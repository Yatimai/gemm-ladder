/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through steps 0 to 4 (nvidia_sample.cuh, step1.cuh to step4.cuh),
 * whose notice follows.
 *
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// The judge's format: C = A B with A (M x K) and B (K x N) fp16 row-major, fp32 accumulation, C (M x N) fp16 row-major.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "step4.cuh"

namespace step5 {

// ---------------------------------------------------------- step 5: fragment double buffering, a serpentine HMMA order
// Two mechanisms on the issue side, on step 4 (step4.cuh, v0: three stages of K64, its stores of C and its groups of
// rows of tiles), whose rest is kept; neither changes the sums (the same C to the bit). CUTLASS's public loop for sm_80
// has both (CUTLASS v4.4.1, include/cutlass/gemm/threadblock/mma_multistage.h and
// include/cutlass/gemm/warp/mma_tensor_op.h).
// (5a) A pipeline of the fragments: two sets of fragment registers, the ldmatrix of the next K16 issued between the
//      HMMA of the current one; in the source, the wait and the barrier move into the middle of the last K16, between
//      its two halves of HMMA, and the first K16 of the next slice loads right after them; the next slice's cp.async
//      are issued among the HMMA of the first K16. CUTLASS's multistage loop also loads the next K16's fragments during
//      the current one's HMMA, and waits, with its barrier, between the HMMA of its last two K16.
// (5b) The order of the HMMA: the operand of four registers (a fragment of A) is held for 8 HMMA, as in step 4, and the
//      operand of two registers (a half of a fragment of B) runs in a serpentine: backwards for every other fragment of
//      A, so that consecutive HMMA share one of their operands and the operand reuse cache can keep it. CUTLASS's
//      warp-level loop visits its fragments in a serpentine too.
// The counts below are read with nvcc 13.2 (a local compilation); the measured binaries are built by nvcc 13.1.1 (the
// Modal images).
// Compiled (nvcc 13.2, sm_80): 5a at 250 registers (step 4: 240), without 5a 236, no spill; one block per SM. Main
// loop, per K64 and per warp (static counts): 5a 270 instructions (128 HMMA, 32 LDSM.x4, 12 LDGSTS, one barrier), no
// branch but the loop's own (the issue of the last slices predicated); without 5a, step 4's 256. HMMA carrying a
// .reuse flag on an operand, per K64 and per warp: step 4 103, 5b alone 117, 5a alone 112, 5a + 5b 124.
// Variants (g_var in step5.cu, for the interleaved probe; each removal keeps the other mechanism):
//   v0 = step 5 (5a + 5b); v1 = without 5a; v2 = without 5b;
//   v3 = step 4 itself (step4::gemm<true, true>), the anchor in the same probe. All four use 147 456 bytes.

using step3::G;              // step 2's swizzled stage
using Smem = step3::Smem<3>;   // step 3's three stages: 147 456 bytes
using step3::cp_async16;
using step3::cp_async_commit;
using step3::cp_async_wait;

// One 128 x 256 tile: step 4's (step4.cuh, v0: three stages, 4a's stores of C), with 5a's pipeline of the fragments
// (kPipe) and 5b's order of the HMMA (kSerp). kEdge: the tile is cut by M.
template <bool kPipe, bool kSerp, bool kEdge>
__device__ __forceinline__ void compute_tile(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                             int row0, int col0)
{
    constexpr int kStages = 3;
    half* As = shmem;                 // kStages x 128 rows of a K64 slice of A
    half* Bs = shmem + Smem::kAAll;   // kStages x 64 rows of a K64 slice of B
    const int warpId = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warpId / 4, wn = warpId % 4;   // the warp's rows wm * 64, columns wn * 64
    float c[4][4][2][4];   // as step 2
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) c[i][j][n][e] = 0.0f;
    int aOff[4], bOff[4];   // step 2's ldmatrix offsets, within a stage
#pragma unroll
    for (int ks = 0; ks < 4; ks++) aOff[ks] = G::a_at(wm * 64 + lane % 16, 2 * ks + lane / 16);
#pragma unroll
    for (int j = 0; j < 4; j++) bOff[j] = G::b_at(lane % 16, wn * 8 + 2 * j + lane / 16);

    auto issue = [&](int k0, int s) {   // step 3's copy of the slice at k0 into stage s
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int r = threadIdx.x / 8 + i * 32;
            const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;
            cp_async16(&As[s * G::kAStage + G::a_at(r, threadIdx.x % 8)],
                       &A[(size_t)gr * K + k0 + 8 * (threadIdx.x % 8)]);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int r = threadIdx.x / 32 + i * 8;
            cp_async16(&Bs[s * G::kBStage + G::b_at(r, threadIdx.x % 32)],
                       &B[(size_t)(k0 + r) * N + col0 + 8 * (threadIdx.x % 32)]);
        }
    };
    // The fragments of one K16: fa[set][i] (rows 16 i of A), fb[set][j] (columns 16 j of B); two sets with 5a.
    uint32_t fa[kPipe ? 2 : 1][4][4], fb[kPipe ? 2 : 1][4][4];
    auto load_frags = [&](int set, int s, int ks) {   // the K16 ks of stage s into set `set`
#pragma unroll
        for (int i = 0; i < 4; i++)
            step2::ldmatrix_x4<false>(fa[set][i], &As[s * G::kAStage + aOff[ks] + i * 16 * G::kAStride]);
#pragma unroll
        for (int j = 0; j < 4; j++)
            step2::ldmatrix_x4<true>(fb[set][j], &Bs[s * G::kBStage + bOff[j] + ks * 16 * G::kBStride]);
    };
    // HMMA h0 to h1 - 1 of the 32 of a K16, on set `set`. HMMA h uses A's fragment i = h / 8 (held for 8 HMMA) and B's
    // n8 half q = h % 8 (j = q / 2, n = q % 2); with 5b, q runs backwards for odd i, so that each A fragment starts on
    // the B half its predecessor ended on (a serpentine).
    auto mma_range = [&](int set, int h0, int h1) {
#pragma unroll
        for (int h = h0; h < h1; h++) {
            const int i = h / 8, q = kSerp && (i & 1) ? 7 - h % 8 : h % 8, j = q / 2, n = q % 2;
            step2::mma_m16n8k16(c[i][j][n], fa[set][i], fb[set][j][2 * n], fb[set][j][2 * n + 1]);
        }
    };
    const int steps = K / 64;
    // Step 3's prologue: slices 0 and 1 in flight, one group each.
#pragma unroll
    for (int s = 0; s < kStages - 1; s++) {
        if (s < steps) issue(s * 64, s);
        cp_async_commit();
    }
    int read = 0, write = kStages - 1;   // the stages of slices k and k + 2
    if constexpr (kPipe) {
        // (5a): the fragments of the next K16 load while the current K16's HMMA run. The first K16 of slice 0 loads
        // before the loop; at K16 0 of slice k, slice k + 2 is issued into the stage of slice k - 1, which every warp
        // left before the last barrier; in the middle of K16 3, between its two halves of HMMA, the wait for slice
        // k + 1 and the barrier, then the first K16 of slice k + 1.
        cp_async_wait<kStages - 2>();
        __syncthreads();
        load_frags(0, 0, 0);
        for (int k = 0; k < steps; k++) {
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const int cur = ks & 1;
                if (ks < 3) load_frags(cur ^ 1, read, ks + 1);
                if (ks == 0) {
                    if (k + kStages - 1 < steps) issue((k + kStages - 1) * 64, write);
                    cp_async_commit();
                }
                if (ks < 3) {
                    mma_range(cur, 0, 32);
                } else {
                    mma_range(cur, 0, 16);
                    cp_async_wait<kStages - 2>();
                    __syncthreads();
                    read = read + 1 == kStages ? 0 : read + 1;
                    write = write + 1 == kStages ? 0 : write + 1;
                    if (k + 1 < steps) load_frags(cur ^ 1, read, 0);
                    mma_range(cur, 16, 32);
                }
            }
        }
    } else {
        // Step 4's loop: per K64 one wait, one barrier, the next issue, then per K16 its fragments and its HMMA.
        for (int k = 0; k < steps; k++) {
            cp_async_wait<kStages - 2>();
            __syncthreads();
            if (k + kStages - 1 < steps) issue((k + kStages - 1) * 64, write);
            cp_async_commit();
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                load_frags(0, read, ks);
                mma_range(0, 0, 32);
            }
            read = read + 1 == kStages ? 0 : read + 1;
            write = write + 1 == kStages ? 0 : write + 1;
        }
    }
    cp_async_wait<0>();   // only empty groups are left
    __syncthreads();      // every warp is done with the last stage before the output reuses it
    // The output, half a tile at a time, as step 4 (4a's stores of C).
    half* Os = shmem;
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        if (wm == h) {
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
#pragma unroll
                    for (int n = 0; n < 2; n++) {
                        half* o = &Os[(i * 16 + g) * G::kOStride + wn * 64 + j * 16 + n * 8 + 2 * t];
                        *(__half2*)o = __floats2half2_rn(c[i][j][n][0], c[i][j][n][1]);
                        *(__half2*)(o + 8 * G::kOStride) = __floats2half2_rn(c[i][j][n][2], c[i][j][n][3]);
                    }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int r = threadIdx.x / 32 + i * 8;   // row in the half tile
            const int gr = row0 + h * 64 + r;
            if (gr < M)
                step4::st16<true>((int4*)&C[(size_t)gr * N + col0] + threadIdx.x % 32,
                                  *((const int4*)&Os[r * G::kOStride] + threadIdx.x % 32));
        }
        __syncthreads();
    }
}

// One block per 128 x 256 tile, in step 4's order (4b's groups of rows of tiles).
template <bool kPipe, bool kSerp>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    int tm, tn;
    step4::tile_of<true>(blockIdx.x, M, N, tm, tn);
    const int row0 = tm * 128, col0 = tn * 256;
    if (row0 + 128 <= M) compute_tile<kPipe, kSerp, false>(shmem, A, B, C, M, N, K, row0, col0);
    else compute_tile<kPipe, kSerp, true>(shmem, A, B, C, M, N, K, row0, col0);
}

}  // namespace step5
