/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through steps 0 to 3 (nvidia_sample.cuh, step1.cuh to step3.cuh),
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
#include "step3.cuh"

namespace step4 {

// ---------------------------------------------------------------- step 4: the memory hierarchy
// Two mechanisms on the way to and from DRAM, on step 3 (step3.cuh, three stages), whose rest is kept: the loop does
// not change (the same 256 instructions per K64 and per warp), nor the arithmetic (the variants give step 3's bits).
// (4a) Stores of C marked evict-first: st.global.cs (STG.E.EF.128 in place of STG.E.128). C is written once and never
//      read back; its lines then leave L2 before those of A and B, which the next tiles read again.
// (4b) The tiles in groups of R = 16 rows of tiles, column by column inside a group (our order and our R; cuBLASLt's
//      API has a CTA swizzle, an on/off option): a wave of 108 blocks then covers R rows of tiles and 108 / R columns
//      instead of 108 / tilesN rows across all the columns. Our count, if the blocks of a wave move through K together
//      and L2 holds what one wave shares: at down_m2048 (16 x 32 tiles, K = 28672, A 117 MB, B 470 MB) step 0's order
//      reads all of B in each of the 4.7 waves (about 2.3 GB from DRAM), the groups of 16 rows read A once per wave and
//      each column of B once (about 1.0 GB); at the ladder (27 x 16 tiles, 31 MB of operands, 40 MB of L2) the two
//      orders read about as much.
// Compiled (nvcc 13.2, sm_80): the three variants of step 4 at 240 registers (step 3: 236), no spill; the main loop
// is step 3's; 4b computes its tile before the loop (about 40 more instructions per tile).
// Variants (g_var in step4.cu, for the interleaved probe; each removal keeps the other mechanism):
//   v0 = step 4 (4a + 4b); v1 = without 4a; v2 = without 4b;
//   v3 = step 3 itself (step3::gemm<3>), the anchor in the same probe. All four use step 3's 147 456 bytes.

using step3::G;                       // step 2's swizzled stage
using Smem = step3::Smem<3>;          // step 3's three stages: 147 456 bytes
using step3::cp_async16;
using step3::cp_async_commit;
using step3::cp_async_wait;

// 4a: a 16-byte store of C; with kEf, marked evict-first (st.global.cs: STG.E.EF.128).
template <bool kEf>
__device__ __forceinline__ void st16(int4* p, int4 v)
{
#ifdef GEMM_LADDER_EMULATE
    *p = v;
#else
    if constexpr (kEf)
        asm volatile("st.global.cs.v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                     : "memory");
    else
        *p = v;
#endif
}

// 4b: the rows of tiles per group.
constexpr int kGroupRows = 16;

// 4b: the tile (row of tiles tm, column of tiles tn) of block b. With kGroup, the blocks of group g cover rows of tiles
// R g to R g + R - 1 (R = kGroupRows; the last group may have fewer), column by column; without it, step 0's order
// (N fastest).
template <bool kGroup>
__host__ __device__ __forceinline__ void tile_of(int b, int M, int N, int& tm, int& tn)
{
    const int tilesN = N / 256;
    if constexpr (kGroup) {
        const int R = kGroupRows, tilesM = (M + 127) / 128, per = R * tilesN, first = b / per * R;
        const int rows = min(tilesM - first, R), r = b % per;
        tm = first + r % rows;
        tn = r / rows;
    } else {
        tm = b / tilesN;
        tn = b % tilesN;
    }
}

// One 128 x 256 tile: step 3's compute_tile (step3.cuh, three stages), with 4a's stores of C. kEdge: the tile is cut
// by M (the row of A each thread copies is clamped to M - 1).
template <bool kEf, bool kEdge>
__device__ __forceinline__ void compute_tile(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                             int row0, int col0)
{
    constexpr int kStages = 3;
    half* As = shmem;                         // kStages x 128 rows of a K64 slice of A
    half* Bs = shmem + Smem::kAAll;           // kStages x 64 rows of a K64 slice of B
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

    // The copy of the slice that starts at k0 into stage s, as step 2's (thread t: rows t / 8 + 32 i of A at chunk
    // t % 8, rows t / 32 + 8 i of B at chunk t % 32, at their swizzled places), each chunk one cp.async.
    auto issue = [&](int k0, int s) {
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
    auto compute = [&](int s) {   // step 2's compute over stage s
        const half* as = As + s * G::kAStage;
        const half* bs = Bs + s * G::kBStage;
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            uint32_t a[4][4], b[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                step2::ldmatrix_x4<false>(a[i], &as[aOff[ks] + i * 16 * G::kAStride]);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (i == 0) step2::ldmatrix_x4<true>(b[j], &bs[bOff[j] + ks * 16 * G::kBStride]);
#pragma unroll
                    for (int n = 0; n < 2; n++) step2::mma_m16n8k16(c[i][j][n], a[i], b[j][2 * n], b[j][2 * n + 1]);
                }
            }
        }
    };
    const int steps = K / 64;
    // Step 3's loop (3b): kStages - 1 slices in flight; per K64 one wait, one barrier, the next issue, the compute.
#pragma unroll
    for (int s = 0; s < kStages - 1; s++) {
        if (s < steps) issue(s * 64, s);
        cp_async_commit();
    }
    int read = 0, write = kStages - 1;   // the stages of slices k and k + kStages - 1
    for (int k = 0; k < steps; k++) {
        cp_async_wait<kStages - 2>();
        __syncthreads();
        if (k + kStages - 1 < steps) issue((k + kStages - 1) * 64, write);
        cp_async_commit();
        compute(read);
        read = read + 1 == kStages ? 0 : read + 1;
        write = write + 1 == kStages ? 0 : write + 1;
    }
    cp_async_wait<0>();   // only empty groups are left
    __syncthreads();      // every warp is done with the last stage before the output reuses it
    // The output, half a tile at a time, as step 2, its stores of C through st16 (4a).
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
                st16<kEf>((int4*)&C[(size_t)gr * N + col0] + threadIdx.x % 32,
                          *((const int4*)&Os[r * G::kOStride] + threadIdx.x % 32));
        }
        __syncthreads();
    }
}

// One block per 128 x 256 tile (step 1's 1a), in 4b's order or in step 0's.
template <bool kEf, bool kGroup>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    int tm, tn;
    tile_of<kGroup>(blockIdx.x, M, N, tm, tn);
    const int row0 = tm * 128, col0 = tn * 256;
    if (row0 + 128 <= M) compute_tile<kEf, false>(shmem, A, B, C, M, N, K, row0, col0);
    else compute_tile<kEf, true>(shmem, A, B, C, M, N, K, row0, col0);
}

}  // namespace step4
