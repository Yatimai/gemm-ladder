/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through steps 0 to 3 (nvidia_sample.cuh, step1.cuh, step2.cuh, step3.cuh), whose notice follows.
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
#include "step3.cuh"

namespace step4 {

// ---------------------------------------------------------------- step 4: instruction issue
// Two mechanisms of cuBLAS's T4 kernel, nested on step 3 (step3.cuh, v0): fragment pipelining (4a), and an order of the
// HMMA (4b), a setting measured on its own. Kept from step 3: step 2's kernel (one block per 128 x 256 tile, 8 warps of
// 64 x 64, two shared stages of K32 filled through registers with one barrier per K32, 2a's mma.sync m16n8k8 and
// ldmatrix.x4, 128 HMMA and 16 LDSM.x4 per K32 and per warp, 2b's swizzle, the fp16 output through shared memory half a
// tile at a time), with 3a's loads, 3b's stores and 3c's order of the tiles. The copy of a slice is step 3's too, with
// its addresses and the loop's tests: thread t loads rows t / 4 + 64 i of A at chunk t % 4 and rows t / 32 + 8 i of B
// at chunk t % 32, at addresses computed from k0 at each slice, and a tile cut by M clamps the row of A each thread
// loads to M - 1. Shapes: M a multiple of 16, N of 256, K of 64.
// Steps 2 and 3 read a stage's fragments K16 at a time, starting at the barrier that publishes the stage, once the
// previous stage's HMMA are all issued: the loop issues LDSM in bursts (up to 13 LDSM, LDG and STS in a row with no
// HMMA between them, after each barrier), and 11 of its 32 LDSM per K64 have fewer than 8 HMMA between them and the
// first HMMA that reads them. Under ncu, the loop's waits sit on these LDSM and on the first HMMA to read one of them
// (the README, step 4).
// (4a) Fragment pipelining, a mechanism of cuBLAS's kernel; ours: the fragments of one k8 slice at a time (4
//      LDSM.x4: A's four 16 x 8 blocks and B's eight 8 x 8 ones, the operands of the slice's 32 HMMA) in two sets of 16
//      registers, the 32 that step 2 declares for one K16; the next slice's 4 LDSM issue among the current slice's 32
//      HMMA. The copy to the other stage and the barrier, which both publishes the other stage and frees this one, come
//      after the stage's last read (slice 3's fragments), before slice 2's HMMA (steps 2 and 3: after their last HMMA).
//      HMMA whose operands are in registers then surround the barrier, and ptxas moves some across it. The other
//      stage's first fragments are read after it, among the stage's last HMMA, as are the next slice's global loads.
//      Under 4a alone (v1), a slice's HMMA keep step 2's order (the rows, then the columns). Each accumulator sums its
//      k8 slices in step 2's order, with or without 4b, so step 3 and both variants of step 4 give the same bits.
// (4b) An order of the HMMA that reuses an operand, as cuBLAS's kernel. Ours, on 4a, orders a slice's 32 HMMA so
//      that B's operand is reused in 3 HMMA of 4 (mma below): ptxas sets 192 .reuse flags on its HMMA per K64, none
//      with step 2's order on 4a; same instructions. What each is worth, measured: the README, step 4.
// Tried on 4a and 4b, and left out: a lighter loop, as cuBLAS's (fewer integer instructions under ncu than our 24 per
// K32 and per warp; the README, step 2): pointers moved as the copy goes, immediate offsets from them, the last K64
// peeled off the loop, its counter on the uniform datapath; drafts, not published (the README, step 4).
// The counts below are read with nvcc 13.2 (a local compilation); the measured binaries are built by nvcc 13.1.1 (the
// Modal images).
// Compiled (nvcc 13.2): 254 registers in v0, v1 and step 3, no spill; 49 152 bytes of dynamic shared memory; one block
// per SM. Main loop, per K32 and per warp (one K64 per turn; full tiles, edge tiles in brackets where they differ):
//                       HMMA   LDSM    LDG.128   STS.128   BAR   BRA   integer     total         per HMMA
//   step 3 (v2)         128    16 x4   6         6         1     1.5   26.5 (26)   185 (184.5)   1.45 (1.44)
//   4a (v1)             128    16 x4   6         6         1     0.5   24 (25.5)   181.5 (183)   1.42 (1.43)
//   4a + 4b (v0)        128    16 x4   6         6         1     0.5   24 (25.5)   181.5 (183)   1.42 (1.43)
// 4a's loop keeps step 3's tests, which ptxas predicates instead of branching: 1 branch per K64 instead of 3.
// From the same bar.sync, ptxas emits BAR.SYNC.DEFER_BLOCKING in v0 and v1, and BAR.SYNC in step 3 (one bit of the
// encoding, undocumented; its cause is not known).
// Shared memory: no bank conflict in v0, v1 or step 3; LDSM.x4 and the STS.128 of the copy take 4 wavefronts each, the
// minimum, as in step 2 (host emulator).
// Where the loop's LDSM sit (SASS; per LDSM, the HMMA issued between it and the first HMMA that reads its registers,
// fewest and mean; the longest run of LDSM, LDG and STS with no HMMA between them):
//   step 3 (v2)   0 and 27 (11 of 32 LDSM under 8); 13;
//   4a (v0, v1)   27 to 30 and 38.4 to 40.3 (none under 8); 1.
// Variants (g_var in step4.cu, for the interleaved probe): v0 = step 4 (4a + 4b); v1 = without 4b (4a, a slice's HMMA
// in step 2's order); v2 = step 3 itself (step3::gemm<true, true, true>), so that the anchor runs in the same probe.
// Without 4a, 4b has nothing to order: v1 against v2 measures 4a alone, v0 against v1 4b alone.

using G = step2::Geometry<true>;   // step 2's stages (2b's swizzle): 49 152 bytes

// One 128 x 256 tile; kOrder: 4b; kEdge: the tile is cut by M (the row of A each thread loads is clamped to M - 1).
template <bool kOrder, bool kEdge>
__device__ __forceinline__ void compute_tile(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                             int row0, int col0)
{
    half* As = shmem;                    // 2 stages x 128 rows of the K32 slice of A
    half* Bs = shmem + 2 * G::kAStage;   // 2 stages x 32 rows of the K32 slice of B
    const int warpId = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warpId / 4, wn = warpId % 4;   // the warp's rows wm * 64, columns wn * 64
    // c[i][j]: the 16 x 8 accumulator of rows wm * 64 + 16 i, columns wn * 64 + 8 j (step 2's c[i][j / 2][j % 2]).
    float c[4][8][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) c[i][j][e] = 0.0f;

    // The copy of a K32 slice, as step 3: global memory to registers (load), then registers to a stage (store). Thread
    // t holds rows t / 4 + 64 i of the slice of A at chunk t % 4, rows t / 32 + 8 i of the slice of B at chunk t % 32,
    // and the global addresses are computed from k0 at each slice.
    int4 ra[2], rb[4];
    auto load = [&](int k0) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int r = threadIdx.x / 4 + i * 64;
            const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;
            ra[i] = step3::ld16<true>((const int4*)&A[(size_t)gr * K + k0] + threadIdx.x % 4);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int r = threadIdx.x / 32 + i * 8;
            rb[i] = step3::ld16<true>((const int4*)&B[(size_t)(k0 + r) * N + col0] + threadIdx.x % 32);
        }
    };
    auto store = [&](int s) {
#pragma unroll
        for (int i = 0; i < 2; i++)
            *(int4*)&As[s * G::kAStage + G::a_at(threadIdx.x / 4 + i * 64, threadIdx.x % 4)] = ra[i];
#pragma unroll
        for (int i = 0; i < 4; i++)
            *(int4*)&Bs[s * G::kBStage + G::b_at(threadIdx.x / 32 + i * 8, threadIdx.x % 32)] = rb[i];
    };

    // 4a: the fragments of one k8 slice at a time, in two sets of 16 registers. Set f holds fa[f][h] (A: lane l points
    // to row wm * 64 + 32 h + l at the slice's chunk, so the four matrices of the .x4 are rows 0-7 and 8-15 of the
    // 16 x 8 blocks 2 h and 2 h + 1) and fb[f][h] (B: lane l points to row l % 8 of the slice at chunk wn * 8 + 4
    // h + l / 8, so the four matrices are the 8 x 8 blocks of columns 32 h to 32 h + 31). 4 LDSM.x4 per k8 keep step
    // 2's 16 per K32. Moving down by 32 rows of A or 8 rows of B adds a constant (the swizzle of a row depends on its
    // bits 1-2 for A, 0-2 for B), so a lane needs one offset per k8 slice of A (aOff[k8]) and one per half of its
    // columns of B (bOff[h]).
    uint32_t fa[2][2][4], fb[2][2][4];
    int aOff[4], bOff[2];
#pragma unroll
    for (int k8 = 0; k8 < 4; k8++) aOff[k8] = G::a_at(wm * 64 + lane, k8);
#pragma unroll
    for (int h = 0; h < 2; h++) bOff[h] = G::b_at(lane % 8, wn * 8 + 4 * h + lane / 8);
    auto frag = [&](int f, int s, int k8) {   // stage s, slice k8 (8 columns of A, 8 rows of B) into set f
#pragma unroll
        for (int h = 0; h < 2; h++)
            step2::ldmatrix_x4<false>(fa[f][h], &As[s * G::kAStage + aOff[k8] + h * 32 * G::kAStride]);
#pragma unroll
        for (int h = 0; h < 2; h++)
            step2::ldmatrix_x4<true>(fb[f][h], &Bs[s * G::kBStage + bOff[h] + k8 * 8 * G::kBStride]);
    };
    auto hmma = [&](int f, int i, int j) {   // c[i][j] += the slice's A block i times its B block j, from set f
        step2::mma_m16n8k8(c[i][j], fa[f][i / 2][2 * (i % 2)], fa[f][i / 2][2 * (i % 2) + 1], fb[f][j / 4][j % 4]);
    };
    auto mma = [&](int f) {   // a k8 slice's 32 HMMA from set f
        if constexpr (kOrder) {   // 4b's order: 8 columns, the rows back and forth
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int ii = 0; ii < 4; ii++) hmma(f, j % 2 ? 3 - ii : ii, j);
        } else {   // step 2's order: the rows, then the columns
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) hmma(f, i, j);
        }
    };
    // One K32 stage s; set 0 holds its slice 0 on entry, and each slice's HMMA follow the read of the next slice into
    // the other set (ptxas spreads the LDSM among the HMMA). The copy to the other stage (st) and the barrier come
    // after the last read of stage s (slice 3's fragments) and before slice 2's HMMA. The barrier publishes the other
    // stage, whose slice 0 is read into set 0 before slice 3's HMMA (nf), and frees stage s for the copy one K32 later;
    // the next loads follow it.
    auto stage = [&](int s, bool st, bool nf, auto&& load_next) {
        frag(1, s, 1);
        mma(0);
        frag(0, s, 2);
        mma(1);
        frag(1, s, 3);
        if (st) store(s ^ 1);
        __syncthreads();
        load_next();
        mma(0);
        if (nf) frag(0, s ^ 1, 0);
        mma(1);
    };
    load(0);
    store(0);
    __syncthreads();
    load(32);
    frag(0, 0, 0);
    for (int k0 = 0; k0 < K; k0 += 64) {   // step 2's loop and tests
        const bool next = k0 + 64 < K;
        stage(0, true, true, [&] { if (next) load(k0 + 64); });
        stage(1, next, next, [&] { if (next) load(k0 + 96); });
    }

    // The output, as step 2: half a tile at a time, the 4 warps of rows [64 h, 64 h + 64) store their accumulators in
    // fp16 (each lane's pairs D[g][2t], D[g][2t + 1] and D[g + 8][2t], D[g + 8][2t + 1] of each 16 x 8 accumulator, as
    // two 32-bit words), then all 256 threads copy the 64 x 256 halves out in 16-byte stores. The loop's last barrier
    // follows every read of the stages.
    half* Os = shmem;   // 64 x kOStride halves, inside the loop's buffers
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        if (wm == h) {   // warps 4h to 4h + 3 hold rows [64 h, 64 h + 64)
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    half* o = &Os[(i * 16 + g) * G::kOStride + wn * 64 + j * 8 + 2 * t];
                    *(__half2*)o = __floats2half2_rn(c[i][j][0], c[i][j][1]);
                    *(__half2*)(o + 8 * G::kOStride) = __floats2half2_rn(c[i][j][2], c[i][j][3]);
                }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int r = threadIdx.x / 32 + i * 8;   // row in the half tile
            const int gr = row0 + h * 64 + r;
            if (gr < M)
                step3::st16<true>((int4*)&C[(size_t)gr * N + col0] + threadIdx.x % 32,
                                  *((const int4*)&Os[r * G::kOStride] + threadIdx.x % 32));
        }
        __syncthreads();
    }
}

// One block per 128 x 256 tile, the tiles in step 3's order (3c).
template <bool kOrder>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    int tm, tn;
    step3::tile_of<true>(blockIdx.x, M, N, tm, tn);   // step 3's order (3c)
    const int row0 = tm * 128, col0 = tn * 256;
    if (row0 + 128 <= M) compute_tile<kOrder, false>(shmem, A, B, C, M, N, K, row0, col0);
    else compute_tile<kOrder, true>(shmem, A, B, C, M, N, K, row0, col0);
}

}  // namespace step4
