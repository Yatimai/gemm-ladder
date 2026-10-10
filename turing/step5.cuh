/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through steps 0 to 4 (nvidia_sample.cuh, step1.cuh to step4.cuh), whose notice follows.
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
#include "step4.cuh"

namespace step5 {

// ---------------------------------------------------------------- step 5: cuBLASLt's choice of tile, shape by shape
// Step 4's kernel, at the tile cuBLASLt uses for each shape. Steps 1 to 4 run one tile, 128 x 256, at every shape (step
// 0, NVIDIA's sample, 128 x 128). The judge's 17 shapes are qkv (N 6144, K 4096), o (4096, 4096), gateup (28672, 4096)
// and down (4096, 14336), each at M = 16, 128, 512 and 2048, and ladder (2048 x 2560 x 2048); its score is the
// geometric mean of the time ratios to its reference, cuBLASLt's best configuration or cuBLAS's default call, whichever
// takes less time (judge/judge.cu). The judge keeps the configurations it has measured on this card and sorts them
// again at each run. In the order it keeps for the card line of the measures (Tesla_T4|70W|lt130201|cuda13030 in
// judge/reference/t4.txt, first rank), the best configurations at the 17 shapes use four tiles, read with M and N
// swapped, as cuBLAS computes C^T, and none splits K:
//   128 x 256   at qkv_m128, gateup_m128 and the 9 shapes with M >= 512 (step 4's tile);
//   128 x 128   at o_m128 and down_m128;
//   64 x 256    at qkv_m16 and gateup_m16;
//   64 x 128    at o_m16 and down_m16.
// The judge sorts again at each run, so the best configuration of a shape can change from one run to the next (the
// REFERENCE lines of its outputs). Under ncu, cuBLASLt's heuristic configuration at down_m16 (the probe's lt arm; the
// judge's best there is 64 x 128) runs a 64 x 256 kernel with step 5's block, over more blocks (the README). Its
// 64 x 128 kernel is of another family ("sliced" in its name under ncu), not reproduced: step 5 changes the tile only,
// so its 64 x 128 is step 4's kernel in 2 warps, with cuBLAS's 32 blocks at o_m16 and down_m16. What 5b is worth there,
// measured: the README, step 5.
// Two rules give these tiles at the 17 shapes, with the T4's 40 SMs (the host emulator checks it; the SM count is read
// at the first call). Their thresholds are ours: at the 17 shapes, 256 columns give 16 blocks or at least 24, M is 16
// or at least 128; any threshold in between gives the same tiles.
// (5a) 64 rows of M when M <= 64, 128 otherwise. At M = 16, a 128-row tile spends 7 HMMA of 8 on the rows past M (the
//      clamp), a 64-row tile 3 of 4.
// (5b) 128 columns when 256 would give fewer blocks than half the SMs, 256 otherwise. At o_m128 and down_m128, 16
//      blocks of 128 x 256 leave 24 of the 40 SMs idle, 32 blocks of 128 x 128 leave 8; at o_m16 and down_m16, 16
//      blocks of 64 x 256 against 32 of 64 x 128.
// The copy of a K32 slice, through registers: thread t loads rows t / 4 + i (T / 4) of A at chunk t % 4, and rows
// t / c + i (T / c) of B at chunk t % c, with T threads and c chunks of 16 bytes in a row of B (Geometry below):
// 512 / BN chunks of A and 512 / BM of B per thread. Kept from step 4, at each tile: warps of 64 x 64, 4a and 4b, step
// 2's stages and swizzle, the output 64 rows at a time, 3a's loads, 3b's stores, 3c's order of the tiles and the edge
// in M.
// The counts below are read with nvcc 13.2 (a local compilation); the measured binaries are built by nvcc 13.1.1 (the
// Modal images).
// Compiled (nvcc 13.2): 254 registers (240 at 64 x 256), no spill. One block per SM at 128 x 256 and 64 x 256 (shared
// memory); two can share an SM at 128 x 128 and 64 x 128. 128 x 256 is step 4's kernel, instruction for instruction.
// Main loop, per K32 and per warp (full tiles; edge tiles in brackets):
//               threads   shared bytes   LDG.128, STS.128   integer       total           per HMMA
//   128 x 256   256       49 152         6, 6               24 (25.5)     181.5 (183)     1.42 (1.43)
//   128 x 128   128       32 768         8, 8               25.5 (29.5)   187 (191)       1.46 (1.49)
//   64 x 256    128       40 960         10, 10             38 (37)       203.5 (202.5)   1.59 (1.58)
//   64 x 128    64        24 576         12, 12             34 (36.5)     203.5 (206)     1.59 (1.61)
// In step 5 (v0), at the judge's shapes, the 64-row tiles run their edge loop only (M = 16), the 128 x 128 its full
// loop only.
// Shared memory: no bank conflict at any of the four tiles; the rows of B's stages and of the output buffer change
// length (256 or 512 bytes; 272 or 528 for the output), not the swizzle nor what a quarter warp touches (host
// emulator).
// Tried and left out: the rows of A past M loaded as zeros, instead of a copy of row M - 1 (the README, step 5).
// Variants (g_var in step5.cu, for the interleaved probe): v0 = step 5 (5a + 5b); v1 = without 5a (128 rows at every M;
// 5b then gives 128 x 128 at o_m16 and down_m16); v2 = without 5b (256 columns at every shape); v3 = step 4 itself
// (step4::gemm<true>), so that the anchor runs in the same probe. v0 against v1 measures 5a at the four M = 16 shapes,
// v0 against v2 5b at o_m16, down_m16, o_m128 and down_m128, v0 against v3 the step. The variants run different kernels
// at these six shapes only; elsewhere all four run step 4's kernel.

// A tile of BM x BN (BM 64 or 128, BN 128 or 256) in warps of 64 x 64: 32 x (BM / 64) x (BN / 64) threads. Step 2's
// swizzled stages of K32 and its output through shared memory, 64 rows at a time, at any of the four sizes; 128 x 256
// is step 4's geometry, number for number.
template <int BM, int BN>
struct Geometry {
    static constexpr int kWarpsN = BN / 64, kThreads = 32 * (BM / 64) * kWarpsN;
    static constexpr int kAStride = 32, kBStride = BN, kOStride = BN + 8;   // halves
    static constexpr int kAStage = BM * kAStride, kBStage = 32 * kBStride;
    static constexpr int kLoopBytes = 2 * (kAStage + kBStage) * 2, kOutBytes = 64 * kOStride * 2;
    static constexpr int kSmemBytes = kLoopBytes > kOutBytes ? kLoopBytes : kOutBytes;
    static_assert(kSmemBytes <= 64 * 1024, "above the 64 KB of shared memory of a block on sm_75");
    // The copy of a K32 slice: thread t holds chunk t % 4 of rows t / 4 + kARows i of A (i < kAPer), and chunk
    // t % kBChunks of rows t / kBChunks + kBRows i of B (i < kBPer); the epilogue copies rows t / kBChunks + kBRows i
    // of 64 (i < kOPer) out, chunk t % kBChunks.
    static constexpr int kBChunks = BN / 8, kARows = kThreads / 4, kBRows = kThreads / kBChunks;
    static constexpr int kAPer = BM / kARows, kBPer = 32 / kBRows, kOPer = 64 / kBRows;
    __host__ __device__ static constexpr int a_at(int r, int c) { return r * kAStride + 8 * (c ^ ((r >> 1) & 3)); }
    __host__ __device__ static constexpr int b_at(int r, int c) { return r * kBStride + 8 * (c ^ (r & 7)); }
};

// Step 3's order of the tiles (3c), for any tile: groups of 8 rows of tiles, column by column.
template <int BM, int BN>
__device__ __forceinline__ void tile_of(int b, int M, int N, int& tm, int& tn)
{
    const int tilesN = N / BN, tilesM = (M + BM - 1) / BM, per = 8 * tilesN, first = b / per * 8;
    const int rows = min(tilesM - first, 8), r = b % per;
    tm = first + r % rows;
    tn = r / rows;
}

// One BM x BN tile, step 4's kernel (4a and 4b) at that size; kEdge: the tile is cut by M (the row of A each thread
// loads is clamped to M - 1).
template <int BM, int BN, bool kEdge>
__device__ __forceinline__ void compute_tile(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                             int row0, int col0)
{
    using G = Geometry<BM, BN>;
    half* As = shmem;                    // 2 stages x BM rows of the K32 slice of A
    half* Bs = shmem + 2 * G::kAStage;   // 2 stages x 32 rows of the K32 slice of B
    const int warpId = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warpId / G::kWarpsN, wn = warpId % G::kWarpsN;   // the warp's rows wm * 64, columns wn * 64
    // c[i][j]: the 16 x 8 accumulator of rows wm * 64 + 16 i, columns wn * 64 + 8 j.
    float c[4][8][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) c[i][j][e] = 0.0f;

    // The copy of a K32 slice, step 3's at this size: global memory to registers (load), then registers to a stage
    // (store), the global addresses computed from k0 at each slice.
    int4 ra[G::kAPer], rb[G::kBPer];
    auto load = [&](int k0) {
#pragma unroll
        for (int i = 0; i < G::kAPer; i++) {
            const int r = threadIdx.x / 4 + i * G::kARows;
            const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;
            ra[i] = step3::ld16<true>((const int4*)&A[(size_t)gr * K + k0] + threadIdx.x % 4);
        }
#pragma unroll
        for (int i = 0; i < G::kBPer; i++) {
            const int r = threadIdx.x / G::kBChunks + i * G::kBRows;
            rb[i] = step3::ld16<true>((const int4*)&B[(size_t)(k0 + r) * N + col0] + threadIdx.x % G::kBChunks);
        }
    };
    auto store = [&](int s) {
#pragma unroll
        for (int i = 0; i < G::kAPer; i++)
            *(int4*)&As[s * G::kAStage + G::a_at(threadIdx.x / 4 + i * G::kARows, threadIdx.x % 4)] = ra[i];
#pragma unroll
        for (int i = 0; i < G::kBPer; i++)
            *(int4*)&Bs[s * G::kBStage + G::b_at(threadIdx.x / G::kBChunks + i * G::kBRows,
                                                   threadIdx.x % G::kBChunks)] = rb[i];
    };

    // 4a and 4b, as step 4 (step4.cuh): the fragments of one k8 slice at a time in two sets of 16 registers, a
    // slice's 32 HMMA in 4b's order; the warp's offsets depend on its place in the tile only.
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
    auto mma = [&](int f) {   // a k8 slice's 32 HMMA from set f, in 4b's order
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int ii = 0; ii < 4; ii++) hmma(f, j % 2 ? 3 - ii : ii, j);
    };
    auto stage = [&](int s, bool st, bool nf, auto&& load_next) {   // one K32 stage, as step 4
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

    // The output, as step 2, 64 rows at a time: the warps of those rows store their accumulators in fp16 (each lane's
    // pairs D[g][2t], D[g][2t + 1] and D[g + 8][2t], D[g + 8][2t + 1] of each 16 x 8 accumulator, as two 32-bit
    // words), then all the block's threads copy the 64 x BN rows out in 16-byte stores. The loop's last barrier follows
    // every read of the stages.
    half* Os = shmem;   // 64 x kOStride halves, inside the loop's buffers
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int h = 0; h < BM / 64; h++) {
        if (wm == h) {   // the warps of rows [64 h, 64 h + 64)
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
        for (int i = 0; i < G::kOPer; i++) {
            const int r = threadIdx.x / G::kBChunks + i * G::kBRows;   // row in the 64
            const int gr = row0 + h * 64 + r;
            if (gr < M)
                step3::st16<true>((int4*)&C[(size_t)gr * N + col0] + threadIdx.x % G::kBChunks,
                                  *((const int4*)&Os[r * G::kOStride] + threadIdx.x % G::kBChunks));
        }
        __syncthreads();
    }
}

// One block of 32 x (BM / 64) x (BN / 64) threads per BM x BN tile, the tiles in step 3's order (3c).
template <int BM, int BN>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    int tm, tn;
    tile_of<BM, BN>(blockIdx.x, M, N, tm, tn);
    const int row0 = tm * BM, col0 = tn * BN;
    if (row0 + BM <= M) compute_tile<BM, BN, false>(shmem, A, B, C, M, N, K, row0, col0);
    else compute_tile<BM, BN, true>(shmem, A, B, C, M, N, K, row0, col0);
}

// The tile, shape by shape, as cuBLASLt's best configuration at the judge's 17 shapes: 64 rows of M when M <= 64
// (5a), 128 otherwise; 128 columns when 256 would give fewer blocks than half the SMs (5b), 256 otherwise. Each
// mechanism can be switched off (the variants of step5.cu).
struct Choice { int bm, bn; };
inline Choice choose(int M, int N, int sms, bool rows64, bool cols128)
{
    const int bm = rows64 && M <= 64 ? 64 : 128;
    const int tiles256 = (M + bm - 1) / bm * (N / 256);
    return {bm, cols128 && 2 * tiles256 < sms ? 128 : 256};
}

}  // namespace step5
