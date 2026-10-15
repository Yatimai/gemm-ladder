/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through steps 0 to 5 (nvidia_sample.cuh, step1.cuh to step5.cuh), whose notice follows.
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
#include "step5.cuh"

namespace step6 {

// ---------------------------------------------------------------- step 6: L2-only loads, 16-row bands, split-K in two
// Step 6 grafts three mechanisms on step 5 as it is, measured by the interleaved probe (each mechanism removed against
// step 6), ncu and the judge; the measures are in the README, step 6.
// (6a) Loads through the L2 only: compute_tile loads A and B with ld.global.cg.L2::128B, not 3a's
//      ld.global.nc.L2::128B; the SASS changes LDG.E.LTC128B.128.CONSTANT.SYS into LDG.E.LTC128B.128.STRONG.GPU (and a
//      few integer instructions, below). At qkv_m128, gateup_m128 and the nine shapes with M >= 512 (ladder among
//      them), 6a is the only mechanism that acts.
// (6b) M <= 16 (decoding): a kernel of its own. The GEMM reads B (K x N) once; A and C are small. Step 5 takes there a
//      tile of 64 rows of M, 48 of them padding. 6b's tile has the 16 rows of mma.sync m16n8k8, none padding: a block
//      per band of BN columns of B (128 or 256) and per slice of K among S, in one round at 3 blocks per SM; the slices
//      summed in fp16 parts by the last block of each band. B is 14336 x 4096 halves at down_m16, 117 440 512 bytes.
//      The rule (bands(), below), from N, K and P (40 on the T4): at least 1.2 P blocks and at most 3 P (one round at 3
//      blocks per SM); then, at long K, a slice added while the blocks stay within 2 P and the parts within a hundredth
//      of B's flow (16 S / K <= 0.01, S the slices with the added one; the slices taken to reach 1.2 P may exceed it).
//      The band 128 columns wide, without slices, when that alone gives the number of blocks; else 256 (N a multiple of
//      256), sliced as needed; at least 4 K steps per slice when S > 1. With more than 3 P bands of 256 columns: 256
//      columns, one slice, several rounds (outside the judge's shapes: N > 30 720 on the T4). B loaded evict-first
//      (ld.global.cs) when S > 1, through the L2 only (6a) otherwise; A through the L2 only. The thresholds are set on
//      the four M = 16 shapes. At the judge's shapes the rule gives qkv_m16 128 x 1 (48 blocks), o_m16 256 x 3 (48),
//      gateup_m16 256 x 1 (112), down_m16 256 x 5 (80), and B through the L2 only at qkv_m16 too: one path, no
//      exception per shape.
//      The kernel: 256 threads, warp w computing the band's columns w BN / 8 to (w + 1) BN / 8 - 1, all 16 rows; a K
//      step of BK = 4096 / BN columns of K (one stage = 8 KB of B, three blocks per SM: 3 x 18 KB); the copy of a K
//      step through registers into one of two shared stages, two register sets, so that the copy runs two K steps
//      ahead, one barrier per K step; A in BK / 16 ldmatrix.x4, B in two ldmatrix.x4.trans, 8 HMMA per K step and warp;
//      step 2's swizzle rule at these row lengths (no bank conflict); the output in fp16 through shared memory (16 rows
//      at a pitch of BN + 8 halves). One slice: stored to C. Several: stored to the block's part in the workspace; a
//      fence; one atomic increment of the band's counter per block; the block that arrives last sums
//      the S parts in the order of the slices in fp32 (its own from registers: the fp16 values it wrote), writes C and
//      puts the counter back to zero. The sum does not depend on the order of arrival: the same bits from run to run.
//      Rows of A >= M are zeros in the stages (each copying thread's registers keep the zeros they start with), rows
//      of C >= M are not written.
//      Its code is kept short and straight (no division: the host splits K; pointers moved as the copy goes; the first
//      three K steps' loads issued before the prologue waits).
// (6c) K cut in two at o_m128 and down_m128: a 128 x 256 tile has 16 tiles there and leaves 24 of the 40 SMs idle;
//      cut in two halves of K, the tiles give 32 halves, one per block for 32 of the 40 blocks. The rule: 2 slices
//      where step 5 takes 128 rows (M > 64) and its 128 x 256 tiles are fewer than half the SMs, K a multiple of 128.
//      In step 5's persistent grid, place 2 t + s of the order (a block's block_pos) is tile t over K's half s: its
//      block computes it (compute_tile over K / 2 columns, A and B moved to the half s) and stores it in fp16 (fp16
//      parts, this rung's choice) into its slice's part, an M x N matrix; a fence, one atomic increment of the tile's
//      counter per half; the block that arrives second sums the two parts in the order of the slices, in fp32, writes C
//      and puts the counter back to zero. The same bits whatever the order of arrival. Its parts are 2 MiB at o_m128.
//      The half is chosen by moving A and B, so that compute_tile's loop runs from 0 to K / 2 at every place.
// Kept from step 5: step 0's persistent grid (one block per SM, block_pos += gridDim.x), compute_tile's tile at the
// two sizes (4a, 2b's swizzle, the output 64 rows at a time, 3b's order of the tiles, the edge in M), step5::choose's
// tile wherever 6b and 6c do not apply. Shapes: M a multiple of 16, N of 256, K of 64 (checked on the host); 6b runs at
// M = 16 when its contract holds (N a multiple of its width, K of its K step, at least 3 K steps per slice, at most
// 3 P blocks when there are slices: the workspace's ceiling), 6c where M > 64, the 128 x 256 tiles are fewer than half
// the SMs and K is a multiple of 128.
// The workspace (step6.cu): allocated once, at the first call, at the ceilings of the rules, never freed. Not for
// concurrent calls.
// Under GEMM_LADDER_EMULATE (turing/emu), which runs one block at a time: the loads are plain 16-byte copies, the
// atomic a plain increment, the fence nothing.
// The counts below are read with nvcc 13.2 (a local compilation); the measured binaries are built by nvcc 13.1.1 (the
// Modal images).
// Compiled (nvcc 13.2, sm_75): step 5's tile with 6a has 370 / 370 instructions per K64 at 128 x 256 (step 5's counts)
// and 398 / 398 at 64 x 256 (step 5: 395 / 394), full / edge tiles; the loads aside, a few integer instructions differ
// (254 and 252 registers; without 6a, step 5's tile instruction for instruction, 254 and 250). gemm_split (with 6a or
// without): 252 registers, 430 / 430 instructions per K64; no spill. 6b
// (__launch_bounds__(256, 3): at most 80 registers): 70 registers at BN 256, 66 at BN 128; one slice / several: 272 /
// 448 instructions at BN 256, 240 / 344 at BN 128; the first load after 37 instructions, the first barrier after 86 (83
// at BN 128), 77 (73) in the loop of two K steps; 17 424 and 18 448 bytes of shared memory.
// Variants (g_var in step6.cu, for the interleaved probe): v0 = step 6 (6a + 6b + 6c); v1 = without 6a (nc in
// compute_tile; 6b keeps its loads); v2 = without 6b (step 5's 64 x 256 tiles at M = 16); v3 = without 6c (step 5's
// 128 x 256 tiles at o_m128 and down_m128); v4 = step 5 itself, the anchor; v5 = 6b with B never marked evict-first
// (through the L2 only, as with one slice). v2 and v3 keep 6a in compute_tile: the tiles of step 5 they fall back to
// load through the L2 only (v1 and v4 do not). v0 against v1 measures 6a at M >= 128; v0 against v2 and v3 the other
// two; v0 against v4 the step; v0 against v5 the evict-first marking, where 6b has slices (o_m16 and down_m16).

// ---------------------------------------------------------------- the device helpers
// 6a: a 16-byte load of A or B for compute_tile, through the L2 only (ld.global.cg), the L2 asked for the whole
// 128-byte line as 3a does; without 6a (kCg false, v1), step 3's ld.global.nc.L2::128B (3a).
template <bool kCg>
__device__ __forceinline__ int4 ld16(const int4* p)
{
#ifdef GEMM_LADDER_EMULATE
    return *p;
#else
    if constexpr (kCg) {
        int4 v;
        asm("ld.global.cg.L2::128B.v4.s32 {%0, %1, %2, %3}, [%4];"
            : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
        return v;
    } else {
        return step3::ld16<true>(p);
    }
#endif
}

// 6b: a 16-byte load of A or B for a band, through the L2 only (6a), or marked evict-first (kEvictFirst: B when the
// band has parts, except in v5); volatile.
template <bool kEvictFirst>
__device__ __forceinline__ int4 ld16_band(const int4* p)
{
#ifdef GEMM_LADDER_EMULATE
    return *p;
#else
    int4 v;
    if constexpr (kEvictFirst)
        asm volatile("ld.global.cs.L2::128B.v4.s32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    else
        asm volatile("ld.global.cg.L2::128B.v4.s32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
#endif
}

// 6b and 6c: a part written by another block of this launch, through the L2 (not a stale L1 line), and not moved by
// the compiler above the atomic that orders it.
__device__ __forceinline__ int4 ld16_part(const int4* p)
{
#ifdef GEMM_LADDER_EMULATE
    return *p;
#else
    int4 v;
    asm volatile("ld.global.cg.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)
                 : "memory");
    return v;
#endif
}

// The block's arrival on a counter (6b, 6c), and the fence before it and after the last arrival. Under
// GEMM_LADDER_EMULATE, one block at a time: a plain increment, and nothing.
#ifdef GEMM_LADDER_EMULATE
inline unsigned arrive(unsigned* count) { return (*count)++; }
inline void fence() {}
#define STEP6_BAND_BOUNDS
#else
__device__ __forceinline__ unsigned arrive(unsigned* count) { return atomicAdd(count, 1u); }
__device__ __forceinline__ void fence() { __threadfence(); }
#define STEP6_BAND_BOUNDS __launch_bounds__(256, 3)
#endif

// ---------------------------------------------------------------- 6a, 6c: step 5's tile
// One BM x 256 tile, step 5's compute_tile (step5.cuh) with two changes: 6a's loads (kCg) and the K columns [0, kEnd)
// only (kEnd a multiple of 64: K / 2 in 6c, A and B moved to the half; K elsewhere; K stays the length of A's rows).
// With kCg false and kEnd = K, step 5's tile in the source (the same SASS, instruction for instruction, at the two
// sizes).
template <int BM, bool kEdge, bool kCg>
__device__ __forceinline__ void compute_tile(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                             int row0, int col0, int kEnd)
{
    using G = step5::Geometry<BM>;
    half* As = shmem;                    // 2 stages x BM rows of the K32 slice of A
    half* Bs = shmem + 2 * G::kAStage;   // 2 stages x 32 rows of the K32 slice of B
    const int warpId = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warpId / 4, wn = warpId % 4;   // the warp's rows wm * 64, columns wn * 64
    // c[i][j]: the 16 x 8 accumulator of rows wm * 64 + 16 i, columns wn * 64 + 8 j.
    float c[4][8][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) c[i][j][e] = 0.0f;

    // The copy of a K32 slice, step 5's, with 6a's loads.
    int4 ra[G::kAPer], rb[G::kBPer];
    auto load = [&](int k0) {
#pragma unroll
        for (int i = 0; i < G::kAPer; i++) {
            const int r = threadIdx.x / 4 + i * G::kARows;
            const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;
            ra[i] = ld16<kCg>((const int4*)&A[(size_t)gr * K + k0] + threadIdx.x % 4);
        }
#pragma unroll
        for (int i = 0; i < G::kBPer; i++) {
            const int r = threadIdx.x / G::kBChunks + i * G::kBRows;
            rb[i] = ld16<kCg>((const int4*)&B[(size_t)(k0 + r) * N + col0] + threadIdx.x % G::kBChunks);
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

    // 4a, as step 5.
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
    auto mma = [&](int f) {   // a k8 slice's 32 HMMA from set f, in step 2's order: the rows, then the columns
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) hmma(f, i, j);
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
    for (int k0 = 0; k0 < kEnd; k0 += 64) {   // step 2's loop and tests, over [0, kEnd)
        const bool next = k0 + 64 < kEnd;
        stage(0, true, true, [&] { if (next) load(k0 + 64); });
        stage(1, next, next, [&] { if (next) load(k0 + 96); });
    }

    // The output, as step 5, 64 rows at a time; the loop's last barrier follows every read of the stages.
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
                *((int4*)&C[(size_t)gr * N + col0] + threadIdx.x % G::kBChunks) =
                    *((const int4*)&Os[r * G::kOStride] + threadIdx.x % G::kBChunks);
        }
        __syncthreads();
    }
}

// Tile t (step 5's order, 3b) over the K columns [0, kEnd) of A and B into C or a part.
template <int BM, bool kCg>
__device__ __forceinline__ void tile(half* shmem, const half* A, const half* B, half* C, int M, int N, int K, int t,
                                     int kEnd)
{
    int tm, tn;
    step5::tile_of<BM>(t, M, N, tm, tn);
    const int row0 = tm * BM, col0 = tn * 256;
    if (row0 + BM <= M) compute_tile<BM, false, kCg>(shmem, A, B, C, M, N, K, row0, col0, kEnd);
    else compute_tile<BM, true, kCg>(shmem, A, B, C, M, N, K, row0, col0, kEnd);
}

// Step 5's persistent grid (one block per SM, block_pos += gridDim.x), with 6a's loads.
template <int BM, bool kCg>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    const int tiles = (M + BM - 1) / BM * (N / 256);
    for (int block_pos = blockIdx.x; block_pos < tiles; block_pos += gridDim.x)
        tile<BM, kCg>(shmem, A, B, C, M, N, K, block_pos, K);
}

// 6c: two halves of K per tile in step 5's persistent grid, place 2 t + s of the order computing tile t over K's half
// s (A and B moved to it: see (6c) above) into its part (an M x N matrix of halves); the block that arrives second at a
// tile sums the two parts in the order of the slices, in fp32, and writes C.
template <int BM, bool kCg>
__global__ void gemm_split(const half* A, const half* B, half* C, int M, int N, int K, half* parts, unsigned* counts)
{
    using G = step5::Geometry<BM>;
    extern __shared__ __align__(128) half shmem[];
    const int tiles = (M + BM - 1) / BM * (N / 256), halfK = K / 2;
    unsigned* last = (unsigned*)((char*)shmem + G::kSmemBytes);   // a word after the tile's buffers (kSmemBytes + 16)
    for (int block_pos = blockIdx.x; block_pos < 2 * tiles; block_pos += gridDim.x) {
        const int t = block_pos / 2, s = block_pos % 2;
        tile<BM, kCg>(shmem, A + (size_t)s * halfK, B + (size_t)s * halfK * N, parts + (size_t)s * M * N, M, N, K, t,
                      halfK);
        fence();          // every thread's stores of the part, before the block arrives
        __syncthreads();
        if (threadIdx.x == 0) *last = arrive(&counts[t]) == 1u;
        __syncthreads();
        if (!*last) continue;
        fence();
        int tm, tn;
        step5::tile_of<BM>(t, M, N, tm, tn);
        const int row0 = tm * BM, col0 = tn * 256;
        // The sum: 16-byte chunk q of the tile is row q / kBChunks, chunk q % kBChunks; slice 0's part first.
        for (int q = threadIdx.x; q < BM * G::kBChunks; q += G::kThreads) {
            const int r = row0 + q / G::kBChunks, cc = q % G::kBChunks;
            if (r >= M) continue;
            float v[8];
#pragma unroll
            for (int e = 0; e < 8; e++) v[e] = 0.0f;
#pragma unroll
            for (int ss = 0; ss < 2; ss++) {
                const int4 x = ld16_part((const int4*)&parts[(size_t)ss * M * N + (size_t)r * N + col0] + cc);
                const __half2* h = reinterpret_cast<const __half2*>(&x);
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const __half2 he = h[e];
                    v[2 * e] += __half2float(he.x);
                    v[2 * e + 1] += __half2float(he.y);
                }
            }
            int4 out;
            __half2* o = reinterpret_cast<__half2*>(&out);
#pragma unroll
            for (int e = 0; e < 4; e++) o[e] = __floats2half2_rn(v[2 * e], v[2 * e + 1]);
            *((int4*)&C[(size_t)r * N + col0] + cc) = out;
        }
        if (threadIdx.x == 0) counts[t] = 0;   // for the next call
    }
}

// ---------------------------------------------------------------- 6b: M <= 16, bands of B
// A band of BN columns (128 or 256) over a slice of K: BK = 4096 / BN columns of K per step, so that a stage holds 8 KB
// of B and three blocks fit in an SM; 256 threads, warp w computes the band's columns w BN / 8 to (w + 1) BN / 8 - 1,
// all 16 rows.
template <int BN>
struct BandGeometry {
    static constexpr int kBK = 4096 / BN;                                   // K per step: 8 KB of B per stage
    static constexpr int kThreads = 256, kWarpCols = BN / 8, kTiles = kWarpCols / 8, kK8 = kBK / 8;
    static constexpr int kAChunks = kBK / 8, kALoaders = 16 * kAChunks;     // A: 16 rows of kBK halves
    static constexpr int kARowsPerLine = 64 / kBK;                          // rows of A in a 128-byte line: 2 or 4
    static constexpr int kBChunks = BN / 8, kBRows = kThreads / kBChunks;  // B: rows t / kBChunks + kBRows i, i < 2
    static constexpr int kAStage = 16 * kBK, kBStage = kBK * BN;            // halves
    static constexpr int kOPitch = BN + 8;                                  // halves: the output's rows
    static constexpr int kLoopBytes = 2 * (kAStage + kBStage) * 2, kOutBytes = 16 * kOPitch * 2;
    static constexpr int kFlagByte = kLoopBytes > kOutBytes ? kLoopBytes : kOutBytes;   // the last block's flag
    static constexpr int kSmemBytes = kFlagByte + 16;
    static constexpr int kOPer = 16 * kBChunks / kThreads;                  // output chunks per thread: 1 or 2
    static_assert(kBK == 2 * kBRows, "two 16-byte loads of B per thread and step");
    static_assert(kTiles * kK8 == 8, "8 HMMA per step and warp: two ldmatrix.x4.trans of B");
    static_assert(3 * kSmemBytes <= 64 * 1024, "three blocks per SM on sm_75");
    __host__ __device__ static constexpr int a_at(int r, int c)
    {
        return r * kBK + 8 * (c ^ ((r / kARowsPerLine) & (kAChunks - 1)));
    }
    __host__ __device__ static constexpr int b_at(int r, int c) { return r * BN + 8 * (c ^ (r & 7)); }
};

// One block: slice blockIdx.x of K (perSlice steps of BK, one more for the first `longer` slices: the host divides),
// band blockIdx.y of BN columns; the slices of a band are neighbours in the order of the blocks. kParts: several
// slices, so parts in the workspace; kEvictFirst: B marked evict-first (with parts, except in v5).
template <int BN, bool kParts, bool kEvictFirst>
__device__ __forceinline__ void band(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                     int perSlice, int longer, half* parts, unsigned* counts)
{
    using G = BandGeometry<BN>;
    half* As = shmem;                    // 2 stages x 16 rows of the K step's slice of A
    half* Bs = shmem + 2 * G::kAStage;   // 2 stages x kBK rows of the K step's slice of B
    const int s = blockIdx.x, b = blockIdx.y, col0 = b * BN;
    const int k0 = s * perSlice + min(s, longer), nk = perSlice + (s < longer);
    const int warpId = threadIdx.x / 32, lane = threadIdx.x % 32;

    // The copy of a K step: thread t < kALoaders stores row t / kAChunks of A at chunk t % kAChunks, loaded if that
    // row is below M, else the zeros its registers hold from the start; every thread loads rows t / kBChunks and
    // t / kBChunks + kBRows of B at chunk t % kBChunks. Two register sets, the copy two K steps ahead; the pointers
    // move on by one K step per copy.
    const int ar = threadIdx.x / G::kAChunks, ac = threadIdx.x % G::kAChunks;
    const bool aStores = threadIdx.x < G::kALoaders, aLoads = aStores && ar < M;
    const half* aNext = A + (size_t)(aLoads ? ar : 0) * K + (size_t)k0 * G::kBK + ac * 8;
    const int br = threadIdx.x / G::kBChunks, bc = threadIdx.x % G::kBChunks;
    const size_t bRows = (size_t)G::kBRows * N, bStep = (size_t)G::kBK * N;
    const half* bNext = B + ((size_t)k0 * G::kBK + br) * N + col0 + bc * 8;
    int4 zero;
    zero.x = zero.y = zero.z = zero.w = 0;
    int4 ra[2] = {}, rb[2][2] = {};   // zeros: A's rows >= M keep them
    auto fetch = [&](int4& a, int4* bb) {   // the next K step of the slice
        bb[0] = ld16_band<kEvictFirst>((const int4*)bNext);
        bb[1] = ld16_band<kEvictFirst>((const int4*)(bNext + bRows));
        bNext += bStep;
        if (aLoads) a = ld16_band<false>((const int4*)aNext);
        aNext += G::kBK;
    };
    auto store = [&](int set, int st) {   // register set `set` into stage st
        *(int4*)&Bs[st * G::kBStage + G::b_at(br, bc)] = rb[set][0];
        *(int4*)&Bs[st * G::kBStage + G::b_at(br + G::kBRows, bc)] = rb[set][1];
        if (aStores) *(int4*)&As[st * G::kAStage + G::a_at(ar, ac)] = ra[set];
    };

    // Fragments. A: ldmatrix.x4 h gives rows 0-7 and 8-15 of k8 slices 2h and 2h + 1 (lane l points to row l % 16 at
    // chunk 2h + l / 16). B: ldmatrix.x4.trans j gives the (k8, n8) blocks i = 4j to 4j + 3, n first (k8 = i / kTiles,
    // n8 = i % kTiles; lane l points to row 8 k8 + l % 8 of block 4j + l / 8, at the warp's chunk w kTiles + n8).
    int aOff[G::kK8 / 2], bOff[2];
#pragma unroll
    for (int h = 0; h < G::kK8 / 2; h++) aOff[h] = G::a_at(lane % 16, 2 * h + lane / 16);
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const int i = 4 * j + lane / 8;
        bOff[j] = G::b_at(8 * (i / G::kTiles) + lane % 8, warpId * G::kTiles + i % G::kTiles);
    }
    // c[n]: the 16 x 8 accumulator of the warp's n8 tile n (rows g and g + 8, columns 2t and 2t + 1 of the lane).
    float c[G::kTiles][4];
#pragma unroll
    for (int n = 0; n < G::kTiles; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) c[n][e] = 0.0f;
    auto compute = [&](int st) {   // the step in stage st: 8 HMMA per warp
        uint32_t fa[G::kK8 / 2][4], fb[2][4];
#pragma unroll
        for (int h = 0; h < G::kK8 / 2; h++) step2::ldmatrix_x4<false>(fa[h], &As[st * G::kAStage + aOff[h]]);
#pragma unroll
        for (int j = 0; j < 2; j++) step2::ldmatrix_x4<true>(fb[j], &Bs[st * G::kBStage + bOff[j]]);
#pragma unroll
        for (int k8 = 0; k8 < G::kK8; k8++)
#pragma unroll
            for (int n = 0; n < G::kTiles; n++) {
                const int i = k8 * G::kTiles + n;
                step2::mma_m16n8k8(c[n], fa[k8 / 2][2 * (k8 % 2)], fa[k8 / 2][2 * (k8 % 2) + 1], fb[i / 4][i % 4]);
            }
    };

    // The loop: K step i computes from stage i % 2; K step i + 1 goes from register set (i + 1) % 2 to the other
    // stage, the barrier publishes it and frees stage i % 2; K step i + 3 is loaded into the set K step i + 1 left. Two
    // K steps per turn, so that the stage and the register set are constants (a set indexed at run time would go to
    // local memory). The prologue issues the loads of K steps 0, 1 and 2 before K step 0's stores wait for their data
    // (nk >= 3: the launch's contract).
    fetch(ra[0], rb[0]);
    fetch(ra[1], rb[1]);
    int4 ra2 = zero, rb2[2];   // K step 2, held apart until K step 0 is stored
    fetch(ra2, rb2);
    store(0, 0);
    ra[0] = ra2;
    rb[0][0] = rb2[0];
    rb[0][1] = rb2[1];
    __syncthreads();
    auto step = [&](int i, int st) {   // st = i % 2
        compute(st);
        if (i + 1 < nk) store(1 - st, 1 - st);
        __syncthreads();
        if (i + 3 < nk) fetch(ra[1 - st], rb[1 - st]);
    };
    for (int i = 0; i < nk; i += 2) {
        step(i, 0);
        if (i + 1 < nk) step(i + 1, 1);
    }

    // The output: fp16 through shared memory (the loop's last barrier follows every read of the stages), then 16-byte
    // chunks: chunk t + 256 u is row (t + 256 u) / kBChunks, chunk (t + 256 u) % kBChunks of the band.
    half* Os = shmem;
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int n = 0; n < G::kTiles; n++) {
        half* o = &Os[g * G::kOPitch + warpId * G::kWarpCols + 8 * n + 2 * t];
        *(__half2*)o = __floats2half2_rn(c[n][0], c[n][1]);
        *(__half2*)(o + 8 * G::kOPitch) = __floats2half2_rn(c[n][2], c[n][3]);
    }
    __syncthreads();
    int4 mine[G::kOPer];
#pragma unroll
    for (int u = 0; u < G::kOPer; u++) {
        const int ch = threadIdx.x + G::kThreads * u;
        mine[u] = *((const int4*)&Os[ch / G::kBChunks * G::kOPitch] + ch % G::kBChunks);
    }
    if constexpr (!kParts) {   // one slice: straight to C
#pragma unroll
        for (int u = 0; u < G::kOPer; u++) {
            const int ch = threadIdx.x + G::kThreads * u, r = ch / G::kBChunks;
            if (r < M) *((int4*)&C[(size_t)r * N + col0] + ch % G::kBChunks) = mine[u];
        }
    } else {   // several slices: the block's part, then the band's counter; the last block sums the parts in order
        const int slices = gridDim.x, first = b * slices;   // the band's first block, in the order of the blocks
        half* part = parts + (size_t)(first + s) * 16 * BN;
#pragma unroll
        for (int u = 0; u < G::kOPer; u++) {
            const int ch = threadIdx.x + G::kThreads * u;
            *((int4*)&part[ch / G::kBChunks * BN] + ch % G::kBChunks) = mine[u];
        }
        fence();
        __syncthreads();
        unsigned* last = (unsigned*)((char*)shmem + G::kFlagByte);
        if (threadIdx.x == 0) *last = arrive(&counts[b]) == (unsigned)slices - 1;
        __syncthreads();
        if (!*last) return;
        fence();
#pragma unroll
        for (int u = 0; u < G::kOPer; u++) {
            const int ch = threadIdx.x + G::kThreads * u, r = ch / G::kBChunks, cc = ch % G::kBChunks;
            float v[8];
#pragma unroll
            for (int e = 0; e < 8; e++) v[e] = 0.0f;
#pragma unroll 1   // short code (the last block of a band runs it once per call)
            for (int ss = 0; ss < slices; ss++) {
                const int4 x = ss == s ? mine[u]
                                       : ld16_part((const int4*)&parts[((size_t)(first + ss) * 16 + r) * BN] + cc);
                const __half2* h = reinterpret_cast<const __half2*>(&x);
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    const __half2 hq = h[q];
                    v[2 * q] += __half2float(hq.x);
                    v[2 * q + 1] += __half2float(hq.y);
                }
            }
            int4 out;
            __half2* o = reinterpret_cast<__half2*>(&out);
#pragma unroll
            for (int q = 0; q < 4; q++) o[q] = __floats2half2_rn(v[2 * q], v[2 * q + 1]);
            if (r < M) *((int4*)&C[(size_t)r * N + col0] + cc) = out;
        }
        if (threadIdx.x == 0) counts[b] = 0;   // for the next call
    }
}

template <int BN, bool kParts, bool kEvictFirst>
__global__ void STEP6_BAND_BOUNDS gemm_bands(const half* A, const half* B, half* C, int M, int N, int K, int perSlice,
                                             int longer, half* parts, unsigned* counts)
{
    extern __shared__ __align__(128) half shmem[];
    band<BN, kParts, kEvictFirst>(shmem, A, B, C, M, N, K, perSlice, longer, parts, counts);
}

// ---------------------------------------------------------------- the rules (host)
// 6b's rule, from N, K and the number of SMs P: the band's width and the number of slices of K (the header).
struct Bands { int bn, slices; };
inline Bands bands(int N, int K, int sms)
{
    const int lo = (12 * sms + 9) / 10, mid = 2 * sms, hi = 3 * sms;   // 1.2 P, 2 P, 3 P blocks (48, 80, 120 on the T4)
    Bands p{128, 1};
    if (N / 128 >= lo && N / 128 <= hi) return p;     // enough bands of 128 columns, without parts
    if (N % 256) return p;                            // 128 columns, one slice (several rounds if N / 128 > 3 P)
    const int nb = N / 256;
    p.bn = 256;
    if (nb > hi) return p;                            // 256 columns, one slice, several rounds
    while (nb * p.slices < lo) p.slices++;                                              // enough blocks
    while (nb * (p.slices + 1) <= mid && 1600 * (p.slices + 1) <= K) p.slices++;       // long K: 16 S / K <= 0.01
    while (p.slices > 1 && K / (16 * p.slices) < 4) p.slices--;                         // at least 4 K steps per slice
    return p;
}

// The kernel that runs a shape, and its tile or band. Each mechanism can be switched off (the variants of step6.cu).
enum Kind { kGrid, kSplit, kBands };
struct Plan { Kind kind; int bm, bn, slices; };
inline Plan plan(int M, int N, int K, int sms, bool with6b, bool with6c)
{
    if (with6b && M <= 16) {   // 6b, when the shape meets the band kernel's contract
        const Bands p = bands(N, K, sms);
        const int steps = K / (4096 / p.bn), blocks = N / p.bn * p.slices;
        if (N % p.bn == 0 && K % (4096 / p.bn) == 0 && steps / p.slices >= 3 && (p.slices == 1 || blocks <= 3 * sms))
            return {kBands, 16, p.bn, p.slices};
    }
    const int tiles256 = (M + 127) / 128 * (N / 256);
    if (with6c && M > 64 && 2 * tiles256 < sms && K % 128 == 0) return {kSplit, 128, 256, 2};   // 6c
    return {kGrid, step5::choose(M, true), 256, 1};
}

}  // namespace step6
