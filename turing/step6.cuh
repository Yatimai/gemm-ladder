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

// --------------------------------------------- step 6: L2-only loads, 16-row bands, split-K in two, synchronized waves
// Step 6 grafts six mechanisms on step 5 as it is, measured by the interleaved probe (each mechanism removed against
// step 6), ncu and the judge; the measures are in the README, step 6.
// (6a) Loads through the L2 only: compute_tile loads A and B with ld.global.cg.L2::128B, not 3a's
//      ld.global.nc.L2::128B; the SASS changes LDG.E.LTC128B.128.CONSTANT.SYS into LDG.E.LTC128B.128.STRONG.GPU (and a
//      few integer instructions, below). At qkv_m128 and gateup_m128, 6a acts with 6e and 6f only.
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
//      cut in two halves of K, the tiles give 32 halves, one per block. The rule: 2 slices where step 5 takes 128 rows
//      (M > 64) and its 128 x 256 tiles are fewer than half the SMs (where v3 takes 128 x 128), K a multiple of 128.
//      One block per half (6e's grid at the halves): block 2 t + s computes tile t over K's half s (compute_tile over
//      K / 2 columns, A and B moved to the half s) and stores it in fp16 (fp16
//      parts, this rung's choice) into its slice's part, an M x N matrix; a fence, one atomic increment of the tile's
//      counter per half; the block that arrives second sums the two parts in the order of the slices, in fp32, writes C
//      and puts the counter back to zero. The same bits whatever the order of arrival. Its parts are 2 MiB at o_m128.
//      The half is chosen by moving A and B, so that compute_tile's loop runs from 0 to K / 2 at every place.
// (6d) Synchronized waves, where step 6 runs 6e's grid at the 128 x 256 tile and 3b's groups of tiles span at
//      least 4 full rows of tiles (M >= 512: the four M = 512 and the four M = 2048 shapes, and ladder): in place of
//      6e's grid, min(tiles, P) blocks of a persistent grid (one block per SM, all the 128 x 256 tile allows), block b
//      computing the places b, b + P, ... of step 5's order, wave w the places [P w, P w + P), a barrier of the whole
//      grid between two waves. 6d is kept at every shape where the rule applies: no exception per shape.
//      The barrier (grid_barrier below) is a counter and a generation in two words of their own (step6.cu), the
//      counter back at zero after each barrier, hence after each call; its waiting threads spin on the generation
//      without a pause. Every block must be resident at once: step6.cu checks at its first call that one block of
//      gemm_waves fits in an SM (cudaOccupancyMaxActiveBlocksPerMultiprocessor), and otherwise runs 6e's grid. The
//      launch is not cooperative: SMs taken by another context could leave a block waiting to start while the others
//      wait at the barrier.
// (6e) One block per tile where step 6 runs tiles: a grid of the tiles, block b the tile at place b of step 3's order
//      (3b), in place of step 5's persistent grid (one block per SM, each block looping over the tiles at places b,
//      b + P, b + 2 P, with P SMs); at 6c's halves, one block per half. When a block ends, the hardware starts the next
//      one on its SM, so an SM that finishes first takes the next tile; the block needs no loop of tiles. At the
//      judge's shapes: qkv_m128 and gateup_m128 (the tiles), o_m128 and down_m128 (6c's halves); 6b keeps its bands,
//      6d its waves.
// (6f) An order of the HMMA that reuses an operand, in every tile of step 6 (the tiles, 6c's halves and 6d's waves: the
//      13 shapes M >= 128 at the judge's): a k8 slice's 32 HMMA in the order of compute_tile's mma, the 8 columns in
//      turn and the 4 rows forward then back, so that B's operand is reused in 3 HMMA of 4, in place of step 2's order
//      (the rows, then the columns). The same instructions, in another order; each accumulator sums its k8 slices in
//      the same order, hence the same bits.
// Kept from step 5: compute_tile's tile (4a, 2b's swizzle, the output 64 rows at a time, 3b's order of the tiles, the
// edge in M) and its rows (5a: 64 where M <= 64) wherever 6b, 6c and 6d do not apply; v2 and v3 take 128 columns where
// 256 would give fewer blocks than half the SMs (plan below). Shapes: M a multiple of 16, N of 256, K of 64 (checked
// on the host); 6b runs at M = 16 when its contract holds (N a multiple of its width, K of its K step, at least 3 K
// steps per slice, at most 3 P blocks when there are slices: the workspace's ceiling), 6c where M > 64, the 128 x 256
// tiles are fewer than half the SMs and K is a multiple of 128.
// The workspace (step6.cu): allocated once, at the first call, at the ceilings of the rules, never freed. Not for
// concurrent calls.
// Under GEMM_LADDER_EMULATE (turing/emu), which runs one block at a time: the loads are plain 16-byte copies, the
// atomic a plain increment, the fence nothing.
// The counts below are read with nvcc 13.2 (a local compilation); the measured binaries are built by nvcc 13.1.1 (the
// Modal images).
// Compiled (nvcc 13.2, sm_75): the tiles with 6a, 6e and 6f, 362 / 367 instructions per K64 at 128 x 256, 382 / 386 at
// 128 x 128, 405 / 405 at 64 x 256 and 429 / 433 at 64 x 128 (full / edge tiles), and 255 registers (254 at 64 x 128,
// 246 at 64 x 256); without 6a (v1), the same counts per K64 and, the loads aside, the same opcodes, but for 2 integer
// instructions of another kind at 128 x 256 and 8 at 128 x 128 (242 registers at 64 x 256). gemm_split (with 6a or
// without): 255 registers, 432 / 432 instructions per K64; no spill. 6b (__launch_bounds__(256, 3): at most 80
// registers): 70 registers at BN 256, 66 at BN 128; one slice / several: 272 / 448 instructions at BN 256, 240 / 344 at
// BN 128; the first load after 37 instructions, the first barrier after 86 (83 at BN 128), 77 (73) in the loop of two K
// steps; 17 424 and 18 448 bytes of shared memory. gemm_waves (6d): 254 registers, no spill, with 6a or without; its
// loop of K, the tile inside a loop of tiles, has 370 / 370 instructions per K64 with 6a; it adds the number of waves
// (a division) and the barrier (one atomic add and one reduction, two strong loads of the generation, three fences).
// v7's kernels (with 6a, no spill): gemm_persistent, the tile inside a loop of tiles, 254, 255, 252 and 254 registers
// and 370 / 370, 409 / 411, 398 / 398 and 416 / 408 instructions per K64 at 128 x 256, 128 x 128, 64 x 256 and 64 x
// 128; gemm_split_persistent, 252 registers, 430 / 430. v8's kernels (gemm_rows at the four sizes, gemm_waves_rows,
// gemm_split_rows): v0's opcodes and counts, without the 192 .reuse flags per K64 that 6f gives v0's HMMA.
// Variants (g_var in step6.cu, for the interleaved probe): v0 = step 6 (6a to 6f); v1 = without 6a (nc in
// compute_tile; 6b keeps its loads, 6d its waves); v2 = without 6b (64 x 256 tiles at qkv_m16 and gateup_m16, 64 x 128
// at o_m16 and down_m16); v3 = without 6c (128 x 128 tiles at o_m128 and down_m128); v4 = step 5 itself, the anchor;
// v5 = 6b with B never marked evict-first (through the L2 only, as with one slice); v6 = without 6d (6e's grid at
// M >= 512); v7 = without 6e: v0, with step 0's persistent grid (one block per SM) where v0 runs 6e's grid, its tiles
// (gemm_persistent) and 6c's halves (gemm_split_persistent); v8 = without 6f: v0, a slice's HMMA in step 2's order in
// every tile (gemm_rows, gemm_waves_rows, gemm_split_rows). v2, v3 and v6 keep 6a, 6e and 6f in the tiles they fall
// back to (v1 and v4 do not keep 6a); v2 and v3 take there 128 columns where 256 would give fewer blocks than half the
// SMs, the best tile without 6b or 6c (plan below). v0 against v1 measures 6a at M >= 128 and its part in 6d; v0
// against v2, v3 and v6 6b, 6c and 6d; v0 against v4 the step; v0 against v5 the evict-first marking, where 6b has
// slices (o_m16 and down_m16); v0 against v7 6e (qkv_m128, gateup_m128, o_m128 and down_m128 at the judge's shapes);
// v0 against v8 6f (the 13 shapes M >= 128).

// ---------------------------------------------------------------- step 6's tiles
// A tile of BM x BN (BM 64 or 128, BN 128 or 256) in warps of 64 x 64: 32 x (BM / 64) x (BN / 64) threads. Step 5's
// geometry (step5.cuh) at BN columns: step 2's swizzled stages of K32 and its output through shared memory, 64 rows at
// a time; at BN = 256, step 5's two tiles, number for number. The 128-column tiles serve v2 and v3 only (plan below).
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

// Step 3's order of the tiles (3b), for any tile: groups of 8 rows of tiles, column by column.
template <int BM, int BN>
__device__ __forceinline__ void tile_of(int b, int M, int N, int& tm, int& tn)
{
    const int tilesN = N / BN, tilesM = (M + BM - 1) / BM, per = 8 * tilesN, first = b / per * 8;
    const int rows = min(tilesM - first, 8), r = b % per;
    tm = first + r % rows;
    tn = r / rows;
}

// The tile where 6b, 6c and 6d do not run: 64 rows of M when M <= 64 (5a), 128 otherwise; with cols128, 128 columns
// when 256 would give fewer blocks than half the SMs, 256 otherwise (plan below asks for cols128 in v2 and v3 only).
struct Choice { int bm, bn; };
inline Choice choose(int M, int N, int sms, bool rows64, bool cols128)
{
    const int bm = rows64 && M <= 64 ? 64 : 128;
    const int tiles256 = (M + bm - 1) / bm * (N / 256);
    return {bm, cols128 && 2 * tiles256 < sms ? 128 : 256};
}

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

// The barrier of the whole grid between two waves (6d, gemm_waves). Under GEMM_LADDER_EMULATE, one block at a time:
// the block's barriers on each side of a call that the emulator counts (no block may wait for another; the waves'
// tiles are independent, so the result is the same; the emulator checks that every block reaches the barrier as many
// times).
#ifdef GEMM_LADDER_EMULATE
void emu_grid_barrier();   // turing/emu: counts the running block's arrivals at the grid barrier
inline void grid_barrier(unsigned*, unsigned)
{
    __syncthreads();
    if (threadIdx.x == 0) emu_grid_barrier();
    __syncthreads();
}
#else
// sync[0]: the counter, sync[1]: the generation. Thread 0 of each block reads the generation, then arrives; the last to
// arrive puts the counter back to zero and then moves the generation on; the others wait until it moves.
__device__ __forceinline__ void grid_barrier(unsigned* sync, unsigned blocks)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned* gen = sync + 1;
        const unsigned g = *gen;
        __threadfence();
        if (atomicAdd(sync, 1u) == blocks - 1) {
            sync[0] = 0;
            __threadfence();
            atomicAdd(sync + 1, 1u);
        } else {
            while (*gen == g) {}
        }
        __threadfence();
    }
    __syncthreads();
}
#endif

// ---------------------------------------------------------------- 6a, 6c, 6f: step 5's tile
// One BM x BN tile, step 5's compute_tile (step5.cuh) at BN columns, with three changes: 6a's loads (kCg), the K
// columns [0, kEnd) only (kEnd a multiple of 64: K / 2 in 6c, A and B moved to the half; K elsewhere; K stays the
// length of A's rows), and 6f's order of a slice's HMMA (kOrder; false, v8: step 2's order, the rows then the columns).
template <int BM, int BN, bool kEdge, bool kCg, bool kOrder = true>
__device__ __forceinline__ void compute_tile(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                             int row0, int col0, int kEnd)
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

    // 4a, as step 5, and 6f's order (kOrder).
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
        if constexpr (kOrder) {   // 6f's order: the 8 columns, the rows back and forth
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int ii = 0; ii < 4; ii++) hmma(f, j % 2 ? 3 - ii : ii, j);
        } else {   // v8: step 2's order, the rows, then the columns
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) hmma(f, i, j);
        }
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

// Tile t (step 5's order, 3b) over the K columns [0, kEnd) of A and B into C or a part; kOrder: 6f (false: v8).
template <int BM, int BN, bool kCg, bool kOrder = true>
__device__ __forceinline__ void tile(half* shmem, const half* A, const half* B, half* C, int M, int N, int K, int t,
                                     int kEnd)
{
    int tm, tn;
    tile_of<BM, BN>(t, M, N, tm, tn);
    const int row0 = tm * BM, col0 = tn * BN;
    if (row0 + BM <= M) compute_tile<BM, BN, false, kCg, kOrder>(shmem, A, B, C, M, N, K, row0, col0, kEnd);
    else compute_tile<BM, BN, true, kCg, kOrder>(shmem, A, B, C, M, N, K, row0, col0, kEnd);
}

// 6e's grid (one block per BM x BN tile, block b the tile at place b of the order), with 6a's loads.
template <int BM, int BN, bool kCg>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    tile<BM, BN, kCg>(shmem, A, B, C, M, N, K, blockIdx.x, K);
}

// v7, without 6e: step 0's persistent grid (one block per SM, block_pos += gridDim.x) over step 5's tiles in its order,
// with 6a's loads.
template <int BM, int BN, bool kCg>
__global__ void gemm_persistent(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    const int tiles = (M + BM - 1) / BM * (N / BN);
    for (int block_pos = blockIdx.x; block_pos < tiles; block_pos += gridDim.x)
        tile<BM, BN, kCg>(shmem, A, B, C, M, N, K, block_pos, K);
}

// v8, without 6f: 6e's grid as gemm, a slice's HMMA in step 2's order.
template <int BM, int BN, bool kCg>
__global__ void gemm_rows(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    tile<BM, BN, kCg, false>(shmem, A, B, C, M, N, K, blockIdx.x, K);
}

// 6d, the waves: a persistent grid of at most one block per SM over step 5's 128 x 256 tiles in its order, block b the
// places b, b + gridDim.x, ..., wave w the places [w gridDim.x, (w + 1) gridDim.x), with a barrier of the whole grid
// between two waves (every block must be resident at once: step6.cu checks that one block of gemm_waves fits in an SM;
// the launch is not cooperative: SMs taken by another context could leave a block waiting to start while the others
// wait at the barrier).
template <int BM, bool kCg, bool kOrder>
__device__ __forceinline__ void waves_of(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                         unsigned* sync)
{
    const int tiles = (M + BM - 1) / BM * (N / 256), waves = (tiles + gridDim.x - 1) / gridDim.x;
    for (int w = 0; w < waves; w++) {
        const int t = w * gridDim.x + blockIdx.x;
        if (t < tiles) tile<BM, 256, kCg, kOrder>(shmem, A, B, C, M, N, K, t, K);
        if (w + 1 < waves) grid_barrier(sync, gridDim.x);   // compute_tile ends at a barrier: the stages are free
    }
}

template <int BM, bool kCg>
__global__ void gemm_waves(const half* A, const half* B, half* C, int M, int N, int K, unsigned* sync)
{
    extern __shared__ __align__(128) half shmem[];
    waves_of<BM, kCg, true>(shmem, A, B, C, M, N, K, sync);
}

// v8, without 6f: the waves, a slice's HMMA in step 2's order.
template <int BM, bool kCg>
__global__ void gemm_waves_rows(const half* A, const half* B, half* C, int M, int N, int K, unsigned* sync)
{
    extern __shared__ __align__(128) half shmem[];
    waves_of<BM, kCg, false>(shmem, A, B, C, M, N, K, sync);
}

// 6c: two halves of K per tile, one block per half (6e's grid at the halves): block 2 t + s computes tile t over K's
// half s (A and B moved to it: see (6c) above) into its part (an M x N matrix of halves); the block that arrives second
// at a tile sums the two parts in the order of the slices, in fp32, and writes C. split_half: tile t over K's half s
// into its part, then the block arrives; true for the block that arrives second. split_sum: that block's sum.
template <int BM, bool kCg, bool kOrder>
__device__ __forceinline__ bool split_half(half* shmem, const half* A, const half* B, half* parts, int M, int N,
                                           int K, int halfK, int t, int s, unsigned* counts, unsigned* last)
{
    tile<BM, 256, kCg, kOrder>(shmem, A + (size_t)s * halfK, B + (size_t)s * halfK * N, parts + (size_t)s * M * N, M,
                               N, K, t, halfK);
    fence();          // every thread's stores of the part, before the block arrives
    __syncthreads();
    if (threadIdx.x == 0) *last = arrive(&counts[t]) == 1u;
    __syncthreads();
    return *last;
}

template <int BM>
__device__ __forceinline__ void split_sum(const half* parts, half* C, int M, int N, int t, unsigned* counts)
{
    using G = Geometry<BM, 256>;
    fence();
    int tm, tn;
    tile_of<BM, 256>(t, M, N, tm, tn);
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

template <int BM, bool kCg, bool kOrder>
__device__ __forceinline__ void split_block(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                            half* parts, unsigned* counts)
{
    using G = Geometry<BM, 256>;
    const int halfK = K / 2, t = blockIdx.x / 2, s = blockIdx.x % 2;
    unsigned* last = (unsigned*)((char*)shmem + G::kSmemBytes);   // a word after the tile's buffers (kSmemBytes + 16)
    if (!split_half<BM, kCg, kOrder>(shmem, A, B, parts, M, N, K, halfK, t, s, counts, last)) return;
    split_sum<BM>(parts, C, M, N, t, counts);
}

template <int BM, bool kCg>
__global__ void gemm_split(const half* A, const half* B, half* C, int M, int N, int K, half* parts, unsigned* counts)
{
    extern __shared__ __align__(128) half shmem[];
    split_block<BM, kCg, true>(shmem, A, B, C, M, N, K, parts, counts);
}

// v8, without 6f: 6c's halves as gemm_split, a slice's HMMA in step 2's order.
template <int BM, bool kCg>
__global__ void gemm_split_rows(const half* A, const half* B, half* C, int M, int N, int K, half* parts,
                                unsigned* counts)
{
    extern __shared__ __align__(128) half shmem[];
    split_block<BM, kCg, false>(shmem, A, B, C, M, N, K, parts, counts);
}

// v7, without 6e: 6c's halves on step 0's persistent grid (one block per SM, block_pos += gridDim.x, block_pos the
// half block_pos % 2 of tile block_pos / 2), each as gemm_split computes it.
template <int BM, bool kCg>
__global__ void gemm_split_persistent(const half* A, const half* B, half* C, int M, int N, int K, half* parts,
                                      unsigned* counts)
{
    using G = Geometry<BM, 256>;
    extern __shared__ __align__(128) half shmem[];
    const int tiles = (M + BM - 1) / BM * (N / 256), halfK = K / 2;
    unsigned* last = (unsigned*)((char*)shmem + G::kSmemBytes);   // a word after the tile's buffers (kSmemBytes + 16)
    for (int block_pos = blockIdx.x; block_pos < 2 * tiles; block_pos += gridDim.x)
        if (split_half<BM, kCg, true>(shmem, A, B, parts, M, N, K, halfK, block_pos / 2, block_pos % 2, counts, last))
            split_sum<BM>(parts, C, M, N, block_pos / 2, counts);
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
enum Kind { kGrid, kSplit, kBands, kWaves };
struct Plan { Kind kind; int bm, bn, slices; };
inline Plan plan(int M, int N, int K, int sms, bool with6b, bool with6c, bool with6d)
{
    if (with6b && M <= 16) {   // 6b, when the shape meets the band kernel's contract
        const Bands p = bands(N, K, sms);
        const int steps = K / (4096 / p.bn), blocks = N / p.bn * p.slices;
        if (N % p.bn == 0 && K % (4096 / p.bn) == 0 && steps / p.slices >= 3 && (p.slices == 1 || blocks <= 3 * sms))
            return {kBands, 16, p.bn, p.slices};
    }
    const int tiles256 = (M + 127) / 128 * (N / 256);
    if (with6c && M > 64 && 2 * tiles256 < sms && K % 128 == 0) return {kSplit, 128, 256, 2};   // 6c
    // Where 6b or 6c would run but is switched off (v2, v3), the best tile without it: 128 columns when 256 would give
    // fewer blocks than half the SMs; elsewhere step 5's tile (5a's rows, 256 columns).
    const bool fallback = (!with6b && M <= 16) || (!with6c && M > 64 && 2 * tiles256 < sms && K % 128 == 0);
    const Choice ch = choose(M, N, sms, true, fallback);
    if (with6d && ch.bm == 128 && ch.bn == 256 && M >= 4 * ch.bm) return {kWaves, 128, 256, 1};   // 6d: the waves
    return {kGrid, ch.bm, ch.bn, 1};
}

}  // namespace step6
