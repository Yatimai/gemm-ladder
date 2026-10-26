/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through steps 0 to 5 (nvidia_sample.cuh, step1.cuh to step5.cuh),
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
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "step5.cuh"
#ifdef GEMM_LADDER_EMULATE
#include <cstring>
#endif

namespace step6 {

// --------------------------------------------------------- step 6: the tile per shape, 16-row bands, a stream in phase
// Steps 1 to 5 run one tile, 128 x 256 (in three stages of K64 from step 3), at every shape. Step 6 adds a tile (6a),
// through one kernel at either tile (step 5's loop, 5a's pipeline of the fragments and 5b's order of the HMMA, with
// warps of 64 x 64), and a rule that picks, for each shape, the tile: the plan (choose). Two mechanisms follow (6b and
// 6c), each with its kernel and its rule; the plan's kernels stay where neither applies. Step 6 keeps K whole in the
// plan's tiles. Step 7 splits it (7a, and 7b on its parts; step7.cuh) at four of the judge's shapes, where cuBLASLt
// splits K too (its configurations kept there for the measured card set CUBLASLT_ALGO_CONFIG_SPLITK_NUM above 1); the
// kernels of 7a, 7b and 7e (the tail of the waves cut in K) are written here, beside the plan's, and step 6 does not
// run them. The kernels of 6b, 6c, 7b and 7e carry step 7's L2 cache hints (7d, step7.cuh) behind a template parameter,
// kHint, and those of 7a's parts in place and of 6c step 7's reads together (7h) behind another, kTogether: step 6 runs
// those of 6b and 6c without either. The judge's 17 shapes: qkv (N 10240, K 8192), o (8192, 8192), gateup (57344,
// 8192), down (8192, 28672), each at M = 16, 128, 512 and 2048, and the ladder (3456 x 4096 x 2048).
// (6a) A 128 x 128 tile, 4 warps of 64 x 64, five stages of K32 (80 KB of shared memory: two blocks fit on an SM), when
//      128 x 256 would give fewer than three waves of tiles: at M = 16 and 128 (qkv, o, gateup, down) and at qkv, o
//      and down M = 512. A's rows of K32 (64 bytes, two to a 128-byte line) are swizzled by pairs: chunk c of row r at
//      c ^ ((r >> 1) & 3).
// (6b) M <= 16: a tile of 16 rows (the plan's tiles of 128 rows spend 7 HMMA of 8 on rows >= M, the rows of A each
//      thread copies clamped to M - 1), 4 warps over a band of B.
//      Bands: one band of 128 columns per block over all of K, 6 stages of K64, when the bands fit in one wave and
//      either cover at least 2/3 of the SMs or their work, 16 x 128 x K, stays under 2^25; otherwise a continuous
//      stream: sms blocks, bands of 256 columns and 4 stages, each block an equal range of (band, slice) units; a band
//      split between ranges leaves fp16 parts that the last block to arrive adds. cuBLASLt's API has 16-row tiles
//      (CUBLASLT_MATMUL_TILE_128x16 and others, in its column-major view).
// (6c) 16 < M <= 512, whole tiles over at least two waves whose last wave holds fewer than half the SMs: a continuous
//      stream in phase, groups of ceil(M / 128) blocks taking B's slices in one order, parts at the ends of the
//      ranges, the last block to arrive adds them.
// The rules, by classes of shapes; their thresholds are our choices, each a mechanical quantity of the shape and the
// card (sms, the SM count, read at the first call); where a setting documented in cuBLASLt's API bears on a mechanism,
// it is named with it:
//   the plan: 128 x 128 when 128 x 256 would give fewer than 3 sms tiles (6a), 128 x 256 otherwise (step 5's kernel,
//   the same SASS), K whole;
//   6b first, at M <= 16; then 6c by its rule on the plan's tiles, but not where 7a's rule (step7.cuh) splits K: 6c
//   takes the same shapes in steps 6 and 7; then the plan.
// The rules give, at the judge's 17 shapes: 6b's bands at qkv_m16 and o_m16, its stream at gateup_m16 and down_m16;
// 6c at gateup_m128 and gateup_m512; 6a's tile at qkv_m128, qkv_m512, o_m128, o_m512, down_m128 and down_m512;
// qkv_m2048, o_m2048, gateup_m2048, down_m2048 and the ladder keep step 5's 128 x 256.
// The counts below are read with nvcc 13.2 (a local compilation); the measured binaries are built by nvcc 13.1.1 (the
// Modal images).
// Compiled (nvcc 13.2, sm_80), no spill. The plan's kernels, main loop per slice and per warp (static counts):
//                             registers    slice   instructions   HMMA   LDSM.x4   LDGSTS   shared memory
//   128 x 256 (step 5)        250          K64     270            128    32        12       147 456 bytes
//   128 x 128 (6a)            242          K32     164            64     16        8        81 920
// The kernels of 6b and 6c as step 6 runs them (without 7d's hints), main loop per slice of K64 and per warp (static
// counts; a kernel with several loops, for tiles cut by M and, in 6c, for whole tiles and parts, gives their range):
//                                 registers   instructions   HMMA   LDSM.x4   LDGSTS   barriers
//   6b bands (16 x 128)           48          101            16     12        9        1
//   6b stream (16 x 256)          128         171            32     20        17       1
//   6c (128 x 256)                254         264 to 288     128    32        12       1
// Variants (g_var in step6.cu, for the interleaved probe; each removal keeps the other mechanisms):
//   v0 = step 6 (6a, 6b, 6c); v1 = without 6a (128 x 256 at every shape); v2 = without 6b; v3 = without 6c; v4 =
//   step 5 itself (step5::gemm<true, true>), the anchor; v5 = 6b's bands at every shape M <= 16; for verif.cu, whose
//   shapes are too small for the rules to choose them: v6 = 6b's stream at every shape M <= 16; v7 = 6c at every shape
//   M > 16.

using step3::cp_async16;
using step3::cp_async_commit;
using step3::cp_async_wait;

// 7a: what a block does with its share of K. kWhole: all of K, the tile to C (6a); kParts: one part of K, its
// result in fp16 to its own place in the workspace, a second kernel sums the parts (as the reduction scheme
// CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE); kInPlace: one part of K, added to C in place, the parts of a tile in the
// order of K, each waiting for the previous on the tile's flag (as CUBLASLT_REDUCTION_SCHEME_INPLACE).
enum Split { kWhole, kParts, kInPlace };

#ifdef GEMM_LADDER_EMULATE
// A host emulation runs the rows of parts one after the other and suspends a block where it sets its flag (other
// blocks may run before its threads go on): a wait finds its flag set already, or the order is wrong; a block that
// sets its tile's flag must have waited on it, for the value before.
void emu_wait_flag(const unsigned* flag, unsigned value);
void emu_set_flag(unsigned* flag, unsigned value);
// It also checks that a read of another block's data (ld16_cg) comes after the block's acquisition (its wait on the
// tile's flag, its arrival on a count, its wait on a count) for the tile it computes (compute_tile marks its start).
void emu_tile_begin();
void emu_read_shared(const void* p);
#endif

// 7a, in place: a 16-byte load of C through the L2 only (the previous part of the tile was written by another block,
// on another SM: L1 does not see it).
__device__ __forceinline__ int4 ld16_cg(const int4* p)
{
#ifdef GEMM_LADDER_EMULATE
    emu_read_shared(p);
    return *p;
#else
    int4 v;
    asm volatile("ld.global.cg.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)
                 : "memory");
    return v;
#endif
}

// 7a, in place: thread 0 waits until the tile's flag reads `value` (acquire), then sets it (release) when its part is
// in C.
__device__ __forceinline__ void wait_flag(const unsigned* flag, unsigned value)
{
#ifdef GEMM_LADDER_EMULATE
    emu_wait_flag(flag, value);
#else
    unsigned x;
    do asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(x) : "l"(flag) : "memory");
    while (x != value);
#endif
}
__device__ __forceinline__ void set_flag(unsigned* flag, unsigned value)
{
#ifdef GEMM_LADDER_EMULATE
    emu_set_flag(flag, value);
#else
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(flag), "r"(value) : "memory");
#endif
}

// A fence for the in-place parts' stores of C before the flag (none in a host emulation: its stores are seen at once).
__device__ __forceinline__ void fence()
{
#ifndef GEMM_LADDER_EMULATE
    __threadfence();
#endif
}

// 8 halves plus 8 halves, each sum in fp32 rounded once to fp16 (the parts' additions, 7a).
__device__ __forceinline__ int4 add8(int4 a, int4 b)
{
    half x[8], y[8];
    memcpy(x, &a, 16);
    memcpy(y, &b, 16);
#pragma unroll
    for (int e = 0; e < 8; e++) x[e] = __float2half(__half2float(x[e]) + __half2float(y[e]));
    int4 r;
    memcpy(&r, x, 16);
    return r;
}

// A tile of BM x BN (BM 64 or 128, BN 128 or 256) in warps of WM x 64 (WM 32 or 64; 32 x BM / WM x BN / 64 threads),
// with kStages stages of SK (32 or 64) along K. A's rows of SK halves are swizzled as step 2's (rows of 128 bytes:
// chunk c ^ (r & 7)) or, at SK = 32, by pairs of rows (rows of 64 bytes, two to a 128-byte line: chunk c ^ ((r >> 1) &
// 3)); B's rows of BN columns as step 2's. The output goes through shared memory 64 rows at a time, as step 2's, in
// rows of BN + 8 halves.
template <int BM, int BN, int WM, int SK, int kStages>
struct Geometry {
    static constexpr int kWarpsN = BN / 64, kThreads = 32 * (BM / WM) * kWarpsN, kFragsM = WM / 16;
    static constexpr int kAStride = SK, kBStride = BN, kOStride = BN + 8;   // halves
    static constexpr int kAStage = BM * SK, kBStage = SK * BN;
    static constexpr int kAAll = kStages * kAStage;
    static constexpr int kLoopBytes = kStages * (kAStage + kBStage) * 2, kOutBytes = 64 * kOStride * 2;
    static constexpr int kSmemBytes = kLoopBytes > kOutBytes ? kLoopBytes : kOutBytes;
    static_assert(kSmemBytes <= 163 * 1024, "above the 163 KB of shared memory of a block on sm_80");
    // The copy of a slice: thread t holds chunk t % kAChunks of rows t / kAChunks + kARows i of A (i < kAPer), and
    // chunk t % kBChunks of rows t / kBChunks + kBRows i of B (i < kBPer); the epilogue copies chunk t % kBChunks of
    // rows t / kBChunks + kBRows i (i < kOPer) of the 64 rows in shared memory.
    static constexpr int kAChunks = SK / 8, kBChunks = BN / 8;
    static constexpr int kARows = kThreads / kAChunks, kBRows = kThreads / kBChunks;
    static constexpr int kAPer = BM / kARows, kBPer = SK / kBRows, kOPer = 64 / kBRows;
    static_assert(kAPer * kARows == BM && kBPer * kBRows == SK && kOPer * kBRows == 64, "copy not spread evenly");
    __host__ __device__ static constexpr int a_at(int r, int c)
    {
        return r * kAStride + 8 * (SK == 64 ? c ^ (r & 7) : c ^ ((r >> 1) & 3));
    }
    __host__ __device__ static constexpr int b_at(int r, int c) { return r * kBStride + 8 * (c ^ (r & 7)); }
};

// Step 4's order of the tiles (4b), for any tile: groups of R rows of tiles (step4::kGroupRows), column by column.
template <int BM, int BN>
__host__ __device__ __forceinline__ void tile_of(int b, int M, int N, int& tm, int& tn)
{
    const int R = step4::kGroupRows, tilesN = N / BN, tilesM = (M + BM - 1) / BM, per = R * tilesN;
    const int first = b / per * R, rows = min(tilesM - first, R), r = b % per;
    tm = first + r % rows;
    tn = r / rows;
}

// The copy of a 16-byte chunk of A or B into a stage: step 3's cp.async. compute_tile takes it as a parameter, so that
// the kernels of 6c and 7b can take step 7's (7d, with L2 cache hints); this one leaves the code of 6a and 7a
// unchanged.
struct PlainCopy {
    __device__ __forceinline__ void a(half* smem, const half* gmem) const { cp_async16(smem, gmem); }
    __device__ __forceinline__ void b(half* smem, const half* gmem) const { cp_async16(smem, gmem); }
};

// One BM x BN tile: step 5's (5a's pipeline of the fragments, 5b's order of the HMMA, 4a's stores of C) at that
// geometry. kEdge: the tile is cut by M (the row of A each thread copies is clamped to M - 1). With kParts, the flag
// is not used: a later step can pass in its place a keeper of its own (step 7's 7g), which keeps some of the chunks of
// the part in registers instead of writing them (chunk threadIdx.x + kThreads u of the tile, u = 0, 1, ... in the
// order of the output); this step writes them all. Flag is never deduced: a null flag stays an unsigned*. kTogether
// (7h, in place): the reads of C issued together (below).
template <int BM, int BN, int WM, int SK, int kStages, int kSplit, bool kEdge, class Copy = PlainCopy,
          class Flag = unsigned*, bool kTogether = false>
__device__ __forceinline__ void compute_tile(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                             int row0, int col0, int kBegin, int steps, half* part,
                                             typename std::common_type<Flag>::type flag, int s, int S)
{
#ifdef GEMM_LADDER_EMULATE
    emu_tile_begin();
#endif
    using G = Geometry<BM, BN, WM, SK, kStages>;
    constexpr int kSteps16 = SK / 16, kFM = G::kFragsM, kH = 8 * kFM;   // K16 steps per slice; HMMA per K16
    half* As = shmem;             // kStages x BM rows of a slice of A
    half* Bs = shmem + G::kAAll;  // kStages x SK rows of a slice of B
    const int warpId = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warpId / G::kWarpsN, wn = warpId % G::kWarpsN;   // the warp's rows wm * WM, columns wn * 64
    float c[kFM][4][2][4];
#pragma unroll
    for (int i = 0; i < kFM; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) c[i][j][n][e] = 0.0f;
    int aOff[kSteps16], bOff[4];   // the ldmatrix offsets within a stage (step 2's rule)
#pragma unroll
    for (int ks = 0; ks < kSteps16; ks++) aOff[ks] = G::a_at(wm * WM + lane % 16, 2 * ks + lane / 16);
#pragma unroll
    for (int j = 0; j < 4; j++) bOff[j] = G::b_at(lane % 16, wn * 8 + 2 * j + lane / 16);

    const Copy copy;
    auto issue = [&](int k, int st) {   // step 3's copy of the block's slice k (of K at (kBegin + k) SK) into stage st
        const int k0 = (kBegin + k) * SK;
#pragma unroll
        for (int i = 0; i < G::kAPer; i++) {
            const int r = threadIdx.x / G::kAChunks + i * G::kARows;
            const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;
            copy.a(&As[st * G::kAStage + G::a_at(r, threadIdx.x % G::kAChunks)],
                   &A[(size_t)gr * K + k0 + 8 * (threadIdx.x % G::kAChunks)]);
        }
#pragma unroll
        for (int i = 0; i < G::kBPer; i++) {
            const int r = threadIdx.x / G::kBChunks + i * G::kBRows;
            copy.b(&Bs[st * G::kBStage + G::b_at(r, threadIdx.x % G::kBChunks)],
                   &B[(size_t)(k0 + r) * N + col0 + 8 * (threadIdx.x % G::kBChunks)]);
        }
    };
    uint32_t fa[2][kFM][4], fb[2][4][4];   // 5a's two sets of fragments
    auto load_frags = [&](int set, int st, int ks) {   // the K16 ks of stage st into set `set`
#pragma unroll
        for (int i = 0; i < kFM; i++)
            step2::ldmatrix_x4<false>(fa[set][i], &As[st * G::kAStage + aOff[ks] + i * 16 * G::kAStride]);
#pragma unroll
        for (int j = 0; j < 4; j++)
            step2::ldmatrix_x4<true>(fb[set][j], &Bs[st * G::kBStage + bOff[j] + ks * 16 * G::kBStride]);
    };
    auto mma_range = [&](int set, int h0, int h1) {   // 5b's order: A's fragment held for 8 HMMA, B's in a serpentine
#pragma unroll
        for (int h = h0; h < h1; h++) {
            const int i = h / 8, q = (i & 1) ? 7 - h % 8 : h % 8, j = q / 2, n = q % 2;
            step2::mma_m16n8k16(c[i][j][n], fa[set][i], fb[set][j][2 * n], fb[set][j][2 * n + 1]);
        }
    };
    // Step 5's loop at kStages stages of SK: the prologue issues slices 0 to kStages - 2; per slice, the next K16's
    // fragments load during the current K16's HMMA, slice k + kStages - 1 is issued during the first K16 into the stage
    // of slice k - 1, and the wait for slice k + 1 and the barrier sit in the middle of the last K16 (in the source).
#pragma unroll
    for (int st = 0; st < kStages - 1; st++) {
        if (st < steps) issue(st, st);
        cp_async_commit();
    }
    int read = 0, write = kStages - 1;
    cp_async_wait<kStages - 2>();
    __syncthreads();
    load_frags(0, 0, 0);
    for (int k = 0; k < steps; k++) {
#pragma unroll
        for (int ks = 0; ks < kSteps16; ks++) {
            const int cur = ks & 1;
            if (ks < kSteps16 - 1) load_frags(cur ^ 1, read, ks + 1);
            if (ks == 0) {
                if (k + kStages - 1 < steps) issue(k + kStages - 1, write);
                cp_async_commit();
            }
            if (ks < kSteps16 - 1) {
                mma_range(cur, 0, kH);
            } else {
                mma_range(cur, 0, kH / 2);
                cp_async_wait<kStages - 2>();
                __syncthreads();
                read = read + 1 == kStages ? 0 : read + 1;
                write = write + 1 == kStages ? 0 : write + 1;
                if (k + 1 < steps) load_frags(cur ^ 1, read, 0);
                mma_range(cur, kH / 2, kH);
            }
        }
    }
    cp_async_wait<0>();   // only empty groups are left
    __syncthreads();      // every warp is done with the last stage before the output reuses it
    // The output, 64 rows at a time: 4a's stores of C (kWhole); the part to the workspace, in the tile's own rows of BN
    // halves (kParts); with kInPlace, once the tile's previous part is in C (thread 0 waits on the flag), C plus this
    // part.
    if constexpr (kSplit == kInPlace && !kTogether) {
        if (threadIdx.x == 0) wait_flag(flag, (unsigned)s);
        __syncthreads();
    }
    half* Os = shmem;
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int h = 0; h < BM / 64; h++) {
        if (wm * WM / 64 == h) {   // the warps of rows [64 h, 64 h + 64)
#pragma unroll
            for (int i = 0; i < kFM; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
#pragma unroll
                    for (int n = 0; n < 2; n++) {
                        half* o = &Os[(wm * WM % 64 + i * 16 + g) * G::kOStride + wn * 64 + j * 16 + n * 8 + 2 * t];
                        *(__half2*)o = __floats2half2_rn(c[i][j][n][0], c[i][j][n][1]);
                        *(__half2*)(o + 8 * G::kOStride) = __floats2half2_rn(c[i][j][n][2], c[i][j][n][3]);
                    }
        }
        __syncthreads();
        if constexpr (kSplit == kInPlace && kTogether) {
            // 7h: the wait on the tile's flag once the first 64 rows are in shared memory; then, in each round, every
            // read of C issued before its additions and stores (one round trip of the L2, not one per chunk). The
            // reads stay after the wait and the barrier that follows it.
            if (h == 0) {
                if (threadIdx.x == 0) wait_flag(flag, (unsigned)s);
                __syncthreads();
            }
            int4 old[G::kOPer];
            if (s > 0) {
#pragma unroll
                for (int i = 0; i < G::kOPer; i++) {
                    const int gr = row0 + h * 64 + threadIdx.x / G::kBChunks + i * G::kBRows;
                    if (gr < M) old[i] = ld16_cg((const int4*)&C[(size_t)gr * N + col0] + threadIdx.x % G::kBChunks);
                }
            }
#pragma unroll
            for (int i = 0; i < G::kOPer; i++) {
                const int r = threadIdx.x / G::kBChunks + i * G::kBRows;   // row among the 64
                const int gr = row0 + h * 64 + r;
                if (gr < M) {
                    const int ch = threadIdx.x % G::kBChunks;
                    int4 v = *((const int4*)&Os[r * G::kOStride] + ch);
                    int4* dst = (int4*)&C[(size_t)gr * N + col0] + ch;
                    if (s > 0) v = add8(v, old[i]);
                    if (s + 1 == S) step4::st16<true>(dst, v);
                    else step4::st16<false>(dst, v);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < G::kOPer; i++) {
                const int r = threadIdx.x / G::kBChunks + i * G::kBRows;   // row among the 64
                const int gr = row0 + h * 64 + r;
                if (gr < M) {
                    const int ch = threadIdx.x % G::kBChunks;
                    int4 v = *((const int4*)&Os[r * G::kOStride] + ch);
                    int4* dst = (int4*)&C[(size_t)gr * N + col0] + ch;
                    if constexpr (kSplit == kParts && std::is_same_v<Flag, unsigned*>) {
                        step4::st16<false>((int4*)&part[(size_t)(h * 64 + r) * BN] + ch, v);
                    } else if constexpr (kSplit == kParts) {   // a keeper in the flag's place
                        if (!flag->keep(h * G::kOPer + i, v))
                            step4::st16<false>((int4*)&part[(size_t)(h * 64 + r) * BN] + ch, v);
                    } else if constexpr (kSplit == kInPlace) {
                        if (s > 0) v = add8(v, ld16_cg(dst));
                        if (s + 1 == S) step4::st16<true>(dst, v);
                        else step4::st16<false>(dst, v);
                    } else {
                        step4::st16<true>(dst, v);
                    }
                }
            }
        }
        __syncthreads();
    }
    if constexpr (kSplit == kInPlace) {   // every thread's stores of C, then the flag: the next part, or 0 for the next
                                          // call
        fence();
        __syncthreads();
        if (threadIdx.x == 0) set_flag(flag, s + 1 == S ? 0u : (unsigned)s + 1);
    }
}

// Block b of a grid of S x tiles: part s = b / tiles of tile b % tiles (step 4's order of the tiles); the parts of a
// tile share its K slices, the first ones one more when they do not divide evenly. kWhole: S = 1.
template <int BM, int BN, int WM, int SK, int kStages, int kSplit, class Copy = PlainCopy, bool kTogether = false>
__device__ __forceinline__ void run(half* shmem, const half* A, const half* B, half* C, int M, int N, int K, int S,
                                    half* parts, unsigned* flags)
{
    const int tilesN = N / BN, tiles = (M + BM - 1) / BM * tilesN;
    const int s = blockIdx.x / tiles;
    int tm, tn;
    tile_of<BM, BN>(blockIdx.x % tiles, M, N, tm, tn);
    const int t = tm * tilesN + tn, row0 = tm * BM, col0 = tn * BN;
    const int slices = K / SK, per = slices / S, extra = slices % S;
    const int kBegin = s * per + min(s, extra), steps = per + (s < extra);
    half* part = kSplit == kParts ? parts + ((size_t)s * tiles + t) * BM * BN : nullptr;
    unsigned* flag = kSplit == kInPlace ? flags + t : nullptr;
    if (row0 + BM <= M)
        compute_tile<BM, BN, WM, SK, kStages, kSplit, false, Copy, unsigned*, kTogether>(
            shmem, A, B, C, M, N, K, row0, col0, kBegin, steps, part, flag, s, S);
    else
        compute_tile<BM, BN, WM, SK, kStages, kSplit, true, Copy, unsigned*, kTogether>(
            shmem, A, B, C, M, N, K, row0, col0, kBegin, steps, part, flag, s, S);
}

// One block per BM x BN tile (6a).
template <int BM, int BN, int WM, int SK, int kStages>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    run<BM, BN, WM, SK, kStages, kWhole>(shmem, A, B, C, M, N, K, 1, nullptr, nullptr);
}

// S blocks per tile, each one part of K (7a): kParts or kInPlace; kTogether: with kInPlace, 7h's reads of C.
template <int BM, int BN, int WM, int SK, int kStages, int kSplit, bool kTogether = false>
__global__ void gemm_split(const half* A, const half* B, half* C, int M, int N, int K, int S, half* parts,
                           unsigned* flags)
{
    extern __shared__ __align__(128) half shmem[];
    run<BM, BN, WM, SK, kStages, kSplit, PlainCopy, kTogether>(shmem, A, B, C, M, N, K, S, parts, flags);
}

// 7a, parts: one block of 256 threads per tile sums the tile's S parts in the order of K, in fp32 rounded once to
// fp16, and writes C (4a's stores); rows >= M are not read.
template <int BM, int BN>
__global__ void reduce_parts(const half* parts, half* C, int M, int N, int S)
{
    const int tilesN = N / BN, tiles = (M + BM - 1) / BM * tilesN, t = blockIdx.x;
    const int row0 = t / tilesN * BM, col0 = t % tilesN * BN;
    for (int q = threadIdx.x; q < BM * BN / 8; q += 256) {
        const int r = q / (BN / 8), ch = q % (BN / 8);
        if (row0 + r >= M) continue;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int s = 0; s < S; s++) {
            const int4 v = *((const int4*)&parts[((size_t)s * tiles + t) * BM * BN + (size_t)r * BN] + ch);
            half x[8];
            memcpy(x, &v, 16);
#pragma unroll
            for (int e = 0; e < 8; e++) acc[e] += __half2float(x[e]);
        }
        half y[8];
#pragma unroll
        for (int e = 0; e < 8; e++) y[e] = __float2half(acc[e]);
        int4 w;
        memcpy(&w, y, 16);
        step4::st16<true>((int4*)&C[(size_t)(row0 + r) * N + col0] + ch, w);
    }
}

// The plan of each shape: its tile (6a) and how its K is split (7a). Our thresholds, each a mechanical quantity of
// the shape and the card (sms: the SM count):
// (6a) 128 x 128 when 128 x 256 would give fewer than three waves of tiles (3 sms), 128 x 256 otherwise;
// (7a) parts, at M <= 128 (one row of tiles): when the tile covers fewer than 2/3 of the SMs and its work,
//      BM x BN x K multiply-adds, reaches 2^27, K in 3 parts, the 128-row tile then 128 x 256;
//      in place, at M > 128 with 6a's 128 x 128: when 0 < tiles mod sms < 2 sms / 3, K in 2 parts.
// Without 6a, the tile is step 5's 128 x 256; without 7a, no split.
enum Tile { k128x256, k128x128 };
struct Plan { Tile tile; int S; Split split; };
inline Plan choose(int M, int N, int K, int sms, bool cols128, bool splitK)
{
    Plan p = {k128x256, 1, kWhole};
    const int tiles256 = (M + 127) / 128 * (N / 256);
    if (cols128 && tiles256 < 3 * sms) p.tile = k128x128;                       // 6a
    if (!splitK) return p;
    const int bn = p.tile == k128x256 ? 256 : 128;
    const int tiles = (M + 127) / 128 * (N / bn);
    if (M <= 128) {                                                             // 7a, parts
        if (3 * tiles < 2 * sms && (long long)128 * bn * K >= (1LL << 27)) {
            p.S = 3;
            p.split = kParts;
            p.tile = k128x256;
        }
    } else if (p.tile == k128x128) {                                            // 7a, in place
        const int last = tiles % sms;
        if (last != 0 && 3 * last < 2 * sms) {
            p.S = 2;
            p.split = kInPlace;
        }
    }
    return p;
}

#ifdef GEMM_LADDER_EMULATE
// A host emulation adds one and suspends the block there: other blocks may run before its threads go on.
unsigned emu_arrive(unsigned* count);
inline unsigned arrive(unsigned* count) { return emu_arrive(count); }
#else
__device__ __forceinline__ unsigned arrive(unsigned* count) { return atomicAdd(count, 1u); }
#endif


// ---------------------------------------------------------------- 7d: L2 cache hints in the kernels of 6b, 6c and 7b
// Step 7's 7d (step7.cuh), here beside the kernels that carry it: kHint = true in step 7, false in step 6.
// In the kernels where each slice of B is read by one block, or by one group of blocks in phase (6b, 6c, 7b; and 7g's
// in step7.cuh): A, read by every column of tiles, is kept (evict_last); B is evicted first (evict_first). The copy
// into a stage carries the policy (cp.async with an L2 cache hint, the policies made once per tile by createpolicy).
// In every kernel of 6b, 6c and 7b with parts of K (6b's stream, 6c, 7b; and 7e's tail, whose copies take no hint),
// the parts, once added, are discarded from the L2 (discard.global.L2, 128-byte lines): their dirty lines are dropped
// instead of written back to DRAM. kHint = false: step 3's copies, no discard.
#ifdef GEMM_LADDER_EMULATE
void emu_discard(const void* line);   // the emulation fills the line with NaN: its content is undefined after a discard
#endif
struct HintCopy {
    uint64_t first = 0, last = 0;
    __device__ __forceinline__ HintCopy()
    {
#ifndef GEMM_LADDER_EMULATE
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(last));
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(first));
#endif
    }
    __device__ __forceinline__ static void copy(half* smem, const half* gmem, uint64_t policy)
    {
#ifdef GEMM_LADDER_EMULATE
        (void)policy;
        cp_async16(smem, gmem);
#else
        const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
        asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::128B [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem),
                     "l"(policy) : "memory");
#endif
    }
    __device__ __forceinline__ void a(half* smem, const half* gmem) const { copy(smem, gmem, last); }
    __device__ __forceinline__ void b(half* smem, const half* gmem) const { copy(smem, gmem, first); }
};
template <bool kHint> struct CopyOf { using type = PlainCopy; };
template <> struct CopyOf<true> { using type = HintCopy; };

__device__ __forceinline__ void discard_line(const void* line)   // a 128-byte line, 128-byte aligned
{
#ifdef GEMM_LADDER_EMULATE
    emu_discard(line);
#else
    asm volatile("discard.global.L2 [%0], 128;" ::"l"(line) : "memory");
#endif
}

// ---------------------------------------------------------------- 6b: M <= 16, bands of B
// A band of BN columns (128 or 256) and 16 rows (the rows of A >= M copy row M - 1, the rows of C >= M are not
// written), kStages stages of K64: A 16 x 64, B 64 x BN, both swizzled as step 2's stages (chunk c of row r at
// c ^ (r & 7)).
// 4 warps, warp w the band's columns w BN / 4 to (w + 1) BN / 4 - 1, all 16 rows: per K16, one ldmatrix.x4 of A,
// BN / 64 ldmatrix.x4.trans of B, BN / 32 HMMA.16816.
template <int BN, int kStages>
struct BandGeometry {
    static constexpr int kThreads = 128, kWarpCols = BN / 4, kNJ = kWarpCols / 8;   // n8 tiles per warp
    static constexpr int kAStage = 16 * 64, kBStage = 64 * BN, kBChunks = BN / 8;   // halves; chunks per row of B
    static constexpr int kBPer = 64 * kBChunks / kThreads, kBRows = kThreads / kBChunks;
    static constexpr int kOPitch = BN + 8, kOPer = 16 * kBChunks / kThreads;
    static constexpr int kLoopBytes = kStages * (kAStage + kBStage) * 2, kOutBytes = 16 * kOPitch * 2;
    static constexpr int kFlagByte = kLoopBytes > kOutBytes ? kLoopBytes : kOutBytes;   // the last block's flag
    static constexpr int kSmemBytes = kFlagByte + 16;
    static_assert(kSmemBytes <= 163 * 1024, "above the 163 KB of shared memory of a block on sm_80");
    __host__ __device__ static constexpr int a_at(int r, int c) { return r * 64 + 8 * (c ^ (r & 7)); }
    __host__ __device__ static constexpr int b_at(int r, int c) { return r * BN + 8 * (c ^ (r & 7)); }
};

// K steps k0 to k0 + nk - 1 (of 64) of band `band`, into c (zeroed first): step 3's multistage loop.
template <int BN, int kStages, bool kHint>
__device__ __forceinline__ void band_steps(half* shmem, const half* A, const half* B, int M, int N, int K, int band,
                                           int k0, int nk, float (&c)[BandGeometry<BN, kStages>::kNJ][4])
{
    using G = BandGeometry<BN, kStages>;
    half* As = shmem;
    half* Bs = shmem + kStages * G::kAStage;
    const int warpId = threadIdx.x / 32, lane = threadIdx.x % 32, col0 = band * BN;
#pragma unroll
    for (int j = 0; j < G::kNJ; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) c[j][e] = 0.0f;
    // The copy: thread t, chunk t % 8 of row t / 8 of A (clamped to M - 1), chunk t % kBChunks of rows t / kBChunks +
    // kBRows i of B.
    const int ar = threadIdx.x / 8, ac = threadIdx.x % 8;
    const int br = threadIdx.x / G::kBChunks, bc = threadIdx.x % G::kBChunks;
    const half* a0 = A + (size_t)min(ar, M - 1) * K + ac * 8;
    const typename CopyOf<kHint>::type copy;   // 7d in step 7
    auto issue = [&](int k, int st) {   // K step k0 + k into stage st
        const int kk = (k0 + k) * 64;
        copy.a(&As[st * G::kAStage + G::a_at(ar, ac)], a0 + kk);
#pragma unroll
        for (int i = 0; i < G::kBPer; i++) {
            const int r = br + i * G::kBRows;
            copy.b(&Bs[st * G::kBStage + G::b_at(r, bc)], &B[(size_t)(kk + r) * N + col0 + bc * 8]);
        }
    };
    int aOff[4], bOff[G::kNJ / 2];
#pragma unroll
    for (int ks = 0; ks < 4; ks++) aOff[ks] = G::a_at(lane % 16, 2 * ks + lane / 16);
#pragma unroll
    for (int p = 0; p < G::kNJ / 2; p++) bOff[p] = G::b_at(lane % 16, warpId * (G::kWarpCols / 8) + 2 * p + lane / 16);
    auto compute = [&](int st) {
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            uint32_t a[4], b[G::kNJ / 2][4];
            step2::ldmatrix_x4<false>(a, &As[st * G::kAStage + aOff[ks]]);
#pragma unroll
            for (int p = 0; p < G::kNJ / 2; p++)
                step2::ldmatrix_x4<true>(b[p], &Bs[st * G::kBStage + bOff[p] + ks * 16 * BN]);
#pragma unroll
            for (int j = 0; j < G::kNJ; j++)
                step2::mma_m16n8k16(c[j], a, b[j / 2][2 * (j % 2)], b[j / 2][2 * (j % 2) + 1]);
        }
    };
#pragma unroll
    for (int st = 0; st < kStages - 1; st++) {
        if (st < nk) issue(st, st);
        cp_async_commit();
    }
    int read = 0, write = kStages - 1;
    for (int k = 0; k < nk; k++) {
        cp_async_wait<kStages - 2>();
        __syncthreads();
        if (k + kStages - 1 < nk) issue(k + kStages - 1, write);
        cp_async_commit();
        compute(read);
        read = read + 1 == kStages ? 0 : read + 1;
        write = write + 1 == kStages ? 0 : write + 1;
    }
    cp_async_wait<0>();
    __syncthreads();   // every warp is done with the stages before the output reuses them
}

// The 16 x BN accumulators of the block, in fp16, through shared memory: thread t gets chunks t + 128 u (row
// (t + 128 u) / kBChunks, chunk (t + 128 u) % kBChunks of the band).
template <int BN, int kStages>
__device__ __forceinline__ void band_out(half* shmem, const float (&c)[BandGeometry<BN, kStages>::kNJ][4],
                                         int4 (&mine)[BandGeometry<BN, kStages>::kOPer])
{
    using G = BandGeometry<BN, kStages>;
    half* Os = shmem;
    const int warpId = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
#pragma unroll
    for (int j = 0; j < G::kNJ; j++) {
        half* o = &Os[g * G::kOPitch + warpId * G::kWarpCols + 8 * j + 2 * t];
        *(__half2*)o = __floats2half2_rn(c[j][0], c[j][1]);
        *(__half2*)(o + 8 * G::kOPitch) = __floats2half2_rn(c[j][2], c[j][3]);
    }
    __syncthreads();
#pragma unroll
    for (int u = 0; u < G::kOPer; u++) {
        const int ch = threadIdx.x + G::kThreads * u;
        mine[u] = *((const int4*)&Os[ch / G::kBChunks * G::kOPitch] + ch % G::kBChunks);
    }
    __syncthreads();   // the next segment's stages reuse the buffer
}

template <int BN, int kStages>
__device__ __forceinline__ void band_store(half* C, int M, int N, int band,
                                           const int4 (&mine)[BandGeometry<BN, kStages>::kOPer])
{
    using G = BandGeometry<BN, kStages>;
#pragma unroll
    for (int u = 0; u < G::kOPer; u++) {
        const int ch = threadIdx.x + G::kThreads * u, r = ch / G::kBChunks;
        if (r < M) step4::st16<true>((int4*)&C[(size_t)r * N + band * BN] + ch % G::kBChunks, mine[u]);
    }
}

// 6b, bands: one block per band over all of K.
template <int BN, int kStages, bool kHint>
__global__ void gemm_bands(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    using G = BandGeometry<BN, kStages>;
    float c[G::kNJ][4];
    int4 mine[G::kOPer];
    band_steps<BN, kStages, kHint>(shmem, A, B, M, N, K, blockIdx.x, 0, K / 64, c);
    band_out<BN, kStages>(shmem, c, mine);
    band_store<BN, kStages>(C, M, N, blockIdx.x, mine);
}

// 6b, the continuous stream: the units (band, K64 step), band by band, cut into gridDim.x equal ranges (block p:
// units p U / P to (p + 1) U / P - 1, U = bands x K / 64). A block runs its range segment by segment (one band's steps
// each): a segment that covers its band's whole K goes to C; otherwise to its place in the workspace (the band's
// segments in the order of the blocks, that is of K), then the band's counter; the last block to arrive sums the
// band's parts in that order, in fp32 rounded once, writes C and puts the counter back to zero.
template <int BN, int kStages, bool kHint>
__global__ void gemm_stream(const half* A, const half* B, half* C, int M, int N, int K, half* parts, unsigned* counts,
                            int maxSegs)
{
    extern __shared__ __align__(128) half shmem[];
    using G = BandGeometry<BN, kStages>;
    const long steps = K / 64, U = (long)(N / BN) * steps, P = gridDim.x, p = blockIdx.x;
    const long u0 = p * U / P, u1 = (p + 1) * U / P;
    auto first_block = [&](long u) {   // the block whose range holds unit u
        long q = u * P / U;
        while ((q + 1) * U / P <= u) q++;
        while (q * U / P > u) q--;
        return q;
    };
    for (long u = u0; u < u1;) {
        const int band = (int)(u / steps), k0 = (int)(u % steps);
        const long bandEnd = (long)(band + 1) * steps, end = u1 < bandEnd ? u1 : bandEnd;
        const int nk = (int)(end - u);
        float c[G::kNJ][4];
        int4 mine[G::kOPer];
        band_steps<BN, kStages, kHint>(shmem, A, B, M, N, K, band, k0, nk, c);
        band_out<BN, kStages>(shmem, c, mine);
        if (nk == steps) {
            band_store<BN, kStages>(C, M, N, band, mine);
        } else {
            const long pFirst = first_block((long)band * steps), pLast = first_block(bandEnd - 1);
            const int segs = (int)(pLast - pFirst + 1), idx = (int)(p - pFirst);
            half* part = parts + ((size_t)band * maxSegs + idx) * 16 * BN;
#pragma unroll
            for (int q = 0; q < G::kOPer; q++) {
                const int ch = threadIdx.x + G::kThreads * q;
                step4::st16<false>((int4*)&part[ch / G::kBChunks * BN] + ch % G::kBChunks, mine[q]);
            }
            fence();
            __syncthreads();
            // The flag, after the stages and the output: thread 0 writes it again only after the next segment's
            // barriers, which every thread reaches after reading it here.
            unsigned* last = (unsigned*)((char*)shmem + G::kFlagByte);
            if (threadIdx.x == 0) *last = arrive(&counts[band]) == (unsigned)segs - 1;
            __syncthreads();
            if (*last) {
                fence();
#pragma unroll
                for (int q = 0; q < G::kOPer; q++) {
                    const int ch = threadIdx.x + G::kThreads * q, r = ch / G::kBChunks, cc = ch % G::kBChunks;
                    float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (int sg = 0; sg < segs; sg++) {
                        const half* seg = parts + (((size_t)band * maxSegs + sg) * 16 + r) * BN;
                        const int4 x = sg == idx ? mine[q] : ld16_cg((const int4*)seg + cc);
                        half h[8];
                        memcpy(h, &x, 16);
#pragma unroll
                        for (int e = 0; e < 8; e++) v[e] += __half2float(h[e]);
                    }
                    half o[8];
#pragma unroll
                    for (int e = 0; e < 8; e++) o[e] = __float2half(v[e]);
                    int4 w;
                    memcpy(&w, o, 16);
                    if (r < M) step4::st16<true>((int4*)&C[(size_t)r * N + band * BN] + cc, w);
                }
                if (threadIdx.x == 0) counts[band] = 0;   // for the next call
                if constexpr (kHint) {   // 7d: the band's parts, every thread's reads done
                    __syncthreads();
                    const char* first = (const char*)(parts + (size_t)band * maxSegs * 16 * BN);
                    for (int l = threadIdx.x; l < segs * 16 * BN * 2 / 128; l += G::kThreads)
                        discard_line(first + 128 * (size_t)l);
                }
            }
        }
        u = end;
    }
}

// 6b's rule (ours; P the SM count): at M <= 16, bands of 128 columns, one per block over all K,
// when they fit in one wave (N / 128 <= P) and either cover at least 2/3 of the SMs or their work, 16 x 128 x K
// multiply-adds, stays under 2^25; otherwise the continuous stream over P blocks, in bands of 256 columns.
enum M16 { kNoBands, kBands, kStream };
inline M16 m16(int M, int N, int K, int sms)
{
    if (M > 16) return kNoBands;
    const int bands = N / 128;
    if (bands <= sms && (3 * bands >= 2 * sms || 16LL * 128 * K < (1LL << 25))) return kBands;
    return kStream;
}

// ---------------------------------------------------------------- 6c: whole tiles in a continuous stream, in phase
// 16 < M <= 512, where the plan's whole tiles leave the last wave mostly empty. P blocks, one per SM, in groups of MT
// (MT = the rows of tiles, M / BM rounded up; P = sms rounded down to a multiple of MT, and at most MT x U blocks, so
// that every group holds at least one unit and a cut tile's groups follow each other): the MT blocks of a group take
// the same units (column of tiles tn, slice of K) in the same order, block tm of the group the tile of row tm, so that
// the MT blocks read the same slices of B in one order (B can then come from DRAM once, the others from L2).
// Group g holds units g U / G to (g + 1) U / G - 1 (U = N / BN x K / SK, G = P / MT): a range cuts at most two columns
// of tiles, at its ends.
// A whole tile in a range is the plan's (its C written by 4a's stores); a cut tile's segment is a part of K in fp16 to
// the group's slot (two per block: the range's first segment and its last), and the last block of the tile to arrive
// adds the tile's parts in the order of K (the groups' order), in fp32 rounded once, writes C, and puts the tile's
// counter back to zero for the next call.
// kTogether (7h, step 7's): the sum of a cut tile's parts reads each part with every read issued together (below).
template <int BM, int BN, int WM, int SK, int kStages, bool kHint, bool kTogether>
__device__ __forceinline__ void phase_tiles(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                            half* parts, unsigned* counts)
{
    using G = Geometry<BM, BN, WM, SK, kStages>;
    const int MT = (M + BM - 1) / BM, tilesN = N / BN, tm = blockIdx.x % MT, row0 = tm * BM;
    const long steps = K / SK, U = (long)tilesN * steps, P = gridDim.x / MT, g = blockIdx.x / MT;
    const long u0 = g * U / P, u1 = (g + 1) * U / P;
    auto group_of = [&](long u) {   // the group whose range holds unit u
        long q = u * P / U;
        while ((q + 1) * U / P <= u) q++;
        while (q * U / P > u) q--;
        return q;
    };
    using Copy = typename CopyOf<kHint>::type;   // 7d in step 7
    auto slot = [&](long q, int tn) {   // group q's part of column tn: its range's first segment, or its last
        const int which = tn == q * U / P / steps ? 0 : 1;
        return parts + (((size_t)q * 2 + which) * MT + tm) * BM * BN;
    };
    for (long u = u0; u < u1;) {
        const int tn = (int)(u / steps), k0 = (int)(u % steps), col0 = tn * BN;
        const long tileEnd = (long)(tn + 1) * steps, end = u1 < tileEnd ? u1 : tileEnd;
        const int nk = (int)(end - u);
        if (nk == steps) {
            if (row0 + BM <= M)
                compute_tile<BM, BN, WM, SK, kStages, kWhole, false, Copy>(
                    shmem, A, B, C, M, N, K, row0, col0, 0, nk, nullptr, nullptr, 0, 1);
            else
                compute_tile<BM, BN, WM, SK, kStages, kWhole, true, Copy>(
                    shmem, A, B, C, M, N, K, row0, col0, 0, nk, nullptr, nullptr, 0, 1);
        } else {
            half* part = slot(g, tn);
            if (row0 + BM <= M)
                compute_tile<BM, BN, WM, SK, kStages, kParts, false, Copy>(
                    shmem, A, B, C, M, N, K, row0, col0, k0, nk, part, nullptr, 0, 1);
            else
                compute_tile<BM, BN, WM, SK, kStages, kParts, true, Copy>(
                    shmem, A, B, C, M, N, K, row0, col0, k0, nk, part, nullptr, 0, 1);
            const long gFirst = group_of((long)tn * steps), gLast = group_of(tileEnd - 1);
            unsigned* count = counts + tm * tilesN + tn;
            fence();   // this block's part, before its arrival
            __syncthreads();
            // The flag, after the stages and the output (as 6b's)
            unsigned* last = (unsigned*)((char*)shmem + G::kSmemBytes);
            if (threadIdx.x == 0) *last = arrive(count) == (unsigned)(gLast - gFirst);
            __syncthreads();
            if (*last) {
                fence();
                constexpr int kChunks = BM * BN / 8, kBChunks = BN / 8;
                if constexpr (kTogether) {
                    // 7h: the tile's chunks of thread t are t + kThreads u (u < kPer), as below, taken kBatch at a
                    // time; part by part, in the order of K, every read of the batch issued before its sums (one round
                    // trip of the L2 per batch and part, not one per chunk and part), the same sums in the same order.
                    // The reads stay after the arrival and the fence.
                    constexpr int kPer = kChunks / G::kThreads, kBatch = kPer < 2 ? kPer : 2;
                    static_assert(kPer * G::kThreads == kChunks && kPer % kBatch == 0, "the chunks not spread evenly");
#pragma unroll 1
                    for (int u0 = 0; u0 < kPer; u0 += kBatch) {
                        float v[kBatch][8];
#pragma unroll
                        for (int u = 0; u < kBatch; u++)
#pragma unroll
                            for (int e = 0; e < 8; e++) v[u][e] = 0.0f;
                        for (long sg = gFirst; sg <= gLast; sg++) {
                            const half* src = slot(sg, tn);
                            int4 x[kBatch];
#pragma unroll
                            for (int u = 0; u < kBatch; u++) {
                                const int q = threadIdx.x + (u0 + u) * G::kThreads, r = q / kBChunks;
                                if (row0 + r < M) x[u] = ld16_cg((const int4*)&src[(size_t)r * BN] + q % kBChunks);
                            }
#pragma unroll
                            for (int u = 0; u < kBatch; u++) {
                                if (row0 + (threadIdx.x + (u0 + u) * G::kThreads) / kBChunks >= M) continue;
                                half h[8];
                                memcpy(h, &x[u], 16);
#pragma unroll
                                for (int e = 0; e < 8; e++) v[u][e] += __half2float(h[e]);
                            }
                        }
#pragma unroll
                        for (int u = 0; u < kBatch; u++) {
                            const int q = threadIdx.x + (u0 + u) * G::kThreads, r = q / kBChunks, cc = q % kBChunks;
                            if (row0 + r >= M) continue;
                            half o[8];
#pragma unroll
                            for (int e = 0; e < 8; e++) o[e] = __float2half(v[u][e]);
                            int4 w;
                            memcpy(&w, o, 16);
                            step4::st16<true>((int4*)&C[(size_t)(row0 + r) * N + col0] + cc, w);
                        }
                    }
                } else {
                    for (int q = threadIdx.x; q < kChunks; q += G::kThreads) {
                        const int r = q / kBChunks, cc = q % kBChunks;
                        if (row0 + r >= M) continue;
                        float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                        for (long sg = gFirst; sg <= gLast; sg++) {
                            const int4 x = ld16_cg((const int4*)&slot(sg, tn)[(size_t)r * BN] + cc);
                            half h[8];
                            memcpy(h, &x, 16);
#pragma unroll
                            for (int e = 0; e < 8; e++) v[e] += __half2float(h[e]);
                        }
                        half o[8];
#pragma unroll
                        for (int e = 0; e < 8; e++) o[e] = __float2half(v[e]);
                        int4 w;
                        memcpy(&w, o, 16);
                        step4::st16<true>((int4*)&C[(size_t)(row0 + r) * N + col0] + cc, w);
                    }
                }
                if (threadIdx.x == 0) *count = 0;   // for the next call
                if constexpr (kHint) {   // 7d: the tile's parts, every thread's reads done
                    __syncthreads();
                    for (long sg = gFirst; sg <= gLast; sg++)
                        for (int l = threadIdx.x; l < BM * BN * 2 / 128; l += G::kThreads)
                            discard_line((const char*)slot(sg, tn) + 128 * (size_t)l);
                }
            }
        }
        u = end;
    }
}

template <int BM, int BN, int WM, int SK, int kStages, bool kHint>
__global__ void gemm_phase(const half* A, const half* B, half* C, int M, int N, int K, half* parts, unsigned* counts)
{
    extern __shared__ __align__(128) half shmem[];
    phase_tiles<BM, BN, WM, SK, kStages, kHint, false>(shmem, A, B, C, M, N, K, parts, counts);
}

// 7h: 6c's kernel as step 7 runs it, the sums of its cut tiles with their reads together.
template <int BM, int BN, int WM, int SK, int kStages, bool kHint>
__global__ void gemm_phase_sums(const half* A, const half* B, half* C, int M, int N, int K, half* parts,
                                unsigned* counts)
{
    extern __shared__ __align__(128) half shmem[];
    phase_tiles<BM, BN, WM, SK, kStages, kHint, true>(shmem, A, B, C, M, N, K, parts, counts);
}

// 6c's rule (ours): at 16 < M <= 512, where the plan keeps whole tiles over at least two
// waves (tiles >= 2 sms) and its last wave holds fewer than half the SMs; the plan with 7a's split of K (step 7's),
// so that 6c takes the same shapes in steps 6 and 7.
inline bool phase(int M, int N, int K, int sms)
{
    if (M <= 16 || M > 512) return false;
    const Plan p = choose(M, N, K, sms, true, true);
    if (p.split != kWhole) return false;
    const int BN = p.tile == k128x256 ? 256 : 128;
    const long tiles = (long)(M + 127) / 128 * (N / BN), rest = tiles % sms;
    return tiles >= 2L * sms && rest > 0 && 2 * rest < sms;
}

// ---------------------------------------------------------------- 7b: K in parts, reduced by their own blocks
// 7a's parts (compute_tile, kParts: each of S blocks per tile writes its part of K in fp16 to its place in the
// workspace), then, instead of 7a's second kernel: thread 0 arrives on the tile's counter (after a fence) and waits
// until all S parts have arrived (an acquire load); each block then adds the S parts, in the order of K, over 1 / S of
// the tile's 16-byte chunks (block s: chunks s Q / S to (s + 1) Q / S - 1, Q = BM x BN / 8, row-major), in fp32 rounded
// once, and writes C (4a's stores); last, it departs on the tile's second counter, and the last block to depart puts
// both counters back to zero for the next call. The wait needs the S parts of a tile resident together: the rule takes
// 7b only when S x tiles blocks fit on the SMs at once (one block per SM at these tiles), and assumes the GPU then
// runs them together (no cooperative launch).

#ifdef GEMM_LADDER_EMULATE
// A host emulation suspends the block while the count is below `value`, and runs the others.
void emu_wait_count(const unsigned* count, unsigned value);
#endif
__device__ __forceinline__ void wait_count(const unsigned* count, unsigned value)
{
#ifdef GEMM_LADDER_EMULATE
    emu_wait_count(count, value);
#else
    unsigned x;
    do asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(x) : "l"(count) : "memory");
    while (x < value);
#endif
}

template <int BM, int BN, int WM, int SK, int kStages, bool kHint>
__global__ void gemm_parallel(const half* A, const half* B, half* C, int M, int N, int K, int S, half* parts,
                              unsigned* counts)
{
    extern __shared__ __align__(128) half shmem[];
    using Copy = typename CopyOf<kHint>::type;
    run<BM, BN, WM, SK, kStages, kParts, Copy>(shmem, A, B, C, M, N, K, S, parts, nullptr);
    const int tilesN = N / BN, tiles = (M + BM - 1) / BM * tilesN;
    const int s = blockIdx.x / tiles;
    int tm, tn;
    tile_of<BM, BN>(blockIdx.x % tiles, M, N, tm, tn);
    const int t = tm * tilesN + tn, row0 = tm * BM, col0 = tn * BN;
    unsigned* arrived = counts + 2 * t;
    unsigned* departed = arrived + 1;
    fence();   // this block's part, before its arrival
    __syncthreads();
    if (threadIdx.x == 0) {
        arrive(arrived);
        wait_count(arrived, (unsigned)S);
    }
    __syncthreads();
    fence();
    constexpr int kChunks = BM * BN / 8, kBChunks = BN / 8;
    const int q0 = s * kChunks / S, q1 = (s + 1) * kChunks / S;
    for (int q = q0 + threadIdx.x; q < q1; q += blockDim.x) {
        const int r = q / kBChunks, cc = q % kBChunks;
        if (row0 + r >= M) continue;
        float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int ss = 0; ss < S; ss++) {
            const int4 x = ld16_cg((const int4*)&parts[((size_t)ss * tiles + t) * BM * BN + (size_t)r * BN] + cc);
            half h[8];
            memcpy(h, &x, 16);
#pragma unroll
            for (int e = 0; e < 8; e++) v[e] += __half2float(h[e]);
        }
        half o[8];
#pragma unroll
        for (int e = 0; e < 8; e++) o[e] = __float2half(v[e]);
        int4 w;
        memcpy(&w, o, 16);
        step4::st16<true>((int4*)&C[(size_t)(row0 + r) * N + col0] + cc, w);
    }
    __syncthreads();   // every thread's reads of the parts, before the departure (and the discard)
    if constexpr (kHint) {   // 7d: the lines wholly in this block's slice, in each part (the others' slices still
                             // read)
        const int l0 = (q0 + 7) / 8, l1 = q1 / 8;
        for (int ss = 0; ss < S; ss++)
            for (int l = l0 + threadIdx.x; l < l1; l += blockDim.x)
                discard_line((const char*)(parts + ((size_t)ss * tiles + t) * BM * BN) + 128 * (size_t)l);
    }
    if (threadIdx.x == 0 && arrive(departed) == (unsigned)S - 1) {   // the last to depart: both counters back to zero
        *arrived = 0;
        *departed = 0;
    }
}


// 7b's condition: S x tiles blocks resident together, one block per SM at the parts' tiles (147 456 bytes of shared
// memory each).
inline bool parallel_fits(int tiles, int S, int sms) { return (long)S * tiles <= sms; }

// ---------------------------------------------------------------- 7e: the tail of the waves, cut in K
// Step 7's 7e (step7.cuh), here beside the plan's kernels; step 6 does not run it. Whole tiles at M > 512 (128 x 256,
// the plan's): T tiles in waves of sms, the last wave R = T mod sms tiles. The first T - R tiles run whole (one block
// each, the plan's order); the last R are cut into S parts of K, R S blocks that the GPU places as the SMs of the last
// whole wave free up. Each part to its place in the workspace in fp16; the last part of a tile to arrive adds the S
// parts in the order of K, in fp32 rounded once, writes C (4a's stores), and puts the tile's counter back to zero for
// the next call; with 7d, the parts are then discarded from the L2 (the copies take no hint: at these shapes each slice
// of B is read by every row of tiles).
template <int BM, int BN, int WM, int SK, int kStages, bool kHint>
__global__ void gemm_tail(const half* A, const half* B, half* C, int M, int N, int K, int whole, int S, half* parts,
                          unsigned* counts)
{
    extern __shared__ __align__(128) half shmem[];
    using G = Geometry<BM, BN, WM, SK, kStages>;
    using Copy = PlainCopy;   // B is read again by every row of tiles here: none of 7d's hints on the copies
    const int tilesN = N / BN, tiles = (M + BM - 1) / BM * tilesN, R = tiles - whole;
    const int b = blockIdx.x;
    const int j = b < whole ? 0 : (b - whole) % R, s = b < whole ? 0 : (b - whole) / R;   // tail tile j, part s
    int tm, tn;
    tile_of<BM, BN>(b < whole ? b : whole + j, M, N, tm, tn);
    const int row0 = tm * BM, col0 = tn * BN, slices = K / SK;
    if (b < whole) {
        if (row0 + BM <= M)
            compute_tile<BM, BN, WM, SK, kStages, kWhole, false, Copy>(
                shmem, A, B, C, M, N, K, row0, col0, 0, slices, nullptr, nullptr, 0, 1);
        else
            compute_tile<BM, BN, WM, SK, kStages, kWhole, true, Copy>(
                shmem, A, B, C, M, N, K, row0, col0, 0, slices, nullptr, nullptr, 0, 1);
        return;
    }
    const int per = slices / S, extra = slices % S, kBegin = s * per + min(s, extra), steps = per + (s < extra);
    half* part = parts + ((size_t)s * R + j) * BM * BN;
    if (row0 + BM <= M)
        compute_tile<BM, BN, WM, SK, kStages, kParts, false, Copy>(
            shmem, A, B, C, M, N, K, row0, col0, kBegin, steps, part, nullptr, s, S);
    else
        compute_tile<BM, BN, WM, SK, kStages, kParts, true, Copy>(
            shmem, A, B, C, M, N, K, row0, col0, kBegin, steps, part, nullptr, s, S);
    fence();   // this block's part, before its arrival
    __syncthreads();
    // The flag, after the stages and the output (as 6b's)
    unsigned* last = (unsigned*)((char*)shmem + G::kSmemBytes);
    if (threadIdx.x == 0) *last = arrive(counts + j) == (unsigned)S - 1;
    __syncthreads();
    if (!*last) return;
    fence();
    constexpr int kChunks = BM * BN / 8, kBChunks = BN / 8;
    for (int q = threadIdx.x; q < kChunks; q += G::kThreads) {
        const int r = q / kBChunks, cc = q % kBChunks;
        if (row0 + r >= M) continue;
        float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int ss = 0; ss < S; ss++) {
            const int4 x = ld16_cg((const int4*)&parts[((size_t)ss * R + j) * BM * BN + (size_t)r * BN] + cc);
            half h[8];
            memcpy(h, &x, 16);
#pragma unroll
            for (int e = 0; e < 8; e++) v[e] += __half2float(h[e]);
        }
        half o[8];
#pragma unroll
        for (int e = 0; e < 8; e++) o[e] = __float2half(v[e]);
        int4 w;
        memcpy(&w, o, 16);
        step4::st16<true>((int4*)&C[(size_t)(row0 + r) * N + col0] + cc, w);
    }
    if (threadIdx.x == 0) counts[j] = 0;   // for the next call
    if constexpr (kHint) {   // 7d: the tile's parts, every thread's reads done
        __syncthreads();
        for (int ss = 0; ss < S; ss++)
            for (int l = threadIdx.x; l < BM * BN * 2 / 128; l += G::kThreads)
                discard_line((const char*)(parts + ((size_t)ss * R + j) * BM * BN) + 128 * (size_t)l);
    }
}

// 7e's rule (ours): at M > 512, where the plan keeps whole tiles of 128 x 256 over at
// least one full wave and the last wave is partial (R = T mod sms > 0): S from 2 to 8 parts with at least 64 slices of
// K64 each (a part's fill of two slices is then 1/32 of it at most), the S that minimizes ceil(R S / sms) / S,
// the tail's time in tiles (the smallest S at a tie), if that time is below one tile. Returns S, or 0: no 7e.
inline int tail_parts(int M, int N, int K, int sms)
{
    if (M <= 512) return 0;
    const Plan p = choose(M, N, K, sms, true, true);
    if (p.split != kWhole || p.tile != k128x256) return 0;
    const long tiles = (long)(M + 127) / 128 * (N / 256), R = tiles % sms;
    if (tiles < sms || R == 0) return 0;
    int best = 0;
    long bestNum = 1, bestDen = 1;   // the best time, a fraction of a tile: ceil(R S / sms) / S < 1
    for (int S = 2; S <= 8 && K / 64 / S >= 64; S++) {
        const long num = (R * S + sms - 1) / sms;
        if (num * bestDen < bestNum * S) {
            best = S;
            bestNum = num;
            bestDen = S;
        }
    }
    return best;
}

}  // namespace step6
