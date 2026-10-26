/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through steps 0 and 1 (nvidia_sample.cuh, step1.cuh), whose notice follows.
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
#ifdef GEMM_LADDER_EMULATE
#include <cstring>
#endif

namespace step2 {

// ---------------------------------------------------------------- step 2: mma.sync and ldmatrix, a swizzle
// Two mechanisms, nested on step 1 (step1.cuh, v0), whose rest is kept: one block per 128 x 256 tile, 8 warps of
// 64 x 64 (2 along M x 4 along N) computed in step 1's order, one shared stage of K64 filled through registers between
// two barriers, every thread copying its share of A and B (4 + 8 chunks of 16 bytes per K64, the 12 loads issued before
// the stores), ordinary global loads, the edge in M (a tile cut by M clamps the row of A each thread loads to M - 1),
// the fp16 output through shared memory half a tile at a time at a pitch of 264 halves, then 16-byte stores.
// Shapes: M a multiple of 16, N of 256, K of 64.
// (2a) The tensor cores without wmma: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 (one HMMA.16816.F32) and
//      ldmatrix.sync.aligned.m8n8.x4 (one LDSM.16.M88.4: four 8 x 8 matrices, one register of each per lane), with
//      .trans for B (LDSM.16.MT88.4), which is row-major (K x N) where mma takes it by columns. On sm_80 wmma already
//      compiles to these (step 1: 128 HMMA.16816 and 32 LDSM.x4 per K64 and per warp), so 2a keeps those counts; it
//      changes the address arithmetic: the fragment addresses become loop-invariant registers plus immediates (8 per
//      lane in the source: one per K16 of A, one per 16 columns of B), where step 1 recomputes them at each K64.
//      The epilogue is rewritten, there being no wmma fragment to store: each lane converts its pairs of accumulators
//      to fp16 and stores them where the mma layout puts them (STS.32), then the same 16-byte copy out.
// (2b) A swizzle instead of the padding: chunk c (16 bytes) of row r is stored at chunk c ^ (r & 7), in A's rows of 128
//      bytes (K64, 8 chunks) and in B's rows of 512 bytes (256 columns, 32 chunks: the swizzle moves a chunk within
//      its 128-byte line); no padding, 48 KB for the stage instead of 54 KB. wmma cannot read it (load_matrix_sync
//      takes one fixed pitch), so 2b only exists on top of 2a.
// Bank conflicts (32 banks of 4 bytes; a 16-byte access by 8 lanes, one 8 x 8 matrix of an LDSM or a quarter warp's
// STS.128, takes one wavefront when its 8 chunks fall in the 8 groups of 4 banks of a 128-byte line, one more per
// extra chunk in the busiest group):
//   padding (step 1, v1): A's rows of 160 bytes put the 8 rows of a matrix in groups c, c + 2, c + 4, c + 6, c, ...
//   (mod 8), B's rows of 544 bytes the same: two-way on every LDSM, 8 wavefronts per LDSM.x4 instead of 4; the copy's
//   stores, 8 chunks of one row per quarter warp, are conflict-free;
//   swizzle (2b): the 8 rows of a matrix (r0 to r0 + 7, r0 a multiple of 8, one chunk c) fall in groups c ^ 0 to
//   c ^ 7 of their lines; a quarter warp stores 8 chunks of one row, permuted within their line. No conflict in the
//   loop. The epilogue is conflict-free in all variants (pitch 132 words, 4 mod 32: an STS.32 hits 32 banks; a quarter
//   warp's LDS.128 reads 128 contiguous bytes). The host emulator (emu/) counts the wavefronts of v0's and v1's shared
//   accesses: 4.00 per LDSM.x4 with the swizzle, 8.00 with the padding, 4.00 per STS.128 and LDS.128, 1.00 per STS.32.
// The counts below are read with nvcc 13.2 (a local compilation); the measured binaries are built by nvcc 13.1.1 (the
// Modal images).
// Compiled (nvcc 13.2, sm_80): v0 at 224 registers, v1 at 228, step 1 at 244, no spill; one block per SM. Main loop,
// per K64 and per warp (static counts; no branch in the loops but their own, so each instruction runs once per K64;
// the integer instructions left in v0 and v1 compute the 64-bit global addresses of the slice's 12 loads and run the
// loop; the swizzle adds one, its lane offsets being computed before the loop):
//                         HMMA   LDSM.x4   LDG.128   STS.128   BAR   integer   total
//   step 1 (v2)           128    32        12        12        2     83        272
//   2a (v1)               128    32        12        12        2     41        228
//   2a + 2b (v0)          128    32        12        12        2     42        229
// The total also counts the instructions that no column holds.
// Variants (g_var in step2.cu, for the interleaved probe; nested, wmma cannot read a swizzle):
//   v0 = step 2 (2a + 2b; 49 152 bytes of dynamic shared memory);
//   v1 = without 2b: 2a over step 1's padded stage (pitches of 80 and 272 halves; 55 296 bytes);
//   v2 = step 1 itself (step1::gemm<true, false, 256>; 55 296 bytes), so that the anchor runs in the same probe: v1
//        against v2 measures 2a, v0 against v1 measures 2b.
constexpr int kSkew = 16;   // step 1's padding, in halves (v1)

#ifdef GEMM_LADDER_EMULATE
// A host emulation runs each thread as a coroutine; this is the warp's meeting point: once all 32 lanes have called it,
// each receives in all[l] the `bytes` that lane l gave at `mine`.
void emu_warp_exchange(const void* mine, void* all, int bytes);
#endif

// One ldmatrix.x4 (one LDSM.16.M88.4; kTrans: .trans, LDSM.16.MT88.4): lane l gives p, the address of row l % 8 of
// 8 x 8 matrix q = l / 8 (8 halves, 16-byte aligned, in shared memory); with g = l / 4 and t = l % 4, register q of
// lane l receives {Mq[g][2t], Mq[g][2t + 1]}, with .trans {Mq[2t][g], Mq[2t + 1][g]} (low half first). The C++
// branch states these semantics and runs in a host emulation of the kernel.
template <bool kTrans>
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const half* p)
{
#ifdef GEMM_LADDER_EMULATE
    const half* rows[32];
    emu_warp_exchange(&p, rows, sizeof p);
    const int g = threadIdx.x % 32 / 4, t = threadIdx.x % 4;
    for (int q = 0; q < 4; q++) {
        const half e[2] = {kTrans ? rows[8 * q + 2 * t][g] : rows[8 * q + g][2 * t],
                           kTrans ? rows[8 * q + 2 * t + 1][g] : rows[8 * q + g][2 * t + 1]};
        memcpy(&r[q], e, 4);
    }
#else
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    if constexpr (kTrans)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#endif
}

// One mma.sync m16n8k16 (one HMMA.16816.F32), D = A B + D, lane l (g, t as above): A 16 x 16 fp16 in a[0] = {A[g][2t],
// A[g][2t + 1]}, a[1] = {A[g + 8][2t], ...}, a[2] = {A[g][2t + 8], ...}, a[3] = {A[g + 8][2t + 8], ...}; B 16 x 8 fp16
// in b0 = {B[2t][g], B[2t + 1][g]} and b1 = {B[2t + 8][g], B[2t + 9][g]}; D 16 x 8 fp32 in d = D[g][2t], D[g][2t + 1],
// D[g + 8][2t], D[g + 8][2t + 1].
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
#ifdef GEMM_LADDER_EMULATE
    struct Lane { uint32_t a[4], b[2]; } mine = {{a[0], a[1], a[2], a[3]}, {b0, b1}}, all[32];
    emu_warp_exchange(&mine, all, sizeof mine);
    auto el = [](uint32_t x, int i) { half e[2]; memcpy(e, &x, 4); return (float)e[i]; };
    const int g = threadIdx.x % 32 / 4, t = threadIdx.x % 4;
    for (int i = 0; i < 4; i++) {   // A[m][k]: lane 4 (m % 8) + (k % 8) / 2, register m / 8 + 2 (k / 8); B[k][n]: lane
        const int m = g + 8 * (i / 2), n = 2 * t + i % 2;   // 4 n + (k % 8) / 2, register k / 8; half k % 2 in both
        for (int k = 0; k < 16; k++)
            d[i] += el(all[4 * (m % 8) + (k % 8) / 2].a[m / 8 + 2 * (k / 8)], k % 2) *
                    el(all[4 * n + (k % 8) / 2].b[k / 8], k % 2);
    }
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// The geometry of a variant (kSwizzle: 2b). Sizes in halves unless said otherwise. A's stage: 128 rows of K64 (8 chunks
// of 16 bytes); B's stage: 64 rows of K, 256 columns (32 chunks). a_at and b_at give the place of chunk c of row r.
template <bool kSwizzle>
struct Geometry {
    static constexpr int kAStride = kSwizzle ? 64 : 64 + kSkew;     // 64 or 80
    static constexpr int kBStride = kSwizzle ? 256 : 256 + kSkew;   // 256 or 272
    static constexpr int kOStride = 256 + 8;                        // 264, as step 1
    static constexpr int kAStage = 128 * kAStride, kBStage = 64 * kBStride;
    static constexpr int kLoopBytes = (kAStage + kBStage) * 2, kOutBytes = 64 * kOStride * 2;
    static constexpr int kSmemBytes = kLoopBytes > kOutBytes ? kLoopBytes : kOutBytes;
    __host__ __device__ static constexpr int a_at(int r, int c)
    {
        return r * kAStride + 8 * (kSwizzle ? c ^ (r & 7) : c);
    }
    __host__ __device__ static constexpr int b_at(int r, int c)
    {
        return r * kBStride + 8 * (kSwizzle ? c ^ (r & 7) : c);
    }
};

// One 128 x 256 tile; kEdge: the tile is cut by M (the row of A each thread loads is clamped to M - 1).
template <bool kSwizzle, bool kEdge>
__device__ __forceinline__ void compute_tile(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                             int row0, int col0)
{
    using G = Geometry<kSwizzle>;
    half* As = shmem;                // 128 rows of the K64 slice of A
    half* Bs = shmem + G::kAStage;   // 64 rows of the K64 slice of B
    const int warpId = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warpId / 4, wn = warpId % 4;   // the warp's rows wm * 64, columns wn * 64
    // c[i][j][n]: the 16 x 8 accumulator of rows wm * 64 + 16 i, columns wn * 64 + 16 j + 8 n (step 1's 16 x 16
    // fragment c[i][j] is c[i][j][0] and c[i][j][1]).
    float c[4][4][2][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) c[i][j][n][e] = 0.0f;
    // The warp's ldmatrix rows: lane l points to row l % 16, chunk l / 16 of a 16 x 16 block (A: rows of M, chunks of
    // K; B: rows of K, chunks of N), so that the four matrices of an .x4 are the block's quarters, rows 0-7 and 8-15 of
    // chunk 0, then of chunk 1. Moving a block by 16 rows moves it by a constant (the swizzle of a row depends on its
    // bits 0-2), so a lane needs one offset per K16 of A (aOff) and one per 16 columns of B (bOff).
    int aOff[4], bOff[4];
#pragma unroll
    for (int ks = 0; ks < 4; ks++) aOff[ks] = G::a_at(wm * 64 + lane % 16, 2 * ks + lane / 16);
#pragma unroll
    for (int j = 0; j < 4; j++) bOff[j] = G::b_at(lane % 16, wn * 8 + 2 * j + lane / 16);

    // A slice in registers, as step 1: thread t holds rows t / 8 + 32 i of the slice of A at chunk t % 8, and rows
    // t / 32 + 8 i of the slice of B at chunk t % 32.
    int4 ra[4], rb[8];
    auto load = [&](int k0) {   // global memory to registers, the slice that starts at k0
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int r = threadIdx.x / 8 + i * 32;
            const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;
            ra[i] = *((const int4*)&A[(size_t)gr * K + k0] + threadIdx.x % 8);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int r = threadIdx.x / 32 + i * 8;
            rb[i] = *((const int4*)&B[(size_t)(k0 + r) * N + col0] + threadIdx.x % 32);
        }
    };
    auto store = [&]() {   // registers to the stage
#pragma unroll
        for (int i = 0; i < 4; i++)
            *(int4*)&As[G::a_at(threadIdx.x / 8 + i * 32, threadIdx.x % 8)] = ra[i];
#pragma unroll
        for (int i = 0; i < 8; i++)
            *(int4*)&Bs[G::b_at(threadIdx.x / 32 + i * 8, threadIdx.x % 32)] = rb[i];
    };
    auto compute = [&]() {   // the warp's 64 x 64 over the stage, K16 at a time, in step 1's order
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            uint32_t a[4][4], b[4][4];   // a[i]: rows 16 i, K16 ks; b[j]: K16 ks, columns 16 j
#pragma unroll
            for (int i = 0; i < 4; i++) {
                ldmatrix_x4<false>(a[i], &As[aOff[ks] + i * 16 * G::kAStride]);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (i == 0) ldmatrix_x4<true>(b[j], &Bs[bOff[j] + ks * 16 * G::kBStride]);
                    // step 1's mma_sync(c[i][j], a[i], b[j]): two n8 halves; b[j][2 n], b[j][2 n + 1] = K rows 0-7
                    // and 8-15 of the n8 half n
#pragma unroll
                    for (int n = 0; n < 2; n++) mma_m16n8k16(c[i][j][n], a[i], b[j][2 * n], b[j][2 * n + 1]);
                }
            }
        }
    };
    // Step 1's loop: copy a K64 slice, barrier, compute, barrier.
    for (int k0 = 0; k0 < K; k0 += 64) {
        load(k0);
        store();
        __syncthreads();
        compute();
        __syncthreads();
    }
    // The output, half a tile at a time, as step 1: the 4 warps of rows [64 h, 64 h + 64) store their accumulators in
    // fp16, then all 256 threads copy the 64 x 256 halves out in 16-byte stores. Each lane converts its pairs D[g][2t],
    // D[g][2t + 1] and D[g + 8][2t], D[g + 8][2t + 1] of each 16 x 8 accumulator and stores them as two 32-bit words.
    half* Os = shmem;   // 64 x kOStride halves, inside the loop's buffers
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        if (wm == h) {   // warps 4h to 4h + 3 hold rows [64 h, 64 h + 64)
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
                *((int4*)&C[(size_t)gr * N + col0] + threadIdx.x % 32) =
                    *((const int4*)&Os[r * G::kOStride] + threadIdx.x % 32);
        }
        __syncthreads();
    }
}

// One block per 128 x 256 tile (step 1's 1a), the tiles in step 0's order (N fastest).
template <bool kSwizzle>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    const int tilesN = N / 256;
    const int row0 = (blockIdx.x / tilesN) * 128, col0 = (blockIdx.x % tilesN) * 256;
    if (row0 + 128 <= M) compute_tile<kSwizzle, false>(shmem, A, B, C, M, N, K, row0, col0);
    else compute_tile<kSwizzle, true>(shmem, A, B, C, M, N, K, row0, col0);
}

}  // namespace step2
