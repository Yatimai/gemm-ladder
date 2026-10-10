/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through step 0 (nvidia_sample.cuh), whose notice follows.
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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace step1 {
using namespace nvcuda;

// ---------------------------------------------------------------- step 1: the granularity of cuBLAS
// Four mechanisms of cuBLAS's T4 kernel (named turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn under ncu),
// applied to step 0 (compute_gemm, nvidia_sample.cuh): a 128 x 256 tile in the judge's row-major view, two shared
// stages of K32 filled through registers, one block per tile, every thread copying its share of both operands. Shapes:
// M a multiple of 16, N of 256 (of 128 for v3 to v5), K of 64. Kept from step 0: wmma 16 x 16 x 16 fragments (b[j] read
// once per K16 and reused by every row of fragments), rows padded by 16 halves, 16-byte global loads and shared stores,
// ordinary global loads, the order of the tiles (N fastest), the edge in M (a tile cut by M clamps the row of A each
// thread loads to M - 1, so only edge tiles pay for the clamp), the fp16 output through shared memory half a tile (64
// rows) at a time, at a pitch of the tile's width + 8 halves, then 16-byte stores.
// (1a) One block per tile, as cuBLAS: the grid has as many blocks as tiles, and when a block ends the hardware
//      starts the next one on its SM, so an SM that finishes first takes the next tile; step 0 runs one block per SM
//      over a list of tiles fixed at launch (block_pos += gridDim.x).
// (1b) Double buffering: two shared stages of K32 instead of one K64 slice. Step 0 issues a slice's loads and waits
//      for them between two barriers, with no HMMA to issue meanwhile. Here each thread's loads of the next K32 slice
//      go to registers before the compute of the current stage and are in flight during its HMMA (128 per warp at the
//      128 x 256 tile); they are stored to the other stage after the compute, and one barrier per K32 both publishes
//      them and frees the stage just read. Step 0 also has one barrier per K32 (two per K64): only their place changes.
// (1c) A 128 x 256 tile, 8 warps of 64 x 64 (2 along M x 4 along N, 4 x 4 fragments: 128 HMMA per K32 and per
//      warp, as cuBLAS's kernel) instead of 128 x 128 with warps of 32 x 64. Per K32, a block reads 24 KB of A and B
//      for 2^20 multiply-adds instead of 16 KB for 2^19 (a quarter fewer bytes read from global memory per
//      multiply-add), and a warp reads its fragments in 32 LDSM.x2 for 128 HMMA instead of 24 for 64 (a third fewer per
//      HMMA). The price: 128 accumulator registers per thread instead of 64; step 0 already runs one block per SM (186
//      registers).
// (1d) The spread of the copy, cuBLAS's: each of the 256 threads copies its share of the slice of A and of the
//      slice of B (per K32, 2 + 4 chunks of 16 bytes at the 128 x 256 tile, as cuBLAS's kernel; 2 + 2 at 128 x 128),
//      where step 0 gives A to warps 0-3 and B to warps 4-7 (at the 128 x 256 tile, B's slice is twice A's). The
//      template has only this copy, so 1d is measured added to step 0 (v4 against v5), not removed from step 1. Its
//      cost, seen in the SASS: the fragment addresses are recomputed, about 95 integer instructions per K64 and per
//      warp in v4 against 23 to 27 in step 0 (full tiles).
// Padding kept, one side effect: a K32 row of A is 96 bytes (64 + 32 of padding), so the 8 threads of a quarter warp
// store into two rows whose banks overlap: two-way conflicts on A's stores, which step 0's 160-byte rows do not have.
// B's stores and every LDSM keep step 0's bank pattern (two-way on the LDSM, from the padding).
// Variants (g_var in step1.cu, for the interleaved probe: the step, and each removal = the step without one mechanism):
//   v0 = step 1 (1a + 1b + 1c + 1d; 254 registers, nvcc 13.2);
//   v1 = without 1a: step 0's persistent grid (one block per SM, block_pos += gridDim.x); at 255 registers with the
//        tile loop, nvcc also recomputes addresses, about 30 more integer instructions per K64 and per warp than v0,
//        a cost its measurement includes;
//   v2 = without 1b: step 0's loop (one K64 slice, copied between two barriers) at the 128 x 256 tile;
//   v3 = without 1c: step 0's 128 x 128 tile (8 warps of 32 x 64), with 1a and 1b;
//   v4 = 1d alone: step 0 except for the spread of the copy;
//   v5 = step 0 itself (nvidia_sample::compute_gemm), so that v4 and v5 run in the same probe.
// Shared memory (dynamic; above 48 KB the launch needs cudaFuncSetAttribute): v0 and v1 59 392 bytes, v2 55 296,
// v3 43 008, v4 and v5 38 912; the output's half tile (33 792 or 17 408 bytes) reuses the loop's buffers.
constexpr int kSkew = 16;   // step 0's padding (the sample's SKEW_HALF), in halves

// The geometry of a variant: kBN, the tile's width (256 with 1c, 128 without); kDouble, two K32 stages (1b) or one K64
// slice. Sizes in halves unless said otherwise.
template <int kBN, bool kDouble>
struct Geometry {
    static constexpr int kSliceK = kDouble ? 32 : 64;    // K per stage
    static constexpr int kStages = kDouble ? 2 : 1;
    static constexpr int kAStride = kSliceK + kSkew;     // 48 or 80
    static constexpr int kBStride = kBN + kSkew;         // 272 or 144
    static constexpr int kOStride = kBN + 8;             // 264 or 136 (no bank conflict at either)
    static constexpr int kAStage = 128 * kAStride, kBStage = kSliceK * kBStride;
    static constexpr int kALanes = kSliceK / 8, kBLanes = kBN / 8;   // 16-byte chunks per row of a slice
    static constexpr int kACopies = 128 * kALanes / 256, kBCopies = kSliceK * kBLanes / 256;   // per thread and slice
    static constexpr int kWarpsN = kBN / 64;             // warps along N (4 or 2), each 64 columns wide
    static constexpr int kWarpRows = 128 * kWarpsN / 8;  // rows per warp (64 or 32)
    static constexpr int kFragsM = kWarpRows / 16;       // rows of fragments per warp (4 or 2)
    static constexpr int kLoopBytes = kStages * (kAStage + kBStage) * 2, kOutBytes = 64 * kOStride * 2;
    static constexpr int kSmemBytes = kLoopBytes > kOutBytes ? kLoopBytes : kOutBytes;
    static_assert(kACopies * 256 == 128 * kALanes && kBCopies * 256 == kSliceK * kBLanes, "copies not spread evenly");
    static_assert(kSmemBytes <= 64 * 1024, "above the 64 KB of shared memory of a block on sm_75");
};

// One 128 x kBN tile; kEdge: the tile is cut by M (the row of A each thread loads is clamped to M - 1).
template <int kBN, bool kDouble, bool kEdge>
__device__ __forceinline__ void compute_tile(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                             int row0, int col0)
{
    using G = Geometry<kBN, kDouble>;
    half* As = shmem;                              // kStages x 128 rows of the slice of A
    half* Bs = shmem + G::kStages * G::kAStage;    // kStages x kSliceK rows of the slice of B
    const int warpId = threadIdx.x / 32;
    const int wm = warpId / G::kWarpsN, wn = warpId % G::kWarpsN;   // the warp's rows wm * kWarpRows, columns wn * 64
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[G::kFragsM][4];
#pragma unroll
    for (int i = 0; i < G::kFragsM; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) wmma::fill_fragment(c[i][j], 0.0f);

    // A slice in registers: thread t holds rows t / kALanes + i * (256 / kALanes) of the slice of A at 16-byte lane
    // t % kALanes, and rows t / kBLanes + i * (256 / kBLanes) of the slice of B at lane t % kBLanes.
    int4 ra[G::kACopies], rb[G::kBCopies];
    auto load = [&](int k0) {   // global memory to registers, the slice that starts at k0
#pragma unroll
        for (int i = 0; i < G::kACopies; i++) {
            const int r = threadIdx.x / G::kALanes + i * (256 / G::kALanes);
            const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;
            ra[i] = *((const int4*)&A[(size_t)gr * K + k0] + threadIdx.x % G::kALanes);
        }
#pragma unroll
        for (int i = 0; i < G::kBCopies; i++) {
            const int r = threadIdx.x / G::kBLanes + i * (256 / G::kBLanes);
            rb[i] = *((const int4*)&B[(size_t)(k0 + r) * N + col0] + threadIdx.x % G::kBLanes);
        }
    };
    auto store = [&](int s) {   // registers to stage s
#pragma unroll
        for (int i = 0; i < G::kACopies; i++) {
            const int r = threadIdx.x / G::kALanes + i * (256 / G::kALanes);
            *((int4*)&As[s * G::kAStage + r * G::kAStride] + threadIdx.x % G::kALanes) = ra[i];
        }
#pragma unroll
        for (int i = 0; i < G::kBCopies; i++) {
            const int r = threadIdx.x / G::kBLanes + i * (256 / G::kBLanes);
            *((int4*)&Bs[s * G::kBStage + r * G::kBStride] + threadIdx.x % G::kBLanes) = rb[i];
        }
    };
    auto compute = [&](int s) {   // the warp's fragments over stage s, K16 at a time, as step 0
#pragma unroll
        for (int ks = 0; ks < G::kSliceK / 16; ks++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[G::kFragsM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
#pragma unroll
            for (int i = 0; i < G::kFragsM; i++) {
                wmma::load_matrix_sync(a[i], &As[s * G::kAStage + (wm * G::kWarpRows + i * 16) * G::kAStride + ks * 16],
                                       G::kAStride);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (i == 0)
                        wmma::load_matrix_sync(b[j], &Bs[s * G::kBStage + (ks * 16) * G::kBStride + wn * 64 + j * 16],
                                               G::kBStride);
                    wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                }
            }
        }
    };
    if constexpr (kDouble) {
        // (1b), the double buffer: the first K32 slice to stage 0; then per K32, the next slice's loads, the compute of
        // the current stage, the stores to the other stage, one barrier. An iteration covers step 0's K64, stage 0 then
        // stage 1, so that each stage keeps a fixed place (a stage index known at run time would add address arithmetic
        // that v2 does not pay).
        load(0);
        store(0);
        __syncthreads();
        for (int k0 = 0; k0 < K; k0 += 64) {
            load(k0 + 32);   // K is a multiple of 64: the second K32 of this K64 always exists
            compute(0);
            store(1);
            __syncthreads();
            const bool next = k0 + 64 < K;
            if (next) load(k0 + 64);
            compute(1);
            if (next) store(0);
            __syncthreads();
        }
    } else {
        // v2 and v4: step 0's loop, copy a K64 slice, barrier, compute, barrier.
        for (int k0 = 0; k0 < K; k0 += 64) {
            load(k0);
            store(0);
            __syncthreads();
            compute(0);
            __syncthreads();
        }
    }
    // The output, half a tile at a time: the 4 warps of rows [64 h, 64 h + 64) store their fragments, then all 256
    // threads copy the 64 x kBN halves out in 16-byte stores. An fp32 and an fp16 accumulator fragment hold their
    // elements in the same order on sm_75 (checked in SASS, see simple_wmma_gemm in nvidia_sample.cuh), so the copy
    // converts in place.
    half* Os = shmem;   // 64 x kOStride halves, inside the loop's buffers
#pragma unroll
    for (int h = 0; h < 2; h++) {
        if (warpId / 4 == h) {   // in both layouts, warps 4h to 4h + 3 hold rows [64 h, 64 h + 64)
#pragma unroll
            for (int i = 0; i < G::kFragsM; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    wmma::fragment<wmma::accumulator, 16, 16, 16, half> o;
#pragma unroll
                    for (int t = 0; t < o.num_elements; t++) o.x[t] = __float2half(c[i][j].x[t]);
                    wmma::store_matrix_sync(&Os[((wm * G::kWarpRows) % 64 + i * 16) * G::kOStride + wn * 64 + j * 16],
                                            o, G::kOStride, wmma::mem_row_major);
                }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 64 * G::kBLanes / 256; i++) {
            const int r = threadIdx.x / G::kBLanes + i * (256 / G::kBLanes);   // row in the half tile
            const int gr = row0 + h * 64 + r;
            if (gr < M)
                *((int4*)&C[(size_t)gr * N + col0] + threadIdx.x % G::kBLanes) =
                    *((const int4*)&Os[r * G::kOStride] + threadIdx.x % G::kBLanes);
        }
        __syncthreads();
    }
}

// kOneTile: (1a), one block per tile (grid = number of tiles); otherwise step 0's persistent grid (one block per SM).
template <bool kOneTile, bool kDouble, int kBN>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    const int tilesN = N / kBN;
    auto tile = [&](int block_pos) {
        const int row0 = (block_pos / tilesN) * 128, col0 = (block_pos % tilesN) * kBN;
        if (row0 + 128 <= M) compute_tile<kBN, kDouble, false>(shmem, A, B, C, M, N, K, row0, col0);
        else compute_tile<kBN, kDouble, true>(shmem, A, B, C, M, N, K, row0, col0);
    };
    if constexpr (kOneTile) {
        tile(blockIdx.x);
    } else {
        const int tiles = (M + 127) / 128 * tilesN;
        for (int block_pos = blockIdx.x; block_pos < tiles; block_pos += gridDim.x) tile(block_pos);
    }
}

}  // namespace step1
