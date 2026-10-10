/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm),
 * whose notice follows. The two kernels below keep the sample's structure and change only what the judge's format
 * requires; every change is listed in the comment above each kernel.
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
// The sample computes D = alpha A B + beta C with B column-major and C, D fp32, at fixed sizes.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace nvidia_sample {
using namespace nvcuda;

// ---------------------------------------------------------------- the reference point: simple_wmma_gemm
// One warp per 16 x 16 tile of C; A and B fragments loaded straight from global memory; no shared memory.
// Changes: B row-major (matrix_b fragment row_major, leading dimension N); the fp32 accumulator converted to an fp16
// fragment and stored to C (no alpha, beta or C read). The sample's two bound tests are kept as they are (an early exit
// instead lets nvcc batch the loads of 4 K steps: about 64 registers against 36).
// Launch as in the sample: block (128, 4) = 16 warps, 4 along M and 4 along N.
__global__ void simple_wmma_gemm(const half* A, const half* B, half* C, int M, int N, int K)
{
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int warpN = blockIdx.y * blockDim.y + threadIdx.y;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int k = 0; k < K; k += 16) {
        const int aRow = warpM * 16, aCol = k, bRow = k, bCol = warpN * 16;
        if (aRow < M && aCol < K && bRow < K && bCol < N) {   // the sample's bound test, at every K step
            wmma::load_matrix_sync(a, A + (size_t)aRow * K + aCol, K);
            wmma::load_matrix_sync(b, B + (size_t)bRow * N + bCol, N);
            wmma::mma_sync(acc, a, b, acc);
        }
    }
    const int cRow = warpM * 16, cCol = warpN * 16;
    if (cRow < M && cCol < N) {   // the sample's bound test on the store
        // An fp32 and an fp16 accumulator fragment hold their elements in the same order on sm_75 (checked in SASS),
        // so the element-wise copy converts in place.
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out;
#pragma unroll
        for (int t = 0; t < out.num_elements; ++t) out.x[t] = __float2half(acc.x[t]);
        wmma::store_matrix_sync(C + (size_t)cRow * N + cCol, out, N, wmma::mem_row_major);
    }
}

// ---------------------------------------------------------------- step 0: compute_gemm
// Kept from the sample: a 128 x 128 tile per block, 8 warps of 32 x 64 (2 x 4 wmma fragments of 16 x 16); a K64 slice
// of A and B per iteration in shared memory (CHUNK_K = 4, the sample's value for 64 KB of shared memory), copied in
// 16-byte loads (warps 0-3 copy A, warps 4-7 copy B), rows padded by SKEW_HALF = 16 halves; one barrier after the copy
// and one after the compute; a persistent grid (one block per SM, tile = block_pos, block_pos += gridDim.x); ordinary
// global loads (no __restrict__, so no read-only path).
// Changes: B row-major, so the B slice is 64 rows of K x 128 columns at a pitch of 128 + 16 halves (16 lanes per row);
// C = A B, so no C tile is read and scaled by beta and no alpha (the sample's 64 KB of shared memory held that fp32 C
// tile; 38 912 bytes here); the output in fp16, through shared memory half a tile (64 rows) at a time at a pitch of 128
// + 8 halves (no bank conflict), then 16-byte stores; M, N, K at run time (N a multiple of 128, K of 64, M of 16);
// edges in M: a tile that holds 128 rows runs the sample's copy as is, a tile cut by M clamps the row of A each thread
// loads to M - 1 (rows past M are computed and never written), so only edge tiles pay for the clamp.
// Not a change of ours but worth knowing: the sample is compiled at fixed sizes, so nvcc unrolls its whole K loop
// (8 192 HMMA in one straight block, all of its integer instructions before the second K64); with K known at run time,
// the loop here stays a loop.
constexpr int kChunkK = 4;
constexpr int kSkew = 16;
constexpr int kAStride = kChunkK * 16 + kSkew;   // 80 halves
constexpr int kBStride = 128 + kSkew;            // 144 halves
constexpr int kOStride = 128 + 8;                // 136 halves
constexpr int kSmemBytes = (128 * kAStride + 64 * kBStride) * 2;   // 20 480 + 18 432 = 38 912 bytes

// One 128 x 128 tile; kEdge: the tile is cut by M (the row of A each thread loads is clamped to M - 1).
template <bool kEdge>
__device__ __forceinline__ void compute_tile(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                             int row0, int col0)
{
    half* As = shmem;                      // 128 rows of the K64 slice of A
    half* Bs = shmem + 128 * kAStride;     // 64 rows of K of the 128-column slice of B
    const int warpId = threadIdx.x / 32, laneId = threadIdx.x % 32;
    {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[2][4];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) wmma::fill_fragment(c[i][j], 0.0f);
        for (int k0 = 0; k0 < K; k0 += kChunkK * 16) {
            if (warpId < 4) {   // A: 128 rows of 128 bytes; 32 rows per warp, 4 rows per instruction (8 lanes per row)
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int r = warpId * 32 + i * 4 + laneId / 8;
                    const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;
                    *((int4*)&As[r * kAStride] + laneId % 8) = *((const int4*)&A[(size_t)gr * K + k0] + laneId % 8);
                }
            } else {            // B: 64 rows of 256 bytes; 16 rows per warp, 2 rows per instruction (16 lanes per row)
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int r = (warpId - 4) * 16 + i * 2 + laneId / 16;
                    *((int4*)&Bs[r * kBStride] + laneId % 16) = *((const int4*)&B[(size_t)(k0 + r) * N + col0] + laneId % 16);
                }
            }
            __syncthreads();
#pragma unroll
            for (int ks = 0; ks < kChunkK; ks++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    wmma::load_matrix_sync(a[i], &As[((warpId / 2) * 32 + i * 16) * kAStride + ks * 16], kAStride);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        if (i == 0) wmma::load_matrix_sync(b[j], &Bs[(ks * 16) * kBStride + (warpId % 2) * 64 + j * 16], kBStride);
                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }
            __syncthreads();
        }
        // The output, half a tile at a time: the 4 warps of rows [64 h, 64 h + 64) store their fragments, then all
        // 256 threads copy the 64 x 128 halves out in 16-byte stores (64 rows x 16 lanes = 4 stores per thread).
        half* Os = shmem;   // 64 x 136 halves = 17 408 bytes, inside the loop's buffers
#pragma unroll
        for (int h = 0; h < 2; h++) {
            if (warpId / 4 == h) {
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        wmma::fragment<wmma::accumulator, 16, 16, 16, half> o;
#pragma unroll
                        for (int t = 0; t < o.num_elements; t++) o.x[t] = __float2half(c[i][j].x[t]);
                        wmma::store_matrix_sync(&Os[(((warpId / 2) % 2) * 32 + i * 16) * kOStride + (warpId % 2) * 64 + j * 16],
                                                o, kOStride, wmma::mem_row_major);
                    }
            }
            __syncthreads();
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int r = (threadIdx.x / 16) + i * 16;   // row in the half tile
                const int gr = row0 + h * 64 + r;
                if (gr < M) *((int4*)&C[(size_t)gr * N + col0] + threadIdx.x % 16) = *((const int4*)&Os[r * kOStride] + threadIdx.x % 16);
            }
            __syncthreads();
        }
    }
}

__global__ void compute_gemm(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    const int tilesN = N / 128, tilesM = (M + 127) / 128;
    for (int block_pos = blockIdx.x; block_pos < tilesM * tilesN; block_pos += gridDim.x) {
        const int row0 = (block_pos / tilesN) * 128, col0 = (block_pos % tilesN) * 128;
        if (row0 + 128 <= M) compute_tile<false>(shmem, A, B, C, M, N, K, row0, col0);
        else compute_tile<true>(shmem, A, B, C, M, N, K, row0, col0);
    }
}

}  // namespace nvidia_sample
