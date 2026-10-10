/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through steps 0 to 2 (nvidia_sample.cuh, step1.cuh, step2.cuh), whose notice follows.
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
#include "step2.cuh"

namespace step3 {

// ---------------------------------------------------------------- step 3: the memory hierarchy
// Three mechanisms of cuBLAS's T4 kernel on the way between DRAM, L2 and the SMs, nested on step 2 (step2.cuh, v0),
// whose kernel is kept as it is: one block per 128 x 256 tile, 8 warps of 64 x 64, two shared stages of K32 filled
// through registers with one barrier per K32, mma.sync m16n8k8 and ldmatrix.x4, the swizzle, the edge in M, the fp16
// output through shared memory half a tile at a time. Shapes: M a multiple of 16, N of 256, K of 64.
// Step 2 reads A and B with plain 16-byte loads (LDG.E.128.SYS) and writes C with plain stores (STG.E.128.SYS), its
// tiles in step 0's order, N fastest. At ladder (2048 x 2560 x 2048: 16 x 10 tiles, a wave of 40 blocks on the T4's 40
// SMs), a wave covers 4 rows of tiles across all 10 columns: it reads 2 MiB of A and all 10 MiB of B, which the 4 MiB
// of L2 cannot keep for the next rows. The DRAM bytes and L2 sector hits that follow are in the README (ncu at ladder,
// step 3).
// (3a) Read-only loads that ask L2 for a whole line, as cuBLAS's kernel (ncu: read-only loads with the L2's
//      128-byte prefetch): ld.global.nc.L2::128B, A and B being read-only during the kernel.
// (3b) Stores of C marked evict-first, as cuBLAS's kernel (ncu: evict-first stores): st.global.cs; C is written
//      once and not read again, so its lines leave L2 before those of A and B.
// (3c) The tiles in groups of 8 rows of tiles (cuBLAS's swizzle flag; the size 8 is ours, from our earlier T4
//      kernels): group g's blocks cover rows of tiles 8 g to 8 g + 7, column by column. At ladder a wave covers 8 rows
//      of tiles and 5 columns: 4 MiB of A and 5 MiB of B, where step 0's order reads 2 MiB of A and 10 MiB of B.
// None of the three changes the arithmetic: the five variants give the same bits.
// The counts below are read with nvcc 13.2 (a local compilation); the measured binaries are built by nvcc 13.1.1 (the
// Modal images).
// Compiled (nvcc 13.2): all five 254 registers, no spill; 49 152 bytes of dynamic shared memory; one block per SM.
// 3a turns the loop's 12 LDG.E.128.SYS per K64 into LDG.E.LTC128B.128.CONSTANT.SYS and adds 5 integer instructions per
// K64 and per warp (370 instructions per K64, 369 at the edge tiles of v0 and v3, against step 2's 365); 3b turns the
// epilogue's STG.E.128.SYS (16 per thread and tile) into STG.E.EF.128.SYS; 3c only changes how a block finds its tile,
// before the loop (46 instructions per tile). ptxas's register allocation also moves the epilogue by up to 60
// instructions between variants, against about 11 800 per warp and tile at ladder.
// Variants (g_var in step3.cu, for the interleaved probe; each removal keeps the other two):
//   v0 = step 3 (3a + 3b + 3c); v1 = without 3a; v2 = without 3b; v3 = without 3c;
//   v4 = step 2 itself (step2::gemm<true>), so that the anchor runs in the same probe.

using G = step2::Geometry<true>;   // step 2's stages (2b's swizzle): 49 152 bytes

// 3a: a 16-byte load of A or B; with kNc, read-only, the L2 asked for the whole 128-byte line.
template <bool kNc>
__device__ __forceinline__ int4 ld16(const int4* p)
{
#ifdef GEMM_LADDER_EMULATE
    return *p;
#else
    if constexpr (kNc) {
        int4 v;
        asm("ld.global.nc.L2::128B.v4.s32 {%0, %1, %2, %3}, [%4];"
            : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
        return v;
    } else {
        return *p;
    }
#endif
}

// 3b: a 16-byte store of C; with kEf, marked evict-first.
template <bool kEf>
__device__ __forceinline__ void st16(int4* p, int4 v)
{
#ifdef GEMM_LADDER_EMULATE
    *p = v;
#else
    if constexpr (kEf)
        asm volatile("st.global.cs.v4.s32 [%0], {%1, %2, %3, %4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                     : "memory");
    else
        *p = v;
#endif
}

// 3c: the tile (row of tiles tm, column of tiles tn) of block b; without kGroup, step 0's order (N fastest).
template <bool kGroup>
__device__ __forceinline__ void tile_of(int b, int M, int N, int& tm, int& tn)
{
    const int tilesN = N / 256;
    if constexpr (kGroup) {
        const int tilesM = (M + 127) / 128, per = 8 * tilesN, first = b / per * 8;
        const int rows = min(tilesM - first, 8), r = b % per;   // the last group may have fewer rows
        tm = first + r % rows;
        tn = r / rows;
    } else {
        tm = b / tilesN;
        tn = b % tilesN;
    }
}

// One 128 x 256 tile: step 2's compute_tile (step2.cuh, v0), with 3a's loads and 3b's stores.
template <bool kNc, bool kEf, bool kEdge>
__device__ __forceinline__ void compute_tile(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                             int row0, int col0)
{
    half* As = shmem;                    // 2 stages x 128 rows of the K32 slice of A
    half* Bs = shmem + 2 * G::kAStage;   // 2 stages x 32 rows of the K32 slice of B
    const int warpId = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warpId / 4, wn = warpId % 4;   // the warp's rows wm * 64, columns wn * 64
    float c[4][4][2][4];   // c[i][j][n]: the 16 x 8 accumulator of rows wm * 64 + 16 i, columns wn * 64 + 16 j + 8 n
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) c[i][j][n][e] = 0.0f;
    int aOff[2], bOff[4];   // the warp's ldmatrix rows, as step 2
#pragma unroll
    for (int ks = 0; ks < 2; ks++) aOff[ks] = G::a_at(wm * 64 + lane % 16, 2 * ks + lane / 16);
#pragma unroll
    for (int j = 0; j < 4; j++) bOff[j] = G::b_at(lane % 16, wn * 8 + 2 * j + lane / 16);

    // A slice in registers, as step 2: thread t holds rows t / 4 + 64 i of the slice of A at chunk t % 4, and rows
    // t / 32 + 8 i of the slice of B at chunk t % 32.
    int4 ra[2], rb[4];
    auto load = [&](int k0) {   // global memory to registers, the slice that starts at k0
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int r = threadIdx.x / 4 + i * 64;
            const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;
            ra[i] = ld16<kNc>((const int4*)&A[(size_t)gr * K + k0] + threadIdx.x % 4);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int r = threadIdx.x / 32 + i * 8;
            rb[i] = ld16<kNc>((const int4*)&B[(size_t)(k0 + r) * N + col0] + threadIdx.x % 32);
        }
    };
    auto store = [&](int s) {   // registers to stage s
#pragma unroll
        for (int i = 0; i < 2; i++)
            *(int4*)&As[s * G::kAStage + G::a_at(threadIdx.x / 4 + i * 64, threadIdx.x % 4)] = ra[i];
#pragma unroll
        for (int i = 0; i < 4; i++)
            *(int4*)&Bs[s * G::kBStage + G::b_at(threadIdx.x / 32 + i * 8, threadIdx.x % 32)] = rb[i];
    };
    auto compute = [&](int s) {   // the warp's 64 x 64 over stage s, K16 at a time, as step 2
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            uint32_t a[4][4], b[4][4];   // a[i]: rows 16 i, K16 ks; b[j]: K16 ks, columns 16 j
#pragma unroll
            for (int i = 0; i < 4; i++) {
                step2::ldmatrix_x4<false>(a[i], &As[s * G::kAStage + aOff[ks] + i * 16 * G::kAStride]);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (i == 0)
                        step2::ldmatrix_x4<true>(b[j], &Bs[s * G::kBStage + bOff[j] + ks * 16 * G::kBStride]);
#pragma unroll
                    for (int kk = 0; kk < 2; kk++)
#pragma unroll
                        for (int n = 0; n < 2; n++)
                            step2::mma_m16n8k8(c[i][j][n], a[i][2 * kk], a[i][2 * kk + 1], b[j][2 * n + kk]);
                }
            }
        }
    };
    // Step 2's loop: the first K32 slice to stage 0; then per K32, the next slice's loads, the compute of the current
    // stage, the stores to the other stage, one barrier.
    load(0);
    store(0);
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += 64) {
        load(k0 + 32);
        compute(0);
        store(1);
        __syncthreads();
        const bool next = k0 + 64 < K;
        if (next) load(k0 + 64);
        compute(1);
        if (next) store(0);
        __syncthreads();
    }
    // The output, as step 2: half a tile at a time, the accumulators in fp16 to shared memory, then 16-byte stores.
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
                st16<kEf>((int4*)&C[(size_t)gr * N + col0] + threadIdx.x % 32,
                          *((const int4*)&Os[r * G::kOStride] + threadIdx.x % 32));
        }
        __syncthreads();
    }
}

// One block per 128 x 256 tile, the tiles in 3c's order or step 0's.
template <bool kNc, bool kEf, bool kGroup>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    int tm, tn;
    tile_of<kGroup>(blockIdx.x, M, N, tm, tn);
    const int row0 = tm * 128, col0 = tn * 256;
    if (row0 + 128 <= M) compute_tile<kNc, kEf, false>(shmem, A, B, C, M, N, K, row0, col0);
    else compute_tile<kNc, kEf, true>(shmem, A, B, C, M, N, K, row0, col0);
}

}  // namespace step3
