/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through step 0 (nvidia_sample.cuh) and steps 1 to 3: the output, whose notice follows.
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

// ---------------------------------------------------------------- step 4: clusters of two
// Clusters of two blocks on step 3 (step3.cuh, v0: 3a alone, step 2's kernel at 4 stages), whose rest is kept: the
// 128 x 256 tile, 4 stages, the TMA boxes, two warpgroups that both compute, thread 0 issuing each slice's loads once
// its stage is released by both, the output through shared memory. Shapes: M a multiple of 16, N of 256, K of 64.
// (4a) Clusters of two blocks: with two rows of tiles or more, the two blocks of a cluster take two tiles one above the
//      other; with one row of tiles (M = 16 and 128), two tiles side by side. Each block loads its own A and B and
//      releases its stages to itself; the cluster is synchronized in one piece once the barriers are initialized
//      (barrier.cluster.arrive.relaxed, then barrier.cluster.wait). A block after the last column of tiles still takes
//      part (its loads read zeros outside the matrices) and writes nothing.
// (4b) The cluster is set at the launch (cudaLaunchKernelEx with a cluster dimension), never by __cluster_dims__ on
//      the kernel.
// The roles (3b) and the descriptors in the uniform datapath (3c), the two pieces of CUTLASS's cooperative
// warp-specialized kernel for sm_90 that left step 3 at the test global, left steps 4 to 6 too, and come back at step
// 7. The shared operand of 4a (the multicast TMA) and 4d left step 4 at the test global, then steps 5 and 6, and come
// back at step 7 (7h, 7i), with their tools there (mapa, the arrivals on another block, the multicast load:
// step7.cuh); 4c left the code. This file keeps the cluster's barrier below for steps 5 to 7.
// Compiled (nvcc 13.2, sm_90a, without -lineinfo, as the judge): 154 registers (as step 3), no spill. Main loop, per
// K64 and per warp (static counts): 150 instructions (ptxas unrolls the loop by 2; 31 R2UR: the descriptors rebuilt
// from the stage's address at each slice; thread 0's 5 UTMALDG inside), 149 in step 3. The cluster barrier: one
// UCGABAR_ARV (the relaxed arrival) and one UCGABAR_WAIT right after it; no MEMBAR.ALL.GPU.
// Variants (g_var in step4.cu, for the interleaved probe):
//   v0 = step 4 (4a, 4b);
//   v1 = step 3 itself (step2::gemm<4>), the anchor in the same probe.

#ifdef GEMM_LADDER_EMULATE
// A host emulation runs the blocks of a cluster together, each with its own shared memory; the cluster's barrier
// suspends a thread until every thread of the cluster has arrived.
uint8_t* emu_shared_base();
uint32_t emu_cluster_rank();
void emu_cluster_arrive();
void emu_cluster_wait();
#endif

// The block's dynamic shared memory.
__device__ __forceinline__ uint8_t* shared_base()
{
#ifdef GEMM_LADDER_EMULATE
    return emu_shared_base();
#else
    extern __shared__ __align__(1024) uint8_t shmem[];
    return shmem;
#endif
}

// %cluster_ctarank: the block's rank in its cluster.
__device__ __forceinline__ uint32_t cluster_rank()
{
#ifdef GEMM_LADDER_EMULATE
    return emu_cluster_rank();
#else
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
#endif
}

// The cluster barrier, split: arrive (relaxed: no ordering of its own; the barriers' initialization is published by
// fence.mbarrier_init.release.cluster before it) and wait (acquire).
__device__ __forceinline__ void cluster_arrive_relaxed()
{
#ifdef GEMM_LADDER_EMULATE
    emu_cluster_arrive();
#else
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cluster_wait()
{
#ifdef GEMM_LADDER_EMULATE
    emu_cluster_wait();
#else
    asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
#endif
}

using G = step3::Geometry<4>;   // step 3's four stages of 48 KB, the output after them, the barriers

// The grid: two blocks per cluster, the clusters along M (two rows of tiles or more) or along N.
inline int blocks(int M, int N, bool alongM)
{
    const int tilesM = (M + 127) / 128, tilesN = N / 256;
    return 2 * (alongM ? (tilesM + 1) / 2 * tilesN : tilesM * ((tilesN + 1) / 2));
}

// The block's tile in its cluster's pair: along M, two tiles one above the other (two rows of tiles or more); along N,
// side by side (one row of tiles).
template <bool kAlongM>
__device__ __forceinline__ void pair_tile(int N, int& row0, int& col0)
{
    const int rank = (int)cluster_rank(), tilesN = N / 256, pair = blockIdx.x / 2;
    int tm, tn;
    if constexpr (kAlongM) {
        tm = (pair / tilesN) * 2 + rank;
        tn = pair % tilesN;
    } else {
        const int pairsN = (tilesN + 1) / 2;
        tm = pair / pairsN;
        tn = (pair % pairsN) * 2 + rank;
    }
    row0 = tm * 128;
    col0 = tn * 256;
}

// Step 4: step 2's kernel at 4 stages (step 3's), two warpgroups that both compute, thread 0 issuing slice p + 4's
// loads once the stage of slice p is released by both, in clusters of two (4a, 4b). kAlongM: the cluster's two tiles
// one above the other, else side by side.
template <bool kAlongM>
__global__ void __launch_bounds__(256, 1) gemm(const __grid_constant__ CUtensorMap tmA,
                                                       const __grid_constant__ CUtensorMap tmB, half* C, int M, int N,
                                                       int K)
{
    constexpr int kStages = 4;
    uint8_t* shmem = shared_base();
    uint8_t* sm = shmem + ((1024 - (step1::smem_addr(shmem) & 1023)) & 1023);
    uint64_t* full = (uint64_t*)(sm + G::kBars);
    uint64_t* empty = full + kStages;
    const int tid = threadIdx.x, wg = tid / 128, nt = K / 64;
    int row0, col0;
    pair_tile<kAlongM>(N, row0, col0);
    if (tid == 0) {
        for (int s = 0; s < kStages; s++) {
            step2::mbar_init(&full[s], 1);
            step2::mbar_init(&empty[s], 2);
        }
        step2::mbar_fence_init();
    }
    __syncthreads();
    cluster_arrive_relaxed();   // the cluster synchronized in one piece, the barriers initialized
    cluster_wait();
    auto load = [&](int t) {   // slice t into its stage (thread 0)
        const int s = t % kStages;
        uint8_t* st = sm + s * G::kStage;
        step2::mbar_arrive_expect_tx(&full[s], G::kStage);
        step2::tma_load_2d(st, &tmA, t * 64, row0, &full[s]);
#pragma unroll
        for (int j = 0; j < 4; j++) step2::tma_load_2d(st + G::kABytes + j * 8192, &tmB, col0 + 64 * j, t * 64, &full[s]);
    };
    if (tid == 0)
        for (int t = 0; t < kStages && t < nt; t++) load(t);
    float acc[128];   // not zeroed: the first wgmma of the tile overwrites it (scaleD 0)
    const uint32_t s0 = step1::smem_addr(sm);
    for (int t = 0; t < nt; t++) {
        const int s = t % kStages;
        step2::mbar_wait(&full[s], (t / kStages) & 1);
        const uint32_t st = s0 + s * G::kStage;
        const uint64_t da = step1::desc_a(st + wg * 64 * 128), db = step1::desc_b(st + G::kABytes);
        step1::wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < 4; kk++) step1::wgmma<256>(acc, da + 2 * kk, db + 128 * kk, t > 0 || kk > 0);
        step1::wgmma_commit();
        step1::wgmma_wait<1>();
        if (t >= 1) {   // the group of slice p = t - 1 is done: its stage is released, then refilled with slice p + 4
            const int p = t - 1, ps = p % kStages;
            if (tid % 128 == 0) step2::mbar_arrive(&empty[ps]);
            if (tid == 0 && p + kStages < nt) {
                step2::mbar_wait(&empty[ps], (p / kStages) & 1);
                load(p + kStages);
            }
        }
    }
    step1::wgmma_wait<0>();
    __syncthreads();   // both warpgroups are done with the stages: the output tile overwrites them
    half* o = (half*)sm;
    const int w = (tid / 32) % 4, l = tid % 32, g = l / 4, q = l % 4, r0 = wg * 64 + w * 16 + g;
#pragma unroll
    for (int j = 0; j < 32; j++) {
        *(__half2*)&o[r0 * G::kOPitch + 8 * j + 2 * q] = __floats2half2_rn(acc[4 * j], acc[4 * j + 1]);
        *(__half2*)&o[(r0 + 8) * G::kOPitch + 8 * j + 2 * q] = __floats2half2_rn(acc[4 * j + 2], acc[4 * j + 3]);
    }
    __syncthreads();
    if (col0 >= N) return;   // a block after the last column of tiles writes nothing
#pragma unroll
    for (int i = 0; i < 16; i++) {   // 16-byte stores, rows >= M left out
        const int x = i * 256 + tid, r = x / 32, gr = row0 + r;
        if (gr < M) *((int4*)&C[(size_t)gr * N + col0] + x % 32) = *((const int4*)&o[r * G::kOPitch] + x % 32);
    }
}

}  // namespace step4
