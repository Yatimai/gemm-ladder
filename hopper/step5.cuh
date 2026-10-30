/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through step 0 (nvidia_sample.cuh) and steps 1 to 4: the output, whose notice follows.
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
#include "step4.cuh"

namespace step5 {

// ---------------------------------------------------------------- step 5: a persistent grid, a tile swizzle
// A persistent grid and an order of the tiles, on step 4 (step4.cuh, v0), whose rest is kept: the 128 x 256 tiles in
// pairs, one pair per cluster of two blocks at a time, each block loading its own A and B and releasing its stages to
// itself, 4 stages, the cluster synchronized in one piece, 256 threads: two warpgroups that both compute, rows 0 to 63
// and 64 to 127 of each tile, warp 0 issuing the loads; the output through shared memory then 16-byte stores. Here
// each computing warpgroup stores its 64 rows into its own zone after the stages, in two passes of 128 columns, each
// lane's pairs of halves at their place in boxes of 64 x 64 at the 128-byte swizzle (on a warp's 8 rows its 32 stores
// of 4 bytes go to 32 different banks), and its 128 threads copy them out: the zones, the passes and that copy live
// from 5a (the stages of the next pair are filled meanwhile). Each warpgroup releases a stage once wait_group 1 has
// retired the group that read it; warp 0 then waits for that stage released by both and one of its lanes
// (elect.sync) issues the slice 4 further on; the whole warp takes the branch (its index broadcast by a shuffle); the
// descriptors are rebuilt from the stage's address at each slice (A's in vector registers: the R2UR below). Shapes: M a
// multiple of 16, N of 256, K of 64.
// (5a) A persistent grid: as many clusters as the GPU holds at once (cudaOccupancyMaxActiveClusters), each going
//      through the pairs of tiles u, u + (number of clusters), ...; the ring of stages and its phases run on from one
//      pair to the next: warp 0 issues the next pair's first slices during the current pair's last ones, and they land
//      while the output is written. Both blocks of a cluster go through the same pairs, and so the same slices.
// (5b) The order of the pairs: in groups of 16 rows of tiles, column after column within a group, the odd groups
//      from the last column back to the first (a serpentine: the columns of B read at the end of a group are read
//      again at the start of the next). Why 16: the 132 tiles that run at once read, from DRAM, 2 MB of A per row of
//      tiles (128 rows of K = 8192 halves) and 4 MB of B per column (256 columns); r rows by 132 / r columns read
//      2 r + 4 * 132 / r MB, least at r = 16 (65 MB per wave, against 100 MB for step 4's 6 rows by 22 columns at the
//      ladder).
// 3b and 3c (the roles: a loading warpgroup and two computing ones; the descriptors in the uniform datapath: the
// computing warpgroup's index and the stage's offset broadcast by a shuffle, then advanced by additions), two pieces of
// CUTLASS's cooperative warp-specialized kernel for sm_90 (public source), left step 3 at the test global, then steps 4
// to 6, and come back at step 7 (7f, 7g). The shared operand of 4a and 4d (the end of the pipeline for it), which left
// steps 4 to 6, come back at step 7 (7h, 7i), and 5c too (7j); 4c, 5d and 5e left the code.
// Compiled (nvcc 13.2, sm_90a, without -lineinfo, as the judge): 167 registers in v0 to v2 (166 in v2 along M); no
// spill, no local memory. One loop, not unrolled, warp 0's loads in branches inside it: on the computing path 60
// instructions per K64 (8 R2UR: the descriptors rebuilt at each slice), 7 fewer at a tile's first slice, which
// releases no stage; no loop around an elect.sync. The output, per pass of 128 columns and per warp: 32 STS.32, then 8
// LDS.128 and 8 STG.E.128. One UCGABAR_ARV and one UCGABAR_WAIT, as in step 4.
// Variants (g_var in step5.cu, for the interleaved probe):
//   v0 = step 5 (5a, 5b: step5::gemm);
//   v1 = without 5a: one pair of tiles per cluster (step 4's grid), the rest of step 5 kept;
//   v2 = without 5b: the pairs row after row, as step 4 takes them;
//   v3 = step 4 itself (step4::gemm), the anchor in the same probe.

#ifdef GEMM_LADDER_EMULATE
// A host emulation checks a TMA store when issued (its source made visible to the async proxy); the store reads its
// source then or as late as the thread's next wait_group.read.
void emu_tma_store_2d(const CUtensorMap* map, int c0, int c1, uint32_t src);
void emu_bulk_commit();
void emu_bulk_wait(int pending, bool read);
#endif

// One TMA store of a 2D box (cp.async.bulk.tensor.2d, global from shared): coordinates in elements, the fast one
// first; elements outside the matrix are not written. It joins the thread's open bulk group.
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map, int c0, int c1, uint32_t src)
{
#ifdef GEMM_LADDER_EMULATE
    emu_tma_store_2d(map, c0, c1, src);
#else
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                 ::"l"(map), "r"(c0), "r"(c1), "r"(src) : "memory");
#endif
}

// cp.async.bulk.commit_group: the thread's bulk operations issued since its last commit form a group.
__device__ __forceinline__ void bulk_commit()
{
#ifdef GEMM_LADDER_EMULATE
    emu_bulk_commit();
#else
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
#endif
}

// cp.async.bulk.wait_group.read N: waits until at most N of the thread's groups may still read their source (shared
// memory read by the others may be written again); without .read, until their writes are done.
template <int N, bool kRead>
__device__ __forceinline__ void bulk_wait()
{
#ifdef GEMM_LADDER_EMULATE
    emu_bulk_wait(N, kRead);
#else
    if constexpr (kRead) asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory");
    else asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}

// Two fp32 values as the halves of one 32-bit register (the first in the low half).
__device__ __forceinline__ uint32_t pack_half2(float a, float b)
{
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

struct Geometry {
    static constexpr int kStages = 4, kABytes = step4::G::kABytes, kStage = step4::G::kStage;   // 16 + 32 KB
    static constexpr int kZone = 2 * 8192;   // a computing warpgroup's output: two boxes of 64 rows x 64 columns
    static constexpr int kZones = kStages * kStage;   // the zones after the stages
    static constexpr int kBars = kZones + 2 * kZone;  // the barriers after the zones: full[kStages], empty[kStages]
    static constexpr int kSmemBytes = kBars + 16 * kStages + 1024;   // + the alignment of the stages to 1024 bytes
    static_assert(kSmemBytes <= 227 * 1024, "227 KB of shared memory at most");
};
using G = Geometry;

constexpr int kGroupRows = 16;   // 5b: rows of tiles per group

// The units of work: pairs of tiles, one per cluster at a time. Along M (two rows of tiles or more), two tiles one
// above the other: UM = half the rows of tiles (rounded up) by UN = the columns. Along N, side by side.
__host__ __device__ inline void units(int M, int N, bool alongM, int& UM, int& UN)
{
    const int tilesM = (M + 127) / 128, tilesN = N / 256;
    UM = alongM ? (tilesM + 1) / 2 : tilesM;
    UN = alongM ? tilesN : (tilesN + 1) / 2;
}

// Unit u's place (row um, column un) among UM x UN: 5b's order (groups of `rows` rows of units, column after column
// within a group, the odd groups from the last column back), or row after row.
__host__ __device__ inline void place(int u, int UM, int UN, int rows, bool grouped, int& um, int& un)
{
    if (!grouped) {
        um = u / UN;
        un = u % UN;
        return;
    }
    const int perGroup = rows * UN, g = u / perGroup, r = u - g * perGroup;
    const int inGroup = UM - g * rows < rows ? UM - g * rows : rows;   // the last group may have fewer rows
    um = g * rows + r % inGroup;
    un = r / inGroup;
    if (g & 1) un = UN - 1 - un;
}

// The grid: two blocks per cluster, one cluster per unit at most, maxClusters at most (5a; one per unit without it).
inline int blocks(int M, int N, bool alongM, int maxClusters)
{
    int UM, UN;
    units(M, N, alongM, UM, UN);
    return 2 * (UM * UN < maxClusters ? UM * UN : maxClusters);
}

// The output of a computing warpgroup, pass q of its two passes of 128 columns: each lane's pairs of halves at their
// place in the warpgroup's zone zp, two boxes of 64 x 64 at the 128-byte swizzle, rows g and g + 8 of warp w's 16
// (g = l / 4), columns 8 c + 2 (l % 4) and the next (rg + 8 has the same row % 8: the same swizzle).
__device__ __forceinline__ void zone_store(uint8_t* zp, const float* acc, int q, int w, int l)
{
    const int rg = 16 * w + l / 4;
#pragma unroll
    for (int c = 0; c < 16; c++) {
        uint8_t* d = zp + (c / 8) * 8192 + (((c % 8) ^ (rg % 8)) << 4) + 4 * (l % 4);
        const int j = 16 * q + c;
        *(__half2*)(d + rg * 128) = __floats2half2_rn(acc[4 * j], acc[4 * j + 1]);
        *(__half2*)(d + (rg + 8) * 128) = __floats2half2_rn(acc[4 * j + 2], acc[4 * j + 3]);
    }
}
// The zone copied out by the warpgroup's 128 threads, 16-byte stores at rows r0 to r0 + 63 and columns c0 to c0 + 127
// of C, rows >= M left out: thread ct copies 8 columns c of rows rr, rr + 8, ..., rr + 56 (rr % 8 = rr).
__device__ __forceinline__ void zone_copy(const uint8_t* zp, half* C, int M, int N, int r0, int c0, int ct)
{
    int n = N;   // n is held back here (an empty asm): else ptxas computes the copy's addresses above the stores into
    asm volatile("" : "+r"(n));   // the zone and spills registers at each tile
    const int rr = ct / 16, c = ct % 16;
    const uint8_t* src = zp + (c / 8) * 8192 + rr * 128 + (((c % 8) ^ rr) << 4);
    half* dst = C + (size_t)(r0 + rr) * n + c0 + 8 * c;
#pragma unroll
    for (int i = 0; i < 8; i++)
        if (r0 + rr + 8 * i < M) *(int4*)(dst + (size_t)8 * i * n) = *(const int4*)(src + 1024 * i);
}

// Step 5 (v0 to v2): 256 threads, two warpgroups that both compute, rows 0 to 63 and 64 to 127 of each tile, on step
// 4's kernel (step4::gemm) made persistent with 5a and 5b. Each warpgroup releases a stage once wait_group 1
// has retired the group that read it; warp 0 then waits for that stage released by both and one of its lanes
// (elect.sync) issues the slice 4 further on, the ring of stages running on from one pair of tiles to the next: the
// next pair's first slices are issued during the current pair's last ones, and land while the output is written. The
// whole warp takes the branch (its condition broadcast from lane 0), one of its lanes issuing the loads. The output
// through each warpgroup's zone (zone_store, zone_copy); each block loads its own A and B; the cluster synchronized
// in one piece. kGrouped: 5b's order, else row after row (v2). The descriptors are rebuilt from the stage's address
// at each slice.
template <bool kAlongM, bool kGrouped>
__global__ void __launch_bounds__(256, 1) gemm(const __grid_constant__ CUtensorMap tmA,
                                                       const __grid_constant__ CUtensorMap tmB, half* C, int M, int N,
                                                       int K)
{
    constexpr int kStages = G::kStages;
    uint8_t* shmem = step4::shared_base();
    uint8_t* sm = shmem + ((1024 - (step1::smem_addr(shmem) & 1023)) & 1023);
    uint64_t* full = (uint64_t*)(sm + G::kBars);
    uint64_t* empty = full + kStages;
    const int tid = threadIdx.x, wg = tid / 128, nt = K / 64;
    const int rank = (int)step4::cluster_rank();
    int UM, UN;
    units(M, N, kAlongM, UM, UN);
    const int nu = UM * UN, cl = blockIdx.x / 2, ncl = gridDim.x / 2;
    const int groupRows = kAlongM ? kGroupRows / 2 : kGroupRows;   // in rows of units
    const unsigned total = (unsigned)((nu - cl + ncl - 1) / ncl) * (unsigned)nt;   // the cluster's slices
    auto tile = [&](int u, int& row0, int& col0) {   // the block's tile in unit u
        int um, un;
        place(u, UM, UN, groupRows, kGrouped, um, un);   // 5b's order, else row after row (v2)
        row0 = (kAlongM ? 2 * um + rank : um) * 128;
        col0 = (kAlongM ? un : 2 * un + rank) * 256;
    };
    if (tid == 0) {
        for (int s = 0; s < kStages; s++) {
            step2::mbar_init(&full[s], 1);
            step2::mbar_init(&empty[s], 2);   // the block's two warpgroups
        }
        step2::mbar_fence_init();
    }
    __syncthreads();
    step4::cluster_arrive_relaxed();   // the cluster synchronized in one piece, as step 4's
    step4::cluster_wait();
    // warp 0's loads, slice after slice over the cluster's units: li the slices issued so far, (lu, lt) the next one
    const bool loader = __shfl_sync(0xffffffffu, tid, 0) < 32;   // the same in the whole warp, broadcast from lane 0
    unsigned li = 0;
    int lu = cl, lt = 0, lrow0 = 0, lcol0 = 0;
    auto load = [&]() {   // warp 0: the next slice into its stage, once the slice 4 before it has released it
        const int s = li % kStages;
        if (li >= kStages) step2::mbar_wait(&empty[s], ((li / kStages) - 1) & 1);
        if (lt == 0) tile(lu, lrow0, lcol0);
        if (step3::elect_one()) {
            uint8_t* st = sm + s * G::kStage;
            step2::mbar_arrive_expect_tx(&full[s], G::kStage);
            step2::tma_load_2d(st, &tmA, lt * 64, lrow0, &full[s]);
#pragma unroll 1   // kept a loop: unrolled, ptxas moves the computing path's descriptors to vector registers
            for (int j = 0; j < 4; j++)
                step2::tma_load_2d(st + G::kABytes + j * 8192, &tmB, lcol0 + 64 * j, lt * 64, &full[s]);
        }
        li++;
        if (++lt == nt) {
            lt = 0;
            lu += ncl;
        }
    };
    if (loader)
        while (li < kStages && li < total) load();
    const int h = wg;   // rows 64 h to 64 h + 63
    const uint32_t s0 = step1::smem_addr(sm);
    const int w = (tid / 32) % 4, l = tid % 32, ct = tid % 128;
    uint8_t* zp = sm + G::kZones + h * G::kZone;   // the warpgroup's output zone
    float acc[128];   // not zeroed: the first wgmma of each tile overwrites it (scaleD 0)
    auto release = [&](unsigned p) {   // slice p's stage is free here: its arrival, then warp 0 refills it
        if (ct == 0) step2::mbar_arrive(&empty[p % kStages]);
        if (loader && li < total) load();
    };
    unsigned it = 0;
    for (int u = cl; u < nu; u += ncl) {
        int row0, col0;
        tile(u, row0, col0);
        for (int t = 0; t < nt; t++, it++) {
            const int s = it % kStages;
            step2::mbar_wait(&full[s], (it / kStages) & 1);
            const uint32_t st = s0 + s * G::kStage;
            const uint64_t da = step1::desc_a(st + h * 64 * 128), db = step1::desc_b(st + G::kABytes);
            step1::wgmma_fence();
#pragma unroll
            for (int kk = 0; kk < 4; kk++) step1::wgmma<256>(acc, da + 2 * kk, db + 128 * kk, t > 0 || kk > 0);
            step1::wgmma_commit();
            step1::wgmma_wait<1>();
            if (t >= 1) release(it - 1);
        }
        step1::wgmma_wait<0>();
        release(it - 1);   // the tile's last slice
        const int r0 = row0 + 64 * h;
#pragma unroll
        for (int q = 0; q < 2; q++) {   // the output: two passes of 128 columns through the warpgroup's zone
            step3::bar_named(2 + h, 128);   // the zone is free
            zone_store(zp, acc, q, w, l);
            step3::bar_named(2 + h, 128);
            if (col0 < N) zone_copy(zp, C, M, N, r0, col0 + 128 * q, ct);
        }
    }
}

}  // namespace step5
