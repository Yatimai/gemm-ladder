/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through step 0 (nvidia_sample.cuh) and steps 1 to 5: the output, whose notice follows.
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
#include "step5.cuh"

namespace step6 {

// ---------------------------------------------------------------- step 6: 16-row bands, the tile per shape
// Steps 1 to 5 run one tile, 128 x 256, at every shape. Step 6 picks a tile by classes of shapes, through two kernels
// and a rule; the rule's quantities are the SM count and the clusters of two that the GPU holds at once, read at the
// first call. Its kernels have every warp computing and warp 0 also loading from the computing path, as step 5's
// (step5::gemm), whose tile it takes; the tiles rebuild their descriptors from the stage's address at each
// slice, the bands offset theirs by the loop's stage. 3b and 3c (the roles and the descriptors in the uniform datapath,
// step5.cuh) left steps 3 to 6 and come back at step 7; the shared operand of 4a and 4d left steps 4 to 6 and come
// back at step 7.
// (6a) M <= 16: bands. A tile of 128 rows spends 7 MMA of 8 on rows >= 16, and one row of 128 x 256 tiles leaves SMs
//      idle when N is small (32 to 40 tiles at N = 8192 to 10240). A band takes 16 rows by J columns of 64 over the
//      whole of K, in the transposed formulation C^T = B^T A^T: the MMA's N carries our M (wgmma m64n16k16), its A
//      operand is our B (MN-major: tnspA), its B operand our A (K-major). J is the least that makes one wave:
//      J = ceil(N / (64 x the SM count)). A block: one warpgroup of 128 threads (J ranges of 8 accumulators), its warp
//      0 also loading. A stage holds 2 KB of A and J boxes of 8 KB of B; as many stages as shared memory holds, 8 at
//      most; below 8, a stage is refilled once its own MMA are retired (wait_group 0): a group fewer in flight for a
//      load more. The output: stmatrix .trans (the fragment of C^T stored as rows of C) into a zone of 2 KB per 64
//      columns, then TMA stores: in the transposed fragment no lane holds two halves side by side in a row of C, so
//      the transposing store is part of 6a.
// (6b) 16 < M <= 128, one row of tiles: a smaller tile when step 5's leaves SMs idle, on a persistent kernel as step
//      5's (5a, 5b). 64 x 128 (two warpgroups on the same 64 rows, 64 columns each, wgmma m64n64) in pairs along M
//      when its pairs make one wave and two tiles cover M; 64 x 128 alone at M <= 64 when its tiles make one wave;
//      else 128 x 128 (wgmma m64n128) when its tiles make one wave, without a cluster; else step 5's tile. 8 stages of
//      24 KB (64 x 128), 6 of 32 KB (128 x 128).
// (6c) M > 128: step 5's tile, 128 x 256, alone (no cluster), on 6b's persistent kernel (wgmma m64n256 per
//      warpgroup). 4 stages of 48 KB. Where N is not a multiple of 256 (step 5's tile cannot take it), 128 x 128
//      alone, as 6b's. Step 7 widens the tile to 128 x 320 where its waves, times its area, are fewer (7e,
//      step7.cuh).
// Clusters: 64 x 128 in pairs along M when two of them cover M; step 5's tile over one row of tiles keeps its pairs
// along N; each block of a pair loads its own A and B and releases its stages to itself (an arrival from each
// warpgroup); the bands, 64 x 128 alone, 128 x 128 and 6c have no cluster. A pair's cluster is synchronized in one
// piece (each thread's wait right after its arrival), once its barriers are initialized.
// The tiles' output, as step 5's: each warpgroup stores its rows in passes of up to 128 columns into its zone after the
// stages (each lane's pairs of halves in boxes of 64 x 64 at the 128-byte swizzle), and its 128 threads copy them out
// by 16-byte stores, rows >= M and boxes >= N left out. 4c, 5d and 5e left the code; 5c comes back at step 7 (7j).
// Compiled (nvcc 13.2, sm_90a, without -lineinfo, as the judge): no spill, no local memory. Main loops, per K64 and per
// warp (static counts), the path of a warp that does not load (warp 0 adds its loads: the arrival with the bytes and
// the UTMALDG; in the tiles, first the wait for the stage released by both warpgroups):
//   bands J = 1: 33.5 (two slices per pass; 4 HGMMA.64x16x16 with tnspA per slice; no R2UR; 34 registers), J = 2: 52,
//   J = 7: 120 (28 HGMMA; 90 registers at J = 8; ptxas keeps the stage in a uniform register); one STSM.16.MT88.4 and
//   one UTMASTG.2D per 64 columns of the output.
//   64 x 128 in pairs: 67 (4 HGMMA.64x64x16; 8 R2UR: the descriptors rebuilt at each slice; 86 registers), alone: 66
//   (94); 128 x 128: 62 (4 HGMMA.64x128x16; 131); 128 x 256: 60 (4 HGMMA.64x256x16; 198); step 5's tile: 60
//   (step5.cuh). The tiles' output, per pass of 128 columns and per warp: 32 STS.32, then 8 LDS.128 and 8 STG.E.128.
//   The tiles in pairs: one UCGABAR_ARV and one UCGABAR_WAIT.
// Variants (g_var in step6.cu, for the interleaved probe; each removal keeps the other pieces):
//   v0 = step 6 (6a, 6b, 6c); v1 = without 6a (M <= 16 by step 5's tile); v2 = without 6b (16 < M <= 128 by step 5's
//   tile); v3 = without 6c (step 5's tile, in pairs); v4 = step 5 itself, the anchor.

#ifdef GEMM_LADDER_EMULATE
// A host emulation runs the transposed wgmma (A MN-major, B K-major) and stmatrix .trans.
void emu_wgmma_t(float* d, int n, uint64_t da, uint64_t db, int scaleD, int tnspA, int tnspB);
void emu_stmatrix_x4_trans(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3);
#endif

// One wgmma.mma_async m64n16k16, fp32 accumulators, A MN-major (tnspA: our B, its columns along the MMA's M), B
// K-major (our A, its rows along the MMA's N). Lane l of warp w (g = l / 4, q = l % 4): d[0], d[1] = D[16 w + g][2 q],
// [2 q + 1]; d[2], d[3] the same in row 16 w + g + 8; d[4] to d[7] the same in columns 8 + 2 q and 8 + 2 q + 1.
__device__ __forceinline__ void wgmma_n16t(float* d, uint64_t da, uint64_t db, int scaleD)
{
#ifdef GEMM_LADDER_EMULATE
    emu_wgmma_t(d, 16, da, db, scaleD, 1, 0);
#else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1, "
                 "1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "l"(da), "l"(db), "r"(scaleD));
#endif
}

// stmatrix.sync.aligned.m8n8.x4.trans.shared.b16: the warp stores four 8 x 8 matrices of halves, each transposed.
// Lane l gives the address of row l % 8 of matrix l / 8 (16 bytes); register i of lane l holds elements 2 (l % 4) and
// 2 (l % 4) + 1 of row l / 4 of matrix i (the layout of the MMA's accumulator fragments); row r of the stored matrix,
// at the address of lane 8 i + r, is column r of the fragment.
__device__ __forceinline__ void stmatrix_x4_trans(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3)
{
#ifdef GEMM_LADDER_EMULATE
    emu_stmatrix_x4_trans(addr, r0, r1, r2, r3);
#else
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                 ::"r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
#endif
}

// 6a's geometry: J columns of 64, NS stages (each A's 16 rows x 64 K, then J boxes of B), the output zone, the
// barriers.
template <int J, int NS>
struct Bands {
    static constexpr int kABytes = 16 * 128, kStage = kABytes + J * 8192;   // 2 KB of A + J boxes of 8 KB of B
    static constexpr int kZone = NS * kStage, kBars = kZone + J * 2048;    // the zone: 16 rows x 64 columns per box
    static constexpr int kSmemBytes = kBars + 16 * NS + 1024;               // + the alignment of the stages to 1024 bytes
    static_assert(kSmemBytes <= 227 * 1024, "227 KB of shared memory at most");
};
// The most stages of J columns that shared memory holds, 8 at most.
template <int J>
constexpr int bandStages()
{
    int ns = 8;
    while (ns > 1 && Bands<J, 1>::kStage * ns + J * 2048 + 16 * ns + 1024 > 227 * 1024) ns--;
    return ns;
}

// A's map for the bands: boxes of 64 K x 16 rows; C's: boxes of 64 columns x 16 rows.
inline CUtensorMap map_a16(const half* A, int M, int K) { return step2::make_map(A, K, M, 64, 16); }
inline CUtensorMap map_c16(half* C, int M, int N) { return step2::make_map(C, N, M, 64, 16); }

// 6a's bands (step 6's v0, and step 7's v6, step 6 itself): one warpgroup of 128 threads that computes, warp 0 issuing
// the loads from the computing path: once wait_group 1 (0 below 8 stages) has retired a slice's MMA (the warpgroup's),
// its stage is free and one lane of warp 0 (elect.sync) issues the slice NS further on into it. No empty barrier: its
// only arrival would be thread 0's, in warp 0 itself, after that same wait_group, so it would tell warp 0 nothing (its
// 8 bytes per stage stay in Bands' layout, unused). The whole warp takes the branch (its condition broadcast from lane
// 0). The descriptors' offsets from the loop's stage (ptxas keeps it in a uniform register).
template <int J, int NS>
__device__ __forceinline__ void bands_body(const CUtensorMap& tmA, const CUtensorMap& tmB,
                                                   const CUtensorMap& tmC, half* C, int M, int N, int K)
{
    using G = Bands<J, NS>;
    constexpr bool kZ = NS < 8;   // a stage refilled once its own MMA are retired (wait_group 0)
    uint8_t* shmem = step4::shared_base();
    uint8_t* sm = shmem + ((1024 - (step1::smem_addr(shmem) & 1023)) & 1023);
    uint64_t* full = (uint64_t*)(sm + G::kBars);
    const int tid = threadIdx.x, nt = K / 64, col0 = blockIdx.x * 64 * J;
    if (tid == 0) {
        for (int s = 0; s < NS; s++)
            step2::mbar_init(&full[s], 1);
        step2::mbar_fence_init();
    }
    __syncthreads();
    // warp 0's loads: li the slices issued so far
    const bool loader = __shfl_sync(0xffffffffu, tid, 0) < 32;   // the same in the whole warp, broadcast from lane 0
    int li = 0;
    auto load = [&]() {   // warp 0: the next slice into its stage, which the slice NS before it has left
        const int s = li % NS;
        if (step3::elect_one()) {
            uint8_t* st = sm + s * G::kStage;
            constexpr uint32_t kBytes = G::kStage;   // A's 2 KB and the J boxes of B
            step2::mbar_arrive_expect_tx(&full[s], kBytes);
            step2::tma_load_2d(st, &tmA, li * 64, 0, &full[s]);
#pragma unroll 1   // kept a loop, as step 5's: the computing path's descriptors stay in uniform registers
            for (int j = 0; j < J; j++)
                step2::tma_load_2d(st + G::kABytes + j * 8192, &tmB, col0 + 64 * j, li * 64, &full[s]);
        }
        li++;
    };
    if (loader)
        while (li < NS && li < nt) load();
    // the warpgroup: J ranges of the transposed MMA (C^T's J x 64 rows, our 16 rows)
    const uint32_t s0 = step1::smem_addr(sm);
    const uint64_t dB0 = step1::desc_b(s0 + G::kABytes), dA0 = step1::desc_a(s0);   // our B: the MMA's A; our A: its B
    float acc[J][8];   // not zeroed: the first wgmma of each range overwrites it (scaleD 0)
    for (int t = 0; t < nt; t++) {
        const int s = t % NS;
        step2::mbar_wait(&full[s], (t / NS) & 1);
        const uint32_t ds = (G::kStage >> 4) * s;   // the stage's offset in units of 16 bytes
        step1::wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
#pragma unroll
            for (int j = 0; j < J; j++) {   // the MMA's A advances 2 KB per K16 (16 rows of K), its B 32 bytes
                float* d = acc[j];
                wgmma_n16t(d, dB0 + (512 * j + 128 * kk) + ds, dA0 + ds + 2 * kk, t > 0 || kk > 0);
            }
        step1::wgmma_commit();
        if constexpr (kZ) step1::wgmma_wait<0>();   // below 8 stages, the slice's own stage
        else step1::wgmma_wait<1>();                // else the stage of the slice before
        if ((kZ || t >= 1) && loader && li < nt) load();   // that stage free: warp 0 refills it
    }
    step1::wgmma_wait<0>();
    // the output: each warp's 16 rows of C^T (columns 16 w to 16 w + 15 of a box of 64), stored transposed: lane l
    // gives the address of row 8 (l / 16) + l % 8 of C, chunk 2 w + (l / 8) % 2 of the box's 8 chunks of 16 bytes
    const int w = tid / 32, l = tid % 32, m = 8 * (l / 16) + l % 8, c = (l / 8) % 2 + 2 * w;
    const uint32_t zone = s0 + G::kZone;
#pragma unroll
    for (int j = 0; j < J; j++)
        stmatrix_x4_trans(zone + j * 2048 + m * 128 + ((c ^ (m % 8)) << 4), step5::pack_half2(acc[j][0], acc[j][1]),
                          step5::pack_half2(acc[j][2], acc[j][3]),
                          step5::pack_half2(acc[j][4], acc[j][5]), step5::pack_half2(acc[j][6], acc[j][7]));
    step1::fence_proxy_async();   // the zone, visible to the TMA
    step3::bar_named(1, 128);
    if (tid == 0) {
#pragma unroll
        for (int j = 0; j < J; j++)
            if (col0 + 64 * j < N) step5::tma_store_2d(&tmC, col0 + 64 * j, 0, zone + 2048 * j);
        step5::bulk_commit();
        step5::bulk_wait<0, false>();
    }
}
template <int J, int NS>
__global__ void __launch_bounds__(128, 1) bands(const __grid_constant__ CUtensorMap tmA,
                                                        const __grid_constant__ CUtensorMap tmB,
                                                        const __grid_constant__ CUtensorMap tmC, half* C, int M, int N,
                                                        int K)
{
    bands_body<J, NS>(tmA, tmB, tmC, C, M, N, K);
}


// ---------------------------------------------------------- 6b, 6c and 7e (step 7): the tiles of the persistent kernel
// wgmma m64nNk16 as step 1's (A K-major, B MN-major), at the widths step 1 does not have: n64 (6b's tile of 64 x 128,
// one warpgroup per 64 columns) and n192 (7e's tile of 128 x 320: n192 then n128).
template <int N>
__device__ __forceinline__ void wgmma(float (&d)[N / 2], uint64_t da, uint64_t db, int scaleD);
template <>
__device__ __forceinline__ void wgmma<64>(float (&d)[32], uint64_t da, uint64_t db, int scaleD)
{
#ifdef GEMM_LADDER_EMULATE
    step1::emu_wgmma(d, 64, da, db, scaleD);
#else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "l"(da), "l"(db), "r"(scaleD));
#endif
}

template <>
__device__ __forceinline__ void wgmma<192>(float (&d)[96], uint64_t da, uint64_t db, int scaleD)
{
#ifdef GEMM_LADDER_EMULATE
    step1::emu_wgmma(d, 192, da, db, scaleD);
#else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %98, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n192k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, %96, %97, p, 1, 1, 0, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])
                 : "l"(da), "l"(db), "r"(scaleD));
#endif
}

// The cluster of two of 6b and 7e: none, or two tiles one above the other, each block loading its own A and B.
enum Pair { kSolo, kAlongM };

// A tile of BM x BN in NS stages of K64. BM = 64: two computing warpgroups on the same 64 rows, each BN / 2 columns;
// BM = 128: each 64 rows, all BN columns. The output by passes of 128 columns at most per warpgroup (boxes of 64 x 64).
template <int BM, int BN, int NS>
struct Tile {
    static constexpr bool kSplitN = BM == 64;
    static constexpr int kWgCols = kSplitN ? BN / 2 : BN;           // a computing warpgroup's columns
    static constexpr int kAcc = kWgCols / 2;                        // its accumulators per thread
    static constexpr int kBoxes = BN / 64;                          // B's boxes of 64 columns in a stage
    static constexpr int kABytes = BM * 128, kStage = kABytes + kBoxes * 8192;
    static constexpr int kPassCols = kWgCols < 128 ? kWgCols : 128, kPasses = (kWgCols + 127) / 128;
    static constexpr int kZone = kPassCols / 64 * 8192;             // a warpgroup's output zone
    static constexpr int kZones = NS * kStage, kBars = kZones + 2 * kZone;
    static constexpr int kSmemBytes = kBars + 16 * NS + 1024;       // + the alignment of the stages to 1024 bytes
    static_assert(kSmemBytes <= 227 * 1024, "227 KB of shared memory at most");
};

// The units of work: tiles, or pairs of tiles one above the other (as step 5's along M).
__host__ __device__ inline void units6(int M, int N, int BM, int BN, Pair pair, int& UM, int& UN)
{
    const int tilesM = (M + BM - 1) / BM, tilesN = (N + BN - 1) / BN;
    UM = pair == kAlongM ? (tilesM + 1) / 2 : tilesM;
    UN = tilesN;
}

// The persistent kernel of step 5's tile (5a, 5b) at another tile and cluster: step 6's tiles (6b, 6c) and 7e's, as
// step 7 runs it without the roles (its v8). 256 threads, two warpgroups that both compute (BM = 64: the same 64 rows,
// BN / 2 columns each; BM = 128: 64 rows each), warp 0 issuing the loads from the computing path, as step 5's
// (step5::gemm): each warpgroup releases a stage once wait_group 1 has retired the group that read it (kWait1 false:
// wait_group 0 and the slice's own stage); warp 0 then waits for that stage released by both and one of its lanes
// (elect.sync) issues the slice NS further on, the ring running on from one tile to the next (the next tile's first
// slices issued during the current tile's last ones). The whole warp takes the branch (its condition broadcast from
// lane 0). The descriptors rebuilt from the stage's address at each slice. groupRows: the rows of units per group of
// 5b's order (step 5's: 16 rows of tiles, 8 rows of pairs along M). In pairs along M, each block loads its own A and B
// and releases its stages to itself; the cluster synchronized in one piece. 7e's 128 x 320: wgmma m64n192 then m64n128.
// The output by passes of up to 128 columns through the warpgroup's zone, then 16-byte stores by its threads.
template <int BM, int BN, int NS, Pair kPair, bool kWait1>
__device__ __forceinline__ void gemm_body(const CUtensorMap& tmA, const CUtensorMap& tmB, half* C, int M,
                                                  int N, int K, int groupRows)
{
    using T = Tile<BM, BN, NS>;
    constexpr bool kCl = kPair != kSolo;
    uint8_t* shmem = step4::shared_base();
    uint8_t* sm = shmem + ((1024 - (step1::smem_addr(shmem) & 1023)) & 1023);
    uint64_t* full = (uint64_t*)(sm + T::kBars);
    uint64_t* empty = full + NS;
    const int tid = threadIdx.x, wg = tid / 128, nt = K / 64;
    const int rank = kCl ? (int)step4::cluster_rank() : 0;
    int UM, UN;
    units6(M, N, BM, BN, kPair, UM, UN);
    const int nu = UM * UN, cl = kCl ? blockIdx.x / 2 : blockIdx.x, ncl = kCl ? gridDim.x / 2 : gridDim.x;
    const unsigned total = (unsigned)((nu - cl + ncl - 1) / ncl) * (unsigned)nt;   // the block's slices
    auto tile = [&](int u, int& row0, int& col0) {
        int um, un;
        step5::place(u, UM, UN, groupRows, true, um, un);
        row0 = BM * (kPair == kAlongM ? 2 * um + rank : um);
        col0 = un * BN;
    };
    if (tid == 0) {
        for (int s = 0; s < NS; s++) {
            step2::mbar_init(&full[s], 1);
            step2::mbar_init(&empty[s], 2);   // the block's two warpgroups
        }
        step2::mbar_fence_init();
    }
    __syncthreads();
    if constexpr (kCl) {   // the cluster synchronized in one piece
        step4::cluster_arrive_relaxed();
        step4::cluster_wait();
    }
    // warp 0's loads, slice after slice over the block's tiles: li the slices issued so far, (lu, lt) the next one
    const bool loader = __shfl_sync(0xffffffffu, tid, 0) < 32;   // the same in the whole warp, broadcast from lane 0
    unsigned li = 0;
    int lu = cl, lt = 0, lrow0 = 0, lcol0 = 0;
    auto load = [&]() {   // warp 0: the next slice into its stage, once the slice NS before it has released it
        const int s = li % NS;
        if (li >= NS) step2::mbar_wait(&empty[s], ((li / NS) - 1) & 1);
        if (lt == 0) tile(lu, lrow0, lcol0);
        if (step3::elect_one()) {
            uint8_t* st = sm + s * T::kStage;
            step2::mbar_arrive_expect_tx(&full[s], T::kStage);
            step2::tma_load_2d(st, &tmA, lt * 64, lrow0, &full[s]);
#pragma unroll 1   // kept a loop, as step 5's: the computing path's descriptors stay in uniform registers
            for (int j = 0; j < T::kBoxes; j++)
                step2::tma_load_2d(st + T::kABytes + j * 8192, &tmB, lcol0 + 64 * j, lt * 64, &full[s]);
        }
        li++;
        if (++lt == nt) {
            lt = 0;
            lu += ncl;
        }
    };
    if (loader)
        while (li < NS && li < total) load();
    const int h = wg;   // the warpgroup's rows (BM = 128) or columns (BM = 64)
    const uint32_t s0 = step1::smem_addr(sm);
    const int w = (tid / 32) % 4, l = tid % 32, ct = tid % 128;
    const uint32_t zone = s0 + T::kZones + h * T::kZone;   // the warpgroup's output zone
    float acc[T::kAcc];   // not zeroed: the first wgmma of each tile overwrites it (scaleD 0)
    auto release = [&](unsigned p) {   // slice p's stage is free here: its arrival, then warp 0 refills it
        if (ct == 0) step2::mbar_arrive(&empty[p % NS]);
        if (loader && li < total) load();
    };
    unsigned it = 0;
    for (int u = cl; u < nu; u += ncl) {
        int row0, col0;
        tile(u, row0, col0);
        for (int t = 0; t < nt; t++, it++) {
            const int s = it % NS;
            step2::mbar_wait(&full[s], (it / NS) & 1);
            const uint32_t st = s0 + T::kStage * s;   // the descriptors rebuilt from the stage's address
            step1::wgmma_fence();
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                const uint32_t ra = st + (T::kSplitN ? 0 : h * 64 * 128);   // A's and B's place
                const uint32_t rb = st + T::kABytes + (T::kSplitN ? 8192 * h * (T::kWgCols / 64) : 0);
                const uint64_t da = step1::desc_a(ra) + 2 * kk;
                const uint64_t db = step1::desc_b(rb) + 128 * kk;
                if constexpr (T::kWgCols == 64) {
                    wgmma<64>(*reinterpret_cast<float(*)[32]>(acc), da, db, t > 0 || kk > 0);
                } else if constexpr (T::kWgCols == 128) {
                    step1::wgmma<128>(*reinterpret_cast<float(*)[64]>(acc), da, db, t > 0 || kk > 0);
                } else if constexpr (T::kWgCols == 256) {   // 6c
                    step1::wgmma<256>(*reinterpret_cast<float(*)[128]>(acc), da, db, kk > 0 || t > 0);
                } else {   // 7e's 320: n192 over B's boxes 0 to 2, then n128 over boxes 3 and 4
                    static_assert(T::kWgCols == 320, "the tiles: 64, 128, 256 or 320 columns per warpgroup");
                    wgmma<192>(*reinterpret_cast<float(*)[96]>(acc), da, db, kk > 0 || t > 0);
                    step1::wgmma<128>(*reinterpret_cast<float(*)[64]>(&acc[96]), da, db + 1536, kk > 0 || t > 0);
                }
            }
            step1::wgmma_commit();
            if constexpr (kWait1) {
                step1::wgmma_wait<1>();
                if (t > 0) release(it - 1);   // the stage of the slice before
            } else {   // kWait1 false: the slice's own stage, once the warpgroup's MMA are retired
                step1::wgmma_wait<0>();
                release(it);   // warp 0 refills it
            }
        }
        if constexpr (kWait1) {
            step1::wgmma_wait<0>();
            release(it - 1);   // its last slice
        }
        // the output, as step 5's, by passes of up to 128 of the warpgroup's columns: each lane's pairs of halves at
        // their place in the boxes of 64 x 64 at the 128-byte swizzle, rows rg and rg + 8 (the same row % 8)
        const int rg = 16 * w + l / 4;
        const int r0 = row0 + (T::kSplitN ? 0 : 64 * h), c0w = col0 + (T::kSplitN ? T::kWgCols * h : 0);
#pragma unroll
        for (int q = 0; q != T::kPasses; q++) {
            constexpr int kFull = T::kPassCols;
            const int qc = T::kWgCols - 128 * q < kFull ? T::kWgCols - 128 * q : kFull;   // the pass's columns
            step3::bar_named(2 + h, 128);
            uint8_t* zw = sm + (zone - s0);
#pragma unroll
            for (int c = 0; c < kFull / 8; c++) {
                if (8 * c >= qc) break;
                const int j = 16 * q + c;
                uint8_t* d = zw + 8192 * (c / 8) + (((c % 8) ^ (rg % 8)) << 4) + 4 * (l % 4);
                *(__half2*)(d + rg * 128) = __floats2half2_rn(acc[4 * j], acc[4 * j + 1]);
                *(__half2*)(d + 128 * (rg + 8)) = __floats2half2_rn(acc[4 * j + 2], acc[4 * j + 3]);
            }
            step3::bar_named(2 + h, 128);
            {   // 16-byte stores by the warpgroup's 128 threads, rows >= M and boxes >= N left out: the threads
                // split between the pass's boxes of 64 columns (a warp in one box), thread t of a box copying
                // chunk t % 8 of rows t / 8, t / 8 + step, ..., in the box's swizzle
                const int nb = qc / 64, tpb = 128 / nb, b = ct / tpb, t = ct % tpb, c8 = t % 8, rr = t / 8;
                const int step = tpb / 8, col = c0w + 128 * q + 64 * b;
                const uint8_t* src = sm + (zone - s0) + b * 8192 + (((rr % 8) ^ c8) << 4);
                if (col < N) {
#pragma unroll
                    for (int i = 0; i < 64 / step; i++) {
                        const int r = rr + step * i;   // step a multiple of 8: r % 8 = rr % 8, the same swizzle
                        if (r0 + r < M)
                            *(int4*)(C + (size_t)(r0 + r) * N + col + 8 * c8) = *(const int4*)(src + r * 128);
                    }
                }
            }
        }
    }
}
template <int BM, int BN, int NS, Pair kPair, bool kWait1 = true>
__global__ void __launch_bounds__(256, 1) gemm(const __grid_constant__ CUtensorMap tmA,
                                                       const __grid_constant__ CUtensorMap tmB, half* C, int M, int N,
                                                       int K)
{
    constexpr int groupRows = kPair == kAlongM ? step5::kGroupRows / 2 : step5::kGroupRows;
    gemm_body<BM, BN, NS, kPair, kWait1>(tmA, tmB, C, M, N, K, groupRows);
}


// ---------------------------------------------------------------- the rule (host)
// Which kernel a shape takes, by classes of shapes (sms: the SM count; pairs: the clusters of two the GPU holds at
// once). M <= 16: the bands (6a), J the least that makes one wave. 16 < M <= 128 (one row of tiles): 64 x 128 in pairs
// along M when the pairs make one wave and two tiles cover M (M > 64), 64 x 128 alone when one covers it;
// else 128 x 128 when its tiles make one wave (no cluster: one tile covers M); else step 5's tile (6b). M > 128: step
// 5's tile alone (6c). Step 5's tile needs N a multiple of 256: other N take 128 x 128. With 7e
// (with7e: step 7's rule, step7.cu), at M > 128: 128 x 320 when its waves times its area (1.25 of 128 x 256's) are
// fewer than step 5's tile's, and at every N not a multiple of 256 (7e), alone when its tiles make one wave, else in
// pairs along M.
enum Kind { kBandsK, kT64Pair, kT64Solo, kT128, kT256, kT320, kT256Solo, kT320Solo };
struct Choice { Kind kind; int J; };
inline Choice choose(int M, int N, int sms, int pairs, bool with7e)
{
    const int J = (N + 64 * sms - 1) / (64 * sms);
    if (M <= 16 && J <= 8) return {kBandsK, J};
    if (M <= 128) {
        const int cols = (N + 127) / 128;   // tiles of 128 columns
        if (M > 64 && cols <= pairs) return {kT64Pair, 0};
        if (M <= 64 && cols <= 2 * pairs) return {kT64Solo, 0};
        if (cols <= 2 * pairs || N % 256) return {kT128, 0};
        return {kT256, 0};
    }
    if (!with7e) return {N % 256 ? kT128 : kT256Solo, 0};   // step 6: 6c, or 128 x 128 where step 5's tile cannot go
    const long rows = (M + 127) / 128, tiles320 = rows * ((N + 319) / 320);
    const Kind k320 = tiles320 <= sms ? kT320Solo : kT320;   // 7e alone when its tiles make one wave
    if (N % 256) return {k320, 0};
    const long w256 = (rows * (N / 256) + 2L * pairs - 1) / (2L * pairs);
    const long w320 = (tiles320 + 2L * pairs - 1) / (2L * pairs);
    return {4 * w320 * 5 < 4 * w256 * 4 ? k320 : kT256Solo, 0};   // w320 x 1.25 < w256
}

}  // namespace step6
