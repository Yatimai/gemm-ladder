/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through step 0 (nvidia_sample.cuh) and step 1 (step1.cuh): the output, whose notice follows.
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
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifndef GEMM_LADDER_EMULATE
#include <cuda.h>   // CUtensorMap and its encoding
#endif
#include "step1.cuh"

namespace step2 {

// ---------------------------------------------------------- step 2: TMA loads in three stages, one MMA group in flight
// The H100's tensor memory accelerator, written from the PTX ISA (CuTe's Hopper tutorial on the TMA shows the same
// engine and the same ring), in place of step 1's copy through registers. Kept from step 1 (step1.cuh, v0): one block
// of two warpgroups per 128 x 256 tile, wgmma m64n256k16 by descriptors on the same swizzled layout, the fp16 output
// through shared memory then 16-byte stores. Shapes: M a multiple of 16, N of 256, K of 64.
// (2a) TMA: one thread loads a slice with one instruction per box (cp.async.bulk.tensor.2d, one UTMALDG): A's box of
//      64 K x 128 rows (16 KB) and B's four boxes of 64 columns x 64 rows of K (8 KB each); the boxes land in shared
//      memory in the layout the descriptors read (the tensor map's 128-byte swizzle), the rows of A from M on as zeros
//      (no clamp; they are computed and never written). A transaction barrier (mbarrier) per stage counts the bytes:
//      the loading thread announces 48 KB (arrive.expect_tx), each box's bytes complete it as they land, and the
//      threads wait for the stage's phase (try_wait.parity) before its MMA.
// (2b) The ring of kStages stages, the tutorial's: a full barrier per stage (one arrival, the announcement, plus the
//      bytes) and an empty barrier (two arrivals, one thread per warpgroup once wait_group 1 has retired the group that
//      read the stage). The loading thread (thread 0, also a consumer) loads the first kStages slices, then, at slice
//      t, once the stage of slice t - 1 is released by both warpgroups, slice t - 1 + kStages into it: kStages - 1
//      slices ahead of the MMA.
// (2c) The boxes of 64 x 64 (and 64 x 128 for A) at the 128-byte swizzle, read as they land; the L2 promotion of the
//      tensor maps at 128 bytes (a miss in L2 fetches a whole 128-byte line from DRAM).
// (2d) The asynchronous chain, which left step 1 at the test global and enters here: wgmma.fence, the 4 wgmma of a
//      slice, wgmma.commit_group, then wgmma.wait_group 1: the group of slice t stays in flight while the ring goes on
//      (as in CUTLASS's warp-specialized kernels for sm_90, public source; the tutorial waits for 0), and the stage of
//      slice t - 1 is released once its group is retired. Without it (v3), wait_group 0: the stage of slice t is
//      released once its own group is retired.
// Compiled (nvcc 13.2, sm_90a, without -lineinfo, as the judge): v0, v1, v3 and v4 at 154 registers, no spill; one
// block per SM. Main loop of v0, per K64 and per warp (static counts): the computing path 79 instructions (the wait for
// the full stage, 4 HGMMA.64x256x16, WARPGROUP.ARRIVE and DEPBAR, the stage's release by one thread per warpgroup, the
// descriptors: 16 R2UR); thread 0 adds 73 (the wait for the empty stage, the announcement, the five UTMALDG.2D and
// their 15 R2UR, each UTMALDG in a loop of ELECT and BRA.U.ANY: ptxas wraps a TMA issued under a per-thread condition
// in a loop that elects one lane). The other variants, whose loops ptxas unrolls by 2, per K64 and per warp, thread
// 0's part included: v1 108.5 (13 R2UR), v3 141 (31), v4 103.5 (13). Step 1's loop: 203 per K64 and per warp.
// Variants (g_var in step2.cu, for the interleaved probe):
//   v0 = step 2 (2a to 2d; 3 stages; 148 528 bytes of dynamic shared memory);
//   v1 = 2 stages (99 360 bytes): step 1's depth, the copy by the TMA;
//   v2 = step 1 itself (step1::gemm<256>), the anchor in the same probe;
//   v3 = without 2d: wait_group 0;
//   v4 = 2 stages without 2d (against v2, which has no 2d either, it measures 2a).

#ifdef GEMM_LADDER_EMULATE
// A host emulation runs each thread as a coroutine; a wait suspends it until the barrier's phase completes; a TMA load
// lands when the emulator decides (when issued, or once nothing else can go on), and completes its bytes then.
void emu_mbar_init(uint64_t* bar, int count);
void emu_mbar_arrive(uint64_t* bar, uint32_t bytes);   // bytes > 0: arrive.expect_tx
void emu_mbar_wait(uint64_t* bar, uint32_t parity);
void emu_tma_load_2d(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar);
void emu_encode_map(CUtensorMap* map, const half* base, uint64_t cols, uint64_t rows, uint32_t boxCols, uint32_t boxRows);
#endif

// mbarrier.init: count arrivals complete a phase (with the bytes announced); fence.mbarrier_init makes the
// initialization visible to the async proxy (the TMA completes its bytes there) before the barrier is used.
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count)
{
#ifdef GEMM_LADDER_EMULATE
    emu_mbar_init(bar, count);
#else
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(step1::smem_addr(bar)), "r"(count) : "memory");
#endif
}
__device__ __forceinline__ void mbar_fence_init()
{
#ifndef GEMM_LADDER_EMULATE
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
}

// mbarrier.arrive.expect_tx: one arrival, and `bytes` more to come from the TMA in the current phase.
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes)
{
#ifdef GEMM_LADDER_EMULATE
    emu_mbar_arrive(bar, bytes);
#else
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(step1::smem_addr(bar)), "r"(bytes)
                 : "memory");
#endif
}

// mbarrier.arrive: one arrival.
__device__ __forceinline__ void mbar_arrive(uint64_t* bar)
{
#ifdef GEMM_LADDER_EMULATE
    emu_mbar_arrive(bar, 0);
#else
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(step1::smem_addr(bar)) : "memory");
#endif
}

// Waits for the phase of parity `parity` of the barrier to complete (try_wait.parity in a loop). STEP2_GUARD (the
// first contact of a new kernel): the wait gives up after 2^28 tries, prints where it is stuck and traps.
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity)
{
#ifdef GEMM_LADDER_EMULATE
    emu_mbar_wait(bar, parity);
#elif defined(STEP2_GUARD)
    for (long i = 0;; i++) {
        uint32_t ok;
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(ok) : "r"(step1::smem_addr(bar)), "r"(parity) : "memory");
        if (ok) break;
        if (i == (1L << 28)) {
            printf("STEP2_GUARD: block %d, thread %d stuck on the barrier at shared byte %u, parity %u\n", blockIdx.x,
                   threadIdx.x, step1::smem_addr(bar), parity);
            __trap();
        }
    }
#else
    asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}\n"
                 ::"r"(step1::smem_addr(bar)), "r"(parity) : "memory");
#endif
}

// One TMA load of a 2D box (cp.async.bulk.tensor.2d): coordinates in elements, the fast one (columns) first; the bytes
// complete on bar.
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar)
{
#ifdef GEMM_LADDER_EMULATE
    emu_tma_load_2d(dst, map, c0, c1, bar);
#else
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 ::"r"(step1::smem_addr(dst)), "l"(map), "r"(c0), "r"(c1), "r"(step1::smem_addr(bar)) : "memory");
#endif
}

// ---------------------------------------------------------------- the tensor maps (host)
// A row-major fp16 matrix of `rows` x `cols`, boxes of boxCols x boxRows, 128-byte swizzle, L2 promotion of 128
// bytes (promo256: 256, step 7's 7c), out-of-bounds elements read as zeros (cuTensorMapEncodeTiled, reached through
// the runtime, so that no library is linked).
inline CUtensorMap make_map(const half* base, uint64_t cols, uint64_t rows, uint32_t boxCols, uint32_t boxRows,
                            bool promo256 = false)
{
    CUtensorMap map;
#ifdef GEMM_LADDER_EMULATE
    (void)promo256;   // a hint to the L2: no result depends on it
    emu_encode_map(&map, base, cols, rows, boxCols, boxRows);
#else
    typedef CUresult (*Encode)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                               const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                               CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    static Encode encode = nullptr;
    if (!encode) {
        void* fn = nullptr;
        cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
        encode = reinterpret_cast<Encode>(fn);
    }
    const cuuint64_t dims[2] = {cols, rows}, strides[1] = {cols * 2};
    const cuuint32_t box[2] = {boxCols, boxRows}, elementStrides[2] = {1, 1};
    encode(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, const_cast<half*>(base), dims, strides, box, elementStrides,
           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
           promo256 ? CU_TENSOR_MAP_L2_PROMOTION_L2_256B : CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
#endif
    return map;
}
inline CUtensorMap map_a(const half* A, int M, int K) { return make_map(A, K, M, 64, 128); }   // boxes of 64 K x 128 rows
inline CUtensorMap map_b(const half* B, int K, int N) { return make_map(B, N, K, 64, 64); }    // 64 columns x 64 K

// ---------------------------------------------------------------- the kernel
template <int kStages>
struct Geometry {
    static constexpr int kABytes = 128 * 128, kBBytes = 64 * 256 * 2, kStage = kABytes + kBBytes;   // 16 + 32 KB
    static constexpr int kOPitch = 256 + 8;   // the output's pitch in halves (132 words, 4 mod 32: no bank conflict)
    static constexpr int kLoopBytes = kStages * kStage, kOutBytes = 128 * kOPitch * 2;
    static constexpr int kBytes = kLoopBytes > kOutBytes ? kLoopBytes : kOutBytes;
    static constexpr int kBars = kBytes;               // the barriers after the stages: full[kStages], empty[kStages]
    static constexpr int kSmemBytes = kBytes + 16 * kStages + 1024;   // + the alignment of the stages to 1024 bytes
    static_assert(kStages >= 2 && kSmemBytes <= 227 * 1024, "2 stages at least; 227 KB of shared memory at most");
};

// One block of 256 threads (two warpgroups) per 128 x 256 tile, tiles row by row (blockIdx.x). kWait1: 2d (wait_group
// 1: a stage released once the next slice's group is in flight), else wait_group 0 (a stage released once its own group
// is retired).
template <int kStages, bool kWait1 = true>
__global__ void __launch_bounds__(256, 1) gemm(const __grid_constant__ CUtensorMap tmA,
                                               const __grid_constant__ CUtensorMap tmB, half* C, int M, int N, int K)
{
    using G = Geometry<kStages>;
    extern __shared__ __align__(1024) uint8_t shmem[];
    uint8_t* sm = shmem + ((1024 - (step1::smem_addr(shmem) & 1023)) & 1023);
    uint64_t* full = (uint64_t*)(sm + G::kBars);
    uint64_t* empty = full + kStages;
    const int tid = threadIdx.x, wg = tid / 128, nt = K / 64;
    const int tilesN = N / 256, row0 = (blockIdx.x / tilesN) * 128, col0 = (blockIdx.x % tilesN) * 256;
    if (tid == 0) {
        for (int s = 0; s < kStages; s++) {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], 2);
        }
        mbar_fence_init();
    }
    __syncthreads();
    auto load = [&](int t) {   // slice t into its stage (thread 0)
        const int s = t % kStages;
        uint8_t* st = sm + s * G::kStage;
        mbar_arrive_expect_tx(&full[s], G::kStage);
        tma_load_2d(st, &tmA, t * 64, row0, &full[s]);
#pragma unroll
        for (int j = 0; j < 4; j++) tma_load_2d(st + G::kABytes + j * 8192, &tmB, col0 + 64 * j, t * 64, &full[s]);
    };
    if (tid == 0)
        for (int t = 0; t < kStages && t < nt; t++) load(t);
    float acc[128];   // not zeroed: the first wgmma of the tile overwrites it (scaleD 0)
    const uint32_t s0 = step1::smem_addr(sm);
    for (int t = 0; t < nt; t++) {
        const int s = t % kStages;
        mbar_wait(&full[s], (t / kStages) & 1);
        const uint32_t st = s0 + s * G::kStage;
        const uint64_t da = step1::desc_a(st + wg * 64 * 128), db = step1::desc_b(st + G::kABytes);
        step1::wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < 4; kk++) step1::wgmma<256>(acc, da + 2 * kk, db + 128 * kk, t > 0 || kk > 0);
        step1::wgmma_commit();
        if constexpr (kWait1) step1::wgmma_wait<1>();
        else step1::wgmma_wait<0>();
        constexpr int kLag = kWait1 ? 1 : 0;   // the slice whose group is retired: t - 1 with 2d, t without
        if (t >= kLag) {   // the group of slice p is done: its stage is released, then refilled with slice p + kStages
            const int p = t - kLag, ps = p % kStages;
            if (tid % 128 == 0) mbar_arrive(&empty[ps]);
            if (tid == 0 && p + kStages < nt) {
                mbar_wait(&empty[ps], (p / kStages) & 1);
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
#pragma unroll
    for (int i = 0; i < 16; i++) {   // 16-byte stores, rows >= M left out
        const int x = i * 256 + tid, r = x / 32, gr = row0 + r;
        if (gr < M) *((int4*)&C[(size_t)gr * N + col0] + x % 32) = *((const int4*)&o[r * G::kOPitch] + x % 32);
    }
}

}  // namespace step2
