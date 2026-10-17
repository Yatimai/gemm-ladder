/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through steps 0 to 2 (nvidia_sample.cuh, step1.cuh, step2.cuh),
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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "step2.cuh"

namespace step3 {

// ---------------------------------------------------------------- step 3: Ampere's asynchronous copy
// Ampere's asynchronous copy, new against the Turing rung, nested on step 2 (step2.cuh, v0), whose rest is kept: the
// 128 x 256 tile, the 8 warps of 64 x 64, step 2's swizzled stage of K64 (48 KB), its compute (ldmatrix.x4, mma.sync
// m16n8k16 in the same order: the same C to the bit, which the emulator checks), its epilogue. Only the copy changes.
// (3a) cp.async.cg.shared.global.L2::128B: each 16-byte chunk goes from global to shared memory without a register (one
//      LDGSTS.E.BYPASS.LTC128B.128 in place of an LDG.E.128 and an STS.128), through L2 only (.cg: not kept in L1),
//      with the hint that L2 may fetch the whole 128-byte line. One stage, step 2's loop: issue the slice, commit,
//      cp.async.wait_group 0, barrier, compute, barrier.
// (3b) Stages: from 1 to 2 to 3 stages of K64 (3 x 48 KB = 144 KB), kStages - 1 slices in flight while one is computed.
//      Per K64, one wait (cp.async.wait_group kStages - 2) and one barrier, which both makes every thread's copies of
//      slice k visible and frees the stage of slice k - 1 for the next issue. The stage indices turn at run time.
//      Nested: 2 stages is half of 3b.
// cp.async.wait_group orders a thread's own copies only; the barrier after it publishes them to the block. The host
// emulator (emu/) performs each copy as late as the program allows (at its wait) in one order of the threads, as early
// as possible (when issued) in another, and either way at random in the third: a missing wait or barrier, or a copy
// into a stage still being read, gives a wrong C (emu/planted_bugs.py plants them).
// Compiled (nvcc 13.2, sm_80): 3 stages at 236 registers, 2 at 228, 3a at 218 (step 2: 224), no spill; one block per SM
// (by the registers; 2 and 3 stages also by their shared memory). Main loop, per K64 and per warp (static counts; in
// 3b one forward branch skips the issue for the last kStages - 1 slices):
//                         HMMA   LDSM.x4   LDGSTS   LDG + STS   BAR   commit + wait   integer   total
//   step 2 (v3)           128    32        0        12 + 12     2     0               42        229
//   3a, 1 stage (v2)      128    32        12       0           2     2               30        213
//   2 stages (v1)         128    32        12       0           1     2               60        243
//   3 stages (v0)         128    32        12       0           1     2               71        256
// The total also counts the instructions that no column holds.
// 3b's extra integer instructions turn the stage indices and move the addresses with them (stage offsets of the
// ldmatrix and cp.async addresses, the 12 source pointers carried in 64 bits).
// Variants (g_var in step3.cu, for the interleaved probe; nested):
//   v0 = step 3 (3 stages; 147 456 bytes of dynamic shared memory);
//   v1 = 2 stages (98 304 bytes);
//   v2 = 3a alone (1 stage; 49 152 bytes);
//   v3 = step 2 itself (step2::gemm<true>; 49 152 bytes), the anchor in the same probe: v2 against v3 measures 3a, v1
//        against v2 and v0 against v1 the stages.

#ifdef GEMM_LADDER_EMULATE
// A host emulation defers each copy until its group is waited for (a copy never waited for leaves its 16 bytes
// unwritten), or performs it at once (see emu/).
void emu_cp_async(void* smem, const void* gmem);
void emu_cp_async_commit();
void emu_cp_async_wait(int pending);
#endif

// One cp.async.cg.shared.global.L2::128B of 16 bytes (one LDGSTS.E.BYPASS.LTC128B.128): global memory to shared memory
// without a register, through L2 only (.cg), with a hint that L2 may fetch the whole 128-byte line (.L2::128B).
__device__ __forceinline__ void cp_async16(half* smem, const half* gmem)
{
#ifdef GEMM_LADDER_EMULATE
    emu_cp_async(smem, gmem);
#else
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" ::"r"(s), "l"(gmem) : "memory");
#endif
}

// cp.async.commit_group: the copies issued since the last commit form a group (possibly empty).
__device__ __forceinline__ void cp_async_commit()
{
#ifdef GEMM_LADDER_EMULATE
    emu_cp_async_commit();
#else
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

// cp.async.wait_group N: waits until at most N of this thread's groups are still in flight (the newest ones). It orders
// this thread's copies only; a barrier then makes every thread's copies visible to the block.
template <int N>
__device__ __forceinline__ void cp_async_wait()
{
#ifdef GEMM_LADDER_EMULATE
    emu_cp_async_wait(N);
#else
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
#endif
}

using G = step2::Geometry<true>;   // step 2's swizzled stage: A 128 rows of K64, B 64 rows of 256 columns

// The shared memory of kStages stages: A's stages first, then B's; the output's half tile reuses them.
template <int kStages>
struct Smem {
    static constexpr int kAAll = kStages * G::kAStage;   // halves
    static constexpr int kLoopBytes = kStages * (G::kAStage + G::kBStage) * 2, kOutBytes = 64 * G::kOStride * 2;
    static constexpr int kBytes = kLoopBytes > kOutBytes ? kLoopBytes : kOutBytes;
    static_assert(kBytes <= 163 * 1024, "above the 163 KB of shared memory of a block on sm_80");
};

// One 128 x 256 tile; kEdge: the tile is cut by M (the row of A each thread copies is clamped to M - 1).
template <int kStages, bool kEdge>
__device__ __forceinline__ void compute_tile(half* shmem, const half* A, const half* B, half* C, int M, int N, int K,
                                             int row0, int col0)
{
    half* As = shmem;                         // kStages x 128 rows of a K64 slice of A
    half* Bs = shmem + Smem<kStages>::kAAll;  // kStages x 64 rows of a K64 slice of B
    const int warpId = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warpId / 4, wn = warpId % 4;   // the warp's rows wm * 64, columns wn * 64
    float c[4][4][2][4];   // as step 2
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 4; e++) c[i][j][n][e] = 0.0f;
    int aOff[4], bOff[4];   // step 2's ldmatrix offsets, within a stage
#pragma unroll
    for (int ks = 0; ks < 4; ks++) aOff[ks] = G::a_at(wm * 64 + lane % 16, 2 * ks + lane / 16);
#pragma unroll
    for (int j = 0; j < 4; j++) bOff[j] = G::b_at(lane % 16, wn * 8 + 2 * j + lane / 16);

    // The copy of the slice that starts at k0 into stage s, as step 2's (thread t: rows t / 8 + 32 i of A at chunk
    // t % 8, rows t / 32 + 8 i of B at chunk t % 32, at their swizzled places), each chunk one cp.async.
    auto issue = [&](int k0, int s) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int r = threadIdx.x / 8 + i * 32;
            const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;
            cp_async16(&As[s * G::kAStage + G::a_at(r, threadIdx.x % 8)],
                       &A[(size_t)gr * K + k0 + 8 * (threadIdx.x % 8)]);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int r = threadIdx.x / 32 + i * 8;
            cp_async16(&Bs[s * G::kBStage + G::b_at(r, threadIdx.x % 32)],
                       &B[(size_t)(k0 + r) * N + col0 + 8 * (threadIdx.x % 32)]);
        }
    };
    auto compute = [&](int s) {   // step 2's compute over stage s
        const half* as = As + s * G::kAStage;
        const half* bs = Bs + s * G::kBStage;
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            uint32_t a[4][4], b[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                step2::ldmatrix_x4<false>(a[i], &as[aOff[ks] + i * 16 * G::kAStride]);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (i == 0) step2::ldmatrix_x4<true>(b[j], &bs[bOff[j] + ks * 16 * G::kBStride]);
#pragma unroll
                    for (int n = 0; n < 2; n++) step2::mma_m16n8k16(c[i][j][n], a[i], b[j][2 * n], b[j][2 * n + 1]);
                }
            }
        }
    };
    const int steps = K / 64;
    if constexpr (kStages == 1) {
        // (3a): step 2's loop, the copy through cp.async: copy, wait for it, barrier, compute, barrier.
        for (int k = 0; k < steps; k++) {
            issue(k * 64, 0);
            cp_async_commit();
            cp_async_wait<0>();
            __syncthreads();
            compute(0);
            __syncthreads();
        }
    } else {
        // (3b): kStages - 1 slices in flight. The prologue issues slices 0 to kStages - 2, one group each (empty
        // outside K). Per K64: wait until slice k has landed (at most kStages - 2 younger groups in flight), one
        // barrier (every thread's copies of slice k are visible, and every warp is done with the stage of slice k - 1),
        // then issue slice k + kStages - 1 into that stage and compute slice k.
#pragma unroll
        for (int s = 0; s < kStages - 1; s++) {
            if (s < steps) issue(s * 64, s);
            cp_async_commit();
        }
        int read = 0, write = kStages - 1;   // the stages of slices k and k + kStages - 1
        for (int k = 0; k < steps; k++) {
            cp_async_wait<kStages - 2>();
            __syncthreads();
            if (k + kStages - 1 < steps) issue((k + kStages - 1) * 64, write);
            cp_async_commit();
            compute(read);
            read = read + 1 == kStages ? 0 : read + 1;
            write = write + 1 == kStages ? 0 : write + 1;
        }
        cp_async_wait<0>();   // only empty groups are left
        __syncthreads();      // every warp is done with the last stage before the output reuses it
    }
    // The output, half a tile at a time, as step 2.
    half* Os = shmem;
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        if (wm == h) {
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
template <int kStages>
__global__ void gemm(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(128) half shmem[];
    const int tilesN = N / 256;
    const int row0 = (blockIdx.x / tilesN) * 128, col0 = (blockIdx.x % tilesN) * 256;
    if (row0 + 128 <= M) compute_tile<kStages, false>(shmem, A, B, C, M, N, K, row0, col0);
    else compute_tile<kStages, true>(shmem, A, B, C, M, N, K, row0, col0);
}

}  // namespace step3
