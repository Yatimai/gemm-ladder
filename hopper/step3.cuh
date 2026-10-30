/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through step 0 (nvidia_sample.cuh) and steps 1 and 2: the output, whose notice follows.
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

// ---------------------------------------------------------------- step 3: a fourth stage
// A fourth stage on step 2 (step2.cuh, v0), whose kernel is kept: the 128 x 256 tile, the TMA boxes, the full and
// empty barriers, wgmma m64n256k16 with wait_group 1, the output through shared memory. Shapes: M a multiple of 16, N of
// 256, K of 64.
// (3a) A ring of 4 stages: 3 slices ahead of the MMA (step2::gemm<4>).
// The organization of the block of CUTLASS's cooperative warp-specialized kernel for sm_90 (3b, the roles: a warpgroup
// that loads, two that compute; 3c, the descriptors in the uniform datapath) left step 3 at the test global, then steps
// 4 to 6, and come back at step 7 (7f, 7g: step7.cuh). The tools below serve steps 4 onward: the geometry of 4 stages
// and of the output from step 4, elect.sync and the named barriers from step 5.
// Compiled (nvcc 13.2, sm_90a, without -lineinfo, as the judge): step2::gemm<4>, 154 registers, no spill (step2.cuh).
// Variants (g_var in step3.cu, for the interleaved probe):
//   v0 = step 3 (3a: step2::gemm<4>; 197 696 bytes of dynamic shared memory);
//   v1 = step 2 itself (step2::gemm<3>), the anchor in the same probe.

#ifdef GEMM_LADDER_EMULATE
// A host emulation runs each thread as a coroutine: elect.sync names one lane of the warp; a named barrier suspends
// the thread until `count` threads have reached it.
bool emu_elect();
void emu_bar_named(int id, int count);
#endif

// elect.sync: true in one lane of the warp (all 32 lanes take part).
__device__ __forceinline__ bool elect_one()
{
#ifdef GEMM_LADDER_EMULATE
    return emu_elect();
#else
    uint32_t p;
    asm volatile("{\n.reg .pred e;\nelect.sync _|e, 0xffffffff;\nselp.u32 %0, 1, 0, e;\n}\n" : "=r"(p));
    return p != 0;
#endif
}

// bar.sync id, count: a named barrier for `count` threads (whole warps).
__device__ __forceinline__ void bar_named(int id, int count)
{
#ifdef GEMM_LADDER_EMULATE
    emu_bar_named(id, count);
#else
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory");
#endif
}

template <int kStages>
struct Geometry {
    static constexpr int kABytes = 128 * 128, kBBytes = 64 * 256 * 2, kStage = kABytes + kBBytes;   // 16 + 32 KB
    static constexpr int kOPitch = 256 + 8;   // the output's pitch in halves (132 words, 4 mod 32: no bank conflict)
    static constexpr int kLoopBytes = kStages * kStage, kOutBytes = 128 * kOPitch * 2;
    static constexpr int kBytes = kLoopBytes > kOutBytes ? kLoopBytes : kOutBytes;
    static constexpr int kBars = kBytes;               // the barriers after the stages: full[kStages], empty[kStages]
    static constexpr int kSmemBytes = kBytes + 16 * kStages + 1024;   // + the alignment of the stages to 1024 bytes
    static_assert(kSmemBytes <= 227 * 1024, "227 KB of shared memory at most");
};

}  // namespace step3
