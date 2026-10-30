/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through step 0 (nvidia_sample.cuh): the copy through registers and the output, whose notice follows.
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

namespace step1 {

// --------------------------------------------------------------- step 1: wgmma, a 128 x 256 tile, every thread copying
// The H100's warpgroup MMA, written from the PTX ISA (wgmma.mma_async; CuTe's Hopper tutorial on wgmma shows the same
// engine), in place of step 0's wmma, which compiles to the per-warp HMMA of sm_80 on this card:
// (1a) wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16: the 128 threads of a warpgroup (4 warps) issue together one
//      MMA of 64 rows x 256 columns x K16, A and B read in shared memory through two 64-bit descriptors (address,
//      strides, swizzle), the fp32 accumulator in registers (128 per thread). One block of two warpgroups per tile of
//      128 x 256 (warpgroup h computes rows 64 h to 64 h + 63), one block per tile (step 0's persistent grid is left;
//      it comes back with the persistence). The shared stage keeps the layout the descriptors read, 128-byte swizzle:
//      A K-major, 128 rows of K64 (128 bytes each), chunk c (16 bytes) of row r at chunk c ^ (r % 8) of its row; B
//      MN-major (the instruction's transposed B: our B is row-major, its N contiguous), kBN / 64 boxes of 64 columns
//      x 64 rows of K (8 KB each), chunk c of K-row k of a box at chunk c ^ (k % 8). Per K64: 4 wgmma per warpgroup
//      (the descriptors advance by 32 bytes in A and by 2 KB in B per K16).
//      The chain of a slice, as the tutorial's: wgmma.fence, the 4 wgmma, wgmma.commit_group, then wgmma.wait_group 0
//      (the group of slice t is done before slice t + 1 is stored). Per K64: the loads of slice t + 1 are issued once
//      group t is, then wait_group 0, a barrier (both warpgroups are done with the stage), the stores of slice t + 1
//      into the other of two stages and fence.proxy.async (the MMA reads shared memory through the async proxy: the
//      threads' stores must be made visible to it), a barrier, the next group. 2d (wait_group 1, the group of slice t
//      kept in flight while slice t + 1 is stored) left step 1 at the test global: it enters step 2 (step2.cuh).
//      Without it, two waits are redundant, kept as measured: the first barrier of slice t (both warpgroups retired
//      group t - 1 before the second barrier of slice t - 1) and the wait_group 0 after the loop (the last slice's
//      wait_group 0 retired the last group).
// (1b) The spread of the copy: each of the 256 threads copies its share of the slice of A and of B (4 + 8 chunks per
//      K64 at 128 x 256), where step 0 gives A to warps 0-3 and B to warps 4-7, and issues all its loads before its
//      stores: the loads of slice t + 1 once group t is issued, its stores after the barrier that follows the wait.
//      In the SASS (nvcc 13.2), v0's loop issues its 12 loads in a row, then its 12 stores; without 1b (v3, step 0's
//      split, each chunk loaded and stored in one statement once the barrier has freed the stage), warps 4 to 7 copy
//      B in five dependent waves (3, 3, 3, 3 and 4 loads, each wave's stores before the next wave's loads) and warps
//      0 to 3 copy A in three (2, 3 and 3).
// Kept from step 0: the copy through registers in 16-byte global loads, ordinary global loads, the edge in M (a tile
// cut by M clamps the row of A each thread loads to M - 1; rows >= M are computed and never written), the fp16 output
// through shared memory (here the whole 128 x 256 tile, at a pitch of 264 halves: no bank conflict), then 16-byte
// stores. Shapes: M a multiple of 16, N of kBN, K of 64.
// Compiled (nvcc 13.2, sm_90a): v0 at 254 registers, v1 at 151, no spill; v3 at 255 with a spill (80 bytes of stack;
// in its loop, 2 to 16 LDL per K64 and no STL); one block per SM. Main loop, per K64 and per warp (static counts; v3
// in the tiles not cut by M, 167 and 331 in those cut by M):
//                        HGMMA            LDG.128   STS.128   BAR   fence   warpgroup   other   total
//   v0 (128 x 256)       4 x 64x256x16    12        12        2     2       2           169     203
//   v1 (128 x 128)       4 x 64x128x16    8         8         2     2       2           112     138
//   v3, warps 0-3 (A)    4 x 64x256x16    8         8         2     2       2           133     159
//   v3, warps 4-7 (B)    4 x 64x256x16    16        16        2     2       2           306     348
// (fence: MEMBAR.ALL.CTA and FENCE.VIEW.ASYNC.S, from fence.proxy.async; warpgroup: WARPGROUP.ARRIVE, from
// wgmma.fence, and WARPGROUP.DEPBAR.LE, from wait_group; other: the address arithmetic of the copy, R2UR, the loop
// and v3's LDL.)
// Step 0's loop, for a tile of 128 x 128: 175 instructions per K64 and per warp, 64 of them HMMA.16816.
// Variants (g_var in step1.cu, for the interleaved probe):
//   v0 = step 1 (1a + 1b; 99 328 bytes of dynamic shared memory);
//   v1 = a tile of 128 x 128 (wgmma m64n128k16, 64 accumulators per thread; 66 560 bytes);
//   v2 = step 0 itself (nvidia_sample::compute_gemm), the anchor in the same probe;
//   v3 = without 1b: step 0's split of the copy (gemm_split; 99 328 bytes).

#ifdef GEMM_LADDER_EMULATE
// A host emulation runs each thread as a coroutine. emu_wgmma is the warpgroup's meeting point: once its 128 threads
// have called it with the same operands, the MMA is issued, and performed when the emulator decides (when issued, or
// when a wait_group retires its group); the accumulators read meanwhile are NaN. emu_fence_proxy_async makes the
// running thread's shared stores visible to the async proxy; a wgmma that reads a byte stored since its writer's last
// fence is an error (see emu/).
void emu_wgmma(float* d, int n, uint64_t da, uint64_t db, int scaleD);
void emu_wgmma_fence();
void emu_wgmma_commit();
void emu_wgmma_wait(int pending);
void emu_fence_proxy_async();
uint32_t emu_smem_addr(const void* p);
#endif

// The address of p in the shared window (cvta.to.shared), as the descriptors take it.
__device__ __forceinline__ uint32_t smem_addr(const void* p)
{
#ifdef GEMM_LADDER_EMULATE
    return emu_smem_addr(p);
#else
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
#endif
}

// The matrix descriptors of the PTX ISA (bits 0-13: address / 16; 16-29: leading byte offset / 16; 32-45: stride byte
// offset / 16; 62-63: swizzle, 1 = 128 bytes), for the layouts of the stage:
// A, K-major, 128-byte swizzle: rows of 128 bytes, atoms of 8 rows (1024 bytes) one after the other (stride 1024); the
//   leading offset is not used (the 16 K of one MMA lie in one row) and set to 1.
__device__ __forceinline__ uint64_t desc_a(uint32_t addr)
{
    return ((uint64_t)(addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
// B, MN-major, 128-byte swizzle: atoms of 64 columns (128 bytes) x 8 rows of K (1024 bytes); leading offset 8 KB, from
//   one box of 64 columns to the next; stride 1024, from 8 rows of K to the next 8.
__device__ __forceinline__ uint64_t desc_b(uint32_t addr)
{
    return ((uint64_t)(addr & 0x3FFFF) >> 4) | ((uint64_t)(8192 >> 4) << 16) | ((uint64_t)(1024 >> 4) << 32) |
           ((uint64_t)1 << 62);
}

// wgmma.fence: orders this warpgroup's register accesses before the MMA that follow (needed before the first wgmma
// and whenever the accumulators were touched by other instructions).
__device__ __forceinline__ void wgmma_fence()
{
#ifdef GEMM_LADDER_EMULATE
    emu_wgmma_fence();
#else
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}

// wgmma.commit_group: the wgmma issued since the last commit form a group.
__device__ __forceinline__ void wgmma_commit()
{
#ifdef GEMM_LADDER_EMULATE
    emu_wgmma_commit();
#else
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}

// wgmma.wait_group N: waits until at most N of this warpgroup's groups are in flight; their accumulators may be read,
// and the shared memory they read written, only then.
template <int N>
__device__ __forceinline__ void wgmma_wait()
{
#ifdef GEMM_LADDER_EMULATE
    emu_wgmma_wait(N);
#else
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
#endif
}

// fence.proxy.async.shared::cta: the running thread's earlier shared stores become visible to the async proxy.
__device__ __forceinline__ void fence_proxy_async()
{
#ifdef GEMM_LADDER_EMULATE
    emu_fence_proxy_async();
#else
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}

// One wgmma.mma_async m64nNk16, fp32 accumulators, A and B from shared memory (A K-major, B transposed: MN-major). The
// accumulator of lane l of warp w of the warpgroup (g = l / 4, t = l % 4), for j = 0 to N / 8 - 1: d[4 j] and
// d[4 j + 1] = D[16 w + g][8 j + 2 t] and [8 j + 2 t + 1], d[4 j + 2] and d[4 j + 3] the same in row 16 w + g + 8.
// scaleD 0: D = A B; 1: D += A B.
template <int N>
__device__ __forceinline__ void wgmma(float (&d)[N / 2], uint64_t da, uint64_t db, int scaleD);

template <>
__device__ __forceinline__ void wgmma<256>(float (&d)[128], uint64_t da, uint64_t db, int scaleD)
{
#ifdef GEMM_LADDER_EMULATE
    emu_wgmma(d, 256, da, db, scaleD);
#else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %130, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 0, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
                 : "l"(da), "l"(db), "r"(scaleD));
#endif
}

template <>
__device__ __forceinline__ void wgmma<128>(float (&d)[64], uint64_t da, uint64_t db, int scaleD)
{
#ifdef GEMM_LADDER_EMULATE
    emu_wgmma(d, 128, da, db, scaleD);
#else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "l"(da), "l"(db), "r"(scaleD));
#endif
}

// The geometry of a tile of 128 x kBN: two stages of A (16 KB) and B (kBN x 128 bytes), the output tile after the loop.
template <int kBN>
struct Geometry {
    static constexpr int kABytes = 128 * 128;             // 128 rows of K64
    static constexpr int kBBytes = 64 * kBN * 2;          // kBN / 64 boxes of 64 rows of K x 64 columns
    static constexpr int kStage = kABytes + kBBytes;
    static constexpr int kOPitch = kBN + 8;               // halves: 132 or 68 words a row, 4 mod 32: no bank conflict
    static constexpr int kLoopBytes = 2 * kStage, kOutBytes = 128 * kOPitch * 2;
    static constexpr int kBytes = kLoopBytes > kOutBytes ? kLoopBytes : kOutBytes;
    static constexpr int kSmemBytes = kBytes + 1024;      // the stages aligned to 1024 bytes (the swizzle's atoms)
    static constexpr int kALoads = 128 * 8 / 256, kBChunks = kBN / 8, kBLoads = 64 * kBChunks / 256;
    static_assert(kSmemBytes <= 227 * 1024, "above the 227 KB of shared memory of a block on sm_90");
};

// One 128 x kBN tile; kSpread: the spread of the copy (1b), else step 0's split; kEdge: the tile is cut by M.
template <int kBN, bool kSpread, bool kEdge>
__device__ __forceinline__ void tile(uint8_t* sm, const half* A, const half* B, half* C, int M, int N, int K, int row0,
                                     int col0)
{
    using G = Geometry<kBN>;
    const int tid = threadIdx.x, wg = tid / 128;
    int4 ra[G::kALoads], rb[G::kBLoads];
    auto load = [&](int k0) {   // slice k0 of A and B into registers: chunk i of A is row i / 8, chunk i % 8
#pragma unroll
        for (int i = 0; i < G::kALoads; i++) {
            const int x = i * 256 + tid, r = x / 8, gr = kEdge ? min(row0 + r, M - 1) : row0 + r;
            ra[i] = *((const int4*)&A[(size_t)gr * K + k0] + x % 8);
        }
#pragma unroll
        for (int i = 0; i < G::kBLoads; i++) {
            const int x = i * 256 + tid, r = x / G::kBChunks;
            rb[i] = *((const int4*)&B[(size_t)(k0 + r) * N + col0] + x % G::kBChunks);
        }
    };
    auto store = [&](int s) {   // the registers into stage s, swizzled, made visible to the async proxy
        uint8_t* sa = sm + s * G::kStage;
        uint8_t* sb = sa + G::kABytes;
#pragma unroll
        for (int i = 0; i < G::kALoads; i++) {
            const int x = i * 256 + tid, r = x / 8, c = x % 8;
            *(int4*)(sa + r * 128 + ((c ^ (r % 8)) << 4)) = ra[i];
        }
#pragma unroll
        for (int i = 0; i < G::kBLoads; i++) {
            const int x = i * 256 + tid, r = x / G::kBChunks, c = x % G::kBChunks;
            *(int4*)(sb + (c / 8) * 8192 + r * 128 + (((c % 8) ^ (r % 8)) << 4)) = rb[i];
        }
        fence_proxy_async();
    };
    auto copy = [&](int k0, int s) {   // without 1b, step 0's split: warps 0 to 3 copy A, warps 4 to 7 copy B, each
                                       // chunk loaded and stored in one statement, into stage s, swizzled
        uint8_t* sa = sm + s * G::kStage;
        uint8_t* sb = sa + G::kABytes;
        const int u = tid % 128;
        if (wg == 0) {
#pragma unroll
            for (int i = 0; i < 2 * G::kALoads; i++) {
                const int x = i * 128 + u, r = x / 8, c = x % 8, ar = kEdge ? min(row0 + r, M - 1) : row0 + r;
                *(int4*)(sa + r * 128 + ((c ^ (r % 8)) << 4)) = *((const int4*)&A[(size_t)ar * K + k0] + c);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 2 * G::kBLoads; i++) {
                const int x = i * 128 + u, r = x / G::kBChunks, c = x % G::kBChunks;
                *(int4*)(sb + (c / 8) * 8192 + r * 128 + (((c % 8) ^ (r % 8)) << 4)) =
                    *((const int4*)&B[(size_t)(k0 + r) * N + col0] + c);
            }
        }
        fence_proxy_async();   // the thread's stores, visible to the async proxy
    };
    float acc[kBN / 2];   // not zeroed: the first wgmma of the tile overwrites it (scaleD 0)
    const uint32_t s0 = smem_addr(sm);
    const int nt = K / 64;
    if constexpr (kSpread) {
        load(0);
        store(0);
    } else {
        copy(0, 0);
    }
    __syncthreads();
    for (int t = 0; t < nt; t++) {
        const uint32_t st = s0 + (t % 2) * G::kStage;
        const uint64_t da = desc_a(st + wg * 64 * 128), db = desc_b(st + G::kABytes);
        wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < 4; kk++) wgmma<kBN>(acc, da + 2 * kk, db + 128 * kk, t > 0 || kk > 0);
        wgmma_commit();
        if (kSpread && t + 1 < nt) load((t + 1) * 64);
        wgmma_wait<0>();
        if (t + 1 < nt) {
            __syncthreads();   // redundant without 2d: group t - 1 retired before the last barrier (kept as measured)
            if constexpr (kSpread) store((t + 1) % 2);
            else copy((t + 1) * 64, (t + 1) % 2);
            __syncthreads();   // slice t + 1 is in place
        }
    }
    wgmma_wait<0>();   // redundant without 2d: the last slice's wait_group 0 retired every group (kept as measured)
    __syncthreads();   // both warpgroups are done with the stages: the output tile overwrites them
    half* o = (half*)sm;
    const int w = (tid / 32) % 4, l = tid % 32, g = l / 4, q = l % 4, r0 = wg * 64 + w * 16 + g;
#pragma unroll
    for (int j = 0; j < kBN / 8; j++) {
        *(__half2*)&o[r0 * G::kOPitch + 8 * j + 2 * q] = __floats2half2_rn(acc[4 * j], acc[4 * j + 1]);
        *(__half2*)&o[(r0 + 8) * G::kOPitch + 8 * j + 2 * q] = __floats2half2_rn(acc[4 * j + 2], acc[4 * j + 3]);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 128 * G::kBChunks / 256; i++) {   // 16-byte stores, rows >= M left out
        const int x = i * 256 + tid, r = x / G::kBChunks, gr = row0 + r;
        if (gr < M) *((int4*)&C[(size_t)gr * N + col0] + x % G::kBChunks) = *((const int4*)&o[r * G::kOPitch] + x % G::kBChunks);
    }
}

// One block of 256 threads (two warpgroups) per 128 x kBN tile, tiles row by row (blockIdx.x).
template <int kBN, bool kSpread>
__device__ __forceinline__ void gemm_body(const half* A, const half* B, half* C, int M, int N, int K)
{
    extern __shared__ __align__(1024) uint8_t shmem[];
    uint8_t* sm = shmem + ((1024 - (smem_addr(shmem) & 1023)) & 1023);
    const int tilesN = N / kBN, row0 = (blockIdx.x / tilesN) * 128, col0 = (blockIdx.x % tilesN) * kBN;
    if (row0 + 128 <= M) tile<kBN, kSpread, false>(sm, A, B, C, M, N, K, row0, col0);
    else tile<kBN, kSpread, true>(sm, A, B, C, M, N, K, row0, col0);
}

// v0 and v1: with 1b.
template <int kBN>
__global__ void __launch_bounds__(256, 1) gemm(const half* A, const half* B, half* C, int M, int N, int K)
{
    gemm_body<kBN, true>(A, B, C, M, N, K);
}

// v3: without 1b (step 0's split of the copy).
template <int kBN>
__global__ void __launch_bounds__(256, 1) gemm_split(const half* A, const half* B, half* C, int M, int N, int K)
{
    gemm_body<kBN, false>(A, B, C, M, N, K);
}

}  // namespace step1
