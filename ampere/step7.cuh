/* Derived from NVIDIA's CUDA sample cudaTensorCoreGemm (cuda-samples v12.4, Samples/3_CUDA_Features/cudaTensorCoreGemm)
 * through steps 0 to 6 (nvidia_sample.cuh, step1.cuh to step6.cuh),
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
#include "step6.cuh"

namespace step7 {

// -------------------------------------------------------- step 7: split-K, parts reduced by their own blocks, L2 hints
// Step 6 takes, shape by shape, a tile and two mechanisms (step6.cuh), K whole in the plan's tiles. Step 7 adds eight
// more, nested on step 6 (step6.cuh, v0), whose kernels and rules stay where none of them applies: a split of K into
// parts (7a), and seven on it or on step 6's kernels (7b to 7h). 7a, 7c, 7e and 7f each have their rule by classes of
// shapes, 7b its condition on the SMs, 7g takes 7b's shapes, and 7d and 7h have no rule of their own; the thresholds
// are our choices, each a mechanical quantity of the shape and the card (sms, the SM count, read at the first call).
// Where a setting documented in cuBLASLt's API bears on a mechanism, it is named with it. The kernels of 7a, 7b and 7e
// are in step6.cuh, beside the plan's.
// (7a) K split into parts, under two of the reduction schemes of cuBLASLt's API (CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE
//      and CUBLASLT_REDUCTION_SCHEME_INPLACE). Parts and reduction: each of S blocks per tile computes its share of the
//      K64 slices and writes its result in fp16 to its own place in a workspace; a second kernel, one block per tile,
//      adds the S parts in the order of K, in fp32 rounded once, and writes C. In place: the S parts of a tile add into
//      C itself, in the order of K; part s waits until the tile's flag reads s (thread 0, an acquire load), adds its
//      result to C (read through L2), stores C, then sets the flag to s + 1, or back to 0 after the last part (a fence,
//      then a release store), so that the next call finds every flag at zero. The parts of a tile are S rows of the
//      grid, the earlier parts at the lower block indices; the code assumes that the GPU starts those first (CUDA does
//      not promise it), so that a part waits only for a block already started.
//      cuBLASLt splits K too at the four shapes where 7a's rule splits it (its configurations kept there for the
//      measured card set CUBLASLT_ALGO_CONFIG_SPLITK_NUM above 1).
// (7b) K in parts, the parts reduced by their own blocks: 7a's parts, then each block of a tile waits until the S parts
//      have arrived and adds them over 1 / S of the tile, in place of 7a's second kernel; taken where the S x tiles
//      blocks fit on the SMs together (they wait on each other; assumed resident together, without a cooperative
//      launch). cuBLASLt's documented reduction schemes add the parts in place, in sequence, or in a separate step.
// (7c) K in parts in place on the 128 x 256 tile: where 7a adds 2 parts of K in place on 6a's 128 x 128 tile, the
//      128 x 256 tile with K in S parts added in place in C one after the other (7a's in-place scheme,
//      CUBLASLT_REDUCTION_SCHEME_INPLACE), S from 2 to 8 with at least 64 slices of K64 each, the S whose blocks end
//      soonest in waves of sms, when those waves are at least 9/10 full.
// (7d) L2 cache hints in the kernels where each slice of B is read by one block or one group in phase (6b, 6c and 7f,
//      7b and 7g): A kept, B evicted first; and, in those with parts of K (6b's stream, 6c and 7f, 7b and 7g, 7e), the
//      parts discarded from L2 once added (step6.cuh's HintCopy and discard_line, behind those kernels' kHint). No
//      rule of its own: it is in those kernels wherever they run.
// (7e) The tail of the waves cut in K: at M > 512, whole tiles of 128 x 256 whose last wave is partial: the last
//      wave's tiles in S parts (2 to 8, at least 64 slices of K64 each), the S that ends the tail soonest, if before
//      one tile's time; the last part to arrive adds them (step6.cuh's gemm_tail and tail_parts). cuBLASLt's split of
//      K is one number per configuration (CUBLASLT_ALGO_CONFIG_SPLITK_NUM).
// (7f) 6c's stream in phase also where 7a adds parts in place and 7c's rule finds no S, by 6c's rule on the waves.
// (7g) 7b without the block's own part: each block keeps in registers the share of its part that it adds itself,
//      writes only the rest, and reads only the other blocks' parts.
// (7h) The reads of an addition issued together: in the parts in place of 7a and 7c, each round of 64 rows of the
//      tile issues every read of C before its additions and stores, the wait on the tile's flag coming once the first
//      round is in shared memory; in 6c's kernel as step 7 runs it (step6.cuh's gemm_phase_sums), the last block of a
//      cut tile reads each part with every read issued together, in the order of K. The reads stay after the
//      acquisition: the wait on the flag and the barrier after it; the arrival and the fence. No rule of its own: it is
//      in those kernels wherever step 7 runs them (behind kTogether).
// The rules of 7a and 7b, on step 6's plan (step6.cuh): at M <= 128 (one row of tiles), K in 3 parts when the tile
// covers fewer than 2/3 of the SMs and its work, BM x BN x K multiply-adds, reaches 2^27 (the 128-row tile then
// 128 x 256); at M > 128 with 6a's tile, K in 2 parts in place when 0 < tiles mod sms < 2 sms / 3 (one tile per SM
// and round) (7a); its parts reduced by their own blocks where the S x tiles blocks fit on the SMs (7b). 6b runs
// first, at M <= 16, and 6c's rule leaves to 7a the shapes it splits; 7f and 7e come next, then the plan.
// The rules give, at the judge's 17 shapes: 7a's 3 parts at o_m128 and down_m128, reduced by their own blocks (7b)
// in 7g's kernel; its 2 parts in place at o_m512 and down_m512, where 7f (o_m512) and 7c (down_m512, 5 parts) take
// over; 7e at down_m2048 (4 parts) and gateup_m2048 (2); 7d in 6b's kernels at the four M = 16, in 6c's at
// gateup_m128 and gateup_m512, in 7f's, in 7g's and in 7e's; 7h in 7c's parts in place at down_m512 and in 6c's
// kernel at gateup_m128, gateup_m512 and o_m512; the other five shapes keep step 6's kernels.
// The counts below are read with nvcc 13.2 (a local compilation); the measured binaries are built by nvcc 13.1.1 (the
// Modal images).
// Compiled (nvcc 13.2, sm_80), no spill. 7a's kernels, main loop per slice and per warp (static counts):
//                             registers    slice   instructions   HMMA   LDSM.x4   LDGSTS   shared memory
//   128 x 256, parts          252          K64     262            128    32        12       147 456 bytes
//   128 x 128, in place       244          K32     164            64     16        8        81 920
//   the reduction of the parts: 32 registers, 256 threads, no shared memory.
// The kernels of 7b, 7c, 7e and 7g, main loop per slice of K64 and per warp (static counts; a kernel with several
// loops, for tiles cut by M and, in 7e, for whole tiles and parts, gives their range):
//                                 registers   instructions   HMMA   LDSM.x4   LDGSTS   barriers
//   7b, 128 x 256 in parts        252         269 to 273     128    32        12       1
//   7c (128 x 256, in place)      252         263 to 269     128    32        12       1
//   7e (128 x 256)                254         263 to 286     128    32        12       1
//   7g, 128 x 256 in parts        252         269 to 273     128    32        12       1
//   7d's hints in the kernels of 6b, 6c, 7b and 7g (registers, instructions; in parentheses, without them): 6b's
//   bands 48, 101 (as without them); its stream 128, 174 (171); 6c 254, 289 to 304 (254, 264 to 288); 7b 252, 269
//   to 273 (250, 266 to 273); 7g 252, 269 to 273 (250, 266 to 273); in 7e's, only the discard of the parts (the
//   same loops).
//   7h in the kernels where it runs (registers, instructions; in parentheses, without it): 7a's parts in place on
//   128 x 128, 248, 168 to 176 (244, 164 to 172); 7c's, 252, 267 to 274 (252, 263 to 269); 6c's with 7d's hints,
//   254, 289 to 316 (254, 289 to 304).
// Variants (g_var in step7.cu, for the interleaved probe; each removal keeps the other mechanisms):
//   v0 = step 7 (7a to 7h); v1 = without 7a (no split of K, and so without 7b, 7c, 7f and 7g); v2 = without 7b (7a's
//   reduction kernel, and so without 7g); v3 = without 7c; v4 = without 7d; v5 = step 6 itself, the anchor; v13 =
//   without 7e; v14 = without 7f; v15 = without 7g; v16 = without 7h; for verif.cu, whose shapes are too small for the
//   rules to choose them: v6 = K in 3 parts at every shape M > 16 where the rules choose none of 6c, 7e and 7f, with 7b
//   and 7g where the parts fit; v7 = 7c's parts in place on 128 x 256 in 2 parts at every shape where 7a adds parts in
//   place and the rules choose neither 6c nor 7f; v8 = 6b's stream at every shape M <= 16; v9 = 6c at every shape M >
//   16 (v8 and v9 with 7d's hints, which step 6's variants run without); v10 = v6 without 7d (7b and 7g without the
//   hints, as in v4); v11 = 7a's parts at every shape, the plan alone (K in 3 on 128 x 256, 7a's reduction kernel);
//   v12 = 7a in place at every shape, the plan alone (K in 2, 128 x 128); v17 = v6 without 7g (7b with 7d's hints, as
//   in v15); v18 = v12 without 7h; v19 = v9 without 7h; v20 = 7e's tail in 2 parts at every shape M > 16 whose 128 x
//   256 tiles leave a partial last wave, where the rules choose none of 6b's kernels, 6c and 7f.

// ---------------------------------------------------------------- 7c: K in parts in place on the 128 x 256 tile
// 7a's in-place scheme (step6::gemm_split with kInPlace: the parts of a tile in the order of K, each waiting for the
// previous on the tile's flag, the parts of a tile S rows of the grid) on step 5's 128 x 256 tile, K in S parts.
// 7c's rule (ours): where 7a adds parts of K in place (6a's 128 x 128 tile, 2 parts), T tiles of 128 x 256
// in S parts, S from 2 to 8 with at least 64 slices of K64 each: the S that minimizes ceil(S T / sms) / S, the time of
// the S T blocks in waves of sms in tiles (the smallest S at a tie), if those waves are at least 9/10 full
// (10 S T >= 9 ceil(S T / sms) sms). Returns S, or 0: no 7c.
inline int inplace_parts(int M, int N, int K, int sms)
{
    const step6::Plan p = step6::choose(M, N, K, sms, true, true);
    if (p.split != step6::kInPlace) return 0;
    const long T = (long)(M + 127) / 128 * (N / 256);
    int best = 0;
    long bestWaves = 0;
    for (int S = 2; S <= 8 && K / 64 / S >= 64; S++) {
        const long waves = (S * T + sms - 1) / sms;
        if (best == 0 || waves * best < bestWaves * S) {
            best = S;
            bestWaves = waves;
        }
    }
    return best && 10 * best * T >= 9 * bestWaves * sms ? best : 0;
}

// ---------------------------------------------------------------- 7f: 6c's stream in phase where 7a adds in place
// 7f's rule (ours): at 16 < M <= 512, where 7a adds parts of K in place and 7c's rule finds no S, 6c's condition on
// the waves of step 6's tiles (at least two waves, the last one holding fewer than half the SMs); the stream in phase
// is 6c's kernel.
inline bool phase_in_place(int M, int N, int K, int sms)
{
    if (M <= 16 || M > 512) return false;
    const step6::Plan p = step6::choose(M, N, K, sms, true, true);
    if (p.split != step6::kInPlace || inplace_parts(M, N, K, sms)) return false;
    const int BN = p.tile == step6::k128x256 ? 256 : 128;
    const long tiles = (long)(M + 127) / 128 * (N / BN), rest = tiles % sms;
    return tiles >= 2L * sms && rest > 0 && 2 * rest < sms;
}

// ---------------------------------------------------------------- 7g: 7b without the block's own part
// 7b's kernel (step6::gemm_parallel) with one change: in 7a's output, thread t writes chunks t + kThreads u of its part
// (u = 0 to U - 1, U = BM x BN / 8 / kThreads); block s of a tile adds chunks t + kThreads u for u from s U / S to
// (s + 1) U / S - 1. Those of its own part it keeps in registers (a keeper passed to step6::compute_tile in the
// flag's place) instead of writing them, and it reads only the other S - 1 parts. The same sums in the same order
// (the S parts in the order of K, each in fp16). The rest is 7b's: the arrival, the wait, the departure.
template <int kU>
struct KeepOwn {
    int u0 = 0, u1 = 0;   // the chunks this block adds: u0 <= u < u1
    int4 mine[kU];
    __device__ __forceinline__ bool keep(int u, int4 v)   // u: a constant of step 6's unrolled output
    {
        mine[u] = v;
        return u >= u0 && u < u1;
    }
};

template <int BM, int BN, int WM, int SK, int kStages, bool kHint>
__global__ void gemm_parallel_own(const half* A, const half* B, half* C, int M, int N, int K, int S, half* parts,
                                  unsigned* counts)
{
    extern __shared__ __align__(128) half shmem[];
    using Copy = typename step6::CopyOf<kHint>::type;
    constexpr int kThreads = step6::Geometry<BM, BN, WM, SK, kStages>::kThreads, kU = BM * BN / 8 / kThreads;
    KeepOwn<kU> own;
    {   // step6::run's part of K, with the keeper in the flag's place
        const int tilesN = N / BN, tiles = (M + BM - 1) / BM * tilesN, s = blockIdx.x / tiles;
        int tm, tn;
        step6::tile_of<BM, BN>(blockIdx.x % tiles, M, N, tm, tn);
        const int t = tm * tilesN + tn, row0 = tm * BM, col0 = tn * BN;
        const int slices = K / SK, per = slices / S, extra = slices % S;
        const int kBegin = s * per + min(s, extra), steps = per + (s < extra);
        half* part = parts + ((size_t)s * tiles + t) * BM * BN;
        own.u0 = s * kU / S;
        own.u1 = (s + 1) * kU / S;
        if (row0 + BM <= M)
            step6::compute_tile<BM, BN, WM, SK, kStages, step6::kParts, false, Copy, KeepOwn<kU>*>(
                shmem, A, B, C, M, N, K, row0, col0, kBegin, steps, part, &own, s, S);
        else
            step6::compute_tile<BM, BN, WM, SK, kStages, step6::kParts, true, Copy, KeepOwn<kU>*>(
                shmem, A, B, C, M, N, K, row0, col0, kBegin, steps, part, &own, s, S);
    }
    const int tilesN = N / BN, tiles = (M + BM - 1) / BM * tilesN;
    const int s = blockIdx.x / tiles;
    int tm, tn;
    step6::tile_of<BM, BN>(blockIdx.x % tiles, M, N, tm, tn);
    const int t = tm * tilesN + tn, row0 = tm * BM, col0 = tn * BN;
    unsigned* arrived = counts + 2 * t;
    unsigned* departed = arrived + 1;
    step6::fence();   // this block's part, before its arrival
    __syncthreads();
    if (threadIdx.x == 0) {
        step6::arrive(arrived);
        step6::wait_count(arrived, (unsigned)S);
    }
    __syncthreads();
    step6::fence();
    constexpr int kBChunks = BN / 8;
    {   // chunks threadIdx.x + kThreads u, u0 <= u < u1; its own part from registers
#pragma unroll
        for (int u = 0; u < kU; u++) {
            const int q = threadIdx.x + kThreads * u, r = q / kBChunks, cc = q % kBChunks;
            if (u < own.u0 || u >= own.u1 || row0 + r >= M) continue;
            float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int ss = 0; ss < S; ss++) {
                const int4 x = ss == s ? own.mine[u]
                                       : step6::ld16_cg((const int4*)&parts[((size_t)ss * tiles + t) * BM * BN +
                                                                            (size_t)r * BN] + cc);
                half h[8];
                memcpy(h, &x, 16);
#pragma unroll
                for (int e = 0; e < 8; e++) v[e] += __half2float(h[e]);
            }
            half o[8];
#pragma unroll
            for (int e = 0; e < 8; e++) o[e] = __float2half(v[e]);
            int4 w;
            memcpy(&w, o, 16);
            step4::st16<true>((int4*)&C[(size_t)(row0 + r) * N + col0] + cc, w);
        }
    }
    __syncthreads();   // every thread's reads of the parts, before the departure (and the discard)
    if constexpr (kHint) {   // 7d with 7g: the lines of this block's slice in the other blocks' parts (in its
                             // own part, that slice was not written)
        const int l0 = own.u0 * kThreads / 8, l1 = own.u1 * kThreads / 8;
        for (int ss = 0; ss < S; ss++)
            if (ss != s)
                for (int l = l0 + threadIdx.x; l < l1; l += blockDim.x)
                    step6::discard_line((const char*)(parts + ((size_t)ss * tiles + t) * BM * BN) + 128 * (size_t)l);
    }
    // The last to depart: both counters back to zero.
    if (threadIdx.x == 0 && step6::arrive(departed) == (unsigned)S - 1) {
        *arrived = 0;
        *departed = 0;
    }
}


}  // namespace step7
