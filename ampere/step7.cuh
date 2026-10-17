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

// ---------------------------------------------------------------- step 7: the refinements
// Step 6 takes, shape by shape, a tile and two mechanisms (step6.cuh), K whole in the plan's tiles. Step 7 adds four
// more, nested on step 6 (step6.cuh, v0), whose kernels and rules stay where none of them applies: a split of K into
// parts (7a), and three on it or on step 6's kernels (7b, 7c and 7d). 7a and 7c each have their rule by classes of
// shapes, 7b its condition on the SMs, and 7d has no rule of its own; the thresholds are our choices, each a mechanical
// quantity of the shape and the card (sms, the SM count, read at the first call). Where a setting documented in
// cuBLASLt's API bears on a mechanism, it is named with it. The kernels of 7a and 7b are in step6.cuh, beside the
// plan's.
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
// (7d) L2 cache hints in the kernels where each slice of B is read by one block or one group in phase (6b, 6c and 7b):
//      A kept, B evicted first; and, in those with parts of K (6b's stream, 6c and 7b), the parts discarded from L2
//      once added (step6.cuh's HintCopy and discard_line, behind those kernels' kHint). No rule of its own: it is in
//      those kernels wherever they run.
// The rules of 7a and 7b, on step 6's plan (step6.cuh): at M <= 128 (one row of tiles), K in 3 parts when the tile
// covers fewer than 2/3 of the SMs and its work, BM x BN x K multiply-adds, reaches 2^27 (the 128-row tile then
// 128 x 256); at M > 128 with 6a's tile, K in 2 parts in place when 0 < tiles mod sms < 2 sms / 3 (one tile per SM
// and round) (7a); its parts reduced by their own blocks where the S x tiles blocks fit on the SMs (7b). 6b runs
// first, at M <= 16, and 6c's rule leaves to 7a the shapes it splits.
// The rules give, at the judge's 17 shapes: 7a's 3 parts at o_m128 and down_m128, reduced by their own blocks (7b); its
// 2 parts in place at o_m512 and down_m512, where 7c takes over at down_m512 (5 parts); 7d in 6b's kernels at the four
// M = 16, in 6c's at gateup_m128 and gateup_m512, and in 7b's at o_m128 and down_m128; the other seven shapes keep
// step 6's kernels.
// Compiled (nvcc 13.2, sm_80), no spill. 7a's kernels, main loop per slice and per warp (static counts):
//                             registers    slice   instructions   HMMA   LDSM.x4   LDGSTS   shared memory
//   128 x 256, parts          252          K64     262            128    32        12       147 456 bytes
//   128 x 128, in place       244          K32     164            64     16        8        81 920
//   the reduction of the parts: 32 registers, 256 threads, no shared memory.
// The kernels of 7b and 7c, main loop per slice of K64 and per warp (static counts; a kernel with several loops, for
// tiles cut by M, gives their range):
//                                 registers   instructions   HMMA   LDSM.x4   LDGSTS   barriers
//   7b, 128 x 256 in parts        252         269 to 273     128    32        12       1
//   7c (128 x 256, in place)      252         263 to 269     128    32        12       1
//   7d's hints in the kernels of 6b, 6c and 7b (registers, instructions; in parentheses, without them): 6b's
//   bands 48, 101 (as without them); its stream 128, 174 (171); 6c 254, 289 to 304 (254, 264 to 288);
//   7b 252, 269 to 273 (250, 266 to 273).
// Variants (g_var in step7.cu, for the interleaved probe; each removal keeps the other mechanisms):
//   v0 = step 7 (7a, 7b, 7c and 7d); v1 = without 7a (no split of K, and so without 7b and 7c); v2 = without 7b (7a's
//   reduction kernel); v3 = without 7c; v4 = without 7d; v5 = step 6 itself, the anchor; for verif.cu, whose shapes are
//   too small for the rules to choose them: v6 = K in 3 parts at every shape M > 16 where the rules do not choose 6c,
//   with 7b where the parts fit; v7 = 7c's parts in place on 128 x 256 in 2 parts at every shape where 7a adds parts in
//   place and the rules do not choose 6c; v8 = 6b's stream at every shape M <= 16; v9 = 6c at every shape M > 16 (v8
//   and v9 with 7d's hints, which step 6's variants run without); v10 = v6 without 7d (7b without the hints, as in v4);
//   v11 = 7a's parts at every shape, the plan alone (K in 3 on 128 x 256, 7a's reduction kernel); v12 = 7a in place at
//   every shape, the plan alone (K in 2, 128 x 128).

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

}  // namespace step7
