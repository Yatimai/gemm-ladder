#!/usr/bin/env python3
"""The planted-bug test of the Hopper rung's host emulator (README.md in this directory).

Each planted bug is one exact text substitution in a copy of a kernel header (nvidia_sample.cuh to step7.cuh). The
emulator, built against that copy, runs the step the header belongs to and must report the bug as a failure: exit code 1
and a FAILED run. The unmodified copy, built the same way, must pass every run (./emu all). A substitution whose text is
not found exactly once in its header stops the test before anything is built: the source has changed, and the
substitution must be brought up to date with it. The script grows with the rung, as the emulator does. Step 0's code is
NVIDIA's sample: its bugs check the emulator's engines at step 0 (barriers, accesses, edges, the persistent grid, the
writes of C, wmma's rules).

    python3 hopper/emu/planted_bugs.py [-j JOBS] [--keep]

-j: builds and runs in parallel (default: the number of CPUs, at most 8); --keep: keep the temporary directory. The
headers in hopper/ are only read. Exit code 0 only if the unmodified copy passes and every planted bug is reported as a
failure.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

EMU = os.path.dirname(os.path.abspath(__file__))   # hopper/emu
RUNG = os.path.dirname(EMU)                         # hopper
HEADERS = ("nvidia_sample.cuh", "step1.cuh", "step2.cuh", "step3.cuh", "step4.cuh", "step5.cuh", "step6.cuh",
           "step7.cuh")
STEP = {"nvidia_sample.cuh": "step0", "step1.cuh": "step1", "step2.cuh": "step2", "step3.cuh": "step3",
        "step4.cuh": "step4", "step5.cuh": "step5", "step6.cuh": "step6", "step7.cuh": "step7"}
CXX = os.environ.get("CXX", "g++")
RUN_BY = {
          # step 7 runs them: the 128 x 320 tile without warp specialization (step6::gemm, step 7's v8) and 7e's rule
          # (step6::choose with 7e), which step 6 does not run
          "the rule: 128 x 320 against 128 x 256 by their waves alone (its area forgotten)": "step7",
          "the rule: 7e in pairs in one wave too (its class forgotten)": "step7",
          "the rule (7e's branch, step 7): M > 128 back to step 5's tile in pairs where 6c runs": "step7",
          "the tiles, 7e: the n128 of 128 x 320 on B's first boxes": "step7",
          "the tiles, 7e: the n128 of 128 x 320 on accumulators that overlap the n192's": "step7",
          "the tiles, 7e: the output's passes after the second forgotten (128 x 320's last "
          "64 columns)": "step7"}

# (header, the bug, exact text, its replacement).
BUGS = [
    ("nvidia_sample.cuh", "no barrier after the copy of a slice (the MMA reads rows other warps have not stored)",
     "            }\n            __syncthreads();\n#pragma unroll\n            for (int ks = 0; ks < kChunkK; ks++) {",
     "            }\n#pragma unroll\n            for (int ks = 0; ks < kChunkK; ks++) {"),
    ("nvidia_sample.cuh", "no barrier after the MMA of a slice (the next copy overwrites rows still being read)",
     "                }\n            }\n            __syncthreads();\n        }\n",
     "                }\n            }\n        }\n"),
    ("nvidia_sample.cuh", "no barrier between the epilogue's fragment stores and its copy out",
     "                    }\n            }\n            __syncthreads();\n#pragma unroll\n            for (int i = 0; i < 4; i++) {",
     "                    }\n            }\n#pragma unroll\n            for (int i = 0; i < 4; i++) {"),
    ("nvidia_sample.cuh", "no barrier after the epilogue's copy out (the next half tile, or tile, overwrites it)",
     "= *((const int4*)&Os[r * kOStride] + threadIdx.x % 16);\n            }\n            __syncthreads();\n",
     "= *((const int4*)&Os[r * kOStride] + threadIdx.x % 16);\n            }\n"),
    ("nvidia_sample.cuh", "no M - 1 clamp of the rows of A (edge tiles read outside A)",
     "const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;",
     "const int gr = row0 + r;"),
    ("nvidia_sample.cuh", "no test of the rows of C against M (edge tiles write outside C)",
     "if (gr < M) *((int4*)&C[", "*((int4*)&C["),
    ("nvidia_sample.cuh", "the persistent grid steps by one tile (blocks share tiles)",
     "block_pos += gridDim.x) {", "block_pos += 1) {"),
    ("nvidia_sample.cuh", "the B fragments of the second half of the warps' columns taken from the first",
     "&Bs[(ks * 16) * kBStride + (warpId % 2) * 64 + j * 16]", "&Bs[(ks * 16) * kBStride + j * 16]"),
    ("nvidia_sample.cuh", "a padding of 4 halves (wmma's pitch no longer a multiple of 16 bytes)",
     "constexpr int kSkew = 16;", "constexpr int kSkew = 4;"),
    ("nvidia_sample.cuh", "the epilogue's half tiles both from rows 0 to 63 of the tile",
     "const int gr = row0 + h * 64 + r;", "const int gr = row0 + r;"),
    ("step1.cuh", "1a: wait_group 2 (the stage of slice t - 1 is stored again while its group may still be in flight)",
     "        wgmma_wait<0>();\n", "        wgmma_wait<2>();\n"),
    ("step1.cuh", "2d without its barrier: wait_group 1 and no barrier before the stores of slice t + 1 (the other "
     "warpgroup's group t - 1 may still read that stage)",
     "        wgmma_wait<0>();\n        if (t + 1 < nt) {\n            __syncthreads();   // redundant without 2d",
     "        wgmma_wait<1>();\n        if (t + 1 < nt) {\n            // redundant without 2d"),
    ("step1.cuh", "no barrier after the stores of slice t + 1 (the MMA may read the stage before it is complete)",
     "            __syncthreads();   // slice t + 1 is in place\n", ""),
    ("step1.cuh", "no fence.proxy.async after the stores (the MMA's async proxy may not see them)",
     "        fence_proxy_async();\n    };", "    };"),
    ("step1.cuh", "slice t + 1 stored into the stage being read",
     "store((t + 1) % 2);", "store(t % 2);"),
    ("step1.cuh", "the last slice's group never waited before the output (the loop's wait_group 0 only before a next "
     "slice, none after the loop)",
     "        wgmma_wait<0>();\n        if (t + 1 < nt) {\n            __syncthreads();   // redundant without 2d: group t - 1 "
     "retired before the last barrier (kept as measured)\n            if constexpr (kSpread) store((t + 1) % 2);\n            "
     "else copy((t + 1) * 64, (t + 1) % 2);\n            __syncthreads();   // slice t + 1 is in place\n        }\n    }\n"
     "    wgmma_wait<0>();   // redundant without 2d: the last slice's wait_group 0 retired every group (kept as measured)\n",
     "        if (t + 1 < nt) {\n            wgmma_wait<0>();\n            __syncthreads();   // redundant without 2d: group t - 1 "
     "retired before the last barrier (kept as measured)\n            if constexpr (kSpread) store((t + 1) % 2);\n            "
     "else copy((t + 1) * 64, (t + 1) % 2);\n            __syncthreads();   // slice t + 1 is in place\n        }\n    }\n"),
    ("step1.cuh", "no barrier before the output tile overwrites the stages",
     "    __syncthreads();   // both warpgroups are done with the stages: the output tile overwrites them\n", ""),
    ("step1.cuh", "no wgmma.fence before the MMA of a slice",
     "        wgmma_fence();\n#pragma unroll", "#pragma unroll"),
    ("step1.cuh", "no commit of the MMA of a slice (wait_group retires nothing)",
     "        wgmma_commit();\n", ""),
    ("step1.cuh", "the accumulators reset at each slice (scaleD 0 at the first K16 of every slice)",
     "t > 0 || kk > 0", "kk > 0"),
    ("step1.cuh", "A's descriptor with a stride of 512 bytes between atoms of 8 rows",
     "((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32)", "((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32)"),
    ("step1.cuh", "B's descriptor with a leading offset of 4 KB between boxes of 64 columns",
     "((uint64_t)(8192 >> 4) << 16)", "((uint64_t)(4096 >> 4) << 16)"),
    ("step1.cuh", "B's descriptor with no swizzle",
     "((uint64_t)(1024 >> 4) << 32) |\n           ((uint64_t)1 << 62);", "((uint64_t)(1024 >> 4) << 32);"),
    ("step1.cuh", "A stored without the swizzle (the descriptor still reads it swizzled)",
     "*(int4*)(sa + r * 128 + ((c ^ (r % 8)) << 4)) = ra[i];", "*(int4*)(sa + r * 128 + (c << 4)) = ra[i];"),
    ("step1.cuh", "B stored without the swizzle",
     "(((c % 8) ^ (r % 8)) << 4)) = rb[i];", "((c % 8) << 4)) = rb[i];"),
    ("step1.cuh", "A's descriptor advanced by 16 bytes per K16 instead of 32",
     "da + 2 * kk", "da + kk"),
    ("step1.cuh", "B's descriptor advanced by 1 KB per K16 instead of 2",
     "db + 128 * kk", "db + 64 * kk"),
    ("step1.cuh", "both warpgroups on rows 0 to 63 of A",
     "desc_a(st + wg * 64 * 128)", "desc_a(st)"),
    ("step1.cuh", "B's rows of the slice taken from row k0 + r % 32",
     "rb[i] = *((const int4*)&B[(size_t)(k0 + r) * N + col0]", "rb[i] = *((const int4*)&B[(size_t)(k0 + r % 32) * N + col0]"),
    ("step1.cuh", "the output's second row of a lane 4 rows down instead of 8",
     "o[(r0 + 8) * G::kOPitch", "o[(r0 + 4) * G::kOPitch"),
    ("step1.cuh", "the output's columns of a lane at 8 j + t instead of 8 j + 2 t",
     "*(__half2*)&o[r0 * G::kOPitch + 8 * j + 2 * q]", "*(__half2*)&o[r0 * G::kOPitch + 8 * j + q]"),
    ("step1.cuh", "no M - 1 clamp of the rows of A (edge tiles read outside A)",
     "gr = kEdge ? min(row0 + r, M - 1) : row0 + r;", "gr = row0 + r;"),
    ("step1.cuh", "no test of the rows of C against M (edge tiles write outside C)",
     "if (gr < M) *((int4*)&C[", "*((int4*)&C["),
    ("step1.cuh", "the tiles' columns at 128 per block whatever the tile (blocks share columns)",
     "col0 = (blockIdx.x % tilesN) * kBN;", "col0 = (blockIdx.x % tilesN) * 128;"),
    ("step1.cuh", "v3: A stored without the swizzle (the descriptor still reads it swizzled)",
     "((c ^ (r % 8)) << 4)) = *((const int4*)&A[", "(c << 4)) = *((const int4*)&A["),
    ("step1.cuh", "v3: B stored without the swizzle",
     "(((c % 8) ^ (r % 8)) << 4)) =\n                    *((const int4*)&B[",
     "((c % 8) << 4)) =\n                    *((const int4*)&B["),
    ("step1.cuh", "v3: B's rows of the slice taken from row k0 + r % 32",
     "*((const int4*)&B[(size_t)(k0 + r) * N + col0] + c);", "*((const int4*)&B[(size_t)(k0 + r % 32) * N + col0] + c);"),
    ("step1.cuh", "v3: warps 0 to 3 copy half of A's slice (the other half left from an earlier slice)",
     "for (int i = 0; i < 2 * G::kALoads; i++) {", "for (int i = 0; i < G::kALoads; i++) {"),
    ("step1.cuh", "v3: no fence.proxy.async after the copy (the MMA's async proxy may not see it)",
     "        fence_proxy_async();   // the thread's stores, visible to the async proxy\n", ""),
    ("step1.cuh", "v3: no M - 1 clamp of the rows of A (edge tiles read outside A)",
     "ar = kEdge ? min(row0 + r, M - 1) : row0 + r;", "ar = row0 + r;"),
    ("step2.cuh", "2d: slice t - 1's stage released, and loaded again, before wait_group 1 retires its MMA",
     "        if constexpr (kWait1) step1::wgmma_wait<1>();\n        else step1::wgmma_wait<0>();\n",
     "        if constexpr (!kWait1) step1::wgmma_wait<0>();\n"),
    ("step2.cuh", "2a: 8 KB fewer announced than the stage's boxes (the phase completes before the last box lands)",
     "mbar_arrive_expect_tx(&full[s], G::kStage);", "mbar_arrive_expect_tx(&full[s], G::kStage - 8192);"),
    ("step2.cuh", "2a: 8 KB more announced than the stage's boxes (the phase never completes)",
     "mbar_arrive_expect_tx(&full[s], G::kStage);", "mbar_arrive_expect_tx(&full[s], G::kStage + 8192);"),
    ("step2.cuh", "the wait for a full stage with the other parity (the MMA reads it before its loads land)",
     "mbar_wait(&full[s], (t / kStages) & 1);", "mbar_wait(&full[s], ((t / kStages) + 1) & 1);"),
    ("step2.cuh", "the wait for an empty stage with the other parity (a load overwrites a stage still read)",
     "mbar_wait(&empty[ps], (p / kStages) & 1);", "mbar_wait(&empty[ps], ((p / kStages) + 1) & 1);"),
    ("step2.cuh", "no wait for the empty stage (a load may overwrite the other warpgroup's stage)",
     "                mbar_wait(&empty[ps], (p / kStages) & 1);\n", ""),
    ("step2.cuh", "the empty barrier counting one arrival (the first warpgroup to release the stage frees it)",
     "mbar_init(&empty[s], 2);", "mbar_init(&empty[s], 1);"),
    ("step2.cuh", "only warpgroup 0 releases stages (the empty barrier never completes)",
     "if (tid % 128 == 0) mbar_arrive(&empty[ps]);", "if (tid == 0) mbar_arrive(&empty[ps]);"),
    ("step2.cuh", "2d: the stage of slice t - 1 released after wait_group 2 (its group may still read it)",
     "        if constexpr (kWait1) step1::wgmma_wait<1>();", "        if constexpr (kWait1) step1::wgmma_wait<2>();"),
    ("step2.cuh", "the empty barriers on the full ones but one (they overlap)",
     "uint64_t* empty = full + kStages;", "uint64_t* empty = full + 1;"),
    ("step2.cuh", "no barrier between the initialization of the mbarriers and their first use",
     "        mbar_fence_init();\n    }\n    __syncthreads();\n", "        mbar_fence_init();\n    }\n"),
    ("step2.cuh", "the first kStages - 1 slices loaded before the loop instead of kStages (a slice is never loaded)",
     "for (int t = 0; t < kStages && t < nt; t++) load(t);", "for (int t = 0; t < kStages - 1 && t < nt; t++) load(t);"),
    ("step2.cuh", "slice p + kStages - 1 loaded into the released stage instead of p + kStages",
     "                load(p + kStages);", "                load(p + kStages - 1);"),
    ("step2.cuh", "A's box at row 0 whatever the tile",
     "tma_load_2d(st, &tmA, t * 64, row0, &full[s]);", "tma_load_2d(st, &tmA, t * 64, 0, &full[s]);"),
    ("step2.cuh", "B's boxes 32 columns apart instead of 64",
     "&tmB, col0 + 64 * j, t * 64", "&tmB, col0 + 32 * j, t * 64"),
    ("step2.cuh", "B's boxes 4 KB apart in the stage instead of 8 KB",
     "st + G::kABytes + j * 8192", "st + G::kABytes + j * 4096"),
    ("step2.cuh", "A's map with boxes of 64 rows (the stage's rows 64 to 127 never loaded)",
     "return make_map(A, K, M, 64, 128);", "return make_map(A, K, M, 64, 64);"),
    ("step2.cuh", "the MMA's stage taken from t % 2 whatever the number of stages",
     "const uint32_t st = s0 + s * G::kStage;", "const uint32_t st = s0 + (t % 2) * G::kStage;"),
    ("step2.cuh", "no wait_group 0 before the output",
     "    step1::wgmma_wait<0>();\n    __syncthreads();   // both warpgroups", "    __syncthreads();   // both warpgroups"),
    ("step4.cuh", "4a: both blocks of a cluster on the same tile (clusters along M)",
     "tm = (pair / tilesN) * 2 + rank;", "tm = (pair / tilesN) * 2;"),
    ("step4.cuh", "4a: a block after the last column of tiles writes (clusters along N, an odd number of columns)",
     "    if (col0 >= N) return;   // a block after the last column of tiles writes nothing\n", ""),
    ("step4.cuh", "step 4: thread 0 refills a stage before both warpgroups have released it",
     "                step2::mbar_wait(&empty[ps], (p / kStages) & 1);\n                load(p + kStages);",
     "                load(p + kStages);"),
    ("step4.cuh", "step 4: one warpgroup's release only (the empty barrier never completes)",
     "            if (tid % 128 == 0) step2::mbar_arrive(&empty[ps]);", "            if (tid == 0) step2::mbar_arrive(&empty[ps]);"),
    ("step4.cuh", "step 4: thread 0's stage from t % 3 with 4 stages in the ring",
     "    auto load = [&](int t) {   // slice t into its stage (thread 0)\n        const int s = t % kStages;",
     "    auto load = [&](int t) {   // slice t into its stage (thread 0)\n        const int s = t % 3;"),
    ("step4.cuh", "step 4: no barrier before the output tile overwrites the stages (a warpgroup's MMA may still read them)",
     "    step1::wgmma_wait<0>();\n    __syncthreads();   // both warpgroups are done with the stages: the output tile overwrites them\n",
     "    step1::wgmma_wait<0>();\n"),
    ("step4.cuh", "step 4: no barrier between the output tile's stores and its copy out",
     "    }\n    __syncthreads();\n    if (col0 >= N) return;   // a block after the last column", "    }\n    if (col0 >= N) return;   // a block after the last column"),
    ("step5.cuh", "step 5: the pairs of halves stored without the box's swizzle (the copy reads it swizzled)",
     "uint8_t* d = zp + (c / 8) * 8192 + (((c % 8) ^ (rg % 8)) << 4) + 4 * (l % 4);",
     "uint8_t* d = zp + (c / 8) * 8192 + ((c % 8) << 4) + 4 * (l % 4);"),
    ("step5.cuh", "step 5: a lane's second row of the fragment 4 rows down instead of 8",
     "*(__half2*)(d + (rg + 8) * 128)", "*(__half2*)(d + (rg + 4) * 128)"),
    ("step5.cuh", "step 5: a lane's pair of halves 2 bytes from the next lane's instead of 4",
     "<< 4) + 4 * (l % 4);\n        const int j", "<< 4) + 2 * (l % 4);\n        const int j"),
    ("step5.cuh", "step 5: the copy out without the box's swizzle",
     "rr * 128 + (((c % 8) ^ rr) << 4);", "rr * 128 + ((c % 8) << 4);"),
    ("step5.cuh", "5b: the last group of rows taken as full (pairs outside the matrix, others never taken)",
     "const int inGroup = UM - g * rows < rows ? UM - g * rows : rows;", "const int inGroup = rows;"),
    ("step5.cuh", "step 5: warp 0 refills a stage before both warpgroups have released it (its "
     "wait on the empty barrier left out)",
     "        if (li >= kStages) step2::mbar_wait(&empty[s], ((li / kStages) - 1) & 1);\n", ""),
    ("step5.cuh", "step 5: slice t - 1's stage released before wait_group 1 retires its MMA",
     "            step1::wgmma_wait<1>();\n            if (t >= 1) release(it - 1);\n        }\n        step1::wgmma_wait<0>();\n        release(it - 1);",
     "            if (t >= 1) release(it - 1);\n            step1::wgmma_wait<1>();\n        }\n        step1::wgmma_wait<0>();\n        release(it - 1);"),
    ("step5.cuh", "step 5: the loads issued by every lane of warp 0 (elect.sync left out)",
     "        if (step3::elect_one()) {\n            uint8_t* st = sm + s * G::kStage;", "        if (true) {\n            uint8_t* st = sm + s * G::kStage;"),
    ("step5.cuh", "step 5: the next pair's tile never taken (warp 0 loads the first pair's slices "
     "again)",
     "        if (lt == 0) tile(lu, lrow0, lcol0);", "        if (li == 0) tile(lu, lrow0, lcol0);"),
    ("step5.cuh", "step 5: the cluster's slices counted for one pair of tiles (warp 0 stops "
     "loading after it)",
     "(unsigned)((nu - cl + ncl - 1) / ncl) * (unsigned)nt;   // the cluster's slices", "(unsigned)nt;   // the cluster's slices"),
    ("step5.cuh", "step 5: every warp taken as the loader (each one's elected lane issues the "
     "slices: their bytes complete several times)",
     "const bool loader = __shfl_sync(0xffffffffu, tid, 0) < 32;", "const bool loader = true;"),
    ("step5.cuh", "step 5: no named barrier between the zone's stores and the copy out",
     "            zone_store(zp, acc, q, w, l);\n            step3::bar_named(2 + h, 128);\n            if (col0 < N) zone_copy",
     "            zone_store(zp, acc, q, w, l);\n            if (col0 < N) zone_copy"),
    ("step5.cuh", "step 5: a block after the last column of tiles writes (clusters along N, an odd "
     "number of columns)",
     "            if (col0 < N) zone_copy(zp, C, M, N, r0, col0 + 128 * q, ct);",
     "            zone_copy(zp, C, M, N, r0, col0 + 128 * q, ct);"),
    ("step5.cuh", "step 5: warp 0 waits for the empty stage with the other parity (it overwrites "
     "a stage still read)",
     "if (li >= kStages) step2::mbar_wait(&empty[s], ((li / kStages) - 1) & 1);",
     "if (li >= kStages) step2::mbar_wait(&empty[s], (li / kStages) & 1);"),
    ("step5.cuh", "step 5: the warpgroups release the stage of slice t instead of t - 1 (still "
     "read by the group in flight)",
     "            if (t >= 1) release(it - 1);\n        }\n        step1::wgmma_wait<0>();\n        release(it - 1);  "
     " // the tile's last slice",
     "            if (t >= 1) release(it);\n        }\n        step1::wgmma_wait<0>();\n        release(it - 1);   // "
     "the tile's last slice"),
    ("step5.cuh", "step 5: the output's named barrier counting 256 threads (each warpgroup's "
     "barrier waits for the other's threads)",
     "            step3::bar_named(2 + h, 128);   // the zone is free\n            zone_store(zp, acc, q, w, l);",
     "            step3::bar_named(2 + h, 256);   // the zone is free\n            zone_store(zp, acc, q, w, l);"),
    ("step5.cuh", "step 5: the copy out indexed by the thread's index in the block (the second "
     "warpgroup's threads start at row 8 of its zone)",
     "if (col0 < N) zone_copy(zp, C, M, N, r0, col0 + 128 * q, ct);",
     "if (col0 < N) zone_copy(zp, C, M, N, r0, col0 + 128 * q, tid);"),
    ("step5.cuh", "step 5: warp 0's stage from li % 3 with 4 stages in the ring",
     "        const int s = li % kStages;\n        if (li >= kStages)",
     "        const int s = li % 3;\n        if (li >= kStages)"),
    ("step5.cuh", "5a: warp 0's stage from the tile's own slice (its ring started again at each "
     "tile, out of step with the warpgroups')",
     "        const int s = li % kStages;\n        if (li >= kStages)",
     "        const int s = lt % kStages;\n        if (li >= kStages)"),
    ("step5.cuh", "5a: a tile's last slice not released (warp 0 waits for it forever)",
     "        release(it - 1);   // the tile's last slice",
     "        if (false) release(it - 1);   // the tile's last slice"),
    ("step5.cuh", "5a: the cluster's index from the block's (each cluster takes the pairs of two)",
     "const int nu = UM * UN, cl = blockIdx.x / 2, ncl = gridDim.x / 2;",
     "const int nu = UM * UN, cl = blockIdx.x, ncl = gridDim.x / 2;"),
    ("step5.cuh", "step 5: the descriptors rebuilt at the stage of the tile's own slice (t % 4)",
     "const uint32_t st = s0 + s * G::kStage;",
     "const uint32_t st = s0 + (t % kStages) * G::kStage;"),
    ("step5.cuh", "step 5: both warpgroups on rows 0 to 63 (the warpgroup's index taken as 0)",
     "const int h = wg;   // rows 64 h to 64 h + 63",
     "const int h = 0;   // rows 64 h to 64 h + 63"),
    ("step6.cuh", "6a: A's boxes of 8 rows instead of 16 (rows 8 to 15 never loaded)",
     "inline CUtensorMap map_a16(const half* A, int M, int K) { return step2::make_map(A, K, M, 64, 16); }",
     "inline CUtensorMap map_a16(const half* A, int M, int K) { return step2::make_map(A, K, M, 64, 8); }"),
    ("step6.cuh", "6a: the MMA's A (our B) read K-major (no tnspA)",
     "emu_wgmma_t(d, 16, da, db, scaleD, 1, 0);", "emu_wgmma_t(d, 16, da, db, scaleD, 0, 0);"),
    ("step6.cuh", "the rule: the bands' J rounded down (no band at all below one wave of 64 columns)",
     "const int J = (N + 64 * sms - 1) / (64 * sms);", "const int J = N / (64 * sms);"),
    ("step6.cuh", "the rule: 128 x 320 against 128 x 256 by their waves alone (its area forgotten)",
     "return {4 * w320 * 5 < 4 * w256 * 4 ? k320 : kT256Solo, 0};", "return {w320 < w256 ? k320 : kT256Solo, 0};"),
    ("step6.cuh", "the rule: 7e in pairs in one wave too (its class forgotten)",
     "const Kind k320 = tiles320 <= sms ? kT320Solo : kT320;", "const Kind k320 = kT320;"),
    ("step6.cuh", "the rule (7e's branch, step 7): M > 128 back to step 5's tile in pairs where 6c runs",
     "? k320 : kT256Solo, 0};", "? k320 : kT256, 0};"),
    ("step6.cuh", "the rule: M > 128 back to step 5's tile in pairs where 6c runs (step 6)",
     "? kT128 : kT256Solo, 0};", "? kT128 : kT256, 0};"),
    ("step6.cuh", "the rule: step 5's tile at an N it cannot take (step 6, N not a multiple of 256)",
     "return {N % 256 ? kT128 : ", "return {N % 256 ? kT256Solo : "),
    ("step6.cuh", "the rule: step 6 with 7e's tile (its switch forgotten)",
     "if (!with7e) return", "if (false) return"),
    ('step7.cuh', "the bands without warp specialization (v8), 7a: both blocks store their rows at the band's first "
                  'row',
     'step5::tma_store_2d(tmC, col0 + 64 * j, row0, zone + j * G::kBox);',
     'step5::tma_store_2d(tmC, col0 + 64 * j, 0, zone + j * G::kBox);'),
    ("step7.cuh", "the rule, 7a: the 64-row pairs without their condition of one wave",
     "if (2 * bands2 <= sms) return {kBands64Pairs, 2};", "return {kBands64Pairs, 2};"),
    ("step7.cuh", "the rule, 7d: the 128-row bands without their condition of one wave",
     "        if (bands2 <= sms) return {kBands128, 2};",
     "        return {kBands128, 2};"),
    ("step6.cuh", "the tiles: warp 0 refills a stage before both warpgroups have released it (its "
     "wait left out)",
     "        if (li >= NS) step2::mbar_wait(&empty[s], ((li / NS) - 1) & 1);\n        if (lt == 0)",
     "        if (lt == 0)"),
    ("step6.cuh", "the tiles: the next tile never taken (warp 0 loads the first tile's slices "
     "again)",
     "        if (lt == 0) tile(lu, lrow0, lcol0);",
     "        if (li == 0) tile(lu, lrow0, lcol0);"),
    ("step6.cuh", "the tiles: the loads issued by every lane of warp 0 (elect.sync left out)",
     "        if (step3::elect_one()) {\n            uint8_t* st = sm + s * T::kStage;",
     "        if (true) {\n            uint8_t* st = sm + s * T::kStage;"),
    ("step6.cuh", "the tiles: slice t - 1's stage released before wait_group 1 retires its MMA",
     "                step1::wgmma_wait<1>();\n                if (t > 0) release(it - 1);   // the stage of the "
     "slice before",
     "                if (t > 0) release(it - 1);   // the stage of the slice before\n                "
     "step1::wgmma_wait<1>();"),
    ("step6.cuh", "the bands: the stage refilled before the wait that retires its MMA",
     "        step1::wgmma_commit();\n        if constexpr (kZ) step1::wgmma_wait<0>();   // below 8 stages, the "
     "slice's own stage\n        else step1::wgmma_wait<1>();                // else the stage of the slice before\n  "
     "      if ((kZ || t >= 1) && loader && li < nt) load();   // that stage free: warp 0 refills it",
     "        step1::wgmma_commit();\n        if ((kZ || t >= 1) && loader && li < nt) load();   // that stage free: "
     "warp 0 refills it\n        if constexpr (kZ) step1::wgmma_wait<0>();   // below 8 stages, the slice's own "
     "stage\n        else step1::wgmma_wait<1>();                // else the stage of the slice before"),
    ("step6.cuh", "the bands: the loads issued by every lane of warp 0 (elect.sync left out)",
     "        if (step3::elect_one()) {\n            uint8_t* st = sm + s * G::kStage;\n            constexpr "
     "uint32_t kBytes",
     "        if (true) {\n            uint8_t* st = sm + s * G::kStage;\n            constexpr uint32_t kBytes"),
    ("step6.cuh", "the bands: every warp taken as the loader (each one's elected lane issues the "
     "slices: their bytes complete several times)",
     "    const bool loader = __shfl_sync(0xffffffffu, tid, 0) < 32;   // the same in the whole warp, broadcast from "
     "lane 0\n    int li = 0;",
     "    const bool loader = true;   // the same in the whole warp, broadcast from lane 0\n    int li = 0;"),
    ("step6.cuh", "the tiles, 7e: the n128 of 128 x 320 on B's first boxes",
     "&acc[96]), da, db + 1536, kk > 0 || t > 0);",
     "&acc[96]), da, db, kk > 0 || t > 0);"),
    ("step6.cuh", "the tiles: the descriptors rebuilt at the stage of the tile's own slice "
     "(t % NS)",
     "const uint32_t st = s0 + T::kStage * s;",
     "const uint32_t st = s0 + T::kStage * (t % NS);"),
    ("step6.cuh", "the tiles, 6b: the two warpgroups of 64 x 128 on the same columns of B",
     "const uint32_t rb = st + T::kABytes + (T::kSplitN ? 8192 * h * (T::kWgCols / 64) : 0);",
     "const uint32_t rb = st + T::kABytes;"),
    ("step6.cuh", "the tiles, 6b: the second warpgroup of 64 x 128 writing the first one's columns",
     "c0w = col0 + (T::kSplitN ? T::kWgCols * h : 0);",
     "c0w = col0;"),
    ("step6.cuh", "the tiles, 7e: the n128 of 128 x 320 on accumulators that overlap the n192's",
     "reinterpret_cast<float(*)[64]>(&acc[96])",
     "reinterpret_cast<float(*)[64]>(&acc[64])"),
    ("step6.cuh", "the tiles, 7e: the output's passes after the second forgotten (128 x 320's last "
     "64 columns)",
     "for (int q = 0; q != T::kPasses; q++) {",
     "for (int q = 0; q != (T::kPasses < 2 ? T::kPasses : 2); q++) {"),
    ("step6.cuh", "the tiles: the threads' copy out of the zone without the swizzle",
     "(((rr % 8) ^ c8) << 4);",
     "(c8 << 4);"),
    ("step6.cuh", "the tiles: the threads' copy out on the first rows of each round only",
     "const int r = rr + step * i;",
     "const int r = rr;"),
    ("step6.cuh", "the tiles, pairs: the second warpgroup out of the cluster barrier (the cluster's "
     "wait never returns)",
     "    if constexpr (kCl) {   // the cluster synchronized in one piece\n",
     "    if (kCl && wg == 0) {   // the cluster synchronized in one piece\n"),
    ("step6.cuh", "the tiles: the pairs of halves stored without the box's swizzle (the copy reads "
     "it swizzled)",
     "uint8_t* d = zw + 8192 * (c / 8) + (((c % 8) ^ (rg % 8)) << 4) + 4 * (l % 4);",
     "uint8_t* d = zw + 8192 * (c / 8) + ((c % 8) << 4) + 4 * (l % 4);"),
    ("step6.cuh", "the tiles: a lane's second row of the fragment 4 rows down instead of 8",
     "*(__half2*)(d + 128 * (rg + 8))",
     "*(__half2*)(d + 128 * (rg + 4))"),
    ("step6.cuh", "the tiles: the pass's last group of 8 columns left out (the loop one short)",
     "                if (8 * c >= qc) break;\n",
     "                if (8 * c >= qc - 8) break;\n"),
    ("step6.cuh", "the tiles: both blocks of a pair along M on the same row of tiles",
     "row0 = BM * (kPair == kAlongM ? 2 * um + rank : um);",
     "row0 = BM * um;"),
    ("step6.cuh", "the tiles, 6c: the n256 MMA reading B from its second box on (B's columns "
     "shifted by 64)",
     "step1::wgmma<256>(*reinterpret_cast<float(*)[128]>(acc), da, db, kk > 0 || t > 0);",
     "step1::wgmma<256>(*reinterpret_cast<float(*)[128]>(acc), da, db + 512, kk > 0 || t > 0);"),
    ("step6.cuh", "the bands: the descriptors' offset from the slice's index instead of its stage",
     "const uint32_t ds = (G::kStage >> 4) * s;",
     "const uint32_t ds = (G::kStage >> 4) * t;"),
    ("step6.cuh", "the bands: the stage's offset in bytes instead of units of 16 bytes",
     "const uint32_t ds = (G::kStage >> 4) * s;",
     "const uint32_t ds = G::kStage * s;"),
    ("step6.cuh", "the bands: every box of the output stored from the zone's first box",
     "step5::tma_store_2d(&tmC, col0 + 64 * j, 0, zone + 2048 * j);",
     "step5::tma_store_2d(&tmC, col0 + 64 * j, 0, zone);"),
    ("step6.cuh", "the bands: the MMA's A advanced by 1 KB per K16 instead of 2",
     "wgmma_n16t(d, dB0 + (512 * j + 128 * kk) + ds,",
     "wgmma_n16t(d, dB0 + (512 * j + 64 * kk) + ds,"),
    ("step6.cuh", "the bands: 8 KB fewer announced than a band's stage brings",
     "constexpr uint32_t kBytes = G::kStage;",
     "constexpr uint32_t kBytes = G::kStage - 8192;"),
    ("step6.cuh", "the bands: the transposed store given the fragment's second and third 8 x 8 "
     "matrices swapped",
     "step5::pack_half2(acc[j][2], acc[j][3]),\n                          step5::pack_half2(acc[j][4], acc[j][5]),",
     "step5::pack_half2(acc[j][4], acc[j][5]),\n                          step5::pack_half2(acc[j][2], acc[j][3]),"),
    ("step6.cuh", "the bands: every band's MMA on the first range of accumulators",
     "float* d = acc[j];",
     "float* d = acc[0];"),
    ("step6.cuh", "the bands: the output's chunk from the wrong half of the lanes",
     "c = (l / 8) % 2 + 2 * w;",
     "c = l / 16 + 2 * w;"),
    ('step7.cuh', 'the bands without warp specialization (v8): the loads issued by every lane of warp 0 (elect.sync '
                  'left out)',
     '        if (step3::elect_one()) {\n            uint8_t* st = sm + s * G::kStage;\n            '
     'step2::mbar_arrive_expect_tx(&full[s], G::kStage);',
     '        if (true) {\n            uint8_t* st = sm + s * G::kStage;\n            '
     'step2::mbar_arrive_expect_tx(&full[s], G::kStage);'),
    ('step7.cuh', 'the bands without warp specialization (v8): the stage refilled before the wait that retires its '
                  'MMA',
     "        if constexpr (kZ) step1::wgmma_wait<0>();   // the 16-row bands below 8 stages: the slice's own "
     'stage\n        else step1::wgmma_wait<1>();                // else the stage of the slice before\n        if '
     '((kZ || t >= 1) && loader && li < nt) load();   // that stage free: warp 0 refills it',
     '        if ((kZ || t >= 1) && loader && li < nt) load();   // that stage free: warp 0 refills it\n        if '
     "constexpr (kZ) step1::wgmma_wait<0>();   // the 16-row bands below 8 stages: the slice's own stage\n        "
     'else step1::wgmma_wait<1>();                // else the stage of the slice before'),
    ('step7.cuh', "the bands without warp specialization (v8), 7a: both blocks load A's first 64 rows",
     'step2::tma_load_2d(st, &tmA, li * 64, row0, &full[s]);',
     'step2::tma_load_2d(st, &tmA, li * 64, 0, &full[s]);'),
    ('step7.cuh', "the bands without warp specialization (v8): every warp taken as the loader (each one's elected "
                  'lane issues the slices: their bytes complete several times)',
     '    const bool loader = __shfl_sync(0xffffffffu, tid, 0) < 32;   // the same in the whole warp, broadcast from '
     'lane 0',
     '    const bool loader = true;   // the same in the whole warp, broadcast from lane 0'),
    ('step7.cuh', 'the bands without warp specialization (v8), 7a: the two blocks of a cluster on two bands',
     'const int band = (int)(blockIdx.x / (P == kAlongM ? 2u : 1u));',
     'const int band = (int)blockIdx.x;'),
    ('step7.cuh', "the bands without warp specialization (v8): the MMA's B (our A) advanced by 16 bytes per K16 "
                  'instead of 32',
     'dA0 + ds + 2 * kk, kk > 0 || t > 0);',
     'dA0 + ds + kk, kk > 0 || t > 0);'),
    ('step7.cuh', 'the bands without warp specialization (v8), in clusters (7a, 7b): warp 0 out of the cluster '
                  "barrier (the cluster's wait never returns)",
     '    if constexpr (kPair) {   // the cluster synchronized in one piece\n',
     '    if (kPair && tid >= 32) {   // the cluster synchronized in one piece\n'),
    ('step7.cuh', "the bands without warp specialization (v8): the descriptors' offset from the slice's index "
                  'instead of its stage',
     'const uint32_t ds = s * (G::kStage >> 4);',
     'const uint32_t ds = t * (G::kStage >> 4);'),
    ("step7.cuh", "the rule, 7c: groups of up to 32 rows",
     "top = UM < 16 ? UM : 16;",
     "top = UM < 32 ? UM : 32;"),
    ("step5.cuh", "step 5: one warpgroup's release only (the empty barrier never completes)",
     "        if (ct == 0) step2::mbar_arrive(&empty[p % kStages]);\n        if (loader",
     "        if (tid == 0) step2::mbar_arrive(&empty[p % kStages]);\n        if (loader"),
    ("step5.cuh", "step 5: its own A from row 0 (every block on the rows of the first tiles)",
     "step2::tma_load_2d(st, &tmA, lt * 64, lrow0, &full[s]);", "step2::tma_load_2d(st, &tmA, lt * 64, 0, &full[s]);"),
    ("step5.cuh", "step 5: each block's empty barriers counting four arrivals (they never "
     "complete)",
     "step2::mbar_init(&empty[s], 2);   // the block's two warpgroups",
     "step2::mbar_init(&empty[s], 4);   // the block's two warpgroups"),
    ("step6.cuh", "the tiles: one warpgroup's release only (the empty barrier never completes)",
     "        if (ct == 0) step2::mbar_arrive(&empty[p % NS]);\n        if (loader && li < total) load();",
     "        if (tid == 0) step2::mbar_arrive(&empty[p % NS]);\n        if (loader && li < total) load();"),
    ("step6.cuh", "the tiles: a block's empty barriers counting four arrivals (they never complete)",
     "step2::mbar_init(&empty[s], 2);   // the block's two warpgroups",
     "step2::mbar_init(&empty[s], 4);   // the block's two warpgroups"),
    ('step7.cuh', "7g: the stage's offset in bytes instead of units of 16 bytes",
     'const uint32_t ds = s * (G::kStage >> 4);',
     'const uint32_t ds = s * G::kStage;'),
    ('step7.cuh', "7h: the empty barriers counting two arrivals (the other block's loads may overwrite a stage still "
                  'read)',
     'step2::mbar_init(&empty[s], kShare ? 4 : 2);',
     'step2::mbar_init(&empty[s], 2);'),
    ('step7.cuh', "7h: both blocks load A's first 64 rows (clusters along N)",
     '&tmA, t * 64, row0 + 64 * rank, &full[s], 3);',
     '&tmA, t * 64, row0, &full[s], 3);'),
    ('step7.cuh', "step 5's tile: the descriptors' stage from the tile's own slices (the stage of slice t of a tile "
                  'taken as t % 4)',
     'const uint32_t ds = ((first + t) % kStages) * (G::kStage >> 4);',
     'const uint32_t ds = (t % kStages) * (G::kStage >> 4);'),
    ('step7.cuh', "step 5's tile: the loading warp's ring started again at each tile (its stages and phases out of "
                  "step with the computing warpgroups')",
     "        unsigned it = 0;   // the slices so far, over the cluster's units\n        for (int u = cl; u < nu; u "
     '+= ncl) {\n',
     "        unsigned it = 0;   // the slices so far, over the cluster's units\n        for (int u = cl; u < nu; u "
     '+= ncl, it = 0) {\n'),
    ('step7.cuh', "7i, step 5's tile: no wait of the loading warp for its last stages (a block may leave while its "
                  'multicast loads still land in the other block)',
     '        for (unsigned p = total > kStages ? total - kStages : 0; p < total; p++)',
     '        for (unsigned p = total; p < total; p++)'),
    ('step7.cuh', "step 5's tile: the cluster's index from the block's (each cluster takes the pairs of two)",
     'cl = blockIdx.x / 2, ncl = gridDim.x / 2;',
     'cl = blockIdx.x, ncl = gridDim.x / 2;'),
    ('step7.cuh', "7j: the second box stored at the first box's columns",
     'tma_store_2d(&tmC, c0 + 64, r0, zone + 8192);',
     'tma_store_2d(&tmC, c0, r0, zone + 8192);'),
    ('step7.cuh', 'the bands with warp specialization: below 8 stages, the stage released before its MMA are retired',
     '            step1::wgmma_wait<0>();\n            release(t);',
     '            release(t);\n            step1::wgmma_wait<0>();'),
    ('step7.cuh', "the bands with warp specialization: the MMA's A advanced by 1 KB per K16 instead of 2",
     'dB0 + ds + 512 * j + 128 * kk',
     'dB0 + ds + 512 * j + 64 * kk'),
    ('step7.cuh', 'the tiles with warp specialization, 6b: the two warpgroups of 64 x 128 on the same columns of B',
     's0 + T::kABytes + (T::kSplitN ? h * (T::kWgCols / 64) * 8192 : 0)',
     's0 + T::kABytes'),
    ('step7.cuh', "the tiles with warp specialization, 6b: the second warpgroup of 64 x 128 writing the first one's "
                  'columns',
     'c0w = col0 + (T::kSplitN ? h * T::kWgCols : 0);',
     'c0w = col0;'),
    ('step7.cuh', "7e: the n128 of 128 x 320 on B's first boxes",
     'da, db + 3 * 512, t > 0 || kk > 0);',
     'da, db, t > 0 || kk > 0);'),
    ('step7.cuh', "7e: the n128 of 128 x 320 on accumulators that overlap the n192's",
     'reinterpret_cast<float(*)[64]>(acc + 96)',
     'reinterpret_cast<float(*)[64]>(acc + 64)'),
    ('step7.cuh', "7e: B's third box loaded by both blocks of the pair (its bytes complete twice)",
     'if ((j < (T::kBoxes + 1) / 2) == (rank == 0))',
     'if ((j < (T::kBoxes + 1) / 2) == (rank == 0) || j == 2)'),
    ('step7.cuh', "the tiles with warp specialization: the output's passes after the second forgotten (128 x 320's "
                  'last 64 columns)',
     'for (int q = 0; q < T::kPasses; q++) {',
     'for (int q = 0; q < (T::kPasses < 2 ? T::kPasses : 2); q++) {'),
    ('step7.cuh', "the tiles with warp specialization, 6c: the n256 MMA reading B from its second box on (B's "
                  'columns shifted by 64)',
     'step1::wgmma<256>(*reinterpret_cast<float(*)[128]>(acc), da, db, t > 0 || kk > 0);',
     'step1::wgmma<256>(*reinterpret_cast<float(*)[128]>(acc), da, db + 512, t > 0 || kk > 0);'),
    ('step7.cuh', "the bands with warp specialization at 8 stages: slice t - 1's stage released before wait_group 1 "
                  'retires its MMA',
     '            step1::wgmma_wait<1>();\n            if (t >= 1) release(t - 1);\n',
     '            if (t >= 1) release(t - 1);\n            step1::wgmma_wait<1>();\n'),
    ('step7.cuh', "the bands with warp specialization: the MMA's B (our A) advanced by 16 bytes per K16 instead of 32",
     'dA0 + ds + 2 * kk, t > 0 || kk > 0);',
     'dA0 + ds + kk, t > 0 || kk > 0);'),
    ('step7.cuh', "7a, pairs along M: both blocks load A's first 64 rows",
     '} else {\n                    step2::tma_load_2d(st, &tmA, t * 64, row0, &full[s]);',
     '} else {\n                    step2::tma_load_2d(st, &tmA, t * 64, 0, &full[s]);'),
    ('step7.cuh', "7a, pairs along M: both blocks multicast all of B's boxes (each stage receives its bytes twice)",
     'if (j % 2 == rank)',
     'if (true)'),
    ('step7.cuh', '7a, pairs along M: the two blocks of a cluster on two bands',
     'const int band = P == kAlongM ? blockIdx.x / 2 : blockIdx.x;',
     'const int band = blockIdx.x;'),
    ('step7.cuh', "7a, pairs along M: both blocks store their rows at the band's first row",
     'step5::tma_store_2d(&tmC, col0 + 64 * j, row0, zone + j * G::kBox);',
     'step5::tma_store_2d(&tmC, col0 + 64 * j, 0, zone + j * G::kBox);'),
    ('step7.cuh', 'the rule, 7b: its clusters of eight without their condition (more than one per GPC, or a second '
                  'wave)',
     'return {(bands5 + 7) / 8 <= (clusters8 + 1) / 2 ? kBands16Wide : kBands16, J16};',
     'return {kBands16Wide, J16};'),
    ('step7.cuh', 'the rule, 7b: two clusters of eight per GPC (as many as the GPU holds)',
     '(bands5 + 7) / 8 <= (clusters8 + 1) / 2 ?',
     '(bands5 + 7) / 8 <= clusters8 ?'),
    ('step7.cuh', 'the rule, 7c: the rounds not kept (the largest group taken whatever its grid)',
     '        if ((tiles + R - 1) / R == rounds) break;',
     '        break;'),
    ('step7.cuh', 'the rule, 7c: the grid not a whole number of groups (all the blocks held)',
     '        R = G * (sms / G);',
     '        R = sms;'),
    ('step7.cuh', "7f, step 5's tile: the loading warp does not wait for the empty stage",
     '                if (it >= kStages) step2::mbar_wait(&empty[s], ((it / kStages) - 1) & 1);   // released by '
     'them all\n',
     ''),
    ('step7.cuh', "7f, step 5's tile: the loading warp waits for the empty stage with the other parity (it "
                  'overwrites a stage still read)',
     'step2::mbar_wait(&empty[s], ((it / kStages) - 1) & 1);   // released by them all',
     'step2::mbar_wait(&empty[s], (it / kStages) & 1);   // released by them all'),
    ('step7.cuh', "7f, step 5's tile: the loads issued by every lane of the loading warp (elect.sync left out)",
     '                if (step3::elect_one()) {\n                    uint8_t* st = sm + s * G::kStage;\n             '
     '       step2::mbar_arrive_expect_tx(&full[s], G::kStage);\n                    if constexpr (kShare)',
     '                {\n                    uint8_t* st = sm + s * G::kStage;\n                    '
     'step2::mbar_arrive_expect_tx(&full[s], G::kStage);\n                    if constexpr (kShare)'),
    ('step7.cuh', "7f, step 5's tile: warps 1 to 3 of the loading warpgroup do not leave (they load too)",
     '        step4::cluster_wait();             // to the other block\n        if (tid >= 32) return;\n',
     '        step4::cluster_wait();             // to the other block\n'),
    ('step7.cuh', "7f, step 5's tile: setmaxnreg in warp 0 of the loading warpgroup only (not the same in its "
                  'warpgroup)',
     '    if (wg == 0) {   // the loading warpgroup: warp 0 loads, warps 1 to 3 leave\n        '
     'setmaxnreg_dec<24>();\n        step4::cluster_arrive_relaxed();',
     '    if (wg == 0) {   // the loading warpgroup: warp 0 loads, warps 1 to 3 leave\n        if (tid < 32) '
     'setmaxnreg_dec<24>();\n        step4::cluster_arrive_relaxed();'),
    ('step7.cuh', "7f, step 5's tile: one computing warpgroup releases the stages (the empty barrier never completes)",
     '        if (ct == 0) {\n            step2::mbar_arrive(&empty[p % kStages]);',
     '        if (tid == 128) {\n            step2::mbar_arrive(&empty[p % kStages]);'),
    ('step7.cuh', "7f, step 5's tile: slice t - 1's stage released before wait_group 1 retires its MMA",
     '            step1::wgmma_wait<1>();\n            if (t >= 1) release(it - 1);\n        }\n        '
     'step1::wgmma_wait<0>();\n        if (!kShare || kTail || u + ncl < nu) release(it - 1);',
     '            if (t >= 1) release(it - 1);\n            step1::wgmma_wait<1>();\n        }\n        '
     'step1::wgmma_wait<0>();\n        if (!kShare || kTail || u + ncl < nu) release(it - 1);'),
    ('step7.cuh', "7f, step 5's tile: the release of slice t instead of t - 1 (still read by the group in flight)",
     '            if (t >= 1) release(it - 1);\n        }\n        step1::wgmma_wait<0>();\n        if (!kShare || '
     'kTail || u + ncl < nu) release(it - 1);',
     '            release(it);\n        }\n        step1::wgmma_wait<0>();\n        if (!kShare || kTail || u + ncl '
     '< nu) release(it - 1);'),
    ('step7.cuh', "7f, step 5's tile: the output's named barrier counting 256 threads (each warpgroup's barrier "
                  "waits for the other's threads)",
     '            step3::bar_named(2 + h, 128);   // the zone is free',
     '            step3::bar_named(2 + h, 256);   // the zone is free'),
    ('step7.cuh', "7h, step 5's tile: A's halves multicast to the block itself only (the other block never gets them)",
     'tma_load_2d_multicast(st + rank * 8192, &tmA, t * 64, row0 + 64 * rank, &full[s], 3);',
     'tma_load_2d_multicast(st + rank * 8192, &tmA, t * 64, row0 + 64 * rank, &full[s], 1 << rank);'),
    ('step7.cuh', "7h, step 5's tile: no release on the other block (its empty barrier never completes)",
     '            if constexpr (kShare) mbar_arrive_cluster(emptyOther + 8 * (p % kStages));\n',
     ''),
    ('step7.cuh', "7h, step 5's tile: the release on the other block always on its stage 0",
     'mbar_arrive_cluster(emptyOther + 8 * (p % kStages));',
     'mbar_arrive_cluster(emptyOther);'),
    ('step7.cuh', "7h, step 5's tile: the other block's empty barrier mapped to its own (a stage refilled while the "
                  'other block may still read it)',
     'emptyOther = kShare ? mapa(step1::smem_addr(empty), (uint32_t)other) : 0;',
     'emptyOther = kShare ? mapa(step1::smem_addr(empty), (uint32_t)rank) : 0;'),
    ('step7.cuh', "7i, step 5's tile: a tile's last slice not released (the loading warp's last wait never returns)",
     'if (!kShare || kTail || u + ncl < nu) release(it - 1);',
     'if (u + ncl < nu) release(it - 1);'),
    ('step7.cuh', "7j, step 5's tile: no wait for the TMA's reads before the zone is written again",
     "            if (kTma && ct == 0) step5::bulk_wait<0, true>();   // the zone read by the previous pass's "
     'stores\n            step3::bar_named(2 + h, 128);   // the zone is free',
     '            step3::bar_named(2 + h, 128);   // the zone is free'),
    ('step7.cuh', "7j, step 5's tile: no fence.proxy.async between the zone's stores and the TMA stores",
     '            if constexpr (kTma) step1::fence_proxy_async();   // the zone, visible to the TMA\n',
     ''),
    ('step7.cuh', "step 5's tile: the first pass's accumulators stored at both passes",
     '            step5::zone_store(zp, acc, q, w, l);\n',
     '            step5::zone_store(zp, acc, 0, w, l);\n'),
    ('step7.cuh', "7j, step 5's tile: no named barrier between the warpgroup's zone stores and the TMA stores",
     '            step3::bar_named(2 + h, 128);\n            const int c0 = col0 + 128 * q;',
     '            const int c0 = col0 + 128 * q;'),
    ('step7.cuh', "7j, step 5's tile: no wait for the TMA stores before the block leaves",
     "    if (kTma && ct == 0) step5::bulk_wait<0, false>();   // the stores done: their source is this block's "
     'shared memory\n}\n\n// ---------------------------------------------------------------- the persistent kernel',
     '}\n\n// ---------------------------------------------------------------- the persistent kernel'),
    ('step7.cuh', "7j, step 5's tile: the TMA stores not committed (no wait covers them)",
     '                    step5::bulk_commit();\n                }\n            } else if (col0 < N) {',
     '                }\n            } else if (col0 < N) {'),
    ('step7.cuh', "without 7j, step 5's tile: the copy out 64 rows down (the warpgroup's rows written at the other's)",
     'step5::zone_copy(zp, C, M, N, r0, c0, ct);',
     'step5::zone_copy(zp, C, M, N, row0 + 64 * (1 - h), c0, ct);'),
    ('step7.cuh', "without 7h, step 5's tile: each block's own A taken 64 rows down",
     '                    else   // all of its own A\n                        step2::tma_load_2d(st, &tmA, t * 64, '
     'row0, &full[s]);',
     '                    else   // all of its own A\n                        step2::tma_load_2d(st, &tmA, t * 64, '
     'row0 + 64, &full[s]);'),
    ('step7.cuh', "without 7h, step 5's tile: the empty barriers counting four arrivals (they never complete)",
     'step2::mbar_init(&empty[s], kShare ? 4 : 2);   // the four computing warpgroups of the cluster (7h)',
     'step2::mbar_init(&empty[s], 4);   // the four computing warpgroups of the cluster (7h)'),
    ('step7.cuh', "without 7i, step 5's tile: no cluster barrier at the end (a block may leave while the other still "
                  'arrives on its barriers)',
     '    if constexpr (kShare && !kTail) {   // without 7i: a cluster barrier before leaving, both blocks done with '
     'stages\n        cluster_arrive();\n        step4::cluster_wait();\n    }\n    if (kTma && ct == 0) '
     "step5::bulk_wait<0, false>();   // the stores done: their source is this block's shared memory\n}\n\n// ------",
     "    if (kTma && ct == 0) step5::bulk_wait<0, false>();   // the stores done: their source is this block's "
     'shared memory\n}\n\n// ------'),
    ('step7.cuh', "without 7g, step 5's tile: the descriptors rebuilt at the stage of the tile's own slice (t % 4)",
     "            const uint32_t st = s0 + s * G::kStage;   // without 7g: the stage's address",
     "            const uint32_t st = s0 + (t % kStages) * G::kStage;   // without 7g: the stage's address"),
    ('step7.cuh', "without 7g, step 5's tile: both warpgroups on rows 0 to 63 (the warpgroup's index taken as 0)",
     'const int h = kUni ? __shfl_sync(0xffffffffu, wg - 1, 0) : wg - 1;\n    const uint32_t s0 = '
     'step1::smem_addr(sm), emptyOther',
     'const int h = kUni ? __shfl_sync(0xffffffffu, wg - 1, 0) : 0;\n    const uint32_t s0 = step1::smem_addr(sm), '
     'emptyOther'),
    ('step7.cuh', '7f, the tiles: the loading warp does not wait for the empty stage',
     '                if (it >= NS) step2::mbar_wait(&empty[s], ((it / NS) - 1) & 1);\n',
     ''),
    ('step7.cuh', '7f, the tiles: the loads issued by every lane of the loading warp (elect.sync left out)',
     '                if (step3::elect_one()) {\n                    uint8_t* st = sm + s * T::kStage;',
     '                {\n                    uint8_t* st = sm + s * T::kStage;'),
    ('step7.cuh', "7f, the tiles: the loading warpgroup keeping 64 registers (the computing ones' setmaxnreg.inc "
                  'never served)',
     '        setmaxnreg_dec<24>();\n        if constexpr (kCl) {',
     '        setmaxnreg_dec<64>();\n        if constexpr (kCl) {'),
    ('step7.cuh', '7f, the tiles: one computing warpgroup releases the stages (the empty barrier never completes)',
     '        if (ct == 0) {\n            step2::mbar_arrive(&empty[p % NS]);',
     '        if (tid == 128) {\n            step2::mbar_arrive(&empty[p % NS]);'),
    ('step7.cuh', '7f, the tiles: warps 1 to 3 of the loading warpgroup do not leave (they load too)',
     '            step4::cluster_wait();\n        }\n        if (tid >= 32) return;\n        unsigned it = 0;',
     '            step4::cluster_wait();\n        }\n        unsigned it = 0;'),
    ('step7.cuh', "7g, the tiles: the descriptors' stage from the tile's own slices (the stage of slice t taken as t "
                  '% NS)',
     '__shfl_sync(0xffffffffu, (int)(it % NS), 0) * (T::kStage >> 4)',
     '__shfl_sync(0xffffffffu, t % NS, 0) * (T::kStage >> 4)'),
    ('step7.cuh', "7g, the tiles: both warpgroups of the 128-row tiles on rows 0 to 63 (the warpgroup's index taken "
                  'as 0)',
     'const int h = kUni ? __shfl_sync(0xffffffffu, wg - 1, 0) : wg - 1;\n    const uint32_t s0 = '
     'step1::smem_addr(sm);\n',
     'const int h = kUni ? __shfl_sync(0xffffffffu, 0, 0) : wg - 1;\n    const uint32_t s0 = step1::smem_addr(sm);\n'),
    ('step7.cuh', "7h, the tiles: both blocks of a pair load all of B's boxes (their bytes complete twice)",
     '                            if ((j < (T::kBoxes + 1) / 2) == (rank == 0))\n',
     '                            if (true)\n'),
    ('step7.cuh', "7h, the tiles: a pair's empty barriers counting two arrivals (the other block's loads may "
                  'overwrite a stage still read)',
     'step2::mbar_init(&empty[s], kSh ? 4 : 2);',
     'step2::mbar_init(&empty[s], 2);'),
    ('step7.cuh', '7h, the tiles: no release on the other block of a pair (its empty barrier never completes)',
     '            if constexpr (kSh) mbar_arrive_cluster(emptyOther + 8 * (p % NS));\n',
     ''),
    ('step7.cuh', '7i, the tiles: no wait of the loading warp for its last stages (a block may leave while its '
                  'multicast loads still land in the other)',
     '        if constexpr (kSh && kTail)   // 7i: the last stages released by the four computing warpgroups\n       '
     '     for (unsigned p = total > NS ? total - NS : 0; p < total; p++)\n                step2::mbar_wait(&empty[p '
     '% NS], (p / NS) & 1);\n',
     ''),
    ('step7.cuh', "7i, the tiles: a tile's last slice not released (the loading warp's last wait never returns)",
     'const bool last = !kSh || kTail || u + ncl < nu;',
     'const bool last = u + ncl < nu;'),
    ('step7.cuh', "7i, the tiles: the cluster's slices counted for one tile (the loading warp's last waits on stages "
                  'already released)',
     '    const unsigned total = (unsigned)((nu - cl + ncl - 1) / ncl) * (unsigned)nt;\n    auto tile = [&](int u, '
     'int& row0, int& col0) {\n        int um, un;\n        step5::place(u, UM, UN, groupRows, true, um, un);',
     '    const unsigned total = (unsigned)nt;\n    auto tile = [&](int u, int& row0, int& col0) {\n        int um, '
     'un;\n        step5::place(u, UM, UN, groupRows, true, um, un);'),
    ('step7.cuh', 'the tiles, pairs: both blocks of a pair on the same row of tiles',
     'row0 = (kPair == step6::kAlongM ? 2 * um + rank : um) * BM;',
     'row0 = (kPair == step6::kAlongM ? 2 * um : um) * BM;'),
    ('step7.cuh', "7j, the tiles: no wait for the TMA's reads before the zone is written again",
     "            if (kTma && ct == 0) step5::bulk_wait<0, true>();   // the zone read by the previous pass's "
     'stores\n            step3::bar_named(2 + h, 128);\n            uint8_t* zw = sm + (zone - s0);',
     '            step3::bar_named(2 + h, 128);\n            uint8_t* zw = sm + (zone - s0);'),
    ('step7.cuh', "7j, the tiles: no fence.proxy.async between the zone's stores and the TMA stores",
     '            if constexpr (kTma) step1::fence_proxy_async();\n            step3::bar_named(2 + h, 128);\n',
     '            step3::bar_named(2 + h, 128);\n'),
    ('step7.cuh', "the tiles, the output: each lane's second row written over its first (rows rg + 8 never written)",
     '                *(__half2*)(d + 128 * (rg + 8)) = __floats2half2_rn(acc[4 * j + 2], acc[4 * j + 3]);\n',
     '                *(__half2*)(d + 128 * rg) = __floats2half2_rn(acc[4 * j + 2], acc[4 * j + 3]);\n'),
    ('step7.cuh', "the tiles, the output: the zone's boxes written without the 128-byte swizzle",
     '                uint8_t* d = zw + 8192 * (c / 8) + (((c % 8) ^ (rg % 8)) << 4) + 4 * (l % 4);\n',
     '                uint8_t* d = zw + 8192 * (c / 8) + ((c % 8) << 4) + 4 * (l % 4);\n'),
    ('step7.cuh', "the tiles, the output: the pass's last 8 columns never written",
     '                if (8 * c >= qc) break;\n',
     '                if (8 * c >= qc - 8) break;\n'),
    ('step7.cuh', '7j, the tiles: the last box before column N left out',
     '                            if (c0w + 128 * q + 64 * b < N)\n',
     '                            if (c0w + 128 * q + 64 * b < N - 64)\n'),
    ('step7.cuh', "without 7j, the tiles: the copy out without the box's swizzle",
     'const uint8_t* src = sm + (zone - s0) + b * 8192 + ((c8 ^ (rr % 8)) << 4);',
     'const uint8_t* src = sm + (zone - s0) + b * 8192 + (c8 << 4);'),
    ('step7.cuh', 'without 7j, the tiles: the copy out on the first rows of each round only',
     '                    for (int i = 0; i < 64 / step; i++) {',
     '                    for (int i = 0; i < 1; i++) {'),
    ('step7.cuh', 'without 7i, the tiles: no cluster barrier at the end (a block may leave while the other still '
                  'arrives on its barriers)',
     '    if constexpr (kSh && !kTail) {   // without 7i: a cluster barrier before leaving, both blocks done with '
     'stages\n        cluster_arrive();\n        step4::cluster_wait();\n    }\n    if (kTma && ct == 0) '
     "step5::bulk_wait<0, false>();   // the stores done: their source is this block's shared memory\n}\ntemplate",
     "    if (kTma && ct == 0) step5::bulk_wait<0, false>();   // the stores done: their source is this block's "
     'shared memory\n}\ntemplate'),
    ('step7.cuh', "without 7g, the tiles: the descriptors rebuilt at the stage of the tile's own slice (t % NS)",
     "            const uint32_t st = s0 + s * T::kStage;   // without 7g: the stage's address",
     "            const uint32_t st = s0 + (t % NS) * T::kStage;   // without 7g: the stage's address"),
    ('step7.cuh', "without 7h, the tiles: a pair's empty barriers counting four arrivals (they never complete)",
     'step2::mbar_init(&empty[s], kSh ? 4 : 2);',
     'step2::mbar_init(&empty[s], kCl ? 4 : 2);'),
    ('step7.cuh', '7f, the bands: the loading warp does not wait for the empty stage',
     '            if (t >= NS) step2::mbar_wait(&empty[s], ((t / NS) - 1) & 1);\n',
     ''),
    ('step7.cuh', '7f, the bands: the loads issued by every lane of the loading warp (elect.sync left out)',
     '            if (step3::elect_one()) {\n                uint8_t* st = sm + s * G::kStage;\n                '
     'step2::mbar_arrive_expect_tx(&full[s], G::kStage);\n                if constexpr (kSh && P == kAlongN)',
     '            {\n                uint8_t* st = sm + s * G::kStage;\n                '
     'step2::mbar_arrive_expect_tx(&full[s], G::kStage);\n                if constexpr (kSh && P == kAlongN)'),
    ('step7.cuh', "7f, the bands: the output's named barrier counting the loading warp too (it has left)",
     '    step3::bar_named(1, 128);\n    if (tid == 0) {\n#pragma unroll\n        for (int j = 0; j < J; j++)   // '
     'rows >= M and columns >= N left out by the TMA',
     '    step3::bar_named(1, 160);\n    if (tid == 0) {\n#pragma unroll\n        for (int j = 0; j < J; j++)   // '
     'rows >= M and columns >= N left out by the TMA'),
    ('step7.cuh', "7g, the bands: the descriptors' offset from the slice's index instead of its stage",
     '(kUni ? __shfl_sync(0xffffffffu, s, 0) : s) * (G::kStage >> 4)',
     '(kUni ? __shfl_sync(0xffffffffu, t, 0) : s) * (G::kStage >> 4)'),
    ('step7.cuh', "without 7g, the bands: the descriptors' offset from the slice's index instead of its stage",
     '(kUni ? __shfl_sync(0xffffffffu, s, 0) : s) * (G::kStage >> 4)',
     '(kUni ? __shfl_sync(0xffffffffu, s, 0) : t) * (G::kStage >> 4)'),
    ('step7.cuh', "7h, the bands in clusters: their empty barriers counting one arrival (another block's loads may "
                  'overwrite a stage)',
     '            step2::mbar_init(&empty[s], kR);',
     '            step2::mbar_init(&empty[s], 1);'),
    ('step7.cuh', "7h, the bands in clusters: the release reaching one other block only (the others' loading warps "
                  'wait forever)',
     'for (int r = 0; r < kR - 1; r++) mbar_arrive_cluster(emptyOther[r] + 8 * (p % NS));',
     'for (int r = 0; r < 1 && r < kR - 1; r++) mbar_arrive_cluster(emptyOther[r] + 8 * (p % NS));'),
    ('step7.cuh', "7h, 7b: A multicast to the cluster's first four blocks only (the others wait for it forever)",
     'if (rank == 0) tma_load_2d_multicast(st, &tmA, t * 64, 0, &full[s], 0xff);',
     'if (rank == 0) tma_load_2d_multicast(st, &tmA, t * 64, 0, &full[s], 0x0f);'),
    ('step7.cuh', '7h, 7b: A loaded by every block of the cluster, each multicast to the eight (its bytes complete '
                  'eight times)',
     'if (rank == 0) tma_load_2d_multicast(st, &tmA, t * 64, 0, &full[s], 0xff);',
     'tma_load_2d_multicast(st, &tmA, t * 64, 0, &full[s], 0xff);'),
    ('step7.cuh', '7i, the bands in clusters: no wait of the loading warp for its last stages (a block may leave '
                  'while its multicast loads still land in another)',
     "        if constexpr (kSh && kTail)   // 7i: the last stages released by the cluster's computing warpgroups\n  "
     '          for (int p = nt > NS ? nt - NS : 0; p < nt; p++) step2::mbar_wait(&empty[p % NS], (p / NS) & 1);\n',
     ''),
    ('step7.cuh', "7i, the bands in clusters at 8 stages: the last stage not released (the loading warps' last wait "
                  'never returns)',
     "    if (!kZ && kSh && kTail) release(nt - 1);   // 7i: the last stage too, for the other blocks' loading "
     'warps\n',
     ''),
    ('step7.cuh', 'without 7i, the bands: no cluster barrier at the end (a block may leave while another still '
                  'arrives on its barriers)',
     '    if constexpr (kSh && !kTail) {   // without 7i: a cluster barrier before leaving, the cluster done with '
     'stages\n        cluster_arrive();\n        step4::cluster_wait();\n    }\n',
     ''),
    ('step7.cuh', "without 7h, the bands in clusters: the empty barriers counting the cluster's blocks (they never "
                  'complete)',
     '    constexpr int kR = kSh ? kC : 1;   // the blocks that release a stage',
     '    constexpr int kR = kPair ? kC : 1;   // the blocks that release a stage'),
    ('step7.cuh', '7l: the hinted loads of B one box to the right (the emulator runs the hinted path as a plain load)',
     '    (void)policy;\n    step2::tma_load_2d(dst, map, c0, c1, bar);\n#else',
     '    (void)policy;\n    step2::tma_load_2d(dst, map, c0 + 64, c1, bar);\n#else'),
    ('step7.cuh', '7l, 7a: the hinted multicast loads of B one box to the right',
     '    (void)policy;\n    tma_load_2d_multicast(dst, map, c0, c1, bar, mask);\n#else',
     '    (void)policy;\n    tma_load_2d_multicast(dst, map, c0 + 64, c1, bar, mask);\n#else'),
    ('step7.cuh', "7d, 7a: the band's rows 16 to MM stored over its first 16 (the passes of 16 rows on one place)",
     '            step6::stmatrix_x4_trans(zone + j * G::kBox + (16 * p + m) * 128 + ((c ^ (m % 8)) << 4),\n         '
     '                            step5::pack_half2(acc[j][8 * p], acc[j][8 * p + 1]),\n                             '
     '        step5::pack_half2(acc[j][8 * p + 2], acc[j][8 * p + 3]),\n                                     '
     'step5::pack_half2(acc[j][8 * p + 4], acc[j][8 * p + 5]),\n                                     '
     'step5::pack_half2(acc[j][8 * p + 6], acc[j][8 * p + 7]));\n    step1::fence_proxy_async();   // the zone, '
     'visible to the TMA\n    step3::bar_named(1, 128);\n    if (tid == 0) {\n#pragma unroll\n        for (int j = '
     '0; j < J; j++)   // rows >= M and columns >= N left out by the TMA',
     '            step6::stmatrix_x4_trans(zone + j * G::kBox + m * 128 + ((c ^ (m % 8)) << 4),\n                    '
     '                 step5::pack_half2(acc[j][8 * p], acc[j][8 * p + 1]),\n                                     '
     'step5::pack_half2(acc[j][8 * p + 2], acc[j][8 * p + 3]),\n                                     '
     'step5::pack_half2(acc[j][8 * p + 4], acc[j][8 * p + 5]),\n                                     '
     'step5::pack_half2(acc[j][8 * p + 6], acc[j][8 * p + 7]));\n    step1::fence_proxy_async();   // the zone, '
     'visible to the TMA\n    step3::bar_named(1, 128);\n    if (tid == 0) {\n#pragma unroll\n        for (int j = '
     '0; j < J; j++)   // rows >= M and columns >= N left out by the TMA'),
    ('step7.cuh', 'the rule, 7c: a grid of all the blocks held (not a whole number of groups)',
     '        R = G * (sms / G);\n',
     '        R = sms;\n'),
]


def check_substitutions():
    """Each bug's text must occur exactly once in its header; otherwise stop, loudly."""
    sources = {h: open(os.path.join(RUNG, h)).read() for h in HEADERS}
    stale = []
    for header, what, old, _ in BUGS:
        n = sources[header].count(old)
        if n != 1:
            stale.append(f"  {header}: '{what}': its text is found {n} times (exactly once expected)")
    if stale:
        print("planted-bug test: SUBSTITUTIONS NO LONGER MATCH THE SOURCE; bring them up to date in planted_bugs.py:")
        print("\n".join(stale))
        sys.exit(2)


def build_and_run(tmp, index, bug, timeout):
    """Copies the headers (the bug applied), builds the emulator against them, runs it; returns (code, output)."""
    d = os.path.join(tmp, f"{index:02d}")
    os.makedirs(d)
    for h in HEADERS:
        shutil.copy(os.path.join(RUNG, h), d)
    if bug:
        header, _, old, new = bug
        path = os.path.join(d, header)
        with open(path) as f:
            text = f.read()
        with open(path, "w") as f:
            f.write(text.replace(old, new, 1))
    exe = os.path.join(d, "emu")
    build = subprocess.run([CXX, "-std=c++17", "-O2", "-march=native", "-I" + d, "-I" + os.path.join(EMU, "shim"),
                            os.path.join(EMU, "emu.cpp"), "-o", exe], capture_output=True, text=True)
    if build.returncode:
        return "build failed", build.stderr
    args = [exe, "all"] if bug is None else [exe, RUN_BY.get(bug[1], STEP[bug[0]]), "-x"]
    try:
        run = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", ""
    return run.returncode, run.stdout


def first_failure(output):
    """The first FAILED run of an emulator output: its label, shape and order, without its reasons or the EMU ERROR
    lines, so that the test's output is the same at every run."""
    run = next((l for l in output.splitlines() if "  FAILED  " in l), "")
    return " ".join(run.partition("  FAILED  ")[0].split())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-j", type=int, default=min(8, os.cpu_count() or 1), help="parallel builds and runs")
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = parser.parse_args()
    check_substitutions()
    tmp = tempfile.mkdtemp(prefix="planted_bugs_")
    print(f"planted-bug test: {len(BUGS)} bugs, each in a copy of its header, the emulator built against each copy "
          f"({args.j} jobs, in a temporary directory)", flush=True)
    jobs = [None] + BUGS   # job 0: the unmodified copy
    with ThreadPoolExecutor(max_workers=args.j) as pool:
        futures = [pool.submit(build_and_run, tmp, i, bug, 10800 if bug is None else 7200) for i, bug in enumerate(jobs)]
        results = [f.result() for f in futures]
    ok = True
    code, output = results[0]
    if code == 0 and output.rstrip().endswith("ALL OK"):
        print("  passes    unmodified copy: " + next(l for l in output.splitlines() if " runs: " in l))
    else:
        ok = False
        print(f"  ERROR     unmodified copy: exit code {code}, it must pass\n{output[-2000:]}")
    reported = 0
    for (header, what, _, _), (code, output) in zip(BUGS, results[1:]):
        if code == 1 and "  FAILED  " in output:
            reported += 1
            print(f"  reported  {header}: {what}\n            first failed run: {first_failure(output)}")
        else:
            ok = False
            verdict = "MISSED  " if code == 0 else "ERROR   "
            detail = "the emulator passed it" if code == 0 else f"exit code {code}, not a reported failure"
            print(f"  {verdict}  {header}: {what}: {detail}\n{output[-2000:]}")
    print(f"{reported} of {len(BUGS)} planted bugs reported as failures; the unmodified copy "
          f"{'passes' if results[0][0] == 0 else 'does NOT pass'}")
    print("PLANTED-BUG TEST OK" if ok else "PLANTED-BUG TEST FAILED")
    if args.keep:
        print(f"kept: {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
