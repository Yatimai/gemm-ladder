#!/usr/bin/env python3
"""The planted-bug test of the Ampere rung's host emulator (README.md in this directory).

Each planted bug is one exact text substitution in a copy of a kernel header (step1.cuh to step7.cuh). The emulator,
built against that copy, runs the step the header belongs to and must report the bug as a failure: exit code 1 and a
FAILED run. The unmodified copy, built the same way, must pass every run (./emu all). A substitution whose text is not
found exactly once in its header stops the test before anything is built: the source has changed, and the substitution
must be brought up to date with it. The script is the Turing rung's (turing/emu/planted_bugs.py) with the Ampere
rung's headers and bugs. A bug that the step of its header does not run is run by the step that RUN_BY names.

    python3 ampere/emu/planted_bugs.py [-j JOBS] [--keep]

-j: builds and runs in parallel (default: the number of CPUs, at most 8); --keep: keep the temporary directory. The
headers in ampere/ are only read. Exit code 0 only if the unmodified copy passes and every planted bug is reported as a
failure.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

EMU = os.path.dirname(os.path.abspath(__file__))   # ampere/emu
RUNG = os.path.dirname(EMU)                         # ampere
HEADERS = ("nvidia_sample.cuh", "step1.cuh", "step2.cuh", "step3.cuh", "step4.cuh", "step5.cuh", "step6.cuh",
           "step7.cuh")
STEP = {"step1.cuh": "step1", "step2.cuh": "step2", "step3.cuh": "step3", "step4.cuh": "step4", "step5.cuh": "step5",
        "step6.cuh": "step6", "step7.cuh": "step7"}
CXX = os.environ.get("CXX", "g++")

# (header, the bug, exact text, its replacement). step1.cuh's double buffer (kDouble, the Turing rung's 1b) is not
# planted here: no variant of the Ampere rung runs it.
BUGS = [
    # step 7, 7c: its rule
    ("step7.cuh", "7c: the rule's waves 3/4 full, not 9/10 (o_m512 in 2 parts in place)",
     "return best && 10 * best * T >= 9 * bestWaves * sms ? best : 0;",
     "return best && 4 * best * T >= 3 * bestWaves * sms ? best : 0;"),
    ("step7.cuh", "7c: the rule without its 64 slices of K64 per part (o_m512 in 5 parts in place)",
     "for (int S = 2; S <= 8 && K / 64 / S >= 64; S++) {\n"
     "        const long waves = (S * T + sms - 1) / sms;",
     "for (int S = 2; S <= 8; S++) {\n"
     "        const long waves = (S * T + sms - 1) / sms;"),
    # step 6, 6b: the bands and the stream (the K steps of a segment, the columns, the parts, the last block, the rule)
    ("step6.cuh", "6b: a segment's K steps from 0, not from its first (the stream's segments overlap)",
     "const int kk = (k0 + k) * 64;",
     "const int kk = k * 64;"),
    ("step6.cuh", "6b: the band's columns left out of B's address (every band reads the first)",
     "&B[(size_t)(kk + r) * N + col0 + bc * 8]",
     "&B[(size_t)(kk + r) * N + bc * 8]"),
    ("step6.cuh", "6b: the warps' ldmatrix of B on half their columns",
     "bOff[p] = G::b_at(lane % 16, warpId * (G::kWarpCols / 8) + 2 * p + lane / 16);",
     "bOff[p] = G::b_at(lane % 16, warpId * (G::kWarpCols / 16) + 2 * p + lane / 16);"),
    ("step6.cuh", "6b: no barrier after reading the output (the next segment's copies overwrite it)",
     "    __syncthreads();   // the next segment's stages reuse the buffer\n",
     ""),
    ("step6.cuh", "6b stream: the first block to arrive sums (the other parts not yet written)",
     "*last = arrive(&counts[band]) == (unsigned)segs - 1;",
     "*last = arrive(&counts[band]) == 0u;"),
    ("step6.cuh", "6b stream: the arrival before the barrier (the other threads' parts not yet written)",
     "            fence();\n"
     "            __syncthreads();\n"
     "            // The flag, after the stages and the output: thread 0",
     "            fence();\n"
     "            // The flag, after the stages and the output: thread 0"),
    ("step6.cuh", "6b stream: the band's counter not put back to zero",
     "                if (threadIdx.x == 0) counts[band] = 0;   // for the next call\n",
     ""),
    ("step6.cuh", "6b stream: every segment of a band at the same place in the workspace",
     "half* part = parts + ((size_t)band * maxSegs + idx) * 16 * BN;",
     "half* part = parts + (size_t)band * maxSegs * 16 * BN;"),
    ("step6.cuh", "6b stream: the sum stops one segment short",
     "                    for (int sg = 0; sg < segs; sg++) {",
     "                    for (int sg = 0; sg < segs - 1; sg++) {"),
    ("step6.cuh", "6b rule: bands even when they exceed a wave",
     "    if (bands <= sms && (3 * bands >= 2 * sms",
     "    if ((3 * bands >= 2 * sms"),
    # step 6, 6b: the bands' own loop of 6 stages (and the stream's of 4): the stage indices, the wait, the copy's stage
    ("step6.cuh", "6b: the bands' write stage turns over kStages - 1 stages",
     "        write = write + 1 == kStages ? 0 : write + 1;\n"
     "    }\n"
     "    cp_async_wait<0>();",
     "        write = write + 1 >= kStages - 1 ? 0 : write + 1;\n"
     "    }\n"
     "    cp_async_wait<0>();"),
    ("step6.cuh", "6b: the bands' wait one group too loose (a stage read before its copies land)",
     "        cp_async_wait<kStages - 2>();\n"
     "        __syncthreads();\n"
     "        if (k + kStages - 1 < nk)",
     "        cp_async_wait<kStages - 1>();\n"
     "        __syncthreads();\n"
     "        if (k + kStages - 1 < nk)"),
    ("step6.cuh", "6b: the bands' next copy issued into the stage about to be read",
     "if (k + kStages - 1 < nk) issue(k + kStages - 1, write);",
     "if (k + kStages - 1 < nk) issue(k + kStages - 1, read);"),
    # step 6, 6b: the stream's flag just outside the launch's shared memory, at byte 139 280: a plain store, which only
    # the canary outside the launch shows
    ("step6.cuh", "6b: the stream's last-block flag just outside the launch's shared memory",
     "unsigned* last = (unsigned*)((char*)shmem + G::kFlagByte);",
     "unsigned* last = (unsigned*)((char*)shmem + G::kSmemBytes);"),
    # step 7, 7d, written in step6.cuh (run by step 7, RUN_BY): the discards of the parts (what a block drops, and when)
    ("step6.cuh", "7d: 7b discards the lines its slice only touches (a neighbour's chunks dropped before it reads them)",
     "const int l0 = (q0 + 7) / 8, l1 = q1 / 8;",
     "const int l0 = q0 / 8, l1 = (q1 + 7) / 8;"),
    ("step6.cuh", "7d: 6c discards the tile's parts before every thread has read them",
     "                if constexpr (kHint) {   // 7d: the tile's parts, every thread's reads done\n"
     "                    __syncthreads();\n",
     "                if constexpr (kHint) {   // 7d: the tile's parts, every thread's reads done\n"),
    ("step6.cuh", "7d: the stream of 6b discards the band's parts before every thread has read them",
     "                if constexpr (kHint) {   // 7d: the band's parts, every thread's reads done\n"
     "                    __syncthreads();\n",
     "                if constexpr (kHint) {   // 7d: the band's parts, every thread's reads done\n"),
    # step 6, 6c: the stream in phase (the slots, the rows of tiles, the last block, the ranges, the parts' K, the rule)
    ("step6.cuh", "6c: a block's two cut tiles share one slot (its last segment's part overwrites its first's)",
     "const int which = tn == q * U / P / steps ? 0 : 1;",
     "const int which = 0;"),
    ("step6.cuh", "6c: the row of tiles left out of the slot (the group's blocks overwrite each other's parts)",
     "return parts + (((size_t)q * 2 + which) * MT + tm) * BM * BN;",
     "return parts + ((size_t)q * 2 + which) * MT * BM * BN;"),
    ("step6.cuh", "6c: the last block found on the count of segments, not one less (no block adds the parts)",
     "*last = arrive(count) == (unsigned)(gLast - gFirst);",
     "*last = arrive(count) == (unsigned)(gLast - gFirst + 1);"),
    ("step6.cuh", "6c: the arrival before the part is computed (the last block adds parts not yet written)",
     "            half* part = slot(g, tn);\n"
     "            if (row0 + BM <= M)\n"
     "                compute_tile<BM, BN, WM, SK, kStages, kParts, false, Copy>(\n"
     "                    shmem, A, B, C, M, N, K, row0, col0, k0, nk, part, nullptr, 0, 1);\n"
     "            else\n"
     "                compute_tile<BM, BN, WM, SK, kStages, kParts, true, Copy>(\n"
     "                    shmem, A, B, C, M, N, K, row0, col0, k0, nk, part, nullptr, 0, 1);\n"
     "            const long gFirst = group_of((long)tn * steps), gLast = group_of(tileEnd - 1);\n"
     "            unsigned* count = counts + tm * tilesN + tn;\n"
     "            fence();   // this block's part, before its arrival\n"
     "            __syncthreads();\n"
     "            // The flag, after the stages and the output (as 6b's)\n"
     "            unsigned* last = (unsigned*)((char*)shmem + G::kSmemBytes);\n"
     "            if (threadIdx.x == 0) *last = arrive(count) == (unsigned)(gLast - gFirst);\n",
     "            half* part = slot(g, tn);\n"
     "            const long gFirst = group_of((long)tn * steps), gLast = group_of(tileEnd - 1);\n"
     "            unsigned* count = counts + tm * tilesN + tn;\n"
     "            // The flag, after the stages and the output (as 6b's)\n"
     "            unsigned* last = (unsigned*)((char*)shmem + G::kSmemBytes);\n"
     "            if (threadIdx.x == 0) *last = arrive(count) == (unsigned)(gLast - gFirst);\n"
     "            if (row0 + BM <= M)\n"
     "                compute_tile<BM, BN, WM, SK, kStages, kParts, false, Copy>(\n"
     "                    shmem, A, B, C, M, N, K, row0, col0, k0, nk, part, nullptr, 0, 1);\n"
     "            else\n"
     "                compute_tile<BM, BN, WM, SK, kStages, kParts, true, Copy>(\n"
     "                    shmem, A, B, C, M, N, K, row0, col0, k0, nk, part, nullptr, 0, 1);\n"
     "            fence();   // this block's part, before its arrival\n"
     "            __syncthreads();\n"),
    ("step6.cuh", "6c: the tile's counter not put back to zero",
     "                if (threadIdx.x == 0) *count = 0;   // for the next call\n",
     ""),
    ("step6.cuh", "6c: the ranges of U / P units rounded down (the last units of K computed by no group)",
     "const long u0 = g * U / P, u1 = (g + 1) * U / P;",
     "const long u0 = g * (U / P), u1 = (g + 1) * (U / P);"),
    ("step6.cuh", "6c: a cut tile's part from the first slice of K, not from its segment's",
     "kParts, false, Copy>(\n"
     "                    shmem, A, B, C, M, N, K, row0, col0, k0, nk, part, nullptr, 0, 1);",
     "kParts, false, Copy>(\n"
     "                    shmem, A, B, C, M, N, K, row0, col0, 0, nk, part, nullptr, 0, 1);"),
    ("step6.cuh", "6c: the rows >= M added and written",
     "                    if (row0 + r >= M) continue;\n",
     ""),
    ("step6.cuh", "6c: every block of a group on the first row of tiles",
     "tilesN = N / BN, tm = blockIdx.x % MT, row0",
     "tilesN = N / BN, tm = 0, row0"),
    ("step6.cuh", "6c: the sum of the parts without the first group's",
     "for (long sg = gFirst; sg <= gLast; sg++) {",
     "for (long sg = gFirst + 1; sg <= gLast; sg++) {"),
    ("step6.cuh", "6c: the rule's last wave below the SMs, not below half of them (qkv_m512 in the stream)",
     "return tiles >= 2L * sms && rest > 0 && 2 * rest < sms;\n"
     "}\n"
     "\n"
     "// ---------------------------------------------------------------- 7b: K in parts, reduced by their own blocks",
     "return tiles >= 2L * sms && rest > 0 && rest < sms;\n"
     "}\n"
     "\n"
     "// ---------------------------------------------------------------- 7b: K in parts, reduced by their own blocks"),
    # step 7, 7b, written in step6.cuh (run by step 7, RUN_BY): the parts reduced by their own blocks (the wait, the
    # slices, the counters, the barrier, residency)
    ("step6.cuh", "7b: the wait on S - 1 arrivals (a block reduces before the last part has arrived)",
     "wait_count(arrived, (unsigned)S);",
     "wait_count(arrived, (unsigned)S - 1);"),
    ("step6.cuh", "7b: the wait removed (the parts read before they arrive)",
     "        arrive(arrived);\n"
     "        wait_count(arrived, (unsigned)S);\n",
     "        arrive(arrived);\n"),
    ("step6.cuh", "7b: the wait on S + 1 arrivals (no block of the tile goes on: a hang)",
     "wait_count(arrived, (unsigned)S);",
     "wait_count(arrived, (unsigned)S + 1);"),
    ("step6.cuh", "7b: the arrival at the block's start (the S parts reduced before they are all written)",
     "    run<BM, BN, WM, SK, kStages, kParts, Copy>(shmem, A, B, C, M, N, K, S, parts, nullptr);\n"
     "    const int tilesN = N / BN, tiles = (M + BM - 1) / BM * tilesN;\n"
     "    const int s = blockIdx.x / tiles;\n"
     "    int tm, tn;\n"
     "    tile_of<BM, BN>(blockIdx.x % tiles, M, N, tm, tn);\n"
     "    const int t = tm * tilesN + tn, row0 = tm * BM, col0 = tn * BN;\n"
     "    unsigned* arrived = counts + 2 * t;\n"
     "    unsigned* departed = arrived + 1;\n"
     "    fence();   // this block's part, before its arrival\n"
     "    __syncthreads();\n"
     "    if (threadIdx.x == 0) {\n"
     "        arrive(arrived);\n"
     "        wait_count(arrived, (unsigned)S);\n"
     "    }\n",
     "    const int tilesN = N / BN, tiles = (M + BM - 1) / BM * tilesN;\n"
     "    const int s = blockIdx.x / tiles;\n"
     "    int tm, tn;\n"
     "    tile_of<BM, BN>(blockIdx.x % tiles, M, N, tm, tn);\n"
     "    const int t = tm * tilesN + tn, row0 = tm * BM, col0 = tn * BN;\n"
     "    unsigned* arrived = counts + 2 * t;\n"
     "    unsigned* departed = arrived + 1;\n"
     "    if (threadIdx.x == 0) arrive(arrived);\n"
     "    run<BM, BN, WM, SK, kStages, kParts, Copy>(shmem, A, B, C, M, N, K, S, parts, nullptr);\n"
     "    fence();   // this block's part, before its arrival\n"
     "    __syncthreads();\n"
     "    if (threadIdx.x == 0) wait_count(arrived, (unsigned)S);\n"),
    ("step6.cuh", "7b: the slices of 1 / S rounded down (the tile's last chunks reduced by no block)",
     "const int q0 = s * kChunks / S, q1 = (s + 1) * kChunks / S;",
     "const int q0 = s * (kChunks / S), q1 = q0 + kChunks / S;"),
    ("step6.cuh", "7b: the slice of the block index divided by S, not of the part (the wrong rows reduced)",
     "const int q0 = s * kChunks / S, q1 = (s + 1) * kChunks / S;",
     "const int q0 = (int)(blockIdx.x / S) * kChunks / S, q1 = ((int)(blockIdx.x / S) + 1) * kChunks / S;"),
    ("step6.cuh", "7b: the arrival counter not put back to zero",
     "        *arrived = 0;\n",
     ""),
    ("step6.cuh", "7b: the departure counter not put back to zero",
     "        *departed = 0;\n",
     ""),
    ("step6.cuh", "7b: the first block to depart puts the counters back to zero (the others still wait on them)",
     "arrive(departed) == (unsigned)S - 1",
     "arrive(departed) == 0u"),
    ("step6.cuh", "7b: no barrier after the wait (the block's threads reduce before the S parts have arrived)",
     "    __syncthreads();\n"
     "    fence();\n"
     "    constexpr int kChunks",
     "    fence();\n"
     "    constexpr int kChunks"),
    ("step6.cuh", "7b: residency at twice the SMs (S x tiles blocks that cannot all be resident: a hang)",
     "return (long)S * tiles <= sms;",
     "return (long)S * tiles <= 2L * sms;"),
    # step 6: its own loop at kStages stages (6a's five of K32, the parts' three of K64): the stage
    # indices, the wait, the copy's stage
    ("step6.cuh", "6: the write stage turns over kStages - 1 stages",
     "                write = write + 1 == kStages ? 0 : write + 1;",
     "                write = write + 1 >= kStages - 1 ? 0 : write + 1;"),
    ("step6.cuh", "6: the wait in the loop one group too loose (a stage read before its copies land)",
     "                cp_async_wait<kStages - 2>();\n"
     "                __syncthreads();\n"
     "                read =",
     "                cp_async_wait<kStages - 1>();\n"
     "                __syncthreads();\n"
     "                read ="),
    ("step6.cuh", "6: the next copy issued into the stage being read",
     "if (k + kStages - 1 < steps) issue(k + kStages - 1, write);",
     "if (k + kStages - 1 < steps) issue(k + kStages - 1, read);"),
    # step 7, 7a, written in step6.cuh (run by step 7, RUN_BY): its parts (their places, the share of K, the reduction),
    # its in-place flags, the plan; then step 6: 6a's geometry, the order of the tiles
    ("step6.cuh", "7a parts: every part of a tile at the same place in the workspace",
     "parts + ((size_t)s * tiles + t) * BM * BN",
     "parts + (size_t)t * BM * BN"),
    ("step6.cuh", "7a parts: the reduction sums one part too few",
     "        for (int s = 0; s < S; s++) {\n"
     "            const int4 v",
     "        for (int s = 0; s < S - 1; s++) {\n"
     "            const int4 v"),
    ("step6.cuh", "7a: the parts' first slices without the extra ones before them (parts overlap, a slice is left out)",
     "const int kBegin = s * per + min(s, extra),",
     "const int kBegin = s * per,"),
    ("step6.cuh", "7a: the first parts without their extra slice",
     "    const int kBegin = s * per + min(s, extra), steps = per + (s < extra);",
     "    const int kBegin = s * per + min(s, extra), steps = per;"),
    ("step6.cuh", "7a parts: the reduction writes rows >= M",
     "        const int r = q / (BN / 8), ch = q % (BN / 8);\n"
     "        if (row0 + r >= M) continue;\n",
     "        const int r = q / (BN / 8), ch = q % (BN / 8);\n"),
    ("step6.cuh", "7a in place: no wait on the tile's flag",
     "        if (threadIdx.x == 0) wait_flag(flag, (unsigned)s);\n",
     ""),
    ("step6.cuh", "7a in place: the previous parts not added",
     "                    if (s > 0) v = add8(v, ld16_cg(dst));\n",
     ""),
    ("step6.cuh", "7a in place: the last part does not put the flag back to zero",
     "set_flag(flag, s + 1 == S ? 0u : (unsigned)s + 1);",
     "set_flag(flag, (unsigned)s + 1);"),
    ("step6.cuh", "7a in place: the flag set as soon as its wait returns (the next part adds before this one's stores)",
     "        if (threadIdx.x == 0) wait_flag(flag, (unsigned)s);\n",
     "        if (threadIdx.x == 0) {\n"
     "            wait_flag(flag, (unsigned)s);\n"
     "            set_flag(flag, s + 1 == S ? 0u : (unsigned)s + 1);\n"
     "        }\n"),
    ("step6.cuh", "7a plan: K in 2 parts instead of 3",
     "            p.S = 3;\n",
     "            p.S = 2;\n"),
    ("step6.cuh", "7a plan: the 128-row tile kept at 128 x 128 when K is split into parts",
     "            p.tile = k128x256;\n",
     ""),
    ("step6.cuh", "6a: A's rows of 64 bytes swizzled on rows, not pairs of rows (right values, LDSM conflicts)",
     "SK == 64 ? c ^ (r & 7) : c ^ ((r >> 1) & 3)",
     "SK == 64 ? c ^ (r & 7) : c ^ (r & 3)"),
    ("step6.cuh", "tile_of: the rows of tiles counted without the one cut by M",
     "tilesM = (M + BM - 1) / BM, per = R * tilesN;",
     "tilesM = M / BM, per = R * tilesN;"),
    # step 5: B's stages 16 KB further, the last one ends after the launch's 147 456 bytes (its copies and ldmatrix)
    ("step5.cuh", "5: B's stages 16 KB further (the last ends after the launch's shared memory)",
     "half* Bs = shmem + Smem::kAAll;   // kStages x 64 rows of a K64 slice of B",
     "half* Bs = shmem + Smem::kAAll + 8192;   // kStages x 64 rows of a K64 slice of B"),
    # step 5: 5a's pipeline of the fragments (the barrier in the middle of K16 3, the sets, the issue), 5b's order
    ("step5.cuh", "5a: no barrier in the middle of K16 3 (slice k + 1 read before every thread's copies are visible)",
     "                    cp_async_wait<kStages - 2>();\n"
     "                    __syncthreads();\n",
     "                    cp_async_wait<kStages - 2>();\n"),
    ("step5.cuh", "5a: the first K16 of slice k + 1 loaded before the wait and the barrier",
     "                    cp_async_wait<kStages - 2>();\n"
     "                    __syncthreads();\n"
     "                    read = read + 1 == kStages ? 0 : read + 1;\n"
     "                    write = write + 1 == kStages ? 0 : write + 1;\n"
     "                    if (k + 1 < steps) load_frags(cur ^ 1, read, 0);\n",
     "                    read = read + 1 == kStages ? 0 : read + 1;\n"
     "                    write = write + 1 == kStages ? 0 : write + 1;\n"
     "                    if (k + 1 < steps) load_frags(cur ^ 1, read, 0);\n"
     "                    cp_async_wait<kStages - 2>();\n"
     "                    __syncthreads();\n"),
    ("step5.cuh", "5a: the next K16's fragments loaded into the set being used",
     "                if (ks < 3) load_frags(cur ^ 1, read, ks + 1);",
     "                if (ks < 3) load_frags(cur, read, ks + 1);"),
    ("step5.cuh", "5a: slice k + 2 issued into the stage being read",
     "                    if (k + kStages - 1 < steps) issue((k + kStages - 1) * 64, write);",
     "                    if (k + kStages - 1 < steps) issue((k + kStages - 1) * 64, read);"),
    ("step5.cuh", "5a: the prologue's first K16 loaded before the wait and the barrier",
     "        cp_async_wait<kStages - 2>();\n"
     "        __syncthreads();\n"
     "        load_frags(0, 0, 0);\n",
     "        load_frags(0, 0, 0);\n"
     "        cp_async_wait<kStages - 2>();\n"
     "        __syncthreads();\n"),
    ("step5.cuh", "5a: the second half of K16 3's HMMA left out",
     "                    mma_range(cur, 16, 32);\n",
     ""),
    ("step5.cuh", "5a: the write stage never advances",
     "                    write = write + 1 == kStages ? 0 : write + 1;\n",
     ""),
    ("step5.cuh", "5b: the serpentine pairs its B halves (some never used, others twice)",
     "q = kSerp && (i & 1) ? 7 - h % 8 : h % 8,",
     "q = kSerp && (i & 1) ? (7 - h % 8) / 2 * 2 : h % 8,"),
    ("step5.cuh", "without 5a: no barrier after the wait (step 4's loop)",
     "            cp_async_wait<kStages - 2>();\n"
     "            __syncthreads();\n"
     "            if (k + kStages - 1 < steps)",
     "            cp_async_wait<kStages - 2>();\n"
     "            if (k + kStages - 1 < steps)"),
    ("step5.cuh", "no barrier before the output reuses the stages",
     "    cp_async_wait<0>();   // only empty groups are left\n"
     "    __syncthreads();",
     "    cp_async_wait<0>();   // only empty groups are left"),
    # step 4: the order of the tiles (4b), the stores of C (4a), the epilogue
    ("step4.cuh", "4b: the last group taken as full (tiles outside the grid when the rows of tiles run out)",
     "const int rows = min(tilesM - first, R), r = b % per;",
     "const int rows = R, r = b % per;"),
    ("step4.cuh", "4b: the column of a tile from the remainder (two blocks on one tile)",
     "        tn = r / rows;\n"
     "    } else {",
     "        tn = r % rows;\n"
     "    } else {"),
    ("step4.cuh", "4b: a group's first row without its R rows per group",
     "per = R * tilesN, first = b / per * R;",
     "per = R * tilesN, first = b / per;"),
    ("step4.cuh", "4b: the rows of tiles counted without the one cut by M",
     "const int R = kGroupRows, tilesM = (M + 127) / 128,",
     "const int R = kGroupRows, tilesM = M / 128,"),
    ("step4.cuh", "the tile's columns from its row of tiles",
     "const int row0 = tm * 128, col0 = tn * 256;",
     "const int row0 = tm * 128, col0 = tm * 256;"),
    ("step4.cuh", "4a: C stored without the bound gr < M (edge tiles write rows >= M)",
     "            if (gr < M)\n"
     "                st16<kEf>(",
     "                st16<kEf>("),
    ("step4.cuh", "the write stage never advances (seen from 4 slices of K64 on)",
     "        write = write + 1 == kStages ? 0 : write + 1;\n",
     ""),
    ("step4.cuh", "no barrier before the output reuses the stages",
     "    cp_async_wait<0>();   // only empty groups are left\n"
     "    __syncthreads();",
     "    cp_async_wait<0>();   // only empty groups are left"),
    # step 3: the cp.async pipeline (waits, commits, barriers, stages) and its copy
    ("step3.cuh", "3b: the wait one group too loose (slice k may not have landed)",
     "            cp_async_wait<kStages - 2>();\n",
     "            cp_async_wait<kStages - 1>();\n"),
    ("step3.cuh", "3b: no barrier after the wait (the other threads' copies of slice k not yet visible)",
     "            cp_async_wait<kStages - 2>();\n"
     "            __syncthreads();\n",
     "            cp_async_wait<kStages - 2>();\n"),
    ("step3.cuh", "3b: slice k + kStages - 1 issued into the stage being read",
     "if (k + kStages - 1 < steps) issue((k + kStages - 1) * 64, write);",
     "if (k + kStages - 1 < steps) issue((k + kStages - 1) * 64, read);"),
    ("step3.cuh", "3b: slice k + kStages - 1 issued before the barrier (into the stage another warp may still read)",
     "            cp_async_wait<kStages - 2>();\n"
     "            __syncthreads();\n"
     "            if (k + kStages - 1 < steps) issue((k + kStages - 1) * 64, write);\n",
     "            cp_async_wait<kStages - 2>();\n"
     "            if (k + kStages - 1 < steps) issue((k + kStages - 1) * 64, write);\n"
     "            __syncthreads();\n"),
    ("step3.cuh", "3b: the compute reads the stage being written",
     "            compute(read);\n",
     "            compute(write);\n"),
    ("step3.cuh", "3b: the read stage never advances",
     "            read = read + 1 == kStages ? 0 : read + 1;\n",
     ""),
    ("step3.cuh", "3b: the write stage never advances (seen from 4 slices of K64 on)",
     "            write = write + 1 == kStages ? 0 : write + 1;\n",
     ""),
    ("step3.cuh", "3b: the prologue issues one slice too few",
     "        for (int s = 0; s < kStages - 1; s++) {",
     "        for (int s = 0; s < kStages - 2; s++) {"),
    ("step3.cuh", "3b: no commit in the loop (the copies of the next slices never form a group)",
     "            if (k + kStages - 1 < steps) issue((k + kStages - 1) * 64, write);\n"
     "            cp_async_commit();\n",
     "            if (k + kStages - 1 < steps) issue((k + kStages - 1) * 64, write);\n"),
    ("step3.cuh", "3b: a slice issued outside K (reads after B's last row)",
     "if (k + kStages - 1 < steps) issue(",
     "if (k + kStages - 1 <= steps) issue("),
    ("step3.cuh", "3b: no barrier before the output reuses the stages",
     "        cp_async_wait<0>();   // only empty groups are left\n"
     "        __syncthreads();",
     "        cp_async_wait<0>();   // only empty groups are left"),
    ("step3.cuh", "3a: no wait before the barrier (the copies land who knows when)",
     "            cp_async_commit();\n"
     "            cp_async_wait<0>();\n"
     "            __syncthreads();\n"
     "            compute(0);",
     "            cp_async_commit();\n"
     "            __syncthreads();\n"
     "            compute(0);"),
    ("step3.cuh", "3a: no barrier after the compute (the next slice's copies overwrite a stage still read)",
     "            compute(0);\n"
     "            __syncthreads();\n",
     "            compute(0);\n"),
    ("step3.cuh", "A's source without k0 (the first K64 of A copied at every step)",
     "&A[(size_t)gr * K + k0 + 8 * (threadIdx.x % 8)]",
     "&A[(size_t)gr * K + 8 * (threadIdx.x % 8)]"),
    ("step3.cuh", "B's source chunk from the lane in A's layout (t % 8 for t % 32)",
     "&B[(size_t)(k0 + r) * N + col0 + 8 * (threadIdx.x % 32)]",
     "&B[(size_t)(k0 + r) * N + col0 + 8 * (threadIdx.x % 8)]"),
    ("step3.cuh", "A copied without the swizzle (read with it)",
     "cp_async16(&As[s * G::kAStage + G::a_at(r, threadIdx.x % 8)]",
     "cp_async16(&As[s * G::kAStage + r * G::kAStride + 8 * (threadIdx.x % 8)]"),
    ("step3.cuh", "B's stage stride of A's (the stages of B overlap)",
     "        const half* bs = Bs + s * G::kBStage;\n",
     "        const half* bs = Bs + s * G::kAStage;\n"),
    ("step3.cuh", "no M - 1 clamp of the rows of A (edge tiles)",
     "const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;",
     "const int gr = row0 + r;"),
    # step 2: each barrier removed in turn
    ("step2.cuh", "no barrier between the copy and the compute",
     "        store();\n"
     "        __syncthreads();\n"
     "        compute();\n",
     "        store();\n"
     "        compute();\n"),
    ("step2.cuh", "no barrier after the compute (the next slice's stores race with the reads)",
     "        compute();\n"
     "        __syncthreads();\n"
     "    }\n",
     "        compute();\n"
     "    }\n"),
    ("step2.cuh", "no barrier between the epilogue's STS.32 and its copy out",
     "        __syncthreads();\n"
     "#pragma unroll\n"
     "        for (int i = 0; i < 8; i++) {",
     "#pragma unroll\n"
     "        for (int i = 0; i < 8; i++) {"),
    ("step2.cuh", "no barrier after the epilogue's copy out of a half tile",
     "threadIdx.x % 32);\n"
     "        }\n"
     "        __syncthreads();\n"
     "    }\n",
     "threadIdx.x % 32);\n"
     "        }\n"
     "    }\n"),
    # step 2: the registers of mma.sync and ldmatrix
    ("step2.cuh", "B's two mma registers swapped (K rows 0-7 and 8-15)",
     "mma_m16n8k16(c[i][j][n], a[i], b[j][2 * n], b[j][2 * n + 1]);",
     "mma_m16n8k16(c[i][j][n], a[i], b[j][2 * n + 1], b[j][2 * n]);"),
    ("step2.cuh", "B's n8 halves swapped",
     "mma_m16n8k16(c[i][j][n], a[i], b[j][2 * n], b[j][2 * n + 1]);",
     "mma_m16n8k16(c[i][j][n], a[i], b[j][2 - 2 * n], b[j][3 - 2 * n]);"),
    ("step2.cuh", "A's ldmatrix quarters in the wrong order (a[1] and a[2] exchanged)",
     "aOff[ks] = G::a_at(wm * 64 + lane % 16, 2 * ks + lane / 16);",
     "aOff[ks] = G::a_at(wm * 64 + lane % 8 + 8 * (lane / 16), 2 * ks + lane / 8 % 2);"),
    ("step2.cuh", "A's rows 8-15 read as rows 0-7 (lane % 8 for lane % 16)",
     "aOff[ks] = G::a_at(wm * 64 + lane % 16, 2 * ks + lane / 16);",
     "aOff[ks] = G::a_at(wm * 64 + lane % 8, 2 * ks + lane / 16);"),
    ("step2.cuh", "B read without .trans",
     "ldmatrix_x4<true>(b[j]",
     "ldmatrix_x4<false>(b[j]"),
    ("step2.cuh", "A read with .trans",
     "ldmatrix_x4<false>(a[i]",
     "ldmatrix_x4<true>(a[i]"),
    # step 2: the swizzle and the offsets
    ("step2.cuh", "A read without the swizzle (ldmatrix rows of A)",
     "aOff[ks] = G::a_at(wm * 64 + lane % 16, 2 * ks + lane / 16);",
     "aOff[ks] = (wm * 64 + lane % 16) * G::kAStride + 8 * (2 * ks + lane / 16);"),
    ("step2.cuh", "A stored without the swizzle",
     "*(int4*)&As[G::a_at(threadIdx.x / 8 + i * 32, threadIdx.x % 8)] = ra[i];",
     "*(int4*)&As[(threadIdx.x / 8 + i * 32) * G::kAStride + 8 * (threadIdx.x % 8)] = ra[i];"),
    ("step2.cuh", "B read without the swizzle",
     "bOff[j] = G::b_at(lane % 16, wn * 8 + 2 * j + lane / 16);",
     "bOff[j] = (lane % 16) * G::kBStride + 8 * (wn * 8 + 2 * j + lane / 16);"),
    ("step2.cuh", "B stored without the swizzle",
     "*(int4*)&Bs[G::b_at(threadIdx.x / 32 + i * 8, threadIdx.x % 32)] = rb[i];",
     "*(int4*)&Bs[(threadIdx.x / 32 + i * 8) * G::kBStride + 8 * (threadIdx.x % 32)] = rb[i];"),
    ("step2.cuh", "B's swizzle on two bits on both sides (right values, two-way LDSM conflicts)",
     "        return r * kBStride + 8 * (kSwizzle ? c ^ (r & 7) : c);",
     "        return r * kBStride + 8 * (kSwizzle ? c ^ (r & 3) : c);"),
    ("step2.cuh", "A's swizzle on rows by pairs on both sides (right values, two-way LDSM conflicts)",
     "        return r * kAStride + 8 * (kSwizzle ? c ^ (r & 7) : c);",
     "        return r * kAStride + 8 * (kSwizzle ? c ^ ((r >> 1) & 7) : c);"),
    ("step2.cuh", "v1's padding of 8 halves (its stated two-way LDSM pattern no longer holds)",
     "constexpr int kSkew = 16;",
     "constexpr int kSkew = 8;"),
    ("step2.cuh", "one A offset for the four K16 steps",
     "ldmatrix_x4<false>(a[i], &As[aOff[ks] + i * 16 * G::kAStride]);",
     "ldmatrix_x4<false>(a[i], &As[aOff[0] + 16 * ks + i * 16 * G::kAStride]);"),
    ("step2.cuh", "B's K16 step left out (the first K16 of B read four times)",
     "&Bs[bOff[j] + ks * 16 * G::kBStride]",
     "&Bs[bOff[j]]"),
    # step 2: the copy, the loop, the tiles, the epilogue
    ("step2.cuh", "A's slice without its k0 (the first K64 of A read at every step)",
     "&A[(size_t)gr * K + k0]",
     "&A[(size_t)gr * K]"),
    ("step2.cuh", "B's slice without its k0",
     "&B[(size_t)(k0 + r) * N + col0]",
     "&B[(size_t)r * N + col0]"),
    ("step2.cuh", "K by steps of 128 (one slice in two left out)",
     "for (int k0 = 0; k0 < K; k0 += 64) {",
     "for (int k0 = 0; k0 < K; k0 += 128) {"),
    ("step2.cuh", "the tile's row from the column index (tiles mixed up)",
     "const int row0 = (blockIdx.x / tilesN) * 128, col0",
     "const int row0 = (blockIdx.x % tilesN) * 128, col0"),
    ("step2.cuh", "no M - 1 clamp of the rows of A (edge tiles)",
     "const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;",
     "const int gr = row0 + r;"),
    ("step2.cuh", "C stored without the bound gr < M (edge tiles write rows >= M)",
     "            if (gr < M)\n"
     "                *((int4*)&C",
     "                *((int4*)&C"),
    ("step2.cuh", "wrong epilogue layout: D[g][2t + 1] taken from D[g + 8][2t]",
     "__floats2half2_rn(c[i][j][n][0], c[i][j][n][1])",
     "__floats2half2_rn(c[i][j][n][0], c[i][j][n][2])"),
    ("step2.cuh", "the epilogue's rows g + 8 written 4 rows down",
     "*(__half2*)(o + 8 * G::kOStride)",
     "*(__half2*)(o + 4 * G::kOStride)"),
    ("step2.cuh", "only the first half tile's warps store their accumulators",
     "        if (wm == h) {",
     "        if (wm == 0) {"),
    ("step2.cuh", "the second half tile copied to the first's rows of C",
     "const int gr = row0 + h * 64 + r;",
     "const int gr = row0 + r;"),
    # step 1 (the Ampere rung runs its single-stage loop only): each barrier removed in turn, and the edge
    ("step1.cuh", "no barrier between the copy and the compute",
     "            store(0);\n"
     "            __syncthreads();\n"
     "            compute(0);\n",
     "            store(0);\n"
     "            compute(0);\n"),
    ("step1.cuh", "no barrier after the compute",
     "            compute(0);\n"
     "            __syncthreads();\n"
     "        }\n",
     "            compute(0);\n"
     "        }\n"),
    ("step1.cuh", "no barrier between the epilogue's fragment stores and its copy out",
     "        __syncthreads();\n"
     "#pragma unroll\n"
     "        for (int i = 0; i < 64 * G::kBLanes / 256; i++) {",
     "#pragma unroll\n"
     "        for (int i = 0; i < 64 * G::kBLanes / 256; i++) {"),
    ("step1.cuh", "no barrier after the epilogue's copy out of a half tile",
     "threadIdx.x % G::kBLanes);\n"
     "        }\n"
     "        __syncthreads();\n"
     "    }\n",
     "threadIdx.x % G::kBLanes);\n"
     "        }\n"
     "    }\n"),
    ("step1.cuh", "that barrier only after the first half tile (a persistent block's next tile races)",
     "threadIdx.x % G::kBLanes);\n"
     "        }\n"
     "        __syncthreads();\n"
     "    }\n",
     "threadIdx.x % G::kBLanes);\n"
     "        }\n"
     "        if (h == 0) __syncthreads();\n"
     "    }\n"),
    ("step1.cuh", "no M - 1 clamp of the rows of A (edge tiles)",
     "const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;",
     "const int gr = row0 + r;"),
    ("step1.cuh", "the persistent grid steps by one tile (blocks share tiles)",
     "block_pos += gridDim.x) tile(block_pos);",
     "block_pos += 1) tile(block_pos);"),
]

# 7a, 7b and 7d, step 7's, written in step6.cuh beside the kernels that carry them (step 6 runs none of 7a's and 7b's
# kernels, and those of 6b and 6c without 7d's hints): their bugs, each named from its letter, are run by step 7.
RUN_BY = {what: "step7" for header, what, _, _ in BUGS if header == "step6.cuh" and what.startswith(("7a", "7b", "7d"))}


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
    """The first FAILED run of an emulator output: its label, shape and order (for a check, its name and range), without
    its reasons or the EMU ERROR lines, so that the test's output is the same at every run."""
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
        futures = [pool.submit(build_and_run, tmp, i, bug, 3600 if bug is None else 1800) for i, bug in enumerate(jobs)]
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
