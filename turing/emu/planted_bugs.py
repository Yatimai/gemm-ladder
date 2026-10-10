#!/usr/bin/env python3
"""The planted-bug test of the host emulator (README.md in this directory).

Each planted bug is one exact text substitution in a copy of a kernel header (step1.cuh to step6.cuh). The emulator,
built against that copy, runs the step the header belongs to and must report the bug as a failure: exit code 1 and a
FAILED run. The unmodified copy, built the same way, must pass every run (./emu all). A substitution whose
text is not found exactly once in its header stops the test before anything is built: the source has changed, and the
substitution must be brought up to date with it.

    python3 turing/emu/planted_bugs.py [-j JOBS] [--keep]

-j: builds and runs in parallel (default: the number of CPUs); --keep: keep the temporary directory. The headers in
turing/ are only read. Exit code 0 only if the unmodified copy passes and every planted bug is reported as a failure.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

EMU = os.path.dirname(os.path.abspath(__file__))   # turing/emu
TURING = os.path.dirname(EMU)                       # turing
HEADERS = ("nvidia_sample.cuh", "step1.cuh", "step2.cuh", "step3.cuh", "step4.cuh", "step5.cuh", "step6.cuh")
STEP = {"step1.cuh": "step1", "step2.cuh": "step2", "step3.cuh": "step3", "step4.cuh": "step4", "step5.cuh": "step5",
        "step6.cuh": "step6"}
CXX = os.environ.get("CXX", "g++")

# (header, the bug, exact text, its replacement)
BUGS = [
    # step 6: 6c's bands (the counters, the last block, the parts, the slices of K, the bounds in M, the barriers)
    ("step6.cuh", "6c: the band's counter not put back to zero by the last block",
     "        if (threadIdx.x == 0) counts[b] = 0;   // for the next call\n",
     ""),
    ("step6.cuh", "6c: the first block to arrive taken as the last (it sums parts not yet written)",
     "*last = arrive(&counts[b]) == (unsigned)slices - 1;",
     "*last = arrive(&counts[b]) == 0u;"),
    ("step6.cuh", "6c: a block's part at its band's index, not its own (the slices of a band overwrite each other)",
     "half* part = parts + (size_t)(first + s) * 16 * BN;",
     "half* part = parts + (size_t)b * 16 * BN;"),
    ("step6.cuh", "6c: the sum stops one slice short",
     "for (int ss = 0; ss < slices; ss++) {",
     "for (int ss = 0; ss < slices - 1; ss++) {"),
    ("step6.cuh", "6c: a slice's first step without the longer slices before it (K's steps overlap, one is left out)",
     "const int k0 = s * perSlice + min(s, longer),",
     "const int k0 = s * perSlice,"),
    ("step6.cuh", "6c: the longer slices without their extra step",
     "nk = perSlice + (s < longer);",
     "nk = perSlice;"),
    ("step6.cuh", "6c: A's rows past M loaded (reads past A's last row)",
     "aLoads = aStores && ar < M;",
     "aLoads = aStores;"),
    ("step6.cuh", "6c: C's rows past M written (one slice)",
     "if (r < M) step3::st16<true>((int4*)&C[(size_t)r * N + col0] + ch % G::kBChunks, mine[u]);",
     "step3::st16<true>((int4*)&C[(size_t)r * N + col0] + ch % G::kBChunks, mine[u]);"),
    ("step6.cuh", "6c: the copy's test i + 3 < nk removed (loads past the slice's last step)",
     "        if (i + 3 < nk) fetch(ra[1 - st], rb[1 - st]);\n",
     "        fetch(ra[1 - st], rb[1 - st]);\n"),
    ("step6.cuh", "6c: no barrier in the loop (between a stage's stores and the next step's reads)",
     "        if (i + 1 < nk) store(1 - st, 1 - st);\n        __syncthreads();\n",
     "        if (i + 1 < nk) store(1 - st, 1 - st);\n"),
    ("step6.cuh", "6c: no barrier after the prologue's stores",
     "    rb[0][1] = rb2[1];\n    __syncthreads();\n",
     "    rb[0][1] = rb2[1];\n"),
    ("step6.cuh", "6c: no barrier between the output's fragment stores and their copy",
     "    __syncthreads();\n    int4 mine[G::kOPer];\n",
     "    int4 mine[G::kOPer];\n"),
    # step 6: 6d's K in two halves
    ("step6.cuh", "6d: the tile's counter not put back to zero by the second block",
     "    if (threadIdx.x == 0) counts[t] = 0;   // for the next call\n",
     ""),
    ("step6.cuh", "6d: the first block to arrive sums (the other part not yet written)",
     "*last = arrive(&counts[t]) == 1u;",
     "*last = arrive(&counts[t]) == 0u;"),
    ("step6.cuh", "6d: both blocks over the whole of K (C twice the product)",
     "t, s * halfK, (s + 1) * halfK);",
     "t, 0, K);"),
    ("step6.cuh", "6d: both slices into slice 0's part",
     "parts + (size_t)s * M * N, M, N, K,",
     "parts, M, N, K,"),
    ("step6.cuh", "6d: the sum reads slice 1's part twice",
     "ld16_part((const int4*)&parts[(size_t)ss * M * N + (size_t)r * N + col0] + cc);",
     "ld16_part((const int4*)&parts[(size_t)M * N + (size_t)r * N + col0] + cc);"),
    ("step6.cuh", "6d: the sum without the bound r < M (rows of a tile cut by M)",
     "        if (r >= M) continue;\n",
     ""),
    # step 6: 6b's waves (its grid barrier is not emulated: see README.md)
    ("step6.cuh", "6b: the tiles past the last not skipped (the last wave's blocks without a tile)",
     "        if (t < tiles) tile<BM, BN, kCg, true>(shmem, A, B, C, M, N, K, t, 0, K);\n",
     "        tile<BM, BN, kCg, true>(shmem, A, B, C, M, N, K, t, 0, K);\n"),
    ("step6.cuh", "6b: the waves rounded down (the last, partial wave never computed)",
     "waves = (tiles + gridDim.x - 1) / gridDim.x;",
     "waves = tiles / gridDim.x;"),
    ("step6.cuh", "6b: a block without a tile in the next wave skips the barrier before it (a hang on the GPU)",
     "        if (w + 1 < waves) grid_barrier(sync, gridDim.x);",
     "        if (w + 1 < waves && t + gridDim.x < tiles) grid_barrier(sync, gridDim.x);"),
    # step 6: compute_tile's K range (6d)
    ("step6.cuh", "compute_tile: the loop from 0, not kBegin (the second half of K computes from its start)",
     "for (int k0 = kBegin; k0 < kEnd; k0 += 64) {",
     "for (int k0 = 0; k0 < kEnd; k0 += 64) {"),
    ("step6.cuh", "compute_tile: the prologue's second slice loaded from kBegin + 64",
     "    load(kBegin + 32);\n",
     "    load(kBegin + 64);\n"),
    # step 6: the rules
    ("step6.cuh", "6c's rule: 128-column bands even past 3 P blocks (two rounds at gateup_m16)",
     "if (N / 128 >= lo && N / 128 <= hi) return p;",
     "if (N / 128 >= lo) return p;"),
    ("step6.cuh", "6c's rule: slices added up to 3 P blocks at long K, not 2 P",
     "while (nb * (p.slices + 1) <= mid && 1600 * (p.slices + 1) <= K) p.slices++;",
     "while (nb * (p.slices + 1) <= hi && 1600 * (p.slices + 1) <= K) p.slices++;"),
    ("step6.cuh", "plan: 6d while the tiles fill fewer than all the SMs, not half",
     "if (with6d && M > 64 && 2 * tiles256 < sms",
     "if (with6d && M > 64 && tiles256 < sms"),
    ("step6.cuh", "plan: 6b from one row of tiles (M = 128)",
     "M >= 4 * t.bm) return {kWaves",
     "M >= t.bm) return {kWaves"),
    # step 5: the four tiles (the copy, the warps, the epilogue, the order of the tiles) and the choice
    ("step5.cuh", "B's copy in step 4's rows (t / 32 + 8 i) at every tile",
     "const int r = threadIdx.x / G::kBChunks + i * G::kBRows;\n            rb[i]",
     "const int r = threadIdx.x / 32 + i * 8;\n            rb[i]"),
    ("step5.cuh", "A's copy in step 4's rows (t / 4 + 64 i) at every tile",
     "const int r = threadIdx.x / 4 + i * G::kARows;",
     "const int r = threadIdx.x / 4 + i * 64;"),
    ("step5.cuh", "the warps placed as step 4's, 4 to a row of the tile",
     "const int wm = warpId / G::kWarpsN, wn = warpId % G::kWarpsN;",
     "const int wm = warpId / 4, wn = warpId % 4;"),
    ("step5.cuh", "the epilogue's chunks of a row of 32 chunks at every tile",
     "step3::st16<true>((int4*)&C[(size_t)gr * N + col0] + threadIdx.x % G::kBChunks,",
     "step3::st16<true>((int4*)&C[(size_t)gr * N + col0] + threadIdx.x % 32,"),
    ("step5.cuh", "one pass of the epilogue at every tile (the second 64 rows of a 128-row tile never written)",
     "for (int h = 0; h < BM / 64; h++)",
     "for (int h = 0; h < 1; h++)"),
    ("step5.cuh", "the output buffer's rows without their 8 halves of padding (bank conflicts)",
     "kOStride = BN + 8;",
     "kOStride = BN;"),
    ("step5.cuh", "no M - 1 clamp of the rows of A (edge tiles)",
     "const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;",
     "const int gr = row0 + r;"),
    ("step5.cuh", "tile_of counts the rows of tiles in 128-row tiles",
     "tilesM = (M + BM - 1) / BM",
     "tilesM = (M + 127) / 128"),
    ("step5.cuh", "choose: 128 columns when 256 gives fewer blocks than the SMs, not than half of them",
     "cols128 && 2 * tiles256 < sms",
     "cols128 && tiles256 < sms"),
    ("step5.cuh", "choose: 64 rows of M up to M = 128",
     "rows64 && M <= 64",
     "rows64 && M <= 128"),
    # step 4: the loop's barrier moved (4a, stage())
    ("step4.cuh", "4a: the barrier before the stage's last read (slice 3's fragments)",
     "        frag(1, s, 3);\n        if (st) store(s ^ 1);\n        __syncthreads();\n",
     "        if (st) store(s ^ 1);\n        __syncthreads();\n        frag(1, s, 3);\n"),
    ("step4.cuh", "4a: the barrier after the read of the other stage's first fragments",
     "        __syncthreads();\n        load_next();\n        mma(0);\n        if (nf) frag(0, s ^ 1, 0);\n",
     "        load_next();\n        mma(0);\n        if (nf) frag(0, s ^ 1, 0);\n        __syncthreads();\n"),
    ("step4.cuh", "4a: the stores to the other stage after the barrier",
     "        if (st) store(s ^ 1);\n        __syncthreads();\n",
     "        __syncthreads();\n        if (st) store(s ^ 1);\n"),
    # step 4: 4a's fragments
    ("step4.cuh", "4a: slice 1's HMMA on set 0, the wrong set",
     "        frag(0, s, 2);\n        mma(1);\n",
     "        frag(0, s, 2);\n        mma(0);\n"),
    ("step4.cuh", "4a: A read at the k8 slice k8 ^ 1",
     "aOff[k8] = G::a_at(wm * 64 + lane, k8);",
     "aOff[k8] = G::a_at(wm * 64 + lane, k8 ^ 1);"),
    ("step4.cuh", "4a: B's ldmatrix rows with lane / 8 and lane % 8 swapped",
     "bOff[h] = G::b_at(lane % 8, wn * 8 + 4 * h + lane / 8);",
     "bOff[h] = G::b_at(lane / 8, wn * 8 + 4 * h + lane % 8);"),
    ("step4.cuh", "4a: the other stage's slice 0 read from the current stage",
     "        if (nf) frag(0, s ^ 1, 0);\n",
     "        if (nf) frag(0, s, 0);\n"),
    ("step4.cuh", "4a: the first fragments never read (the prologue)",
     "    load(32);\n    frag(0, 0, 0);\n",
     "    load(32);\n"),
    # step 4: the copy and the loop's tests around 4a, and the edge
    ("step4.cuh", "4a: the last K64 does not store its second slice",
     "        stage(0, true, true, [&] { if (next) load(k0 + 64); });\n",
     "        stage(0, next, true, [&] { if (next) load(k0 + 64); });\n"),
    ("step4.cuh", "the next K64's second slice loaded from k0 + 32 instead of k0 + 96",
     "load(k0 + 96); });",
     "load(k0 + 32); });"),
    ("step4.cuh", "the next K64's first slice loaded without the test next (A and B read past K)",
     "stage(0, true, true, [&] { if (next) load(k0 + 64); });",
     "stage(0, true, true, [&] { load(k0 + 64); });"),
    ("step4.cuh", "no M - 1 clamp of the rows of A (edge tiles)",
     "const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;",
     "const int gr = row0 + r;"),
    # step 4: 4b's order, the B register of hmma (v0, v1), and the bits
    ("step4.cuh", "4b: the rows back and forth on the row's parity, not the column's: rows 0, 2, 2, 0",
     "hmma(f, j % 2 ? 3 - ii : ii, j);",
     "hmma(f, ii % 2 ? 3 - ii : ii, j);"),
    ("step4.cuh", "4a, 4b: a wrong B register in hmma (fb[f][j % 2][j / 2] for fb[f][j / 4][j % 4])",
     "fb[f][j / 4][j % 4]",
     "fb[f][j % 2][j / 2]"),
    ("step4.cuh", "4a: slices 1 and 2 of a stage swapped in every accumulator (C within the tolerance, other bits)",
     "        frag(1, s, 1);\n        mma(0);\n        frag(0, s, 2);\n",
     "        frag(1, s, 2);\n        mma(0);\n        frag(0, s, 1);\n"),
    # step 3: 3c's order of the tiles (tile_of; v0, v1, v2, and step 4's kernels)
    ("step3.cuh", "3c: tm and tn swapped (a group's blocks run along its columns of tiles, not its rows)",
     "        tm = first + r % rows;\n        tn = r / rows;\n",
     "        tn = first + r % rows;\n        tm = r / rows;\n"),
    ("step3.cuh", "3c: the last group's rows not bounded (8 rows of tiles even when fewer are left)",
     "const int rows = min(tilesM - first, 8), r = b % per;",
     "const int rows = 8, r = b % per;"),
    ("step3.cuh", "3c: the last group's rows bounded by all the rows of tiles, not by those left (min(tilesM, 8))",
     "const int rows = min(tilesM - first, 8), r = b % per;",
     "const int rows = min(tilesM, 8), r = b % per;"),
    ("step3.cuh", "3c: a group's first row without the factor 8 (b / per for b / per * 8)",
     "first = b / per * 8;",
     "first = b / per;"),
    ("step3.cuh", "3c: the rows of tiles rounded down (M / 128: the row of tiles cut by M forgotten)",
     "const int tilesM = (M + 127) / 128, per",
     "const int tilesM = M / 128, per"),
    # step 3: 3a's loads, 3b's stores, the edge in M
    ("step3.cuh", "3a: A loaded one chunk off ((t + 1) % 4 for t % 4: inside A, a wrong C)",
     "ra[i] = ld16<kNc>((const int4*)&A[(size_t)gr * K + k0] + threadIdx.x % 4);",
     "ra[i] = ld16<kNc>((const int4*)&A[(size_t)gr * K + k0] + (threadIdx.x + 1) % 4);"),
    ("step3.cuh", "3b: C stored without the half tile's offset (row row0 + r for row0 + 64 h + r)",
     "st16<kEf>((int4*)&C[(size_t)gr * N + col0]",
     "st16<kEf>((int4*)&C[(size_t)(row0 + r) * N + col0]"),
    ("step3.cuh", "no M - 1 clamp of the rows of A (edge tiles)",
     "const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;",
     "const int gr = row0 + r;"),
    ("step3.cuh", "3b: C stored without the bound gr < M (edge tiles write rows past M)",
     "            if (gr < M)\n                st16<kEf>(",
     "                st16<kEf>("),
    # step 2: each barrier removed in turn
    ("step2.cuh", "no barrier after the first slice's stores (prologue)",
     "    load(0);\n    store(0);\n    __syncthreads();\n",
     "    load(0);\n    store(0);\n"),
    ("step2.cuh", "no barrier after the stores to stage 1",
     "        store(1);\n        __syncthreads();\n",
     "        store(1);\n"),
    ("step2.cuh", "no barrier after the stores to stage 0 (end of a K64)",
     "        if (next) store(0);\n        __syncthreads();\n",
     "        if (next) store(0);\n"),
    ("step2.cuh", "no barrier between the epilogue's STS.32 and its copy out",
     "        __syncthreads();\n#pragma unroll\n        for (int i = 0; i < 8; i++) {",
     "#pragma unroll\n        for (int i = 0; i < 8; i++) {"),
    ("step2.cuh", "no barrier after the epilogue's copy out of a half tile",
     "threadIdx.x % 32);\n        }\n        __syncthreads();\n    }\n",
     "threadIdx.x % 32);\n        }\n    }\n"),
    # step 2: addressing and layout
    ("step2.cuh", "swapped register: b[j][2 kk + n] for b[j][2 n + kk]",
     "b[j][2 * n + kk]",
     "b[j][2 * kk + n]"),
    ("step2.cuh", "A read without the swizzle (ldmatrix rows of A)",
     "aOff[ks] = G::a_at(wm * 64 + lane % 16, 2 * ks + lane / 16);",
     "aOff[ks] = (wm * 64 + lane % 16) * G::kAStride + 8 * (2 * ks + lane / 16);"),
    ("step2.cuh", "B stored without the swizzle",
     "*(int4*)&Bs[s * G::kBStage + G::b_at(threadIdx.x / 32 + i * 8, threadIdx.x % 32)] = rb[i];",
     "*(int4*)&Bs[s * G::kBStage + (threadIdx.x / 32 + i * 8) * G::kBStride + 8 * (threadIdx.x % 32)] = rb[i];"),
    ("step2.cuh", "B read without .trans",
     "ldmatrix_x4<true>(b[j]",
     "ldmatrix_x4<false>(b[j]"),
    ("step2.cuh", "wrong epilogue layout: D[g][2t + 1] taken from D[g + 8][2t]",
     "__floats2half2_rn(c[i][j][n][0], c[i][j][n][1])",
     "__floats2half2_rn(c[i][j][n][0], c[i][j][n][2])"),
    ("step2.cuh", "no M - 1 clamp of the rows of A (edge tiles)",
     "const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;",
     "const int gr = row0 + r;"),
    # step 1: each barrier removed in turn, and the edge
    ("step1.cuh", "no barrier after the first slice's stores (double buffer)",
     "        load(0);\n        store(0);\n        __syncthreads();\n",
     "        load(0);\n        store(0);\n"),
    ("step1.cuh", "no barrier after the stores to stage 1",
     "            store(1);\n            __syncthreads();\n",
     "            store(1);\n"),
    ("step1.cuh", "no barrier after the stores to stage 0 (end of a K64)",
     "            if (next) store(0);\n            __syncthreads();\n",
     "            if (next) store(0);\n"),
    ("step1.cuh", "no barrier between the copy and the compute (v2, v4)",
     "            store(0);\n            __syncthreads();\n            compute(0);\n",
     "            store(0);\n            compute(0);\n"),
    ("step1.cuh", "no barrier after the compute (v2, v4)",
     "            compute(0);\n            __syncthreads();\n",
     "            compute(0);\n"),
    ("step1.cuh", "no barrier between the epilogue's fragment stores and its copy out",
     "        __syncthreads();\n#pragma unroll\n        for (int i = 0; i < 64 * G::kBLanes / 256; i++) {",
     "#pragma unroll\n        for (int i = 0; i < 64 * G::kBLanes / 256; i++) {"),
    ("step1.cuh", "no barrier after the epilogue's copy out of a half tile",
     "threadIdx.x % G::kBLanes);\n        }\n        __syncthreads();\n    }\n",
     "threadIdx.x % G::kBLanes);\n        }\n    }\n"),
    ("step1.cuh", "that barrier only after the first half tile (a persistent block's next tile races)",
     "threadIdx.x % G::kBLanes);\n        }\n        __syncthreads();\n    }\n",
     "threadIdx.x % G::kBLanes);\n        }\n        if (h == 0) __syncthreads();\n    }\n"),
    ("step1.cuh", "no M - 1 clamp of the rows of A (edge tiles)",
     "const int gr = kEdge ? min(row0 + r, M - 1) : row0 + r;",
     "const int gr = row0 + r;"),
    # every step with two stages: the write stage frozen at stage 1 (seen only once the stages go round, from the
    # second K64 on: 128 x 512 x 128 for steps 1 to 5, 16 x 1024 x 1024 for step 6's tiles)
    ("step1.cuh", "1b: the next K64's first slice stored to stage 1, the one just read (the write stage frozen)",
     "if (next) store(0);",
     "if (next) store(1);"),
    ("step2.cuh", "the next K64's first slice stored to stage 1, the one just read (the write stage frozen)",
     "if (next) store(0);",
     "if (next) store(1);"),
    ("step3.cuh", "the next K64's first slice stored to stage 1, the one just read (the write stage frozen)",
     "if (next) store(0);",
     "if (next) store(1);"),
    ("step4.cuh", "4a: every copy stored to stage 1 (the write stage frozen)",
     "if (st) store(s ^ 1);",
     "if (st) store(1);"),
    ("step5.cuh", "every copy stored to stage 1 (the write stage frozen)",
     "if (st) store(s ^ 1);",
     "if (st) store(1);"),
    ("step6.cuh", "compute_tile: every copy stored to stage 1 (the write stage frozen)",
     "if (st) store(s ^ 1);",
     "if (st) store(1);"),
]


def check_substitutions():
    """Each bug's text must occur exactly once in its header; otherwise stop, loudly."""
    sources = {h: open(os.path.join(TURING, h)).read() for h in HEADERS}
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
        shutil.copy(os.path.join(TURING, h), d)
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
    args = [exe, "all"] if bug is None else [exe, STEP[bug[0]], "-x"]
    try:
        run = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", ""
    return run.returncode, run.stdout


def first_failure(output):
    """The first FAILED run (label, shape, order and reasons) and the first EMU ERROR line of an emulator output."""
    run = next((l for l in output.splitlines() if "  FAILED  " in l), "")
    head, _, tail = run.partition("  FAILED  ")
    why = tail[tail.find("[") + 1:tail.rfind("]")] if "[" in tail else tail.strip()
    error = next((l for l in output.splitlines() if l.startswith("EMU ERROR")), "")
    return " ".join(head.split()) + ": " + why, error


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-j", type=int, default=os.cpu_count() or 1, help="parallel builds and runs")
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = parser.parse_args()
    check_substitutions()
    tmp = tempfile.mkdtemp(prefix="planted_bugs_")
    print(f"planted-bug test: {len(BUGS)} bugs, each in a copy of its header, the emulator built against each copy "
          f"({args.j} jobs, in {tmp})", flush=True)
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
            run, error = first_failure(output)
            print(f"  reported  {header}: {what}\n            first failed run: {run}")
            if error:
                print(f"            {error}")
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
