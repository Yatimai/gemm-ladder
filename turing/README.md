# The Turing rung (Tesla T4)

The repository's GEMM (A and B fp16 row-major, fp32 accumulation, C fp16 row-major) on the T4 (sm_75), built in six
steps from NVIDIA's tensor-core sample. Each step is timed against the judge's reference at the 17 shapes, both defined
in the repository's README (`../README.md`).

Score 1.05; shapes from 1.0024 (down_m16) to 1.1059 (down_m128) (`results/judge/20261025-172649_step6`, the arrival:
container 2, where step 6 scores lower).

| step | what it adds | container 1 | container 2 |
|---|---|---|---|
| reference point | NVIDIA's `simple_wmma_gemm`: wmma from global memory, no shared memory | 0.1863 | 0.1868 |
| 0 | NVIDIA's `compute_gemm`: 128 x 128 tiles, wmma, a persistent kernel (one block per SM) | 0.6124 | 0.6193 |
| 1 | two shared stages of K32, a 128 x 256 tile, the copy spread over all threads | 0.7535 | 0.7559 |
| 2 | mma.sync and ldmatrix instead of wmma, a swizzle instead of the padding | 0.8048 | 0.8042 |
| 3 | read-only loads that ask the L2 for whole lines, the tiles in groups of 8 rows of tiles | 0.8241 | 0.8249 |
| 4 | fragment double buffering in registers | 0.8980 | 0.8968 |
| 5 | a tile of 64 x 256 where M <= 64, step 4's 128 x 256 elsewhere | 0.9874 | 0.9839 |
| 6 | loads through the L2 only, 16-row bands at M <= 16, split-K in two at two shapes, synchronized waves at M >= 512, one block per tile, the HMMA in an order that reuses an operand | 1.0695 | 1.0590 |

Score: the geometric mean over the 17 shapes of the reference's time over the step's (above 1: faster than the
reference), in each of two containers, numbered by the time stamps of their outputs (`results/judge/20261025-172635_*`
and `results/judge/20261025-172649_*`). Below, what each mechanism saves, then each step; the probe tables shape by
shape, the folders of the probes and sessions, the counters and the list of the raw outputs are in `results/README.md`.

## How it is measured

Paths start from this folder (`results/`, `emu/`) or from the repository's root (`judge/`, `probe/`, `modal/`).
- **Judge** (`judge/judge.cu`, version 6; launcher `judge/modal_judge.py`; `judge/README.md`): checks each step's
  result, then times it against the reference, with the reference cached in `judge/reference/t4.txt`, in each of two
  containers with the same card line (`Tesla_T4|70W|lt130201|cuda13030`). The outputs are in `results/judge/`; the
  first two lines of each give the md5 of the files it judged.
- **Probe** (`probe/probe.cu`; launcher `probe/modal_probe.py --card T4`): in one process, interleaved, 5 rounds of
  1 s, the step and its variants `v<i>` (each defined in `step<i>.cu`); the time per call is the median of the rounds.
  The step is timed again as an arm of its own (`twin`); the noise of a probe is the largest gap, in absolute value,
  between the step and an arm that runs exactly the same code (same instructions, grid and arguments), whatever its
  kernel's name, at any of its shapes: `twin`, a variant whose cell shows `-`, or a variant named under the table (in %,
  to one decimal, at least 0.1 %); its rms noise is the square root of the mean of the squares of these same gaps,
  unrounded. Under ncu, one process per arm, the step, a removal and the configuration of a REFERENCE line (arm `x:`);
  the counters cited are those of the arm's second call.
- **ncu** (`modal/session.py`): at ladder, clocks not locked, in one session per step (its folder: `results/README.md`),
  one call of the step, of each of its variants and of the configuration ranked first in the judge's final at ladder
  for that step in the arrival (`--cublaslt`). Counters are compared within one session or one probe only.
- **compute-sanitizer** (memcheck, racecheck, synccheck) on the shapes of `verif.cu`, in the same sessions: memcheck 0
  errors, racecheck 0 hazards displayed, synccheck 0 errors, in each of the 8. The **host emulator** (`emu/`) runs every
  kernel on the CPU and checks its accesses, barriers, bank conflicts and results; its runs on these sources are in
  `results/emu/` (`emu_all.txt`: 1370 runs, 1370 ok, 0 failed, `ALL OK`; `planted_bugs.txt`: 95 of 95 planted bugs
  reported as failures, the unmodified copy passes, `PLANTED-BUG TEST OK`).
- **The card**: the T4's power cap is 70 W (the judge's header and card line); the card's power, clocks and throttle
  reasons in the judge's outputs: `results/README.md`. The two scores of a row differ by 0.07 to 1.13 %, so a step is
  read against the previous step in the same container.

Each step's probe runs in two series, each run in a container of its own; at each step, the two probes that
decide ran at the same time, so on two different cards (`results/two_cards.txt`). "Faster" is the geometric mean,
over the shapes where a mechanism acts (the cases without a dash), of the time without it over the time with it,
minus 1, in % to one decimal. Most mechanisms are measured removed from the step; two are measured added to the
version without them, 1c (the only copy in step 1's code) and 6b without its hint, and their cases (`results/README.md`)
are turned here the same way. A mechanism stays if, in both series, it saves more than its threshold: three times the
standard error of its mean (the rms noise of its probe over the square root of the number of shapes where it acts), to
one decimal; at least 0.2 %; and at least the noise of its probe if it acts at 3 shapes or fewer; two that work only
together count as one (2a and 2b; 2a alone saves no time: step 2). All stay:

| step | mechanism | measured | shapes | faster, series 1 | threshold | faster, series 2 | threshold |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | 1a | removed | 17 | +6.3 % | 0.2 % | +5.3 % | 0.2 % |
| 1 | 1b | removed | 17 | +23.5 % | 0.2 % | +24.5 % | 0.2 % |
| 1 | 1c | added to step 0 | 17 | +4.6 % | 0.2 % | +4.9 % | 0.2 % |
| 2 | 2a and 2b | removed together (step 1) | 17 | +6.6 % | 0.2 % | +7.2 % | 0.2 % |
| 3 | 3a | removed | 17 | +0.4 % | 0.2 % | +2.3 % | 0.3 % |
| 3 | 3b | removed | 17 | +2.5 % | 0.2 % | +4.3 % | 0.3 % |
| 4 | 4a | removed (step 3) | 17 | +10.2 % | 0.2 % | +8.8 % | 0.2 % |
| 5 | 5a | removed | 4 | +48.5 % | 0.3 % | +49.2 % | 0.2 % |
| 6 | 6a | removed | 13 | +1.8 % | 0.2 % | +2.1 % | 0.2 % |
| 6 | 6b | added without its hint to step 6 without 6b | 4 | +10.4 % | 0.3 % | +12.9 % | 0.2 % |
| 6 | 6b's hint | removed | 2 | +1.9 % | 1.0 % | +1.9 % | 0.6 % |
| 6 | 6c | removed | 2 | +5.6 % | 1.0 % | +5.4 % | 0.6 % |
| 6 | 6d | removed | 9 | +0.5 % | 0.2 % | +0.3 % | 0.2 % |
| 6 | 6e | removed | 4 | +0.6 % | 0.3 % | +0.7 % | 0.2 % |
| 6 | 6f | removed | 13 | +0.3 % | 0.2 % | +0.5 % | 0.2 % |

## Step 0: NVIDIA's compute_gemm, adapted

`compute_gemm`, from cuda-samples v12.4 (`cudaTensorCoreGemm`): 128 x 128 tiles, 8 warps of 32 x 64, wmma
16 x 16 x 16, one K64 slice copied to padded shared rows between two barriers, one persistent block per SM; adapted:
fp16 output, B row-major, sizes at run time, an edge in M. The reference point is the same sample's `simple_wmma_gemm`.
ncu at ladder: tensor active 37.4 % against cuBLASLt's 87.9 %; 1 435 988 cycles against 608 660; DRAM read 120.0 MB
against 52.4.

## Step 1: two shared stages, a 128 x 256 tile, every thread copying

- 1a: two shared stages of K32 filled through registers, one barrier per K32 (`compute_tile`, `kDouble`).
- 1b: a 128 x 256 tile in 8 warps of 64 x 64 (`Geometry<256, ...>`), where step 0 has 128 x 128 in warps of 32 x 64.
- 1c: every thread copies its share of A and B (`load`, `store`), where step 0 gives A to warps 0-3 and B to 4-7.
- ncu at ladder: tensor active 57.0 % against cuBLASLt's 88.0 %; 943 299 cycles against 610 535.

## Step 2: mma.sync and ldmatrix, a swizzle

- 2a: mma.sync m16n8k8 and ldmatrix.x4 instead of wmma (`mma_m16n8k8`, `ldmatrix_x4`): 16 LDSM.x4 per K32 and warp.
- 2b: a swizzle of the shared stages instead of the padding (`Geometry<true>`, `a_at`, `b_at`): no bank conflict by the
  headers' bank model, 48 KB of stages instead of 58. wmma cannot read it, so 2b exists only on top of 2a.
- 2a alone saves no time, so 2a and 2b count as one mechanism, judged without both (step 1).
- ncu at ladder, step 1 -> step 2: LDSM instructions 2 621 440 -> 1 310 720 (cuBLASLt 1 315 840); excess shared-memory
  wavefronts 5 898 240 -> 0 (cuBLASLt 0; the bank model of `step2.cuh`: 5 898 240 -> 0); tensor active 56.9 -> 62.2 %
  (cuBLASLt 87.7 %).

## Step 3: read-only loads, a tile swizzle

- 3a: read-only loads of A and B that ask the L2 for the whole 128-byte line (`ld16`, ld.global.nc.L2::128B).
- 3b: the tiles in groups of 8 rows of tiles, column by column (`tile_of`).
- ncu at ladder, step 2 -> step 3: DRAM read 67.5 -> 49.4 MB (cuBLASLt 51.1); L2 sector hits 80.0 -> 88.7 % (cuBLASLt
  88.8 %); tensor active 60.7 % against cuBLASLt's 87.7 %.

## Step 4: fragment double buffering

- 4a: the fragments of one k8 slice at a time in two sets of 16 registers; the next slice's 4 LDSM issue among the
  current slice's 32 HMMA (`compute_tile`). Without it, step 4 is step 3.
- ncu at ladder: tensor active 81.4 % against cuBLASLt's 87.9 %; 666 324 cycles against 612 266.

## Step 5: the tile per shape

- 5a: 64 rows of M when M <= 64, 128 otherwise (`choose`), 256 columns: at the 17 shapes, 64 x 256 at the four M = 16
  shapes, and elsewhere step 4's 128 x 256, which is step 4's kernel, instruction for instruction.
- ncu at ladder, where 5a does not act: instructions 16 201 920, and 16 201 920 in step 4.

## Step 6: L2-only loads, 16-row bands, split-K in two, synchronized waves

- 6a: `compute_tile` loads A and B through the L2 only (`ld16`, ld.global.cg.L2::128B), at the 13 shapes M >= 128.
- 6b: at M <= 16, a kernel of 16-row bands (`gemm_bands`): a block per band of 128 or 256 columns of B and per slice
  of K (rule `bands`; at the judge's shapes 1 to 5 slices, all the blocks in one round), fp16 parts summed by the last
  block of a band. Without its hint, it saves time on its own: the hint is a mechanism of its own.
- 6b's hint: B loaded evict-first when a band has slices (`ld16_band`, ld.global.cs), at o_m16 and down_m16.
- 6c: split-K in two where step 5 takes 128 rows, its 128 x 256 tiles are fewer than half the SMs and K is a multiple
  of 128 (`gemm_split`): at o_m128 and down_m128, 32 halves of K, one block per half (6e), where step 5 has 16 tiles.
- 6d: synchronized waves on the 128 x 256 tile at M >= 512 (`gemm_waves`): one block per SM over step 5's order, a
  barrier of the whole grid between two waves (`grid_barrier`): at the four M = 512 and four M = 2048 shapes and ladder.
- 6e: one block per tile where step 6 runs tiles (`grid`), in place of step 5's persistent grid of one block per SM: at
  qkv_m128 and gateup_m128, and at 6c's halves (o_m128 and down_m128).
- 6f: a k8 slice's 32 HMMA in an order that reuses an operand (`kOrder`): the 8 columns in turn, the 4 rows forward then
  back (B's operand reused in 3 HMMA of 4), in every tile of step 6 (the 13 shapes M >= 128); the same C, bit for bit.
- ncu at ladder, where 6a, 6d and 6f act: 649 132 cycles against cuBLASLt's 613 325, tensor active 82.7 % against
  88.2 %. ncu in the probes, at one shape for each of 6a to 6f and 6b's hint, against cuBLASLt: `results/README.md`.

## Limits

- One card: the T4 at its 70 W power cap, with cuBLASLt 130201 (the judge's header) and the driver 610.57.04
  (`results/*/00_machine.txt`); another card, driver or power limit can move every ratio.
- cuBLAS only, and the judge's search does not try every cuBLASLt configuration (`judge/README.md`).
- The 17 shapes only: M a multiple of 16, N of 256, K of 64 (the host checks it).
- 6d assumes, without a cooperative launch, that all its blocks are resident at once (`step6.cu` checks that one block
  fits in an SM, else runs 6e's grid); another context on an SM can leave the rest waiting at the barrier (`step6.cuh`).
