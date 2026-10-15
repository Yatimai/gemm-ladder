# The Turing rung (Tesla T4)

The repository's GEMM (A and B fp16 row-major, fp32 accumulation, C fp16 row-major) on the T4 (sm_75), built in six
steps from NVIDIA's tensor-core sample. Each step is timed against the judge's reference at the 17 shapes, both defined
in the repository's README (`../README.md`).

Score 1.05; shapes from 1.0016 (down_m16) to 1.1053 (down_m128) (`results/judge/20261015-132913_step6`, the arrival:
container 1, where step 6 scores lower).

| step | what it adds | container 1 | container 2 |
|---|---|---|---|
| reference point | NVIDIA's `simple_wmma_gemm`: wmma from global memory, no shared memory | 0.1903 | 0.1911 |
| 0 | NVIDIA's `compute_gemm`: 128 x 128 tiles, wmma, a persistent kernel (one block per SM) | 0.6220 | 0.6272 |
| 1 | two shared stages of K32, a 128 x 256 tile, the copy spread over all threads | 0.7586 | 0.7624 |
| 2 | mma.sync and ldmatrix instead of wmma, a swizzle instead of the padding | 0.8094 | 0.8125 |
| 3 | read-only loads that ask the L2 for whole lines, the tiles in groups of 8 rows of tiles | 0.8343 | 0.8404 |
| 4 | fragment double buffering in registers | 0.8965 | 0.9014 |
| 5 | a tile of 64 x 256 where M <= 64, step 4's 128 x 256 elsewhere | 0.9827 | 0.9886 |
| 6 | loads through the L2 only, 16-row bands at M <= 16, split-K in two at two shapes | 1.0561 | 1.0566 |

Score: the geometric mean over the 17 shapes of the reference's time over the step's (above 1: faster than the
reference), in each of two containers, numbered by the time stamps of their outputs (`results/judge/20261015-132913_*`
and `results/judge/20261015-133006_*`). Below, what each mechanism saves, then each step; the probe tables shape by
shape, the counters and the list of the raw outputs are in `results/README.md`.

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
  to one decimal, at least 0.1 %). Under ncu, one process per arm, the step, a removal and the configuration of a
  REFERENCE line (arm `x:`); the counters cited are those of the arm's second call.
- **ncu** (`modal/session.py`): at ladder, clocks not locked, in one session, one call of the step, of each of its
  variants and of the configuration ranked first in the judge's final at ladder for that step in the arrival
  (`--cublaslt`). Counters are compared within one session or one probe only (their names: `results/README.md`).
- **compute-sanitizer** (memcheck, racecheck, synccheck) on the shapes of `verif.cu`, and the **host emulator**
  (`emu/`), which runs every kernel on the CPU and checks its accesses, barriers, bank conflicts and results; its runs
  on these sources are in `results/emu/` (`emu_all.txt`: 771 runs, 771 ok, 0 failed, `ALL OK`; `planted_bugs.txt`: 79
  of 79 planted bugs reported as failures, the unmodified copy passes, `PLANTED-BUG TEST OK`).
- **The card**: the T4's power cap is 70 W (the judge's header and card line); the card's power, clocks and throttle
  reasons in the judge's outputs: `results/README.md`. The two scores of a row differ by 0.05 to 0.84 %, so a step is
  read against the previous step in the same container.

Each step's probe runs in two series, each run in a container of its own; at each step, the two probes that
decide ran at the same time, so on two different cards (`results/two_cards.txt`). A mechanism's case is the time
without it over the step's, minus 1 (above 0: it saves time), or, measured added, the time with it over the time
without it, minus 1 (below 0: it saves time); its mean, the geometric mean of these time ratios over the shapes where
it acts (the cases without a dash), minus 1, in % to one decimal. A mechanism stays if, in both series, its mean shows
a saving of more than twice the noise of its probe; two that work only together count as one (2a and 2b). All stay:

| step | mechanism | case | shapes | mean, series 1 | twice the noise | mean, series 2 | twice the noise |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | 1a | without it | 17 | +5.9 % | 0.4 % | +3.5 % | 1.6 % |
| 1 | 1b | without it | 17 | +23.7 % | 0.4 % | +23.8 % | 1.6 % |
| 1 | 1c | added to step 0 | 17 | -4.2 % | 0.4 % | -4.0 % | 1.6 % |
| 2 | 2a alone | added to step 1 | 17 | +1.9 % | 0.6 % | +0.7 % | 2.6 % |
| 2 | 2a and 2b | without both (step 1) | 17 | +6.9 % | 0.6 % | +8.6 % | 2.6 % |
| 3 | 3a | without it | 17 | +2.2 % | 1.2 % | +1.6 % | 0.6 % |
| 3 | 3b | without it | 17 | +4.5 % | 1.2 % | +3.1 % | 0.6 % |
| 4 | 4a | without it (step 3) | 17 | +9.4 % | 0.6 % | +7.6 % | 0.8 % |
| 5 | 5a | without it | 4 | +49.5 % | 1.2 % | +45.0 % | 4.2 % |
| 6 | 6a | without it | 13 | +3.2 % | 0.6 % | +3.4 % | 0.8 % |
| 6 | 6b | without its hint, added to step 6 without 6b | 4 | -15.4 % | 0.6 % | -11.8 % | 0.8 % |
| 6 | 6b's hint | without it | 2 | +1.9 % | 0.6 % | +1.9 % | 0.8 % |
| 6 | 6c | without it | 2 | +12.0 % | 0.6 % | +11.9 % | 0.8 % |

## Step 0: NVIDIA's compute_gemm, adapted

`compute_gemm`, from cuda-samples v12.4 (`cudaTensorCoreGemm`): 128 x 128 tiles, 8 warps of 32 x 64, wmma
16 x 16 x 16, one K64 slice copied to padded shared rows between two barriers, one persistent block per SM; adapted:
fp16 output, B row-major, sizes at run time, an edge in M. The reference point is the same sample's `simple_wmma_gemm`.
compute-sanitizer (`results/20261015-133214_step0`; the reference point: `results/20261015-133133_ref_simple`):
memcheck 0 errors, racecheck 0 hazards displayed, synccheck 0 errors, in both. ncu at ladder (step 0's folder): tensor
active 37.9 % against cuBLASLt's 87.8 %; 1 396 309 cycles against 607 174; DRAM read 120.0 MB against 53.6.

## Step 1: two shared stages, a 128 x 256 tile, every thread copying

- 1a: two shared stages of K32 filled through registers, one barrier per K32 (`compute_tile`, `kDouble`).
- 1b: a 128 x 256 tile in 8 warps of 64 x 64 (`Geometry<256, ...>`), where step 0 has 128 x 128 in warps of 32 x 64.
- 1c: every thread copies its share of A and B (`load`, `store`), where step 0 gives A to warps 0-3 and B to 4-7.
- Probes: `results/20261015-1355_step1_probe` (series 1), `results/20261015-1356_step1_probe` (series 2).
- compute-sanitizer (`results/20261015-133212_step1`): memcheck 0 errors, racecheck 0 hazards displayed, synccheck 0
  errors. ncu at ladder (same folder): tensor active 56.9 % against cuBLASLt's 88.3 %; 938 240 cycles against 610 286.

## Step 2: mma.sync and ldmatrix, a swizzle

- 2a: mma.sync m16n8k8 and ldmatrix.x4 instead of wmma (`mma_m16n8k8`, `ldmatrix_x4`): 16 LDSM.x4 per K32 and warp.
- 2b: a swizzle of the shared stages instead of the padding (`Geometry<true>`, `a_at`, `b_at`): no bank conflict by the
  headers' bank model, 48 KB of stages instead of 58. wmma cannot read it, so 2b exists only on top of 2a.
- 2a alone saves no time, so 2a and 2b count as one mechanism, judged without both (step 1).
- Probes: `results/20261015-1334_step2_probe` (series 1), `results/20261015-1335_step2_probe` (series 2).
- compute-sanitizer (`results/20261015-133237_step2`): memcheck 0 errors, racecheck 0 hazards displayed, synccheck 0
  errors. ncu at ladder (same folder), step 1 -> step 2: LDSM instructions 2 621 440 -> 1 310 720 (cuBLASLt
  1 315 840); excess shared-memory wavefronts 5 898 240 -> 0 (cuBLASLt 0; the bank model of `step2.cuh`:
  5 898 240 -> 0); tensor active 57.1 -> 64.3 % (cuBLASLt 88.0 %).

## Step 3: read-only loads, a tile swizzle

- 3a: read-only loads of A and B that ask the L2 for the whole 128-byte line (`ld16`, ld.global.nc.L2::128B).
- 3b: the tiles in groups of 8 rows of tiles, column by column (`tile_of`).
- Probes: `results/20261015-1355_step3_probe` (series 1), `results/20261015-1356_step3_probe` (series 2).
- compute-sanitizer (`results/20261015-133146_step3`): memcheck 0 errors, racecheck 0 hazards displayed, synccheck 0
  errors. ncu at ladder (same folder), step 2 -> step 3: DRAM read 68.9 -> 49.5 MB (cuBLASLt 51.2); L2 sector hits
  77.2 -> 88.7 % (cuBLASLt 88.8 %); tensor active 60.7 % against cuBLASLt's 87.9 %.

## Step 4: fragment double buffering

- 4a: the fragments of one k8 slice at a time in two sets of 16 registers; the next slice's 4 LDSM issue among the
  current slice's 32 HMMA (`compute_tile`). Without it, step 4 is step 3.
- Probes: `results/20261015-1414_step4_probe` (series 1), `results/20261015-1415_step4_probe` (series 2).
- compute-sanitizer (`results/20261015-133140_step4`): memcheck 0 errors, racecheck 0 hazards displayed, synccheck 0
  errors. ncu at ladder (same folder): tensor active 81.5 % against cuBLASLt's 87.9 %; 666 170 cycles against 612 240.

## Step 5: the tile per shape

- 5a: 64 rows of M when M <= 64, 128 otherwise (`choose`), 256 columns: at the 17 shapes, 64 x 256 at the four M = 16
  shapes, and elsewhere step 4's 128 x 256, which is step 4's kernel, instruction for instruction.
- Probes: `results/20261015-1334_step5_probe` (series 1), `results/20261015-1335_step5_probe` (series 2). Their noise
  counts v2 (step 4) at the 13 shapes M >= 128, where it runs step 5's code under another kernel name.
- compute-sanitizer (`results/20261015-133228_step5`): memcheck 0 errors, racecheck 0 hazards displayed, synccheck 0
  errors. ncu at ladder (same folder), where 5a does not act: instructions 16 201 920, and 16 201 920 in step 4.

## Step 6: L2-only loads, 16-row bands, split-K in two

- 6a: `compute_tile` loads A and B through the L2 only (`ld16`, ld.global.cg.L2::128B), at the 13 shapes M >= 128.
- 6b: at M <= 16, a kernel of 16-row bands (`gemm_bands`): a block per band of 128 or 256 columns of B and per slice
  of K (rule `bands`; at the judge's shapes 1 to 5 slices, all the blocks in one round), fp16 parts summed by the last
  block of a band. Without its hint, it saves time on its own: the hint is a mechanism of its own.
- 6b's hint: B loaded evict-first when a band has slices (`ld16_band`, ld.global.cs), at o_m16 and down_m16.
- 6c: split-K in two where step 5 takes 128 rows, its 128 x 256 tiles are fewer than half the SMs and K is a multiple
  of 128 (`gemm_split`): at o_m128 and down_m128, 32 halves of K for the 40 blocks, where step 5 has 16 tiles.
- Probes: `results/20261015-1334_step6_probe` (series 1), `results/20261015-1335_step6_probe` (series 2).
- compute-sanitizer (`results/20261015-133303_step6`): memcheck 0 errors, racecheck 0 hazards displayed, synccheck 0
  errors. ncu at ladder (same folder), where only 6a acts: 624 614 cycles against cuBLASLt's 611 428, tensor active
  85.6 % against 87.5 %. ncu in the probes at down_m16 (6b) and o_m128 (6c), against cuBLASLt: `results/README.md`.

## Limits

- One card: the T4 at its 70 W power cap, with cuBLASLt 130201 (the judge's header) and the driver 610.57.04
  (`results/*/00_machine.txt`); another card, driver or power limit can move every ratio.
- cuBLAS only, and the judge's search does not try every cuBLASLt configuration (`judge/README.md`).
- The 17 shapes only: M a multiple of 16, N of 256, K of 64 (the host checks it).
