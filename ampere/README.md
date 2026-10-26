# The Ampere rung (A100-SXM4-80GB)

The repository's GEMM (A and B fp16 row-major, fp32 accumulation, C fp16 row-major) on the A100 (sm_80), built in seven
steps from NVIDIA's tensor-core sample. Each step is timed against the judge's reference at the 17 shapes, both defined
in the repository's README (`../README.md`).

Score 1.02; shapes from 0.9789 (qkv_m128) to 1.1095 (o_m128) (`results/judge/20261026-163446_step7`, the arrival:
container 2, where step 7's score, recomputed from its written ratios, is lower; both show 1.0224: `results/README.md`).

| step | what it adds | container 1 | container 2 |
|---|---|---|---|
| reference point | NVIDIA's `simple_wmma_gemm`: wmma from global memory, no shared memory | 0.1030 | 0.1017 |
| 0 | NVIDIA's `compute_gemm`: 128 x 128 tiles, wmma, a persistent kernel (one block per SM) | 0.2736 | 0.2707 |
| 1 | one block per tile, a 128 x 256 tile, the copy spread over all threads | 0.3809 | 0.3780 |
| 2 | mma.sync and ldmatrix instead of wmma, a swizzle instead of the padding | 0.3855 | 0.3825 |
| 3 | cp.async through the L2 only, three shared stages | 0.5674 | 0.5668 |
| 4 | stores of C marked evict-first, the tiles in groups of 16 rows of tiles | 0.5836 | 0.5828 |
| 5 | the next fragments loaded among the current HMMA, a serpentine order of the HMMA | 0.7197 | 0.7178 |
| 6 | a tile per shape: 128 x 128, 16 rows at M = 16, a stream in phase | 0.9752 | 0.9757 |
| 7 | K in parts at six more shapes, reduced by their own blocks, in place, or by the last to arrive; L2 cache hints; the reads of an addition issued together | 1.0224 | 1.0224 |

Score: the geometric mean over the 17 shapes of the reference's time over the step's (above 1: faster than the
reference), in each of two containers, numbered by the time stamps of their outputs (`results/judge/20261026-163413_*`
and `results/judge/20261026-163446_*`). Below, what each mechanism saves, then each step; the probe tables shape by
shape, the folders of the probes and sessions, the counters and the list of the raw outputs are in `results/README.md`.

## How it is measured

Paths start from this folder (`results/`, `emu/`) or from the repository's root (`judge/`, `probe/`, `modal/`).
- **Judge** (`judge/judge.cu`, version 6; launcher `judge/modal_judge.py`; `judge/README.md`): checks each step's
  result, then times it against the reference, with the reference cached in `judge/reference/a100.txt`, in each of two
  containers with the same card line (`NVIDIA_A100-SXM4-80GB|400W|lt130201|cuda13030`). The outputs are in
  `results/judge/`; the first two lines of each give the md5 of the files it judged.
- **Probe** (`probe/probe.cu`; launcher `probe/modal_probe.py --card A100-80GB`): in one process, interleaved, 5 rounds
  of 1 s, the step and its variants `v<i>` (each defined in `step<i>.cu`); the time per call is the median of the
  rounds. The step is timed again as an arm of its own (`twin`); the noise of a probe is the largest gap, in absolute
  value, between the step and an arm that runs exactly the same code (same instructions, grid and arguments), whatever
  its kernel's name, at any of its shapes: `twin`, a variant whose cell shows `-`, or a variant named under the table
  (in %, to one decimal, at least 0.1 %); its rms noise is the square root of the mean of the squares of these same
  gaps, unrounded. Under ncu, one process per arm, the step, a removal and the configuration of a REFERENCE line (arm
  `x:`); the counters cited are those of the arm's second call.
- **ncu** (`modal/session.py`): at ladder, clocks not locked, in one session per step (its folder: `results/README.md`),
  one call of the step, of each of its variants and of the configuration ranked first in the judge's final at ladder for
  that step in the arrival (`--cublaslt`). Counters are compared within one session or one probe only.
- **compute-sanitizer** (memcheck, racecheck, synccheck) on the shapes of `verif.cu`, in the same sessions: memcheck 0
  errors, racecheck 0 hazards displayed, synccheck 0 errors, in each of the 9. The **host emulator** (`emu/`) runs every
  kernel on the CPU and checks its accesses, barriers, bank conflicts and results; its runs on these sources are in
  `results/emu/` (`emu_all.txt`: 1076 runs, 1076 ok, 0 failed, `ALL OK`; `planted_bugs.txt`: 164 of 164 planted bugs
  reported as failures, the unmodified copy passes, `PLANTED-BUG TEST OK`).
- **The card**: the A100's power cap is 400 W (the judge's header and card line); the card's power, clocks and throttle
  reasons in the judge's outputs: `results/README.md`. The two scores of a row differ by up to 1.28 %, so a step is read
  against the previous step in the same container.

Each step's probe runs in two series, each run in a container of its own; at each step, the two probes that decide ran
at the same time, so on two different cards (`results/two_cards.txt`). "Faster" is the geometric mean, over the shapes
where a mechanism acts (the cases without a dash), of the time without it over the time with it, minus 1, in % to one
decimal. Most mechanisms are measured removed from the step; three are measured added to the version without them, 1c,
3a and 7a, and their cases (`results/README.md`) are turned here the same way. A mechanism stays if, in both series, it
saves more than its threshold: three times the standard error of its mean (the rms noise of its probe over the square
root of the number of shapes where it acts), to one decimal; at least 0.2 %; and at least the noise of its probe if it
acts at 3 shapes or fewer; two that work only together count as one (2a and 2b). All stay:

| step | mechanism | measured | shapes | faster, series 1 | threshold | faster, series 2 | threshold |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | 1a | removed | 17 | +3.7 % | 0.2 % | +3.6 % | 0.2 % |
| 1 | 1b | removed | 17 | +6.2 % | 0.2 % | +6.1 % | 0.2 % |
| 1 | 1c | added to step 0 | 17 | +25.1 % | 0.2 % | +25.3 % | 0.2 % |
| 2 | 2a and 2b | removed together (step 1) | 17 | +1.3 % | 0.2 % | +1.2 % | 0.2 % |
| 3 | 3a | added to step 2 | 17 | +10.1 % | 0.2 % | +10.2 % | 0.2 % |
| 3 | 3b | removed (3a alone, one stage) | 17 | +35.7 % | 0.2 % | +34.3 % | 0.2 % |
| 4 | 4a | removed | 17 | +1.2 % | 0.2 % | +1.3 % | 0.2 % |
| 4 | 4b | removed | 17 | +1.7 % | 0.2 % | +1.6 % | 0.2 % |
| 5 | 5a | removed | 17 | +22.1 % | 0.2 % | +21.3 % | 0.2 % |
| 5 | 5b | removed | 17 | +0.6 % | 0.2 % | +0.6 % | 0.2 % |
| 6 | 6a | removed | 6 | +38.7 % | 0.2 % | +37.4 % | 0.2 % |
| 6 | 6b | removed | 4 | +140.4 % | 0.2 % | +143.0 % | 0.2 % |
| 6 | 6c | removed | 2 | +1.8 % | 0.2 % | +2.7 % | 0.2 % |
| 7 | 7a | added without 7b (7f at o_m512, 7c at down_m512) to step 7 without 7a | 4 | +13.8 % | 0.2 % | +12.3 % | 0.3 % |
| 7 | 7b | removed, with 7g and 7d's hints there | 2 | +9.5 % | 0.5 % | +12.1 % | 0.5 % |
| 7 | 7c | removed | 1 | +6.5 % | 0.5 % | +6.0 % | 0.5 % |
| 7 | 7d | removed | 11 | +1.0 % | 0.2 % | +1.3 % | 0.2 % |
| 7 | 7e | removed | 2 | +0.7 % | 0.5 % | +0.8 % | 0.5 % |
| 7 | 7f | removed | 1 | +1.0 % | 0.5 % | +1.5 % | 0.5 % |
| 7 | 7g | removed | 2 | +0.9 % | 0.5 % | +1.5 % | 0.5 % |
| 7 | 7h | removed | 4 | +0.7 % | 0.2 % | +0.7 % | 0.3 % |

## Step 0: NVIDIA's compute_gemm, adapted

`compute_gemm`, from cuda-samples v12.4 (`cudaTensorCoreGemm`), the Turing rung's step 0 line for line, on sm_80:
128 x 128 tiles, 8 warps of 32 x 64, one K64 slice copied to padded shared rows between two barriers, one persistent
block per SM. The reference point is the same sample's `simple_wmma_gemm`. ncu at ladder: tensor active 24.6 % against
cuBLASLt's 87.4 %; 1 078 274 cycles against 306 015.

## Step 1: one block per tile, a 128 x 256 tile, every thread copying

- 1a: one block per tile (`kOneTile`), where step 0 runs one block per SM over a list of tiles.
- The template's double buffer (the Turing rung's 1a) is not taken: step 3 copies asynchronously.
- 1b: a 128 x 256 tile in 8 warps of 64 x 64 (`kBN` = 256), where step 0 has 128 x 128 in warps of 32 x 64.
- 1c: every thread copies its share of A and B, where step 0 gives A to warps 0-3 and B to 4-7.
- ncu at ladder: tensor active 45.0 % against cuBLASLt's 87.4 %; 588 746 cycles against 305 553.

## Step 2: mma.sync and ldmatrix, a swizzle

- 2a: mma.sync m16n8k16 and ldmatrix.x4 instead of wmma (`mma_m16n8k16`, `ldmatrix_x4`): the same HMMA and LDSM as wmma
  on sm_80, the fragment addresses held in registers across the loop.
- 2b: a swizzle of the shared stage instead of the padding (`Geometry<true>`): no bank conflict by the headers' bank
  model. wmma cannot read it, so 2b exists only on top of 2a; 2a alone costs time, so the two count as one.
- ncu at ladder, step 1 -> step 2: excess shared-memory wavefronts 14 155 776 -> 0 (cuBLASLt 0); instructions
  31 916 160 -> 27 523 584; tensor active 44.9 -> 45.9 % (cuBLASLt 87.4 %).

## Step 3: cp.async, three stages

- 3a: cp.async (`cp_async16`, cp.async.cg.shared.global.L2::128B): each 16-byte chunk goes from global to shared memory
  through the L2 only, without a register; one stage, as step 2. Alone, it saves time: judged apart from 3b.
- 3b: three stages of K64 (`gemm<3>`), two slices in flight while one is computed; one wait and one barrier per K64.
- ncu at ladder, step 2 -> step 3: tensor active 45.9 -> 68.7 %, cycles 576 596 -> 390 436 (cuBLASLt: 87.4 %, 305 975).

## Step 4: evict-first stores, a tile swizzle

- 4a: stores of C marked evict-first (`st16<true>`, st.global.cs): C is written once and never read back.
- 4b: the tiles in groups of 16 rows of tiles, column by column inside a group (`tile_of<true>`).
- ncu at ladder, step 3 -> step 4: DRAM read 78.6 -> 60.8 MB, L2 sector hits 82.2 -> 85.8 % (cuBLASLt: 60.8 MB, 85.4 %).

## Step 5: fragment double buffering, a serpentine HMMA order

- 5a: a pipeline of the fragments (`kPipe`): the next K16's ldmatrix issued among the current K16's HMMA, into a second
  set of registers; the wait and the barrier moved into the last K16 of a slice.
- 5b: the order of the HMMA (`kSerp`): B's halves in a serpentine, so that consecutive HMMA share an operand.
- ncu at ladder, step 4 -> step 5: tensor active 70.0 -> 89.3 %, cycles 381 815 -> 300 195 (the cuBLASLt configuration
  of this session: 90.1 %, 297 845).

## Step 6: the tile per shape, 16-row bands, a stream in phase

- 6a: a 128 x 128 tile, five stages of K32, two blocks per SM, where 128 x 256 would give fewer than three waves of
  tiles (`choose`): at qkv_m128, qkv_m512, o_m128, o_m512, down_m128 and down_m512.
- 6b: at M <= 16, a tile of 16 rows (rule `m16`): a band of 128 columns per block over all of K (`gemm_bands`) at
  qkv_m16 and o_m16; a stream of (band, slice) units over the SMs (`gemm_stream`) at gateup_m16 and down_m16.
- 6c: a stream in phase (`gemm_phase`, rule `phase`) where whole tiles leave the last wave mostly empty: gateup_m128 and
  gateup_m512. Step 6 keeps K whole in the tiles of `choose`; 6b's stream and 6c cut it at the ends of their ranges.
- ncu at ladder, step 5's kernel: 302 475 cycles against cuBLASLt's 307 679, tensor active 89.2 % against 87.3 %.

## Step 7: split-K, parts reduced by their own blocks, L2 hints

- 7a: K in parts (`gemm_split`, rule `choose`): 3 parts on 128 x 256 at o_m128 and down_m128, written in fp16 and added
  by a second kernel (`reduce_parts`); 2 parts on 6a's tile, added in place in C, at o_m512 and down_m512, where 7f and
  7c take over.
- 7b: at o_m128 and down_m128, the parts reduced by their own blocks (`gemm_parallel`) instead of 7a's second kernel (7a
  alone saves time, so the two are judged apart).
- 7c: at down_m512, K in 5 parts added in place on the 128 x 256 tile (`inplace_parts`).
- 7d: L2 cache hints (`HintCopy`, `discard_line`): A kept and B evicted first in the kernels of 6b, 6c, 7b, 7f and 7g;
  the parts dropped from the L2 once added in those that cut K (6b's stream, 6c, 7b, 7e, 7f and 7g): at the four M = 16
  shapes, gateup_m128, gateup_m512 and the shapes of 7e, 7f and 7g.
- 7e: the tail of the waves cut in K (`gemm_tail`, rule `tail_parts`): at M > 512, where whole 128 x 256 tiles leave the
  last wave partial, that wave's tiles in the number of parts (2 to 8) that ends the tail soonest, the last part to
  arrive adding them: at down_m2048 (4 parts) and gateup_m2048 (2).
- 7f: 6c's stream in phase also where 7a adds parts in place and 7c finds no number of parts (rule `phase_in_place`): at
  o_m512.
- 7g: 7b without the block's own part (`gemm_parallel_own`): each block keeps in registers the share of its part that it
  adds itself, writes only the rest and reads only the other blocks' parts: at o_m128 and down_m128.
- 7h: the reads of an addition issued together (`kTogether`): in the parts in place of 7a and 7c, each round of 64 rows
  of the tile issues every read of C before its additions and stores; in 6c's kernel as step 7 runs it
  (`gemm_phase_sums`), the last block of a cut tile reads each part with every read issued together: at down_m512,
  gateup_m128, gateup_m512 and o_m512.
- ncu at ladder: step 5's kernel, as in step 6; in the probes, at one shape for each of 7a to 7h: `results/README.md`.

## Limits

- One card: the A100-SXM4-80GB at its 400 W power cap, with cuBLASLt 130201 (the judge's header) and the driver
  610.57.04 (`results/*/00_machine.txt`); another card, driver or power limit can move every ratio.
- cuBLAS only, and the judge's search does not try every cuBLASLt configuration (`judge/README.md`).
- The 17 shapes only: M a multiple of 16, N of 256, K of 64 (the host checks it); the rules of steps 6 and 7 use
  thresholds chosen by hand (`step6.cuh`, `step7.cuh`).
- 7a and 7c in place, and 7b and 7g, assume what CUDA does not promise: that a tile's earlier parts start first, and
  that the blocks of a tile's parts are resident together, without a cooperative launch (`step7.cuh`).
