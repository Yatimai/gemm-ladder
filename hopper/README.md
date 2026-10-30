# The Hopper rung (H100)

The repository's GEMM (A and B fp16 row-major, fp32 accumulation, C fp16 row-major) on the H100 80GB HBM3 (sm_90a),
built in seven steps from NVIDIA's tensor-core sample. Each step is timed against the judge's reference at the 17
shapes, both defined in the repository's README (`../README.md`).

Score 1.00 (1.0094, rounded down); shapes from 0.9875 (qkv_m512) to 1.0415 (o_m128), 13 won and 2 lost by all their
pairs (`results/judge/20261030-171612_step7`, the arrival: container 2, where step 7's score is lower; container 1
shows 1.0100; every shape in both: `results/README.md`).

| step | what it adds | container 1 | container 2 |
|---|---|---|---|
| reference point | NVIDIA's `simple_wmma_gemm`: wmma from global memory, no shared memory | 0.0661 | 0.0654 |
| 0 | NVIDIA's `compute_gemm`: 128 x 128 tiles, wmma (HMMA), one persistent block per SM | 0.1686 | 0.1662 |
| 1 | wgmma (the warpgroup MMA) on a 128 x 256 tile, one block per tile, every thread copying | 0.4018 | 0.4013 |
| 2 | TMA loads into a ring of three stages, one MMA group kept in flight | 0.7243 | 0.7232 |
| 3 | a fourth stage | 0.7510 | 0.7494 |
| 4 | clusters of two blocks, set at the launch | 0.8173 | 0.8124 |
| 5 | a persistent grid; the pairs of tiles in a serpentine of 16 rows of tiles | 0.8390 | 0.8328 |
| 6 | a tile per shape: 16-row bands at M <= 16, smaller tiles where 128 x 256 leaves SMs idle | 0.9676 | 0.9643 |
| 7 | warp specialization, taller and wider bands, a 128 x 320 tile, multicast loads, TMA stores | 1.0100 | 1.0094 |

Score: the geometric mean over the 17 shapes of the reference's time over the step's (above 1: faster), in each of
two containers (`results/judge/20261030-171549_*`, `results/judge/20261030-171612_*`). Below, what each mechanism saves,
then each step; the probe tables, the folders, the counters and the list of the raw outputs: `results/README.md`.

## How it is measured

Paths start from this folder (`results/`, `emu/`) or from the repository's root (`judge/`, `probe/`, `modal/`).
- **Judge** (`judge/judge.cu`, version 6; launcher `judge/modal_judge.py`; `judge/README.md`): checks each step's
  result, then times it against the reference, with the reference cached in `judge/reference/h100.txt`, in each of two
  containers with the same card line (`NVIDIA_H100_80GB_HBM3|700W|lt130201|cuda13030`). The outputs are in
  `results/judge/`; the first two lines of each give the md5 of the files it judged.
- **Probe** (`probe/probe.cu`; launcher `probe/modal_probe.py --card H100!`): in one process, interleaved, 5 rounds of
  1 s, the step and its variants `v<i>` (each defined in `step<i>.cu`); the time per call is the median of the rounds.
  The step is timed again as an arm of its own (`twin`); the noise of a probe is the largest gap, in absolute value,
  between the step and an arm that runs exactly the same code at a shape (`twin`, or a variant whose cell shows `-`),
  in %, to one decimal, at least 0.1 %; its rms noise is the square root of the mean of the squares of these gaps,
  unrounded. At step 7, two arms run at every shape and judge nothing: step 6 (`v6`) and the cuBLASLt configuration
  ranked first for the card line in the judge's cache (`ref0`).
- **ncu**, clocks not locked: in one session per step (`modal/session.py`), at ladder, one call of the step, of each of
  its variants and of the configuration ranked first in the judge's final at ladder for that step in the arrival
  (`--cublaslt`); in the probes of step 7, one process per arm (the step, a removal, the configuration of a REFERENCE
  line: arm `x:`), the second call. Counters are compared within one session or one probe only.
- **compute-sanitizer** (memcheck, racecheck, synccheck) on the 26 shapes of `verif.cu`, in the same sessions: 0 errors,
  0 hazards displayed, 0 errors, in each of the 9 sessions that count (the first sessions of steps 1 and 7 ran out of
  their racecheck time budget, 600 s per pass, with 0 hazards on the part done, and were run again, unchanged).
  The **host emulator** (`emu/`, its `README.md`) runs every kernel on the CPU, its threads in three orders, and checks
  waits, barriers, accesses and results (`results/emu/`: 7041 runs, 7041 ok, 0 failed, `ALL OK`; 239 of 239 planted
  bugs reported as failures, the unmodified copy passes, `PLANTED-BUG TEST OK`).
- **The card**: the H100's power cap is 700 W (the judge's header and card line); its power, clocks and throttle
  reasons in the judge's outputs: `results/README.md` (container 1's output of step 4 reports thermal or hardware
  throttling at down_m128). The two scores of a row differ by up to 1.44 %, so a step is read against the previous
  step in the same container.

Each step's probe runs in two series, each in a container of its own; at each step, the two probes that decide ran at
the same time, so on two different cards (`results/two_cards.txt`; steps 1, 3 and 4 were probed again in both series:
`results/README.md`). "Faster": the geometric mean, over the shapes where a mechanism acts (the cases without a dash),
of the time without it over the time with it, minus 1, in % to one decimal (2a, 7f alone and 7h alone are measured
added, and turned the same way). A mechanism stays if, in both series, it saves more than its threshold: three times
the standard error of its mean (its probe's rms noise over the square root of its number of shapes), to one decimal;
at least 0.2 %; at least its probe's noise if it acts at 3 shapes or fewer. 4a and 4b are measured together. 7h alone
does not save: 7h and 7i count as one; 7f alone does not save: 7f counts as one with the five that run on it (7g to
7l), each of which also passes its own case (7h by the pair). All stay:

| step | mechanism | measured | shapes | faster, series 1 | threshold | faster, series 2 | threshold |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | 1a | removed with the tile and 1b (step 0) | 17 | +140.2 % | 0.2 % | +142.0 % | 0.8 % |
| 1 | the 128 x 256 tile | 128 x 128 instead | 17 | +13.0 % | 0.2 % | +13.9 % | 0.8 % |
| 1 | 1b | removed | 17 | +122.4 % | 0.2 % | +124.2 % | 0.8 % |
| 2 | 2a | added to step 1, two stages, without 2d | 17 | +61.7 % | 0.2 % | +61.8 % | 0.2 % |
| 2 | 2b | two stages instead of three, without 2d | 17 | +7.1 % | 0.2 % | +7.0 % | 0.2 % |
| 2 | 2d | removed | 17 | +3.5 % | 0.2 % | +3.8 % | 0.2 % |
| 3 | 3a | removed (step 2) | 17 | +4.5 % | 0.2 % | +3.9 % | 0.2 % |
| 4 | 4a and 4b | removed together (step 3) | 17 | +6.5 % | 0.3 % | +8.5 % | 0.2 % |
| 5 | 5a | removed | 9 | +0.9 % | 0.3 % | +0.4 % | 0.2 % |
| 5 | 5b | removed | 17 | +1.7 % | 0.2 % | +1.6 % | 0.2 % |
| 6 | 6a | removed | 4 | +37.8 % | 0.2 % | +37.5 % | 0.2 % |
| 6 | 6b | removed | 3 | +41.2 % | 0.3 % | +40.1 % | 0.3 % |
| 6 | 6c | removed | 9 | +1.7 % | 0.2 % | +1.6 % | 0.2 % |
| 7 | 7a | removed | 2 | +2.1 % | 1.4 % | +2.3 % | 1.1 % |
| 7 | 7b | removed | 3 | +2.0 % | 1.4 % | +2.0 % | 1.1 % |
| 7 | 7c | removed | 7 | +0.7 % | 0.3 % | +0.4 % | 0.3 % |
| 7 | 7d | removed | 1 | +2.4 % | 1.4 % | +2.5 % | 1.1 % |
| 7 | 7e | removed | 2 | +15.7 % | 1.4 % | +15.9 % | 1.1 % |
| 7 | 7e alone in one wave | in pairs instead | 1 | +2.2 % | 1.4 % | +2.0 % | 1.1 % |
| 7 | 7f alone | added to step 7 without 7f and the five | 17 | -0.8 % | 0.2 % | -0.5 % | 0.2 % |
| 7 | 7f and the five | removed together | 17 | +1.0 % | 0.2 % | +1.0 % | 0.2 % |
| 7 | 7g | removed | 15 | +0.8 % | 0.2 % | +0.6 % | 0.2 % |
| 7 | 7h alone | added to step 7 without 7h and 7i | 7 | +0.2 % | 0.3 % | +0.2 % | 0.3 % |
| 7 | 7h and 7i | removed together | 7 | +0.7 % | 0.3 % | +0.6 % | 0.3 % |
| 7 | 7i | removed | 7 | +0.5 % | 0.3 % | +0.5 % | 0.3 % |
| 7 | 7j | removed | 10 | +0.5 % | 0.3 % | +0.5 % | 0.2 % |
| 7 | 7l | removed | 7 | +0.9 % | 0.3 % | +0.9 % | 0.3 % |

## Step 0: NVIDIA's compute_gemm, adapted

`compute_gemm`, from cuda-samples v12.4 (`cudaTensorCoreGemm`, `nvidia_sample.cuh`), for sm_90a: wmma gives sm_80's
HMMA.16816, no warpgroup MMA; 128 x 128 tiles, one persistent block per SM; reference point: its `simple_wmma_gemm`.
ncu at ladder: tensor active 17.6 % against cuBLASLt's 97.7 %; 6 828 095 cycles against 825 796.

## Step 1: wgmma, a 128 x 256 tile, every thread copying

- 1a: wgmma.mma_async m64n256k16 (`step1.cuh`): a warpgroup's 128 threads issue one MMA of 64 x 256 x K16, A and B
  read in shared memory by descriptors at the 128-byte swizzle; two warpgroups per tile, one block per tile, two stages.
- The 128 x 256 tile, where `v1` takes 128 x 128 (wgmma m64n128k16).
- 1b: every thread copies its share of A and B (12 chunks of 16 bytes per K64), all its loads before its stores, where
  step 0 gives A to warps 0-3 and B to warps 4-7 (`gemm_split`).
- ncu at ladder: tensor active 38.5 % against cuBLASLt's 97.6 %; 2 058 372 cycles against 829 002.

## Step 2: TMA loads in three stages, one MMA group in flight

- 2a: the TMA (cp.async.bulk.tensor.2d, `step2.cuh`): one thread loads a slice, one instruction per box, the boxes
  landing at the 128-byte swizzle the descriptors read (2c, measured with 2a); an mbarrier per stage counts the bytes.
- 2b: the ring of three stages, a full and an empty barrier each, two slices ahead of the MMA.
- 2d: wgmma.wait_group 1: a slice's MMA group stays in flight while the ring goes on (without it, wait_group 0).
- ncu at ladder, step 1 -> step 2: tensor active 38.2 -> 87.7 % (cuBLASLt 97.6 %), cycles 2 069 076 -> 922 640.

## Step 3: a fourth stage

- 3a: a ring of four stages, three slices ahead of the MMA (`step2::gemm<4>`, `step3.cuh`).
- ncu at ladder, step 2 -> step 3: tensor active 87.8 -> 95.4 % (cuBLASLt 97.6 %); 923 789 -> 858 405 cycles (831 134).

## Step 4: clusters of two

- 4a: clusters of two blocks (`step4.cuh`): two tiles one above the other with two rows of tiles or more, side by side
  with one row (M = 16 and 128); each block loads its own A and B; the cluster synchronized once its barriers are set.
- 4b: the cluster set at the launch (cudaLaunchKernelEx with a cluster dimension), not by `__cluster_dims__`.
- ncu at ladder, step 3 -> step 4: 910 513 -> 851 798 cycles (cuBLASLt 830 538); tensor active 92.3 -> 95.7 % (97.6 %).

## Step 5: a persistent grid, a tile swizzle

- 5a: a persistent grid (`step5.cuh`): as many clusters as the GPU holds at once, each going through the pairs of tiles;
  the ring runs on, the next pair's first slices landing while the current pair's output is written.
- 5b: the order of the pairs: groups of 16 rows of tiles, column after column, the odd groups from the last column back.
- ncu at ladder, step 4 -> step 5: DRAM read 629.3 -> 511.8 MB (cuBLASLt 427.8 MB); 849 289 -> 827 761 cycles (827 069).

## Step 6: 16-row bands, the tile per shape

- 6a: M <= 16, bands (`step6::bands`): 16 rows by J columns of 64 over all of K, transposed (C^T = B^T A^T, wgmma
  m64n16k16), J the least that makes one wave; up to 8 stages; at the four M = 16 shapes.
- 6b: 16 < M <= 128 with one row of tiles: 64 x 128 (in pairs along M, or alone) or 128 x 128 when its tiles make one
  wave, on a persistent kernel (`step6::gemm`), at qkv_m128, o_m128 and down_m128; else step 5's tile.
- 6c: M > 128: step 5's tile, 128 x 256, alone (no cluster) on 6b's persistent kernel: at the nine shapes with M > 128.
- ncu at ladder (6c): 826 806 cycles against cuBLASLt's 826 087; DRAM read 511.7 MB against 427.8 MB.

## Step 7: warp specialization, taller and wider bands, a 128 x 320 tile

Eleven mechanisms, each a rule by classes of shapes with its variant that takes it out (`step7.cu`, `step7.cuh`):
- 7a: 64 < M <= 128: 6a's bands at 64 rows, in pairs along M, when they make one wave: at o_m128 and down_m128.
- 7b: M <= 16: the bands twice as wide, in clusters of eight along N: at qkv_m16, o_m16 and down_m16.
- 7c: 6c's waves as whole rectangles of tiles (`gemm_rows`); in one wave, the L2 promotion at 256 bytes: at 7 shapes.
- 7d: 64 < M <= 128 where 7a's pairs do not make one wave: the bands at 128 rows, alone: at qkv_m128.
- 7e: M > 128: a 128 x 320 tile (wgmma m64n192 then m64n128 per warpgroup) where its waves times its area are fewer
  than 6c's; alone in one wave (one tile per block), else in pairs along M: at qkv_m512 and gateup_m2048.
- 7f: warp specialization: a loading warp beside the bands' computing warpgroup, a loading warpgroup beside the tiles'
  two (setmaxnreg: down to 24, up to 232 in 6c's and 7e's tiles); one elected lane issues each stage's boxes.
- 7g: the descriptors in the uniform datapath: computed once from shuffled indices, then advanced by additions.
- 7h: the shared operand, multicast by the TMA in clusters: in 7b's clusters, A's 16 rows loaded once for the eight; in
  7a's pairs and in the pairs of tiles, each block loads a share of what both read.
- 7i: the end of the loading warp: it waits until the cluster has released its last stages before its block leaves.
- 7j: the tiles' output by TMA stores from each warpgroup's zone, issued by one thread and not waited for.
- 7l: an L2 evict_first hint on the bands' TMA loads of B, read once.
- ncu at ladder (7c's `gemm_rows`): 825 348 cycles against cuBLASLt's 824 822, DRAM read 427.9 MB against 427.9 MB
  (6c's order, `v7`: 511.8 MB); the other probes: `results/README.md`. `ref0` over `v0`: 1.0054 and
  1.0046 (series 1 and 2); over `v6`: 0.9623 and 0.9657 (geometric means over the 17 shapes; not the judge's score).

## Limits

- One card: the H100 80GB HBM3 at its 700 W power cap, with cuBLASLt 130201 (the judge's header) and the driver
  610.57.04 (`results/*/00_machine.txt`); another card, driver or power limit can move every ratio.
- cuBLAS only, and the judge's search does not try every cuBLASLt configuration (`judge/README.md`).
- The 17 shapes only: M a multiple of 16, N of 64 (of 128 at step 0, of 256 at steps 1 to 5), K of 64 (the hosts check
  it); the rules of steps 6 and 7 use thresholds chosen by hand (`step6.cuh`, `step7.cuh`).
