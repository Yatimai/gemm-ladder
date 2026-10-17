# The Ampere rung (A100-SXM4-80GB)

The repository's GEMM (A and B fp16 row-major, fp32 accumulation, C fp16 row-major) on the A100 (sm_80), built in seven
steps from NVIDIA's tensor-core sample. Each step is timed against the judge's reference at the 17 shapes of
`judge/shapes/A100.txt`, both defined in the repository's README (`../README.md`).

Score 1.01; shapes from 0.9774 (qkv_m128) to 1.0781 (o_m128) (`results/judge/20261017-190653_step7`, the arrival:
container 2, where step 7 scores lower). Step 6 stays under the reference; step 7 cuts K at four more shapes, where the
reference splits it too (`splitk` above 1 in the `reference` column of step 7's outputs), and adds three refinements.

| step | what it adds | container 1 | container 2 |
|---|---|---|---|
| reference point | NVIDIA's `simple_wmma_gemm`: wmma from global memory, no shared memory | 0.1029 | 0.1038 |
| 0 | NVIDIA's `compute_gemm`: 128 x 128 tiles, wmma, a persistent kernel (one block per SM) | 0.2739 | 0.2767 |
| 1 | one block per tile, a 128 x 256 tile, the copy spread over all threads | 0.3814 | 0.3844 |
| 2 | mma.sync and ldmatrix instead of wmma, a swizzle instead of the padding | 0.3864 | 0.3894 |
| 3 | cp.async through the L2 only, three shared stages | 0.5674 | 0.5708 |
| 4 | stores of C marked evict-first, the tiles in groups of 16 rows of tiles | 0.5832 | 0.5874 |
| 5 | the next fragments loaded among the current HMMA, a serpentine order of the HMMA | 0.7184 | 0.7214 |
| 6 | a tile per shape: 128 x 128, 16 rows at M = 16, a stream in phase | 0.9760 | 0.9772 |
| 7 | K in parts at four shapes, reduced by their own blocks or in place; L2 cache hints | 1.0182 | 1.0180 |

Score: the geometric mean over the 17 shapes of the reference's time over the step's (above 1: faster than the
reference), in each of two containers, numbered by the time stamps of their outputs (`results/judge/20261017-190559_*`
and `results/judge/20261017-190653_*`). The probe tables, the counters and the raw outputs: `results/README.md`.

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
  (in %, to one decimal, at least 0.1 %). Under ncu, one process per arm, the step, a removal and the configuration of a
  REFERENCE line (arm `x:`); the counters cited are those of the arm's second call.
- **ncu** (`modal/session.py`): at ladder, clocks not locked, in one session per step (its folder: `results/README.md`),
  one call of the step, of each of its variants and of the configuration ranked first in the judge's final at ladder for
  that step in the arrival (`--cublaslt`). Counters are compared within one session (each step's own) or one probe only.
- **compute-sanitizer** on the shapes of `verif.cu`, in the same sessions: memcheck 0 errors, racecheck 0 hazards
  displayed, synccheck 0 errors, in each of the 9. The **host emulator** (`emu/`) runs every kernel on the CPU and
  checks its accesses, barriers, bank conflicts and results (`results/emu/`: `emu_all.txt`, 767 runs, 767 ok, `ALL OK`;
  `planted_bugs.txt`, 130 of 130 planted bugs reported as failures, `PLANTED-BUG TEST OK`).
- **The card**: power cap 400 W (the card line; power, clocks, throttle reasons: `results/README.md`).
  The two scores of a row differ by 0.02 to 1.02 %, so a step is read against the previous step in the same container.

Each step's probe runs in two series, each run in a container of its own; at each step, the two probes that decide ran
at the same time, so on two different cards (`results/two_cards.txt`). "Faster" is the geometric mean, over the shapes
where a mechanism acts (the cases without a dash), of the time without it over the time with it, minus 1, in % to one
decimal. Most mechanisms are measured removed from the step; 1d, 3a and 7a are measured added to the version without
them, their cases (`results/README.md`) turned here the same way. A mechanism stays if, in both series, it saves more
than twice the noise of its probe; two that work only together count as one (2a and 2b). All stay:

| step | mechanism | measured | shapes | faster, series 1 | twice the noise | faster, series 2 | twice the noise |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | 1a | removed | 17 | +3.5 % | 0.2 % | +3.6 % | 0.2 % |
| 1 | 1c | removed | 17 | +5.4 % | 0.2 % | +6.2 % | 0.2 % |
| 1 | 1d | added to step 0 | 17 | +24.9 % | 0.2 % | +25.2 % | 0.2 % |
| 2 | 2a and 2b | removed together (step 1) | 17 | +1.2 % | 0.2 % | +1.2 % | 0.2 % |
| 3 | 3a | added to step 2 | 17 | +10.0 % | 0.2 % | +10.0 % | 0.2 % |
| 3 | 3b | removed (3a alone, one stage) | 17 | +34.1 % | 0.2 % | +34.1 % | 0.2 % |
| 4 | 4a | removed | 17 | +1.2 % | 0.2 % | +1.3 % | 0.2 % |
| 4 | 4b | removed | 17 | +1.7 % | 0.2 % | +1.6 % | 0.2 % |
| 5 | 5a | removed | 17 | +21.8 % | 0.2 % | +21.5 % | 0.2 % |
| 5 | 5b | removed | 17 | +0.6 % | 0.2 % | +0.6 % | 0.2 % |
| 6 | 6a | removed | 6 | +37.1 % | 0.2 % | +37.7 % | 0.2 % |
| 6 | 6b | removed | 4 | +152.4 % | 0.2 % | +142.8 % | 0.2 % |
| 6 | 6c | removed | 2 | +2.6 % | 0.2 % | +1.8 % | 0.2 % |
| 7 | 7a | added without 7b (7c at down_m512) to step 7 without 7a | 4 | +13.2 % | 1.2 % | +12.8 % | 1.2 % |
| 7 | 7b | removed, with 7d's hints there | 2 | +7.7 % | 1.2 % | +8.1 % | 1.2 % |
| 7 | 7c | removed | 1 | +4.1 % | 1.2 % | +4.5 % | 1.2 % |
| 7 | 7d | removed | 8 | +1.7 % | 1.2 % | +1.8 % | 1.2 % |

## Step 0: NVIDIA's compute_gemm, adapted

`compute_gemm`, from cuda-samples v12.4 (`cudaTensorCoreGemm`), the Turing rung's step 0 line for line, on sm_80:
128 x 128 tiles, 8 warps of 32 x 64, one K64 slice copied to padded shared rows between two barriers, one persistent
block per SM. The reference point is the same sample's `simple_wmma_gemm`. ncu at ladder: tensor active 24.4 % against
cuBLASLt's 87.3 %; 1 079 924 cycles against 307 396.

## Step 1: granularity

- 1a: one block per tile (`kOneTile`), where step 0 runs one block per SM over a list of tiles.
- 1b, the template's double buffer (the Turing rung's 1a), is not taken: step 3 copies asynchronously.
- 1c: a 128 x 256 tile in 8 warps of 64 x 64 (`kBN` = 256), where step 0 has 128 x 128 in warps of 32 x 64.
- 1d: every thread copies its share of A and B, where step 0 gives A to warps 0-3 and B to 4-7.
- ncu at ladder: tensor active 44.8 % against cuBLASLt's 87.4 %; 590 701 cycles against 306 760.

## Step 2: the path to the tensor cores

- 2a: mma.sync m16n8k16 and ldmatrix.x4 instead of wmma (`mma_m16n8k16`, `ldmatrix_x4`): the same HMMA and LDSM as wmma
  on sm_80, the fragment addresses held in registers across the loop.
- 2b: a swizzle of the shared stage instead of the padding (`Geometry<true>`): no bank conflict by the headers' bank
  model. wmma cannot read it, so 2b exists only on top of 2a; 2a alone costs time, so the two count as one.
- ncu at ladder, step 1 -> step 2: excess shared-memory wavefronts 14 155 776 -> 0 (cuBLASLt 0); instructions
  31 916 160 -> 27 523 584; tensor active 44.6 -> 45.9 % (cuBLASLt 87.3 %).

## Step 3: Ampere's asynchronous copy

- 3a: cp.async (`cp_async16`, cp.async.cg.shared.global.L2::128B): each 16-byte chunk goes from global to shared memory
  through the L2 only, without a register; one stage, as step 2. Alone, it saves time: judged apart from 3b.
- 3b: three stages of K64 (`gemm<3>`), two slices in flight while one is computed; one wait and one barrier per K64.
- ncu at ladder, step 2 -> step 3: tensor active 46.0 -> 68.8 %, cycles 575 495 -> 387 766 (cuBLASLt: 87.4 %, 305 998).

## Step 4: the memory hierarchy

- 4a: stores of C marked evict-first (`st16<true>`, st.global.cs): C is written once and never read back.
- 4b: the tiles in groups of 16 rows of tiles, column by column inside a group (`tile_of<true>`).
- ncu at ladder, step 3 -> step 4: DRAM read 79.6 -> 58.3 MB, L2 sector hits 81.8 -> 86.2 % (cuBLASLt: 57.6 MB, 85.9 %).

## Step 5: the issue

- 5a: a pipeline of the fragments (`kPipe`): the next K16's ldmatrix issued among the current K16's HMMA, into a second
  set of registers; the wait and the barrier moved into the last K16 of a slice.
- 5b: the order of the HMMA (`kSerp`): B's halves in a serpentine, so that consecutive HMMA share an operand.
- ncu at ladder, step 4 -> step 5: tensor active 69.9 -> 89.2 %, cycles 382 712 -> 301 075 (cuBLASLt: 87.3 %, 306 703).

## Step 6: the schedule per shape

- 6a: a 128 x 128 tile, five stages of K32, two blocks per SM, where 128 x 256 would give fewer than three waves of
  tiles (`choose`): at qkv_m128, qkv_m512, o_m128, o_m512, down_m128 and down_m512.
- 6b: at M <= 16, a tile of 16 rows (rule `m16`): a band of 128 columns per block over all of K (`gemm_bands`) at
  qkv_m16 and o_m16; a stream of (band, slice) units over the SMs (`gemm_stream`) at gateup_m16 and down_m16.
- 6c: a stream in phase (`gemm_phase`, rule `phase`) where whole tiles leave the last wave mostly empty: gateup_m128 and
  gateup_m512. Step 6 keeps K whole in the tiles of `choose`; 6b's stream and 6c cut it at the ends of their ranges.
- ncu at ladder, step 5's kernel: 302 172 cycles against cuBLASLt's 306 786, tensor active 89.1 % against 87.3 %.

## Step 7: the refinements

- 7a: K in parts (`gemm_split`, rule `choose`): 3 parts on 128 x 256 at o_m128 and down_m128, written in fp16 and added
  by a second kernel (`reduce_parts`); 2 parts on 6a's tile, added in place in C, at o_m512 (at down_m512, 7c instead).
- 7b: at o_m128 and down_m128, the parts reduced by their own blocks (`gemm_parallel`) instead of 7a's second kernel (7a
  alone saves time, so the two are judged apart).
- 7c: at down_m512, K in 5 parts added in place on the 128 x 256 tile (`inplace_parts`).
- 7d: L2 cache hints in the kernels of 6b, 6c and 7b (`HintCopy`, `discard_line`): A kept, B evicted first, the parts
  dropped from the L2 once added.
- ncu at ladder: step 5's kernel, as in step 6; in the probes, at one shape for each of 7a to 7d: `results/README.md`.

## Limits
- One card: the A100-SXM4-80GB at its 400 W power cap, with cuBLASLt 130201 (the judge's header) and the driver
  610.57.04 (`results/*/00_machine.txt`); another card, driver or power limit can move every ratio. cuBLAS only, and the
  judge's search does not try every cuBLASLt configuration (`judge/README.md`).
- The 17 shapes only: M a multiple of 16, N of 256, K of 64 (the host checks it); the rules of steps 6 and 7 use
  thresholds chosen by hand (`step6.cuh`, `step7.cuh`).
- 7a and 7c in place, and 7b, assume what CUDA does not promise: that a tile's earlier parts start first, and that the
  blocks of a tile's parts are resident together, without a cooperative launch (`step7.cuh`).
