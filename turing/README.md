# The Turing rung (Tesla T4)

The rung rebuilds, one mechanism at a time, the fp16 GEMM of cuBLAS on the T4 (sm_75), from NVIDIA's own tensor-core
sample to a kernel of our own. The format is fixed: A (M x K) and B (K x N) fp16 row-major, fp32 accumulation, C (M x N)
fp16 row-major; a shape is written M x N x K. The target at each shape is the judge's **reference**: the faster of
cuBLAS's default call (cublasGemmEx) and the fastest cuBLASLt configuration the judge's search finds (the first
configurations cuBLASLt's heuristic offers at several workspace sizes, and variations around the first of each family:
judge/README.md; below, "cuBLASLt's best configuration").

## Progression

One pass of the judge (cached reference) over every step, in each of two containers (Modal GPU instances, possibly two
cards) with the same card line: card, power cap, cuBLASLt, driver (results/judge/20261009-163456_* and
20261009-163524_*). Score: the geometric mean over the 17 shapes of the reference's time over the step's (above 1:
faster than the reference). A shape is won when every pair of the judge's measures is above 1, lost when every pair is
below.

| step | mechanisms | container 1 | container 2 |
|---|---|---|---|
| reference point | NVIDIA's `simple_wmma_gemm`: the wmma API alone, without shared memory | 0.1890 | 0.1912 |
| 0 | NVIDIA's `compute_gemm`: 128 x 128 tiles, wmma, one K64 slice between two barriers, one block per SM | 0.6192 | 0.6225 |
| 1 | the granularity of cuBLAS: one block per tile, double buffering, a 128 x 256 tile, every thread copying both operands | 0.7616 | 0.7642 |
| 2 | the path to the tensor cores: mma.sync and ldmatrix.x4 instead of wmma, a swizzle instead of the padding | 0.7987 | 0.8017 |
| 3 | the memory hierarchy: read-only loads with the L2's prefetch, evict-first stores, tiles in groups of 8 rows (our size) | 0.8431 | 0.8464 |
| 4 | instruction issue: fragment pipelining, an order of the HMMA that reuses an operand | 0.9150 | 0.9197 |
| 5 | cuBLASLt's choice of tile: 64 rows at M <= 64, 128 columns where 256 would leave more than half the SMs idle | 1.0041 | 1.0111 |
| 6 | beyond cuBLASLt: loads through the L2 only, synchronized waves, a kernel of its own at M = 16, K in two where step 5 takes 128 x 128 | 1.0750 | 1.0721 |

## How it is measured

Paths below start from this folder (as `results/`) or from the repository's root (`judge/`, `probe/`, `modal/`,
`turing/`).

- **The judge** (`judge/judge.cu`, md5 26f43859; launcher `judge/modal_judge.py`; see `judge/README.md`) times a
  candidate at 17 shapes: the four GEMM of a Llama-3-8B layer (hidden size 4096, 8 key-value heads of 128, intermediate
  size 14336), qkv (N 6144, K 4096), o (4096, 4096), gateup (28672, 4096) and down (4096, 14336), each at M = 16, 128,
  512 and 2048 (qkv_m16 is the qkv GEMM at M = 16, and so on), and 2048 x 2560 x 2048, this ladder's own shape, named
  ladder. Every score below comes from one pass of the judge with the cached reference (the configurations it has found
  best on this card, 117 for this card line in `judge/reference/t4.txt`, ranked again at each run, with cuBLAS's default
  call in the final), every step run in each of two containers with the same card line
  (`Tesla_T4|70W|lt130201|cuda13030`). The judge's outputs are in `results/judge/`; the first line of each gives the md5
  of the sources it judged, those of this repository.
- **The probe** (`probe/probe_t4.cu`; launcher `probe/modal_probe_t4.py`): an interleaved A/B in one process, the step
  and each of its removals (the step with one mechanism taken out: the variants of `step<i>.cu`, `g_var`) and cuBLASLt's
  heuristic's first configuration (arm lt), 5 rounds; time, clock and energy per GEMM. It measures what a mechanism is
  worth; the judge measures the step.
- **ncu** at ladder (`modal/session.py`, clocks not locked): the step, its removals and cuBLASLt's best configuration in
  one session. Tensor activity (sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active) and cycles
  (sm__cycles_elapsed.max) are compared within a session only; DRAM bytes read (dram__bytes_read.sum), L2 sector hits
  (lts__t_sector_hit_rate.pct) and instruction counts from one session to the next. The counters are in the
  `*_raw.csv` files of each session's folder (the ncu reports themselves are not in the repository).
- **compute-sanitizer** (memcheck, racecheck, synccheck) on `verif.cu`'s shapes, and the **host emulator**
  (`turing/emu`), which runs every kernel on the CPU and checks its accesses, its barriers, its bank conflicts and its
  results (to the bit where the headers say so) before any GPU session. On the T4, `verif.cu` under compute-sanitizer
  gives the emulator's error figures at the six shapes the two share for every step, from 16 x 256 x 64 to 1168 x 512 x
  64, 2.34e-04 to 3.96e-04 (results/20261009-150509_step1/step1_01_sanitizer.txt; equal within the rounding of C to
  fp16, not the same bits), and ncu the excess shared-memory wavefronts of the emulator's bank model at ladder
  (results/20261009-150440_step2, below).
- The SASS counts in the headers are read with nvcc 13.2; the measured binaries are built by nvcc 13.1.1 in the Modal
  images, with the same registers under ncu.

Two facts of the card shape every reading. The T4 runs at its 70 W cap under these GEMM: the clock falls until the power
fits, so time follows the energy a GEMM spends, and a mechanism can win cycles and lose time. And the same code scores
differently from one container to the next: in the common pass, the two containers differ by 0.27 to 0.70 % on the same
step, 1.16 % on the reference point (the table above). A step is therefore read against the previous step in the same
container, and every score is given for both.

## Step 0: NVIDIA's sample

NVIDIA's `compute_gemm` (cuda-samples `cudaTensorCoreGemm`), adapted as little as possible: fp16 output, B row-major, an
edge in M. 128 x 128 tiles, 8 warps of 32 x 64, wmma 16 x 16 x 16, rows padded by 16 halves (32 bytes), one K64 slice
copied between two barriers, one persistent block per SM, the output through shared memory. `simple_wmma_gemm`, from the
same sample, is the reference point.
- Judge, the common pass (results/judge/20261009-163456_* and 20261009-163524_*): step 0 0.6192 and 0.6225 in the two
  containers, the reference point 0.1890 and 0.1912; 0 and 0 shapes won.
- ncu at ladder (results/20261009-150505_step0-ref_simple): tensor pipe active 37.4 % of the active cycles, cuBLASLt
  88.6 %; 1 430 002 cycles against 613 409; 119.5 MB read from DRAM against 52.6, 73.3 % of L2 sector hits against 88.8.

## Step 1: the granularity of cuBLAS

Four mechanisms of cuBLAS's T4 kernel (turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn under ncu, for
C^T): one block per tile (1a), two shared stages of K32 filled through registers, one barrier per K32 (1b), a 128 x 256
tile in 8 warps of 64 x 64 (1c), every thread copying its share of both operands (1d).
- Judge, the common pass (results/judge/20261009-163456_* and 20261009-163524_*): step 1 0.7616 and 0.7642 in the two
  containers, step 0 0.6192 and 0.6225; 1 and 1 shapes won.
- Probe, 5 rounds (results/20261009-1505_step1_probe): at ladder, step 1 895.0 us; without 1c 1172.6 (+31.0 %); without
  1b 911.6 (+1.9 %); without 1a 894.7 (0.0 %); 1d alone against step 0, 1131.7 against 1211.3 (-6.6 %). 1a pays at
  gateup_m2048 (without it +5.7 %) and costs at o_m128 (without it -5.9 %).
- ncu at ladder (results/20261009-150509_step1): tensor active 37.5 -> 53.6 %; cycles 1 429 219 -> 1 005 716; DRAM read
  119.8 -> 66.3 MB. cuBLASLt's kernel issues as many 16-byte global loads as step 1 (491 520 at ladder, 6 per K32 and
  per warp).

## Step 2: the path to the tensor cores

Two mechanisms, nested: mma.sync m16n8k8 and ldmatrix.x4 instead of wmma (2a: per K32 and per warp, 16 LDSM.x4 instead
of 32 LDSM.x2; ncu counts 16 LDSM in cuBLAS's kernel), and a swizzle of the shared stages instead of the padding (2b: no
bank conflict, 48 KB of stages instead of 58).
- Judge, the common pass (results/judge/20261009-163456_* and 20261009-163524_*): step 2 0.7987 and 0.8017 in the two
  containers, step 1 0.7616 and 0.7642; 1 and 1 shapes won.
- Probe, 5 rounds (results/20261009-1505_step2_probe), step 2 / 2a alone / step 1: ladder 836.5 / 920.5 / 909.2 us;
  gateup_m2048 21 656 / 23 340 / 22 759; qkv_m512 1170.7 / 1299.6 / 1286.3; o_m128 296.5 / 324.4 / 303.8. 2b gains at
  the four shapes (-7.2 to -9.9 %); 2a alone costs at the four (+1.0 % at qkv_m512 to +6.8 % at o_m128): it pays
  only with 2b.
- ncu at ladder (results/20261009-150440_step2): LDSM instructions 2 621 440 -> 1 310 720 and excess shared-memory
  wavefronts 5 898 240 -> 0, both as predicted before the session; tensor active 53.5 -> 60.7 % (cuBLASLt 87.6 %).
  cuBLASLt's kernel: 49 152 bytes of static shared memory and 16 384 dynamic, no excess wavefront; per K32 and per
  warp, 128 HMMA (its tile), 16 LDSM and 6 global loads, as ours; fewer instructions in all, 15 377 920 against step 2's
  15 973 120, and fewer integer instructions than ours under ncu.

## Step 3: the memory hierarchy

Three mechanisms of cuBLAS's kernel between DRAM, L2 and the SMs, the first two seen under ncu: read-only loads asking
L2 for a whole 128-byte line (3a, ld.global.nc.L2::128B), stores of C marked evict-first (3b, st.global.cs), and the
tiles in groups of 8 rows of tiles, column by column (3c: cuBLAS's swizzle flag; the 8 is ours).
- Judge, the common pass (results/judge/20261009-163456_* and 20261009-163524_*): step 3 0.8431 and 0.8464 in the two
  containers, step 2 0.7987 and 0.8017; 5 and 6 shapes won.
- Probe, 5 rounds (results/20261009-1505_step3_probe), step 3 against step 2: ladder 860.2 against 874.2 us (-1.6 %),
  gateup_m2048 18 990 against 24 166 (-21.4 %), qkv_m512 1118.0 against 1203.6 (-7.1 %), o_m128 -1.2 %, qkv_m16 +0.9 %.
  Without 3c: gateup_m2048 +29.0 %, qkv_m512 +9.1 %, ladder +0.7 %; without 3a: ladder +4.4 %, gateup_m2048 +5.8 %;
  without 3b: -0.3 to +0.1 % at ladder, gateup_m2048 and qkv_m512, -0.3 % at o_m128 and +0.7 % at qkv_m16.
- ncu at ladder (results/20261009-150455_step3): DRAM read 63.9 -> 49.9 MB (cuBLASLt 52.7), L2 sector hits 80.8 ->
  88.8 % (88.8); without 3c 61.1 MB and 82.2 %, without 3a 57.8 MB and 85.6 %. Tensor active 62.3 % against cuBLASLt's
  87.9 %.

## Step 4: instruction issue

Two mechanisms of cuBLAS's kernel: fragment pipelining (4a; ours: the fragments of one k8 slice at a time, in two sets
of registers, the next slice's LDSM issued among the current slice's HMMA, the barrier surrounded by HMMA) and an order
of the HMMA that reuses an operand (4b; ours reuses B's operand in 3 HMMA of 4).
- Judge, the common pass (results/judge/20261009-163456_* and 20261009-163524_*): step 4 0.9150 and 0.9197 in the two
  containers, step 3 0.8431 and 0.8464; 12 and 12 shapes won.
- ncu at ladder (results/20261009-150442_step4): tensor active 83.1 % against cuBLASLt's 88.1 %; 657 070 cycles against
  615 176; DRAM read 49.5 MB (52.6).
- ncu at ladder, the same session, step 3's kernel, per instruction (its report, not published): before 4a, most of
  the loop's mio_throttle samples sit on LDSM, and most of its short_scoreboard ones on one HMMA, the first to read an
  LDSM issued right after the burst that follows a barrier.
- Probe, 5 rounds (results/20261009-1505_step4_probe), step 4 / without 4b (4a alone) / step 3, time, clock, energy per
  GEMM: ladder 805.1 / 810.2 / 829.4 us, 782 / 774 / 1015 MHz, 57.1 / 57.0 / 57.4 mJ; qkv_m2048 3971.3 / 3987.8 /
  4027.3 us, 769 / 763 / 1014 MHz, 267.7 / 272.1 / 299.0 mJ; o_m128 243.6 / 244.6 / 292.3 us, 1227 / 1220 / 1361 MHz,
  16.7 / 16.8 / 20.1 mJ. 4a takes 25 to 26 % of the cycles off at the three shapes (time x clock): at ladder and
  qkv_m2048 the clock falls from ~1015 to ~770 MHz under the 70 W cap and the time only by 2.3 and 1.0 %, the energy per
  GEMM by 0.7 and 9.0 %; at o_m128, 16 blocks for 40 SMs, the time falls by 16.3 %. 4b: -0.6 % at ladder, -0.4 % at
  o_m128 and at qkv_m2048, at the edge of the probe's resolution.
- Tried and left out: a lighter loop, as cuBLAS's (fewer integer instructions under ncu). With step 3's copy of B it
  tied with step 4 at the judge; with another copy of B, our variant (thread t copies row t / 8 of the slice, chunks t %
  8 + 8 j, its 4 loads at one pointer), it read more DRAM at ladder and took more time at the large shapes (drafts, not
  published). Our reading: under the T4's 70 W cap, time follows energy, and the integer instructions it saves cost
  little of it.

## Step 5: cuBLASLt's choice of tile, shape by shape

Step 4's kernel at the tile cuBLASLt's best configuration uses at each shape: 64 rows of M when M <= 64 (5a), 128
columns when 256 would give fewer blocks than half the SMs (5b). Four tiles: 128 x 256 (step 4's, instruction for
instruction), 128 x 128, 64 x 256, 64 x 128 (cuBLAS's own 64 x 128, of another family, "sliced" under ncu in earlier
unpublished measures, is not reproduced).
- Judge, the common pass (results/judge/20261009-163456_* and 20261009-163524_*): step 5 1.0041 and 1.0111 in the two
  containers, step 4 0.9150 and 0.9197; 13 and 13 shapes won.
- Probe, 5 rounds (results/20261009-1505_step5_probe), step 5 against step 4: qkv_m16 202.2 against 289.1 us,
  o_m16 150.1 against 229.1, gateup_m16 1012.1 against 1407.4, down_m16 547.9 against 809.1 (-28 to -34 %); o_m128 241.0
  against 245.8 (-2.0 %), down_m128 835.6 against 843.2 (-0.9 %). 5a carries the M = 16 shapes (without it, +37 to
  +42 %); without 5b: +2.2 and +0.9 % at o_m128 and down_m128, -1.2 % at o_m16 and -4.7 % at down_m16.
- ncu in the probe (results/20261009-1505_step5_probe, ncu_lt_16x4096x14336_raw.csv): at down_m16, cuBLASLt's heuristic
  configuration (the probe's lt arm; the judge's best there is 64 x 128) runs a 64 x 256 kernel with the block of
  step 5's 64 x 256 (128 threads, 40 960 bytes of shared memory), over 128 blocks where ours has 16, and 234 registers
  (240 here).
- compute-sanitizer and ncu at ladder (results/20261009-150458_step5): `verif.cu` passes for step 5's variants at its
  shapes (largest error over max |ref| 3.96e-04, tolerance 2e-3), with no error under memcheck and synccheck and no
  hazard under racecheck. At ladder, where neither 5a nor 5b acts, step 5 runs step 4's kernel: 15 979 520 instructions
  executed for both, 653 352 cycles against 657 432, tensor active 82.2 % for both (cuBLASLt 611 343 cycles, 87.7 %).
- Tried and left out: the rows of A past M loaded as zeros instead of a copy of row M - 1: at gateup_m16 energy per GEMM
  fell and clock rose, time hardly moved; qkv_m16, o_m16 and down_m16, which wait on DRAM, lost time (a draft,
  not published).

## Step 6: beyond cuBLASLt

Four mechanisms of our own, first tried on our earlier T4 kernels (written before this rung, not in this repository) and
grafted here on step 5, each measured on this rung: loads of A and B through the L2 only (6a, ld.global.cg.L2::128B, at
every tile); synchronized waves at M >= 512, ladder included (6b: one persistent block per SM, a barrier of the whole
grid between two waves of tiles; its loop also has 10 to 11 fewer integer instructions per K64 than step 5's, by
ptxas's choice: step6.cuh); at M = 16, a kernel of its own (6c: a tile of the 16 rows of the MMA, bands of B over
slices of K, enough blocks resident at once to keep the DRAM busy, fp16 parts summed by the last block of a band); K cut
in two where step 5 takes 128 x 128 and K is a multiple of 128 (6d: 32 blocks of 128 x 256, each over half of K, in
place of step 5's 32 blocks of 128 x 128).
- Judge, the common pass (results/judge/20261009-163456_* and 20261009-163524_*): step 6 1.0750 and 1.0721 in the two
  containers, step 5 1.0041 and 1.0111; 17 and 17 shapes won. Step 6 against step 5, ratio to the reference: at qkv_m128
  and gateup_m128, where 6a alone acts, -0.1 to -0.6 %; at the four M = 16 shapes +7.9 % to +27.2 %, step 6's ratio
  there 1.0043 to 1.0262; at o_m128 and down_m128 +5.2 % to +7.6 %.
- Probe, 5 rounds (results/20261009-1505_step6_probe), each mechanism removed against step 6 (time):
  - without 6a: +1.1 to +2.5 % at the seven shapes with M >= 512 that the probe ran (ladder included), the clock within
    20 MHz of step 6's there; +1.9 and +2.4 % at o_m128 and down_m128; -0.6 % at qkv_m128; no difference at M = 16,
    where 6c keeps its own loads;
  - without 6b: +0.2 to +1.1 % at gateup_m512, qkv_m2048, down_m2048 and ladder, -0.2 to -1.1 % at qkv_m512, o_m512
    and o_m2048;
  - without 6c (step 5's 64-row tiles): +5.3 % at qkv_m16, +18.2 % at o_m16, +3.3 % at gateup_m16, +22.4 % at
    down_m16, those tiles running at a lower clock (qkv_m16: 1009 against 1245 MHz);
  - without 6d (step 5's 128 x 128): +5.7 % at o_m128 and +6.4 % at down_m128;
  - step 5 against step 6: +3.2 to +23.2 % at M = 16, +5.5 and +6.3 % at o_m128 and down_m128, -0.5 % at qkv_m128,
    +1.1 to +3.8 % at M >= 512 (ladder included). The product of the four removals' time ratios, then step 5's, against
    step 6: ladder +2.6 / +2.8 %, qkv_m2048 +3.9 / +3.8 %, o_m512 -0.2 / +1.3 %: at qkv_m2048 the removals cost
    together about what the step gains; at ladder and o_m512, less.
- ncu at ladder (results/20261009-150525_step6; only 6a and 6b act there): step 6 649 272 cycles, tensor active 82.5 %;
  without 6a 669 223 (80.4 %); without 6b, a grid of 160 blocks, 641 724 (84.0 %); step 5 654 572 (82.7 %); cuBLASLt
  616 016 (88.2 %). DRAM read 49.2 MB (49.7 without 6a; cuBLASLt 52.3), L2 sector hits 88.8 %. In isolated calls under
  ncu the waves take 1.2 % more cycles than the grid; in the probe's sustained calls they take 0.2 % less time.
- ncu in the probe (results/20261009-1505_step6_probe): at down_m16, 6c's 80 blocks of 70 registers read 132.6 MB from
  DRAM (B is 117.4 MB) against 143.8 MB without 6c (step 5's tiles) and 134.4 MB for cuBLASLt; at o_m128, 6d writes
  1.8 MB to DRAM against 0.4 MB without it: most of its parts (2 MiB, 2.1 MB) leave the L2.
- 6c's code is kept short and straight: a first, longer form of it (measured before this rung, not published) ran a few
  microseconds longer per call at the four M = 16 shapes, whatever their size, and the shorter code won them back (our
  reading, not measured: B's stream would evict the code from the L2, so each call starts it cold).
- On our earlier T4 kernels (written before this rung, not published):
  - 6a's path had been ahead at every shape tried with M >= 128, and the L2::128B hint needed (without it, the large
    shapes lost);
  - 6b's waves with their barrier had won at the nine shapes with M >= 512 (ladder included), the same persistent
    blocks without the barrier had lost, and at M = 128 the waves were neutral to losing (6b is not used there);
  - for 6c, the rate at which B is read followed the number of resident blocks, in one round, slices of K cost more
    than the bytes of their parts, B evict-first helped when there were parts, and at qkv_m16 nc for B was a little
    faster than the L2-only path;
  - for 6d, with fp32 parts, 32 blocks with K in two had beaten 16 blocks of the same tile at both shapes (the T4's
    energy per SM cycle falls with its clock down to about 1 GHz: more SMs at a lower clock do the same work for less
    energy), and 4 slices (64 blocks, 1.6 waves) were less good.
- Tried on our earlier kernels and left out (not published): the first slice of the next tile loaded before the epilogue
  in 6b's loop (not confirmed at the judge); 6b at M = 128 (neutral to losing); 6d with 4 slices; K split over all 40
  SMs at M = 128 (lost at the four shapes); at M = 16, a continuous flow across bands and a deeper copy (neither
  gained), more slices than the rule's (slower at down_m16 and o_m16), and nc for B at qkv_m16 alone (the rule keeps
  one path); ld.global.nc.L1::no_allocate for 6a (gained nothing).

## Arrival

Step 6 against the reference, the common pass (results/judge/20261009-163456_* and 20261009-163524_*), in the two
containers: 1.0750 and 1.0721, 17 and 17 shapes won. The ratio at each shape, container 1 / container 2:

| shape | M = 16 | M = 128 | M = 512 | M = 2048 |
|---|---|---|---|---|
| qkv (N 6144, K 4096) | 1.0135 / 1.0043 | 1.0959 / 1.0809 | 1.0857 / 1.0900 | 1.0871 / 1.0892 |
| o (4096, 4096) | 1.0262 / 1.0179 | 1.0983 / 1.0901 | 1.0731 / 1.0654 | 1.0647 / 1.0640 |
| gateup (28672, 4096) | 1.0098 / 1.0078 | 1.1196 / 1.1137 | 1.1076 / 1.1063 | 1.1236 / 1.1320 |
| down (4096, 14336) | 1.0074 / 1.0149 | 1.1415 / 1.1388 | 1.0797 / 1.0788 | 1.0830 / 1.0869 |

ladder (2048 x 2560 x 2048): 1.0696 / 1.0574.

## What it does not show

- The T4 only, at its 70 W cap, with cuBLASLt 13.2.1 (`lt130201`) and the driver for CUDA 13.3 (`cuda13030`,
  610.57.04): another driver, card or power limit can move every ratio.
- The 17 shapes only: M a multiple of 16, N of 256, K of 64 (the host checks it).
- cuBLAS only: NVIDIA's other route to these GEMM, the CUTLASS templates, is not scored here.
- Two containers: the absolute score moves from one container to the next on the same code; a step is read against the
  previous one in the same container.
