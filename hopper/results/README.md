# Results of the Hopper rung

Raw outputs of the GPU sessions on Modal's H100, one folder per session, named by its date and time, and of the host
emulator (`emu/`). The tools write their outputs in English. Paths start from this folder; those of the tools
(`judge/judge.cu`, `probe/probe.cu`, `modal/session.py`, `hopper/emu`) from the repository's root.

## The outputs

- `judge/<date>-<time>_<candidate>/`: the judge's output (`judge/judge.cu`), `output.txt`. Its lines 1 and 2 give the
  md5 of each judged source, of the judge, of its launcher, of the shapes file and of the reference file: those of this
  repository's files, which `md5sum` checks. Container 1: `judge/20261030-171549_*`; container 2, the arrival:
  `judge/20261030-171612_*`; 9 candidates each (`ref_simple`, the reference point, and `step0` to `step7`). The arrival
  is the container whose step 7 SCORE is lower: 1.0094 in container 2, against 1.0100 in container 1.
- `<date>-<time>_<steps>/`: a session of `modal/session.py`: the run (`00_run.txt`: the md5 of the launcher and of each
  source, the card line, the cuBLASLt configuration analyzed and where it comes from), the machine (`00_machine.txt`),
  `verif.cu` under compute-sanitizer (`*_01_sanitizer.txt`), ncu at ladder (`*_02_ncu_run.txt`, and its counters in
  `*_02_ncu_raw.csv`). Here, one session per candidate that counts, 9 folders, and the first sessions of steps 1 and 7,
  2 folders (below, "The sessions").
- `<date>-<time>_<step>_probe/`: an interleaved probe (`probe/probe.cu`): `probe.txt` (its lines 1 and 2: the md5 of
  each source, of the probe and of its launcher, the card line). Here, two series of the seven probes, one step at a
  time, the two series of a step at the same time, each probe in a container of its own: series 1,
  `20261030-1823_step1_probe`, `20261030-1834_step2_probe`, `20261030-1845_step3_probe`, `20261030-1852_step4_probe`,
  `20261030-1859_step5_probe`, `20261030-1908_step6_probe` and `20261030-1919_step7_probe`; series 2, the same steps, a
  few seconds later: `20261030-1824_step1_probe`, `20261030-1835_step2_probe`, `20261030-1846_step3_probe`,
  `20261030-1853_step4_probe`, `20261030-1900_step5_probe`, `20261030-1909_step6_probe` and `20261030-1920_step7_probe`.
  Steps 1, 3 and 4 were run again in both series (below, "The probes"): `20261030-1956_step1_probe`,
  `20261030-2007_step3_probe` and `20261030-2014_step4_probe` (series 1), `20261030-1957_step1_probe`,
  `20261030-2008_step3_probe` and `20261030-2015_step4_probe` (series 2).
- `<date>-<time>_<step>_<mechanism>_ncu_probe/`: the probe under ncu only, at a shape where that mechanism acts:
  `probe.txt`, the counters of each arm in `ncu_<arm>_<shape>_raw.csv`, and the version of Nsight Compute that exported
  them (`ncu_export_version.txt`). Here, one per mechanism of step 7, `20261030-1955_step7_7a_ncu_probe` to
  `20261030-1955_step7_7l_ncu_probe` (12 folders, 7e alone included: `20261030-1955_step7_7e_alone_ncu_probe`).
- `two_cards.txt`: for each step, the two probes that decide (one per series), the start and the end of each one's app
  on Modal (in UTC), its ARM lines and rounds, hence a lower bound of its work on the card (each round, a third of a
  block of warm-up, then the timed block: `probe/probe.cu`) and the time it surely ran; then the time the two surely ran
  together. A container keeps its card for its whole life, so two containers running at the same time have two different
  cards: the file shows it at the 7 steps, and says nothing about the hosts.
- `two_cards_first.txt`: the same, with the first runs of steps 1, 3 and 4 (the four other steps as in `two_cards.txt`):
  these runs also ran together, two by two, so on two different cards; they were redone for a gap (below, "The probes"),
  not for the proof.
- `full_pass/20261020-105243_step7/output.txt`: the judge's full pass (without `--reference`) of 20/10/2026 on this card
  line, whose finalists the launcher kept in `judge/reference/h100.txt` (`judge/README.md`). Its candidate is not the
  step 7 published here (the md5 of its line 1), and its SCORE is not this rung's: only its finalists, the reference,
  are used here.
- `emu/`: the host emulator (`hopper/emu`, no GPU) on these sources: `emu_all.txt` (`./emu all`: 7041 runs, 7041 ok, 0
  failed, `ALL OK`) and `planted_bugs.txt` (`python3 hopper/emu/planted_bugs.py -j 2`: 239 of 239 planted bugs reported
  as failures, the unmodified copy passes, `PLANTED-BUG TEST OK`).

The ncu counters are in the `*_raw.csv` files, exported with `ncu --import <report> --csv --page raw`. Paths in the
candidate lines are relative to the repository's root.

## The card, in the judge's outputs

In the judge's 18 outputs, at their 17 shapes, each arm's mean power is 312.1 to 701.5 W and its mean SM clock 1199 to
1980 MHz (columns `w_ref`, `w_cand`, `mhz_ref`, `mhz_cand`); its NVML throttle reasons (`throttle_ref`, `throttle_cand`)
are 0x0 (132 times), 0x4 (479 times) or 0x24 (once). In NVIDIA's `nvml.h`, 0x4 is `nvmlClocksEventReasonSwPowerCap`
("The clocks have been optimized to ensure not to exceed currently set power limits") and 0x20 is
`nvmlClocksEventReasonSwThermalSlowdown`; the judge counts 0x20 as a thermal or hardware throttle
(`judge/judge.cu`:482). The one 0x24 is the candidate's, at down_m128, in container 1's output for step 4
(`judge/20261030-171549_step4/output.txt`): it writes `WARNING: thermal or hardware throttling while timing down_m128`
and, at its end, `CARD THROTTLED (thermal or hardware): this pass does not count as an official pair`. Container 2's
outputs, the arrival, write no such line. The scores of container 1 are published as written.

## Step 7 in the judge, shape by shape

Columns `ratio`, `t_ref_us` and `t_cand_us` of `judge/<container>_step7/output.txt`, times in us per call. The ratio is
the median, over the judge's pairs of timed blocks (reference, candidate, in random order), of the reference's time over
step 7's; each time is the median of its own blocks (`judge/judge.cu`:652-663), so the quotient of the two times can
differ a little from the ratio:

| shape | container 1: ratio | t_ref / t_step7 | container 2 (the arrival): ratio | t_ref / t_step7 |
|---|---:|---:|---:|---:|
| qkv_m16 | 1.0017 | 56.05 / 55.95 | 1.0079 | 55.74 / 55.30 |
| qkv_m128 | 1.0213 | 59.00 / 57.77 | 1.0171 | 58.50 / 57.52 |
| qkv_m512 | 0.9884 | 132.29 / 133.88 | 0.9875 | 131.29 / 132.94 |
| qkv_m2048 | 1.0093 | 515.89 / 511.04 | 1.0091 | 514.25 / 509.57 |
| o_m16 | 1.0361 | 46.29 / 44.67 | 1.0224 | 45.53 / 44.53 |
| o_m128 | 1.0442 | 48.75 / 46.68 | 1.0415 | 47.97 / 46.05 |
| o_m512 | 1.0086 | 106.13 / 105.26 | 1.0095 | 106.26 / 105.21 |
| o_m2048 | 1.0152 | 411.81 / 405.56 | 1.0126 | 412.13 / 407.15 |
| gateup_m16 | 0.9782 | 304.23 / 311.03 | 0.9968 | 299.47 / 300.44 |
| gateup_m128 | 1.0037 | 327.31 / 326.12 | 1.0039 | 317.92 / 316.69 |
| gateup_m512 | 1.0122 | 736.29 / 727.77 | 1.0163 | 742.20 / 729.94 |
| gateup_m2048 | 1.0045 | 2843.36 / 2831.23 | 1.0002 | 2850.36 / 2851.49 |
| down_m16 | 1.0296 | 154.01 / 149.59 | 1.0118 | 150.46 / 148.68 |
| down_m128 | 1.0022 | 160.21 / 159.84 | 1.0058 | 158.02 / 157.11 |
| down_m512 | 0.9957 | 355.99 / 357.53 | 0.9977 | 358.04 / 358.96 |
| down_m2048 | 1.0089 | 1401.86 / 1389.68 | 1.0075 | 1412.27 / 1401.81 |
| ladder | 1.0129 | 618.56 / 611.15 | 1.0139 | 622.59 / 614.37 |

## The sessions

One session per candidate (`modal/session.py`, `--cublaslt` the configuration that the judge's final ranked first at
ladder for that candidate in the arrival): `verif.cu` under compute-sanitizer (`*_01_sanitizer.txt`: plain run,
memcheck, racecheck and synccheck each rc 0 and `OK (tolerance 2e-3)`; memcheck and synccheck `ERROR SUMMARY: 0 errors`,
racecheck `0 hazards displayed`), then ncu at ladder, one call of each kernel (`*_02_ncu_run.txt`: `kernels profiled`;
the same number of rows in `*_02_ncu_raw.csv`): each variant of the step, then cuBLASLt. The nine sessions that count:

| candidate | folder | kernels profiled | cuBLASLt's kernel at ladder, blocks x threads |
|---|---|---:|---|
| reference point | `20261030-172413_ref_simple` | 2 | `nvjet_sm90_hsh_256x128_64x4_1x2_h_bz_coopA_NNT`, 132 x 384 |
| step 0 | `20261030-172408_step0` | 2 | `nvjet_sm90_hsh_256x128_64x4_1x2_h_bz_coopA_NNT`, 132 x 384 |
| step 1 | `20261030-181929_step1` | 5 | `nvjet_sm90_hsh_256x128_64x4_1x2_h_bz_coopA_NNN`, 132 x 384 |
| step 2 | `20261030-172444_step2` | 6 | `nvjet_sm90_hsh_256x128_64x4_1x2_h_bz_coopA_NNN`, 132 x 384 |
| step 3 | `20261030-172624_step3` | 3 | `nvjet_sm90_hsh_256x128_64x4_1x2_h_bz_coopA_NNN`, 132 x 384 |
| step 4 | `20261030-172338_step4` | 3 | `nvjet_sm90_hsh_256x128_64x4_1x2_h_bz_coopA_NNN`, 132 x 384 |
| step 5 | `20261030-172538_step5` | 5 | `nvjet_sm90_hsh_256x128_64x4_1x2_h_bz_coopA_NNN`, 132 x 384 |
| step 6 | `20261030-173039_step6` | 6 | `nvjet_sm90_hsh_256x128_64x4_1x2_h_bz_coopA_NNT`, 132 x 384 |
| step 7 | `20261030-182302_step7` | 16 | `nvjet_sm90_hsh_256x128_64x4_1x2_h_bz_coopA_NNT`, 132 x 384 |

The first sessions of steps 1 and 7, `20261030-173336_step1` and `20261030-173522_step7`, ran with the seven others, at
the same time; they stay here. In each, the plain run, memcheck and synccheck end rc 0 with `OK (tolerance 2e-3)`, and
memcheck and synccheck with `ERROR SUMMARY: 0 errors`; racecheck ends rc 1, `RACECHECK SUMMARY: 0 hazards displayed`,
after `HANG: shape 16 x 65536 x 448, v2: the call has not finished after 2 s (limits: 600 s per call, a budget of 600 s
for the run)` (step 1) and `HANG: shape 4224 x 2304 x 64, v14: the call has not finished after 0 s (limits: 600 s per
call, a budget of 600 s for the run)` (step 7), then `FAILED`: `verif.cu` stops a run that outlasts its budget
(`GEMM_LADDER_BUDGET`, set by `modal/session.py` 60 s below the run's limit), and these calls had 2 s and 0 s of it
left. Step 1's racecheck had made 102 of its 104 calls, step 7's 284 of its 390 (shapes 1 to 18, and 14 of the 15
variants of shape 19, of the 26 shapes of `verif.cu`); in each file, the end of the output, which `modal/session.py`
keeps (the last 45 calls), shows each within the tolerance. Both sessions were run again together, without the seven
others, with the same command: `20261030-181929_step1` and `20261030-182302_step7`, the two in the table above, whose
racecheck ends within its budget, rc 0.

## The probes

A case is t(arm) / t(anchor) - 1, in %, from the two medians printed in the same `probe.txt` at the same shape, rounded
to one decimal (halves away from zero); the anchor is v0, the step, unless the column names another arm. `twin` is the
step timed a second time. A dash: at that shape the arm launches the same kernels as the arm it is compared with, with
the same grids and arguments (each step's variants and their classes are in the header of `step<i>.cu`); the probe times
it all the same, and its median stays in `probe.txt`. The noise of a probe is the largest, in absolute value, of its
`twin` cases and of its cases shown as a dash against the step (at least 0.1 %); its rms noise is the square root of the
mean of the squares of these same gaps, unrounded. When the largest of these gaps is more than twice the second largest
(counted at least 0.1 %), the probe is run again once, the same way, in both series at the same time: the second runs
decide, and the first ones stay here. Each step has two probes that decide, one per series, side by side in its table:
each cell gives series 1 / series 2. The mechanisms' means, from these same medians, are in the rung's README
(`../README.md`).

Step 1 (`20261030-1956_step1_probe` / `20261030-1957_step1_probe`), each cell series 1 / series 2:

| shape | twin | v2, step 0 | v1, a 128 x 128 tile | v3, without 1b |
|---|---:|---:|---:|---:|
| qkv_m16 | +0.5 % / 0.0 % | +68.3 % / +68.0 % | -21.7 % / -21.7 % | +150.6 % / +150.9 % |
| qkv_m128 | 0.0 % / 0.0 % | +67.4 % / +67.4 % | -21.2 % / -21.0 % | +144.2 % / +144.8 % |
| qkv_m512 | 0.0 % / 0.0 % | +140.1 % / +137.8 % | +14.7 % / +14.7 % | +128.7 % / +126.5 % |
| qkv_m2048 | -0.2 % / +0.1 % | +211.3 % / +216.8 % | +47.3 % / +47.9 % | +115.8 % / +118.5 % |
| o_m16 | 0.0 % / 0.0 % | +65.2 % / +66.0 % | -24.1 % / -23.6 % | +143.0 % / +144.0 % |
| o_m128 | +0.3 % / 0.0 % | +64.1 % / +64.5 % | -22.9 % / -22.6 % | +138.5 % / +139.4 % |
| o_m512 | +0.4 % / -0.8 % | +209.3 % / +212.8 % | +46.9 % / +47.8 % | +112.6 % / +114.0 % |
| o_m2048 | +0.1 % / 0.0 % | +210.2 % / +214.4 % | +47.4 % / +47.8 % | +115.7 % / +117.4 % |
| gateup_m16 | +0.1 % / -0.2 % | +191.1 % / +193.2 % | +37.0 % / +38.4 % | +115.3 % / +116.4 % |
| gateup_m128 | 0.0 % / 0.0 % | +185.7 % / +187.6 % | +37.7 % / +38.7 % | +109.4 % / +110.6 % |
| gateup_m512 | -0.3 % / 0.0 % | +171.6 % / +174.2 % | +33.8 % / +33.4 % | +99.9 % / +101.5 % |
| gateup_m2048 | -0.2 % / -0.1 % | +164.0 % / +166.6 % | +31.3 % / +31.0 % | +97.1 % / +98.7 % |
| down_m16 | 0.0 % / 0.0 % | +66.8 % / +67.5 % | -23.8 % / -23.4 % | +147.3 % / +147.8 % |
| down_m128 | 0.0 % / 0.0 % | +71.9 % / +72.0 % | -23.0 % / -22.7 % | +141.1 % / +141.8 % |
| down_m512 | -0.2 % / +4.5 % | +211.7 % / +225.8 % | +43.6 % / +51.9 % | +103.0 % / +114.3 % |
| down_m2048 | 0.0 % / +0.3 % | +221.3 % / +222.5 % | +49.0 % / +50.6 % | +118.6 % / +121.1 % |
| ladder | 0.0 % / 0.0 % | +214.2 % / +215.3 % | +46.0 % / +47.6 % | +111.6 % / +113.1 % |

The noise: 0.5 % (qkv_m16 by twin) / 4.5 % (down_m512 by twin); the rms noise: 0.201 % / 1.110 %. Run again in both
series: the first runs, `20261030-1823_step1_probe` (series 1: the twin's gap 1.2 % at down_m512, more than twice the
second largest, 0.5 %) and `20261030-1824_step1_probe` (series 2: the twin's gap 3.8 % at down_m512, more than twice the
second largest, 0.2 %), stay here; the table above takes the second runs only, which ran together (`two_cards.txt`; the
first runs too: `two_cards_first.txt`).

Step 2 (`20261030-1834_step2_probe` / `20261030-1835_step2_probe`), each cell series 1 / series 2:

| shape | twin | v3, without 2d | v4 (2 stages, without 2d) against v2 (step 1) | v4 against v3 (without 2d) |
|---|---:|---:|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | -0.1 % / +1.2 % | -44.1 % / -44.1 % | +21.9 % / +20.8 % |
| qkv_m128 | 0.0 % / 0.0 % | +2.0 % / +2.4 % | -44.6 % / -44.8 % | +20.1 % / +20.3 % |
| qkv_m512 | +0.2 % / 0.0 % | +2.7 % / +2.7 % | -43.1 % / -43.5 % | +5.7 % / +6.0 % |
| qkv_m2048 | -0.2 % / -0.2 % | +4.5 % / +4.4 % | -36.0 % / -36.2 % | +2.5 % / +3.3 % |
| o_m16 | -0.2 % / 0.0 % | +5.6 % / +6.8 % | -47.1 % / -46.9 % | +16.5 % / +14.6 % |
| o_m128 | 0.0 % / 0.0 % | +5.3 % / +5.5 % | -47.6 % / -47.4 % | +15.5 % / +14.8 % |
| o_m512 | -0.2 % / -0.1 % | +3.2 % / +3.6 % | -36.5 % / -37.8 % | +3.8 % / +3.9 % |
| o_m2048 | 0.0 % / -0.1 % | +4.9 % / +5.0 % | -37.0 % / -37.4 % | +0.6 % / +1.0 % |
| gateup_m16 | 0.0 % / 0.0 % | -3.0 % / -1.7 % | -19.2 % / -18.7 % | +5.7 % / +4.4 % |
| gateup_m128 | 0.0 % / 0.0 % | -0.1 % / -0.1 % | -26.1 % / -25.9 % | +3.3 % / +3.1 % |
| gateup_m512 | 0.0 % / 0.0 % | +1.5 % / +1.5 % | -22.4 % / -22.1 % | 0.0 % / -0.1 % |
| gateup_m2048 | 0.0 % / 0.0 % | -1.5 % / -1.4 % | -21.1 % / -20.7 % | +3.0 % / +2.7 % |
| down_m16 | 0.0 % / 0.0 % | +4.1 % / +4.6 % | -47.4 % / -47.4 % | +19.1 % / +17.8 % |
| down_m128 | 0.0 % / 0.0 % | +4.2 % / +4.3 % | -48.3 % / -48.2 % | +18.4 % / +17.4 % |
| down_m512 | -0.2 % / -0.2 % | +6.1 % / +6.4 % | -40.2 % / -40.2 % | +0.7 % / +1.2 % |
| down_m2048 | +0.3 % / 0.0 % | +15.7 % / +14.1 % | -37.2 % / -37.4 % | -7.9 % / -6.2 % |
| ladder | -0.1 % / -0.2 % | +6.4 % / +6.7 % | -39.2 % / -39.1 % | -2.2 % / -1.5 % |

The noise: 0.3 % (down_m2048 by twin) / 0.2 % (qkv_m2048 by twin, down_m512 by twin, ladder by twin); the rms noise:
0.138 % / 0.080 %.

Step 3 (`20261030-2007_step3_probe` / `20261030-2008_step3_probe`), each cell series 1 / series 2:

| shape | twin | v1, step 2 |
|---|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | +13.7 % / +12.9 % |
| qkv_m128 | 0.0 % / +0.1 % | +9.7 % / +9.9 % |
| qkv_m512 | 0.0 % / 0.0 % | +5.2 % / +5.0 % |
| qkv_m2048 | -0.1 % / +0.2 % | -0.7 % / +1.0 % |
| o_m16 | 0.0 % / 0.0 % | +9.2 % / +8.9 % |
| o_m128 | +0.4 % / 0.0 % | +8.7 % / +9.5 % |
| o_m512 | +0.1 % / -0.1 % | +5.6 % / +4.9 % |
| o_m2048 | +0.1 % / +0.2 % | +4.4 % / +3.6 % |
| gateup_m16 | 0.0 % / 0.0 % | +4.1 % / +2.8 % |
| gateup_m128 | 0.0 % / 0.0 % | -0.9 % / -2.8 % |
| gateup_m512 | 0.0 % / 0.0 % | -1.6 % / -2.0 % |
| gateup_m2048 | 0.0 % / 0.0 % | +0.4 % / -0.1 % |
| down_m16 | 0.0 % / 0.0 % | +9.4 % / +9.2 % |
| down_m128 | -0.6 % / 0.0 % | +5.6 % / +9.2 % |
| down_m512 | +0.1 % / +0.2 % | +5.8 % / +5.3 % |
| down_m2048 | -0.2 % / -1.0 % | -2.9 % / -9.9 % |
| ladder | 0.0 % / +0.5 % | +2.2 % / +1.8 % |

The noise: 0.6 % (down_m128 by twin) / 1.0 % (down_m2048 by twin); the rms noise: 0.184 % / 0.290 %. Run again in both
series: the first runs, `20261030-1845_step3_probe` (series 1: the twin's gap 3.4 % at down_m2048, more than twice the
second largest, 0.9 %) and `20261030-1846_step3_probe` (series 2: the twin's gap 3.6 % at down_m2048, more than twice
the second largest, 0.7 %), stay here; the table above takes the second runs only, which ran together (`two_cards.txt`;
the first runs too: `two_cards_first.txt`).

Step 4 (`20261030-2014_step4_probe` / `20261030-2015_step4_probe`), each cell series 1 / series 2:

| shape | twin | v1, step 3 |
|---|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | -0.3 % / -0.8 % |
| qkv_m128 | -0.8 % / -0.1 % | -6.9 % / -0.4 % |
| qkv_m512 | -0.3 % / 0.0 % | +4.2 % / +4.1 % |
| qkv_m2048 | 0.0 % / -0.1 % | +8.5 % / +7.8 % |
| o_m16 | 0.0 % / 0.0 % | -0.3 % / -0.3 % |
| o_m128 | +1.3 % / 0.0 % | -6.9 % / -0.6 % |
| o_m512 | 0.0 % / 0.0 % | +3.3 % / +2.7 % |
| o_m2048 | -0.1 % / -0.1 % | +3.6 % / +3.8 % |
| gateup_m16 | 0.0 % / +0.1 % | -1.1 % / -0.3 % |
| gateup_m128 | 0.0 % / +0.1 % | 0.0 % / +0.3 % |
| gateup_m512 | -0.1 % / +0.1 % | +53.7 % / +62.4 % |
| gateup_m2048 | 0.0 % / -0.1 % | +54.8 % / +62.3 % |
| down_m16 | 0.0 % / 0.0 % | -0.1 % / -0.2 % |
| down_m128 | +0.4 % / 0.0 % | -7.4 % / -0.2 % |
| down_m512 | +0.2 % / 0.0 % | +3.7 % / +3.3 % |
| down_m2048 | +0.3 % / -0.2 % | +16.8 % / +19.5 % |
| ladder | 0.0 % / -0.2 % | +5.4 % / +5.7 % |

The noise: 1.3 % (o_m128 by twin) / 0.2 % (down_m2048 by twin, ladder by twin); the rms noise: 0.412 % / 0.085 %. Run
again in both series: the first runs, `20261030-1852_step4_probe` (series 1: the twin's gap 0.4 % at gateup_m512, more
than twice the second largest, 0.1 %) and `20261030-1853_step4_probe` (series 2: largest gap 0.2 %, second 0.2 %, no gap
of its own), stay here; the table above takes the second runs only, which ran together (`two_cards.txt`; the first runs
too: `two_cards_first.txt`).

Step 5 (`20261030-1859_step5_probe` / `20261030-1900_step5_probe`), each cell series 1 / series 2:

| shape | twin | v1, without 5a | v2, without 5b |
|---|---:|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | - | -0.1 % / -0.1 % |
| qkv_m128 | +0.6 % / +0.1 % | - | +1.1 % / 0.0 % |
| qkv_m512 | -0.2 % / 0.0 % | +0.7 % / +0.9 % | +0.8 % / +1.1 % |
| qkv_m2048 | 0.0 % / +0.1 % | +1.0 % / +0.8 % | +4.9 % / +5.1 % |
| o_m16 | 0.0 % / +0.5 % | - | -0.1 % / +0.4 % |
| o_m128 | +0.7 % / 0.0 % | - | -0.9 % / -0.1 % |
| o_m512 | +0.1 % / +0.1 % | - | +0.2 % / -0.1 % |
| o_m2048 | 0.0 % / 0.0 % | +0.8 % / +0.5 % | +3.3 % / +3.4 % |
| gateup_m16 | -0.1 % / 0.0 % | +2.5 % / +1.0 % | -0.1 % / -0.1 % |
| gateup_m128 | +0.4 % / 0.0 % | +1.9 % / -0.9 % | +0.4 % / 0.0 % |
| gateup_m512 | +0.2 % / 0.0 % | +0.7 % / +0.7 % | +5.2 % / +5.5 % |
| gateup_m2048 | +0.2 % / +0.2 % | -0.6 % / -0.4 % | +8.2 % / +8.7 % |
| down_m16 | +0.3 % / 0.0 % | - | +0.2 % / 0.0 % |
| down_m128 | +0.3 % / 0.0 % | - | +2.0 % / 0.0 % |
| down_m512 | 0.0 % / +0.2 % | - | +0.1 % / 0.0 % |
| down_m2048 | 0.0 % / +0.1 % | 0.0 % / +0.1 % | +3.9 % / +4.1 % |
| ladder | -0.1 % / -0.2 % | +1.2 % / +0.8 % | +0.1 % / 0.0 % |

The noise: 1.1 % (o_m128 by v1) / 0.5 % (o_m16 by twin, o_m16 by v1); the rms noise: 0.318 % / 0.171 %. Where an arm
differs from the step (a dash at its other shapes): v1 at qkv_m512, qkv_m2048, o_m2048, gateup_m16, gateup_m128,
gateup_m512, gateup_m2048, down_m2048, ladder.

Step 6 (`20261030-1908_step6_probe` / `20261030-1909_step6_probe`), each cell series 1 / series 2:

| shape | twin | v1, without 6a | v2, without 6b | v3, without 6c |
|---|---:|---:|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | +27.9 % / +27.6 % | - | - |
| qkv_m128 | 0.0 % / -0.1 % | - | +28.4 % / +27.7 % | - |
| qkv_m512 | -0.1 % / 0.0 % | - | - | +3.2 % / +3.1 % |
| qkv_m2048 | -0.1 % / -0.1 % | - | - | +1.2 % / +1.2 % |
| o_m16 | 0.0 % / 0.0 % | +65.2 % / +65.0 % | - | - |
| o_m128 | -0.1 % / -0.1 % | - | +49.0 % / +48.9 % | - |
| o_m512 | -0.2 % / +0.1 % | - | - | +1.2 % / +1.3 % |
| o_m2048 | -0.1 % / -0.3 % | - | - | +0.9 % / +0.8 % |
| gateup_m16 | 0.0 % / 0.0 % | +4.5 % / +4.3 % | - | - |
| gateup_m128 | 0.0 % / 0.0 % | - | - | - |
| gateup_m512 | -0.2 % / -0.2 % | - | - | +2.2 % / +2.4 % |
| gateup_m2048 | -0.1 % / -0.1 % | - | - | +3.2 % / +2.5 % |
| down_m16 | 0.0 % / 0.0 % | +63.2 % / +63.1 % | - | - |
| down_m128 | -0.2 % / -0.2 % | - | +47.0 % / +44.8 % | - |
| down_m512 | +0.1 % / -0.2 % | - | - | +1.5 % / +1.2 % |
| down_m2048 | +0.1 % / -0.1 % | - | - | +0.9 % / +1.0 % |
| ladder | -0.1 % / -0.2 % | - | - | +1.0 % / +1.0 % |

The noise: 0.3 % (o_m128 by v1, ladder by v2) / 0.3 % (o_m2048 by twin, o_m2048 by v2, ladder by v1, ladder by v2); the
rms noise: 0.116 % / 0.126 %. Where an arm differs from the step (a dash at its other shapes): v1 at qkv_m16, o_m16,
gateup_m16, down_m16; v2 at qkv_m128, o_m128, down_m128; v3 at qkv_m512, qkv_m2048, o_m512, o_m2048, gateup_m512,
gateup_m2048, down_m512, down_m2048, ladder.

Step 7 (`20261030-1919_step7_probe` / `20261030-1920_step7_probe`), each cell series 1 / series 2:

| shape | twin | v1, without 7a | v2, without 7b | v3, without 7d | v4, without 7e | v5, 7e in pairs | v7, without 7c | v9 (7f alone) against v8 | v8, without 7f and the five | v10, without 7g | v12, without 7i | v13, without 7j | v14, without 7l | v12 against v11 (7h alone) | v11, without 7h and 7i |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv_m16 | -0.2 % / -0.2 % | - | +3.4 % / +3.4 % | - | - | - | - | +3.8 % / +3.6 % | +0.1 % / +0.4 % | +0.1 % / +0.1 % | +0.9 % / +0.9 % | - | +0.3 % / +0.4 % | -2.3 % / -2.4 % | +3.3 % / +3.3 % |
| qkv_m128 | -1.3 % / -1.1 % | - | - | +2.4 % / +2.5 % | - | - | - | -0.6 % / -0.5 % | +2.8 % / +2.6 % | -0.7 % / -0.5 % | - | - | +1.9 % / +1.9 % | - | - |
| qkv_m512 | 0.0 % / -0.2 % | - | - | - | +34.3 % / +34.6 % | +2.2 % / +2.0 % | - | -0.3 % / -0.6 % | +1.3 % / +1.4 % | +1.1 % / +0.9 % | - | 0.0 % / 0.0 % | - | - | - |
| qkv_m2048 | +0.1 % / 0.0 % | - | - | - | - | - | +0.6 % / +0.4 % | +1.9 % / +1.4 % | +0.1 % / +0.1 % | +1.6 % / +1.1 % | - | +0.6 % / +0.8 % | - | - | - |
| o_m16 | -0.1 % / -0.2 % | - | +0.6 % / +0.5 % | - | - | - | - | -0.6 % / -0.6 % | +1.4 % / +1.3 % | 0.0 % / -0.1 % | +1.1 % / +1.0 % | - | +0.4 % / +0.3 % | +0.8 % / +0.8 % | +0.2 % / +0.2 % |
| o_m128 | -0.7 % / -0.8 % | +3.6 % / +3.6 % | - | - | - | - | - | -1.3 % / -1.4 % | +4.4 % / +4.4 % | - | +0.9 % / +1.0 % | - | +2.3 % / +2.4 % | +1.1 % / +1.3 % | -0.1 % / -0.4 % |
| o_m512 | 0.0 % / -0.1 % | - | - | - | - | - | +0.7 % / +0.5 % | +1.5 % / +0.9 % | +0.4 % / +0.3 % | +1.5 % / +1.2 % | - | 0.0 % / 0.0 % | - | - | - |
| o_m2048 | 0.0 % / -0.2 % | - | - | - | - | - | +0.5 % / +0.3 % | +2.0 % / +1.5 % | +0.1 % / +0.1 % | +1.3 % / +0.9 % | - | +0.5 % / +0.7 % | - | - | - |
| gateup_m16 | 0.0 % / 0.0 % | - | - | - | - | - | - | -0.3 % / -0.3 % | +0.1 % / +0.1 % | 0.0 % / 0.0 % | - | - | -0.2 % / -0.2 % | - | - |
| gateup_m128 | 0.0 % / 0.0 % | - | - | - | - | - | - | +0.1 % / +0.1 % | +0.5 % / +0.5 % | 0.0 % / 0.0 % | +0.1 % / +0.1 % | +0.1 % / +0.1 % | - | -0.5 % / -0.5 % | +0.6 % / +0.6 % |
| gateup_m512 | -0.1 % / -0.2 % | - | - | - | - | - | -0.1 % / 0.0 % | +2.0 % / +1.9 % | +0.4 % / +0.5 % | +1.7 % / +1.6 % | - | +1.2 % / +1.2 % | - | - | - |
| gateup_m2048 | +0.2 % / -0.2 % | - | - | - | -0.3 % / -0.3 % | - | - | +1.3 % / +0.8 % | +0.6 % / +0.3 % | +0.3 % / +0.1 % | +0.2 % / -0.1 % | +0.4 % / +0.2 % | - | -0.6 % / -0.4 % | +0.8 % / +0.2 % |
| down_m16 | 0.0 % / 0.0 % | - | +2.0 % / +2.0 % | - | - | - | - | -1.0 % / -0.9 % | +2.1 % / +2.0 % | 0.0 % / 0.0 % | +0.3 % / +0.3 % | - | +0.2 % / +0.3 % | -0.5 % / -0.5 % | +0.8 % / +0.8 % |
| down_m128 | 0.0 % / 0.0 % | +0.7 % / +1.0 % | - | - | - | - | - | -1.5 % / -1.4 % | +2.2 % / +2.3 % | - | +0.2 % / +0.2 % | - | +1.0 % / +1.1 % | +0.6 % / +0.4 % | -0.4 % / -0.2 % |
| down_m512 | 0.0 % / -0.1 % | - | - | - | - | - | +1.0 % / +0.3 % | +1.6 % / +1.3 % | +0.3 % / +0.1 % | +1.6 % / +1.3 % | - | 0.0 % / -0.2 % | - | - | - |
| down_m2048 | -0.2 % / -0.1 % | - | - | - | - | - | +0.6 % / +0.3 % | +2.1 % / +1.2 % | +0.2 % / +0.2 % | +1.5 % / +1.0 % | - | +0.7 % / +0.5 % | - | - | - |
| ladder | -0.1 % / 0.0 % | - | - | - | - | - | +1.5 % / +1.2 % | +2.3 % / +1.9 % | +0.6 % / +0.5 % | +2.1 % / +1.5 % | - | +1.5 % / +1.2 % | - | - | - |

The noise: 1.4 % (qkv_m128 by v7) / 1.1 % (qkv_m128 by twin, qkv_m128 by v7, qkv_m128 by v11); the rms noise: 0.292 % /
0.254 %. Where an arm differs from the step (a dash at its other shapes): v1 at o_m128, down_m128; v2 at qkv_m16, o_m16,
down_m16; v3 at qkv_m128; v4 at qkv_m512, gateup_m2048; v5 at qkv_m512; v7 at qkv_m2048, o_m512, o_m2048, gateup_m512,
down_m512, down_m2048, ladder; v10 at qkv_m16, qkv_m128, qkv_m512, qkv_m2048, o_m16, o_m512, o_m2048, gateup_m16,
gateup_m128, gateup_m512, gateup_m2048, down_m16, down_m512, down_m2048, ladder; v12 at qkv_m16, o_m16, o_m128,
gateup_m128, gateup_m2048, down_m16, down_m128; v13 at qkv_m512, qkv_m2048, o_m512, o_m2048, gateup_m128, gateup_m512,
gateup_m2048, down_m512, down_m2048, ladder; v14 at qkv_m16, qkv_m128, o_m16, o_m128, gateup_m16, down_m16, down_m128;
v12 against v11 at qkv_m16, o_m16, o_m128, gateup_m128, gateup_m2048, down_m16, down_m128; v11 at qkv_m16, o_m16,
o_m128, gateup_m128, gateup_m2048, down_m16, down_m128.

Step 7's probes also time two reading arms at each shape, which judge nothing: v6, step 6 itself, and ref0, the cuBLASLt
configuration of rank 0 for that shape and this card line in `judge/reference/h100.txt`. Their ratio to the step, ref0 /
v0 and ref0 / v6, each cell series 1 / series 2; over the 17 shapes (geometric mean), step 7 1.0054 / 1.0046 and step 6
0.9623 / 0.9657. A reading of the probe, not the judge's score:

| shape | ref0 / v0 | ref0 / v6 |
|---|---:|---:|
| qkv_m16 | 0.9923 / 0.9918 | 0.9353 / 0.9367 |
| qkv_m128 | 1.0002 / 0.9959 | 0.9446 / 0.9594 |
| qkv_m512 | 0.9875 / 0.9861 | 0.7321 / 0.7284 |
| qkv_m2048 | 1.0112 / 1.0114 | 1.0048 / 1.0058 |
| o_m16 | 1.0317 / 1.0305 | 1.0202 / 1.0196 |
| o_m128 | 1.0234 / 1.0215 | 0.9226 / 0.9373 |
| o_m512 | 1.0103 / 1.0059 | 0.9985 / 0.9965 |
| o_m2048 | 1.0145 / 1.0107 | 1.0070 / 1.0053 |
| gateup_m16 | 0.9812 / 0.9792 | 0.9805 / 0.9785 |
| gateup_m128 | 0.9839 / 0.9814 | 0.9793 / 0.9766 |
| gateup_m512 | 1.0077 / 1.0124 | 1.0036 / 1.0074 |
| gateup_m2048 | 1.0079 / 1.0000 | 0.9947 / 0.9954 |
| down_m16 | 1.0282 / 1.0265 | 1.0016 / 1.0001 |
| down_m128 | 0.9972 / 1.0054 | 0.9035 / 0.9220 |
| down_m512 | 0.9990 / 0.9965 | 0.9883 / 0.9876 |
| down_m2048 | 1.0061 / 1.0096 | 0.9953 / 1.0042 |
| ladder | 1.0116 / 1.0151 | 0.9915 / 0.9998 |

## ncu in the probes of step 7

At one shape for each mechanism of step 7, where it acts (`20261030-1955_step7_<mechanism>_ncu_probe`), each arm in its
own process, called twice under ncu: the counters are those of its second call (ID 1 of its CSV). `x:` is the
configuration that the judge's final ranked first at that shape for step 7 in the arrival
(`judge/20261030-171612_step7/output.txt`, its REFERENCE line of rank 0). These counters are a snapshot: under ncu, each
call is replayed in many passes, the clocks not locked, so they show what an arm runs (its kernel, its grid, its
traffic), not its time, and judge nothing; the timed probes, in two series, judge each mechanism (the tables above).

| mechanism | shape, arm | kernel, blocks | cycles | tensor active | DRAM read | DRAM write | time under ncu |
|---|---|---|---:|---:|---:|---:|---:|
| 7a | o_m128, step 7 | `bands<64, 2, 8, 2, 1, 1, 1, 1>`, 128 | 85 741 | 42.2 % | 136.3 MB | 2.37 MB | 49.92 us |
| 7a | o_m128, v1 (without 7a) | `gemm<64, 128, 8, 1, 0, 1, 1, 1, 1>`, 128 | 86 731 | 41.7 % | 136.3 MB | 3.32 MB | 49.73 us |
| 7a | o_m128, `x:` | `nvjet_sm90_hsh_128x64_64x8_1x2_h_bz_NNN`, 128 | 87 253 | 41.0 % | 136.3 MB | 3.49 MB | 49.70 us |
| 7b | qkv_m16, step 7 | `bands<16, 4, 6, 1, 1, 1, 1, 1>`, 40 | 115 686 | 16.8 % | 168.1 MB | 2.51 MB | 59.33 us |
| 7b | qkv_m16, v2 (without 7b) | `bands<16, 2, 8, 0, 1, 0, 0, 1>`, 80 | 110 124 | 8.5 % | 168.1 MB | 2.86 MB | 61.25 us |
| 7b | qkv_m16, `x:` | `nvjet_sm90_hsh_256x16_64x6_4x1_v_bz_NNT`, 40 | 103 095 | 16.7 % | 168.1 MB | 3.23 MB | 58.02 us |
| 7c | ladder, step 7 | `gemm_rows<128, 256, 4, 232, 1, 1>`, 132 | 827 395 | 97.9 % | 427.8 MB | 47.00 MB | 518.40 us |
| 7c | ladder, v7 (without 7c) | `gemm<128, 256, 4, 0, 232, 1, 0, 0, 1>`, 132 | 830 509 | 97.8 % | 511.8 MB | 46.77 MB | 525.15 us |
| 7c | ladder, `x:` | `nvjet_sm90_hsh_256x128_64x4_1x2_h_bz_coopA_NNT`, 132 | 828 696 | 97.7 % | 427.9 MB | 46.73 MB | 521.12 us |
| 7d | qkv_m128, step 7 | `bands<128, 2, 6, 0, 1, 0, 0, 1>`, 80 | 107 114 | 65.6 % | 169.9 MB | 2.60 MB | 61.38 us |
| 7d | qkv_m128, v3 (without 7d) | `gemm<128, 128, 6, 0, 0, 1, 0, 0, 1>`, 80 | 109 799 | 66.5 % | 169.9 MB | 2.85 MB | 63.46 us |
| 7d | qkv_m128, `x:` | `nvjet_sm90_hsh_112x128_64x6_4x1_v_bz_NNN`, 92 | 105 600 | 59.0 % | 169.9 MB | 3.35 MB | 61.09 us |
| 7e | qkv_m512, step 7 | `gemm<128, 320, 3, 0, 232, 1, 0, 0, 1>`, 128 | 186 734 | 93.8 % | 176.2 MB | 2.96 MB | 114.98 us |
| 7e | qkv_m512, v4 (without 7e) | `gemm_rows<128, 256, 4, 232, 1, 1>`, 132 | 294 945 | 95.4 % | 184.6 MB | 10.48 MB | 169.63 us |
| 7e | qkv_m512, `x:` | `nvjet_sm90_hsh_320x128_64x3_1x2_v_bz_coopA_NNN`, 128 | 182 725 | 95.2 % | 176.2 MB | 3.58 MB | 113.82 us |
| 7e alone | qkv_m512, step 7 | `gemm<128, 320, 3, 0, 232, 1, 0, 0, 1>`, 128 | 183 057 | 93.8 % | 176.2 MB | 3.00 MB | 118.53 us |
| 7e alone | qkv_m512, v5 (7e in pairs) | `gemm<128, 320, 3, 1, 232, 1, 1, 1, 1>`, 128 | 184 986 | 93.3 % | 176.2 MB | 3.45 MB | 120.96 us |
| 7e alone | qkv_m512, `x:` | `nvjet_sm90_hsh_320x128_64x3_1x2_v_bz_coopA_NNN`, 128 | 180 764 | 95.4 % | 176.2 MB | 2.82 MB | 117.66 us |
| 7f | gateup_m16, step 7 | `bands<16, 7, 3, 0, 1, 0, 0, 1>`, 128 | 560 037 | 5.6 % | 939.8 MB | 3.08 MB | 311.23 us |
| 7f | gateup_m16, v8 (without 7f and the five) | `bands_noroles<16, 7, 3, 0>`, 128 | 560 302 | 5.6 % | 940.0 MB | 3.53 MB | 312.22 us |
| 7f | gateup_m16, `x:` | `nvjet_sm90_hsh_448x64_64x2_2x1_v_bz_coopB_NNN`, 128 | 549 243 | 21.5 % | 939.8 MB | 3.85 MB | 306.30 us |
| 7g | gateup_m16, step 7 | `bands<16, 7, 3, 0, 1, 0, 0, 1>`, 128 | 557 902 | 5.6 % | 939.8 MB | 3.18 MB | 311.52 us |
| 7g | gateup_m16, v10 (without 7g) | `bands<16, 7, 3, 0, 1, 0, 0, 0>`, 128 | 556 944 | 5.6 % | 939.8 MB | 3.20 MB | 311.01 us |
| 7g | gateup_m16, `x:` | `nvjet_sm90_hsh_448x64_64x2_2x1_v_bz_coopB_NNN`, 128 | 549 440 | 21.5 % | 939.8 MB | 3.19 MB | 307.26 us |
| 7h | gateup_m128, step 7 | `gemm5<1, 1, 1, 1>`, 132 | 563 738 | 47.5 % | 946.5 MB | 12.99 MB | 323.07 us |
| 7h | gateup_m128, v11 (without 7h and 7i) | `gemm5<1, 0, 0, 1>`, 132 | 579 750 | 47.2 % | 946.7 MB | 13.18 MB | 326.21 us |
| 7h | gateup_m128, `x:` | `nvjet_sm90_hsh_256x168_64x3_4x1_v_bz_coopA_NNT`, 120 | 567 311 | 64.2 % | 943.8 MB | 10.70 MB | 315.84 us |
| 7i | o_m128, step 7 | `bands<64, 2, 8, 2, 1, 1, 1, 1>`, 128 | 86 179 | 41.7 % | 136.3 MB | 2.72 MB | 48.22 us |
| 7i | o_m128, v12 (without 7i) | `bands<64, 2, 8, 2, 1, 1, 0, 1>`, 128 | 88 052 | 41.4 % | 136.3 MB | 2.48 MB | 50.27 us |
| 7i | o_m128, `x:` | `nvjet_sm90_hsh_128x64_64x8_1x2_h_bz_NNN`, 128 | 87 284 | 41.5 % | 136.3 MB | 3.43 MB | 50.46 us |
| 7j | gateup_m512, step 7 | `gemm_rows<128, 256, 4, 232, 1, 1>`, 132 | 964 574 | 97.8 % | 998.3 MB | 55.07 MB | 618.85 us |
| 7j | gateup_m512, v13 (without 7j) | `gemm_rows<128, 256, 4, 232, 0, 1>`, 132 | 966 157 | 97.5 % | 998.3 MB | 54.78 MB | 616.70 us |
| 7j | gateup_m512, `x:` | `nvjet_sm90_hsh_128x256_64x4_1x2_h_bz_coopA_NNT`, 132 | 960 730 | 97.7 % | 998.3 MB | 54.75 MB | 619.23 us |
| 7l | gateup_m16, step 7 | `bands<16, 7, 3, 0, 1, 0, 0, 1>`, 128 | 561 838 | 5.5 % | 939.8 MB | 3.08 MB | 310.94 us |
| 7l | gateup_m16, v14 (without 7l) | `bands<16, 7, 3, 0, 0, 0, 0, 1>`, 128 | 566 176 | 5.5 % | 940.0 MB | 3.66 MB | 313.06 us |
| 7l | gateup_m16, `x:` | `nvjet_sm90_hsh_448x64_64x2_2x1_v_bz_coopB_NNN`, 128 | 558 523 | 21.3 % | 939.8 MB | 3.35 MB | 309.76 us |

## The counters

The names, in the CSV files, of the counters cited here and in the rung's README: cycles `sm__cycles_elapsed.max`;
tensor active `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`; DRAM read and write
`dram__bytes_read.sum`, `dram__bytes_write.sum`; blocks and threads `launch__grid_size`, `launch__block_size`; time
under ncu `gpu__time_duration.sum`. Each value is that of the kernel's row (its Kernel Name and ID), in % and MB to one
decimal (DRAM write to two).
