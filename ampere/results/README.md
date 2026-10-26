# Results of the Ampere rung

Raw outputs of the GPU sessions on Modal's A100, one folder per session, named by its date and time, and of the host
emulator (`emu/`). The tools write their outputs in English. Paths start from this folder; those of the tools
(`judge/judge.cu`, `probe/probe.cu`, `modal/session.py`, `ampere/emu`) from the repository's root.

## The outputs

- `judge/<date>-<time>_<candidate>/`: the judge's output (`judge/judge.cu`), `output.txt`. Its lines 1 and 2 give the
  md5 of each judged source, of the judge, of its launcher, of the shapes file and of the reference file: those of this
  repository's files, which `md5sum` checks. Container 1: `judge/20261026-163413_*`; container 2, the arrival:
  `judge/20261026-163446_*`; 9 candidates each (`ref_simple`, the reference point, and `step0` to `step7`). The two
  SCOREs of step 7 are equal to four decimals (1.0224); the arrival is then the container whose step 7 score,
  recomputed by the judge's formula (the geometric mean over the 17 shapes) from the ratios written in its output, is
  lower: 1.02237 in container 2, against 1.02243 in container 1.
- `<date>-<time>_<steps>/`: a session of `modal/session.py`: the run (`00_run.txt`: the md5 of the launcher and of each
  source, the card line, the cuBLASLt configuration analyzed and where it comes from), the machine (`00_machine.txt`),
  `verif.cu` under compute-sanitizer (`*_01_sanitizer.txt`), ncu at ladder (`*_02_ncu_run.txt`, and its counters in
  `*_02_ncu_raw.csv`). Here, one session per candidate, 9 folders (below, "The sessions").
- `<date>-<time>_<step>_probe/`: an interleaved probe (`probe/probe.cu`): `probe.txt` (its lines 1 and 2: the md5 of
  each source, of the probe and of its launcher, the card line). Here, two series of the seven probes, each probe in a
  container of its own: series 1, `20261026-1646_step2_probe` to `20261026-1646_step7_probe` (step 1's first run in
  series 1 left no folder: below); series 2, `20261026-1647_step1_probe` to `20261026-1647_step7_probe`. Steps 1, 2, 3,
  4 and 6 were run again in both series (below): `20261026-1733_step1_probe`, `20261026-1747_step2_probe`,
  `20261026-1756_step3_probe`, `20261026-1718_step4_probe` and `20261026-1805_step6_probe` (series 1),
  `20261026-1734_step1_probe`, `20261026-1748_step2_probe`, `20261026-1757_step3_probe`, `20261026-1719_step4_probe`
  and `20261026-1806_step6_probe` (series 2).
- `<date>-<time>_<step>_<mechanism>_ncu_probe/`: the probe under ncu only, at a shape where that mechanism acts:
  `probe.txt`, the counters of each arm in `ncu_<arm>_<shape>_raw.csv`, and the version of Nsight Compute that
  exported them (`ncu_export_version.txt`). Here, one per mechanism of step 7, `20261026-1714_step7_7a_ncu_probe` to
  `20261026-1714_step7_7h_ncu_probe`.
- `two_cards.txt`: for each step, the two probes that decide (one per series), the start and the end of each one's app
  on Modal (in UTC), its ARM lines and rounds, hence a lower bound of its work on the card (each round, a third of a
  block of warm-up, then the timed block: `probe/probe.cu`) and the time it surely ran; then the time the two surely
  ran together. A container keeps its card for its whole life, so two containers running at the same time have two
  different cards: the file shows it at the 7 steps, and says nothing about the hosts.
- `two_cards_first.txt`: the same, with the first runs of steps 1, 2, 3 and 6 (the three other steps as in
  `two_cards.txt`): no `probe.txt` for step 1's run of series 1; no sure moment for the runs of series 1 of steps 2 and
  3 and for the run of series 2 of step 6 (their apps on Modal lasted more than twice their work); hence no proof for
  steps 1, 2, 3 and 6.
- `full_pass/20261023-152424_step7/output.txt`: the judge's full pass (without `--reference`) of 23/10/2026 on this
  card line, whose finalists the launcher kept in `judge/reference/a100.txt` (`judge/README.md`). Its candidate is not
  the step 7 published here (the md5 of its line 1), and its SCORE is not this rung's: only its finalists, the
  reference, are used here.
- `emu/`: the host emulator (`ampere/emu`, no GPU) on these sources: `emu_all.txt` (`./emu all`) and `planted_bugs.txt`
  (`python3 ampere/emu/planted_bugs.py -j 2`).

The ncu counters are in the `*_raw.csv` files, exported with `ncu --import <report> --csv --page raw`. Paths in the
candidate lines are relative to the repository's root.

## The card, in the judge's outputs

In the judge's 18 outputs, at their 17 shapes, each arm's mean power is 187.6 to 399.6 W, its mean SM clock 1107 to
1410 MHz and its NVML throttle reasons 0x0 or 0x4 (columns `w_ref`, `w_cand`, `mhz_ref`, `mhz_cand`, `throttle_ref`,
`throttle_cand`), and no `WARNING` line is written. In NVIDIA's `nvml.h`, 0x4 is `nvmlClocksEventReasonSwPowerCap`: "The
clocks have been optimized to ensure not to exceed currently set power limits"; it is not one of the bits that the judge
counts as a thermal or hardware throttle (`judge/judge.cu`:482).

## The sessions

One session per candidate (`modal/session.py`, `--cublaslt` the configuration that the judge's final ranked first at
ladder for that candidate in the arrival): `verif.cu` under compute-sanitizer (`*_01_sanitizer.txt`: plain run,
memcheck, racecheck and synccheck each rc 0 and `OK (tolerance 2e-3)`; memcheck and synccheck `ERROR SUMMARY: 0
errors`, racecheck `0 hazards displayed`), then ncu at ladder, one call of each kernel (`*_02_ncu_run.txt`: `kernels
profiled`; the same number of rows in `*_02_ncu_raw.csv`): each variant of the step, then cuBLASLt.

| candidate | session | kernels under ncu |
|---|---|---:|
| reference point | `20261026-164134_ref_simple` | 2 |
| step 0 | `20261026-164109_step0` | 2 |
| step 1 | `20261026-164504_step1` | 6 |
| step 2 | `20261026-164320_step2` | 4 |
| step 3 | `20261026-164521_step3` | 5 |
| step 4 | `20261026-164421_step4` | 5 |
| step 5 | `20261026-164239_step5` | 5 |
| step 6 | `20261026-164223_step6` | 9 |
| step 7 | `20261026-164426_step7` | 26 |

## The probes

A case is t(arm) / t(v0) - 1, in %, from the two medians printed in the same `probe.txt` at the same shape, rounded to
one decimal (halves away from zero); a case "alone against" an arm is t(that variant) / t(that arm) - 1. `twin` is the
step timed a second time. A dash: at that shape the arm launches the kernel of the arm it is compared with, with the
same grid and arguments (the note under the table gives the lines of the code); the probe times it all the same, and its
median stays in `probe.txt`. The noise of a probe is the largest, in absolute value, of its `twin` cases, of its cases
shown as a dash against the step and of the cases of a variant named under its table (at least 0.1 %); its rms noise is
the square root of the mean of the squares of these same gaps, unrounded. When the largest of these gaps is more than
twice the second largest (counted at least 0.1 %), the probe is run again once, the same way, in both series at the same
time: the second runs decide, and the first ones stay here. Each step has two probes that decide, one per series, side
by side in its table: each cell gives series 1 / series 2. The mechanisms' means, from these same medians, are in the
rung's README (`../README.md`).

Step 1 (`20261026-1733_step1_probe` / `20261026-1734_step1_probe`), time against step 1:

| shape | twin | without 1a | without 1b | 1c alone against step 0 | step 0 |
|---|---:|---:|---:|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | +2.2 % / +2.3 % | -22.9 % / -23.0 % | -11.9 % / -12.0 % | 0.0 % / 0.0 % |
| qkv_m128 | 0.0 % / 0.0 % | +3.5 % / +3.5 % | -27.1 % / -27.0 % | -20.8 % / -20.8 % | -1.9 % / -1.8 % |
| qkv_m512 | 0.0 % / 0.0 % | +3.4 % / +3.4 % | +9.3 % / +9.4 % | -22.2 % / -22.5 % | +46.8 % / +47.0 % |
| qkv_m2048 | 0.0 % / 0.0 % | +3.8 % / +3.5 % | +45.7 % / +45.6 % | -22.5 % / -22.5 % | +96.8 % / +96.7 % |
| o_m16 | 0.0 % / 0.0 % | +3.1 % / +3.0 % | -24.0 % / -23.9 % | -12.4 % / -12.7 % | -3.8 % / -3.5 % |
| o_m128 | 0.0 % / 0.0 % | +3.7 % / +3.6 % | -27.0 % / -27.0 % | -20.9 % / -21.2 % | -5.6 % / -5.4 % |
| o_m512 | 0.0 % / 0.0 % | +3.7 % / +3.6 % | +8.9 % / +8.9 % | -20.8 % / -21.1 % | +40.5 % / +41.3 % |
| o_m2048 | 0.0 % / 0.0 % | +3.7 % / +3.7 % | +44.8 % / +44.9 % | -20.2 % / -20.7 % | +86.6 % / +87.8 % |
| gateup_m16 | 0.0 % / 0.0 % | +3.8 % / +3.7 % | +25.6 % / +25.6 % | -17.9 % / -17.9 % | +67.1 % / +67.2 % |
| gateup_m128 | 0.0 % / 0.0 % | +4.0 % / +3.9 % | +21.0 % / +20.9 % | -23.4 % / -23.3 % | +63.1 % / +63.3 % |
| gateup_m512 | 0.0 % / 0.0 % | +3.6 % / +3.6 % | +35.9 % / +35.0 % | -23.2 % / -23.2 % | +83.2 % / +82.3 % |
| gateup_m2048 | 0.0 % / 0.0 % | +3.7 % / +3.6 % | +40.0 % / +38.7 % | -23.7 % / -23.5 % | +89.8 % / +88.1 % |
| down_m16 | 0.0 % / 0.0 % | +3.0 % / +3.1 % | -24.1 % / -24.0 % | -11.8 % / -11.9 % | -3.4 % / -3.2 % |
| down_m128 | 0.0 % / +0.2 % | +4.5 % / +4.5 % | -27.2 % / -27.2 % | -21.4 % / -21.6 % | -4.2 % / -4.0 % |
| down_m512 | 0.0 % / 0.0 % | +4.1 % / +4.0 % | +8.6 % / +8.6 % | -21.4 % / -21.5 % | +43.1 % / +43.1 % |
| down_m2048 | 0.0 % / 0.0 % | +3.9 % / +3.8 % | +44.5 % / +44.5 % | -20.7 % / -21.0 % | +89.8 % / +90.4 % |
| ladder | 0.0 % / 0.0 % | +4.6 % / +4.6 % | +41.7 % / +41.7 % | -24.0 % / -24.0 % | +82.7 % / +82.7 % |

The noise: 0.1 % / 0.2 %; the rms noise: 0.009 % / 0.047 %. `20261026-1647_step1_probe` (series 2) is the first run of
this probe; in series 1, its first run got no card of the card line in 8 tries and left no folder. Redone in both
series, at the same time, and not for a gap: without a run in series 1, no two runs could be shown to have run at the
same time (`two_cards_first.txt`: no `probe.txt` for the run of series 1, no proof for step 1); the table above takes
the second runs only, which `two_cards.txt` shows running together.

Step 2 (`20261026-1747_step2_probe` / `20261026-1748_step2_probe`), time against step 2:

| shape | twin | without 2b | step 1 | 2a alone against step 1 |
|---|---:|---:|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | +9.4 % / +9.4 % | +2.2 % / +2.2 % | +7.0 % / +7.0 % |
| qkv_m128 | 0.0 % / 0.0 % | +8.8 % / +8.8 % | +1.2 % / +0.9 % | +7.5 % / +7.9 % |
| qkv_m512 | 0.0 % / 0.0 % | +8.9 % / +8.8 % | +1.4 % / +1.2 % | +7.4 % / +7.5 % |
| qkv_m2048 | 0.0 % / 0.0 % | +8.9 % / +8.8 % | +2.0 % / +1.8 % | +6.7 % / +7.0 % |
| o_m16 | 0.0 % / 0.0 % | +9.2 % / +9.2 % | +1.6 % / +1.6 % | +7.5 % / +7.5 % |
| o_m128 | 0.0 % / 0.0 % | +8.8 % / +8.8 % | +0.8 % / +0.5 % | +8.0 % / +8.2 % |
| o_m512 | 0.0 % / 0.0 % | +8.8 % / +8.8 % | +1.5 % / +1.4 % | +7.2 % / +7.3 % |
| o_m2048 | 0.0 % / 0.0 % | +8.7 % / +8.8 % | +1.7 % / +1.6 % | +6.9 % / +7.1 % |
| gateup_m16 | -0.1 % / 0.0 % | +8.9 % / +9.0 % | +1.8 % / +1.9 % | +7.0 % / +7.0 % |
| gateup_m128 | 0.0 % / 0.0 % | +8.5 % / +8.5 % | +0.4 % / +0.5 % | +8.0 % / +8.0 % |
| gateup_m512 | -0.1 % / 0.0 % | +7.2 % / +8.3 % | 0.0 % / +0.1 % | +7.2 % / +8.2 % |
| gateup_m2048 | +0.1 % / 0.0 % | +7.0 % / +6.7 % | +0.1 % / +0.1 % | +6.9 % / +6.6 % |
| down_m16 | 0.0 % / 0.0 % | +9.4 % / +9.4 % | +1.8 % / +1.8 % | +7.5 % / +7.5 % |
| down_m128 | 0.0 % / 0.0 % | +9.3 % / +9.3 % | +0.8 % / +0.4 % | +8.5 % / +8.8 % |
| down_m512 | 0.0 % / 0.0 % | +9.1 % / +9.1 % | +1.3 % / +1.0 % | +7.8 % / +8.0 % |
| down_m2048 | 0.0 % / 0.0 % | +9.1 % / +9.1 % | +1.6 % / +1.4 % | +7.4 % / +7.6 % |
| ladder | 0.0 % / 0.0 % | +8.7 % / +8.7 % | +2.4 % / +2.3 % | +6.2 % / +6.3 % |

The noise: 0.1 % / 0.1 %; the rms noise: 0.045 % / 0.016 %. `20261026-1646_step2_probe` (series 1) and
`20261026-1647_step2_probe` (series 2) are the first runs of this probe. Redone in both series, at the same time, and
not for a gap: these two runs could not be shown to have run at the same time (`two_cards_first.txt`: no sure moment for
the run of series 1, no proof for step 2); the table above takes the second runs only, which `two_cards.txt` shows
running together.

Step 3 (`20261026-1756_step3_probe` / `20261026-1757_step3_probe`), time against step 3:

| shape | twin | without 3b (one stage) | step 2 | 3a alone against step 2 |
|---|---:|---:|---:|---:|
| qkv_m16 | -0.1 % / 0.0 % | +44.1 % / +42.7 % | +41.8 % / +40.6 % | +1.6 % / +1.5 % |
| qkv_m128 | 0.0 % / 0.0 % | +35.2 % / +34.9 % | +56.8 % / +56.7 % | -13.8 % / -13.9 % |
| qkv_m512 | 0.0 % / 0.0 % | +37.7 % / +35.4 % | +55.9 % / +53.6 % | -11.7 % / -11.8 % |
| qkv_m2048 | 0.0 % / 0.0 % | +27.8 % / +26.4 % | +45.3 % / +42.3 % | -12.0 % / -11.2 % |
| o_m16 | 0.0 % / 0.0 % | +41.0 % / +39.7 % | +45.8 % / +46.8 % | -3.3 % / -4.8 % |
| o_m128 | 0.0 % / 0.0 % | +34.7 % / +34.3 % | +56.9 % / +56.6 % | -14.1 % / -14.2 % |
| o_m512 | 0.0 % / 0.0 % | +36.1 % / +36.3 % | +55.6 % / +55.5 % | -12.5 % / -12.3 % |
| o_m2048 | -0.1 % / 0.0 % | +31.3 % / +28.8 % | +48.5 % / +45.7 % | -11.6 % / -11.6 % |
| gateup_m16 | -0.1 % / 0.0 % | +41.5 % / +40.0 % | +40.6 % / +39.8 % | +0.6 % / +0.2 % |
| gateup_m128 | +0.2 % / -0.1 % | +37.8 % / +34.3 % | +55.0 % / +51.0 % | -11.1 % / -11.0 % |
| gateup_m512 | -0.1 % / 0.0 % | +30.6 % / +29.6 % | +45.4 % / +42.9 % | -10.2 % / -9.3 % |
| gateup_m2048 | 0.0 % / +0.1 % | +29.9 % / +28.3 % | +43.0 % / +41.1 % | -9.2 % / -9.1 % |
| down_m16 | 0.0 % / 0.0 % | +39.1 % / +38.0 % | +48.4 % / +48.7 % | -6.3 % / -7.2 % |
| down_m128 | 0.0 % / 0.0 % | +35.5 % / +34.9 % | +56.4 % / +56.3 % | -13.3 % / -13.7 % |
| down_m512 | 0.0 % / 0.0 % | +38.6 % / +39.2 % | +56.1 % / +56.2 % | -11.2 % / -10.9 % |
| down_m2048 | -0.2 % / 0.0 % | +32.9 % / +30.7 % | +48.4 % / +45.9 % | -10.4 % / -10.4 % |
| ladder | 0.0 % / 0.0 % | +34.3 % / +31.6 % | +43.0 % / +40.4 % | -6.1 % / -6.3 % |

The noise: 0.2 % / 0.1 %; the rms noise: 0.073 % / 0.040 %. `20261026-1646_step3_probe` (series 1) and
`20261026-1647_step3_probe` (series 2) are the first runs of this probe. Redone in both series, at the same time, and
not for a gap: these two runs could not be shown to have run at the same time (`two_cards_first.txt`: no sure moment for
the run of series 1, no proof for step 3); the table above takes the second runs only, which `two_cards.txt` shows
running together.

Step 4 (`20261026-1718_step4_probe` / `20261026-1719_step4_probe`), time against step 4:

| shape | twin | without 4a | without 4b |
|---|---:|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | 0.0 % / 0.0 % | -0.1 % / 0.0 % |
| qkv_m128 | 0.0 % / 0.0 % | +1.9 % / +1.9 % | -0.1 % / -0.1 % |
| qkv_m512 | +0.1 % / -0.3 % | +1.4 % / +1.1 % | +1.7 % / +1.3 % |
| qkv_m2048 | 0.0 % / 0.0 % | +1.2 % / +1.3 % | +3.4 % / +3.3 % |
| o_m16 | 0.0 % / 0.0 % | +0.3 % / +0.2 % | +0.1 % / +0.1 % |
| o_m128 | 0.0 % / 0.0 % | +2.1 % / +2.1 % | -0.1 % / 0.0 % |
| o_m512 | 0.0 % / 0.0 % | +1.8 % / +1.8 % | +0.3 % / 0.0 % |
| o_m2048 | -0.1 % / -0.1 % | +1.2 % / +1.2 % | +2.1 % / +1.9 % |
| gateup_m16 | -0.1 % / 0.0 % | 0.0 % / 0.0 % | -0.5 % / -0.2 % |
| gateup_m128 | 0.0 % / -0.1 % | +0.3 % / +0.9 % | 0.0 % / +0.1 % |
| gateup_m512 | -0.1 % / 0.0 % | +1.2 % / +1.2 % | +8.7 % / +7.7 % |
| gateup_m2048 | 0.0 % / 0.0 % | +1.3 % / +1.3 % | +11.2 % / +10.7 % |
| down_m16 | 0.0 % / 0.0 % | 0.0 % / +0.1 % | 0.0 % / 0.0 % |
| down_m128 | 0.0 % / 0.0 % | +2.9 % / +2.9 % | 0.0 % / -0.1 % |
| down_m512 | 0.0 % / 0.0 % | +2.3 % / +2.2 % | +0.7 % / +0.3 % |
| down_m2048 | -0.1 % / 0.0 % | +1.4 % / +1.7 % | +2.2 % / +2.3 % |
| ladder | 0.0 % / 0.0 % | +1.6 % / +1.6 % | +0.3 % / +0.5 % |

The noise: 0.1 % / 0.3 %; the rms noise: 0.046 % / 0.085 %. `20261026-1646_step4_probe` (series 1) and
`20261026-1647_step4_probe` (series 2) are the first runs of this probe. Redone: in series 2, the twin's largest gap
(0.4 %, at gateup_m128) was more than twice its second largest (0.1 %); the probe was run again the same way in both
series, at the same time, and the table above takes the second runs only. A probe is redone once only: in series 2, the
run that decides has its twin's largest gap (0.3 %, at qkv_m512) more than twice its second largest (0.1 %).

Step 5 (`20261026-1646_step5_probe` / `20261026-1647_step5_probe`), time against step 5:

| shape | twin | without 5a | without 5b |
|---|---:|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | +25.7 % / +22.6 % | +0.4 % / +0.4 % |
| qkv_m128 | 0.0 % / 0.0 % | +28.9 % / +28.8 % | +0.8 % / +0.8 % |
| qkv_m512 | 0.0 % / 0.0 % | +21.2 % / +21.1 % | +0.6 % / +0.7 % |
| qkv_m2048 | 0.0 % / 0.0 % | +18.8 % / +18.6 % | +0.8 % / +0.7 % |
| o_m16 | 0.0 % / 0.0 % | +24.7 % / +22.7 % | +0.6 % / +0.5 % |
| o_m128 | 0.0 % / 0.0 % | +28.9 % / +28.9 % | +0.8 % / +0.8 % |
| o_m512 | 0.0 % / 0.0 % | +27.2 % / +26.9 % | +0.7 % / +0.8 % |
| o_m2048 | -0.1 % / 0.0 % | +19.4 % / +19.3 % | +0.7 % / +0.7 % |
| gateup_m16 | 0.0 % / 0.0 % | +11.5 % / +6.8 % | 0.0 % / +0.1 % |
| gateup_m128 | 0.0 % / 0.0 % | +8.8 % / +8.4 % | +0.4 % / +0.4 % |
| gateup_m512 | 0.0 % / -0.1 % | +19.0 % / +18.5 % | +0.7 % / +0.6 % |
| gateup_m2048 | 0.0 % / 0.0 % | +19.2 % / +18.8 % | +0.8 % / +0.8 % |
| down_m16 | 0.0 % / 0.0 % | +30.9 % / +30.3 % | +0.4 % / +0.4 % |
| down_m128 | 0.0 % / 0.0 % | +29.9 % / +29.9 % | +0.8 % / +0.8 % |
| down_m512 | 0.0 % / 0.0 % | +27.5 % / +27.2 % | +0.7 % / +0.8 % |
| down_m2048 | -0.1 % / +0.1 % | +19.7 % / +19.7 % | +0.8 % / +0.8 % |
| ladder | -0.1 % / 0.0 % | +16.6 % / +16.6 % | +0.5 % / +0.5 % |

The noise: 0.1 % / 0.1 %; the rms noise: 0.037 % / 0.031 %.

Step 6 (`20261026-1805_step6_probe` / `20261026-1806_step6_probe`), time against step 6:

| shape | twin | without 6a | without 6b | without 6c |
|---|---:|---:|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | - | +194.1 % / +208.2 % | - |
| qkv_m128 | 0.0 % / 0.0 % | +59.1 % / +56.1 % | - | - |
| qkv_m512 | 0.0 % / 0.0 % | +12.0 % / +12.9 % | - | - |
| qkv_m2048 | -0.1 % / 0.0 % | - | - | - |
| o_m16 | 0.0 % / 0.0 % | - | +169.3 % / +171.3 % | - |
| o_m128 | +0.1 % / -0.1 % | +72.9 % / +68.8 % | - | - |
| o_m512 | 0.0 % / 0.0 % | +15.7 % / +16.5 % | - | - |
| o_m2048 | -0.1 % / 0.0 % | - | - | - |
| gateup_m16 | 0.0 % / 0.0 % | - | +96.2 % / +100.8 % | - |
| gateup_m128 | 0.0 % / 0.0 % | - | - | +3.0 % / +4.4 % |
| gateup_m512 | 0.0 % / +0.1 % | - | - | +0.7 % / +1.0 % |
| gateup_m2048 | +0.1 % / 0.0 % | - | - | - |
| down_m16 | 0.0 % / 0.0 % | - | +115.0 % / +107.6 % | - |
| down_m128 | +0.1 % / +0.1 % | +73.5 % / +69.4 % | - | - |
| down_m512 | 0.0 % / 0.0 % | +15.1 % / +14.4 % | - | - |
| down_m2048 | 0.0 % / 0.0 % | - | - | - |
| ladder | 0.0 % / 0.0 % | - | - | - |

The noise: 0.1 % / 0.1 %; the rms noise: 0.049 % / 0.044 %. A dash: at that shape the variant launches step 6's kernels,
with the same grids and arguments (`candidate_gemm` in `step6.cu`:152-181): 6b comes first, at M <= 16 (`step6.cu`:173,
`m16` in `step6.cuh`:787-793), so there v1 and v3 run 6b's kernels; 6c then takes gateup_m128 and gateup_m512
(`step6.cu`:176, `phase` in `step6.cuh`:952-960), where v1 and v2 run it too; elsewhere, the plan keeps step 5's 128 x
256 where it gives at least three waves of tiles (`choose`, `step6.cuh`:491-495, and `step6.cu`:57-62): at qkv_m2048,
o_m2048, gateup_m2048, down_m2048 and the ladder, v1 runs the same kernel. v2 (without 6b) differs only at M <= 16, v3
(without 6c) only where `phase` holds. `20261026-1646_step6_probe` (series 1) and `20261026-1647_step6_probe` (series 2)
are the first runs of this probe. Redone in both series, at the same time, and not for a gap: these two runs could not
be shown to have run at the same time (`two_cards_first.txt`: no sure moment for the run of series 2, no proof for step
6); the table above takes the second runs only, which `two_cards.txt` shows running together.

Step 7 (`20261026-1646_step7_probe` / `20261026-1647_step7_probe`), time against step 7:

| shape | twin | without 7a | without 7b | without 7c | without 7d | without 7e | without 7f | without 7g | without 7h | 7a alone against without 7a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv_m16 | -0.5 % / -0.5 % | - | - | - | +0.5 % / +0.5 % | - | - | - | - | - |
| qkv_m128 | 0.0 % / 0.0 % | - | - | - | - | - | - | - | - | - |
| qkv_m512 | -0.1 % / 0.0 % | - | - | - | - | - | - | - | - | - |
| qkv_m2048 | -0.1 % / 0.0 % | - | - | - | - | - | - | - | - | - |
| o_m16 | -0.4 % / -0.5 % | - | - | - | +0.6 % / +0.5 % | - | - | - | - | - |
| o_m128 | 0.0 % / 0.0 % | +19.7 % / +19.5 % | +16.4 % / +19.9 % | - | +5.3 % / +7.7 % | - | - | +1.4 % / +2.5 % | - | -2.8 % / +0.3 % |
| o_m512 | 0.0 % / 0.0 % | +9.9 % / +10.0 % | - | - | -1.7 % / -1.5 % | - | +1.0 % / +1.5 % | - | +0.5 % / +0.5 % | -9.1 % / -9.1 % |
| o_m2048 | +0.1 % / -0.1 % | - | - | - | - | - | - | - | - | - |
| gateup_m16 | -0.4 % / -0.4 % | - | - | - | +1.5 % / +1.4 % | - | - | - | - | - |
| gateup_m128 | 0.0 % / 0.0 % | - | - | - | +3.0 % / +3.1 % | - | - | - | +0.9 % / +1.1 % | - |
| gateup_m512 | -0.1 % / 0.0 % | - | - | - | -1.3 % / -0.9 % | - | - | - | -0.2 % / -0.3 % | - |
| gateup_m2048 | -0.1 % / 0.0 % | - | - | - | +0.1 % / 0.0 % | -0.1 % / 0.0 % | - | - | - | - |
| down_m16 | -0.2 % / -0.3 % | - | - | - | +3.0 % / +3.0 % | - | - | - | - | - |
| down_m128 | -0.1 % / 0.0 % | +28.7 % / +29.1 % | +3.1 % / +4.9 % | - | 0.0 % / +0.9 % | - | - | +0.4 % / +0.5 % | - | -19.9 % / -18.8 % |
| down_m512 | 0.0 % / -0.1 % | +18.6 % / +17.9 % | - | +6.5 % / +6.0 % | - | - | - | - | +1.7 % / +1.5 % | -15.7 % / -15.2 % |
| down_m2048 | 0.0 % / 0.0 % | - | - | - | 0.0 % / 0.0 % | +1.5 % / +1.5 % | - | - | - | - |
| ladder | +0.1 % / 0.0 % | - | - | - | - | - | - | - | - | - |

The noise: 0.5 % / 0.5 %; the rms noise: 0.163 % / 0.173 %. A dash: at that shape the variant launches step 7's kernels,
with the same grids and arguments (`candidate_gemm` in `step7.cu`:294-350, and `step6_run`, :131-155): 6b at M <= 16
(:328) runs in every variant, with 7d's hints but in v4; 6c by its rule (`phase`, `step6.cuh`:952-960) at gateup_m128
and gateup_m512, and 7f by its own (`phase_in_place`, `step7.cuh`:156-164) at o_m512 (:331-332); 7e by `tail_parts`
(`step6.cuh`:1126-1144) at gateup_m2048 and down_m2048 (:336); 7a's rule (`choose`, `step6.cuh`:491-513) splits K at
o_m128 and down_m128 (3 parts, reduced by their own blocks in 7g's kernel) and at o_m512 and down_m512 (2 parts in
place, where 7f and 7c take over: at down_m512, 5 parts, `inplace_parts`, `step7.cuh`:135-150); 7d's hints are only in
the kernels of 6b, 6c, 7b, 7e, 7f and 7g, 7h's reads only in the parts in place of 7a and 7c and in 6c's kernel. So v1
(without 7a) differs from step 7 only at the four shapes where 7a splits K, v2 (without 7b) and v15 (without 7g) only at
o_m128 and down_m128, v3 (without 7c) only at down_m512, v4 (without 7d) only at the eleven shapes where those kernels
run, v13 (without 7e) only at gateup_m2048 and down_m2048, v14 (without 7f) only at o_m512, and v16 (without 7h) only at
o_m512, gateup_m128, gateup_m512 and down_m512. In the last column, 7a without 7b against step 7 without 7a, at the four
shapes where 7a splits K (at o_m512 and down_m512, the arm without 7b runs 7f and 7c). The noise is reached at qkv_m16
in both series, by the twin and at the dashes of v1 and v15, and in series 2 also at o_m16, by the twin and at the
dashes of v1, v3, v13, v14 and v15.

## ncu in the probes of step 7

At one shape for each mechanism of step 7, where it acts (`20261026-1714_step7_<mechanism>_ncu_probe`: 7a and 7c at
down_m512, 7b and 7g at o_m128, 7d at down_m16, 7e at down_m2048, 7f at o_m512, 7h at gateup_m128), each arm in its own
process, called twice under ncu: the counters are those of its second call (ID 1 of its CSV; IDs 2 and 3 when a call
runs two kernels, each cited on its own row). `x:` is the configuration that the judge's final ranked first at that
shape for step 7 in the arrival (`judge/20261026-163446_step7/output.txt`, its REFERENCE line of rank 0). These counters
are a snapshot: under ncu, each call is replayed in 51 or 52 passes (`probe.txt`), the clocks not locked, so they show
what an arm runs (its kernel, its grid, its traffic), not its time, and judge nothing; the timed probes, in two series,
judge each mechanism (the tables above). At o_m512 and o_m128, the arm without 7f or 7g has fewer cycles than step 7,
though the timed probe finds it slower there.

| shape, arm | kernel, blocks | cycles | tensor active | DRAM read | DRAM write |
|---|---|---:|---:|---:|---:|
| down_m512, step 7 | `gemm_split<128, 256, 64, 64, 3, 2, 1>`, 640 | 1 221 716 | 91.0 % | 562.1 MB | 37.91 MB |
| down_m512, without 7a | `gemm<128, 128, 64, 32, 5>`, 256 | 1 468 378 | 94.6 % | 833.3 MB | 9.59 MB |
| down_m512, `x:` | ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_nn, 640 | 1 212 762 | 91.9 % | 562.1 MB | 37.90 MB |
| o_m128, step 7 | `gemm_parallel_own<128, 256, 64, 64, 3, 1>`, 96 | 117 979 | 78.0 % | 136.4 MB | 1.82 MB |
| o_m128, without 7b | `gemm_split<128, 256, 64, 64, 3, 1, 0>`, 96 | 113 099 | 82.5 % | 136.4 MB | 2.65 MB |
| o_m128, without 7b | `reduce_parts<128, 256>`, 32 | 42 115 | 0.0 % | 6.3 MB | 0.00 MB |
| o_m128, `x:` | ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_nn, 96 | 112 576 | 81.4 % | 136.3 MB | 2.55 MB |
| o_m128, `x:` | `cublasLt::splitKreduce_kernel<32, 16, ...>`, 512 | 19 074 | 2.3 % | 6.3 MB | 0.00 MB |
| down_m512, step 7 | `gemm_split<128, 256, 64, 64, 3, 2, 1>`, 640 | 1 224 789 | 91.1 % | 562.1 MB | 37.92 MB |
| down_m512, without 7c | `gemm_split<128, 128, 64, 32, 5, 2, 1>`, 512 | 1 253 490 | 92.4 % | 733.7 MB | 16.50 MB |
| down_m512, `x:` | ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_nn, 640 | 1 208 209 | 92.1 % | 562.1 MB | 37.91 MB |
| down_m16, step 7 | `gemm_stream<256, 4, 1>`, 108 | 363 713 | 9.8 % | 470.7 MB | 1.91 MB |
| down_m16, without 7d | `gemm_stream<256, 4, 0>`, 108 | 375 769 | 9.6 % | 477.4 MB | 2.84 MB |
| down_m16, `x:` | `cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x6_nn_align8>`, 192 | 378 306 | 41.5 % | 471.0 MB | 2.99 MB |
| down_m16, `x:` | `cublasLt::splitKreduce_kernel<32, 16, ...>`, 256 | 6 358 | 1.1 % | 0.8 MB | 0.00 MB |
| down_m2048, step 7 | `gemm_tail<128, 256, 64, 64, 3, 1>`, 752 | 4 557 719 | 96.4 % | 1181.3 MB | 45.00 MB |
| down_m2048, without 7e | `gemm<1, 1>`, 512 | 4 708 167 | 97.9 % | 1101.2 MB | 30.82 MB |
| down_m2048, `x:` | `Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_64x3_nn_align8>`, 512 | 4 659 589 | 98.8 % | 1424.0 MB | 30.81 MB |
| o_m512, step 7 | `gemm_phase_sums<128, 256, 64, 64, 3, 1>`, 108 | 390 850 | 81.8 % | 145.8 MB | 6.87 MB |
| o_m512, without 7f | `gemm_split<128, 128, 64, 32, 5, 2, 1>`, 512 | 374 570 | 88.5 % | 182.3 MB | 14.69 MB |
| o_m512, `x:` | `Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_nn_align8>`, 512 | 367 390 | 90.8 % | 183.6 MB | 15.24 MB |
| o_m128, step 7 | `gemm_parallel_own<128, 256, 64, 64, 3, 1>`, 96 | 120 886 | 76.2 % | 136.4 MB | 1.77 MB |
| o_m128, without 7g | `gemm_parallel<128, 256, 64, 64, 3, 1>`, 96 | 118 726 | 76.9 % | 136.4 MB | 1.93 MB |
| o_m128, `x:` | ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_nn, 96 | 113 646 | 80.9 % | 136.3 MB | 2.57 MB |
| o_m128, `x:` | `cublasLt::splitKreduce_kernel<32, 16, ...>`, 512 | 19 096 | 2.3 % | 6.3 MB | 0.00 MB |
| gateup_m128, step 7 | `gemm_phase_sums<128, 256, 64, 64, 3, 1>`, 108 | 712 585 | 78.6 % | 942.1 MB | 10.57 MB |
| gateup_m128, without 7h | `gemm_phase<128, 256, 64, 64, 3, 1>`, 108 | 714 851 | 78.1 % | 942.1 MB | 10.85 MB |
| gateup_m128, `x:` | `Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_nn_align8>`, 448 | 850 520 | 73.1 % | 949.4 MB | 15.70 MB |

## The counters

The names, in the CSV files, of the counters cited here and in the rung's README: cycles `sm__cycles_elapsed.max`;
tensor active `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`; DRAM read and write
`dram__bytes_read.sum`, `dram__bytes_write.sum`; L2 sector hits `lts__t_sector_hit_rate.pct`; excess shared-memory
wavefronts `derived__memory_l1_wavefronts_shared_excessive`; instructions `smsp__inst_executed.sum`; blocks
`launch__grid_size`. Each value is that of the kernel's row (its Kernel Name and ID), in % and MB to one decimal (DRAM
write to two). In 8 of the 9 sessions, the cuBLASLt configuration analyzed at ladder runs
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_nn, 432 blocks; in step 5's, the configuration ranked first at
ladder for step 5 runs `cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_nn_align8>`, 864
blocks.
