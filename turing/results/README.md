# Results of the Turing rung

Raw outputs of the GPU sessions on Modal's T4, one folder per session, named by its date and time, and of the host
emulator (`emu/`). The tools write their outputs in English. Paths start from this folder; those of the tools
(`judge/judge.cu`, `probe/probe.cu`, `modal/session.py`, `turing/emu`) from the repository's root.

## The outputs

- `judge/<date>-<time>_<candidate>/`: the judge's output (`judge/judge.cu`), `output.txt`. Its lines 1 and 2 give the
  md5 of each judged source, of the judge, of its launcher, of the shapes file and of the reference file: those of this
  repository's files, which `md5sum` checks. Container 1: `judge/20261025-172635_*`; container 2, the arrival:
  `judge/20261025-172649_*`; 8 candidates each (`ref_simple`, the reference point, and `step0` to `step6`).
- `<date>-<time>_<steps>/`: a session of `modal/session.py`: the run (`00_run.txt`: the md5 of the launcher and of each
  source, the card line, the cuBLASLt configuration analyzed and where it comes from), the machine (`00_machine.txt`),
  `verif.cu` under compute-sanitizer (`*_01_sanitizer.txt`), ncu at ladder (`*_02_ncu_run.txt`, and its counters in
  `*_02_ncu_raw.csv`). Here, one session per candidate, 8 folders (below, "The sessions").
- `<date>-<time>_<step>_probe/`: an interleaved probe (`probe/probe.cu`): `probe.txt` (its lines 1 and 2: the md5 of
  each source, of the probe and of its launcher, the card line). Here, two series of the six probes, each probe in a
  container of its own: series 1, `20261025-1737_step1_probe` to `20261025-1737_step6_probe`; series 2,
  `20261025-1738_step1_probe` to `20261025-1738_step6_probe`. Steps 3, 4 and 5 were run again in both series (below):
  `20261025-1808_step3_probe`, `20261025-1809_step4_probe` and `20261025-1819_step5_probe` (series 1),
  `20261025-1809_step3_probe`, `20261025-1810_step4_probe` and `20261025-1820_step5_probe` (series 2).
- `<date>-<time>_<step>_<mechanism>_ncu_probe/`: the probe under ncu only, at a shape where that mechanism acts:
  `probe.txt`, the counters of each arm in `ncu_<arm>_<shape>_raw.csv`, and the version of Nsight Compute that
  exported them (`ncu_export_version.txt`). Here, one per mechanism of step 6, `20261025-1737_step6_6a_ncu_probe` to
  `20261025-1737_step6_6f_ncu_probe`, and `20261025-1737_step6_indice_6b_ncu_probe` for 6b's hint.
- `two_cards.txt`: for each step, the two probes that decide (one per series), the start and the end of each one's app
  on Modal (in UTC), its ARM lines and rounds, hence a lower bound of its work on the card (each round, a third of a
  block of warm-up, then the timed block: `probe/probe.cu`) and the time it surely ran; then the time the two surely
  ran together. A container keeps its card for its whole life, so two containers running at the same time have two
  different cards: the file shows it at the 6 steps, and says nothing about the hosts.
- `two_cards_first.txt`: the same, with the first runs of step 5 (the five other steps as in `two_cards.txt`): no sure
  moment for the run of series 2 (its app on Modal lasted more than twice its work), hence no proof for step 5.
- `full_pass/20261023-165615_step6/output.txt`: the judge's full pass (without `--reference`) of 23/10/2026 on this
  card line, whose finalists the launcher kept in `judge/reference/t4.txt` (`judge/README.md`). Its candidate is not
  the step 6 published here (the md5 of its line 1), and its SCORE is not this rung's: only its finalists, the
  reference, are used here.
- `emu/`: the host emulator (`turing/emu`, no GPU) on these sources: `emu_all.txt` (`./emu all`) and `planted_bugs.txt`
  (`python3 turing/emu/planted_bugs.py -j 2`).

The ncu counters are in the `*_raw.csv` files, exported with `ncu --import <report> --csv --page raw`. Paths in the
candidate lines are relative to the repository's root.

## The card, in the judge's outputs

In the judge's 16 outputs, at their 17 shapes, each arm's mean power is 67.6 to 70.5 W, its mean SM clock 525 to
1531 MHz and its NVML throttle reasons 0x4 (columns `w_ref`, `w_cand`, `mhz_ref`, `mhz_cand`, `throttle_ref`,
`throttle_cand`), and no `WARNING` line is written. In NVIDIA's `nvml.h`, 0x4 is `nvmlClocksEventReasonSwPowerCap`:
"The clocks have been optimized to ensure not to exceed currently set power limits"; it is not one of the bits that
the judge counts as a thermal or hardware throttle (`judge/judge.cu`:482).

## The sessions

One session per candidate (`modal/session.py`, `--cublaslt` the configuration that the judge's final ranked first at
ladder for that candidate in the arrival): `verif.cu` under compute-sanitizer (`*_01_sanitizer.txt`: plain run,
memcheck, racecheck and synccheck each rc 0 and `OK (tolerance 2e-3)`; memcheck and synccheck `ERROR SUMMARY: 0
errors`, racecheck `0 hazards displayed`), then ncu at ladder, one call of each kernel (`*_02_ncu_run.txt`: `kernels
profiled`; the same number of rows in `*_02_ncu_raw.csv`): each variant of the step, then cuBLASLt.

| candidate | session | kernels under ncu |
|---|---|---:|
| reference point | `20261025-173234_ref_simple` | 2 |
| step 0 | `20261025-173206_step0` | 2 |
| step 1 | `20261025-173352_step1` | 6 |
| step 2 | `20261025-173201_step2` | 4 |
| step 3 | `20261025-173229_step3` | 5 |
| step 4 | `20261025-173234_step4` | 3 |
| step 5 | `20261025-173204_step5` | 4 |
| step 6 | `20261025-173621_step6` | 10 |

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

Step 1 (`20261025-1737_step1_probe` / `20261025-1738_step1_probe`), time against step 1:

| shape | twin | without 1a | without 1b | 1c alone against step 0 | step 0 |
|---|---:|---:|---:|---:|---:|
| qkv_m16 | -0.3 % / +0.1 % | +6.8 % / +6.7 % | +22.2 % / +21.6 % | -3.3 % / -3.1 % | +27.8 % / +24.9 % |
| qkv_m128 | -0.1 % / -0.1 % | +4.6 % / +4.3 % | +21.9 % / +22.3 % | -3.2 % / -3.4 % | +22.8 % / +22.4 % |
| qkv_m512 | +0.1 % / -0.1 % | +5.4 % / +4.7 % | +32.2 % / +33.8 % | -4.4 % / -4.4 % | +29.1 % / +29.9 % |
| qkv_m2048 | -0.1 % / -0.1 % | +0.9 % / +0.4 % | +36.9 % / +37.8 % | -4.4 % / -3.6 % | +34.2 % / +33.9 % |
| o_m16 | +0.2 % / 0.0 % | +23.3 % / +16.0 % | +5.2 % / +5.8 % | -2.1 % / -2.0 % | +6.5 % / +7.6 % |
| o_m128 | 0.0 % / 0.0 % | +19.2 % / +18.1 % | +5.2 % / +8.1 % | -5.3 % / -5.7 % | +2.3 % / +4.9 % |
| o_m512 | -0.1 % / -0.1 % | 0.0 % / -0.2 % | +31.8 % / +32.3 % | -4.6 % / -4.9 % | +29.3 % / +28.6 % |
| o_m2048 | +0.3 % / -0.1 % | +3.0 % / +1.7 % | +37.3 % / +38.6 % | -6.5 % / -7.8 % | +35.7 % / +36.2 % |
| gateup_m16 | +0.2 % / 0.0 % | +0.8 % / -0.1 % | +23.6 % / +21.4 % | -2.4 % / -1.4 % | +24.5 % / +21.1 % |
| gateup_m128 | +0.2 % / +0.3 % | -0.4 % / -0.9 % | +23.6 % / +23.9 % | -5.5 % / -5.1 % | +22.5 % / +19.9 % |
| gateup_m512 | +0.2 % / +0.1 % | -0.6 % / -0.5 % | +23.3 % / +23.9 % | -3.1 % / -2.1 % | +19.0 % / +16.8 % |
| gateup_m2048 | 0.0 % / +0.2 % | -0.7 % / -0.6 % | +22.8 % / +23.1 % | -2.9 % / -4.2 % | +18.2 % / +18.3 % |
| down_m16 | 0.0 % / +0.1 % | +22.0 % / +16.0 % | +6.6 % / +8.4 % | -2.7 % / -1.6 % | +7.4 % / +7.9 % |
| down_m128 | 0.0 % / 0.0 % | +18.5 % / +16.9 % | +5.9 % / +9.8 % | -5.5 % / -6.5 % | +3.1 % / +6.5 % |
| down_m512 | +0.2 % / -0.3 % | -0.6 % / -1.7 % | +30.8 % / +31.2 % | -4.7 % / -4.9 % | +28.6 % / +28.5 % |
| down_m2048 | +0.2 % / -0.1 % | +3.6 % / +6.1 % | +39.8 % / +40.5 % | -8.6 % / -11.4 % | +40.5 % / +41.4 % |
| ladder | -0.1 % / 0.0 % | +7.0 % / +6.8 % | +41.2 % / +44.0 % | -5.7 % / -6.1 % | +35.8 % / +37.1 % |

The noise: 0.3 % / 0.3 %; the rms noise: 0.165 % / 0.129 %.

Step 2 (`20261025-1737_step2_probe` / `20261025-1738_step2_probe`), time against step 2:

| shape | twin | without 2b | step 1 | 2a alone against step 1 |
|---|---:|---:|---:|---:|
| qkv_m16 | -0.1 % / -0.3 % | +12.1 % / +10.4 % | +10.6 % / +10.9 % | +1.3 % / -0.4 % |
| qkv_m128 | -0.2 % / -0.1 % | +17.2 % / +18.0 % | +14.6 % / +14.9 % | +2.2 % / +2.6 % |
| qkv_m512 | 0.0 % / 0.0 % | +10.8 % / +12.0 % | +7.8 % / +8.2 % | +2.8 % / +3.5 % |
| qkv_m2048 | +0.4 % / +0.3 % | +3.5 % / +5.4 % | +2.5 % / +3.3 % | +1.0 % / +2.0 % |
| o_m16 | 0.0 % / 0.0 % | +6.8 % / +7.0 % | +0.7 % / +0.9 % | +6.1 % / +6.0 % |
| o_m128 | 0.0 % / 0.0 % | +10.0 % / +10.1 % | +0.5 % / +0.6 % | +9.5 % / +9.4 % |
| o_m512 | -0.2 % / +0.1 % | +11.9 % / +12.0 % | +12.4 % / +13.3 % | -0.5 % / -1.2 % |
| o_m2048 | 0.0 % / +0.1 % | +10.6 % / +11.9 % | +9.5 % / +10.6 % | +1.0 % / +1.2 % |
| gateup_m16 | 0.0 % / -0.1 % | +5.1 % / +7.1 % | +7.4 % / +7.7 % | -2.2 % / -0.6 % |
| gateup_m128 | 0.0 % / -0.1 % | +8.8 % / +9.2 % | +10.2 % / +10.6 % | -1.3 % / -1.2 % |
| gateup_m512 | +0.1 % / -0.1 % | +8.2 % / +8.4 % | +9.6 % / +10.0 % | -1.3 % / -1.5 % |
| gateup_m2048 | -0.2 % / +0.1 % | +8.6 % / +9.4 % | +10.3 % / +10.7 % | -1.5 % / -1.2 % |
| down_m16 | -0.1 % / 0.0 % | +7.1 % / +6.8 % | +0.2 % / 0.0 % | +6.9 % / +6.8 % |
| down_m128 | -0.1 % / 0.0 % | +9.2 % / +9.6 % | -0.1 % / 0.0 % | +9.4 % / +9.6 % |
| down_m512 | 0.0 % / +0.1 % | +10.7 % / +11.2 % | +10.4 % / +11.2 % | +0.3 % / 0.0 % |
| down_m2048 | -0.4 % / +0.1 % | -0.1 % / +4.5 % | -2.4 % / +0.2 % | +2.3 % / +4.3 % |
| ladder | 0.0 % / 0.0 % | +11.6 % / +13.1 % | +9.8 % / +10.8 % | +1.6 % / +2.1 % |

The noise: 0.4 % / 0.3 %; the rms noise: 0.158 % / 0.130 %.

Step 3 (`20261025-1808_step3_probe` / `20261025-1809_step3_probe`), time against step 3:

| shape | twin | without 3a | without 3b |
|---|---:|---:|---:|
| qkv_m16 | -0.2 % / -0.2 % | -3.8 % / -4.0 % | -1.0 % / -1.6 % |
| qkv_m128 | 0.0 % / 0.0 % | -2.4 % / -1.5 % | +0.3 % / +0.6 % |
| qkv_m512 | 0.0 % / -1.3 % | 0.0 % / -0.1 % | +5.3 % / +5.1 % |
| qkv_m2048 | 0.0 % / -0.2 % | +5.5 % / +9.6 % | +7.4 % / +15.4 % |
| o_m16 | 0.0 % / +0.1 % | -7.2 % / -6.8 % | -0.5 % / -1.4 % |
| o_m128 | 0.0 % / -0.3 % | -4.2 % / -3.3 % | +0.8 % / 0.0 % |
| o_m512 | 0.0 % / -0.1 % | -1.3 % / 0.0 % | +1.9 % / +4.0 % |
| o_m2048 | -0.1 % / +0.1 % | +3.4 % / +9.6 % | +3.0 % / +6.1 % |
| gateup_m16 | 0.0 % / -0.1 % | -2.0 % / +0.4 % | -0.8 % / -0.4 % |
| gateup_m128 | +0.1 % / 0.0 % | +0.2 % / +0.3 % | +0.1 % / -0.8 % |
| gateup_m512 | +0.2 % / +0.1 % | -3.3 % / -1.1 % | +8.1 % / +20.1 % |
| gateup_m2048 | 0.0 % / +0.2 % | +17.9 % / +19.3 % | +17.0 % / +28.3 % |
| down_m16 | +0.1 % / 0.0 % | -7.4 % / -6.6 % | 0.0 % / -1.6 % |
| down_m128 | -0.1 % / +0.1 % | -3.0 % / -2.3 % | +0.7 % / +0.7 % |
| down_m512 | 0.0 % / -0.1 % | +1.4 % / +2.5 % | +2.5 % / +3.1 % |
| down_m2048 | +0.2 % / -0.6 % | +13.2 % / +21.8 % | 0.0 % / -0.7 % |
| ladder | 0.0 % / +1.2 % | +2.3 % / +7.0 % | -0.7 % / +1.1 % |

The noise: 0.2 % / 1.3 %; the rms noise: 0.091 % / 0.467 %. `20261025-1737_step3_probe` (series 1) and
`20261025-1738_step3_probe` (series 2) are the first runs of this probe. Redone: in series 2, the twin's largest gap
(0.3 %, at qkv_m128) was more than twice its second largest (0.1 %); the probe was run again the same way in both
series, at the same time, and the table above takes the second runs only.

Step 4 (`20261025-1809_step4_probe` / `20261025-1810_step4_probe`), time against step 4:

| shape | twin | step 3 (without 4a) |
|---|---:|---:|
| qkv_m16 | -0.2 % / 0.0 % | +11.2 % / +11.7 % |
| qkv_m128 | 0.0 % / +0.1 % | +10.3 % / +11.0 % |
| qkv_m512 | 0.0 % / -0.5 % | +15.3 % / +13.8 % |
| qkv_m2048 | -0.1 % / -0.1 % | +8.4 % / +5.1 % |
| o_m16 | -0.2 % / 0.0 % | +16.3 % / +16.5 % |
| o_m128 | +0.1 % / +0.1 % | +20.0 % / +20.8 % |
| o_m512 | +0.1 % / +0.1 % | +15.2 % / +15.8 % |
| o_m2048 | -0.1 % / -0.1 % | +9.9 % / +7.8 % |
| gateup_m16 | 0.0 % / -0.1 % | +1.0 % / +0.1 % |
| gateup_m128 | 0.0 % / 0.0 % | -1.6 % / -2.7 % |
| gateup_m512 | 0.0 % / -0.1 % | +3.8 % / +0.8 % |
| gateup_m2048 | -0.4 % / -0.1 % | +3.0 % / -4.5 % |
| down_m16 | -0.1 % / 0.0 % | +15.9 % / +16.1 % |
| down_m128 | +0.2 % / +0.1 % | +19.8 % / +19.8 % |
| down_m512 | 0.0 % / -0.1 % | +14.5 % / +14.5 % |
| down_m2048 | 0.0 % / 0.0 % | +7.5 % / +4.5 % |
| ladder | 0.0 % / +0.1 % | +6.4 % / +3.0 % |

The noise: 0.4 % / 0.5 %; the rms noise: 0.133 % / 0.145 %. `20261025-1737_step4_probe` (series 1) and
`20261025-1738_step4_probe` (series 2) are the first runs of this probe. Redone: in series 1, the twin's largest gap
(1.2 %, at gateup_m2048) was more than twice its second largest (0.2 %); the probe was run again the same way in both
series, at the same time, and the table above takes the second runs only. A probe is redone once only: in series 2, the
run that decides has its twin's largest gap (0.5 %, at qkv_m512) more than twice its second largest (0.1 %).

Step 5 (`20261025-1819_step5_probe` / `20261025-1820_step5_probe`), time against step 5:

| shape | twin | without 5a |
|---|---:|---:|
| qkv_m16 | +0.3 % / -0.1 % | +45.5 % / +45.9 % |
| qkv_m128 | +0.4 % / 0.0 % | - |
| qkv_m512 | -0.1 % / -0.1 % | - |
| qkv_m2048 | +0.3 % / +0.2 % | - |
| o_m16 | +0.1 % / +0.1 % | +52.7 % / +53.3 % |
| o_m128 | 0.0 % / +0.1 % | - |
| o_m512 | +0.1 % / +0.2 % | - |
| o_m2048 | +0.1 % / +0.1 % | - |
| gateup_m16 | +0.2 % / +0.3 % | +42.0 % / +42.5 % |
| gateup_m128 | +0.2 % / +0.1 % | - |
| gateup_m512 | 0.0 % / -0.1 % | - |
| gateup_m2048 | -0.1 % / +0.1 % | - |
| down_m16 | -0.1 % / 0.0 % | +54.0 % / +55.8 % |
| down_m128 | 0.0 % / -0.1 % | - |
| down_m512 | -0.1 % / 0.0 % | - |
| down_m2048 | 0.0 % / -0.1 % | - |
| ladder | -0.2 % / 0.0 % | - |

The noise: 0.5 % / 0.3 %; the rms noise: 0.190 % / 0.134 %. A dash: at that shape the variant launches step 5's kernel,
with the same grid and arguments (`step5.cu`:50-51 calls `choose`, `step5.cuh`:242-245): 5a acts only at M <= 64
(`step5.cuh`:244); these 13 cases count in the noise. `20261025-1737_step5_probe` (series 1) and
`20261025-1738_step5_probe` (series 2) are the first runs of this probe. Redone in both series, at the same time, and
not for a gap: these two runs could not be shown to have run at the same time (`two_cards_first.txt`: no sure moment for
the run of series 2, no proof for step 5); the table above takes the second runs only, which `two_cards.txt` shows
running together.

Step 6 (`20261025-1737_step6_probe` / `20261025-1738_step6_probe`), time against step 6:

| shape | twin | without 6a | without 6b | without 6c | without 6b's hint | 6b alone against without 6b | without 6d | without 6e | without 6f |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv_m16 | +0.1 % / +0.1 % | - | +3.9 % / +6.1 % | - | - | -3.5 % / -5.6 % | - | - | - |
| qkv_m128 | +0.9 % / 0.0 % | +2.2 % / +3.2 % | - | - | - | - | - | +0.8 % / +0.6 % | +0.3 % / +0.6 % |
| qkv_m512 | +0.2 % / -0.3 % | +1.9 % / +2.8 % | - | - | - | - | +1.2 % / +0.4 % | - | +0.2 % / -0.3 % |
| qkv_m2048 | -0.2 % / -0.1 % | +2.6 % / +2.5 % | - | - | - | - | +0.5 % / +0.8 % | - | +0.6 % / +0.9 % |
| o_m16 | +0.1 % / +0.1 % | - | +15.6 % / +19.4 % | - | +2.7 % / +2.7 % | -11.1 % / -14.0 % | - | - | - |
| o_m128 | -0.1 % / +0.2 % | +1.3 % / +2.2 % | - | +5.3 % / +5.5 % | - | - | - | +0.3 % / +1.0 % | +0.3 % / +1.1 % |
| o_m512 | -0.2 % / -0.2 % | +1.9 % / +2.3 % | - | - | - | - | +0.4 % / +0.2 % | - | +0.1 % / +0.1 % |
| o_m2048 | +0.2 % / +0.4 % | +1.6 % / +2.8 % | - | - | - | - | +0.8 % / +0.2 % | - | +0.2 % / +0.3 % |
| gateup_m16 | 0.0 % / 0.0 % | - | +7.1 % / +9.2 % | - | - | -6.6 % / -8.4 % | - | - | - |
| gateup_m128 | +0.1 % / 0.0 % | +0.6 % / 0.0 % | - | - | - | - | - | +0.6 % / 0.0 % | +0.2 % / +0.5 % |
| gateup_m512 | -0.2 % / +0.2 % | +2.6 % / +1.8 % | - | - | - | - | +0.6 % / +0.3 % | - | -0.2 % / +0.4 % |
| gateup_m2048 | -0.1 % / 0.0 % | +1.7 % / +2.3 % | - | - | - | - | +0.1 % / +0.4 % | - | -0.1 % / +0.4 % |
| down_m16 | 0.0 % / 0.0 % | - | +20.2 % / +22.4 % | - | +1.1 % / +1.1 % | -15.9 % / -17.3 % | - | - | - |
| down_m128 | +0.1 % / +0.2 % | +2.4 % / +2.4 % | - | +5.9 % / +5.3 % | - | - | - | +0.8 % / +1.2 % | +1.5 % / +1.4 % |
| down_m512 | 0.0 % / -0.1 % | +1.2 % / +1.5 % | - | - | - | - | +0.3 % / +0.2 % | - | +0.1 % / -0.1 % |
| down_m2048 | +0.2 % / 0.0 % | +1.0 % / +1.8 % | - | - | - | - | +0.2 % / +0.5 % | - | +0.5 % / +0.4 % |
| ladder | -0.1 % / +0.2 % | +2.7 % / +2.4 % | - | - | - | - | +0.4 % / +0.2 % | - | 0.0 % / +0.6 % |

The noise: 1.0 % / 0.6 %; the rms noise: 0.199 % / 0.151 %. A dash: at that shape the variant launches step 6's kernel,
with the same grid and arguments (`candidate_gemm` in `step6.cu`:246-279, `plan` in `step6.cuh`:794-810): at M = 16, 6b
comes first (`step6.cuh`:796-801) and its kernel takes no 6a parameter (`step6.cu`:271-273), nor any of 6e and 6f
(`step6.cu`:259-264); 6b and its hint act only at M <= 16 (`step6.cuh`:796); 6c only where twice the tiles of 128 x 256
are fewer than the SMs (:803); 6d only on the 128 x 256 tile at M >= 512 (:808), where 6e does not act
(`step6.cu`:216-217); where a band has one slice (qkv_m16 and gateup_m16, `bands` in `step6.cuh`:776-789), step 6
launches the band kernel without the hint (`step6.cu`:143-145). In the column of 6b alone, 6b without its hint against
step 6 without 6b: at M >= 128, both launch step 6's kernel. Where 6b or 6c acts, the arm without it takes the best tile
without it: 128 columns where 256 would give fewer blocks than half the SMs (`plan` in `step6.cuh`:804-807), 64 x 128 at
o_m16 and down_m16, 128 x 128 at o_m128 and down_m128.

## ncu in the probes of step 6

At one shape for each mechanism of step 6, where it acts (`20261025-1737_step6_<mechanism>_ncu_probe`: 6a at
gateup_m512, 6b at gateup_m16, 6b's hint at o_m16, 6c at down_m128, 6d at qkv_m512, 6e at qkv_m128, 6f at down_m2048),
each arm in its own process, called twice under ncu: the counters are those of its second call (ID 1 of its CSV). `x:`
is the configuration that the judge's final ranked first at that shape for step 6 in the arrival
(`judge/20261025-172649_step6/output.txt`, its REFERENCE line of rank 0). These counters are a snapshot:
under ncu, each call is replayed in 33 passes (`probe.txt`), the clocks not locked, so they show what an arm runs (its
kernel, its grid, its traffic), not its time, and judge nothing; the timed probes, in two series, judge each mechanism
(the tables above). At qkv_m512 and down_m2048, the arm without 6d or 6f has fewer cycles than step 6, though the timed
probe finds it slower there.

| shape, arm | kernel, blocks | cycles | tensor active | DRAM read | DRAM write |
|---|---|---:|---:|---:|---:|
| gateup_m512, step 6 | `gemm_waves<128, 1>`, 40 | 3 726 302 | 85.9 % | 350.5 MB | 35.71 MB |
| gateup_m512, without 6a | `gemm_waves<128, 0>`, 40 | 3 888 448 | 82.3 % | 348.8 MB | 35.09 MB |
| gateup_m512, `x:` | turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn, 448 | 3 477 449 | 90.9 % | 406.0 MB | 35.50 MB |
| gateup_m16, step 6 | `gemm_bands<256, 0, 0>`, 112 | 1 411 398 | 6.7 % | 274.8 MB | 1.28 MB |
| gateup_m16, without 6b | `gemm<64, 256, 1>`, 112 | 1 416 499 | 28.8 % | 273.8 MB | 1.35 MB |
| gateup_m16, `x:` | turing_fp16_s1688gemm_fp16_256x64_ldg8_f2f_stages_32x1_nn, 112 | 1 418 388 | 28.1 % | 272.9 MB | 1.15 MB |
| o_m16, step 6 | `gemm_bands<256, 1, 1>`, 48 | 200 535 | 8.5 % | 38.6 MB | 0.13 MB |
| o_m16, without 6b's hint | `gemm_bands<256, 1, 0>`, 48 | 200 896 | 8.8 % | 38.7 MB | 0.76 MB |
| o_m16, `x:` | turing_fp16_s1688gemm_fp16_128x64_sliced1x2_ldg8_f2f_nn, 32 | 206 323 | 48.7 % | 43.2 MB | 0.48 MB |
| down_m128, step 6 | `gemm_split<128, 1>`, 32 | 723 808 | 66.5 % | 141.1 MB | 1.75 MB |
| down_m128, without 6c | `gemm<128, 128, 1>`, 32 | 754 890 | 63.8 % | 149.0 MB | 0.72 MB |
| down_m128, `x:` | turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn, 32 | 732 291 | 66.6 % | 139.9 MB | 0.40 MB |
| qkv_m512, step 6 | `gemm_waves<128, 1>`, 40 | 907 576 | 87.0 % | 77.6 MB | 7.49 MB |
| qkv_m512, without 6d | `gemm<128, 256, 1>`, 96 | 906 923 | 87.9 % | 77.8 MB | 6.73 MB |
| qkv_m512, `x:` | turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn, 96 | 876 540 | 90.7 % | 82.3 MB | 6.57 MB |
| qkv_m128, step 6 | `gemm<128, 256, 1>`, 24 | 316 606 | 86.0 % | 58.9 MB | 0.48 MB |
| qkv_m128, without 6e | `gemm_persistent<128, 256, 1>`, 40 | 317 649 | 90.0 % | 59.7 MB | 0.70 MB |
| qkv_m128, `x:` | turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn, 24 | 323 349 | 84.2 % | 65.6 MB | 0.42 MB |
| down_m2048, step 6 | `gemm_waves<128, 1>`, 40 | 7 610 303 | 86.3 % | 629.2 MB | 20.10 MB |
| down_m2048, without 6f | `gemm_waves_rows<128, 1>`, 40 | 7 554 375 | 87.1 % | 628.3 MB | 19.99 MB |
| down_m2048, `x:` | turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn, 256 | 7 092 809 | 91.2 % | 993.6 MB | 20.01 MB |

## The counters

The names, in the CSV files, of the counters cited here and in the rung's README: cycles `sm__cycles_elapsed.max`;
tensor active `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`; DRAM read and write
`dram__bytes_read.sum`, `dram__bytes_write.sum`; L2 sector hits `lts__t_sector_hit_rate.pct`; LDSM instructions
`smsp__inst_executed_op_ldsm.sum`; excess shared-memory wavefronts `derived__memory_l1_wavefronts_shared_excessive`;
instructions `smsp__inst_executed.sum`; blocks `launch__grid_size`. Each value is that of the kernel's row (its Kernel
Name and ID), in % and MB to one decimal (DRAM write to two). In the 8 sessions, the cuBLASLt configuration analyzed at
ladder runs turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn, 160 blocks.
