# Results of the Ampere rung

Raw outputs of the GPU sessions on Modal's A100, one folder per session, named by its date and time, and of the host
emulator (`emu/`). The tools write their outputs in English. Paths start from this folder; those of the tools
(`judge/judge.cu`, `probe/probe.cu`, `modal/session.py`, `ampere/emu`) from the repository's root.

## The outputs

- `judge/<date>-<time>_<candidate>/`: the judge's output (`judge/judge.cu`), `output.txt`. Its lines 1 and 2 give the
  md5 of each judged source, of the judge, of its launcher, of the shapes file and of the reference file: those of this
  repository's files, which `md5sum` checks. Container 1: `judge/20261017-190559_*`; container 2, the arrival:
  `judge/20261017-190653_*`; 9 candidates each (`ref_simple`, the reference point, and `step0` to `step7`).
- `<date>-<time>_<steps>/`: a session of `modal/session.py`: the run (`00_run.txt`: the md5 of the launcher and of each
  source, the card line, the cuBLASLt configuration analyzed and where it comes from), the machine (`00_machine.txt`),
  `verif.cu` under compute-sanitizer (`*_01_sanitizer.txt`), ncu at ladder (`*_02_ncu_run.txt`, and its counters in
  `*_02_ncu_raw.csv`). Here, one session per candidate, 9 folders: `20261017-191330_ref_simple`,
  `20261017-191331_step0`, `20261017-191426_step1`, `20261017-191338_step2`, `20261017-191340_step3`,
  `20261017-191341_step4`, `20261017-191342_step5`, `20261017-191400_step6` and `20261017-191438_step7`.
- `<date>-<time>_<step>_probe/`: an interleaved probe (`probe/probe.cu`): `probe.txt` (its lines 1 and 2: the md5 of
  each source, of the probe and of its launcher, the card line). Here, two series of the seven probes, each probe in a
  container of its own: series 1, `20261017-1917_step1_probe` to `20261017-1917_step7_probe`; series 2,
  `20261017-1918_step1_probe` to `20261017-1918_step7_probe`. Steps 1 and 5, then steps 2 and 7, were run again in both
  series (below): `20261017-1952_step1_probe`, `20261017-1952_step5_probe`, `20261017-2013_step2_probe` and
  `20261017-2013_step7_probe` (series 1), `20261017-1953_step1_probe`, `20261017-1953_step5_probe`,
  `20261017-2014_step2_probe` and `20261017-2014_step7_probe` (series 2).
- `<date>-<time>_<step>_<mechanism>_ncu_probe/`: the probe under ncu only, at a shape where that mechanism acts:
  `probe.txt`, the counters of each arm in `ncu_<arm>_<shape>_raw.csv`, and the version of Nsight Compute that exported
  them (`ncu_export_version.txt`). Here, `20261017-1917_step7_7a_ncu_probe` to `20261017-1917_step7_7d_ncu_probe`.
- `two_cards.txt`: for each step, the two probes that decide (one per series), the start and the end of each one's app
  on Modal (in UTC), its ARM lines and rounds, hence a lower bound of its work on the card (each round, a third of a
  block of warm-up, then the timed block: `probe/probe.cu`) and the time it surely ran; then the time the two surely ran
  together. A container keeps its card for its whole life, so two containers running at the same time have two different
  cards: the file shows it at the 7 steps, and says nothing about the hosts.
- `two_cards_first.txt`: the same, with the first runs of steps 2 and 7 (the five other steps as in `two_cards.txt`): no
  sure moment for their runs of series 1 (their apps on Modal lasted more than twice their work), hence no proof for
  steps 2 and 7.
- `emu/`: the host emulator (`ampere/emu`, no GPU) on these sources: `emu_all.txt` (`./emu all`) and `planted_bugs.txt`
  (`python3 ampere/emu/planted_bugs.py -j 2`).

The ncu counters are in the `*_raw.csv` files, exported with `ncu --import <report> --csv --page raw`. Paths in the
candidate lines are relative to the repository's root.

## The card, in the judge's outputs

In the judge's 18 outputs, at their 17 shapes, each arm's mean power is 200.1 to 402.1 W, its mean SM clock 1029 to
1410 MHz and its NVML throttle reasons 0x0 or 0x4 (columns `w_ref`, `w_cand`, `mhz_ref`, `mhz_cand`, `throttle_ref`,
`throttle_cand`), and no `WARNING` line is written. In NVIDIA's `nvml.h`, 0x4 is `nvmlClocksEventReasonSwPowerCap`: "The
clocks have been optimized to ensure not to exceed currently set power limits"; it is not one of the bits that the judge
counts as a thermal or hardware throttle (`judge/judge.cu`:482).

## The probes

A case is t(arm) / t(v0) - 1, in %, from the two medians printed in the same `probe.txt` at the same shape, rounded to
one decimal (halves away from zero); a case "alone against" an arm is t(that variant) / t(that arm) - 1. `twin` is the
step timed a second time. A dash: at that shape the arm launches the kernel of the arm it is compared with, with the
same grid and arguments (the note under the table gives the lines of the code); the probe times it all the same, and its
median stays in `probe.txt`. The noise of a probe is the largest, in absolute value, of its `twin` cases, of its cases
shown as a dash against the step and of the cases of a variant named under its table (at least 0.1 %). When the largest
of these gaps is more than twice the second largest (counted at least 0.1 %), the probe is run again once, the same way,
in both series at the same time: the second runs decide, and the first ones stay here. Each step has two probes that
decide, one per series, side by side in its table: each cell gives series 1 / series 2. The mechanisms' means, from
these same medians, are in the rung's README (`../README.md`).

Step 1 (`20261017-1952_step1_probe` / `20261017-1953_step1_probe`), time against step 1:

| shape | twin | without 1a | without 1c | 1d alone against step 0 | step 0 |
|---|---:|---:|---:|---:|---:|
| qkv_m16 | +0.1 % / 0.0 % | +2.2 % / +2.3 % | -23.0 % / -22.8 % | -11.8 % / -11.8 % | 0.0 % / +0.3 % |
| qkv_m128 | 0.0 % / 0.0 % | +3.3 % / +3.6 % | -27.3 % / -26.9 % | -20.8 % / -21.0 % | -2.1 % / -1.5 % |
| qkv_m512 | 0.0 % / 0.0 % | +3.4 % / +3.4 % | +9.7 % / +9.3 % | -22.4 % / -22.2 % | +46.8 % / +47.2 % |
| qkv_m2048 | 0.0 % / 0.0 % | +3.5 % / +3.7 % | +42.7 % / +45.8 % | -22.5 % / -22.6 % | +92.0 % / +97.2 % |
| o_m16 | 0.0 % / 0.0 % | +2.9 % / +3.0 % | -24.0 % / -23.9 % | -12.4 % / -12.5 % | -3.7 % / -3.4 % |
| o_m128 | 0.0 % / 0.0 % | +3.6 % / +3.7 % | -27.0 % / -26.9 % | -20.9 % / -21.0 % | -5.8 % / -5.2 % |
| o_m512 | 0.0 % / 0.0 % | +3.6 % / +3.7 % | +8.8 % / +9.0 % | -21.0 % / -21.1 % | +41.0 % / +41.2 % |
| o_m2048 | 0.0 % / 0.0 % | +3.2 % / +3.6 % | +41.9 % / +44.9 % | -20.5 % / -20.4 % | +83.8 % / +87.4 % |
| gateup_m16 | 0.0 % / 0.0 % | +3.7 % / +3.7 % | +25.5 % / +25.6 % | -17.7 % / -17.9 % | +66.6 % / +67.5 % |
| gateup_m128 | 0.0 % / 0.0 % | +4.1 % / +4.0 % | +20.9 % / +21.0 % | -23.1 % / -23.5 % | +62.8 % / +63.6 % |
| gateup_m512 | -0.1 % / -0.1 % | +3.3 % / +3.4 % | +31.9 % / +35.1 % | -23.0 % / -23.2 % | +76.3 % / +82.7 % |
| gateup_m2048 | 0.0 % / 0.0 % | +3.4 % / +3.6 % | +36.2 % / +38.8 % | -23.1 % / -23.6 % | +82.7 % / +88.7 % |
| down_m16 | 0.0 % / 0.0 % | +2.9 % / +3.0 % | -24.0 % / -24.0 % | -11.7 % / -11.9 % | -3.5 % / -3.1 % |
| down_m128 | 0.0 % / 0.0 % | +4.4 % / +4.5 % | -27.2 % / -27.1 % | -21.5 % / -21.5 % | -4.2 % / -3.8 % |
| down_m512 | 0.0 % / 0.0 % | +4.1 % / +4.1 % | +8.6 % / +8.7 % | -21.1 % / -21.5 % | +42.8 % / +43.5 % |
| down_m2048 | 0.0 % / 0.0 % | +3.5 % / +3.8 % | +42.2 % / +44.5 % | -20.7 % / -20.9 % | +87.1 % / +90.3 % |
| ladder | +0.1 % / 0.0 % | +4.3 % / +4.5 % | +39.9 % / +41.6 % | -23.7 % / -24.1 % | +81.6 % / +83.0 % |

The noise: 0.1 % / 0.1 %. `20261017-1917_step1_probe` (series 1) and `20261017-1918_step1_probe` (series 2) are the
first runs of this probe. Redone: in series 2, the twin's largest gap (0.3 %, at down_m2048) was more than twice its
second largest (0.1 %); the probe was run again the same way in both series, at the same time, and the table above takes
the second runs only.

Step 2 (`20261017-2013_step2_probe` / `20261017-2014_step2_probe`), time against step 2:

| shape | twin | without 2b | step 1 | 2a alone against step 1 |
|---|---:|---:|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | +9.4 % / +9.4 % | +2.2 % / +2.2 % | +7.1 % / +7.0 % |
| qkv_m128 | 0.0 % / 0.0 % | +8.9 % / +8.8 % | +1.0 % / +1.0 % | +7.8 % / +7.7 % |
| qkv_m512 | 0.0 % / 0.0 % | +8.8 % / +8.8 % | +1.2 % / +1.4 % | +7.6 % / +7.4 % |
| qkv_m2048 | 0.0 % / 0.0 % | +7.4 % / +8.9 % | +1.7 % / +1.9 % | +5.6 % / +6.9 % |
| o_m16 | 0.0 % / 0.0 % | +9.3 % / +9.2 % | +1.7 % / +1.6 % | +7.5 % / +7.6 % |
| o_m128 | 0.0 % / 0.0 % | +8.9 % / +8.8 % | +0.7 % / +0.6 % | +8.1 % / +8.2 % |
| o_m512 | 0.0 % / 0.0 % | +8.8 % / +8.8 % | +1.4 % / +1.4 % | +7.3 % / +7.4 % |
| o_m2048 | 0.0 % / 0.0 % | +8.9 % / +8.8 % | +1.7 % / +1.6 % | +7.1 % / +7.0 % |
| gateup_m16 | 0.0 % / 0.0 % | +9.0 % / +9.0 % | +1.9 % / +1.9 % | +7.0 % / +7.0 % |
| gateup_m128 | 0.0 % / 0.0 % | +8.6 % / +8.5 % | +0.5 % / +0.5 % | +8.1 % / +8.0 % |
| gateup_m512 | 0.0 % / 0.0 % | +7.1 % / +7.2 % | -0.1 % / -0.1 % | +7.2 % / +7.3 % |
| gateup_m2048 | 0.0 % / -0.1 % | +6.7 % / +6.6 % | 0.0 % / -0.1 % | +6.6 % / +6.7 % |
| down_m16 | 0.0 % / 0.0 % | +9.4 % / +9.4 % | +1.9 % / +1.8 % | +7.4 % / +7.4 % |
| down_m128 | 0.0 % / 0.0 % | +9.3 % / +9.3 % | +0.4 % / +0.5 % | +8.9 % / +8.8 % |
| down_m512 | 0.0 % / 0.0 % | +9.1 % / +9.2 % | +0.9 % / +1.1 % | +8.1 % / +8.0 % |
| down_m2048 | 0.0 % / 0.0 % | +8.8 % / +9.1 % | +1.4 % / +1.5 % | +7.4 % / +7.5 % |
| ladder | 0.0 % / 0.0 % | +8.2 % / +8.7 % | +2.4 % / +2.4 % | +5.7 % / +6.1 % |

The noise: 0.1 % / 0.1 %. `20261017-1917_step2_probe` (series 1) and `20261017-1918_step2_probe` (series 2) are the
first runs of this probe. Redone in both series, at the same time, and not for a gap: these two runs could not be shown
to have run at the same time (`two_cards_first.txt`: no sure moment for the run of series 1, no proof for step 2); the
table above takes the second runs only, which `two_cards.txt` shows running together.

Step 3 (`20261017-1917_step3_probe` / `20261017-1918_step3_probe`), time against step 3:

| shape | twin | without 3b (one stage) | step 2 | 3a alone against step 2 |
|---|---:|---:|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | +42.8 % / +43.2 % | +40.8 % / +41.1 % | +1.4 % / +1.5 % |
| qkv_m128 | 0.0 % / 0.0 % | +35.0 % / +35.2 % | +56.7 % / +56.9 % | -13.8 % / -13.8 % |
| qkv_m512 | 0.0 % / 0.0 % | +34.7 % / +33.5 % | +52.6 % / +51.1 % | -11.7 % / -11.7 % |
| qkv_m2048 | -0.1 % / 0.0 % | +26.7 % / +26.6 % | +41.6 % / +41.9 % | -10.5 % / -10.8 % |
| o_m16 | 0.0 % / 0.0 % | +40.1 % / +39.9 % | +47.0 % / +47.6 % | -4.7 % / -5.2 % |
| o_m128 | 0.0 % / 0.0 % | +34.4 % / +34.9 % | +56.5 % / +57.2 % | -14.1 % / -14.2 % |
| o_m512 | 0.0 % / +0.1 % | +36.1 % / +36.5 % | +55.1 % / +55.7 % | -12.2 % / -12.3 % |
| o_m2048 | 0.0 % / 0.0 % | +28.6 % / +28.2 % | +44.8 % / +42.9 % | -11.2 % / -10.3 % |
| gateup_m16 | 0.0 % / -0.1 % | +37.7 % / +37.7 % | +37.6 % / +37.1 % | +0.1 % / +0.4 % |
| gateup_m128 | 0.0 % / +0.1 % | +33.2 % / +31.9 % | +49.4 % / +48.2 % | -10.8 % / -11.0 % |
| gateup_m512 | 0.0 % / -0.1 % | +30.0 % / +30.2 % | +43.1 % / +43.3 % | -9.2 % / -9.1 % |
| gateup_m2048 | -0.1 % / 0.0 % | +29.0 % / +29.3 % | +41.7 % / +42.0 % | -9.0 % / -8.9 % |
| down_m16 | 0.0 % / 0.0 % | +38.1 % / +38.4 % | +48.8 % / +49.4 % | -7.2 % / -7.4 % |
| down_m128 | 0.0 % / 0.0 % | +35.0 % / +35.2 % | +56.1 % / +56.8 % | -13.5 % / -13.8 % |
| down_m512 | 0.0 % / 0.0 % | +37.7 % / +39.4 % | +54.5 % / +56.3 % | -10.9 % / -10.8 % |
| down_m2048 | 0.0 % / 0.0 % | +30.9 % / +30.7 % | +44.7 % / +43.7 % | -9.5 % / -9.0 % |
| ladder | +0.1 % / 0.0 % | +30.4 % / +30.4 % | +39.1 % / +38.9 % | -6.3 % / -6.1 % |

The noise: 0.1 % / 0.1 %.

Step 4 (`20261017-1917_step4_probe` / `20261017-1918_step4_probe`), time against step 4:

| shape | twin | without 4a | without 4b |
|---|---:|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | +0.1 % / 0.0 % | +0.1 % / 0.0 % |
| qkv_m128 | 0.0 % / 0.0 % | +1.9 % / +1.9 % | 0.0 % / 0.0 % |
| qkv_m512 | -0.1 % / 0.0 % | +1.3 % / +1.4 % | +1.6 % / +1.6 % |
| qkv_m2048 | 0.0 % / 0.0 % | +1.2 % / +1.1 % | +3.4 % / +2.9 % |
| o_m16 | 0.0 % / 0.0 % | +0.2 % / +0.3 % | +0.1 % / +0.1 % |
| o_m128 | 0.0 % / 0.0 % | +2.1 % / +2.1 % | -0.1 % / 0.0 % |
| o_m512 | 0.0 % / 0.0 % | +1.7 % / +1.8 % | 0.0 % / 0.0 % |
| o_m2048 | -0.1 % / 0.0 % | +1.2 % / +1.2 % | +2.1 % / +2.2 % |
| gateup_m16 | 0.0 % / 0.0 % | 0.0 % / 0.0 % | -0.4 % / 0.0 % |
| gateup_m128 | -0.1 % / 0.0 % | +1.0 % / +1.4 % | -0.1 % / +0.1 % |
| gateup_m512 | -0.1 % / +0.1 % | +1.2 % / +1.4 % | +8.6 % / +8.6 % |
| gateup_m2048 | 0.0 % / 0.0 % | +1.2 % / +1.3 % | +11.2 % / +10.8 % |
| down_m16 | 0.0 % / 0.0 % | +0.1 % / +0.1 % | +0.1 % / 0.0 % |
| down_m128 | +0.1 % / 0.0 % | +2.9 % / +2.9 % | 0.0 % / -0.1 % |
| down_m512 | 0.0 % / 0.0 % | +2.2 % / +2.2 % | -0.1 % / -0.1 % |
| down_m2048 | 0.0 % / -0.1 % | +1.6 % / +1.4 % | +2.1 % / +1.9 % |
| ladder | -0.1 % / 0.0 % | +1.5 % / +1.6 % | +0.5 % / +0.6 % |

The noise: 0.1 % / 0.1 %.

Step 5 (`20261017-1952_step5_probe` / `20261017-1953_step5_probe`), time against step 5:

| shape | twin | without 5a | without 5b |
|---|---:|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | +25.9 % / +25.7 % | +0.5 % / +0.4 % |
| qkv_m128 | 0.0 % / 0.0 % | +28.8 % / +28.8 % | +0.8 % / +0.8 % |
| qkv_m512 | -0.1 % / 0.0 % | +20.9 % / +20.2 % | +0.4 % / +0.7 % |
| qkv_m2048 | -0.1 % / -0.1 % | +18.8 % / +17.9 % | +0.6 % / +0.8 % |
| o_m16 | 0.0 % / 0.0 % | +24.9 % / +24.7 % | +0.6 % / +0.4 % |
| o_m128 | 0.0 % / 0.0 % | +28.9 % / +28.8 % | +0.8 % / +0.8 % |
| o_m512 | 0.0 % / 0.0 % | +25.8 % / +25.7 % | +0.8 % / +1.1 % |
| o_m2048 | 0.0 % / 0.0 % | +18.9 % / +18.7 % | +0.6 % / +0.8 % |
| gateup_m16 | -0.1 % / 0.0 % | +10.8 % / +10.0 % | 0.0 % / +0.1 % |
| gateup_m128 | 0.0 % / 0.0 % | +9.2 % / +9.6 % | +0.5 % / +0.5 % |
| gateup_m512 | 0.0 % / -0.1 % | +18.7 % / +18.2 % | +0.7 % / +0.7 % |
| gateup_m2048 | -0.1 % / -0.1 % | +18.9 % / +18.3 % | +0.6 % / +0.8 % |
| down_m16 | 0.0 % / 0.0 % | +30.9 % / +30.8 % | +0.4 % / +0.4 % |
| down_m128 | 0.0 % / 0.0 % | +29.9 % / +29.9 % | +0.8 % / +0.8 % |
| down_m512 | 0.0 % / 0.0 % | +26.1 % / +25.7 % | +0.8 % / +0.5 % |
| down_m2048 | 0.0 % / -0.1 % | +19.5 % / +18.9 % | +0.7 % / +0.6 % |
| ladder | 0.0 % / 0.0 % | +16.7 % / +15.8 % | +0.6 % / +0.6 % |

The noise: 0.1 % / 0.1 %. `20261017-1917_step5_probe` (series 1) and `20261017-1918_step5_probe` (series 2) are the
first runs of this probe. Redone: in series 1, the twin's largest gap (0.3 %, at qkv_m16) was more than twice its second
largest (0.1 %); the probe was run again the same way in both series, at the same time, and the table above takes the
second runs only.

Step 6 (`20261017-1917_step6_probe` / `20261017-1918_step6_probe`), time against step 6:

| shape | twin | without 6a | without 6b | without 6c |
|---|---:|---:|---:|---:|
| qkv_m16 | 0.0 % / 0.0 % | - | +235.1 % / +193.0 % | - |
| qkv_m128 | +0.1 % / 0.0 % | +56.2 % / +57.8 % | - | - |
| qkv_m512 | -0.1 % / 0.0 % | +12.2 % / +11.7 % | - | - |
| qkv_m2048 | 0.0 % / +0.1 % | - | - | - |
| o_m16 | 0.0 % / 0.0 % | - | +175.7 % / +179.7 % | - |
| o_m128 | 0.0 % / 0.0 % | +69.3 % / +70.6 % | - | - |
| o_m512 | 0.0 % / -0.1 % | +16.1 % / +16.1 % | - | - |
| o_m2048 | +0.1 % / 0.0 % | - | - | - |
| gateup_m16 | 0.0 % / 0.0 % | - | +100.8 % / +96.7 % | - |
| gateup_m128 | 0.0 % / 0.0 % | - | - | +4.6 % / +4.0 % |
| gateup_m512 | 0.0 % / 0.0 % | - | - | +0.5 % / -0.4 % |
| gateup_m2048 | 0.0 % / 0.0 % | - | - | - |
| down_m16 | 0.0 % / 0.0 % | - | +118.9 % / +115.6 % | - |
| down_m128 | 0.0 % / +0.1 % | +69.5 % / +70.8 % | - | - |
| down_m512 | +0.1 % / 0.0 % | +13.7 % / +14.3 % | - | - |
| down_m2048 | +0.1 % / -0.1 % | - | - | - |
| ladder | +0.1 % / 0.0 % | - | - | - |

The noise: 0.1 % / 0.1 %. A dash: at that shape the variant launches step 6's kernels, with the same grids and arguments
(`step6.cu`:163-179): 6b comes first, at M <= 16 (`step6.cu`:172, `m16` in `step6.cuh`:736-742), so there v1 and v3 run
6b's kernels; 6c then takes gateup_m128 and gateup_m512 (`step6.cu`:175, `phase` in `step6.cuh`:839-847), where v1 and
v2 run it too; elsewhere, the plan keeps step 5's 128 x 256 where it gives at least three waves of tiles (`choose`,
`step6.cuh`:440-445, and `step6.cu`:57-62): at qkv_m2048, o_m2048, gateup_m2048, down_m2048 and the ladder, v1 runs the
same kernel. v2 (without 6b) differs only at M <= 16, v3 (without 6c) only where `phase` holds.

Step 7 (`20261017-2013_step7_probe` / `20261017-2014_step7_probe`), time against step 7:

| shape | twin | without 7a | without 7b | without 7c | without 7d | 7a alone against without 7a |
|---|---:|---:|---:|---:|---:|---:|
| qkv_m16 | -0.3 % / -0.3 % | - | - | - | +0.4 % / +0.4 % | - |
| qkv_m128 | -0.1 % / 0.0 % | - | - | - | - | - |
| qkv_m512 | +0.2 % / +0.1 % | - | - | - | - | - |
| qkv_m2048 | 0.0 % / 0.0 % | - | - | - | - | - |
| o_m16 | -0.1 % / -0.1 % | - | - | - | +0.3 % / +0.3 % | - |
| o_m128 | 0.0 % / 0.0 % | +17.4 % / +17.3 % | +13.2 % / +13.8 % | - | +5.2 % / +5.6 % | -3.6 % / -2.9 % |
| o_m512 | 0.0 % / 0.0 % | +8.7 % / +8.4 % | - | - | - | -7.9 % / -7.8 % |
| o_m2048 | -0.1 % / +0.1 % | - | - | - | - | - |
| gateup_m16 | 0.0 % / 0.0 % | - | - | - | +1.4 % / +1.4 % | - |
| gateup_m128 | +0.1 % / +0.1 % | - | - | - | +3.5 % / +2.9 % | - |
| gateup_m512 | -0.1 % / 0.0 % | - | - | - | 0.0 % / +0.7 % | - |
| gateup_m2048 | 0.0 % / 0.0 % | - | - | - | - | - |
| down_m16 | +0.1 % / 0.0 % | - | - | - | +3.0 % / +2.9 % | - |
| down_m128 | 0.0 % / 0.0 % | +28.2 % / +27.6 % | +2.5 % / +2.6 % | - | +0.1 % / +0.2 % | -20.1 % / -19.6 % |
| down_m512 | 0.0 % / 0.0 % | +16.5 % / +16.4 % | - | +4.1 % / +4.5 % | - | -14.2 % / -14.1 % |
| down_m2048 | 0.0 % / -0.1 % | - | - | - | - | - |
| ladder | 0.0 % / +0.1 % | - | - | - | - | - |

The noise: 0.6 % / 0.6 %. A dash: at that shape the variant launches step 7's kernels, with the same grids and arguments
(`step7.cu`:261-276, and `step6_run`, :117-139): 6b at M <= 16 and 6c by its rule run whichever of 7a, 7b and 7c is
switched off (:261-264, :267-271); 7a's rule splits K at o_m128 and down_m128 (3 parts, reduced by 7b) and at o_m512 and
down_m512 (2 parts in place; 7c at down_m512, in 5 parts) (`choose`, `step6.cuh`:446-461; `inplace_parts`,
`step7.cuh`:110-125); 7d's hints are only in the kernels of 6b, 6c and 7b (`kHint`, `step6.cuh`). So v1 (without 7a)
differs from step 7 only at the four shapes where 7a splits K, v2 (without 7b) only at o_m128 and down_m128, v3 (without
7c) only at down_m512, and v4 (without 7d) only at the eight shapes where 6b, 6c or 7b run. In the last column, 7a
without 7b against step 7 without 7a, at the four shapes where 7a splits K (at down_m512, the arm without 7b runs 7c).
The noise is reached at a dash: v3 (without 7c) at o_m16, -0.6 % in both series, and at qkv_m16, -0.6 % in series 2.
`20261017-1917_step7_probe` (series 1) and `20261017-1918_step7_probe` (series 2) are the first runs of this probe.
Redone in both series, at the same time, and not for a gap: these two runs could not be shown to have run at the same
time (`two_cards_first.txt`: no sure moment for the run of series 1, no proof for step 7); the table above takes the
second runs only, which `two_cards.txt` shows running together.

## ncu in the probes of step 7

At down_m128, where 7a acts (`20261017-1917_step7_7a_ncu_probe`), at o_m128, where 7b acts
(`20261017-1917_step7_7b_ncu_probe`), at down_m512, where 7c acts (`20261017-1917_step7_7c_ncu_probe`), and at
gateup_m128, where 7d acts (`20261017-1917_step7_7d_ncu_probe`), each arm in its own process, called twice under ncu:
the counters are those of its second call (ID 1 of its CSV; IDs 2 and 3 when a call runs two kernels, each cited on its
own row). `x:` is the configuration that the judge's final ranked first at that shape for step 7 in the arrival
(`judge/20261017-190653_step7/output.txt`, its REFERENCE line of rank 0).

| shape, arm | kernel, blocks | cycles | tensor active | DRAM read | DRAM write |
|---|---|---:|---:|---:|---:|
| down_m128, step 7 | `gemm_parallel<128, 256, 64, 64, 3, 1>`, 96 | 399 433 | 78.9 % | 477.1 MB | 1.90 MB |
| down_m128, without 7a | `gemm<128, 128, 64, 32, 5>`, 64 | 495 410 | 93.9 % | 477.1 MB | 2.52 MB |
| down_m128, `x:` | `cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x6_nn_align8>`, 96 | 386 319 | 81.3 % | 477.2 MB | 2.51 MB |
| down_m128, `x:` | `cublasLt::splitKreduce_kernel<32, 16, ...>`, 512 | 19 170 | 2.3 % | 6.3 MB | 0.00 MB |
| o_m128, step 7 | `gemm_parallel<128, 256, 64, 64, 3, 1>`, 96 | 126 577 | 72.2 % | 136.4 MB | 1.97 MB |
| o_m128, without 7b | `gemm_split<128, 256, 64, 64, 3, 1>`, 96 | 114 400 | 80.6 % | 136.4 MB | 2.55 MB |
| o_m128, without 7b | `reduce_parts<128, 256>`, 32 | 42 484 | 0.0 % | 6.3 MB | 0.00 MB |
| o_m128, `x:` | ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_nn, 96 | 114 701 | 79.7 % | 136.3 MB | 2.54 MB |
| o_m128, `x:` | `cublasLt::splitKreduce_kernel<32, 16, ...>`, 512 | 19 427 | 2.3 % | 6.3 MB | 0.00 MB |
| down_m512, step 7 | `gemm_split<128, 256, 64, 64, 3, 2>`, 640 | 1 253 703 | 88.8 % | 562.1 MB | 37.94 MB |
| down_m512, without 7c | `gemm_split<128, 128, 64, 32, 5, 2>`, 512 | 1 252 114 | 93.1 % | 805.2 MB | 16.65 MB |
| down_m512, `x:` | ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_nn, 640 | 1 210 289 | 91.9 % | 562.1 MB | 37.92 MB |
| gateup_m128, step 7 | `gemm_phase<128, 256, 64, 64, 3, 1>`, 108 | 766 537 | 73.7 % | 942.1 MB | 9.98 MB |
| gateup_m128, without 7d | `gemm_phase<128, 256, 64, 64, 3, 0>`, 108 | 849 472 | 66.9 % | 964.8 MB | 18.48 MB |
| gateup_m128, `x:` | `Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_nn_align8>`, 448 | 823 106 | 76.0 % | 949.5 MB | 15.42 MB |

## The counters

The names, in the CSV files, of the counters cited here and in the rung's README: cycles `sm__cycles_elapsed.max`;
tensor active `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`; DRAM read and write
`dram__bytes_read.sum`, `dram__bytes_write.sum`; L2 sector hits `lts__t_sector_hit_rate.pct`; excess shared-memory
wavefronts `derived__memory_l1_wavefronts_shared_excessive`; instructions `smsp__inst_executed.sum`; blocks
`launch__grid_size`. Each value is that of the kernel's row (its Kernel Name and ID), in % and MB to one decimal (DRAM
write to two). In the 9 sessions, the cuBLASLt configuration analyzed at ladder runs
ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_nn, 432 blocks.
