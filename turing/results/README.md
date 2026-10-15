# Results of the Turing rung

Raw outputs of the GPU sessions on Modal's T4, one folder per session, named by its date and time, and of the host
emulator (`emu/`). The tools write their outputs in English. Paths start from this folder; those of the tools
(`judge/judge.cu`, `probe/probe.cu`, `modal/session.py`, `turing/emu`) from the repository's root.

## The outputs

- `judge/<date>-<time>_<candidate>/`: the judge's output (`judge/judge.cu`), `output.txt`. Its lines 1 and 2 give the
  md5 of each judged source, of the judge, of its launcher, of the shapes file and of the reference file: those of this
  repository's files, which `md5sum` checks. Container 1, the arrival: `judge/20261015-132913_*`; container 2:
  `judge/20261015-133006_*`; 8 candidates each (`ref_simple`, the reference point, and `step0` to `step6`).
- `<date>-<time>_<steps>/`: a session of `modal/session.py`: the run (`00_run.txt`: the md5 of the launcher and of each
  source, the card line, the cuBLASLt configuration analyzed and where it comes from), the machine (`00_machine.txt`),
  `verif.cu` under compute-sanitizer (`*_01_sanitizer.txt`), ncu at ladder (`*_02_ncu_run.txt`, and its counters in
  `*_02_ncu_raw.csv`). Here, one session per candidate, 8 folders from `20261015-133133_ref_simple` to
  `20261015-133303_step6`.
- `<date>-<time>_<step>_probe/`: an interleaved probe (`probe/probe.cu`): `probe.txt` (its lines 1 and 2: the md5 of
  each source, of the probe and of its launcher, the card line). Here, two series of the six probes, each probe in a
  container of its own: series 1, `20261015-1334_step1_probe` to `20261015-1334_step6_probe`; series 2,
  `20261015-1335_step1_probe` to `20261015-1335_step6_probe`. Steps 1, 3 and 4 were run again in both series (below):
  `20261015-1355_step1_probe`, `20261015-1355_step3_probe` and `20261015-1414_step4_probe` (series 1),
  `20261015-1356_step1_probe`, `20261015-1356_step3_probe` and `20261015-1415_step4_probe` (series 2).
- `<date>-<time>_<step>_<mechanism>_ncu_probe/`: the probe under ncu only, at a shape where that mechanism acts:
  `probe.txt`, the counters of each arm in `ncu_<arm>_<shape>_raw.csv`, and the version of Nsight Compute that
  exported them (`ncu_export_version.txt`). Here, `20261015-1334_step6_6b_ncu_probe` and
  `20261015-1334_step6_6c_ncu_probe`.
- `two_cards.txt`: for each step, the two probes that decide (one per series), the start and the end of each one's app
  on Modal (in UTC), its ARM lines and rounds, hence a lower bound of its work on the card (each round, a third of a
  block of warm-up, then the timed block: `probe/probe.cu`) and the time it surely ran; then the time the two surely
  ran together. A container keeps its card for its whole life, so two containers running at the same time have two
  different cards: the file shows it at the 6 steps, and says nothing about the hosts.
- `two_cards_first.txt`: the same, with the first runs of step 4 (the five other steps as in `two_cards.txt`): no sure
  moment for the run of series 1 (its app on Modal lasted more than twice its work), hence no proof for step 4.
- `emu/`: the host emulator (`turing/emu`, no GPU) on these sources: `emu_all.txt` (`./emu all`) and `planted_bugs.txt`
  (`python3 turing/emu/planted_bugs.py -j 2`).

The ncu counters are in the `*_raw.csv` files, exported with `ncu --import <report> --csv --page raw`. Paths in the
candidate lines are relative to the repository's root.

## The card, in the judge's outputs

In the judge's 16 outputs, at their 17 shapes, each arm's mean power is 67.5 to 70.0 W, its mean SM clock 499 to
1507 MHz and its NVML throttle reasons 0x4 (columns `w_ref`, `w_cand`, `mhz_ref`, `mhz_cand`, `throttle_ref`,
`throttle_cand`), and no `WARNING` line is written. In NVIDIA's `nvml.h`, 0x4 is `nvmlClocksEventReasonSwPowerCap`:
"The clocks have been optimized to ensure not to exceed currently set power limits"; it is not one of the bits that
the judge counts as a thermal or hardware throttle (`judge/judge.cu`:482).

## The probes

A case is t(arm) / t(v0) - 1, in %, from the two medians printed in the same `probe.txt` at the same shape, rounded to
one decimal (halves away from zero); a case "alone against" an arm is t(that variant) / t(that arm) - 1. `twin` is the
step timed a second time. A dash: at that shape the arm launches the kernel of the arm it is compared with, with the
same grid and arguments (the note under the table gives the lines of the code); the probe times it all the same, and
its median stays in `probe.txt`. The noise of a probe is the largest, in absolute value, of its `twin` cases, of its
cases shown as a dash against the step and of the cases of a variant named under its table (at least 0.1 %). When the
largest of these gaps is more than twice the second largest (counted at least 0.1 %), the probe is run again once, the
same way, in both series at the same time: the second runs decide, and the first ones stay here. Each step has two
probes that decide, one per series, side by side in its table: each cell gives series 1 / series 2. The mechanisms'
means, from these same medians, are in the rung's README (`../README.md`).

Step 1 (`20261015-1355_step1_probe` / `20261015-1356_step1_probe`), time against step 1:

| shape | twin | without 1a | without 1b | 1c alone against step 0 | step 0 |
|---|---:|---:|---:|---:|---:|
| qkv_m16 | -0.2 % / -0.8 % | +7.5 % / +6.2 % | +22.9 % / +21.9 % | -3.6 % / -2.2 % | +28.4 % / +23.5 % |
| qkv_m128 | -0.1 % / +0.3 % | +5.1 % / +3.9 % | +21.8 % / +22.5 % | -3.1 % / -3.3 % | +22.8 % / +22.4 % |
| qkv_m512 | 0.0 % / +0.2 % | +4.8 % / -0.4 % | +34.6 % / +33.3 % | -3.1 % / -1.5 % | +28.8 % / +26.2 % |
| qkv_m2048 | -0.1 % / 0.0 % | -1.8 % / -2.9 % | +36.6 % / +37.9 % | -2.1 % / -1.9 % | +30.5 % / +32.0 % |
| o_m16 | 0.0 % / -0.1 % | +22.9 % / +13.8 % | +6.1 % / +5.2 % | -2.4 % / -0.7 % | +7.7 % / +5.6 % |
| o_m128 | +0.1 % / -0.2 % | +19.8 % / +16.4 % | +7.1 % / +6.2 % | -5.5 % / -5.7 % | +3.9 % / +4.6 % |
| o_m512 | +0.1 % / +0.1 % | +0.9 % / +0.5 % | +32.5 % / +31.1 % | -4.6 % / -5.0 % | +29.8 % / +31.4 % |
| o_m2048 | +0.1 % / +0.1 % | +1.5 % / -2.1 % | +38.4 % / +39.5 % | -5.5 % / -7.0 % | +33.4 % / +35.3 % |
| gateup_m16 | -0.1 % / -0.1 % | +0.3 % / -0.6 % | +19.7 % / +21.8 % | -2.1 % / +1.4 % | +21.1 % / +16.6 % |
| gateup_m128 | 0.0 % / +0.1 % | -0.8 % / -0.5 % | +22.4 % / +23.8 % | -4.0 % / -2.1 % | +17.9 % / +16.7 % |
| gateup_m512 | -0.1 % / -0.4 % | -1.2 % / -1.4 % | +22.3 % / +23.1 % | -1.6 % / -1.6 % | +15.6 % / +15.5 % |
| gateup_m2048 | -0.2 % / +0.1 % | -1.1 % / -0.2 % | +21.8 % / +23.4 % | -2.0 % / -5.1 % | +15.6 % / +19.9 % |
| down_m16 | 0.0 % / +0.1 % | +22.0 % / +11.3 % | +7.9 % / +6.6 % | -2.4 % / -3.0 % | +8.6 % / +5.8 % |
| down_m128 | 0.0 % / -0.3 % | +18.6 % / +15.2 % | +7.3 % / +7.0 % | -5.3 % / -6.7 % | +4.4 % / +5.9 % |
| down_m512 | 0.0 % / -0.2 % | -0.3 % / +0.3 % | +30.7 % / +31.3 % | -5.3 % / -5.3 % | +29.2 % / +29.9 % |
| down_m2048 | -0.1 % / -0.2 % | +0.7 % / -1.8 % | +39.7 % / +41.5 % | -11.4 % / -10.9 % | +41.5 % / +41.2 % |
| ladder | +0.1 % / 0.0 % | +6.2 % / +4.2 % | +40.4 % / +38.3 % | -7.0 % / -6.5 % | +35.1 % / +34.5 % |

The noise: 0.2 % / 0.8 %. `20261015-1334_step1_probe` (series 1) and `20261015-1335_step1_probe` (series 2) are the
first runs of this probe. Redone: in series 2, the twin's largest gap (1.5 %, at o_m16) was more than twice its second
largest (0.4 %); the probe was run again the same way in both series, at the same time, and the table above takes the
second runs only.

Step 2 (`20261015-1334_step2_probe` / `20261015-1335_step2_probe`), time against step 2:

| shape | twin | without 2b | step 1 | 2a alone against step 1 |
|---|---:|---:|---:|---:|
| qkv_m16 | -0.1 % / -1.2 % | +11.2 % / +4.5 % | +11.0 % / +4.9 % | +0.2 % / -0.4 % |
| qkv_m128 | -0.3 % / +1.3 % | +17.6 % / +16.9 % | +15.0 % / +15.1 % | +2.2 % / +1.6 % |
| qkv_m512 | 0.0 % / -0.2 % | +11.8 % / +12.4 % | +8.3 % / +11.1 % | +3.2 % / +1.2 % |
| qkv_m2048 | -0.2 % / -0.5 % | +4.0 % / +9.2 % | +4.8 % / +9.6 % | -0.8 % / -0.4 % |
| o_m16 | 0.0 % / +0.6 % | +6.1 % / +4.4 % | +0.3 % / +0.4 % | +5.9 % / +3.9 % |
| o_m128 | 0.0 % / -0.1 % | +10.3 % / +11.9 % | +0.9 % / +2.3 % | +9.3 % / +9.4 % |
| o_m512 | -0.3 % / -0.5 % | +12.4 % / +9.1 % | +13.2 % / +13.9 % | -0.7 % / -4.2 % |
| o_m2048 | 0.0 % / +0.2 % | +9.9 % / +8.4 % | +8.1 % / +10.2 % | +1.6 % / -1.6 % |
| gateup_m16 | 0.0 % / +0.1 % | +3.1 % / +4.8 % | +8.4 % / +11.1 % | -4.9 % / -5.7 % |
| gateup_m128 | -0.1 % / -0.2 % | +10.3 % / +13.8 % | +11.6 % / +14.1 % | -1.1 % / -0.3 % |
| gateup_m512 | +0.1 % / +0.5 % | +10.7 % / +15.5 % | +11.1 % / +16.1 % | -0.4 % / -0.5 % |
| gateup_m2048 | 0.0 % / 0.0 % | +11.1 % / +16.7 % | +12.9 % / +17.2 % | -1.6 % / -0.4 % |
| down_m16 | 0.0 % / +0.1 % | +6.6 % / +5.6 % | -0.3 % / +2.5 % | +6.9 % / +3.0 % |
| down_m128 | 0.0 % / +0.1 % | +9.1 % / +12.3 % | +0.1 % / +1.5 % | +9.0 % / +10.6 % |
| down_m512 | +0.1 % / -0.4 % | +11.7 % / +10.0 % | +11.0 % / +10.9 % | +0.6 % / -0.9 % |
| down_m2048 | +0.1 % / -0.1 % | -4.2 % / -1.9 % | -5.2 % / +0.3 % | +1.1 % / -2.3 % |
| ladder | +0.1 % / +0.2 % | +10.6 % / +7.5 % | +8.2 % / +7.7 % | +2.1 % / -0.1 % |

The noise: 0.3 % / 1.3 %.

Step 3 (`20261015-1355_step3_probe` / `20261015-1356_step3_probe`), time against step 3:

| shape | twin | without 3a | without 3b | step 2 |
|---|---:|---:|---:|---:|
| qkv_m16 | +0.6 % / +0.3 % | -3.7 % / -3.6 % | -1.3 % / -1.3 % | -3.1 % / -3.8 % |
| qkv_m128 | -0.5 % / -0.1 % | -3.4 % / -2.6 % | +0.9 % / -0.2 % | -4.4 % / -3.1 % |
| qkv_m512 | +0.1 % / 0.0 % | -0.4 % / -0.3 % | +7.8 % / +5.0 % | +8.5 % / +4.6 % |
| qkv_m2048 | +0.2 % / -0.1 % | +9.7 % / +6.3 % | +15.4 % / +9.1 % | +17.6 % / +16.2 % |
| o_m16 | 0.0 % / +0.1 % | -6.0 % / -6.4 % | -0.5 % / -1.0 % | -6.1 % / -6.6 % |
| o_m128 | 0.0 % / -0.2 % | -3.5 % / -4.1 % | +0.4 % / +0.4 % | -3.4 % / -3.9 % |
| o_m512 | -0.2 % / 0.0 % | -0.3 % / -0.9 % | +3.7 % / +2.5 % | +3.0 % / +1.3 % |
| o_m2048 | +0.2 % / -0.1 % | +9.6 % / +3.5 % | +6.2 % / +1.7 % | +8.5 % / +3.3 % |
| gateup_m16 | -0.3 % / 0.0 % | +0.5 % / -1.4 % | -1.0 % / -1.3 % | +0.4 % / -1.6 % |
| gateup_m128 | +0.1 % / +0.1 % | +0.7 % / +1.2 % | -0.3 % / +0.4 % | +1.1 % / +0.4 % |
| gateup_m512 | +0.5 % / +0.2 % | -1.3 % / -3.1 % | +18.9 % / +15.0 % | +18.9 % / +15.4 % |
| gateup_m2048 | -0.5 % / -0.2 % | +20.6 % / +25.8 % | +27.3 % / +25.5 % | +26.1 % / +25.1 % |
| down_m16 | 0.0 % / -0.1 % | -6.9 % / -6.7 % | -0.7 % / -0.9 % | -7.0 % / -7.1 % |
| down_m128 | 0.0 % / +0.2 % | -2.1 % / -2.5 % | +0.5 % / +1.0 % | -1.7 % / -2.2 % |
| down_m512 | -0.1 % / +0.2 % | +0.4 % / +1.8 % | +2.2 % / +2.8 % | +8.7 % / +4.9 % |
| down_m2048 | -0.4 % / -0.3 % | +22.9 % / +22.0 % | +0.5 % / -1.7 % | +13.9 % / +13.7 % |
| ladder | 0.0 % / 0.0 % | +5.8 % / +4.5 % | +0.4 % / +0.1 % | +2.4 % / +0.7 % |

The noise: 0.6 % / 0.3 %. `20261015-1334_step3_probe` (series 1) and `20261015-1335_step3_probe` (series 2) are the
first runs of this probe. Redone: in series 2, the twin's largest gap (3.5 %, at gateup_m2048) was more than twice its
second largest (0.7 %); the probe was run again the same way in both series, at the same time, and the table above
takes the second runs only.

Step 4 (`20261015-1414_step4_probe` / `20261015-1415_step4_probe`), time against step 4:

| shape | twin | step 3 (without 4a) |
|---|---:|---:|
| qkv_m16 | 0.0 % / -0.2 % | +12.1 % / +11.8 % |
| qkv_m128 | +0.1 % / -0.1 % | +11.1 % / +10.3 % |
| qkv_m512 | 0.0 % / 0.0 % | +13.1 % / +10.2 % |
| qkv_m2048 | 0.0 % / -0.3 % | +4.0 % / +2.2 % |
| o_m16 | 0.0 % / 0.0 % | +18.0 % / +16.3 % |
| o_m128 | +0.1 % / 0.0 % | +21.6 % / +20.5 % |
| o_m512 | 0.0 % / 0.0 % | +15.4 % / +12.8 % |
| o_m2048 | +0.1 % / +0.1 % | +8.3 % / +5.4 % |
| gateup_m16 | -0.1 % / +0.4 % | +0.9 % / +0.6 % |
| gateup_m128 | 0.0 % / -0.1 % | -1.4 % / -2.0 % |
| gateup_m512 | -0.3 % / +0.1 % | +0.3 % / -0.5 % |
| gateup_m2048 | 0.0 % / +0.2 % | -4.1 % / -6.2 % |
| down_m16 | +0.1 % / -0.2 % | +17.6 % / +16.2 % |
| down_m128 | 0.0 % / 0.0 % | +21.5 % / +19.9 % |
| down_m512 | 0.0 % / 0.0 % | +15.6 % / +12.3 % |
| down_m2048 | +0.1 % / +0.1 % | +5.0 % / +3.6 % |
| ladder | +0.1 % / -0.1 % | +5.5 % / +1.3 % |

The noise: 0.3 % / 0.4 %. `20261015-1334_step4_probe` (series 1) and `20261015-1335_step4_probe` (series 2) are the
first runs of this probe. Redone in both series, at the same time, and not for a gap: these two runs could not be shown
to have run at the same time (`two_cards_first.txt`: no sure moment for the run of series 1, no proof for step 4); the
table above takes the second runs only, which `two_cards.txt` shows running together. A probe is redone once only: in
series 1, the run that decides has its twin's largest gap (0.3 %, at gateup_m512) more than twice its second largest
(0.1 %).

Step 5 (`20261015-1334_step5_probe` / `20261015-1335_step5_probe`), time against step 5:

| shape | twin | without 5a | step 4 |
|---|---:|---:|---:|
| qkv_m16 | +0.1 % / -2.0 % | +46.8 % / +37.9 % | +46.5 % / +36.8 % |
| qkv_m128 | +0.3 % / +2.1 % | - | -0.3 % / +1.6 % |
| qkv_m512 | -0.1 % / -0.4 % | - | 0.0 % / -0.5 % |
| qkv_m2048 | -0.1 % / +0.1 % | - | -0.2 % / -0.3 % |
| o_m16 | 0.0 % / 0.0 % | +56.0 % / +53.9 % | +56.2 % / +53.6 % |
| o_m128 | -0.1 % / -0.5 % | - | +0.1 % / -0.4 % |
| o_m512 | 0.0 % / -0.5 % | - | -0.1 % / -0.4 % |
| o_m2048 | -0.5 % / -0.3 % | - | -0.2 % / -0.2 % |
| gateup_m16 | -0.6 % / -0.1 % | +40.2 % / +33.9 % | +40.3 % / +34.0 % |
| gateup_m128 | +0.1 % / -0.1 % | - | -0.3 % / -0.6 % |
| gateup_m512 | +0.1 % / -0.2 % | - | 0.0 % / 0.0 % |
| gateup_m2048 | -0.1 % / -1.3 % | - | -0.2 % / -0.9 % |
| down_m16 | 0.0 % / -0.1 % | +55.5 % / +55.8 % | +55.3 % / +55.8 % |
| down_m128 | +0.1 % / 0.0 % | - | 0.0 % / +0.1 % |
| down_m512 | -0.1 % / -0.1 % | - | -0.5 % / -0.2 % |
| down_m2048 | -0.1 % / -1.2 % | - | 0.0 % / -1.0 % |
| ladder | 0.0 % / +0.1 % | - | -0.1 % / 0.0 % |

The noise: 0.6 % / 2.1 %. A dash: at that shape the variant launches step 5's kernel, with the same grid and arguments
(`step5.cu`:50-51 calls `choose`, `step5.cuh`:242-245): 5a acts only at M <= 64 (`step5.cuh`:244). The variant named
for the noise: v2 (step 4), at the 13 shapes with M >= 128, where it runs exactly the same code as step 5 under another
kernel name: `step4::gemm` (`step5.cu`:47) and step 5's `step5::gemm<128>` (`step5.cu`:51 and :20) are launched with
one block of 256 threads per SM, the same shared memory and the same arguments, and 128 x 256 "is step 4's kernel,
instruction for instruction" (`step5.cuh`:55); its 13 cases count in the noise.

Step 6 (`20261015-1334_step6_probe` / `20261015-1335_step6_probe`), time against step 6:

| shape | twin | without 6a | without 6b | without 6c | step 5 | without 6b's hint | 6b alone against without 6b |
|---|---:|---:|---:|---:|---:|---:|---:|
| qkv_m16 | +0.1 % / 0.0 % | - | +9.5 % / +5.4 % | - | +6.0 % / +2.7 % | - | -8.6 % / -5.0 % |
| qkv_m128 | -0.3 % / +0.1 % | +0.7 % / +1.1 % | - | - | +0.8 % / +1.2 % | - | - |
| qkv_m512 | 0.0 % / 0.0 % | +2.0 % / +1.4 % | - | - | +1.9 % / +1.0 % | - | - |
| qkv_m2048 | +0.1 % / 0.0 % | +2.7 % / +3.5 % | - | - | +2.6 % / +3.2 % | - | - |
| o_m16 | +0.1 % / 0.0 % | - | +27.3 % / +22.6 % | - | +22.5 % / +19.4 % | +2.6 % / +2.7 % | -19.4 % / -16.3 % |
| o_m128 | +0.3 % / +0.3 % | +1.6 % / +1.5 % | - | +11.5 % / +12.0 % | +12.2 % / +12.8 % | - | - |
| o_m512 | +0.2 % / +0.2 % | +2.2 % / +2.2 % | - | - | +2.1 % / +2.2 % | - | - |
| o_m2048 | +0.1 % / +0.1 % | +2.5 % / +2.5 % | - | - | +2.4 % / +2.5 % | - | - |
| gateup_m16 | 0.0 % / 0.0 % | - | +11.8 % / +6.0 % | - | +10.5 % / +4.3 % | - | -10.6 % / -5.7 % |
| gateup_m128 | -0.1 % / -0.2 % | 0.0 % / +0.1 % | - | - | -0.1 % / -0.6 % | - | - |
| gateup_m512 | 0.0 % / 0.0 % | +10.1 % / +13.4 % | - | - | +10.1 % / +13.2 % | - | - |
| gateup_m2048 | +0.1 % / -0.1 % | +8.0 % / +9.0 % | - | - | +8.6 % / +9.6 % | - | - |
| down_m16 | 0.0 % / 0.0 % | - | +30.1 % / +25.6 % | - | +24.7 % / +21.0 % | +1.2 % / +1.2 % | -22.2 % / -19.4 % |
| down_m128 | -0.1 % / 0.0 % | +2.5 % / +1.5 % | - | +12.5 % / +11.8 % | +12.8 % / +13.1 % | - | - |
| down_m512 | -0.3 % / -0.1 % | +2.2 % / +1.9 % | - | - | +2.1 % / +1.8 % | - | - |
| down_m2048 | 0.0 % / +0.2 % | +4.8 % / +4.5 % | - | - | +4.9 % / +4.5 % | - | - |
| ladder | +0.2 % / +0.1 % | +2.2 % / +2.3 % | - | - | +2.3 % / +2.0 % | - | - |

The noise: 0.3 % / 0.4 %. A dash: at that shape the variant launches step 6's kernel, with the same grid and arguments
(`step6.cu`:141-164, `plan` in `step6.cuh`:586-597): at M = 16, 6b comes first (`step6.cuh`:588-593) and its kernel
takes no 6a parameter (`step6.cu`:154-158); 6b and its hint act only at M <= 16 (`step6.cuh`:588); 6c only where twice
the tiles of 128 x 256 are fewer than the SMs (:595); where a band has one slice (qkv_m16 and gateup_m16, `bands` in
`step6.cuh`:568-581), step 6 launches the band kernel without the hint (`step6.cu`:114-116). In the last column, 6b
without its hint against step 6 without 6b: at M >= 128, both launch step 6's kernel.

## ncu in the probes of step 6

At down_m16, where 6b acts (`20261015-1334_step6_6b_ncu_probe`), and at o_m128, where 6c acts
(`20261015-1334_step6_6c_ncu_probe`), each arm in its own process, called twice under ncu: the counters are those of its
second call (ID 1 of its CSV). `x:` is the configuration that the judge's final ranked first at that shape for step 6
in the arrival (`judge/20261015-132913_step6/output.txt`, its REFERENCE line of rank 0).

| shape, arm | kernel, blocks | cycles | tensor active | DRAM read | DRAM write |
|---|---|---:|---:|---:|---:|
| down_m16, step 6 | `gemm_bands<256, 1, 1>`, 80 | 686 551 | 7.3 % | 132.2 MB | 0.12 MB |
| down_m16, without 6b | `gemm<64, 1>`, 40 | 698 910 | 78.3 % | 136.9 MB | 0.69 MB |
| down_m16, `x:` | turing_fp16_s1688gemm_fp16_128x64_sliced1x2_ldg8_f2f_nn, 32 | 696 492 | 45.9 % | 152.3 MB | 0.70 MB |
| o_m128, step 6 | `gemm_split<128, 1>`, 40 | 227 074 | 77.0 % | 40.6 MB | 1.81 MB |
| o_m128, without 6c | `gemm<128, 1>`, 40 | 293 677 | 92.0 % | 39.2 MB | 0.44 MB |
| o_m128, `x:` | turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn, 32 | 217 092 | 62.6 % | 46.0 MB | 0.64 MB |

## The counters

The names, in the CSV files, of the counters cited here and in the rung's README: cycles `sm__cycles_elapsed.max`;
tensor active `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`; DRAM read and write
`dram__bytes_read.sum`, `dram__bytes_write.sum`; L2 sector hits `lts__t_sector_hit_rate.pct`; LDSM instructions
`smsp__inst_executed_op_ldsm.sum`; excess shared-memory wavefronts `derived__memory_l1_wavefronts_shared_excessive`;
instructions `smsp__inst_executed.sum`; blocks `launch__grid_size`. Each value is that of the kernel's row (its Kernel
Name and ID), in % and MB to one decimal (DRAM write to two). In the 8 sessions, the cuBLASLt configuration analyzed at
ladder runs turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn, 160 blocks.
