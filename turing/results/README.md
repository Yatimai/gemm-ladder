# Results of the Turing rung

Raw outputs of the GPU sessions on Modal's T4, one folder per session, named by its date and time. The tools write their
outputs in English.

- `<date>-<time>_<steps>/`: a session of `modal/session.py`: the run (`00_run.txt`: the md5 of the launcher and of each
  source, the card line), the machine (`00_machine.txt`), `verif.cu` under compute-sanitizer (`*_01_sanitizer.txt`), ncu
  at ladder (`*_02_ncu_run.txt`, and its counters in `*_02_ncu_raw.csv`).
- `<date>-<time>_<step>_probe/`: an interleaved probe (`probe/probe_t4.cu`): `probe.txt` (its lines 1 and 2: the md5 of
  each source, of the probe and of its launcher, the card line), and the counters of its ncu calls in
  `ncu_<arm>_<shape>_raw.csv`.
- `judge/<date>-<time>_<candidate>/`: the judge's output (`judge/judge.cu`) in the common pass, `output.txt`. Its lines
  1 and 2 give the md5 of each judged source, of the judge, of its launcher, of the shapes file and of the reference
  file: those of this repository's files, which `md5sum` checks. The launcher's copies of the judged sources, written
  next to `output.txt`, are left out.

The ncu reports themselves (`.ncu-rep`) are not in the repository; their counters are in the `*_raw.csv` files, exported
with `ncu --import <report> --csv --page raw`. Paths in the candidate lines are relative to the repository's root.
