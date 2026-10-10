# The judge

`judge.cu` times a candidate GEMM against the judge's reference on one card. A candidate is a `.cu` file that defines
`candidate_gemm` (C = A B, A (M x K) and B (K x N) fp16 row-major, fp32 accumulation, C (M x N) fp16 row-major). At each
shape of `shapes/<card>.txt`, the judge takes the ratio of the reference's time to the candidate's; the score is the
geometric mean of these ratios (above 1: faster than the reference).

The reference. Without `--reference`, the judge ranks by time cuBLASLt's configurations on the card (the heuristic's
first ones at several workspace sizes and, around the first of each family, each setting changed alone); after such a
full pass (not with `--quick`), the launcher keeps the best ones in `reference/<card>.txt` (lower case, e.g. t4.txt),
with the card line. With `--reference`, the judge ranks again those kept for its card line, with the first ones of
cuBLASLt's heuristic. A final then times the five best of the ranking and cuBLAS's default call (cublasGemmEx); the
fastest is the reference.

The timing. Each arm runs a CUDA graph of at least 3 calls, each call on one of three input sets in turn, in blocks of
1.5 s (0.5 s of warm-up), 4 pairs (reference, candidate) in random order; the ratio at a shape is the median of the
pairs. The programmatic dependencies (PDL) between two calls are neutralized in both arms' graphs (they become ordinary
dependencies): both arms are timed as a chain of dependent layers; PDL inside a call stays allowed. When the reference
chains its calls by PDL (cuBLAS does on the B200), the judge reports, for the record, what that PDL would have brought
it.

The checks. The candidate's result must be within max |C - ref| / max |ref| <= 2e-3 of an fp32 reference, with every
element finite and written: before the timing (direct calls on the three input sets), during it (the timed graph's
output after every block of the candidate) and after it (new contents at the same addresses, the timed graph run again,
then direct calls). A and B must not change, nor the GPU's state (limits, cache configuration, the L2 window of the
reference's stream). A failed check writes `FAILED` (and marks the shape's line `# timing refused` when it comes during
or after the timing), and the pass ends with `SCORE INVALID`. A candidate may not call cuBLAS or cuDNN, load a library,
capture graphs or change the GPU's state: the launcher checks its symbols before the link.

```
modal run judge/modal_judge.py --candidates turing/step6.cu,turing/step5.cu --card T4 --reference \
    --variant "Tesla_T4|70W|lt130201|cuda13030" --tries 4
```

The output lands in `<rung>/results/judge/<date>-<time>_<candidate>/output.txt` (a folder of its own:
`<date>-<time>-2_...` when two outputs land in the same second), with a copy of the judged sources. Line 1 gives the
candidate's path and the md5 of each file sent; line 2 the md5 of judge.cu, of the launcher, of the shapes file and of
the reference file sent, then the options, the variant and the try. Then the judge's output, line by line: a header
(`judge v4 (md5 ...)`, `card: <card line>`, `token: <token>`); per shape, the `info:` lines of the ranking and the
final, the `REFERENCE` lines (the finalists, for the cache) and one CSV line under the header
`shape,M,N,K,reference,t_ref_us,t_cand_us,ratio,ratio_min,
ratio_max,mhz_ref,mhz_cand,w_ref,w_cand,ratio_cycles,err_before,err_after,throttle_ref,throttle_cand` (`ratio` is the
median of the pairs, `throttle_*` the NVML clock event reasons of each arm); at the end, the `SCORE` line and a `CYCLES`
line (the ratio in cycles at the shapes with M >= 512, for the record: the score is the SCORE line). A thermal or
hardware throttle writes `WARNING` and `CARD THROTTLED`.

Options of `judge.cu`: `--pairs P` (4), `--duration S` (1.5), `--warmup S` (0.5), `--quick` (2 pairs of 0.6 s with 0.2 s
of warm-up, one round of the final), `--only a,b` (some shapes only), `--default` (also times cuBLAS's default call with
the pool workspace: column `ratio_default`), `--seed G`, `--reference FILE`. Options of the launcher: `--candidates`,
`--card` (T4, A100-80GB, H100!, B200), `--options "..."` (the options of `judge.cu` above but `--seed` and
`--reference`), `--reference`, `--shapes FILE`, `--variant "<card line>"` with `--tries N` (the GPU container reads its
card line before judging and stops if it differs), `--recover FILE.json` (stores a pass copied to the Modal volume
`judge-outputs`, with the files it judged; refused if a judged source, judge.cu or the shapes file has changed on the
disk since), `--control` (lifts the symbol filter, for control candidates kept in a `judge/tests/` folder, which is not
part of this repository).
