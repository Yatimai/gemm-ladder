# gemm-ladder

One GEMM (fp16 inputs, fp32 accumulation, fp16 output) on the Tesla T4, built step by step from NVIDIA's tensor-core
sample. The score is taken against the judge's reference, the fastest cuBLASLt configuration the judge's search finds
(the first configurations cuBLASLt's heuristic offers at several workspace sizes, and variations around the first of
each family: judge/README.md), at 17 shapes written M x N x K: the four GEMM of a Llama-3-8B layer, qkv (N 6144,
K 4096), o (4096, 4096), gateup (28672, 4096) and down (4096, 14336), at M = 16, 128, 512 and 2048 tokens (qkv_m16 is
qkv at M = 16, and so on), plus 2048 x 2560 x 2048, this ladder's own shape, named ladder. The score is the geometric
mean of the time ratios, measured in two containers (two Modal GPU instances, which may land on two different cards).

With the cached reference, the judge sorts again the configurations it keeps for this card line
(`judge/reference/t4.txt`) together with the first ones of cuBLASLt's heuristic (its first 3 with 64 MB of workspace,
its first 8 not yet present with 32, 4, 1 and 0 MB), then times the first five against each other in a final, in
blocks of 1.5 s; the fastest is the reference (`judge/judge.cu`). cuBLAS's documentation describes cuBLASLt in these
words ([1. Introduction](https://docs.nvidia.com/cuda/archive/13.1.1/cublas/index.html#introduction), the
documentation of CUDA 13.1 Update 1, whose cuBLAS is 13.2.1):

> The cuBLASLt is a lightweight library dedicated to GEneral Matrix-to-matrix Multiply (GEMM) operations with a new
> flexible API. This library adds flexibility in matrix data layouts, input types, compute types, and also in choosing
> the algorithmic implementations and heuristics through parameter programmability.

## The rung

| Rung | GPU | Mechanisms | Score |
|---|---|---|---|
| [Turing](turing/) | T4, sm_75 | tensor cores through `mma.sync` and `ldmatrix`, swizzled shared stages, fragment double buffering; then loads through the L2 only, 16-row bands at M <= 16, split-K in two | 1.05 |

Score: the reference's time over the rung's, geometric mean over the 17 shapes (above 1: faster than the reference); the
lower of two containers, rounded down to two decimals. `turing/README.md` gives the steps, the measures and their
sources.

## How it is measured

- The judge (`judge/`, see its README) checks each candidate's result and times it against the reference at the 17
  shapes.
- The interleaved probe (`probe/probe.cu`; launcher `probe/modal_probe.py`) times a step and its variants (each defined
  in `step<i>.cu`) in one process, with the step timed a second time (arm `twin`) to measure the noise; under ncu, one
  process per arm, the step, a removal and the configuration of a REFERENCE line of the judge (arm `x:`).
- The host emulator (`turing/emu`) runs every kernel on the CPU and checks its accesses, barriers, bank conflicts and
  results (to the bit where the headers say so); its planted-bug test checks that the emulator catches bugs planted in
  the kernels. Their outputs on the T4 rung's sources: `turing/results/emu/` (`emu_all.txt`: 771 runs, 771 ok, 0
  failed; `planted_bugs.txt`: 79 of 79 planted bugs reported as failures).
- compute-sanitizer (memcheck, racecheck, synccheck) on `turing/verif.cu`, ncu at ladder, and ncu in the probes of
  step 6. cuBLAS is described only by its documentation (above), ncu's counters, its kernels' names and grids, and
  cuBLASLt's documented attributes.
- The raw outputs are in `turing/results/`.

## Running it

From the repository's root, with Modal for the GPU runs:

```
modal run judge/modal_judge.py --candidates turing/step6.cu --card T4 --reference \
    --variant "Tesla_T4|70W|lt130201|cuda13030" --tries 4                   # the judge
modal run modal/session.py --rung turing --steps step6 \
    --variant "Tesla_T4|70W|lt130201|cuda13030" --tries 4                   # compute-sanitizer and ncu
modal run probe/modal_probe.py --card T4 --candidate turing/step6.cu --shapes "2048,2560,2048" \
    --arms "v0,twin,v1" --rounds 5 --variant "Tesla_T4|70W|lt130201|cuda13030" --tries 4  # the probe
(cd turing/emu && g++ -std=c++17 -O2 -march=native -I.. -Ishim emu.cpp -o emu && ./emu)   # the emulator, no GPU
```

## License

MIT, see [LICENSE](LICENSE). NVIDIA's BSD notice stays in `nvidia_sample.cuh` and in the headers derived from it.
