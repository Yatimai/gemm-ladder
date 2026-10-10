# gemm-ladder

The same GEMM (fp16 inputs, fp32 accumulation, fp16 output) on four generations of NVIDIA GPUs. Each rung starts from
NVIDIA's own tensor-core sample, rebuilds cuBLAS's recipe step by step (read under ncu or in cuBLASLt's settings), then
goes beyond it. The score is taken against the judge's reference, the faster of cuBLASLt's best configuration and
cuBLAS's default call (cublasGemmEx), at 17 shapes written M x N x K: the four GEMM of a Llama-3-8B layer, qkv (N 6144,
K 4096), o (4096, 4096), gateup (28672, 4096) and down (4096, 14336), at M = 16, 128, 512 and 2048 tokens, plus 2048 x
2560 x 2048, this ladder's own shape, named ladder. The score is the geometric mean of the time ratios, measured in two
containers (two Modal GPU instances, which may land on two different cards).

## The rungs

| Rung | GPU | What is new on this card | Score | State |
|---|---|---|---|---|
| [Turing](turing/) | T4, sm_75 | tensor cores through `mma.sync` and `ldmatrix`, swizzled shared stages, fragment pipelining; then loads through the L2 only, synchronized waves, a kernel of its own at M = 16, K split in two | 1.07 | measured |
| Ampere | A100 80GB, sm_80 | `cp.async`, deeper pipelines of shared stages | | in progress |
| Hopper | H100, sm_90a | TMA, `wgmma`, warp specialization, clusters | | planned |
| Blackwell | B200, sm_100a | `tcgen05`, tensor memory, MMA over two SMs | | planned |

Score: the reference's time over the rung's, geometric mean over the 17 shapes (above 1: faster than the reference); the
lower of two containers, rounded down to two decimals. Each rung's README gives the steps, the measures and their
sources.

## How it is measured

- The judge (`judge/`, see its README) checks each candidate's result and times it against the reference at the 17
  shapes.
- The interleaved probe (`probe/`, the T4's and the B200's, on the same method) times a step against each of its
  removals (the step with one mechanism taken out) and cuBLASLt in one process: time, clock and energy per GEMM. The
  B200's is published ahead of the Blackwell rung, which will use it.
- Each rung's host emulator (`<rung>/emu`) runs every kernel on the CPU and checks its accesses, barriers, bank
  conflicts and results (to the bit where the headers say so); its planted-bug test checks that the emulator catches
  bugs planted in the kernels.
- compute-sanitizer (memcheck, racecheck, synccheck) on each rung's `verif.cu`, and ncu at ladder. cuBLAS is described
  by ncu's counts, mechanism names and cuBLASLt's settings only.
- The raw outputs are in each rung's `results/`.

## Running it

From the repository's root, with Modal for the GPU runs:

```
modal run judge/modal_judge.py --candidates turing/step6.cu --card T4 --reference \
    --variant "Tesla_T4|70W|lt130201|cuda13030" --tries 4                   # the judge
modal run modal/session.py --rung turing --steps step6 \
    --variant "Tesla_T4|70W|lt130201|cuda13030" --tries 4                   # compute-sanitizer and ncu
modal run probe/modal_probe_t4.py --candidate turing/step6.cu --shapes "2048,2560,2048" --arms "lt,v0,v1" \
    --rounds 5 --variant "Tesla_T4|70W|lt130201|cuda13030" --tries 4       # the probe
(cd turing/emu && g++ -std=c++17 -O2 -march=native -I.. -Ishim emu.cpp -o emu && ./emu)   # the emulator, no GPU
```

## License

MIT, see [LICENSE](LICENSE). NVIDIA's BSD notice stays in `nvidia_sample.cuh` and in the headers derived from it.
