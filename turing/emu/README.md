# Host emulator

`emu` runs the Turing kernels of this rung on the CPU, compiled from their own source (`nvidia_sample.cuh`, `step1.cuh`
to `step6.cuh`, under `GEMM_LADDER_EMULATE`, with the headers of `shim/` in place of CUDA's), at 6 of `verif.cu`'s 9
shapes (all 9 for step 6, and two with M < 16). It finds a kernel's bugs on a machine without a GPU: a missing barrier,
a fragment register out of place, a swizzle applied on one side only, an access outside an array, an element of C
written twice or never. One block runs at a time, its threads as coroutines that meet at the barriers and at the
warp-wide instructions (`ldmatrix` and `mma.sync`, whose per-lane semantics `step2.cuh` states under
`GEMM_LADDER_EMULATE`; wmma at the tile level), in three orders of the blocks and of the threads: forward, reverse and
random. Cache hints, fences and atomics become plain accesses.

## What it checks

At each run (a kernel, a shape, an order):
- every element of C written exactly once, within `verif.cu`'s tolerance (2e-3, against a reference in double);
- the 16- and 4-byte copies (`int4`, `__half2`), the `ldmatrix` rows and the wmma loads and stores: inside A, B, C or
  the launch's shared memory, aligned, and no store to A or B (A, B and C lie between guard regions; shared memory and
  C are filled with NaN before each run);
- the barriers: without one, a thread reading what another writes in the same phase gets NaN or another slice's value,
  in one of the three orders;
- the shared-memory wavefronts of each instruction (32 banks of 4 bytes), against the bank conflicts the headers state;
- the same bits as step 2's v0 where the headers say so (the variants of steps 3 to 5, step 6's tiles), and the same
  bits in the three orders for step 6's sums of parts (6b, 6c);
- step 6's parts written and read in place, and its counters back at zero after each run.

Before the runs, at every shape up to 8192 x 12288: the order of the tiles of steps 3 to 6 (each tile at one place).
At the judge's 17 shapes: step 5's choice of tile and step 6's plan, against the tables in `emu.cpp`.

## How to run it

Linux and g++ 12 or later:

```
cd turing/emu
g++ -std=c++17 -O2 -march=native -I.. -Ishim emu.cpp -o emu
./emu                     # every kernel
./emu step1 step2         # some steps (step0 to step6)
./emu step2 -x            # stop at the first failed run
python3 planted_bugs.py -j 3  # the planted-bug test
```

Per kernel run, one line: the kernel, shape, order, verdict, error, wavefronts per shared instruction. A failed run
comes after its `EMU ERROR` lines (array, element, block and thread). `ALL OK` and exit code 0 only if every run passed.
The outputs of `./emu all` and of `python3 planted_bugs.py -j 2` on this rung's sources: `../results/emu/emu_all.txt`
(771 runs, 771 ok, 0 failed, `ALL OK`) and `../results/emu/planted_bugs.txt` (79 of 79 planted bugs reported as
failures, the unmodified copy passes, `PLANTED-BUG TEST OK`).

## What it does not prove

- Two blocks at once: the publication of 6b's and 6c's parts (fences, atomics) is checked on the GPU only, by
  `verif.cu` (the counters' state after each call).
- Time, traffic or SASS: no clocks, occupancy, cache hints or traffic between DRAM, L2 and the SMs, and nothing of what
  ptxas makes of the source. The probe, ncu and the SASS show them.
- That the semantics it states are the hardware's. They are the PTX ISA's; on the T4, ncu gives the excess wavefronts of
  its bank model (the rung's README, `../README.md`).
- Every schedule: three orders among many.
- 6b in 12 slices, as step 6 runs it on the GPU at 16 x 1024 x 1024: the emulator runs 6b in 1, 2, 3 or 5 slices
  (the judge's shapes use 1, 3 and 5).
- `step5.cu` and `step6.cu` themselves: the choice of a variant and its launch are read, not emulated.

## The planted-bug test

`planted_bugs.py` (from anywhere; `-j N` parallel jobs, `--keep` to keep the temporary directory) makes 79 exact text
substitutions, one at a time, in a copy of `step1.cuh` to `step6.cuh` in a temporary directory: barriers removed or
moved, registers swapped, a swizzle on one side only, the write stage frozen, clamps removed, wrong orders of tiles,
counters not put back to zero, wrong rules (the list, each bug described, is in the file). The emulator, built against
each copy, must report each bug as a failure, and the unmodified copy must pass every run: then `PLANTED-BUG TEST OK`,
exit code 0. A substitution whose text is not found exactly once stops the test before any build (exit code 2): the
source has changed, and the table must follow it.
