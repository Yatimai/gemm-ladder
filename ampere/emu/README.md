# Host emulator (Ampere rung)

`emu` runs the Ampere kernels of this rung on the CPU, compiled from their own source, at the first six shapes of
`verif.cu` (and three more, below): `nvidia_sample.cuh` (step 0, `compute_gemm`, and `simple_wmma_gemm`, the reference
point), `step1.cuh` (v0 to v3; v4 is step 0), `step2.cuh` (v0 and v1; v2 is step 1's v0), `step3.cuh` (v0 to v2; v3 is
step 2's v0), `step4.cuh` (v0 to v2; v3 is step 3's v0), `step5.cuh` (v0 to v2; v3 is step 4's v0), `step6.cuh` (its
tiles, step 5's v0 at 128 x 256, 6b's bands and stream and 6c's stream in phase, all without 7d's hints, as step 6 runs
them; 7a's parts of K and their reduction and its parts in place, 7e's tail in parts, 7h's reads together in the parts
in place and in 6c's kernel, as step 7 runs them) and `step7.cuh` (7c's parts in place, 7g's parts reduced by their own
blocks with 7d's hints and without, and 6b's, 7b's and 6c's kernels with 7d's hints and discards, as step 7 runs them).
It is the Turing rung's emulator (`turing/emu`, whose README describes how it works) with these changes:
- the persistent grids (step 0, step 1's v1 and v3) have 108 blocks, the A100's SMs, as `verif.cu` runs them, and also
  3, so that each block goes through several tiles even at the small shapes;
- `step2.cuh` issues `mma.sync.aligned.m16n8k16` (one HMMA.16816 on sm_80) where the Turing rung's issues m16n8k8; its
  C++ branch states the per-lane semantics of the PTX ISA (g = lane / 4, t = lane % 4): a[0] = {A[g][2t], A[g][2t + 1]},
  a[1] = {A[g + 8][2t], ...}, a[2] = {A[g][2t + 8], ...}, a[3] = {A[g + 8][2t + 8], ...}; b0 = {B[2t][g], B[2t + 1][g]},
  b1 = {B[2t + 8][g], B[2t + 9][g]}; d = D[g][2t], D[g][2t + 1], D[g + 8][2t], D[g + 8][2t + 1], accumulated in fp32;
- the bank patterns the headers state: the copy's 16-byte stores conflict-free on step 0's and step 1's rows of 160
  bytes (K64 + 16 halves of padding) as on step 2's swizzled rows; LDSM.x4 two-way with the padding (step 2's v1),
  conflict-free with the swizzle (v0); the variants of steps 2 to 6 without a split of K must give the same bits (the
  orders of the tiles of steps 4 and 6 are also checked on their own, as the Turing rung's, at every shape up to
  8192 x 12288, and step 6's plan at the judge's 17 shapes);
- `step3.cuh`'s `cp.async` (one LDGSTS.128) is emulated as a deferred copy: checked and counted for the bank model when
  issued (class LDGSTS.128 A or B), performed as late as the program allows (when a `cp.async.wait_group` leaves its
  group out of those still in flight) in the forward order of the threads, as early as possible (when issued) in the
  reverse order, and either way, copy by copy at random, in the random order; a copy never waited for at the end of the
  kernel is an error. A missing wait or barrier leaves stale data in shared memory in the first order, a copy into a
  stage still being read overwrites it in the second: both give a wrong C. A wait does not hand over to the other
  threads: it waits for its own thread's copies only, and another thread sees a copy landed or not, the two ends that
  the first two orders give. The bank model covers the shared side of each copy only: its global side (the sectors it
  reads, L1, L2) is not modeled;
- 7a's parts of K (step 7's) run against a workspace and flags between guard regions, like A, B and C: the parts are
  filled with NaN before each run, the reduction runs as a second grid after the first, the parts in place run row of
  the grid by row (each part after the earlier parts of its tile, as step6.cuh assumes), a part that sets its tile's
  flag must have waited on it for the value before, each element of C is written once per part in place, and every flag
  must be back at zero after the run;
- the shared array of each step is 164 KB (a block's 163 KB on sm_80, and a canary): a launch above it is refused, and
  an access that lands in another step's array is an error;
- the kernels whose blocks wait on each other or arrive on a count or a flag of the grid (7b and 7g: each block of a
  tile waits until the tile's S parts have arrived; 6b's stream, 6c and 7e: the last block to arrive adds the parts; 7a
  and 7c in place: a part sets its tile's flag for the next) run with the blocks as coroutines, up to 16 alive at once,
  the emulator's SMs (7b's and 7g's shapes are the ones whose blocks fit on them). A block that waits on a count of the
  grid not yet reached is suspended, its shared memory kept, and the others run. An arriving block is suspended there,
  its other threads where they are; the forward and reverse orders resume it only when no other block can start or
  resume, the last one first: a block that arrives before its other threads' writes are done shows, since the block that
  reads them runs in between. The fences are not modeled (each store is seen at once): a barrier that only orders the
  threads' fences before an arrival, after a barrier that already follows their stores, can go unseen. When every alive
  block waits on a count that will not reach its value, or the next block cannot start, the run fails (on the GPU, a
  hang). The forward order resumes a block whose wait is met before it starts the next, the reverse order starts the
  next first, the random order draws among all. 7b's and 7g's kernels run twice in a row, the second call on the
  counters the first put back to zero, C set to NaN between the calls and compared; a read of another block's data (C
  in place, the parts) before the block's acquisition for its tile (its wait or its arrival) is an error;
- 7d's `discard.global.L2` (in step6.cuh's kernels, as step 7 runs them, and in 7g's kernel) fills the discarded line
  with NaN (its content is undefined after it), so that a part read after its discard shows; its cache hints run as
  plain copies; 6b's stream and 6c run at the A100's 108 SMs and at 7, where the ranges cut more tiles; the rules of
  steps 6 and 7 (`step6::m16`, `parallel_fits`, `phase`, `tail_parts`; `step7::inplace_parts`, `phase_in_place`) are
  checked at the judge's 17 shapes (`step6::m16` at seven of them);
- on 108 SMs, `verif.cu`'s shapes do not run on the GPU a block of 6b's stream that holds more than one unit, or 7c
  in more than 2 parts; the emulator runs them all (the stream at 7 blocks, 7c in 5 parts).

```
cd ampere/emu
g++ -std=c++17 -O2 -march=native -I.. -Ishim emu.cpp -o emu
./emu                     # every kernel: 1076 runs, the three orders of the tiles, step 6's plan, steps 6 and 7's rules
./emu step6 -x            # step 6 (and step 5's v0): 139 runs
```

The planted-bug test, `planted_bugs.py` (the Turing rung's script with this rung's headers): 164 bugs, each one exact
substitution in a copy of `step1.cuh` to `step7.cuh`, run by the step of its header (those of 7a, 7b, 7d, 7e, 7g and 7h
written in `step6.cuh`, by step 7): barriers removed or moved, waits one group too loose, copies into a stage still
being read, commits left out, an arrival before the block's part is written, counts and flags off by one or never put
back to zero, discards before a read, a flag written just outside the launch's shared memory, registers swapped, a
swizzle on one side only, wrong orders of tiles, wrong rules (each bug is described in the file). The emulator built
against each copy must report a failure, and the unmodified copy must pass every run; for each bug, the output says that
it is reported and at which first failed run. Steps 3 to 7 also run at two shapes of `verif.cu` with 4 and 16 slices of
K64 (with at most 3 slices the three stages never come round), and steps 6 and 7 at a third, 16 x 6144 x 128.

```
python3 ampere/emu/planted_bugs.py [-j JOBS] [--keep]
```
