# Host emulator

`emu` runs the kernels of this rung on the CPU, compiled from their own source (`nvidia_sample.cuh`, `step1.cuh` to
`step7.cuh`, under `GEMM_LADDER_EMULATE`, with the headers of `shim/` in place of CUDA's). Each kernel runs at the
shapes of the emulator's list given for its step that it can take, whatever the step's rule picks there: ten of
`verif.cu`'s shapes and others that turn each ring of stages, cut the tiles and the bands by M, and run the persistent
grids over several rounds. It finds a kernel's bugs on a machine without a GPU: a missing wait or barrier, a stage
reused while an MMA or a TMA store still reads it, a descriptor or a swizzle that does not match the data, an access
outside an array, an element of C written twice or never, a deadlock. The threads of a block (of a cluster, from step 4)
run as coroutines that meet at the barriers and at the warp-wide and warpgroup-wide instructions, in three orders:
forward, reverse and random. Each step adds only the engines it brings; their semantics are the PTX ISA's.

## The engines, step by step

- Step 0: `wmma` at the tile level; the persistent grid at 132 blocks and at 3.
- Step 1: `wgmma.mma_async` (operands through their descriptors, the 128-byte swizzle, the accumulators in the layout of
  the instruction); `wgmma.fence`, `commit_group`, `wait_group`: an MMA is performed when issued in the forward order,
  as late as its `wait_group` allows in the reverse order (its accumulators NaN until then), chain by chain at random in
  the random order. A wait with a group of a late chain to retire first lets the other threads run for two rounds of the
  scheduler, one more per warpgroup of the block before its own (the warpgroups of a block retire their groups at
  different times), those that a barrier completed meanwhile wakes among them, as the other warps run on while the MMA
  are in flight: a stage released before the wait that retires its MMA, or loaded again before another warpgroup has
  released it, can then be overwritten before the MMA read it. The async proxy: a byte stored by a thread and read by an
  MMA or a TMA store with no `fence.proxy.async` of that thread in between is an error.
- Step 2: `mbarrier` (arrivals, bytes, phases, parity; a barrier never initialized, too many arrivals or bytes are
  errors); the TMA's 2D loads (the tensor map our host code builds, zeros outside the matrix, the swizzle), landing late
  in the forward order, when issued in the reverse order, at random in the random order. Threads waiting at barriers
  that nothing can complete end the run: a DEADLOCK, with the barrier and what it awaits.
- Step 3: `elect.sync`, named barriers, `__shfl_sync`.
- Step 4: the blocks of a cluster (up to eight) run together, each with its own shared memory, and the cluster
  barrier. A block may leave only once its TMA loads have landed, in it and in the others.
- Step 5: no engine of its own; the order of its tiles is checked apart from the runs (below).
- Step 6: the MMA with either operand MN-major (the bands' transposed MMA), several accumulator ranges in one chain,
  `stmatrix .trans`, the TMA's 2D stores (the bands' output, early or late), warpgroups per block, step 6's rule.
- Step 7: `setmaxnreg` (a pool of registers per block; `.inc` waits for it; the same in a warpgroup), `mapa`, arrivals
  on another block's barriers and the multicast TMA (none before a wait at the cluster barrier, none to a block that
  has left). It runs its bands, step 5's tile in pairs and its persistent kernel at each
  tile, at v0's switches and at each variant's, and step 6's kernels at its classes (v8).

## What it checks

At each run (a kernel, a shape, an order): every element of C written exactly once, within `verif.cu`'s tolerance
(against a reference in double); every global and shared access inside its array and aligned; the shared-memory
wavefronts against the bank conflicts the headers state; the same bits for the variants of a step and in the three
orders. Apart from the runs: the order of the tiles of step 5 at every shape up to 8192 x 12288 and at the judge's
shapes (each pair of tiles once, each tile once); step 6's rule at the judge's 17 shapes (the kernels a table of
`emu.cpp` names) and at every shape up to 4096 x 65 536 (a kernel that can take the shape); step 7's rule the same way,
with 4, 13 and 15 clusters of eight held at once (7b's clusters one per GPC), 7e's (step 6's rule with 7e) at M above
128, and 7c's (its groups and grid, up to 40 x 100 tiles; each tile once in groups of 1 to 16 rows). For the rings, the
shapes turn each ring of stages several times, and one slice more; 16 x 2048 x 320 takes a cluster of step 5's tile
through two units along N.

## How to run it

```
cd hopper/emu
g++ -std=c++17 -O2 -march=native -I.. -Ishim emu.cpp -o emu
./emu                     # every kernel
./emu step6 -x            # one step, stopping at the first failure
```

## The planted-bug test

`planted_bugs.py`: 239 bugs, each one exact text substitution in a copy of `nvidia_sample.cuh` or of `step1.cuh` to
`step7.cuh` (step 0: 10, which check the emulator's engines on NVIDIA's code; step 1: 30; step 2: 19; step 4: 7; step 5:
26; step 6: 40, 6 of them (7e's rule, and its tile without warp specialization) run by step 7; step 7: 107; step 3, step
2's kernel at 4 stages, has none). They cover the waits, barriers and fences removed or relaxed, the stages released
early (among them, at steps 2, 5, 6 and 7, a stage released before the wait that retires its MMA), the barriers' counts
and parities, the bytes announced, the descriptors and swizzles, the loading warp and its registers, the multicast, the
cluster's arrivals and barriers, the block that leaves too early, the order of the tiles, the output, the rules of steps
6 and 7. The emulator built against each copy must report a failure, and the unmodified copy must pass every run:
`python3 hopper/emu/planted_bugs.py [-j JOBS] [--keep]`.

## What it does not prove

- That the semantics it states are the silicon's: `verif.cu` on the GPU checks each step at all its shapes, within the
  tolerance, and makes each call 8 times, its outputs compared bit for bit (a race shows as outputs that differ).
- The host code of the steps' `.cu` files (their tensor maps, grids and choices): the emulator launches the kernels with
  its own, written as theirs; `verif.cu` runs the `.cu` files.
- Everything the PTX ISA leaves unstated, such as the order of the tensor cores' sums.
