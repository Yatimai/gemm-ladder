// The Hopper rung's step 7 (see step7.cu): its kernels and its rule, on step 6 (step6.cuh).
#pragma once
#include <cstdlib>
#include "step6.cuh"

namespace step7 {

// ----------------------------------------------- step 7: warp specialization, taller and wider bands, a 128 x 320 tile
// Step 6 picks a tile by classes of shapes, on kernels where every warp computes and warp 0 also loads. Step 7 adds
// eleven mechanisms, each a rule by classes of shapes, each with its variant that takes it out (step7.cu). Five shape
// the work:
// (7a) 64 < M <= 128: the bands of 6a, taller: 64 rows (wgmma m64n64k16, transposed as 6a's) in pairs along M (the
//      two blocks of a cluster take the same columns, rows 0 to 63 and 64 to 127), bands of two columns of 64 (J = 2),
//      when its blocks make one wave (2 x ceil(N / 128) <= the SM count); else step 6's choice.
// (7b) M <= 16: the bands twice as wide (J doubled, at most 8 columns of 64), in clusters of eight along N, when its
//      clusters can each have a GPC of their own: a cluster takes eight SMs of one GPC, a GPC of 16 to 18 SMs holds
//      two, and the GPU places one per GPC first, so no more clusters than half those the GPU holds at once, rounded
//      up; else the 16-row bands alone at step 6's J.
// (7c) 6c's waves as whole rectangles of tiles: 5b's order in groups of G rows of tiles, G the largest (at most 16, the
//      rows and the blocks) whose grid of R = G x floor(SMs / G) blocks keeps the rounds, ceil(tiles / SMs); R blocks
//      (fewer when the tiles are fewer). In one wave (its tiles at most the blocks the GPU holds at once), A's and B's
//      maps at the L2 promotion of 256 bytes instead of 128.
// (7d) 64 < M <= 128 when the pairs of 7a do not make one wave: the bands of 6a at 128 rows (wgmma m64n128k16,
//      transposed), one block per band of two columns of 64, alone, when its blocks make one wave (ceil(N / 128) <= the
//      SM count); else step 6's choice.
// (7e) M > 128: 6c's tile widened to 128 x 320 (wgmma m64n192 then m64n128 per warpgroup: 160 accumulators per
//      thread), when its waves times its area, 1.25 of 6c's, are fewer than 6c's waves, and wherever N is not a
//      multiple of 256; else 6c. Alone when its tiles make one wave (each block takes one tile), else in pairs along M.
//      The output in passes of 128, 128 and 64 columns; a last column of tiles cut by N reads zeros from column N on
//      and writes nothing there. Its rule is step6::choose with the 128 x 320 tile.
// Of the six below, 7f to 7j come from CUTLASS's warp-specialized kernels for sm_90 (public source, CUTLASS 4.4.1:
// the cooperative kernel, its pipeline and its TMA epilogue); 3b, 3c, the shared operand of 4a, 4d and 5c left their
// steps at the test global and come back here:
// (7f) Warp specialization (the roles, 3b), at every shape: a loading warp beside the computing warpgroup of the bands
//      (160 threads), a loading warpgroup beside the two computing ones of the tiles (384 threads), whose four warps
//      give registers back (setmaxnreg.dec to 24) and of which warps 1 to 3 leave; the loading warp (warp 0 of the
//      tiles, warp 4 of the bands) waits for each stage to be released, and one of its lanes (elect.sync) announces it
//      and issues its boxes; the computing warpgroups release each stage (one arrival each) and, in 6c's tile and
//      7e's, raise their registers to 232 (setmaxnreg.inc).
// (7g) The descriptors in the uniform datapath (3c), at every shape: the computing warpgroup's index and the stage of
//      each slice (of a tile's first slice, in step 5's tile) broadcast by a shuffle, the descriptors computed once,
//      then advanced by additions (without it, at the tiles and the 128-row bands, ptxas rebuilds them in vector
//      registers: R2UR at each slice).
// (7h) The shared operand (4a's multicast TMA), at the shapes whose kernel runs in clusters: in 7b's clusters, the
//      first block loads A's 16 rows and multicasts them to the eight; in 7a's pairs, each block loads half of B's
//      boxes for both; in the pairs of step 5's tile (side by side, M <= 128), each block loads 64 of A's 128 rows for
//      both; in the pairs of 7e's tile and of 64 x 128 (one above the other), B's first boxes come from the first
//      block, the others from the second. A stage, filled by several blocks, is released by all their computing
//      warpgroups (an arrival on another block's barrier through mapa).
// (7i) The end of the loading warp (4d), at 7h's shapes: the computing warpgroups release every stage to the cluster,
//      the last one included, and the loading warp, once it has issued its last loads, waits until they have released
//      its last stages: when a block leaves, nothing of either block can still reach the other (the CUDA Programming
//      Guide, Distributed Shared Memory, leaves this to the code).
// (7j) The tiles' output by TMA stores (5c), at the shapes of the tiles: each warpgroup's pass of up to 128 columns,
//      written into its zone of boxes of 64 x 64 at the 128-byte swizzle, made visible to the async proxy, then
//      copied out by TMA stores issued by one thread and not waited for: the zone is written again once the stores
//      have read it. Rows >= M are left out by the TMA.
// (7l) An L2 hint (.L2::cache_hint, evict_first) on the bands' TMA loads of B, read once, at the shapes of the bands.
// The bands of 16 rows keep 6a's ring (wait_group 1 at 8 stages; below, a stage released once its MMA are retired);
// the bands of 64 and 128 rows and the tiles keep one group of MMA in flight (wait_group 1). The bands keep 6a's
// output (stmatrix .trans into a zone, then TMA stores). A cluster is synchronized in one piece, each thread's arrival
// at the cluster barrier right before its wait (4c, the arrival split from the wait, left the code); without 7h (v9,
// v11), no block reaches another, and the barrier stays. The waits are step 2's (5d, the suspended waits, left the
// code). Without the roles (step7.cu's v8), step 7's classes run on step 6's kernels: the bands below (bands_noroles),
// step 6's persistent kernel and step 5's tile.
// Shapes: M a multiple of 16, N of 64, K of 64.

#ifdef GEMM_LADDER_EMULATE
// A host emulation: setmaxnreg is checked to be the same in the whole warpgroup; the blocks of a cluster run together,
// an access to another block checked; the hinted loads run as plain loads (the hint changes no result).
void emu_setmaxnreg(int regs, bool inc);
uint32_t emu_mapa(uint32_t addr, uint32_t rank);
void emu_mbar_arrive_remote(uint32_t addr);
void emu_tma_load_2d_multicast(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar, uint16_t mask);
#endif

// setmaxnreg.dec / .inc: the warp's registers lowered to (raised to) R per thread (the same in a whole warpgroup); .inc
// takes them from those the block's other warps gave back by setmaxnreg.dec, and waits until they are there.
template <int R>
__device__ __forceinline__ void setmaxnreg_dec()
{
#ifdef GEMM_LADDER_EMULATE
    emu_setmaxnreg(R, false);
#else
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(R));
#endif
}
template <int R>
__device__ __forceinline__ void setmaxnreg_inc()
{
#ifdef GEMM_LADDER_EMULATE
    emu_setmaxnreg(R, true);
#else
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(R));
#endif
}

// mapa: the address, in the cluster's shared window, of the same shared variable in block `rank`.
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank)
{
#ifdef GEMM_LADDER_EMULATE
    return emu_mapa(addr, rank);
#else
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(addr), "r"(rank));
    return r;
#endif
}

// mbarrier.arrive on a barrier of the cluster (through mapa's address), with the default semantics (release at the
// cluster's scope, as CUTLASS's ClusterBarrier::arrive).
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t addr)
{
#ifdef GEMM_LADDER_EMULATE
    emu_mbar_arrive_remote(addr);
#else
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n" ::"r"(addr) : "memory");
#endif
}

// A multicast TMA load: the box lands at the same offset in the shared memory of every block of `mask`, and its bytes
// complete on each one's barrier at the offset of bar.
__device__ __forceinline__ void tma_load_2d_multicast(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar,
                                                      uint16_t mask)
{
#ifdef GEMM_LADDER_EMULATE
    emu_tma_load_2d_multicast(dst, map, c0, c1, bar, mask);
#else
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                 " [%0], [%1, {%2, %3}], [%4], %5;\n"
                 ::"r"(step1::smem_addr(dst)), "l"(map), "r"(c0), "r"(c1), "r"(step1::smem_addr(bar)), "h"(mask)
                 : "memory");
#endif
}

// A's map for step 5's tile in pairs along N: boxes of 64 K x 64 rows (8 KB), two per stage, one from each block.
inline CUtensorMap map_a64(const half* A, int M, int K) { return step2::make_map(A, K, M, 64, 64); }

// C's map for the TMA stores: boxes of 64 columns x 64 rows (8 KB), 128-byte swizzle.
inline CUtensorMap map_c(half* C, int M, int N) { return step2::make_map(C, N, M, 64, 64); }

// The L2 cache policy of the hinted loads (createpolicy, the whole of an access).
__device__ __forceinline__ uint64_t policy_evict_first()
{
#ifdef GEMM_LADDER_EMULATE
    return 2;
#else
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
    return p;
#endif
}

// A TMA load of a 2D box with an L2 hint (step2::tma_load_2d with .L2::cache_hint).
__device__ __forceinline__ void tma_load_2d_hint(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar,
                                                 uint64_t policy)
{
#ifdef GEMM_LADDER_EMULATE
    (void)policy;
    step2::tma_load_2d(dst, map, c0, c1, bar);
#else
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                 " [%0], [%1, {%2, %3}], [%4], %5;\n"
                 ::"r"(step1::smem_addr(dst)), "l"(map), "r"(c0), "r"(c1), "r"(step1::smem_addr(bar)), "l"(policy)
                 : "memory");
#endif
}
// The same, multicast to the blocks of `mask` (tma_load_2d_multicast with .L2::cache_hint).
__device__ __forceinline__ void tma_load_2d_multicast_hint(void* dst, const CUtensorMap* map, int c0, int c1,
                                                           uint64_t* bar, uint16_t mask, uint64_t policy)
{
#ifdef GEMM_LADDER_EMULATE
    (void)policy;
    tma_load_2d_multicast(dst, map, c0, c1, bar, mask);
#else
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                 ".L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5, %6;\n"
                 ::"r"(step1::smem_addr(dst)), "l"(map), "r"(c0), "r"(c1), "r"(step1::smem_addr(bar)), "h"(mask),
                 "l"(policy)
                 : "memory");
#endif
}
template <bool kHint>
__device__ __forceinline__ void load(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar, uint64_t pol)
{
    if constexpr (kHint) tma_load_2d_hint(dst, map, c0, c1, bar, pol);
    else step2::tma_load_2d(dst, map, c0, c1, bar);
}
template <bool kHint>
__device__ __forceinline__ void load_multicast(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar,
                                               uint16_t mask, uint64_t pol)
{
    if constexpr (kHint) tma_load_2d_multicast_hint(dst, map, c0, c1, bar, mask, pol);
    else tma_load_2d_multicast(dst, map, c0, c1, bar, mask);
}

// The transposed wgmma of 6a (A MN-major: our B; B K-major: our A) at N = 16, 64 and 128 (our M of a band). The
// accumulators as step6::wgmma_n16t's, for each group of 8 of the MMA's columns: d[4 c], d[4 c + 1] in row g of the
// warp's 16, columns 8 c + 2 q and the next; d[4 c + 2], d[4 c + 3] the same in row g + 8.
template <int N>
__device__ __forceinline__ void wgmma_t(float* d, uint64_t da, uint64_t db, int scaleD);
template <>
__device__ __forceinline__ void wgmma_t<16>(float* d, uint64_t da, uint64_t db, int scaleD)
{
    step6::wgmma_n16t(d, da, db, scaleD);
}
template <>
__device__ __forceinline__ void wgmma_t<64>(float* d, uint64_t da, uint64_t db, int scaleD)
{
#ifdef GEMM_LADDER_EMULATE
    step6::emu_wgmma_t(d, 64, da, db, scaleD, 1, 0);
#else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, "
                 "%11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, "
                 "%31}, %32, %33, p, 1, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                   "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]),
                   "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
                   "+f"(d[30]), "+f"(d[31])
                 : "l"(da), "l"(db), "r"(scaleD));
#endif
}
template <>
__device__ __forceinline__ void wgmma_t<128>(float* d, uint64_t da, uint64_t db, int scaleD)
{
#ifdef GEMM_LADDER_EMULATE
    step6::emu_wgmma_t(d, 128, da, db, scaleD, 1, 0);
#else
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, "
                 "%11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, "
                 "%31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, "
                 "%51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 1, 0;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                   "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]),
                   "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
                   "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]),
                   "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
                   "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]),
                   "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]),
                   "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "l"(da), "l"(db), "r"(scaleD));
#endif
}

// The arrival at the cluster barrier with the default semantics (release: the thread's earlier accesses, those to the
// other blocks included, are performed before the cluster's threads return from their wait): the barrier at the end of
// the kernels in clusters without 7i.
__device__ __forceinline__ void cluster_arrive()
{
#ifdef GEMM_LADDER_EMULATE
    step4::emu_cluster_arrive();
#else
    asm volatile("barrier.cluster.arrive.aligned;\n" ::: "memory");
#endif
}

// ---------------------------------------------------------------- step 5's tile in pairs, with the roles
// One block of 384 threads per 128 x 256 tile at a time, in clusters of two side by side (M <= 128, one row of tiles),
// 5b's order, the ring of stages running on from one tile to the next (step 5's tile, as step 6 names it at M <= 128,
// with 7f). kTma: 7j's output, else the zone copied out by 16-byte stores of the warpgroup's threads; kShare: 7h, the
// two blocks share A (each loads 64 of its 128 rows, multicast: tmA's boxes of 64 rows, map_a64), else each block
// loads its own A and B and releases its stages to itself; kTail (with kShare): 7i, else no last release and no last
// wait of the loading warp, the computing warpgroups arriving at the cluster barrier (release) and waiting there
// (acquire) before leaving; kUni: 7g, else the descriptors rebuilt from the stage's address at each slice.
template <bool kTma, bool kShare, bool kTail, bool kUni>
__global__ void __launch_bounds__(384, 1) gemm5(const __grid_constant__ CUtensorMap tmA,
                                                const __grid_constant__ CUtensorMap tmB,
                                                const __grid_constant__ CUtensorMap tmC, half* C, int M, int N, int K)
{
    using G = step5::G;
    constexpr int kStages = G::kStages;
    uint8_t* shmem = step4::shared_base();
    uint8_t* sm = shmem + ((1024 - (step1::smem_addr(shmem) & 1023)) & 1023);
    uint64_t* full = (uint64_t*)(sm + G::kBars);
    uint64_t* empty = full + kStages;
    const int tid = threadIdx.x, wg = tid / 128, nt = K / 64;
    const int rank = (int)step4::cluster_rank(), other = rank ^ 1;
    int UM, UN;
    step5::units(M, N, false, UM, UN);
    const int nu = UM * UN, cl = blockIdx.x / 2, ncl = gridDim.x / 2;
    // the slices of the cluster's units, the same for both blocks
    const unsigned total = (unsigned)((nu - cl + ncl - 1) / ncl) * (unsigned)nt;
    auto tile = [&](int u, int& row0, int& col0) {   // the block's tile in unit u
        int um, un;
        step5::place(u, UM, UN, step5::kGroupRows, true, um, un);
        row0 = um * 128;
        col0 = (2 * un + rank) * 256;
    };
    if (tid == 0) {
        for (int s = 0; s < kStages; s++) {
            step2::mbar_init(&full[s], 1);
            step2::mbar_init(&empty[s], kShare ? 4 : 2);   // the four computing warpgroups of the cluster (7h)
        }
        step2::mbar_fence_init();
    }
    __syncthreads();
    if (wg == 0) {   // the loading warpgroup: warp 0 loads, warps 1 to 3 leave
        setmaxnreg_dec<24>();
        step4::cluster_arrive_relaxed();   // the cluster's arrival and wait in one piece, before the first access
        step4::cluster_wait();             // to the other block
        if (tid >= 32) return;
        unsigned it = 0;   // the slices so far, over the cluster's units
        for (int u = cl; u < nu; u += ncl) {
            int row0, col0;
            tile(u, row0, col0);
            for (int t = 0; t < nt; t++, it++) {
                const int s = it % kStages;
                if (it >= kStages) step2::mbar_wait(&empty[s], ((it / kStages) - 1) & 1);   // released by them all
                if (step3::elect_one()) {
                    uint8_t* st = sm + s * G::kStage;
                    step2::mbar_arrive_expect_tx(&full[s], G::kStage);
                    if constexpr (kShare)   // 64 of A's 128 rows for both blocks
                        tma_load_2d_multicast(st + rank * 8192, &tmA, t * 64, row0 + 64 * rank, &full[s], 3);
                    else   // all of its own A
                        step2::tma_load_2d(st, &tmA, t * 64, row0, &full[s]);
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        step2::tma_load_2d(st + G::kABytes + j * 8192, &tmB, col0 + 64 * j, t * 64, &full[s]);
                }
            }
        }
        if constexpr (kShare && kTail)   // 7i: the last 4 stages released by the four computing warpgroups
            for (unsigned p = total > kStages ? total - kStages : 0; p < total; p++)
                step2::mbar_wait(&empty[p % kStages], (p / kStages) & 1);
        return;
    }
    // the computing warpgroups: rows 64 h to 64 h + 63 of each tile
    const int h = kUni ? __shfl_sync(0xffffffffu, wg - 1, 0) : wg - 1;
    const uint32_t s0 = step1::smem_addr(sm), emptyOther = kShare ? mapa(step1::smem_addr(empty), (uint32_t)other) : 0;
    const uint64_t dA0 = step1::desc_a(s0 + h * 64 * 128), dB0 = step1::desc_b(s0 + G::kABytes);
    const int w = (tid / 32) % 4, l = tid % 32, ct = tid % 128;
    const uint32_t zone = s0 + G::kZones + h * G::kZone;   // the warpgroup's output zone
    uint8_t* zp = sm + G::kZones + h * G::kZone;
    float acc[128];   // not zeroed: the first wgmma of each tile overwrites it (scaleD 0)
    step4::cluster_arrive_relaxed();
    step4::cluster_wait();
    auto release = [&](unsigned p) {   // slice p's stage is free here, and for the other block's loads too (7h)
        if (ct == 0) {
            step2::mbar_arrive(&empty[p % kStages]);
            if constexpr (kShare) mbar_arrive_cluster(emptyOther + 8 * (p % kStages));
        }
    };
    unsigned it = 0;
    for (int u = cl; u < nu; u += ncl) {
        int row0, col0;
        tile(u, row0, col0);
        // 7g: the stage of the tile's first slice, broadcast by a shuffle as the warpgroup's index: the descriptors'
        // offsets stay in the uniform datapath (ptxas keeps the counter of slices across tiles in vector registers,
        // and would rebuild the descriptors there)
        const int first = kUni ? __shfl_sync(0xffffffffu, (int)(it % kStages), 0) : 0;
        for (int t = 0; t < nt; t++, it++) {
            const int s = it % kStages;
            step2::mbar_wait(&full[s], (it / kStages) & 1);
            const uint32_t ds = ((first + t) % kStages) * (G::kStage >> 4);
            const uint32_t st = s0 + s * G::kStage;   // without 7g: the stage's address
            step1::wgmma_fence();
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                const uint64_t da = kUni ? dA0 + ds + 2 * kk : step1::desc_a(st + h * 64 * 128) + 2 * kk;
                const uint64_t db = kUni ? dB0 + ds + 128 * kk : step1::desc_b(st + G::kABytes) + 128 * kk;
                step1::wgmma<256>(acc, da, db, t > 0 || kk > 0);
            }
            step1::wgmma_commit();
            step1::wgmma_wait<1>();
            if (t >= 1) release(it - 1);
        }
        step1::wgmma_wait<0>();
        if (!kShare || kTail || u + ncl < nu) release(it - 1);   // the tile's last slice (without 7i, the cluster's
                                                                  // last one is waited for by no one)
        // the output: two passes of 128 columns through the warpgroup's zone
        const int r0 = row0 + 64 * h;
#pragma unroll
        for (int q = 0; q < 2; q++) {
            if (kTma && ct == 0) step5::bulk_wait<0, true>();   // the zone read by the previous pass's stores
            step3::bar_named(2 + h, 128);   // the zone is free
            step5::zone_store(zp, acc, q, w, l);
            if constexpr (kTma) step1::fence_proxy_async();   // the zone, visible to the TMA
            step3::bar_named(2 + h, 128);
            const int c0 = col0 + 128 * q;
            if constexpr (kTma) {
                if (ct == 0) {   // a block after the last column of tiles writes nothing
                    if (r0 < M && col0 < N) {
                        step5::tma_store_2d(&tmC, c0, r0, zone);
                        step5::tma_store_2d(&tmC, c0 + 64, r0, zone + 8192);
                    }
                    step5::bulk_commit();
                }
            } else if (col0 < N) {   // 16-byte stores by the warpgroup's 128 threads, rows >= M left out
                step5::zone_copy(zp, C, M, N, r0, c0, ct);
            }
        }
    }
    if constexpr (kShare && !kTail) {   // without 7i: a cluster barrier before leaving, both blocks done with stages
        cluster_arrive();
        step4::cluster_wait();
    }
    if (kTma && ct == 0) step5::bulk_wait<0, false>();   // the stores done: their source is this block's shared memory
}

// ---------------------------------------------------------------- the persistent kernel with the roles
// wgmma m64n64 and m64n192 are step 6's (step6::wgmma). A tile of BM x BN in NS stages of K64. BM = 64: two computing
// warpgroups on the same 64 rows, each BN / 2 columns; BM = 128: each 64 rows, all BN columns. The output by passes of
// 128 columns at most per warpgroup (boxes of 64 x 64), into the warpgroups' output zones after the stages.
template <int BM, int BN, int NS>
struct Tile {
    static constexpr bool kSplitN = BM == 64;
    static constexpr int kWgCols = kSplitN ? BN / 2 : BN;           // a computing warpgroup's columns
    static constexpr int kAcc = kWgCols / 2;                        // its accumulators per thread
    static constexpr int kBoxes = BN / 64;                          // B's boxes of 64 columns in a stage
    static constexpr int kABytes = BM * 128, kStage = kABytes + kBoxes * 8192;
    static constexpr int kPassCols = kWgCols < 128 ? kWgCols : 128, kPasses = (kWgCols + 127) / 128;
    static constexpr int kZone = kPassCols / 64 * 8192;             // a warpgroup's output zone
    static constexpr int kZones = NS * kStage, kBars = kZones + 2 * kZone;
    static constexpr int kSmemBytes = kBars + 16 * NS + 1024;       // + the alignment of the stages to 1024 bytes
    static_assert(kSmemBytes <= 227 * 1024, "227 KB of shared memory at most");
};

// Step 5's persistent kernel (5a, 5b) with the roles (7f) at a tile of step 6 (64 x 128, 128 x 128, 6c's 128 x 256) or
// 7e's, and a cluster (step6::Pair: none, or two tiles one above the other). kInc: the computing warpgroups' registers
// raised by setmaxnreg.inc (0: kept). groupRows: the rows of units per group of 5b's order. kTma: 7j, else
// the zone copied out by 16-byte stores of the warpgroup's threads; kShare (in pairs): 7h, B shared, else each block
// loads its own A and B and releases its stages to itself; kTail (with kShare): 7i, else no last release and no last
// wait of the loading warp, a cluster barrier before leaving; kUni: 7g, else the descriptors rebuilt from the stage's
// address at each slice.
template <int BM, int BN, int NS, step6::Pair kPair, int kInc, bool kTma, bool kShare, bool kTail, bool kUni>
__device__ __forceinline__ void gemm_body(const CUtensorMap& tmA, const CUtensorMap& tmB, const CUtensorMap& tmC,
                                          half* C, int M, int N, int K, int groupRows)
{
    using T = Tile<BM, BN, NS>;
    constexpr bool kCl = kPair != step6::kSolo, kSh = kCl && kShare;   // kSh: the pair shares B
    uint8_t* shmem = step4::shared_base();
    uint8_t* sm = shmem + ((1024 - (step1::smem_addr(shmem) & 1023)) & 1023);
    uint64_t* full = (uint64_t*)(sm + T::kBars);
    uint64_t* empty = full + NS;
    const int tid = threadIdx.x, wg = tid / 128, nt = K / 64;
    const int rank = kCl ? (int)step4::cluster_rank() : 0, other = rank ^ 1;
    int UM, UN;
    step6::units6(M, N, BM, BN, kPair, UM, UN);
    const int nu = UM * UN, cl = kCl ? blockIdx.x / 2 : blockIdx.x, ncl = kCl ? gridDim.x / 2 : gridDim.x;
    const unsigned total = (unsigned)((nu - cl + ncl - 1) / ncl) * (unsigned)nt;
    auto tile = [&](int u, int& row0, int& col0) {
        int um, un;
        step5::place(u, UM, UN, groupRows, true, um, un);
        row0 = (kPair == step6::kAlongM ? 2 * um + rank : um) * BM;
        col0 = un * BN;
    };
    if (tid == 0) {
        for (int s = 0; s < NS; s++) {
            step2::mbar_init(&full[s], 1);
            step2::mbar_init(&empty[s], kSh ? 4 : 2);
        }
        step2::mbar_fence_init();
    }
    __syncthreads();
    if (wg == 0) {   // the loading warpgroup: warp 0 loads, warps 1 to 3 leave
        setmaxnreg_dec<24>();
        if constexpr (kCl) {   // the cluster's arrival and wait in one piece, before the first access to the other
            step4::cluster_arrive_relaxed();   // block
            step4::cluster_wait();
        }
        if (tid >= 32) return;
        unsigned it = 0;
        for (int u = cl; u < nu; u += ncl) {
            int row0, col0;
            tile(u, row0, col0);
            for (int t = 0; t < nt; t++, it++) {
                const int s = it % NS;
                if (it >= NS) step2::mbar_wait(&empty[s], ((it / NS) - 1) & 1);
                if (step3::elect_one()) {
                    uint8_t* st = sm + s * T::kStage;
                    step2::mbar_arrive_expect_tx(&full[s], T::kStage);
                    step2::tma_load_2d(st, &tmA, t * 64, row0, &full[s]);
#pragma unroll
                    for (int j = 0; j < T::kBoxes; j++) {
                        if constexpr (kSh) {   // B's first boxes from rank 0, the others from rank 1
                            if ((j < (T::kBoxes + 1) / 2) == (rank == 0))
                                tma_load_2d_multicast(st + T::kABytes + j * 8192, &tmB, col0 + 64 * j, t * 64,
                                                      &full[s], 3);
                        } else {
                            step2::tma_load_2d(st + T::kABytes + j * 8192, &tmB, col0 + 64 * j, t * 64, &full[s]);
                        }
                    }
                }
            }
        }
        if constexpr (kSh && kTail)   // 7i: the last stages released by the four computing warpgroups
            for (unsigned p = total > NS ? total - NS : 0; p < total; p++)
                step2::mbar_wait(&empty[p % NS], (p / NS) & 1);
        return;
    }
    if constexpr (kInc > 0) setmaxnreg_inc<kInc>();
    const int h = kUni ? __shfl_sync(0xffffffffu, wg - 1, 0) : wg - 1;
    const uint32_t s0 = step1::smem_addr(sm);
    const uint32_t emptyOther = kSh ? mapa(step1::smem_addr(empty), (uint32_t)other) : 0;
    const uint64_t dA0 = step1::desc_a(s0 + (T::kSplitN ? 0 : h * 64 * 128));
    const uint64_t dB0 = step1::desc_b(s0 + T::kABytes + (T::kSplitN ? h * (T::kWgCols / 64) * 8192 : 0));
    const int w = (tid / 32) % 4, l = tid % 32, ct = tid % 128;
    const uint32_t zone = s0 + T::kZones + h * T::kZone;   // the warpgroup's output zone
    float acc[T::kAcc];   // not zeroed: the first wgmma of each tile overwrites it (scaleD 0)
    if constexpr (kCl) {
        step4::cluster_arrive_relaxed();
        step4::cluster_wait();
    }
    auto release = [&](unsigned p) {   // slice p's stage is free here, and for the other block's loads too (7h)
        if (ct == 0) {
            step2::mbar_arrive(&empty[p % NS]);
            if constexpr (kSh) mbar_arrive_cluster(emptyOther + 8 * (p % NS));
        }
    };
    unsigned it = 0;
    for (int u = cl; u < nu; u += ncl) {
        int row0, col0;
        tile(u, row0, col0);
        for (int t = 0; t < nt; t++, it++) {
            const int s = it % NS;
            step2::mbar_wait(&full[s], (it / NS) & 1);
            // 7g: the stage broadcast by a shuffle at each slice (step 5's tile broadcasts the tile's first stage;
            // here, with or without a cluster, this keeps the descriptors in the uniform datapath)
            const uint32_t ds = kUni ? __shfl_sync(0xffffffffu, (int)(it % NS), 0) * (T::kStage >> 4) : 0;
            const uint32_t st = s0 + s * T::kStage;   // without 7g: the stage's address
            step1::wgmma_fence();
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                const uint32_t ra = st + (T::kSplitN ? 0 : h * 64 * 128);
                const uint32_t rb = st + T::kABytes + (T::kSplitN ? h * (T::kWgCols / 64) * 8192 : 0);
                const uint64_t da = kUni ? dA0 + ds + 2 * kk : step1::desc_a(ra) + 2 * kk;
                const uint64_t db = kUni ? dB0 + ds + 128 * kk : step1::desc_b(rb) + 128 * kk;
                if constexpr (T::kWgCols == 64) {
                    step6::wgmma<64>(*reinterpret_cast<float(*)[32]>(acc), da, db, t > 0 || kk > 0);
                } else if constexpr (T::kWgCols == 128) {
                    step1::wgmma<128>(*reinterpret_cast<float(*)[64]>(acc), da, db, t > 0 || kk > 0);
                } else if constexpr (T::kWgCols == 256) {   // 6c
                    step1::wgmma<256>(*reinterpret_cast<float(*)[128]>(acc), da, db, t > 0 || kk > 0);
                } else {   // 320: n192 over B's boxes 0 to 2, then n128 over boxes 3 and 4
                    step6::wgmma<192>(*reinterpret_cast<float(*)[96]>(acc), da, db, t > 0 || kk > 0);
                    step1::wgmma<128>(*reinterpret_cast<float(*)[64]>(acc + 96), da, db + 3 * 512, t > 0 || kk > 0);
                }
            }
            step1::wgmma_commit();
            step1::wgmma_wait<1>();
            if (t >= 1) release(it - 1);
        }
        step1::wgmma_wait<0>();
        // the tile's last slice (without 7i, the cluster's last one is waited for by no one)
        const bool last = !kSh || kTail || u + ncl < nu;
        if (last) release(it - 1);
        // the output, as step 5's tile's, by passes of up to 128 of the warpgroup's columns: each lane's pairs of
        // halves at their place in the boxes of 64 x 64 at the 128-byte swizzle, rows rg and rg + 8 (the same row % 8)
        const int rg = 16 * w + l / 4;
        const int r0 = row0 + (T::kSplitN ? 0 : 64 * h), c0w = col0 + (T::kSplitN ? h * T::kWgCols : 0);
#pragma unroll
        for (int q = 0; q < T::kPasses; q++) {
            constexpr int kFull = T::kPassCols;
            const int qc = T::kWgCols - 128 * q < kFull ? T::kWgCols - 128 * q : kFull;   // the pass's columns
            if (kTma && ct == 0) step5::bulk_wait<0, true>();   // the zone read by the previous pass's stores
            step3::bar_named(2 + h, 128);
            uint8_t* zw = sm + (zone - s0);
#pragma unroll
            for (int c = 0; c < kFull / 8; c++) {
                if (8 * c >= qc) break;
                const int j = 16 * q + c;
                uint8_t* d = zw + 8192 * (c / 8) + (((c % 8) ^ (rg % 8)) << 4) + 4 * (l % 4);
                *(__half2*)(d + rg * 128) = __floats2half2_rn(acc[4 * j], acc[4 * j + 1]);
                *(__half2*)(d + 128 * (rg + 8)) = __floats2half2_rn(acc[4 * j + 2], acc[4 * j + 3]);
            }
            if constexpr (kTma) step1::fence_proxy_async();
            step3::bar_named(2 + h, 128);
            if constexpr (kTma) {
                if (ct == 0) {   // boxes outside M or N are left out (a box partly outside is cut by the TMA)
                    if (r0 < M)
                        for (int b = 0; b < qc / 64; b++)
                            if (c0w + 128 * q + 64 * b < N)
                                step5::tma_store_2d(&tmC, c0w + 128 * q + 64 * b, r0, zone + b * 8192);
                    step5::bulk_commit();
                }
            } else {   // 16-byte stores by the warpgroup's 128 threads, rows >= M and boxes >= N left out: the threads
                       // split between the pass's boxes of 64 columns (a warp in one box), thread t of a box copying
                       // chunk t % 8 of rows t / 8, t / 8 + step, ..., in the box's swizzle
                const int nb = qc / 64, tpb = 128 / nb, b = ct / tpb, t = ct % tpb, c8 = t % 8, rr = t / 8;
                const int step = tpb / 8, col = c0w + 128 * q + 64 * b;
                const uint8_t* src = sm + (zone - s0) + b * 8192 + ((c8 ^ (rr % 8)) << 4);
                if (col < N) {
#pragma unroll
                    for (int i = 0; i < 64 / step; i++) {
                        const int r = rr + i * step;   // step a multiple of 8: r % 8 = rr % 8, the same swizzle
                        if (r0 + r < M)
                            *(int4*)(C + (size_t)(r0 + r) * N + col + 8 * c8) = *(const int4*)(src + r * 128);
                    }
                }
            }
        }
    }
    if constexpr (kSh && !kTail) {   // without 7i: a cluster barrier before leaving, both blocks done with stages
        cluster_arrive();
        step4::cluster_wait();
    }
    if (kTma && ct == 0) step5::bulk_wait<0, false>();   // the stores done: their source is this block's shared memory
}
template <int BM, int BN, int NS, step6::Pair kPair, int kInc, bool kTma, bool kShare, bool kTail, bool kUni>
__global__ void __launch_bounds__(384, 1) gemm(const __grid_constant__ CUtensorMap tmA,
                                               const __grid_constant__ CUtensorMap tmB,
                                               const __grid_constant__ CUtensorMap tmC, half* C, int M, int N, int K)
{
    gemm_body<BM, BN, NS, kPair, kInc, kTma, kShare, kTail, kUni>(
        tmA, tmB, tmC, C, M, N, K, kPair == step6::kAlongM ? step5::kGroupRows / 2 : step5::kGroupRows);
}
// 7c: the same kernel alone, the rows of its groups given at the launch.
template <int BM, int BN, int NS, int kInc, bool kTma, bool kUni>
__global__ void __launch_bounds__(384, 1) gemm_rows(const __grid_constant__ CUtensorMap tmA,
                                                    const __grid_constant__ CUtensorMap tmB,
                                                    const __grid_constant__ CUtensorMap tmC, half* C, int M, int N,
                                                    int K, int groupRows)
{
    gemm_body<BM, BN, NS, step6::kSolo, kInc, kTma, false, false, kUni>(tmA, tmB, tmC, C, M, N, K, groupRows);
}
// 7c without the roles (v8): step 6's persistent kernel alone (step6::gemm_body), the rows of its groups given at the
// launch.
template <int BM, int BN, int NS>
__global__ void __launch_bounds__(256, 1) gemm_rows_noroles(const __grid_constant__ CUtensorMap tmA,
                                                            const __grid_constant__ CUtensorMap tmB, half* C, int M,
                                                            int N, int K, int groupRows)
{
    step6::gemm_body<BM, BN, NS, step6::kSolo, true>(tmA, tmB, C, M, N, K, groupRows);
}

// ---------------------------------------------------------------- the bands (7a, 7b, 7d)
// kSolo: one block per band (16 or 128 rows); kAlongN: 16-row bands in clusters of eight along N (7b); kAlongM: 64-row
// bands in clusters of two along M (7a: the two blocks of a cluster take the same columns, rows 0 to 63 and 64 to 127).
enum Mode { kSolo, kAlongN, kAlongM };

// The blocks of a cluster of the bands (kAlongN: 8; kAlongM: 2).
__host__ __device__ constexpr int clusterBlocks(Mode P)
{
    return P == kSolo ? 1 : P == kAlongM ? 2 : 8;
}

// The bands' geometry: MM rows by J columns of 64, NS stages (A's MM rows x 64 K, then J boxes of B), the output zone
// (MM rows x 64 columns per box), the barriers.
template <int MM, int J, int NS>
struct Bands {
    static constexpr int kABytes = MM * 128, kStage = kABytes + J * 8192, kBox = MM * 128;
    static constexpr int kZone = NS * kStage, kBars = kZone + J * kBox;
    static constexpr int kSmemBytes = kBars + 16 * NS + 1024;   // + the alignment of the stages to 1024 bytes
    static_assert(kSmemBytes <= 227 * 1024, "227 KB of shared memory at most");
};
// The most stages that shared memory holds, 8 at most.
template <int MM, int J>
constexpr int bandStages()
{
    int ns = 8;
    while (ns > 1 && Bands<MM, J, 1>::kStage * ns + J * MM * 128 + 16 * ns + 1024 > 227 * 1024) ns--;
    return ns;
}

// The bands' shapes: 16 rows alone or in clusters of eight along N, 64 rows in pairs along M, 128 rows alone.
template <int MM, Mode P>
__host__ __device__ constexpr bool bandsShape()
{
    return P == kAlongN ? MM == 16 : P == kAlongM ? MM == 64 : MM == 16 || MM == 128;
}

// The bands with the roles (7f): one block of 160 threads per band of MM rows x J columns of 64, all of K (6a's kernel,
// at MM rows): a loading warp (threads 128 to 159) and a computing warpgroup. kHint: 7l; kShare (in clusters): 7h, A
// (7b) or the pair's B (7a) shared, else each block loads its own A and B and releases its stages to itself; kTail
// (with kShare): 7i, else no last wait of the loading warp (nor, at 8 stages, the last release), a cluster barrier
// before leaving; kUni: 7g, else the stage not broadcast.
template <int MM, int J, int NS, Mode P, bool kHint, bool kShare, bool kTail, bool kUni>
__global__ void __launch_bounds__(160, 1) bands(const __grid_constant__ CUtensorMap tmA,
                                                const __grid_constant__ CUtensorMap tmB,
                                                const __grid_constant__ CUtensorMap tmC, half* C, int M, int N, int K)
{
    using G = Bands<MM, J, NS>;
    constexpr bool kZ = MM == 16 && NS < 8;   // a stage released once its MMA are retired (wait_group 0)
    constexpr bool kPair = P != kSolo;
    constexpr bool kSh = kPair && kShare;   // the cluster shares an operand
    constexpr int kC = clusterBlocks(P);   // the blocks of a cluster
    constexpr int kR = kSh ? kC : 1;   // the blocks that release a stage
    static_assert(bandsShape<MM, P>(), "16 rows alone or in clusters of eight, 64 in pairs along M, 128 alone");
    static_assert(P != kAlongM || J % 2 == 0, "pairs along M: J even");
    uint8_t* shmem = step4::shared_base();
    uint8_t* sm = shmem + ((1024 - (step1::smem_addr(shmem) & 1023)) & 1023);
    uint64_t* full = (uint64_t*)(sm + G::kBars);
    uint64_t* empty = full + NS;
    const int tid = threadIdx.x, nt = K / 64;
    const int rank = kPair ? (int)step4::cluster_rank() : 0;
    const int band = P == kAlongM ? blockIdx.x / 2 : blockIdx.x;
    const int col0 = band * 64 * J, row0 = P == kAlongM ? 64 * rank : 0;
    if (tid == 0) {
        for (int s = 0; s < NS; s++) {
            step2::mbar_init(&full[s], 1);
            step2::mbar_init(&empty[s], kR);
        }
        step2::mbar_fence_init();
    }
    __syncthreads();
    if (tid >= 128) {   // the loading warp
        if constexpr (kPair) {   // the cluster's arrival and wait in one piece, before the first access to another
            step4::cluster_arrive_relaxed();   // block
            step4::cluster_wait();
        }
        const uint64_t polB = kHint ? policy_evict_first() : 0;
        for (int t = 0; t < nt; t++) {
            const int s = t % NS;
            if (t >= NS) step2::mbar_wait(&empty[s], ((t / NS) - 1) & 1);
            if (step3::elect_one()) {
                uint8_t* st = sm + s * G::kStage;
                step2::mbar_arrive_expect_tx(&full[s], G::kStage);
                if constexpr (kSh && P == kAlongN) {   // all of A by the first block, for the cluster's eight
                    if (rank == 0) tma_load_2d_multicast(st, &tmA, t * 64, 0, &full[s], 0xff);
                } else {
                    step2::tma_load_2d(st, &tmA, t * 64, row0, &full[s]);
                }
#pragma unroll
                for (int j = 0; j < J; j++) {
                    if constexpr (kSh && P == kAlongM) {   // half of B's boxes each, for both blocks
                        if (j % 2 == rank)
                            load_multicast<kHint>(st + G::kABytes + j * 8192, &tmB, col0 + 64 * j, t * 64, &full[s], 3,
                                                  polB);
                    } else {
                        load<kHint>(st + G::kABytes + j * 8192, &tmB, col0 + 64 * j, t * 64, &full[s], polB);
                    }
                }
            }
        }
        if constexpr (kSh && kTail)   // 7i: the last stages released by the cluster's computing warpgroups
            for (int p = nt > NS ? nt - NS : 0; p < nt; p++) step2::mbar_wait(&empty[p % NS], (p / NS) & 1);
        return;
    }
    // the computing warpgroup: J ranges of the transposed MMA (C^T's J x 64 rows, our MM rows)
    const uint32_t s0 = step1::smem_addr(sm);
    uint32_t emptyOther[kR > 1 ? kR - 1 : 1];   // the other blocks' empty barriers
    for (int r = 1; r < kR; r++) emptyOther[r - 1] = mapa(step1::smem_addr(empty), (uint32_t)((rank + r) % kC));
    const uint64_t dB0 = step1::desc_b(s0 + G::kABytes), dA0 = step1::desc_a(s0);   // our B: the MMA's A; our A: its B
    float acc[J][MM / 2];   // not zeroed: the first wgmma of each range overwrites it (scaleD 0)
    if constexpr (kPair) {
        step4::cluster_arrive_relaxed();
        step4::cluster_wait();
    }
    auto release = [&](int p) {   // slice p's stage is free here, and for the other blocks' loads too (7h)
        if (tid == 0) {
            step2::mbar_arrive(&empty[p % NS]);
#pragma unroll
            for (int r = 0; r < kR - 1; r++) mbar_arrive_cluster(emptyOther[r] + 8 * (p % NS));
        }
    };
    for (int t = 0; t < nt; t++) {
        const int s = t % NS;
        step2::mbar_wait(&full[s], (t / NS) & 1);
        // 7g: the stage broadcast by a shuffle: the descriptors' offsets stay in the uniform datapath
        const uint32_t ds = (kUni ? __shfl_sync(0xffffffffu, s, 0) : s) * (G::kStage >> 4);
        step1::wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
#pragma unroll
            for (int j = 0; j < J; j++)   // the MMA's A advances 2 KB per K16 (16 rows of K), its B 32 bytes
                wgmma_t<MM>(acc[j], dB0 + ds + 512 * j + 128 * kk, dA0 + ds + 2 * kk, t > 0 || kk > 0);
        step1::wgmma_commit();
        if constexpr (kZ) {
            step1::wgmma_wait<0>();
            release(t);
        } else {
            step1::wgmma_wait<1>();
            if (t >= 1) release(t - 1);
        }
    }
    step1::wgmma_wait<0>();
    if (!kZ && kSh && kTail) release(nt - 1);   // 7i: the last stage too, for the other blocks' loading warps
    // the output: each warp's 16 rows of C^T (columns 16 w to 16 w + 15 of a box of 64), stored transposed, 16 of our
    // rows at a time (rows 16 p to 16 p + 15): lane l gives the address of row 16 p + 8 (l / 16) + l % 8 of the band,
    // chunk 2 w + (l / 8) % 2 of the box's 8 chunks of 16 bytes
    const int w = tid / 32, l = tid % 32, m = 8 * (l / 16) + l % 8, c = 2 * w + (l / 8) % 2;
    const uint32_t zone = s0 + G::kZone;
#pragma unroll
    for (int j = 0; j < J; j++)
#pragma unroll
        for (int p = 0; p < MM / 16; p++)
            step6::stmatrix_x4_trans(zone + j * G::kBox + (16 * p + m) * 128 + ((c ^ (m % 8)) << 4),
                                     step5::pack_half2(acc[j][8 * p], acc[j][8 * p + 1]),
                                     step5::pack_half2(acc[j][8 * p + 2], acc[j][8 * p + 3]),
                                     step5::pack_half2(acc[j][8 * p + 4], acc[j][8 * p + 5]),
                                     step5::pack_half2(acc[j][8 * p + 6], acc[j][8 * p + 7]));
    step1::fence_proxy_async();   // the zone, visible to the TMA
    step3::bar_named(1, 128);
    if (tid == 0) {
#pragma unroll
        for (int j = 0; j < J; j++)   // rows >= M and columns >= N left out by the TMA
            if (col0 + 64 * j < N && row0 < M) step5::tma_store_2d(&tmC, col0 + 64 * j, row0, zone + j * G::kBox);
        step5::bulk_commit();
        step5::bulk_wait<0, false>();
    }
    if constexpr (kSh && !kTail) {   // without 7i: a cluster barrier before leaving, the cluster done with stages
        cluster_arrive();
        step4::cluster_wait();
    }
}

// The output of the bands without the roles: each warp's 16 rows of C^T (columns 16 w to 16 w + 15 of a box of 64),
// stored transposed, 16 of our rows at a time (rows 16 p to 16 p + 15): lane l gives the address of row 16 p + 8 (l /
// 16) + l % 8 of the band, chunk 2 w + (l / 8) % 2 of the box's 8 chunks of 16 bytes; then the TMA stores by thread 0,
// rows >= M and columns >= N left out.
template <int MM, int J, int NS>
__device__ __forceinline__ void bands_output(float (&acc)[J][MM / 2], uint32_t s0, int tid, int col0, int row0,
                                             const CUtensorMap* tmC, int M, int N)
{
    using G = Bands<MM, J, NS>;
    const int w = tid / 32, l = tid % 32, m = 8 * (l / 16) + l % 8, c = 2 * w + (l / 8) % 2;
    const uint32_t zone = s0 + G::kZone;
#pragma unroll
    for (int j = 0; j < J; j++)
#pragma unroll
        for (int p = 0; p < MM / 16; p++)
            step6::stmatrix_x4_trans(zone + j * G::kBox + (16 * p + m) * 128 + ((c ^ (m % 8)) << 4),
                                     step5::pack_half2(acc[j][8 * p], acc[j][8 * p + 1]),
                                     step5::pack_half2(acc[j][8 * p + 2], acc[j][8 * p + 3]),
                                     step5::pack_half2(acc[j][8 * p + 4], acc[j][8 * p + 5]),
                                     step5::pack_half2(acc[j][8 * p + 6], acc[j][8 * p + 7]));
    step1::fence_proxy_async();   // the zone, visible to the TMA
    step3::bar_named(1, 128);
    if (tid == 0) {
#pragma unroll
        for (int j = 0; j < J; j++)
            if (col0 + 64 * j < N && row0 < M) step5::tma_store_2d(tmC, col0 + 64 * j, row0, zone + j * G::kBox);
        step5::bulk_commit();
        step5::bulk_wait<0, false>();
    }
}

// The bands without the roles (v8): one block per band of MM rows x J columns of 64, all of K (6a's kernel, at MM
// rows), one warpgroup of 128 threads that computes, warp 0 issuing the loads from the computing path (as
// step6::bands): once wait_group 1 (0 for the 16-row bands below 8 stages) has retired a slice's MMA (the
// warpgroup's), its stage is free and one lane of warp 0 (elect.sync) issues the slice NS further on into it. No empty
// barrier: its only arrival would be thread 0's, in warp 0 itself, after that same wait_group, so it would tell warp 0
// nothing (its 8 bytes per stage stay in Bands' layout, unused). The whole warp takes the branch (its condition
// broadcast from lane 0). The descriptors' offsets from the loop's stage. In a cluster, each block loads its own A and
// B; the cluster synchronized in one piece. The output: bands_output.
template <int MM, int J, int NS, Mode P>
__device__ __forceinline__ void bands_noroles_body(const CUtensorMap& tmA, const CUtensorMap& tmB,
                                                   const CUtensorMap& tmC, half* C, int M, int N, int K)
{
    using G = Bands<MM, J, NS>;
    constexpr bool kZ = MM == 16 && NS < 8;   // a stage refilled once its own MMA are retired (wait_group 0)
    constexpr bool kPair = P != kSolo;
    static_assert(bandsShape<MM, P>(), "16 rows alone or in clusters of eight, 64 in pairs along M, 128 alone");
    uint8_t* shmem = step4::shared_base();
    uint8_t* sm = shmem + ((1024 - (step1::smem_addr(shmem) & 1023)) & 1023);
    uint64_t* full = (uint64_t*)(sm + G::kBars);
    const int tid = threadIdx.x, nt = K / 64;
    const int rank = kPair ? (int)step4::cluster_rank() : 0;
    const int band = (int)(blockIdx.x / (P == kAlongM ? 2u : 1u));
    const int col0 = band * 64 * J, row0 = P == kAlongM ? 64 * rank : 0;
    if (tid == 0) {
        for (int s = 0; s < NS; s++)
            step2::mbar_init(&full[s], 1);
        step2::mbar_fence_init();
    }
    __syncthreads();
    if constexpr (kPair) {   // the cluster synchronized in one piece
        step4::cluster_arrive_relaxed();
        step4::cluster_wait();
    }
    // warp 0's loads: li the slices issued so far
    const bool loader = __shfl_sync(0xffffffffu, tid, 0) < 32;   // the same in the whole warp, broadcast from lane 0
    int li = 0;
    auto load = [&]() {   // warp 0: the next slice into its stage, which the slice NS before it has left
        const int s = li % NS;
        if (step3::elect_one()) {
            uint8_t* st = sm + s * G::kStage;
            step2::mbar_arrive_expect_tx(&full[s], G::kStage);
            step2::tma_load_2d(st, &tmA, li * 64, row0, &full[s]);
#pragma unroll 1   // kept a loop, as step 6's: the computing path's descriptors stay in uniform registers
            for (int j = 0; j < J; j++)
                step2::tma_load_2d(st + G::kABytes + j * 8192, &tmB, col0 + 64 * j, li * 64, &full[s]);
        }
        li++;
    };
    if (loader)
        while (li < NS && li < nt) load();
    // the warpgroup: J ranges of the transposed MMA (C^T's J x 64 rows, our MM rows)
    const uint32_t s0 = step1::smem_addr(sm);
    const uint64_t dB0 = step1::desc_b(s0 + G::kABytes), dA0 = step1::desc_a(s0);   // our B: the MMA's A; our A: its B
    float acc[J][MM / 2];   // not zeroed: the first wgmma of each range overwrites it (scaleD 0)
    for (int t = 0; t < nt; t++) {
        const int s = t % NS;
        step2::mbar_wait(&full[s], (t / NS) & 1);
        const uint32_t ds = s * (G::kStage >> 4);   // the stage's offset in units of 16 bytes
        step1::wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
#pragma unroll
            for (int j = 0; j < J; j++)   // the MMA's A advances 2 KB per K16, its B 32 bytes
                wgmma_t<MM>(acc[j], dB0 + (512 * j + 128 * kk) + ds, dA0 + ds + 2 * kk, kk > 0 || t > 0);
        step1::wgmma_commit();
        if constexpr (kZ) step1::wgmma_wait<0>();   // the 16-row bands below 8 stages: the slice's own stage
        else step1::wgmma_wait<1>();                // else the stage of the slice before
        if ((kZ || t >= 1) && loader && li < nt) load();   // that stage free: warp 0 refills it
    }
    step1::wgmma_wait<0>();
    bands_output<MM, J, NS>(acc, s0, tid, col0, row0, &tmC, M, N);
}
template <int MM, int J, int NS, Mode P>
__global__ void __launch_bounds__(128, 1) bands_noroles(const __grid_constant__ CUtensorMap tmA,
                                                        const __grid_constant__ CUtensorMap tmB,
                                                        const __grid_constant__ CUtensorMap tmC, half* C, int M, int N,
                                                        int K)
{
    bands_noroles_body<MM, J, NS, P>(tmA, tmB, tmC, C, M, N, K);
}

// ---------------------------------------------------------------- the rule
// The kernels of step 7's rule.
enum Kind { kStep6, kBands16, kBands16Wide, kBands64Pairs, kBands128 };
struct Choice {
    Kind kind;
    int J;   // the bands' columns of 64 (kBands16Wide: step 6's J, doubled at the launch)
};
// 7c's rule: the rows G of 6c's groups and its grid R, for UM x UN tiles and sms blocks held at once.
inline void rows7c(int UM, int UN, int sms, int& G, int& R)
{
    const int tiles = UM * UN, rounds = (tiles + sms - 1) / sms, top = UM < 16 ? UM : 16;
    for (G = top < sms ? top : sms; G > 1; G--) {   // at most the blocks too: R holds a whole group
        R = G * (sms / G);
        if ((tiles + R - 1) / R == rounds) break;
    }
    if (G <= 1) {
        G = 1;
        R = sms;
    }
    if (R > tiles) R = tiles;
}
// 7a, 7b and 7d's classes; 7c and 7e are decided by the launch (step7.cu: step6::choose with the 128 x 320 tile).
// sms: the SM count; clusters8: the clusters of eight of 7b's kernel the GPU holds at once.
inline Choice choose(int M, int N, int sms, int clusters8)
{
    const int J16 = (N + 64 * sms - 1) / (64 * sms);
    if (M <= 16 && J16 <= 8) {
        const int J5 = 2 * J16 < 8 ? 2 * J16 : 8, bands5 = (N + 64 * J5 - 1) / (64 * J5);
        return {(bands5 + 7) / 8 <= (clusters8 + 1) / 2 ? kBands16Wide : kBands16, J16};   // a GPC for each cluster
    }
    const int bands2 = (N + 127) / 128;   // bands of two columns of 64
    if (M > 64 && M <= 128) {
        if (2 * bands2 <= sms) return {kBands64Pairs, 2};
        if (bands2 <= sms) return {kBands128, 2};
    }
    return {kStep6, 0};
}

}  // namespace step7
