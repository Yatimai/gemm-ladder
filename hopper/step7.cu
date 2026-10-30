// Step 7 of the Hopper rung, as a candidate for the judge: warp specialization, taller and wider bands, a 128 x 320
// tile (see step7.cuh). Its variants, for the interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar): v0 =
// step 7 (7a to 7l); v1 = without 7a (the 64-row pairs' class by step 6's choice); v2 = without 7b (the 16-row bands
// alone, at step 6's J); v3 = without 7d (the 128-row bands' class by step 6's choice); v4 = without 7e (6c where 7e
// would run, 128 x 128 where N is not a multiple of 256); v5 = 7e in pairs at every shape it takes (alone in one wave
// left out); v6 = step 6 itself; v7 = without 7c (6c in step 6's order, grid and maps); v8 = without 7f and the five
// that live on it (7g to 7l: step 7's classes on step 6's kernels, every warp computing and warp 0 also loading); v9 =
// 7f alone (warp specialization, the roles, without 7g to 7l); v10 = without 7g; v11 = without 7h (each block loads its
// own A and B and releases its stages to itself, so without 7i too); v12 = without 7i (a cluster barrier before
// leaving); v13 = without 7j (the tiles' zones copied out by 16-byte stores of the warpgroups' threads); v14 = without
// 7l (no L2 hint).
#include <cstdio>
#include <cstdlib>
#include "step7.cuh"

int g_var = 0;
int g_nvar = 15;

static int sms()   // the SM count, read once
{
    static int n = 0;
    if (!n) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
    return n;
}

static cudaLaunchConfig_t config(int blocks, int threads, int smem, bool cluster, cudaLaunchAttribute* attr,
                                 cudaStream_t s, int clusterBlocks = 2)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks);
    cfg.blockDim = dim3(threads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = s;
    attr[0].id = cudaLaunchAttributeClusterDimension;   // 4b: the cluster set at the launch
    attr[0].val.clusterDim.x = clusterBlocks;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = cluster ? 1 : 0;
    return cfg;
}

// How many clusters of clusterBlocks (or blocks, without clusters) of a kernel the GPU holds at once; asked once per
// kernel.
template <typename F>
static int resident(F kernel, int smem, int threads, bool cluster, int clusterBlocks = 2)
{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int n = 0;
    if (cluster) {
        cudaLaunchAttribute attr[1];
        const cudaLaunchConfig_t cfg = config(clusterBlocks, threads, smem, true, attr, 0, clusterBlocks);
        cudaOccupancyMaxActiveClusters(&n, kernel, &cfg);
    } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, threads, smem);
        n *= sms();
    }
    if (n < 1) {
        fprintf(stderr, "step7: the occupancy query failed\n");
        abort();
    }
    return n;
}

// ---------------------------------------------------------------- step 6's kernels (v6, and v8 at step 7's classes)
// Step 5's tile, in clusters of two side by side (step6::choose gives it at M <= 128 only, one row of tiles).
static void step6_launch5(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    auto kernel = step5::gemm<false, true>;
    static const int pairs = resident(kernel, step5::G::kSmemBytes, 256, true);
    const CUtensorMap a = step2::map_a(A, M, K), b = step2::map_b(B, K, N);
    cudaLaunchAttribute attr[1];
    const cudaLaunchConfig_t cfg = config(step5::blocks(M, N, false, pairs), 256, step5::G::kSmemBytes, true, attr, s);
    cudaLaunchKernelEx(&cfg, kernel, a, b, C, M, N, K);
}

// 6b, 6c and 7e without the roles: step 6's persistent kernel at a tile of BM x BN, NS stages, a cluster of two along
// M or none.
template <int BM, int BN, int NS, step6::Pair P>
static void step6_launch6(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    constexpr bool cl = P != step6::kSolo;
    auto kernel = step6::gemm<BM, BN, NS, P>;
    using T = step6::Tile<BM, BN, NS>;
    static const int res = resident(kernel, T::kSmemBytes, 256, cl);
    const CUtensorMap a = step2::make_map(A, K, M, 64, BM), b = step2::map_b(B, K, N);
    int UM, UN;
    step6::units6(M, N, BM, BN, P, UM, UN);
    const int units = UM * UN < res ? UM * UN : res;
    cudaLaunchAttribute attr[1];
    const cudaLaunchConfig_t cfg = config(cl ? 2 * units : units, 256, T::kSmemBytes, cl, attr, s);
    cudaLaunchKernelEx(&cfg, kernel, a, b, C, M, N, K);
}

// 6a: step 6's bands of J columns of 64.
template <int J>
static void step6_bands(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    constexpr int NS = step6::bandStages<J>();
    auto kernel = step6::bands<J, NS>;
    using G = step6::Bands<J, NS>;
    static bool allowed = false;
    if (!allowed) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, G::kSmemBytes);
        allowed = true;
    }
    const CUtensorMap a = step6::map_a16(A, M, K), b = step2::map_b(B, K, N), c = step6::map_c16(C, M, N);
    cudaLaunchAttribute attr[1];
    const cudaLaunchConfig_t cfg = config((N + 64 * J - 1) / (64 * J), 128, G::kSmemBytes, false, attr, s);
    cudaLaunchKernelEx(&cfg, kernel, a, b, c, C, M, N, K);
}

// The clusters of two of step 5's kernel the GPU holds at once (the rule's unit of a wave), as step 6's.
static int step6_pairs()
{
    static const int n = resident(step5::gemm<true, true>, step5::G::kSmemBytes, 256, true);
    return n;
}

// Step 6 itself (v6): its choice and its kernels.
static void step6_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    const step6::Choice c = step6::choose(M, N, sms(), step6_pairs(), false);
    switch (c.kind) {
    case step6::kBandsK:
        switch (c.J) {
        case 1: step6_bands<1>(A, B, C, M, N, K, s); break;
        case 2: step6_bands<2>(A, B, C, M, N, K, s); break;
        case 3: step6_bands<3>(A, B, C, M, N, K, s); break;
        case 4: step6_bands<4>(A, B, C, M, N, K, s); break;
        case 5: step6_bands<5>(A, B, C, M, N, K, s); break;
        case 6: step6_bands<6>(A, B, C, M, N, K, s); break;
        case 7: step6_bands<7>(A, B, C, M, N, K, s); break;
        default: step6_bands<8>(A, B, C, M, N, K, s); break;
        }
        break;
    case step6::kT64Pair: step6_launch6<64, 128, 8, step6::kAlongM>(A, B, C, M, N, K, s); break;
    case step6::kT64Solo: step6_launch6<64, 128, 8, step6::kSolo>(A, B, C, M, N, K, s); break;
    case step6::kT128: step6_launch6<128, 128, 6, step6::kSolo>(A, B, C, M, N, K, s); break;
    case step6::kT256Solo: step6_launch6<128, 256, 4, step6::kSolo>(A, B, C, M, N, K, s); break;
    case step6::kT256: step6_launch5(A, B, C, M, N, K, s); break;   // M <= 128 (step6::choose)
    default:   // 7e's tiles: not in step 6's choice
        fprintf(stderr, "step7 v6: no 7e in step 6\n");
        abort();
    }
}

// 7c without the roles (v8): step 6's 6c, the rows of its groups and its grid from 7c's rule, its maps' L2 promotion
// of 256 bytes in one wave.
static void step6_launch6c_rows(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    auto kernel = step7::gemm_rows_noroles<128, 256, 4>;
    using T = step6::Tile<128, 256, 4>;
    static const int res = resident(kernel, T::kSmemBytes, 256, false);
    int UM, UN, G, R;
    step6::units6(M, N, 128, 256, step6::kSolo, UM, UN);
    step7::rows7c(UM, UN, res, G, R);
    const bool oneWave = UM * UN <= res;   // in one wave, A's and B's maps at the L2 promotion of 256 bytes
    const CUtensorMap a = step2::make_map(A, K, M, 64, 128, oneWave);
    const CUtensorMap b = step2::make_map(B, N, K, 64, 64, oneWave);
    cudaLaunchAttribute attr[1];
    const cudaLaunchConfig_t cfg = config(R, 256, T::kSmemBytes, false, attr, s);
    cudaLaunchKernelEx(&cfg, kernel, a, b, C, M, N, K, G);
}

// ---------------------------------------------------------------- step 7's switches
// The switches of a variant's kernels: 7f (the roles), 7g, 7h, 7i (with 7h only), 7j, 7l.
template <bool F, bool G, bool H, bool I, bool J, bool L>
struct Arm {
    static constexpr bool f = F, g = G, h = H, i = H && I, j = J, l = L;
};
using All = Arm<true, true, true, true, true, true>;   // step 7's

// ---------------------------------------------------------------- step 7's tiles
// Step 5's tile in clusters of two side by side, with the roles and W's switches (7h: A in boxes of 64 rows).
template <class W>
static void launch5(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    auto kernel = step7::gemm5<W::j, W::h, W::i, W::g>;
    static const int pairs = resident(kernel, step5::G::kSmemBytes, 384, true);
    const CUtensorMap a = W::h ? step7::map_a64(A, M, K) : step2::map_a(A, M, K);
    const CUtensorMap b = step2::map_b(B, K, N), c = step7::map_c(C, M, N);
    cudaLaunchAttribute attr[1];
    const cudaLaunchConfig_t cfg = config(step5::blocks(M, N, false, pairs), 384, step5::G::kSmemBytes, true, attr, s);
    cudaLaunchKernelEx(&cfg, kernel, a, b, c, C, M, N, K);
}

// The persistent kernel with the roles at a tile of BM x BN, NS stages, W's switches (7h and 7i in pairs only).
template <int BM, int BN, int NS, step6::Pair P, int kInc, class W>
static void launch6(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    constexpr bool cl = P != step6::kSolo, kSh = cl && W::h;
    auto kernel = step7::gemm<BM, BN, NS, P, kInc, W::j, kSh, kSh && W::i, W::g>;
    using T = step7::Tile<BM, BN, NS>;
    static const int res = resident(kernel, T::kSmemBytes, 384, cl);
    const CUtensorMap a = step2::make_map(A, K, M, 64, BM);
    const CUtensorMap b = step2::map_b(B, K, N), c = step7::map_c(C, M, N);
    int UM, UN;
    step6::units6(M, N, BM, BN, P, UM, UN);
    const int units = UM * UN < res ? UM * UN : res;
    cudaLaunchAttribute attr[1];
    const cudaLaunchConfig_t cfg = config(cl ? 2 * units : units, 384, T::kSmemBytes, cl, attr, s);
    cudaLaunchKernelEx(&cfg, kernel, a, b, c, C, M, N, K);
}

static int pairs5()   // the clusters of two of step 5's tile the GPU holds at once (the rule's unit of a wave)
{
    static const int n = resident(step7::gemm5<true, true, true, true>, step5::G::kSmemBytes, 384, true);
    return n;
}

// 7c: 6c's persistent kernel with the roles, the rows of its groups and its grid from 7c's rule, its maps' L2
// promotion of 256 bytes in one wave.
template <class W>
static void launch6c_rows(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    auto kernel = step7::gemm_rows<128, 256, 4, 232, W::j, W::g>;
    using T = step7::Tile<128, 256, 4>;
    static const int res = resident(kernel, T::kSmemBytes, 384, false);
    int UM, UN, G, R;
    step6::units6(M, N, 128, 256, step6::kSolo, UM, UN);
    step7::rows7c(UM, UN, res, G, R);
    const bool oneWave = UM * UN <= res;   // in one wave, A's and B's maps at the L2 promotion of 256 bytes
    const CUtensorMap a = step2::make_map(A, K, M, 64, 128, oneWave);
    const CUtensorMap b = step2::make_map(B, N, K, 64, 64, oneWave), c = step7::map_c(C, M, N);
    cudaLaunchAttribute attr[1];
    const cudaLaunchConfig_t cfg = config(R, 384, T::kSmemBytes, false, attr, s);
    cudaLaunchKernelEx(&cfg, kernel, a, b, c, C, M, N, K, G);
}

// Step 6's choice as step 7 runs it (step6::choose, with 7e's tile when with7e; alone7e: 7e alone when its tiles make
// one wave, else in pairs at every shape it takes; t6c: 6c by 7c's rule), on the kernels with the roles and W's
// switches, or, without the roles (v8), on step 6's.
template <class W>
static void tiles(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s, bool with7e,
                  bool alone7e, bool t6c)
{
    step6::Choice c = step6::choose(M, N, sms(), pairs5(), with7e);
    if (!alone7e && c.kind == step6::kT320Solo) c.kind = step6::kT320;
    if constexpr (!W::f) {
        switch (c.kind) {
        case step6::kT64Pair: step6_launch6<64, 128, 8, step6::kAlongM>(A, B, C, M, N, K, s); return;
        case step6::kT64Solo: step6_launch6<64, 128, 8, step6::kSolo>(A, B, C, M, N, K, s); return;
        case step6::kT128: step6_launch6<128, 128, 6, step6::kSolo>(A, B, C, M, N, K, s); return;
        case step6::kT320: step6_launch6<128, 320, 3, step6::kAlongM>(A, B, C, M, N, K, s); return;
        case step6::kT320Solo: step6_launch6<128, 320, 3, step6::kSolo>(A, B, C, M, N, K, s); return;
        case step6::kT256Solo:
            if (t6c) step6_launch6c_rows(A, B, C, M, N, K, s);
            else step6_launch6<128, 256, 4, step6::kSolo>(A, B, C, M, N, K, s);
            return;
        case step6::kT256: step6_launch5(A, B, C, M, N, K, s); return;   // M <= 128 (step6::choose)
        default: break;
        }
    } else {
        switch (c.kind) {
        case step6::kT64Pair: launch6<64, 128, 8, step6::kAlongM, 0, W>(A, B, C, M, N, K, s); return;
        case step6::kT64Solo: launch6<64, 128, 8, step6::kSolo, 0, W>(A, B, C, M, N, K, s); return;
        case step6::kT128: launch6<128, 128, 6, step6::kSolo, 0, W>(A, B, C, M, N, K, s); return;
        case step6::kT320: launch6<128, 320, 3, step6::kAlongM, 232, W>(A, B, C, M, N, K, s); return;
        case step6::kT320Solo: launch6<128, 320, 3, step6::kSolo, 232, W>(A, B, C, M, N, K, s); return;
        case step6::kT256Solo:
            if (t6c) launch6c_rows<W>(A, B, C, M, N, K, s);
            else launch6<128, 256, 4, step6::kSolo, 232, W>(A, B, C, M, N, K, s);
            return;
        case step6::kT256: launch5<W>(A, B, C, M, N, K, s); return;   // M <= 128 (step6::choose)
        default: break;
        }
    }
    fprintf(stderr, "step7: step 6's bands named at step 7\n");   // M <= 16 and J <= 8: step 7's bands (step7::choose)
    abort();
}

// ---------------------------------------------------------------- step 7's bands
// The bands of MM rows by J columns of 64 (7a, 7b, 7d), with the roles and W's switches (7h and 7i in clusters only),
// or, without the roles (v8), the bands of step 6's kind.
template <int MM, int J, step7::Mode P, class W>
static void launch_bands(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    constexpr int NS = step7::bandStages<MM, J>();
    constexpr int CL = step7::clusterBlocks(P);   // the blocks of a cluster
    constexpr bool kSh = P != step7::kSolo && W::h;
    using G = step7::Bands<MM, J, NS>;
    const CUtensorMap a = step2::make_map(A, K, M, 64, MM);
    const CUtensorMap b = step2::make_map(B, N, K, 64, 64), c = step2::make_map(C, N, M, 64, MM);
    const int bands = (N + 64 * J - 1) / (64 * J);
    int blocks = bands;
    if (P == step7::kAlongN) blocks = (bands + CL - 1) / CL * CL;   // bands outside the matrix: zeros, no store
    if (P == step7::kAlongM) blocks = 2 * bands;    // two blocks per band, rows 0 to 63 and 64 to 127
    cudaLaunchAttribute attr[1];
    static bool allowed = false;
    if constexpr (W::f) {
        auto kernel = step7::bands<MM, J, NS, P, W::l, kSh, kSh && W::i, W::g>;
        if (!allowed) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, G::kSmemBytes);
            allowed = true;
        }
        const cudaLaunchConfig_t cfg = config(blocks, 160, G::kSmemBytes, P != step7::kSolo, attr, s, CL);
        cudaLaunchKernelEx(&cfg, kernel, a, b, c, C, M, N, K);
    } else {
        auto kernel = step7::bands_noroles<MM, J, NS, P>;
        if (!allowed) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, G::kSmemBytes);
            allowed = true;
        }
        const cudaLaunchConfig_t cfg = config(blocks, 128, G::kSmemBytes, P != step7::kSolo, attr, s, CL);
        cudaLaunchKernelEx(&cfg, kernel, a, b, c, C, M, N, K);
    }
}
// 7b's rule: the clusters of eight of its kernel the GPU holds at once (one block per SM, as each J's), asked once.
static int clusters8()
{
    constexpr int NS = step7::bandStages<16, 8>();
    static const int n = resident(step7::bands<16, 8, NS, step7::kAlongN, true, true, true, true>,
                                  step7::Bands<16, 8, NS>::kSmemBytes, 160, true, 8);
    return n;
}
// The 16-row bands of step 6's J; 7b: twice as wide (at most 8 columns of 64), in clusters of eight along N.
template <int J, class W>
static void bands16(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s, bool wide)
{
    constexpr int J7b = 2 * J < 8 ? 2 * J : 8;
    if (wide) launch_bands<16, J7b, step7::kAlongN, W>(A, B, C, M, N, K, s);
    else launch_bands<16, J, step7::kSolo, W>(A, B, C, M, N, K, s);
}

// The kernels of step 7's choice, with W's switches and without the pieces of the rule variant v takes out.
template <class W>
static void run(int v, const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    const bool with7a = v != 1, with7b = v != 2, with7d = v != 3, with7e = v != 4, alone7e = v != 5, with7c = v != 7;
    step7::Choice c = step7::choose(M, N, sms(), clusters8());
    if ((!with7a && c.kind == step7::kBands64Pairs) || (!with7d && c.kind == step7::kBands128)) c.kind = step7::kStep6;
    if (!with7b && c.kind == step7::kBands16Wide) c.kind = step7::kBands16;
    const bool wide = c.kind == step7::kBands16Wide;
    switch (c.kind) {
    case step7::kBands16:
    case step7::kBands16Wide:
        switch (c.J) {
        case 1: bands16<1, W>(A, B, C, M, N, K, s, wide); break;
        case 2: bands16<2, W>(A, B, C, M, N, K, s, wide); break;
        case 3: bands16<3, W>(A, B, C, M, N, K, s, wide); break;
        case 4: bands16<4, W>(A, B, C, M, N, K, s, wide); break;
        case 5: bands16<5, W>(A, B, C, M, N, K, s, wide); break;
        case 6: bands16<6, W>(A, B, C, M, N, K, s, wide); break;
        case 7: bands16<7, W>(A, B, C, M, N, K, s, wide); break;
        default: bands16<8, W>(A, B, C, M, N, K, s, wide); break;
        }
        break;
    case step7::kBands64Pairs: launch_bands<64, 2, step7::kAlongM, W>(A, B, C, M, N, K, s); break;
    case step7::kBands128: launch_bands<128, 2, step7::kSolo, W>(A, B, C, M, N, K, s); break;
    case step7::kStep6: tiles<W>(A, B, C, M, N, K, s, with7e, alone7e, with7c); break;
    }
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    const int v = g_var;
    if (v < 0 || v > 14 || M <= 0 || M % 16 || N <= 0 || N % 64 || K <= 0 || K % 64) {
        fprintf(stderr, "step7 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    constexpr bool y = true, n = false;
    switch (v) {
    case 6: step6_gemm(A, B, C, M, N, K, s); break;   // step 6 itself
    case 8: run<Arm<n, n, n, n, n, n>>(v, A, B, C, M, N, K, s); break;   // without 7f and the five
    case 9: run<Arm<y, n, n, n, n, n>>(v, A, B, C, M, N, K, s); break;   // 7f alone
    case 10: run<Arm<y, n, y, y, y, y>>(v, A, B, C, M, N, K, s); break;   // without 7g
    case 11: run<Arm<y, y, n, n, y, y>>(v, A, B, C, M, N, K, s); break;   // without 7h (and 7i)
    case 12: run<Arm<y, y, y, n, y, y>>(v, A, B, C, M, N, K, s); break;   // without 7i
    case 13: run<Arm<y, y, y, y, n, y>>(v, A, B, C, M, N, K, s); break;   // without 7j
    case 14: run<Arm<y, y, y, y, y, n>>(v, A, B, C, M, N, K, s); break;   // without 7l
    default: run<All>(v, A, B, C, M, N, K, s); break;   // v0 to v5 and v7
    }
}
