// Step 6 of the Hopper rung, as a candidate for the judge: 16-row bands, the tile per shape (see step6.cuh). Its
// variants, for the interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 6 (6a, 6b, 6c:
// step6::bands, step6::gemm, and step 5's tile as step 5 runs it: step5::gemm); v1 = without
// 6a (M <= 16 by step 5's tile); v2 = without 6b (16 < M <= 128 by step 5's tile);
// v3 = without 6c (step 5's tile, in pairs); v4 = step 5 itself. The tiles in pairs keep their clusters, each block
// loading its own A and B. Where step 5's tile cannot take the shape (N not a multiple of 256), a variant keeps step
// 6's choice. Step 6's rule runs without step 7's tile of 128 x 320 (7e).
#include <cstdio>
#include <cstdlib>
#include "step6.cuh"

int g_var = 0;
int g_nvar = 5;

static int sms()   // the SM count, read once
{
    static int n = 0;
    if (!n) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
    return n;
}

static cudaLaunchConfig_t config(int blocks, int threads, int smem, bool cluster, cudaLaunchAttribute* attr,
                                 cudaStream_t s)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks);
    cfg.blockDim = dim3(threads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = s;
    attr[0].id = cudaLaunchAttributeClusterDimension;   // 4b: the cluster set at the launch
    attr[0].val.clusterDim.x = 2;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = cluster ? 1 : 0;
    return cfg;
}

// How many clusters of two (or blocks, without clusters) of a kernel the GPU holds at once; asked once per kernel.
template <typename F>
static int resident(F kernel, int smem, int threads, bool cluster)
{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int n = 0;
    if (cluster) {
        cudaLaunchAttribute attr[1];
        const cudaLaunchConfig_t cfg = config(2, threads, smem, true, attr, 0);
        cudaOccupancyMaxActiveClusters(&n, kernel, &cfg);
    } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, threads, smem);
        n *= sms();
    }
    if (n < 1) {
        fprintf(stderr, "step6: the occupancy query failed\n");
        abort();
    }
    return n;
}

// Step 5's tile at step 6, as step 5 runs it (step5::gemm: 5a, 5b), each block loading its own A and B, the
// output by the threads; v4 runs it at every shape (step 5 itself).
static void step5_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    auto kernel = M > 128 ? step5::gemm<true, true> : step5::gemm<false, true>;
    static int res[2] = {0, 0};
    int& r = res[M > 128];
    if (!r) r = resident(kernel, step5::G::kSmemBytes, 256, true);
    const CUtensorMap a = step2::map_a(A, M, K), b = step2::map_b(B, K, N);
    cudaLaunchAttribute attr[1];
    const cudaLaunchConfig_t cfg = config(step5::blocks(M, N, M > 128, r), 256, step5::G::kSmemBytes, true, attr, s);
    cudaLaunchKernelEx(&cfg, kernel, a, b, C, M, N, K);
}

// 6b and 6c: the persistent kernel (step6::gemm) at a tile of BM x BN, NS stages, a cluster of two along M or
// none; the output by the threads; in a cluster, each block loads its own A and B.
template <int BM, int BN, int NS, step6::Pair P>
static void launch6(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
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

// 6a: the bands of J columns of 64 (step6::bands), as many stages as shared memory holds, 8 at most.
template <int J>
static void launch_bands(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
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
static void bands(int J, const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    switch (J) {
    case 1: launch_bands<1>(A, B, C, M, N, K, s); break;
    case 2: launch_bands<2>(A, B, C, M, N, K, s); break;
    case 3: launch_bands<3>(A, B, C, M, N, K, s); break;
    case 4: launch_bands<4>(A, B, C, M, N, K, s); break;
    case 5: launch_bands<5>(A, B, C, M, N, K, s); break;
    case 6: launch_bands<6>(A, B, C, M, N, K, s); break;
    case 7: launch_bands<7>(A, B, C, M, N, K, s); break;
    default: launch_bands<8>(A, B, C, M, N, K, s); break;
    }
}

static int pairs()   // the clusters of two of step 5's kernel the GPU holds at once (the rule's unit of a wave)
{
    static const int n = resident(step5::gemm<true, true>, step5::G::kSmemBytes, 256, true);
    return n;
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    const int v = g_var;
    if (v < 0 || v > 4 || M <= 0 || M % 16 || N <= 0 || N % 64 || K <= 0 || K % 64) {
        fprintf(stderr, "step6 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    step6::Choice c = step6::choose(M, N, sms(), pairs(), false);   // step 6's rule, without step 7's 7e
    // the variants: a piece taken out falls back to step 5's tile where it can take the shape
    const bool step5ok = N % 256 == 0;
    if (step5ok && (v == 4 || (v == 1 && M <= 16) || (v == 2 && M > 16 && M <= 128))) c.kind = step6::kT256;
    switch (c.kind) {
    case step6::kBandsK: bands(c.J, A, B, C, M, N, K, s); break;   // no cluster, and 6a's own output
    case step6::kT64Pair: launch6<64, 128, 8, step6::kAlongM>(A, B, C, M, N, K, s); break;
    case step6::kT64Solo: launch6<64, 128, 8, step6::kSolo>(A, B, C, M, N, K, s); break;
    case step6::kT128: launch6<128, 128, 6, step6::kSolo>(A, B, C, M, N, K, s); break;
    case step6::kT256: step5_gemm(A, B, C, M, N, K, s); break;
    case step6::kT256Solo:
        if (v == 3) step5_gemm(A, B, C, M, N, K, s);
        else launch6<128, 256, 4, step6::kSolo>(A, B, C, M, N, K, s);
        break;
    case step6::kT320:
    case step6::kT320Solo:   // step 7's 7e: the rule without it never names it
        fprintf(stderr, "step6 v%d: 7e's tile named without 7e\n", v);
        abort();
    }
}
