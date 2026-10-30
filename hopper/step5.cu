// Step 5 of the Hopper rung, as a candidate for the judge: a persistent grid, a tile swizzle (see step5.cuh). Its
// variants, for the interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 5 (5a, 5b), v1 =
// without 5a (one pair of tiles per cluster), v2 = without 5b (the pairs row after row), v3 = step 4 itself. The
// clusters go along M with two rows of tiles or more, along N with one; each block loads its own A and B; two
// warpgroups that both compute, warp 0 loading (3b and 3c come back at step 7 only).
#include <cstdio>
#include <cstdlib>
#include "step5.cuh"

int g_var = 0;
int g_nvar = 4;

// A launch of step 5's tile (step5::gemm: 256 threads), in clusters of two; persistent: as many clusters as
// the GPU holds at once (5a), else one cluster per pair of tiles (v1).
template <auto kernel>
static void launch(const half* A, const half* B, half* C, int M, int N, int K, bool alongM, bool persistent,
                   cudaStream_t s)
{
    cudaLaunchConfig_t cfg = {};
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = step5::G::kSmemBytes;
    cfg.stream = s;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;   // 4b: the cluster set at the launch
    attr[0].val.clusterDim.x = 2;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    static int maxClusters = 0;   // 5a: the clusters the GPU holds at once, asked once
    if (!maxClusters) {   // above 48 KB, a kernel's dynamic shared memory must be allowed first
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, step5::G::kSmemBytes);
        cfg.gridDim = dim3(2);
        if (cudaOccupancyMaxActiveClusters(&maxClusters, kernel, &cfg) != cudaSuccess || maxClusters < 1) {
            fprintf(stderr, "step5: cudaOccupancyMaxActiveClusters failed\n");
            abort();
        }
    }
    const CUtensorMap a = step2::map_a(A, M, K), b = step2::map_b(B, K, N);
    cfg.gridDim = dim3(step5::blocks(M, N, alongM, persistent ? maxClusters : 1 << 30));
    cudaLaunchKernelEx(&cfg, kernel, a, b, C, M, N, K);
}

// Step 5's tile: kGrouped: 5b; the descriptors rebuilt at each slice, each block loads its own A and B, 16-byte stores
// by the threads.
template <bool kGrouped>
static void tile5(const half* A, const half* B, half* C, int M, int N, int K, bool persistent, cudaStream_t s)
{
    if (M > 128) launch<step5::gemm<true, kGrouped>>(A, B, C, M, N, K, true, persistent, s);
    else launch<step5::gemm<false, kGrouped>>(A, B, C, M, N, K, false, persistent, s);
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    const int v = g_var;
    if (v < 0 || v > 3 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step5 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    const bool alongM = M > 128;
    switch (v) {
    case 0: tile5<true>(A, B, C, M, N, K, true, s); break;
    case 1: tile5<true>(A, B, C, M, N, K, false, s); break;   // without 5a
    case 2: tile5<false>(A, B, C, M, N, K, true, s); break;   // without 5b
    case 3: {   // step 4 (v0: step 2's kernel at 4 stages, in clusters of two)
        auto kernel = alongM ? step4::gemm<true> : step4::gemm<false>;
        static bool allowed[2] = {false, false};
        if (!allowed[alongM]) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, step4::G::kSmemBytes);
            allowed[alongM] = true;
        }
        const CUtensorMap a = step2::map_a(A, M, K), b = step2::map_b(B, K, N);
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(step4::blocks(M, N, alongM));
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = step4::G::kSmemBytes;
        cfg.stream = s;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = 2;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, kernel, a, b, C, M, N, K);
        break;
    }
    }
}
