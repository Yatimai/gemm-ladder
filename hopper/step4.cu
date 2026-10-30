// Step 4 of the Hopper rung, as a candidate for the judge: clusters of two (see step4.cuh). Its variants, for the
// interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 4 (4a, 4b: step 2's kernel at 4 stages,
// in clusters of two), v1 = step 3 itself (step2::gemm<4>). The clusters go along M with two rows of tiles or more,
// along N with one.
#include <cstdio>
#include <cstdlib>
#include "step4.cuh"

int g_var = 0;
int g_nvar = 2;

template <auto kernel>   // one instance per kernel: its dynamic shared memory allowed once
static void launch(int threads, const half* A, const half* B, half* C, int M, int N, int K, bool alongM, cudaStream_t s)
{
    static bool allowed = false;   // above 48 KB, a kernel's dynamic shared memory must be allowed once
    if (!allowed) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, step4::G::kSmemBytes);
        allowed = true;
    }
    const CUtensorMap a = step2::map_a(A, M, K), b = step2::map_b(B, K, N);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(step4::blocks(M, N, alongM));
    cfg.blockDim = dim3(threads);
    cfg.dynamicSmemBytes = step4::G::kSmemBytes;
    cfg.stream = s;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;   // 4b: the cluster set at the launch
    attr[0].val.clusterDim.x = 2;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, a, b, C, M, N, K);
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    const int v = g_var;
    if (v < 0 || v > 1 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step4 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    const bool alongM = M > 128;   // two rows of tiles or more
    switch (v) {
    case 0:
        if (alongM) launch<step4::gemm<true>>(256, A, B, C, M, N, K, true, s);
        else launch<step4::gemm<false>>(256, A, B, C, M, N, K, false, s);
        break;
    case 1: {   // step 3: 3a alone
        static bool allowed = false;
        if (!allowed) {
            cudaFuncSetAttribute(step2::gemm<4>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 step2::Geometry<4>::kSmemBytes);
            allowed = true;
        }
        const CUtensorMap a = step2::map_a(A, M, K), b = step2::map_b(B, K, N);
        step2::gemm<4><<<(M + 127) / 128 * (N / 256), 256, step2::Geometry<4>::kSmemBytes, s>>>(a, b, C, M, N, K);
        break;
    }
    }
}
