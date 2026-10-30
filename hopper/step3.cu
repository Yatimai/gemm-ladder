// Step 3 of the Hopper rung, as a candidate for the judge: a fourth stage (see step3.cuh). Its variants, for the
// interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 3 (3a: step2::gemm<4>), v1 = step 2
// itself (step2::gemm<3>).
#include <cstdio>
#include <cstdlib>
#include "step3.cuh"

int g_var = 0;
int g_nvar = 2;

template <typename Kernel>
static void launch(Kernel kernel, int threads, int smem, const half* A, const half* B, half* C, int M, int N, int K,
                   cudaStream_t s)
{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);   // above 48 KB
    const CUtensorMap a = step2::map_a(A, M, K), b = step2::map_b(B, K, N);
    kernel<<<(M + 127) / 128 * (N / 256), threads, smem, s>>>(a, b, C, M, N, K);
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    const int v = g_var;
    if (v < 0 || v > 1 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step3 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    switch (v) {
    case 0: launch(step2::gemm<4>, 256, step2::Geometry<4>::kSmemBytes, A, B, C, M, N, K, s); break;   // step 3: 3a
    case 1: launch(step2::gemm<3>, 256, step2::Geometry<3>::kSmemBytes, A, B, C, M, N, K, s); break;   // step 2
    }
}
