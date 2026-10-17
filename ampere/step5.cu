// Step 5 of the Ampere rung, as a candidate for the judge: the issue (see step5.cuh). Its variants, for the interleaved
// probe (arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 5 (5a + 5b), v1 = without 5a, v2 = without 5b, v3 =
// step 4 itself (step4::gemm<true, true>, the anchor).
#include <cstdio>
#include <cstdlib>
#include "step4.cuh"
#include "step5.cuh"

int g_var = 0;
int g_nvar = 4;

template <class Kernel>
static void allow_smem(Kernel kernel, int bytes)   // above 48 KB, a kernel's dynamic shared memory must be allowed once
{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    constexpr int smem = step5::Smem::kBytes;   // 147 456 bytes, as steps 3 and 4
    static bool allowed = false;
    if (!allowed) {
        allow_smem(step5::gemm<true, true>, smem);
        allow_smem(step5::gemm<false, true>, smem);
        allow_smem(step5::gemm<true, false>, smem);
        allow_smem(step4::gemm<true, true>, smem);
        allowed = true;
    }
    const int v = g_var;
    if (v < 0 || v > 3 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step5 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    const int tiles = (M + 127) / 128 * (N / 256);
    switch (v) {
    case 0: step5::gemm<true, true><<<tiles, 256, smem, s>>>(A, B, C, M, N, K); break;     // step 5
    case 1: step5::gemm<false, true><<<tiles, 256, smem, s>>>(A, B, C, M, N, K); break;    // without 5a
    case 2: step5::gemm<true, false><<<tiles, 256, smem, s>>>(A, B, C, M, N, K); break;    // without 5b
    case 3: step4::gemm<true, true><<<tiles, 256, smem, s>>>(A, B, C, M, N, K); break;     // step 4
    }
}
