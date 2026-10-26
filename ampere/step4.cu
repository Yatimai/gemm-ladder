// Step 4 of the Ampere rung, as a candidate for the judge: evict-first stores, a tile swizzle (see step4.cuh). Its
// variants, for the interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 4 (4a + 4b),
// v1 = without 4a, v2 = without 4b, v3 = step 3 itself (step3::gemm<3>, the anchor).
#include <cstdio>
#include <cstdlib>
#include "step3.cuh"
#include "step4.cuh"

int g_var = 0;
int g_nvar = 4;

template <class Kernel>
static void allow_smem(Kernel kernel, int bytes)   // above 48 KB, a kernel's dynamic shared memory must be allowed once
{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    constexpr int smem = step4::Smem::kBytes;   // 147 456 bytes, as step 3
    static bool allowed = false;
    if (!allowed) {
        allow_smem(step4::gemm<true, true>, smem);
        allow_smem(step4::gemm<false, true>, smem);
        allow_smem(step4::gemm<true, false>, smem);
        allow_smem(step3::gemm<3>, smem);
        allowed = true;
    }
    const int v = g_var;
    if (v < 0 || v > 3 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step4 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    const int tiles = (M + 127) / 128 * (N / 256);
    switch (v) {
    case 0: step4::gemm<true, true><<<tiles, 256, smem, s>>>(A, B, C, M, N, K); break;     // step 4
    case 1: step4::gemm<false, true><<<tiles, 256, smem, s>>>(A, B, C, M, N, K); break;    // without 4a
    case 2: step4::gemm<true, false><<<tiles, 256, smem, s>>>(A, B, C, M, N, K); break;    // without 4b
    case 3: step3::gemm<3><<<tiles, 256, smem, s>>>(A, B, C, M, N, K); break;              // step 3
    }
}
