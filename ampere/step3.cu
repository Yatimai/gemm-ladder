// Step 3 of the Ampere rung, as a candidate for the judge: cp.async, three stages (see step3.cuh). Its variants, for
// the interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar), nested: v0 = step 3 (3a + 3b, three stages),
// v1 = two stages, v2 = 3a alone (one stage), v3 = step 2 itself (step2::gemm<true>, the anchor).
#include <cstdio>
#include <cstdlib>
#include "step2.cuh"
#include "step3.cuh"

int g_var = 0;
int g_nvar = 4;

template <class Kernel>
static void allow_smem(Kernel kernel, int bytes)   // above 48 KB, a kernel's dynamic shared memory must be allowed once
{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    constexpr int smem3 = step3::Smem<3>::kBytes, smem2 = step3::Smem<2>::kBytes, smem1 = step3::Smem<1>::kBytes;
    constexpr int smemS2 = step2::Geometry<true>::kSmemBytes;
    static bool allowed = false;
    if (!allowed) {   // 147 456, 98 304, 49 152 and 49 152 bytes
        allow_smem(step3::gemm<3>, smem3);
        allow_smem(step3::gemm<2>, smem2);
        allow_smem(step3::gemm<1>, smem1);
        allow_smem(step2::gemm<true>, smemS2);
        allowed = true;
    }
    const int v = g_var;
    if (v < 0 || v > 3 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step3 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    const int tiles = (M + 127) / 128 * (N / 256);
    switch (v) {
    case 0: step3::gemm<3><<<tiles, 256, smem3, s>>>(A, B, C, M, N, K); break;           // step 3
    case 1: step3::gemm<2><<<tiles, 256, smem2, s>>>(A, B, C, M, N, K); break;           // two stages
    case 2: step3::gemm<1><<<tiles, 256, smem1, s>>>(A, B, C, M, N, K); break;           // 3a alone
    case 3: step2::gemm<true><<<tiles, 256, smemS2, s>>>(A, B, C, M, N, K); break;       // step 2
    }
}
