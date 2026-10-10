// Step 2 of the Turing rung, as a candidate for the judge (judge/judge.cu): the path to the tensor cores (see
// step2.cuh). Its variants, for the interleaved probe (probe/probe_t4.cu: arm v<i> sets g_var) and for verif.cu
// (g_nvar): v0 = step 2 (2a + 2b), v1 = step 2 without 2b (2a over step 1's padded stages), v2 = step 1 itself
// (step1::gemm<true, true, 256>, the anchor).
#include <cstdio>
#include <cstdlib>
#include "step1.cuh"
#include "step2.cuh"

int g_var = 0;
int g_nvar = 3;

template <class Kernel>
static void allow_smem(Kernel kernel, int bytes)   // above 48 KB, a kernel's dynamic shared memory must be allowed once
{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    constexpr int smem0 = step2::Geometry<true>::kSmemBytes, smem1 = step2::Geometry<false>::kSmemBytes;
    constexpr int smem2 = step1::Geometry<256, true>::kSmemBytes;
    static bool allowed = false;
    if (!allowed) {   // v0 needs 49 152 bytes, exactly 48 KB: allowed as well, without effect
        allow_smem(step2::gemm<true>, smem0);
        allow_smem(step2::gemm<false>, smem1);
        allow_smem(step1::gemm<true, true, 256>, smem2);
        allowed = true;
    }
    const int v = g_var;
    if (v < 0 || v > 2 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step2 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    const int tiles = (M + 127) / 128 * (N / 256);
    switch (v) {
    case 0: step2::gemm<true><<<tiles, 256, smem0, s>>>(A, B, C, M, N, K); break;                // step 2
    case 1: step2::gemm<false><<<tiles, 256, smem1, s>>>(A, B, C, M, N, K); break;               // without 2b
    case 2: step1::gemm<true, true, 256><<<tiles, 256, smem2, s>>>(A, B, C, M, N, K); break;     // step 1
    }
}
