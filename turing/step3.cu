// Step 3 of the Turing rung, as a candidate for the judge (judge/judge.cu): the memory hierarchy (see step3.cuh). Its
// variants, for the interleaved probe (probe/probe_t4.cu: arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 3
// (3a + 3b + 3c), v1 = without 3a, v2 = without 3b, v3 = without 3c, v4 = step 2 itself (step2::gemm<true>, the
// anchor).
#include <cstdio>
#include <cstdlib>
#include "step3.cuh"

int g_var = 0;
int g_nvar = 5;

template <class Kernel>
static void allow_smem(Kernel kernel, int bytes)   // above 48 KB, a kernel's dynamic shared memory must be allowed once
{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    constexpr int smem = step3::G::kSmemBytes;   // step 2's 49 152 bytes in all five variants
    static bool allowed = false;
    if (!allowed) {   // exactly 48 KB: allowed as well, without effect, as step 2
        allow_smem(step3::gemm<true, true, true>, smem);
        allow_smem(step3::gemm<false, true, true>, smem);
        allow_smem(step3::gemm<true, false, true>, smem);
        allow_smem(step3::gemm<true, true, false>, smem);
        allow_smem(step2::gemm<true>, smem);
        allowed = true;
    }
    const int v = g_var;
    if (v < 0 || v > 4 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step3 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    const int tiles = (M + 127) / 128 * (N / 256);
    switch (v) {
    case 0: step3::gemm<true, true, true><<<tiles, 256, smem, s>>>(A, B, C, M, N, K); break;    // step 3
    case 1: step3::gemm<false, true, true><<<tiles, 256, smem, s>>>(A, B, C, M, N, K); break;   // without 3a
    case 2: step3::gemm<true, false, true><<<tiles, 256, smem, s>>>(A, B, C, M, N, K); break;   // without 3b
    case 3: step3::gemm<true, true, false><<<tiles, 256, smem, s>>>(A, B, C, M, N, K); break;   // without 3c
    case 4: step2::gemm<true><<<tiles, 256, smem, s>>>(A, B, C, M, N, K); break;                // step 2
    }
}
