// Step 4 of the Turing rung, as a candidate for the judge: fragment double buffering (see step4.cuh).
// Its variants, for the interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 4
// (4a), v1 = step 3 itself (step3::gemm<true, true>, the anchor).
#include <cstdio>
#include <cstdlib>
#include "step4.cuh"

int g_var = 0;
int g_nvar = 2;

template <class Kernel>
static void allow_smem(Kernel kernel, int bytes)   // above 48 KB, a kernel's dynamic shared memory must be allowed once
{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    constexpr int smem = step4::G::kSmemBytes;   // step 2's 49 152 bytes in both variants
    static int sms = 0;
    if (!sms) {   // exactly 48 KB: allowed as well, without effect, as step 2
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        allow_smem(step4::gemm, smem);
        allow_smem(step3::gemm<true, true>, smem);
    }
    const int v = g_var;
    if (v < 0 || v > 1 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step4 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    switch (v) {   // one block per SM (step 0's persistent grid)
    case 0: step4::gemm<<<sms, 256, smem, s>>>(A, B, C, M, N, K); break;                // step 4
    case 1: step3::gemm<true, true><<<sms, 256, smem, s>>>(A, B, C, M, N, K); break;    // step 3
    }
}
