// Step 1 of the Ampere rung, as a candidate for the judge: granularity (see step1.cuh; the template is the Turing
// rung's, used without its double buffer). Its variants, for the interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar):
// v0 = step 1 (1a + 1c + 1d), v1 = without 1a, v2 = without 1c; v3 = 1d alone (step 0 with step 1's spread of the copy); v4 = step 0
// itself (compute_gemm).
#include <cstdio>
#include <cstdlib>
#include "nvidia_sample.cuh"
#include "step1.cuh"

int g_var = 0;
int g_nvar = 5;

template <bool kOneTile, bool kDouble, int kBN>
static void allow_smem()   // above 48 KB (v0, v1), a kernel's dynamic shared memory must be allowed once
{
    cudaFuncSetAttribute(step1::gemm<kOneTile, kDouble, kBN>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         step1::Geometry<kBN, kDouble>::kSmemBytes);
}

template <bool kOneTile, bool kDouble, int kBN>
static void launch(int blocks, const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    constexpr int smem = step1::Geometry<kBN, kDouble>::kSmemBytes;
    step1::gemm<kOneTile, kDouble, kBN><<<blocks, 256, smem, s>>>(A, B, C, M, N, K);
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    static int sms = 0;
    if (!sms) {
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        allow_smem<true, false, 256>();
        allow_smem<false, false, 256>();
    }
    const int v = g_var, bn = v >= 2 ? 128 : 256;
    if (v < 0 || v > 4 || M <= 0 || M % 16 || N <= 0 || N % bn || K <= 0 || K % 64) {
        fprintf(stderr, "step1 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    const int tiles = (M + 127) / 128 * (N / bn);
    switch (v) {
    case 0: launch<true, false, 256>(tiles, A, B, C, M, N, K, s); break;    // step 1
    case 1: launch<false, false, 256>(sms, A, B, C, M, N, K, s); break;     // without 1a: one block per SM
    case 2: launch<true, false, 128>(tiles, A, B, C, M, N, K, s); break;    // without 1c
    case 3: launch<false, false, 128>(sms, A, B, C, M, N, K, s); break;     // 1d alone: step 0 with the copy spread
    case 4: nvidia_sample::compute_gemm<<<sms, 256, nvidia_sample::kSmemBytes, s>>>(A, B, C, M, N, K); break;   // step 0
    }
}
