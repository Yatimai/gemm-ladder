// Step 1 of the Hopper rung, as a candidate for the judge: wgmma, a 128 x 256 tile, every thread copying (see
// step1.cuh). Its variants, for the interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 1
// (1a + 1b), v1 = a tile of 128 x 128, v2 = step 0 itself (compute_gemm), v3 = without 1b (step 0's split of the copy:
// gemm_split).
#include <cstdio>
#include <cstdlib>
#include "nvidia_sample.cuh"
#include "step1.cuh"

int g_var = 0;
int g_nvar = 4;

template <int kBN, bool kSpread = true>
static void allow_smem()   // above 48 KB, a kernel's dynamic shared memory must be allowed once
{
    if constexpr (kSpread)
        cudaFuncSetAttribute(step1::gemm<kBN>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             step1::Geometry<kBN>::kSmemBytes);
    else
        cudaFuncSetAttribute(step1::gemm_split<kBN>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             step1::Geometry<kBN>::kSmemBytes);
}

template <int kBN, bool kSpread = true>
static void launch(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    const int tiles = (M + 127) / 128 * (N / kBN);
    constexpr int smem = step1::Geometry<kBN>::kSmemBytes;
    if constexpr (kSpread) step1::gemm<kBN><<<tiles, 256, smem, s>>>(A, B, C, M, N, K);
    else step1::gemm_split<kBN><<<tiles, 256, smem, s>>>(A, B, C, M, N, K);
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    static int sms = 0;
    if (!sms) {
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        allow_smem<256>();
        allow_smem<128>();
        allow_smem<256, false>();
    }
    const int v = g_var, bn = v == 1 ? 128 : 256;
    if (v < 0 || v > 3 || M <= 0 || M % 16 || N <= 0 || N % bn || K <= 0 || K % 64) {
        fprintf(stderr, "step1 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    switch (v) {
    case 0: launch<256>(A, B, C, M, N, K, s); break;    // step 1
    case 1: launch<128>(A, B, C, M, N, K, s); break;    // a tile of 128 x 128
    case 2: nvidia_sample::compute_gemm<<<sms, 256, nvidia_sample::kSmemBytes, s>>>(A, B, C, M, N, K); break;   // step 0
    case 3: launch<256, false>(A, B, C, M, N, K, s); break;   // without 1b
    }
}
