// Step 2 of the Hopper rung, as a candidate for the judge: TMA loads in three stages, one MMA group in flight (see
// step2.cuh). Its variants, for the interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 2 (3
// stages, 2d), v1 = 2 stages, v2 = step 1 itself (step1::gemm<256>), v3 = without 2d (wait_group 0), v4 = 2 stages
// without 2d.
#include <cstdio>
#include <cstdlib>
#include "step2.cuh"

int g_var = 0;
int g_nvar = 5;

template <int kStages, bool kWait1 = true>
static void launch(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    static bool allowed = false;   // above 48 KB, a kernel's dynamic shared memory must be allowed once
    if (!allowed) {
        cudaFuncSetAttribute(step2::gemm<kStages, kWait1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             step2::Geometry<kStages>::kSmemBytes);
        allowed = true;
    }
    const CUtensorMap a = step2::map_a(A, M, K), b = step2::map_b(B, K, N);
    const int tiles = (M + 127) / 128 * (N / 256);
    step2::gemm<kStages, kWait1><<<tiles, 256, step2::Geometry<kStages>::kSmemBytes, s>>>(a, b, C, M, N, K);
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    const int v = g_var;
    if (v < 0 || v > 4 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step2 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    switch (v) {
    case 0: launch<3>(A, B, C, M, N, K, s); break;   // step 2
    case 1: launch<2>(A, B, C, M, N, K, s); break;   // 2 stages
    case 3: launch<3, false>(A, B, C, M, N, K, s); break;   // without 2d
    case 4: launch<2, false>(A, B, C, M, N, K, s); break;   // 2 stages without 2d
    case 2: {                                         // step 1
        static bool allowed = false;
        if (!allowed) {
            cudaFuncSetAttribute(step1::gemm<256>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 step1::Geometry<256>::kSmemBytes);
            allowed = true;
        }
        step1::gemm<256><<<(M + 127) / 128 * (N / 256), 256, step1::Geometry<256>::kSmemBytes, s>>>(A, B, C, M, N, K);
        break;
    }
    }
}
