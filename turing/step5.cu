// Step 5 of the Turing rung, as a candidate for the judge: the tile per shape (see step5.cuh). Its
// variants, for the interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 5
// (5a), v1 = without 5a, v2 = step 4 itself (step4::gemm, 128 x 256 at every shape, the anchor).
#include <cstdio>
#include <cstdlib>
#include "step5.cuh"

int g_var = 0;
int g_nvar = 3;

template <int BM>
static void launch(int blocks, const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    using G = step5::Geometry<BM>;
    static bool allowed = false;
    if (!allowed) {   // neither is above 48 KB: the call has no effect, kept as in step 4
        cudaFuncSetAttribute(step5::gemm<BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::kSmemBytes);
        allowed = true;
    }
    step5::gemm<BM><<<blocks, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K);
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    static int sms = 0;
    if (!sms) {   // the current device's SM count, read once (one block per SM)
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) {
            fprintf(stderr, "step5: no SM count\n");
            abort();
        }
    }
    const int v = g_var;
    if (v < 0 || v > 2 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step5 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    if (v == 2) {   // step 4
        constexpr int smem = step4::G::kSmemBytes;
        static bool allowed = false;
        if (!allowed) {
            cudaFuncSetAttribute(step4::gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            allowed = true;
        }
        step4::gemm<<<sms, 256, smem, s>>>(A, B, C, M, N, K);
        return;
    }
    if (step5::choose(M, v != 1) == 64) launch<64>(sms, A, B, C, M, N, K, s);   // v1: without 5a
    else launch<128>(sms, A, B, C, M, N, K, s);
}
