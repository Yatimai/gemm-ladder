// Step 5 of the Turing rung, as a candidate for the judge (judge/judge.cu): cuBLASLt's choice of tile, shape by shape
// (see step5.cuh). Its variants, for the interleaved probe (probe/probe_t4.cu: arm v<i> sets g_var) and for verif.cu
// (g_nvar): v0 = step 5 (5a + 5b), v1 = without 5a, v2 = without 5b, v3 = step 4 itself (step4::gemm<true>, 128 x 256
// at every shape, the anchor).
#include <cstdio>
#include <cstdlib>
#include "step5.cuh"

int g_var = 0;
int g_nvar = 4;

template <int BM, int BN>
static void launch(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    using G = step5::Geometry<BM, BN>;
    static bool allowed = false;
    if (!allowed) {   // none of the four is above 48 KB: the call has no effect, kept as in step 4
        cudaFuncSetAttribute(step5::gemm<BM, BN>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::kSmemBytes);
        allowed = true;
    }
    const int tiles = (M + BM - 1) / BM * (N / BN);
    step5::gemm<BM, BN><<<tiles, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K);
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    static int sms = 0;
    if (!sms) {   // the current device's SM count, read once (5b)
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) {
            fprintf(stderr, "step5: no SM count\n");
            abort();
        }
    }
    const int v = g_var;
    if (v < 0 || v > 3 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step5 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    if (v == 3) {   // step 4
        constexpr int smem = step4::G::kSmemBytes;
        static bool allowed = false;
        if (!allowed) {
            cudaFuncSetAttribute(step4::gemm<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            allowed = true;
        }
        step4::gemm<true><<<(M + 127) / 128 * (N / 256), 256, smem, s>>>(A, B, C, M, N, K);
        return;
    }
    const step5::Choice ch = step5::choose(M, N, sms, v != 1, v != 2);   // v1: without 5a, v2: without 5b
    if (ch.bm == 64 && ch.bn == 128) launch<64, 128>(A, B, C, M, N, K, s);
    else if (ch.bm == 64) launch<64, 256>(A, B, C, M, N, K, s);
    else if (ch.bn == 128) launch<128, 128>(A, B, C, M, N, K, s);
    else launch<128, 256>(A, B, C, M, N, K, s);
}
