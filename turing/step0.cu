// Step 0 of the Turing rung, as a candidate for the judge (judge/judge.cu): NVIDIA's compute_gemm, adapted (see
// nvidia_sample.cuh).
#include <cstdio>
#include <cstdlib>
#include "nvidia_sample.cuh"

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    static int sms = 0;
    if (!sms) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    if (M % 16 || N % 128 || K % 64) { fprintf(stderr, "step0: shape %d x %d x %d not supported\n", M, N, K); abort(); }
    nvidia_sample::compute_gemm<<<sms, 256, nvidia_sample::kSmemBytes, s>>>(A, B, C, M, N, K);   // persistent: one block per SM
}
