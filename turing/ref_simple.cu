// The reference point of the Turing rung, as a candidate for the judge (judge/judge.cu): NVIDIA's simple_wmma_gemm,
// adapted (see nvidia_sample.cuh). Not a step: the wmma API alone, without shared memory.
#include <cstdio>
#include <cstdlib>
#include "nvidia_sample.cuh"

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    if (M % 16 || N % 64 || K % 16) { fprintf(stderr, "ref_simple: shape %d x %d x %d not supported\n", M, N, K); abort(); }
    nvidia_sample::simple_wmma_gemm<<<dim3((M + 63) / 64, N / 64), dim3(128, 4), 0, s>>>(A, B, C, M, N, K);
}
