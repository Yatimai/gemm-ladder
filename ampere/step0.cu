// Step 0 of the Ampere rung, as a candidate for the judge: NVIDIA's compute_gemm, adapted (see nvidia_sample.cuh). The
// code is the Turing rung's step 0, line for line (only these comment lines differ), compiled for sm_80: wmma gives
// HMMA.16816 and LDSM.x4 there (sm_75: HMMA.1688 and LDSM.x2); 168 registers, no spill, 38 912 bytes of shared memory,
// one block per SM (by the registers). The epilogue converts its fp32 fragments to fp16 in place: the two accumulator
// fragments order their elements alike on sm_80 as on sm_75 (verif.cu checks the result at each of its shapes).
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
