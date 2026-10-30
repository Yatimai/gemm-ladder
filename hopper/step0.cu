// Step 0 of the Hopper rung, as a candidate for the judge: NVIDIA's compute_gemm, adapted (see nvidia_sample.cuh).
// Compiled for sm_90a, wmma gives HMMA.16816 and LDSM.x4, as on sm_80, and nothing of the H100's warpgroup MMA (no
// HGMMA); 128 registers, no spill, 38 912 bytes of shared memory. Its K64 loop: 175 instructions (SASS), of which per
// warp 64 HMMA.16816.F32, 24 LDSM.x4, 2 BAR.SYNC, and one of two branches of copy (8 LDG.E.128 and 8 STS.128 each).
// Two blocks would fit on an SM (registers and shared memory); the grid has one per SM.
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
