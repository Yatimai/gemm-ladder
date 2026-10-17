// Step 6 of the Ampere rung, as a candidate for the judge: the schedule per shape (see step6.cuh). Its variants, for
// the interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 6 (6a, 6b, 6c), v1 = without 6a,
// v2 = without 6b, v3 = without 6c, v4 = step 5 itself (step5::gemm<true, true>, 128 x 256 at every shape, the
// anchor), v5 = 6b's bands at every shape M <= 16; for verif.cu and compute-sanitizer only, whose shapes are too small
// for the rules to choose 6b's stream or 6c: v6 = 6b's stream at every shape M <= 16, v7 = 6c at every shape M > 16.
// Step 6's plan keeps K whole (7a and 7b split it, step7.cu). The kernels of 6b and 6c run without step 7's L2
// cache hints (kHint = false; 7d in step7.cu).
// candidate_dirty, for verif.cu: the counters left non-zero by a call (each call must leave them at zero, as it found
// them).
#include <cstdio>
#include <cstdlib>
#include "step5.cuh"
#include "step6.cuh"

int g_var = 0;
int g_nvar = 8;

static void check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        fprintf(stderr, "step6: %s: %s\n", what, cudaGetErrorString(e));
        abort();
    }
}

template <class Kernel>
static void allow_smem(Kernel kernel, int bytes)   // above 48 KB, a kernel's dynamic shared memory must be allowed once
{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

template <int BM, int BN, int WM, int SK, int kStages>
static void launch(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    using G = step6::Geometry<BM, BN, WM, SK, kStages>;
    static bool allowed = false;
    if (!allowed) {
        allow_smem(step6::gemm<BM, BN, WM, SK, kStages>, G::kSmemBytes);
        allowed = true;
    }
    const int tiles = (M + BM - 1) / BM * (N / BN);
    step6::gemm<BM, BN, WM, SK, kStages><<<tiles, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K);
}

static void step5_launch(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    constexpr int smem = step5::Smem::kBytes;
    static bool allowed = false;
    if (!allowed) {
        allow_smem(step5::gemm<true, true>, smem);
        allowed = true;
    }
    step5::gemm<true, true><<<(M + 127) / 128 * (N / 256), 256, smem, s>>>(A, B, C, M, N, K);
}

// The plan by 6a (switched off by its variant), without a split of K: 6a's tile, or step 5's kernel at 128 x 256.
static void step6_run(const half* A, const half* B, half* C, int M, int N, int K, int sms, bool cols128, cudaStream_t s)
{
    const step6::Plan p = step6::choose(M, N, K, sms, cols128, false);
    if (p.tile == step6::k128x128) launch<128, 128, 64, 32, 5>(A, B, C, M, N, K, s);
    else step5_launch(A, B, C, M, N, K, s);   // step 5's kernel, the same SASS
}

// 6b's workspace, allocated at the first call that needs it and never freed: the parts of the stream (16 x 256 halves
// for each of at most maxSegs segments of each band) and one counter per band, zeroed on the candidate's stream before
// its first kernel, then zero between calls; 6c's parts and counters share it. A launch that would need more stops.
struct Stream { half* parts = nullptr; unsigned* counts = nullptr; long partHalves = 0; int ncounts = 0; };
static Stream st6;
static Stream stream_workspace(cudaStream_t s)
{
    if (!st6.parts) {
        st6.partHalves = (long)1 << 23;   // 16 MB of parts
        st6.ncounts = 1 << 12;
        check(cudaMalloc(&st6.parts, (size_t)st6.partHalves * sizeof(half)), "cudaMalloc (stream parts)");
        check(cudaMalloc(&st6.counts, (size_t)st6.ncounts * sizeof(unsigned)), "cudaMalloc (stream counters)");
        check(cudaMemsetAsync(st6.counts, 0, (size_t)st6.ncounts * sizeof(unsigned), s),
              "cudaMemsetAsync (stream counters)");
    }
    return st6;
}

template <int BN, int kStages>
static void run_bands(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    using G = step6::BandGeometry<BN, kStages>;
    static bool allowed = false;
    if (!allowed) {
        allow_smem(step6::gemm_bands<BN, kStages, false>, G::kSmemBytes);
        allowed = true;
    }
    step6::gemm_bands<BN, kStages, false><<<N / BN, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K);
}

template <int BN, int kStages>
static void run_stream(const half* A, const half* B, half* C, int M, int N, int K, int sms, cudaStream_t s)
{
    using G = step6::BandGeometry<BN, kStages>;
    static bool allowed = false;
    if (!allowed) {
        allow_smem(step6::gemm_stream<BN, kStages, false>, G::kSmemBytes);
        allowed = true;
    }
    const long bands = N / BN, steps = K / 64, U = bands * steps;
    const int P = U < sms ? (int)U : sms;
    const long per = U / P;                                 // each block holds at least per units
    const int maxSegs = (int)((steps + per - 1) / per) + 1;   // segments of one band, at most
    const Stream w = stream_workspace(s);
    if (bands * maxSegs * 16 * BN > w.partHalves || bands > w.ncounts) {
        fprintf(stderr, "step6: %d x %d x %d needs more stream workspace than allocated\n", M, N, K);
        abort();
    }
    step6::gemm_stream<BN, kStages, false>
        <<<P, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K, w.parts, w.counts, maxSegs);
}

// 6c: P = sms rounded down to a multiple of MT blocks (MT rows of 128 x 256 tiles, one group per MT blocks); the
// parts in the stream's workspace (two slots of 128 x 256 halves per block), one counter per tile.
static void run_phase(const half* A, const half* B, half* C, int M, int N, int K, int sms, cudaStream_t s)
{
    using G = step6::Geometry<128, 256, 64, 64, 3>;
    constexpr int kSmem = G::kSmemBytes + 16;   // and the last block's flag
    static bool allowed = false;
    if (!allowed) {
        allow_smem(step6::gemm_phase<128, 256, 64, 64, 3, false>, kSmem);
        allowed = true;
    }
    const long U = (long)(N / 256) * (K / 64);   // units (column of tiles, slice of K64): at least one per group
    const int MT = (M + 127) / 128, P = (int)(sms / MT < U ? sms / MT : U) * MT;
    const Stream w = stream_workspace(s);
    if (P < MT || (long)P * 2 * 128 * 256 > w.partHalves || (long)MT * (N / 256) > w.ncounts) {
        fprintf(stderr, "step6: %d x %d x %d needs more stream workspace than allocated\n", M, N, K);
        abort();
    }
    step6::gemm_phase<128, 256, 64, 64, 3, false><<<P, G::kThreads, kSmem, s>>>(A, B, C, M, N, K, w.parts, w.counts);
}

int candidate_dirty()
{
    int dirty = 0;
    auto count = [&](const unsigned* d, int n) {
        if (!d) return;
        unsigned* h = (unsigned*)malloc(n * sizeof(unsigned));
        cudaMemcpy(h, d, n * sizeof(unsigned), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++) dirty += h[i] != 0;
        free(h);
    };
    count(st6.counts, st6.ncounts);
    return dirty;
}

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    static int sms = 0;
    if (!sms) {   // the current device's SM count, read once
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) {
            fprintf(stderr, "step6: no SM count\n");
            abort();
        }
    }
    const int v = g_var;
    if (v < 0 || v > 7 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step6 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    if (v == 4) {   // step 5
        step5_launch(A, B, C, M, N, K, s);
        return;
    }
    step6::M16 m = v == 2 ? step6::kNoBands : step6::m16(M, N, K, sms);   // v2: no 6b
    if (m != step6::kNoBands && v == 5) m = step6::kBands;                // v5: bands only
    if (m != step6::kNoBands && v == 6) m = step6::kStream;               // v6: the stream only
    const bool phase = v == 7 ? M > 16 : v != 3 && step6::phase(M, N, K, sms);   // v7: 6c forced; v3: no 6c
    if (m == step6::kBands) run_bands<128, 6>(A, B, C, M, N, K, s);
    else if (m == step6::kStream) run_stream<256, 4>(A, B, C, M, N, K, sms, s);
    else if (phase) run_phase(A, B, C, M, N, K, sms, s);
    else step6_run(A, B, C, M, N, K, sms, v != 1, s);
}
