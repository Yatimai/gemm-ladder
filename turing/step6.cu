// Step 6 of the Turing rung, as a candidate for the judge (judge/judge.cu): beyond cuBLASLt (see step6.cuh). Its
// variants, for the interleaved probe (probe/probe_t4.cu: arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 6
// (6a + 6b + 6c + 6d), v1 = without 6a, v2 = without 6b, v3 = without 6c, v4 = without 6d, v5 = step 5 itself
// (step5::gemm at step5::choose's tile, the anchor). candidate_dirty, for verif.cu: the words of the workspace's state
// that a call left non-zero.
#include <cstdio>
#include <cstdlib>
#include "step6.cuh"

int g_var = 0;
int g_nvar = 6;

namespace {

// The workspace, allocated once, at the first call (never inside a measured call), at the ceilings of the rules, and
// never freed (a graph captured later keeps valid pointers): 6c's parts, 16 x 256 halves for each of at most 3 P
// blocks, and its bands' counters (3 P); 6d's parts, 2 x M x N halves with at most (P - 1) / 2 tiles of 128 x 256
// (2 tiles256 < P), and its tiles' counters; 6b's counter and generation. The counters are zeroed then, on the
// candidate's stream before its first kernel; afterwards, the last block of a band or a tile zeroes its counter, and
// the last block to reach a barrier its counter. Not for concurrent calls.
struct Workspace {
    int sms = 0, splitTiles = 0;
    half* bandParts = nullptr;
    half* splitParts = nullptr;
    unsigned* words = nullptr;   // 3 P bands' counters, splitTiles tiles' counters, 6b's counter and generation
    unsigned* bandCounts() const { return words; }
    unsigned* splitCounts() const { return words + 3 * sms; }
    unsigned* sync() const { return words + 3 * sms + splitTiles; }
    int counters() const { return 3 * sms + splitTiles + 1; }   // the words that must be zero after a call
    bool wavesResident[2] = {false, false};   // per kCg: P blocks of 6b's kernel can all be resident at once
};
Workspace ws;

void check(cudaError_t e, const char* what)   // setup's calls: a failure stops there, not at the first kernel
{
    if (e != cudaSuccess) {
        fprintf(stderr, "step6: %s: %s\n", what, cudaGetErrorString(e));
        abort();
    }
}

template <class Kernel>
void allow_smem(Kernel kernel, int bytes)   // above 48 KB, a kernel's dynamic shared memory must be allowed once
{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

template <bool kCg>
bool resident()   // 6b: one block of gemm_waves per SM at least, so that its P blocks are all resident
{
    using G = step5::Geometry<128, 256>;
    check(cudaFuncSetAttribute(step6::gemm_waves<128, 256, kCg>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               G::kSmemBytes), "cudaFuncSetAttribute (6b)");
    int per = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, step6::gemm_waves<128, 256, kCg>, G::kThreads,
                                                        G::kSmemBytes), "occupancy (6b)");
    return per >= 1;
}

void setup(cudaStream_t s)
{
    int dev = 0;
    check(cudaGetDevice(&dev), "cudaGetDevice");
    check(cudaDeviceGetAttribute(&ws.sms, cudaDevAttrMultiProcessorCount, dev), "SM count");
    if (ws.sms <= 0) {
        fprintf(stderr, "step6: no SM count\n");
        abort();
    }
    ws.splitTiles = (ws.sms - 1) / 2;
    check(cudaMalloc(&ws.bandParts, (size_t)3 * ws.sms * 16 * 256 * sizeof(half)), "cudaMalloc (6c's parts)");
    check(cudaMalloc(&ws.splitParts, (size_t)2 * ws.splitTiles * 128 * 256 * sizeof(half)), "cudaMalloc (6d's parts)");
    check(cudaMalloc(&ws.words, (size_t)(ws.counters() + 1) * sizeof(unsigned)), "cudaMalloc (counters)");
    check(cudaMemsetAsync(ws.words, 0, (size_t)(ws.counters() + 1) * sizeof(unsigned), s), "cudaMemsetAsync");
    ws.wavesResident[0] = resident<false>();
    ws.wavesResident[1] = resident<true>();
    int per[2] = {0, 0};   // 6c: three blocks per SM at both widths (the 64 KB carve-out of shared memory)
    const auto carveout = cudaFuncAttributePreferredSharedMemoryCarveout;
    check(cudaFuncSetAttribute(step6::gemm_bands<128, false>, carveout, 100), "carve-out (6c)");
    check(cudaFuncSetAttribute(step6::gemm_bands<128, true>, carveout, 100), "carve-out (6c)");
    check(cudaFuncSetAttribute(step6::gemm_bands<256, false>, carveout, 100), "carve-out (6c)");
    check(cudaFuncSetAttribute(step6::gemm_bands<256, true>, carveout, 100), "carve-out (6c)");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per[0], step6::gemm_bands<128, false>, 256,
                                                        step6::BandGeometry<128>::kSmemBytes), "occupancy (6c)");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per[1], step6::gemm_bands<256, true>, 256,
                                                        step6::BandGeometry<256>::kSmemBytes), "occupancy (6c)");
    if (per[0] < 3 || per[1] < 3)
        fprintf(stderr, "step6: %d and %d blocks of 6c per SM (3 expected): its rule's single round is not met\n",
                per[0], per[1]);
}

template <int BM, int BN, bool kCg>
void grid(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)   // step 5's grid, 6a's loads
{
    using G = step5::Geometry<BM, BN>;
    static bool allowed = false;
    if (!allowed) {   // none of the four is above 48 KB: no effect, kept as in step 5
        allow_smem(step6::gemm<BM, BN, kCg>, G::kSmemBytes);
        allowed = true;
    }
    step6::gemm<BM, BN, kCg><<<(M + BM - 1) / BM * (N / BN), G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K);
}

template <bool kCg>
void waves(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)   // 6b
{
    using G = step5::Geometry<128, 256>;
    const int tiles = (M + 127) / 128 * (N / 256);
    step6::gemm_waves<128, 256, kCg><<<tiles < ws.sms ? tiles : ws.sms, G::kThreads, G::kSmemBytes, s>>>(
        A, B, C, M, N, K, ws.sync());
}

template <bool kCg>
void split(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)   // 6d
{
    using G = step5::Geometry<128, 256>;
    static bool allowed = false;
    if (!allowed) {   // the last block's flag past the tile's 48 KB
        allow_smem(step6::gemm_split<128, 256, kCg>, G::kSmemBytes + 16);
        allowed = true;
    }
    const int tiles = (M + 127) / 128 * (N / 256);
    step6::gemm_split<128, 256, kCg><<<2 * tiles, G::kThreads, G::kSmemBytes + 16, s>>>(A, B, C, M, N, K,
                                                                                         ws.splitParts,
                                                                                         ws.splitCounts());
}

template <int BN>
void bands(const half* A, const half* B, half* C, int M, int N, int K, int slices, cudaStream_t s)   // 6c
{
    using G = step6::BandGeometry<BN>;
    const int steps = K / G::kBK, perSlice = steps / slices, longer = steps % slices;
    const dim3 blocks(slices, N / BN);
    if (slices > 1)
        step6::gemm_bands<BN, true><<<blocks, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K, perSlice, longer,
                                                                               ws.bandParts, ws.bandCounts());
    else
        step6::gemm_bands<BN, false><<<blocks, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K, perSlice, 0,
                                                                                ws.bandParts, ws.bandCounts());
}

template <int BM, int BN>
void step5_grid(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)   // v5: step 5 itself
{
    using G = step5::Geometry<BM, BN>;
    static bool allowed = false;
    if (!allowed) {
        allow_smem(step5::gemm<BM, BN>, G::kSmemBytes);
        allowed = true;
    }
    step5::gemm<BM, BN><<<(M + BM - 1) / BM * (N / BN), G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K);
}

template <bool kCg>
void tiles(const step6::Plan& p, const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    if (p.kind == step6::kWaves) waves<kCg>(A, B, C, M, N, K, s);
    else if (p.kind == step6::kSplit) split<kCg>(A, B, C, M, N, K, s);
    else if (p.bm == 64 && p.bn == 128) grid<64, 128, kCg>(A, B, C, M, N, K, s);
    else if (p.bm == 64) grid<64, 256, kCg>(A, B, C, M, N, K, s);
    else if (p.bn == 128) grid<128, 128, kCg>(A, B, C, M, N, K, s);
    else grid<128, 256, kCg>(A, B, C, M, N, K, s);
}

}  // namespace

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    if (!ws.sms) setup(s);
    const int v = g_var;
    if (v < 0 || v > 5 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step6 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    if (v == 5) {   // step 5
        const step5::Choice t = step5::choose(M, N, ws.sms, true, true);
        if (t.bm == 64 && t.bn == 128) step5_grid<64, 128>(A, B, C, M, N, K, s);
        else if (t.bm == 64) step5_grid<64, 256>(A, B, C, M, N, K, s);
        else if (t.bn == 128) step5_grid<128, 128>(A, B, C, M, N, K, s);
        else step5_grid<128, 256>(A, B, C, M, N, K, s);
        return;
    }
    const bool cg = v != 1;   // v1: without 6a
    step6::Plan p = step6::plan(M, N, K, ws.sms, v != 2, v != 3, v != 4);   // v2, v3, v4: without 6b, 6c, 6d
    if (p.kind == step6::kWaves && !ws.wavesResident[cg]) p = {step6::kGrid, 128, 256, 1};   // step 5's grid
    if (p.kind == step6::kBands) {
        if (p.bn == 128) bands<128>(A, B, C, M, N, K, p.slices, s);
        else bands<256>(A, B, C, M, N, K, p.slices, s);
    } else if (cg) {
        tiles<true>(p, A, B, C, M, N, K, s);
    } else {
        tiles<false>(p, A, B, C, M, N, K, s);
    }
}

// For verif.cu: after a call (synchronized), the counters of 6b, 6c and 6d that are not back at zero.
int candidate_dirty()
{
    if (!ws.sms) return 0;
    const int n = ws.counters();
    unsigned* h = (unsigned*)malloc(n * sizeof(unsigned));
    cudaMemcpy(h, ws.words, n * sizeof(unsigned), cudaMemcpyDeviceToHost);
    int dirty = 0;
    for (int i = 0; i < n; i++) dirty += h[i] != 0;
    free(h);
    return dirty;
}
