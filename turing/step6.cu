// Step 6 of the Turing rung, as a candidate for the judge: L2-only loads, 16-row bands, split-K in two (see step6.cuh).
// Its variants, for the interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 6
// (6a + 6b + 6c), v1 = without 6a, v2 = without 6b, v3 = without 6c, v4 = step 5 itself (step5::gemm at
// step5::choose's tile, the anchor), v5 = 6b with B never marked evict-first. candidate_dirty, for verif.cu: the words
// of the workspace's state that a call left non-zero.
#include <cstdio>
#include <cstdlib>
#include "step6.cuh"

int g_var = 0;
int g_nvar = 6;

namespace {

// The workspace, allocated once, at the first call (never inside a measured call), at the ceilings of the rules, and
// never freed (a graph captured later keeps valid pointers): 6b's parts, 16 x 256 halves for each of at most 3 P
// blocks, and its bands' counters (3 P); 6c's parts, 2 x M x N halves with at most (P - 1) / 2 tiles of 128 x 256
// (2 tiles256 < P), and its tiles' counters. The counters are zeroed then, on the candidate's stream before its first
// kernel; afterwards, the last block of a band or a tile zeroes its counter. Not for concurrent calls.
struct Workspace {
    int sms = 0, splitTiles = 0;
    half* bandParts = nullptr;
    half* splitParts = nullptr;
    unsigned* words = nullptr;   // 3 P bands' counters, splitTiles tiles' counters
    unsigned* bandCounts() const { return words; }
    unsigned* splitCounts() const { return words + 3 * sms; }
    int counters() const { return 3 * sms + splitTiles; }   // the words that must be zero after a call
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
    check(cudaMalloc(&ws.bandParts, (size_t)3 * ws.sms * 16 * 256 * sizeof(half)), "cudaMalloc (6b's parts)");
    check(cudaMalloc(&ws.splitParts, (size_t)2 * ws.splitTiles * 128 * 256 * sizeof(half)), "cudaMalloc (6c's parts)");
    check(cudaMalloc(&ws.words, (size_t)ws.counters() * sizeof(unsigned)), "cudaMalloc (counters)");
    check(cudaMemsetAsync(ws.words, 0, (size_t)ws.counters() * sizeof(unsigned), s), "cudaMemsetAsync");
    int per[2] = {0, 0};   // 6b: three blocks per SM at both widths (the 64 KB carve-out of shared memory)
    const auto carveout = cudaFuncAttributePreferredSharedMemoryCarveout;
    check(cudaFuncSetAttribute(step6::gemm_bands<128, false, false>, carveout, 100), "carve-out (6b)");
    check(cudaFuncSetAttribute(step6::gemm_bands<128, true, true>, carveout, 100), "carve-out (6b)");
    check(cudaFuncSetAttribute(step6::gemm_bands<128, true, false>, carveout, 100), "carve-out (6b, v5)");
    check(cudaFuncSetAttribute(step6::gemm_bands<256, false, false>, carveout, 100), "carve-out (6b)");
    check(cudaFuncSetAttribute(step6::gemm_bands<256, true, true>, carveout, 100), "carve-out (6b)");
    check(cudaFuncSetAttribute(step6::gemm_bands<256, true, false>, carveout, 100), "carve-out (6b, v5)");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per[0], step6::gemm_bands<128, false, false>, 256,
                                                        step6::BandGeometry<128>::kSmemBytes), "occupancy (6b)");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per[1], step6::gemm_bands<256, true, true>, 256,
                                                        step6::BandGeometry<256>::kSmemBytes), "occupancy (6b)");
    if (per[0] < 3 || per[1] < 3)
        fprintf(stderr, "step6: %d and %d blocks of 6b per SM (3 expected): its rule's single round is not met\n",
                per[0], per[1]);
}

template <int BM, bool kCg>
void grid(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)   // step 5's grid, 6a's loads
{
    using G = step5::Geometry<BM>;
    static bool allowed = false;
    if (!allowed) {   // neither is above 48 KB: no effect, kept as in step 5
        allow_smem(step6::gemm<BM, kCg>, G::kSmemBytes);
        allowed = true;
    }
    step6::gemm<BM, kCg><<<ws.sms, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K);   // one block per SM
}

template <bool kCg>
void split(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)   // 6c
{
    using G = step5::Geometry<128>;
    static bool allowed = false;
    if (!allowed) {   // the last block's flag after the tile's 48 KB
        allow_smem(step6::gemm_split<128, kCg>, G::kSmemBytes + 16);
        allowed = true;
    }
    step6::gemm_split<128, kCg><<<ws.sms, G::kThreads, G::kSmemBytes + 16, s>>>(A, B, C, M, N, K, ws.splitParts,
                                                                                ws.splitCounts());   // one per SM
}

template <int BN>   // 6b; evictFirst false: v5
void bands(const half* A, const half* B, half* C, int M, int N, int K, int slices, bool evictFirst, cudaStream_t s)
{
    using G = step6::BandGeometry<BN>;
    const int steps = K / G::kBK, perSlice = steps / slices, longer = steps % slices;
    const dim3 blocks(slices, N / BN);
    if (slices > 1 && evictFirst)
        step6::gemm_bands<BN, true, true><<<blocks, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K, perSlice, longer,
                                                                                     ws.bandParts, ws.bandCounts());
    else if (slices > 1)
        step6::gemm_bands<BN, true, false><<<blocks, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K, perSlice,
                                                                                      longer, ws.bandParts,
                                                                                      ws.bandCounts());
    else
        step6::gemm_bands<BN, false, false><<<blocks, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K, perSlice, 0,
                                                                                       ws.bandParts, ws.bandCounts());
}

template <int BM>
void step5_grid(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)   // v4: step 5 itself
{
    using G = step5::Geometry<BM>;
    static bool allowed = false;
    if (!allowed) {
        allow_smem(step5::gemm<BM>, G::kSmemBytes);
        allowed = true;
    }
    step5::gemm<BM><<<ws.sms, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K);   // one block per SM
}

template <bool kCg>
void tiles(const step6::Plan& p, const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    if (p.kind == step6::kSplit) split<kCg>(A, B, C, M, N, K, s);
    else if (p.bm == 64) grid<64, kCg>(A, B, C, M, N, K, s);
    else grid<128, kCg>(A, B, C, M, N, K, s);
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
    if (v == 4) {   // step 5
        if (step5::choose(M, true) == 64) step5_grid<64>(A, B, C, M, N, K, s);
        else step5_grid<128>(A, B, C, M, N, K, s);
        return;
    }
    const bool cg = v != 1;   // v1: without 6a
    const step6::Plan p = step6::plan(M, N, K, ws.sms, v != 2, v != 3);   // v2, v3: without 6b, 6c
    if (p.kind == step6::kBands) {
        if (p.bn == 128) bands<128>(A, B, C, M, N, K, p.slices, v != 5, s);   // v5: B never marked evict-first
        else bands<256>(A, B, C, M, N, K, p.slices, v != 5, s);
    } else if (cg) {
        tiles<true>(p, A, B, C, M, N, K, s);
    } else {
        tiles<false>(p, A, B, C, M, N, K, s);
    }
}

// For verif.cu: after a call (synchronized), the counters of 6b and 6c that are not back at zero.
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
