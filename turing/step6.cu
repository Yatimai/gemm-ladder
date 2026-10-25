// Step 6 of the Turing rung, as a candidate for the judge: L2-only loads, 16-row bands, split-K in two, synchronized
// waves (see step6.cuh). Its variants, for the interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar):
// v0 = step 6 (6a to 6f), v1 = without 6a, v2 = without 6b, v3 = without 6c, v4 = step 5 itself (step5::gemm at
// step5::choose's rows, on step 5's persistent grid: the anchor), v5 = 6b with B never marked evict-first, v6 = without
// 6d (6e's grid where 6d runs its waves), v7 = without 6e (v0 with step 0's persistent grid where v0 runs 6e's grid),
// v8 = without 6f (v0 with step 2's order of the HMMA). candidate_dirty, for verif.cu: the words of the workspace's
// state that a call left non-zero.
#include <cstdio>
#include <cstdlib>
#include "step6.cuh"

int g_var = 0;
int g_nvar = 9;

namespace {

// The workspace, allocated once, at the first call (never inside a measured call), at the ceilings of the rules, and
// never freed (a graph captured later keeps valid pointers): 6b's parts, 16 x 256 halves for each of at most 3 P
// blocks, and its bands' counters (3 P); 6c's parts, 2 x M x N halves with at most (P - 1) / 2 tiles of 128 x 256
// (2 tiles256 < P), and its tiles' counters; 6d's counter and generation, in two words of their own. The words are
// zeroed then, on the candidate's stream before its first kernel; afterwards, the last block of a band or a tile zeroes
// its counter, and the last block to reach one of 6d's barriers the barrier's counter. Not for concurrent calls.
struct Workspace {
    int sms = 0, splitTiles = 0;
    half* bandParts = nullptr;
    half* splitParts = nullptr;
    unsigned* words = nullptr;   // 3 P bands' counters, splitTiles tiles' counters
    unsigned* bandCounts() const { return words; }
    unsigned* splitCounts() const { return words + 3 * sms; }
    int counters() const { return 3 * sms + splitTiles; }   // the words that must be zero after a call
    unsigned* waveSync = nullptr;   // 6d: the waves' counter (zero after a call) and generation, allocated apart
    bool wavesResident[2] = {};     // 6d, without 6a and with: one block of gemm_waves per SM fits (its P blocks can
                                    // all be resident)
    bool wavesRowsResident = false;   // the same for v8's waves (gemm_waves_rows)
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
    check(cudaMalloc(&ws.waveSync, 2 * sizeof(unsigned)), "cudaMalloc (6d's counter)");
    check(cudaMemsetAsync(ws.waveSync, 0, 2 * sizeof(unsigned), s), "cudaMemsetAsync (6d)");
    using G = step6::Geometry<128, 256>;
    check(cudaFuncSetAttribute(step6::gemm_waves<128, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               G::kSmemBytes), "cudaFuncSetAttribute (6d, v1)");
    check(cudaFuncSetAttribute(step6::gemm_waves<128, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               G::kSmemBytes), "cudaFuncSetAttribute (6d)");
    int waves[2] = {0, 0};
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&waves[0], step6::gemm_waves<128, false>, G::kThreads,
                                                        G::kSmemBytes), "occupancy (6d, v1)");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&waves[1], step6::gemm_waves<128, true>, G::kThreads,
                                                        G::kSmemBytes), "occupancy (6d)");
    ws.wavesResident[0] = waves[0] >= 1;
    ws.wavesResident[1] = waves[1] >= 1;
    check(cudaFuncSetAttribute(step6::gemm_waves_rows<128, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               G::kSmemBytes), "cudaFuncSetAttribute (6d, v8)");
    int wavesRows = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&wavesRows, step6::gemm_waves_rows<128, true>, G::kThreads,
                                                        G::kSmemBytes), "occupancy (6d, v8)");
    ws.wavesRowsResident = wavesRows >= 1;
}

template <int BM, int BN, bool kCg>
void grid(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)   // 6e's grid, 6a's loads
{
    using G = step6::Geometry<BM, BN>;
    static bool allowed = false;
    if (!allowed) {   // none of the four is above 48 KB: no effect, kept as in step 5
        allow_smem(step6::gemm<BM, BN, kCg>, G::kSmemBytes);
        allowed = true;
    }
    const int tiles = (M + BM - 1) / BM * (N / BN);   // one block per tile (6e)
    step6::gemm<BM, BN, kCg><<<tiles, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K);
}

template <bool kCg>
void split(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)   // 6c
{
    using G = step6::Geometry<128, 256>;
    static bool allowed = false;
    if (!allowed) {   // the last block's flag after the tile's 48 KB
        allow_smem(step6::gemm_split<128, kCg>, G::kSmemBytes + 16);
        allowed = true;
    }
    const int tiles = (M + 127) / 128 * (N / 256);   // one block per half of a tile (6e's grid at 6c's halves)
    step6::gemm_split<128, kCg><<<2 * tiles, G::kThreads, G::kSmemBytes + 16, s>>>(A, B, C, M, N, K, ws.splitParts,
                                                                                   ws.splitCounts());
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
void waves(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)   // 6d
{
    using G = step6::Geometry<128, 256>;
    const int n = (M + 127) / 128 * (N / 256);
    step6::gemm_waves<128, kCg><<<n < ws.sms ? n : ws.sms, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K,
                                                                                        ws.waveSync);
}

// v7 and v8 (the header): v0's plan with 6a's loads; v7 without 6e (step 0's persistent grid, one block per SM, where
// v0 runs 6e's grid: its tiles and 6c's halves), v8 without 6f (step 2's order of the HMMA in every tile).
template <int BM, int BN>
void grid_v(int v, const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    using G = step6::Geometry<BM, BN>;
    static bool allowed = false;
    if (!allowed) {   // none of the four is above 48 KB: no effect, kept as in step 5
        allow_smem(step6::gemm_persistent<BM, BN, true>, G::kSmemBytes);
        allow_smem(step6::gemm_rows<BM, BN, true>, G::kSmemBytes);
        allowed = true;
    }
    const int tiles = (M + BM - 1) / BM * (N / BN);
    if (v == 7) step6::gemm_persistent<BM, BN, true><<<ws.sms, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K);
    else step6::gemm_rows<BM, BN, true><<<tiles, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K);
}

void split_v(int v, const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)   // 6c, v7 and v8
{
    using G = step6::Geometry<128, 256>;
    static bool allowed = false;
    if (!allowed) {   // the last block's flag after the tile's 48 KB
        allow_smem(step6::gemm_split_persistent<128, true>, G::kSmemBytes + 16);
        allow_smem(step6::gemm_split_rows<128, true>, G::kSmemBytes + 16);
        allowed = true;
    }
    const int tiles = (M + 127) / 128 * (N / 256);
    if (v == 7)   // one block per SM over the halves
        step6::gemm_split_persistent<128, true><<<ws.sms, G::kThreads, G::kSmemBytes + 16, s>>>(A, B, C, M, N, K,
                                                                                            ws.splitParts,
                                                                                            ws.splitCounts());
    else          // one block per half
        step6::gemm_split_rows<128, true><<<2 * tiles, G::kThreads, G::kSmemBytes + 16, s>>>(A, B, C, M, N, K,
                                                                                         ws.splitParts,
                                                                                         ws.splitCounts());
}

void waves_rows(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)   // 6d, v8
{
    using G = step6::Geometry<128, 256>;
    const int n = (M + 127) / 128 * (N / 256);
    step6::gemm_waves_rows<128, true><<<n < ws.sms ? n : ws.sms, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K,
                                                                                              ws.waveSync);
}

void tiles_v(int v, const step6::Plan& p, const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    if (p.kind == step6::kWaves) {   // 6d: v7 keeps v0's waves (6e does not act there), v8 has its own, without 6f;
        if (v == 7 && ws.wavesResident[1]) waves<true>(A, B, C, M, N, K, s);
        else if (v == 8 && ws.wavesRowsResident) waves_rows(A, B, C, M, N, K, s);
        else grid_v<128, 256>(v, A, B, C, M, N, K, s);   // where v0 falls back on 6e's grid (the waves cannot all be
        return;                                          // resident): v7 the persistent grid, v8 6e's without 6f
    }
    if (p.kind == step6::kSplit) split_v(v, A, B, C, M, N, K, s);
    else if (p.bm == 64 && p.bn == 128) grid_v<64, 128>(v, A, B, C, M, N, K, s);
    else if (p.bm == 64) grid_v<64, 256>(v, A, B, C, M, N, K, s);
    else if (p.bn == 128) grid_v<128, 128>(v, A, B, C, M, N, K, s);
    else grid_v<128, 256>(v, A, B, C, M, N, K, s);
}

template <bool kCg>
void tiles(const step6::Plan& p, const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    if (p.kind == step6::kWaves) {   // 6d; 6e's grid if its blocks cannot all be resident
        if (ws.wavesResident[kCg]) waves<kCg>(A, B, C, M, N, K, s);
        else grid<128, 256, kCg>(A, B, C, M, N, K, s);
        return;
    }
    if (p.kind == step6::kSplit) split<kCg>(A, B, C, M, N, K, s);
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
    if (v < 0 || v > 8 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step6 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    if (v == 4) {   // step 5
        if (step5::choose(M, true) == 64) step5_grid<64>(A, B, C, M, N, K, s);
        else step5_grid<128>(A, B, C, M, N, K, s);
        return;
    }
    if (v >= 7) {   // v0's plan; v7: without 6e, v8: without 6f
        const step6::Plan p = step6::plan(M, N, K, ws.sms, true, true, true);
        if (p.kind == step6::kBands) {   // v0's bands (neither 6e nor 6f acts there)
            if (p.bn == 128) bands<128>(A, B, C, M, N, K, p.slices, true, s);
            else bands<256>(A, B, C, M, N, K, p.slices, true, s);
        } else {
            tiles_v(v, p, A, B, C, M, N, K, s);
        }
        return;
    }
    const bool cg = v != 1;   // v1: without 6a
    const step6::Plan p = step6::plan(M, N, K, ws.sms, v != 2, v != 3, v != 6);   // v2, v3, v6: without 6b, 6c, 6d
    if (p.kind == step6::kBands) {
        if (p.bn == 128) bands<128>(A, B, C, M, N, K, p.slices, v != 5, s);   // v5: B never marked evict-first
        else bands<256>(A, B, C, M, N, K, p.slices, v != 5, s);
    } else if (cg) {
        tiles<true>(p, A, B, C, M, N, K, s);
    } else {
        tiles<false>(p, A, B, C, M, N, K, s);
    }
}

// For verif.cu: after a call (synchronized), the counters of 6b, 6c and 6d's waves that are not back at zero.
int candidate_dirty()
{
    if (!ws.sms) return 0;
    const int n = ws.counters();
    unsigned* h = (unsigned*)malloc(n * sizeof(unsigned));
    cudaMemcpy(h, ws.words, n * sizeof(unsigned), cudaMemcpyDeviceToHost);
    int dirty = 0;
    for (int i = 0; i < n; i++) dirty += h[i] != 0;
    free(h);
    unsigned w = 0;
    cudaMemcpy(&w, ws.waveSync, sizeof(unsigned), cudaMemcpyDeviceToHost);
    return dirty + (w != 0);
}
