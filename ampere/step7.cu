// Step 7 of the Ampere rung, as a candidate for the judge: split-K, parts reduced by their own blocks, L2 hints (see
// step7.cuh). Its variants, for the interleaved probe (arm v<i> sets g_var) and for verif.cu (g_nvar): v0 = step 7 (7a
// to 7h), v1 = without 7a (no split of K, and so without 7b, 7c, 7f and 7g), v2 = without 7b (7a's reduction kernel,
// and so without 7g), v3 = without 7c, v4 = without 7d, v5 = step 6 itself (the anchor), v13 = without 7e, v14 =
// without 7f, v15 = without 7g, v16 = without 7h; for verif.cu, whose shapes are too small for the rules to choose 7a's
// parts, 7c, 6b's stream, 6c, 7e or 7f: v6 = K in 3 parts at every shape M > 16 where the rules choose none of 6c, 7e
// and 7f, with 7b and 7g where the parts fit on the SMs, v7 = 7c's parts in place on 128 x 256 in 2 parts at every
// shape where 7a adds parts in place and the rules choose neither 6c nor 7f, v8 = 6b's stream at every shape M <= 16,
// v9 = 6c at every shape M > 16 (v8 and v9 with 7d's hints), v10 = v6 without 7d (7b and 7g without the hints, as in
// v4), v11 = 7a's parts at every shape, the plan alone (K in 3 on 128 x 256, 7a's reduction kernel), v12 = 7a in place
// at every shape, the plan alone (K in 2, 128 x 128), v17 = v6 without 7g (7b with 7d's hints, as in v15), v18 = v12
// without 7h, v19 = v9 without 7h, v20 = 7e's tail in 2 parts at every shape M > 16 whose 128 x 256 tiles leave a
// partial last wave, where the rules choose none of 6b's kernels, 6c and 7f.
// candidate_dirty, for verif.cu: the counters and flags left non-zero by a call (each call must leave them at zero, as
// it found them).
#include <cstdio>
#include <cstdlib>
#include "step5.cuh"
#include "step6.cuh"
#include "step7.cuh"

int g_var = 0;
int g_nvar = 21;

static void check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        fprintf(stderr, "step7: %s: %s\n", what, cudaGetErrorString(e));
        abort();
    }
}

template <class Kernel>
static void allow_smem(Kernel kernel, int bytes)   // above 48 KB, a kernel's dynamic shared memory must be allowed once
{
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

// 7a's workspace, allocated at its first use and never freed (a graph captured later keeps valid pointers): the parts,
// 3 x 128 x 256 halves for each of max(4 sms, 128) tiles, and the in-place flags, one per tile of 128 x 128 for up to 8
// sms tiles, zeroed on the candidate's stream before its first kernel, then zero between calls (the last part of each
// tile puts its flag back to zero); 7b's counters, two per tile for at most sms tiles (S x tiles <= sms), likewise. The
// rule splits fewer than 2/3 sms tiles into parts and keeps in place to 6a's three waves of 128 x 256 (6 sms tiles of
// 128 x 128); the sizes cover also the variants that force parts or in place at the session's shape, the ladder (4 sms
// tiles of 128 x 256 and 8 sms of 128 x 128 on 108 SMs), and verif.cu's shapes (128 tiles). A launch that would need
// more stops.
struct Workspace { half* parts = nullptr; unsigned* flags = nullptr; unsigned* counts7b = nullptr; long partTiles = 0;
                   int nflags = 0, ncounts7b = 0; };
static Workspace ws;
static Workspace workspace(int sms, cudaStream_t s)
{
    if (!ws.parts) {
        ws.partTiles = 4 * sms > 128 ? 4 * sms : 128;
        ws.nflags = 8 * sms;
        check(cudaMalloc(&ws.parts, (size_t)3 * ws.partTiles * 128 * 256 * sizeof(half)), "cudaMalloc (parts)");
        check(cudaMalloc(&ws.flags, (size_t)ws.nflags * sizeof(unsigned)), "cudaMalloc (flags)");
        check(cudaMemsetAsync(ws.flags, 0, (size_t)ws.nflags * sizeof(unsigned), s), "cudaMemsetAsync (flags)");
        ws.ncounts7b = 2 * sms;   // 7b: two counters per tile, at most sms tiles (S x tiles <= sms)
        check(cudaMalloc(&ws.counts7b, (size_t)ws.ncounts7b * sizeof(unsigned)), "cudaMalloc (7b counters)");
        check(cudaMemsetAsync(ws.counts7b, 0, (size_t)ws.ncounts7b * sizeof(unsigned), s),
              "cudaMemsetAsync (7b counters)");
    }
    return ws;
}

template <int BM, int BN, int WM, int SK, int kStages, bool kHint>
static void launch(const step6::Plan& p, const half* A, const half* B, half* C, int M, int N, int K, int sms,
                   bool with7b, bool with7g, bool together, cudaStream_t s)
{
    using G = step6::Geometry<BM, BN, WM, SK, kStages>;
    static bool allowed = false;
    if (!allowed) {
        allow_smem(step6::gemm<BM, BN, WM, SK, kStages>, G::kSmemBytes);
        allow_smem(step6::gemm_split<BM, BN, WM, SK, kStages, step6::kParts>, G::kSmemBytes);
        allow_smem(step6::gemm_split<BM, BN, WM, SK, kStages, step6::kInPlace>, G::kSmemBytes);
        allow_smem(step6::gemm_split<BM, BN, WM, SK, kStages, step6::kInPlace, true>, G::kSmemBytes);
        allowed = true;
    }
    const int tiles = (M + BM - 1) / BM * (N / BN);
    if (p.split == step6::kWhole) {
        step6::gemm<BM, BN, WM, SK, kStages><<<tiles, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K);
        return;
    }
    const Workspace w = workspace(sms, s);
    if ((p.split == step6::kParts && (long)p.S * tiles * BM * BN > 3 * w.partTiles * 128 * 256) ||
        (p.split == step6::kInPlace && tiles > w.nflags)) {
        fprintf(stderr, "step7: %d x %d x %d needs more workspace than allocated\n", M, N, K);
        abort();
    }
    if (p.split == step6::kParts && with7b && step6::parallel_fits(tiles, p.S, sms)) {   // 7b, and 7g
        static bool allowed7b = false;
        if (!allowed7b) {
            allow_smem(step6::gemm_parallel<BM, BN, WM, SK, kStages, kHint>, G::kSmemBytes);
            allow_smem(step7::gemm_parallel_own<BM, BN, WM, SK, kStages, kHint>, G::kSmemBytes);
            allowed7b = true;
        }
        if (with7g)
            step7::gemm_parallel_own<BM, BN, WM, SK, kStages, kHint>
                <<<p.S * tiles, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K, p.S, w.parts, w.counts7b);
        else
            step6::gemm_parallel<BM, BN, WM, SK, kStages, kHint>
                <<<p.S * tiles, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K, p.S, w.parts, w.counts7b);
    } else if (p.split == step6::kParts) {
        step6::gemm_split<BM, BN, WM, SK, kStages, step6::kParts>
            <<<p.S * tiles, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K, p.S, w.parts, w.flags);
        step6::reduce_parts<BM, BN><<<tiles, 256, 0, s>>>(w.parts, C, M, N, p.S);
    } else if (together) {   // 7h: the reads of C together
        step6::gemm_split<BM, BN, WM, SK, kStages, step6::kInPlace, true>
            <<<p.S * tiles, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K, p.S, w.parts, w.flags);
    } else {
        step6::gemm_split<BM, BN, WM, SK, kStages, step6::kInPlace>
            <<<p.S * tiles, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K, p.S, w.parts, w.flags);
    }
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

// Step 6's plan (6a) with 7a's split of K, 7b (and 7g) in place of 7a's reduction kernel where the parts fit on the
// SMs, and 7c where its rule gives S; forceParts: K in 3 parts at every shape (v6, v10, v17), the 128-row tile then
// 128 x 256, as 7a's rule does; force7c: 7c in 2 parts wherever 7a adds parts in place (v7); together: 7h in the parts
// in place.
template <bool kHint>
static void step6_run(const half* A, const half* B, half* C, int M, int N, int K, int sms, bool splitK, bool with7b,
                      bool with7g, bool with7c, bool forceParts, bool force7c, bool together, cudaStream_t s)
{
    step6::Plan p = step6::choose(M, N, K, sms, true, splitK && !forceParts);
    if (forceParts) {
        p.S = 3;
        p.split = step6::kParts;
        if (p.tile == step6::k128x128) p.tile = step6::k128x256;
    }
    const int S7c = with7c ? step7::inplace_parts(M, N, K, sms) : 0;
    if (p.split == step6::kInPlace && (force7c || S7c)) {   // 7c: 128 x 256 in S parts in place
        p.tile = step6::k128x256;
        p.S = force7c ? 2 : S7c;
    }
    switch (p.tile) {
    case step6::k128x128:
        launch<128, 128, 64, 32, 5, kHint>(p, A, B, C, M, N, K, sms, with7b, with7g, together, s);
        break;
    case step6::k128x256:
        if (p.split == step6::kWhole) step5_launch(A, B, C, M, N, K, s);
        else launch<128, 256, 64, 64, 3, kHint>(p, A, B, C, M, N, K, sms, with7b, with7g, together, s);
        break;
    }
}

// 6b's workspace, allocated at the first call that needs it and never freed: the parts of the stream (16 x 256 halves
// for each of at most maxSegs segments of each band) and one counter per band, zeroed on the candidate's stream before
// its first kernel, then zero between calls; 6c's parts and counters share it. A launch that would need more stops.
struct Stream { half* parts = nullptr; unsigned* counts = nullptr; long partHalves = 0; int ncounts = 0; };
static Stream st7;
static Stream stream_workspace(cudaStream_t s)
{
    if (!st7.parts) {
        st7.partHalves = (long)1 << 23;   // 16 MB of parts
        st7.ncounts = 1 << 12;
        check(cudaMalloc(&st7.parts, (size_t)st7.partHalves * sizeof(half)), "cudaMalloc (stream parts)");
        check(cudaMalloc(&st7.counts, (size_t)st7.ncounts * sizeof(unsigned)), "cudaMalloc (stream counters)");
        check(cudaMemsetAsync(st7.counts, 0, (size_t)st7.ncounts * sizeof(unsigned), s),
              "cudaMemsetAsync (stream counters)");
    }
    return st7;
}

template <int BN, int kStages, bool kHint>
static void run_bands(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s)
{
    using G = step6::BandGeometry<BN, kStages>;
    static bool allowed = false;
    if (!allowed) {
        allow_smem(step6::gemm_bands<BN, kStages, kHint>, G::kSmemBytes);
        allowed = true;
    }
    step6::gemm_bands<BN, kStages, kHint><<<N / BN, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K);
}

template <int BN, int kStages, bool kHint>
static void run_stream(const half* A, const half* B, half* C, int M, int N, int K, int sms, cudaStream_t s)
{
    using G = step6::BandGeometry<BN, kStages>;
    static bool allowed = false;
    if (!allowed) {
        allow_smem(step6::gemm_stream<BN, kStages, kHint>, G::kSmemBytes);
        allowed = true;
    }
    const long bands = N / BN, steps = K / 64, U = bands * steps;
    const int P = U < sms ? (int)U : sms;
    const long per = U / P;                                 // each block holds at least per units
    const int maxSegs = (int)((steps + per - 1) / per) + 1;   // segments of one band, at most
    const Stream w = stream_workspace(s);
    if (bands * maxSegs * 16 * BN > w.partHalves || bands > w.ncounts) {
        fprintf(stderr, "step7: %d x %d x %d needs more stream workspace than allocated\n", M, N, K);
        abort();
    }
    step6::gemm_stream<BN, kStages, kHint>
        <<<P, G::kThreads, G::kSmemBytes, s>>>(A, B, C, M, N, K, w.parts, w.counts, maxSegs);
}

// 6c: P = sms rounded down to a multiple of MT blocks (MT rows of 128 x 256 tiles, one group per MT blocks); the
// parts in the stream's workspace (two slots of 128 x 256 halves per block), one counter per tile; together: 7h's sums.
template <bool kHint>
static void run_phase(const half* A, const half* B, half* C, int M, int N, int K, int sms, bool together,
                      cudaStream_t s)
{
    using G = step6::Geometry<128, 256, 64, 64, 3>;
    constexpr int kSmem = G::kSmemBytes + 16;   // and the last block's flag
    static bool allowed = false;
    if (!allowed) {
        allow_smem(step6::gemm_phase<128, 256, 64, 64, 3, kHint>, kSmem);
        allow_smem(step6::gemm_phase_sums<128, 256, 64, 64, 3, kHint>, kSmem);
        allowed = true;
    }
    const long U = (long)(N / 256) * (K / 64);   // units (column of tiles, slice of K64): at least one per group
    const int MT = (M + 127) / 128, P = (int)(sms / MT < U ? sms / MT : U) * MT;
    const Stream w = stream_workspace(s);
    if (P < MT || (long)P * 2 * 128 * 256 > w.partHalves || (long)MT * (N / 256) > w.ncounts) {
        fprintf(stderr, "step7: %d x %d x %d needs more stream workspace than allocated\n", M, N, K);
        abort();
    }
    if (together)   // 7h
        step6::gemm_phase_sums<128, 256, 64, 64, 3, kHint>
            <<<P, G::kThreads, kSmem, s>>>(A, B, C, M, N, K, w.parts, w.counts);
    else
        step6::gemm_phase<128, 256, 64, 64, 3, kHint>
            <<<P, G::kThreads, kSmem, s>>>(A, B, C, M, N, K, w.parts, w.counts);
}

// 7e's workspace, allocated at the first call that needs it and never freed: the parts of the tail (at most 8 sms parts
// of 128 x 256 halves: S <= 8 parts of R < sms tiles) and one counter per tile of the tail, zeroed on the candidate's
// stream before its first kernel, then zero between calls.
struct Tail { half* parts = nullptr; unsigned* counts = nullptr; long parts128x256 = 0; int ncounts = 0; };
static Tail tl7;
static Tail tail_workspace(int sms, cudaStream_t s)
{
    if (!tl7.parts) {
        tl7.parts128x256 = 8L * sms;
        tl7.ncounts = sms;
        check(cudaMalloc(&tl7.parts, (size_t)tl7.parts128x256 * 128 * 256 * sizeof(half)), "cudaMalloc (tail parts)");
        check(cudaMalloc(&tl7.counts, (size_t)tl7.ncounts * sizeof(unsigned)), "cudaMalloc (tail counters)");
        check(cudaMemsetAsync(tl7.counts, 0, (size_t)tl7.ncounts * sizeof(unsigned), s),
              "cudaMemsetAsync (tail counters)");
    }
    return tl7;
}

// 7e: the first tiles - R tiles whole, the last R (R = tiles mod sms) in S parts of K.
template <bool kHint>
static void run_tail(const half* A, const half* B, half* C, int M, int N, int K, int sms, int S, cudaStream_t s)
{
    using G = step6::Geometry<128, 256, 64, 64, 3>;
    constexpr int kSmem = G::kSmemBytes + 16;   // and the last part's flag
    static bool allowed = false;
    if (!allowed) {
        allow_smem(step6::gemm_tail<128, 256, 64, 64, 3, kHint>, kSmem);
        allowed = true;
    }
    const int tiles = (M + 127) / 128 * (N / 256), R = tiles % sms, whole = tiles - R;
    const Tail w = tail_workspace(sms, s);
    if (R == 0 || (long)S * R > w.parts128x256 || R > w.ncounts) {
        fprintf(stderr, "step7: %d x %d x %d, tail of %d tiles in %d parts: not supported\n", M, N, K, R, S);
        abort();
    }
    step6::gemm_tail<128, 256, 64, 64, 3, kHint><<<whole + R * S, G::kThreads, kSmem, s>>>(A, B, C, M, N, K, whole, S,
                                                                                         w.parts, w.counts);
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
    count(ws.flags, ws.nflags);
    count(ws.counts7b, ws.ncounts7b);
    count(st7.counts, st7.ncounts);
    count(tl7.counts, tl7.ncounts);
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
            fprintf(stderr, "step7: no SM count\n");
            abort();
        }
    }
    const int v = g_var;
    if (v < 0 || v > 20 || M <= 0 || M % 16 || N <= 0 || N % 256 || K <= 0 || K % 64) {
        fprintf(stderr, "step7 v%d: shape %d x %d x %d not supported\n", v, M, N, K);
        abort();
    }
    const bool together = v != 5 && v != 16 && v != 18 && v != 19;                          // 7h
    if (v == 11 || v == 12 || v == 18) {   // 7a forced, the plan alone: its parts with its reduction kernel (v11), in
                                           // place (v12, and v18 without 7h)
        step6::Plan p = step6::choose(M, N, K, sms, true, false);
        if (v == 11) {
            p.S = 3;
            p.split = step6::kParts;
            if (p.tile == step6::k128x128) p.tile = step6::k128x256;
        } else {
            p = {step6::k128x128, 2, step6::kInPlace};
        }
        if (p.tile == step6::k128x128)
            launch<128, 128, 64, 32, 5, true>(p, A, B, C, M, N, K, sms, false, false, together, s);
        else
            launch<128, 256, 64, 64, 3, true>(p, A, B, C, M, N, K, sms, false, false, together, s);
        return;
    }
    step6::M16 m = step6::m16(M, N, K, sms);                                             // 6b
    if (m != step6::kNoBands && v == 8) m = step6::kStream;                              // v8: the stream only
    const bool splitK = v != 1 && v != 5;                                                // 7a (v1: none; v5: step 6)
    const bool phase = v == 9 || v == 19 ? M > 16 : step6::phase(M, N, K, sms) ||       // 6c (v9, v19: forced)
                       (splitK && v != 14 && step7::phase_in_place(M, N, K, sms));      // 7f (v14: no 7f)
    const bool forceParts = v == 6 || v == 10 || v == 17;                                // K in 3 parts
    const bool with7b = v != 2, with7g = v != 2 && v != 15 && v != 17, with7c = v != 3, force7c = v == 7;
    const int tailTiles = (M + 127) / 128 * (N / 256) % sms;                             // 7e (v20: forced)
    const int tail = v == 20 ? (M > 16 && tailTiles ? 2 : 0) : v != 5 && v != 13 ? step6::tail_parts(M, N, K, sms) : 0;
    if (v != 4 && v != 5 && v != 10) {   // 7d
        if (m == step6::kBands) run_bands<128, 6, true>(A, B, C, M, N, K, s);
        else if (m == step6::kStream) run_stream<256, 4, true>(A, B, C, M, N, K, sms, s);
        else if (phase) run_phase<true>(A, B, C, M, N, K, sms, together, s);
        else if (tail) run_tail<true>(A, B, C, M, N, K, sms, tail, s);
        else step6_run<true>(A, B, C, M, N, K, sms, splitK, with7b, with7g, with7c, forceParts, force7c, together, s);
    } else {   // v4: without 7d (in 7e's and 7g's kernels too); v5: step 6 itself; v10: v6 without 7d
        if (m == step6::kBands) run_bands<128, 6, false>(A, B, C, M, N, K, s);
        else if (m == step6::kStream) run_stream<256, 4, false>(A, B, C, M, N, K, sms, s);
        else if (phase) run_phase<false>(A, B, C, M, N, K, sms, together, s);
        else if (tail) run_tail<false>(A, B, C, M, N, K, sms, tail, s);
        else step6_run<false>(A, B, C, M, N, K, sms, splitK, with7b, with7g, with7c, forceParts, force7c, together, s);
    }
}
