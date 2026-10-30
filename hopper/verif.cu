// The Hopper rung's verif.cu: the shared verif.cu of the rungs, plus eight repeated calls compared bit for bit.
// Checks a step (linked with it) on small shapes with edges in M against a CPU reference, for
// compute-sanitizer; or runs it once at one shape, next to one call of cuBLASLt, for ncu. A step with variants (it
// defines g_var and g_nvar, as step1.cu) is checked, or run, in each of them: v0, v1, ... A step that keeps state
// across calls (it defines candidate_dirty; none of this rung's steps does) must leave it as it found it: the check
// fails a call after which candidate_dirty counts words of its state not back at zero. A call that has not finished
// after kHangSeconds, or after the run's budget (GEMM_LADDER_BUDGET, set by the session; an mbarrier phase that never
// completes, for instance), ends the check: HANG, with its shape and variant, exit code 1. Each call is made kRepeats
// times, and its outputs must agree bit for bit (this rung's kernels sum in a fixed order, so outputs that differ from
// one call to the next show a race, among them those of the async proxy, which compute-sanitizer may not see). The
// output goes out line by line, so that a killed run keeps what it has checked.
//   ./verif                       shapes 16 x 256 x 64, 144 x 256 x 64, 128 x 512 x 128, 208 x 512 x 192,
//                                 272 x 5376 x 128, 1168 x 512 x 64, 144 x 512 x 256, 16 x 1024 x 1024,
//                                 16 x 6144 x 128, 144 x 512 x 320, 144 x 512 x 832, 2064 x 6656 x 320,
//                                 16 x 8704 x 64, 16 x 50944 x 64, 48 x 4096 x 128, 128 x 8448 x 128,
//                                 128 x 8704 x 128, 128 x 17152 x 64, 4224 x 2304 x 64, 1536 x 10496 x 320,
//                                 80 x 4096 x 640, 112 x 10240 x 448, 3344 x 1536 x 320, 512 x 10240 x 320,
//                                 16 x 28672 x 448, 16 x 65536 x 448
//   ./verif M N K [hex]           one call of each variant of the step, then one of cuBLASLt (fp16 output), at
//                                 M x N x K, on inputs drawn like the judge's (uniform in [-1, 1)); with hex (a
//                                 cublasLtMatmulAlgo_t as the judge stores it in its cache), cuBLASLt runs that
//                                 configuration (as the session passes it), otherwise its heuristic's first choice
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thread>
#include <unistd.h>

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s);
// The variant a call runs, and how many there are; a step without variants leaves these defaults.
__attribute__((weak)) int g_var = 0;
__attribute__((weak)) int g_nvar = 1;
// After a call (synchronized), the words of a step's state across calls left non-zero; a step without state leaves it
// undefined.
__attribute__((weak)) int candidate_dirty();

// A check's call, waited for on the host: after kHangSeconds, or after the run's budget, the call is taken as hung. The
// budget, GEMM_LADDER_BUDGET (seconds from the start of main), is set by the session for each run of the check, the
// plain run and each tool of compute-sanitizer: the session's time limit for that run on that card, less 60 s. So the
// budget ends about 60 s before that run's limit (the rule is written in the session's code too, next to its limits).
// Without it (a run by hand), kHangSeconds alone.
constexpr int kHangSeconds = 600;
constexpr int kRepeats = 8;   // the calls of each variant at each shape, compared bit for bit
static std::chrono::steady_clock::time_point g_start;
static int g_budget = 0;   // seconds from g_start; 0: no budget
static void wait_call(int M, int N, int K, int v)
{
    const auto t0 = std::chrono::steady_clock::now();
    while (cudaStreamQuery(0) == cudaErrorNotReady) {
        const auto now = std::chrono::steady_clock::now();
        if (now - t0 > std::chrono::seconds(kHangSeconds) || (g_budget > 0 && now - g_start > std::chrono::seconds(g_budget))) {
            const double waited = std::chrono::duration<double>(now - t0).count();
            if (g_budget > 0)
                printf("HANG: shape %d x %d x %d, v%d: the call has not finished after %.0f s (limits: %d s per call, a budget "
                       "of %d s for the run)\nFAILED\n", M, N, K, v, waited, kHangSeconds, g_budget);
            else
                printf("HANG: shape %d x %d x %d, v%d: the call has not finished after %.0f s (limit: %d s per call, no budget)"
                       "\nFAILED\n", M, N, K, v, waited, kHangSeconds);
            fflush(stdout);
            _exit(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static float rnd(unsigned& s) { s = s * 1664525u + 1013904223u; return ((s >> 9) & 0xffff) / 32768.0f - 1.0f; }

// The judge's fill: uniform in [-1, 1) (the judge's function fill).
__global__ void fill(half* p, size_t n, unsigned seed)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned x = ((unsigned)i * 2654435761u) ^ (seed * 0x9e3779b9u) ^ (unsigned)(i >> 32);
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        p[i] = __float2half((x >> 8) * (1.0f / 8388608.0f) - 1.0f);
    }
}
#define LT(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { printf("cuBLASLt %s:%d status %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);   // line by line, even into a pipe
    g_start = std::chrono::steady_clock::now();
    if (const char* b = getenv("GEMM_LADDER_BUDGET")) g_budget = atoi(b);
    if (argc == 4 || argc == 5) {
        const int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
        half *A, *B, *C; void* ws; const size_t wsMax = 64u << 20;
        CK(cudaMalloc(&A, (size_t)M * K * 2)); CK(cudaMalloc(&B, (size_t)K * N * 2)); CK(cudaMalloc(&C, (size_t)M * N * 2));
        CK(cudaMalloc(&ws, wsMax));
        fill<<<1024, 256>>>(A, (size_t)M * K, 1); fill<<<1024, 256>>>(B, (size_t)K * N, 2); CK(cudaDeviceSynchronize());
        for (int v = 0; v < g_nvar; ++v) {
            g_var = v;
            candidate_gemm(A, B, C, M, N, K, 0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        }
        g_var = 0;
        // cuBLASLt, as the judge calls it: C^T (N x M) = B^T (N x K) . A^T (K x M), column-major.
        cublasLtHandle_t h; LT(cublasLtCreate(&h));
        cublasLtMatmulDesc_t op; LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasLtMatrixLayout_t la, lb, lc;
        LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, N, K, N)); LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, M, K));
        LT(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, N, M, N));
        cublasLtMatmulAlgo_t algo;
        if (argc == 5) {
            const char* x = argv[4];
            if (strlen(x) != 2 * sizeof algo) { printf("hex: %zu digits expected\n", 2 * sizeof algo); return 1; }
            unsigned char* q = reinterpret_cast<unsigned char*>(&algo);
            auto v = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c - 'a' + 10; };
            for (size_t i = 0; i < sizeof algo; ++i) q[i] = (unsigned char)(v(x[2 * i]) * 16 + v(x[2 * i + 1]));
            cublasLtMatmulHeuristicResult_t chk; LT(cublasLtMatmulAlgoCheck(h, op, la, lb, lc, lc, &algo, &chk));
        } else {
            cublasLtMatmulPreference_t pref; LT(cublasLtMatmulPreferenceCreate(&pref));
            LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsMax, sizeof wsMax));
            cublasLtMatmulHeuristicResult_t r; int got = 0;
            LT(cublasLtMatmulAlgoGetHeuristic(h, op, la, lb, lc, lc, pref, 1, &r, &got));
            algo = r.algo;
        }
        const float one = 1.f, zero = 0.f;
        LT(cublasLtMatmul(h, op, &one, B, la, A, lb, &zero, C, lc, C, lc, &algo, ws, wsMax, 0));
        CK(cudaDeviceSynchronize());
        printf("once: done (cuBLASLt %s)\n", argc == 5 ? "configuration given" : "heuristic");
        return 0;
    }
    // N is a multiple of 256 everywhere (the 128 x 256 tile). 16 x 256 x 64 and 144 x 256 x 64: a tile cut by M (16
    // rows left); 128 x 512 x 128: two K64 slices, each stage of step 1's double buffer filled once; 208 x 512 x 192:
    // an edge in the second half tile of the output (80 rows left), three K64 slices; 272 x 5376 x 128: 63 tiles of 128
    // x 256 in three rows, the last one cut by M; 1168 x 512 x 64: 10 rows of tiles, the last one cut by M (16 rows
    // left); 144 x 512 x 256: four K64 slices at a tile cut by M; 16 x 1024 x 1024 and 16 x 6144 x 128: bands of one
    // column of 64 (16 and 96 bands), of 16 and 2 slices. 144 x 512 x 320 and 144 x 512 x 832, 5 and 13 slices of K64,
    // turn a ring of 4 stages once and three times, and one slice more (step 3); 2064 x 6656 x 320, 17 rows of 26 tiles
    // in pairs (9 rows of pairs, the last one cut by M), 234 pairs for 66 clusters (step 5's persistent grid): 3 or 4
    // pairs per cluster, each of 5 slices, so that the ring's stages and phases run on from one pair to the next,
    // starting each pair at another stage; two groups of step 5's order, the second from the last column back. Step 6,
    // one shape per branch of its rule on the H100 (132 SMs, 66 clusters of two), N still a multiple of 256: bands of J
    // = 2 and 7 columns of 64 (16 x 8704 x 64, 16 x 50944 x 64; J = 1 at 16 x 1024 x 1024 and 16 x 6144 x 128 above);
    // 64 x 128 alone (48 x 4096 x 128); 64 x 128 in pairs at 66 columns of 128, 128 x 128 at 68, step 5's tile above
    // 132 (128 x 8448, 8704, 17152); step 5's tile alone (6c) at every shape with M > 128, among them persistent, 162
    // tiles for 132 blocks, the last row cut by M (16 rows left), 5 slices of K64 each for its 4 stages (3344 x 1536 x
    // 320). Step 7, on the H100: the bands of 64 rows in pairs along M, the second block's rows cut by M (16 left), 10
    // slices of K64 for 8 stages (80 x 4096 x 640); the bands of 128 rows alone (7d), cut by M, 7 slices for 6 stages
    // (112 x 10240 x 448); 7e, 128 x 320, with a last column of tiles of 64 columns (4224 x 2304 x 64) and persistent,
    // 198 pairs of tiles for 66 clusters, 5 slices of K64 each for its 3 stages, a last column of 256
    // (1536 x 10496 x 320), and 189 pairs for 66 clusters (2064 x 6656 x 320), alone, its 128 tiles in one wave, 5
    // slices of K64 (512 x 10240 x 320), and in pairs there too (v5). Step 7's other classes run at shapes above: the
    // bands of 64 rows in pairs at 128 x 512 and 128 x 8448, of 128 rows alone at 128 x 8704, the bands of 16 rows
    // twice as wide in clusters of eight along N (7b, a GPC for each) at 16 x 256, 16 x 1024, 16 x 6144 and 16 x 8704
    // (the last cluster partly outside the matrix (columns N and on) at 16 x 256 and 16 x 8704); at 16 x 50944, 13
    // clusters of eight would put two in some GPCs: the 16-row bands alone at step 6's J (J = 7). The bands of 8
    // columns of 64 (3 stages, 7 slices of K64): 7b at 16 x 28672 x 448 (7 clusters of eight), the 16-row bands alone
    // and 6a at 16 x 65536 x 448 (16 clusters of eight would put two in some GPCs). Step 0's grid (one block per SM)
    // moves on to further tiles at the shapes with more than 132 tiles of 128 x 128.
    const int shapes[][3] = {{16, 256, 64}, {144, 256, 64}, {128, 512, 128}, {208, 512, 192}, {272, 5376, 128},
                             {1168, 512, 64}, {144, 512, 256}, {16, 1024, 1024}, {16, 6144, 128}, {144, 512, 320},
                             {144, 512, 832}, {2064, 6656, 320}, {16, 8704, 64}, {16, 50944, 64}, {48, 4096, 128},
                             {128, 8448, 128}, {128, 8704, 128}, {128, 17152, 64}, {4224, 2304, 64},
                             {1536, 10496, 320}, {80, 4096, 640}, {112, 10240, 448}, {3344, 1536, 320},
                             {512, 10240, 320}, {16, 28672, 448}, {16, 65536, 448}};
    double worst = 0;
    for (auto& sh : shapes) {
        const int M = sh[0], N = sh[1], K = sh[2];
        std::vector<half> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
        unsigned s = 7;
        for (auto& v : hA) v = __float2half(rnd(s));
        for (auto& v : hB) v = __float2half(rnd(s));
        std::vector<double> ref((size_t)M * N, 0.0);   // in double, from the fp16 inputs; once for all the variants
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) {
                const double a = __half2float(hA[(size_t)i * K + k]);
                for (int j = 0; j < N; ++j) ref[(size_t)i * N + j] += a * __half2float(hB[(size_t)k * N + j]);
            }
        double mref = 0;
        for (const double r : ref) mref = std::max(mref, std::fabs(r));
        half *A, *B, *C;
        CK(cudaMalloc(&A, hA.size() * 2)); CK(cudaMalloc(&B, hB.size() * 2)); CK(cudaMalloc(&C, hC.size() * 2));
        CK(cudaMemcpy(A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(B, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));
        for (int v = 0; v < g_nvar; ++v) {
            g_var = v;
            CK(cudaMemset(C, 0xff, hC.size() * 2));
            candidate_gemm(A, B, C, M, N, K, 0);
            CK(cudaGetLastError());
            wait_call(M, N, K, v);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hC.data(), C, hC.size() * 2, cudaMemcpyDeviceToHost));
            double e = 0;
            for (size_t i = 0; i < hC.size(); ++i) {
                const double d = std::fabs(__half2float(hC[i]) - ref[i]);
                if (std::isnan(d)) { e = NAN; break; }   // an element left unwritten (or NaN): failure
                if (d > e) e = d;
            }
            const std::vector<half> first = hC;   // the next calls must give it again, bit for bit
            long differ = 0;
            for (int r = 1; r < kRepeats; ++r) {
                CK(cudaMemset(C, 0xff, hC.size() * 2));
                candidate_gemm(A, B, C, M, N, K, 0);
                CK(cudaGetLastError());
                wait_call(M, N, K, v);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(hC.data(), C, hC.size() * 2, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < hC.size(); ++i)
                    differ += memcmp(&hC[i], &first[i], sizeof(half)) != 0;
            }
            if (g_nvar > 1) printf("shape %d x %d x %d, v%d: max error / max |ref| = %.2e\n", M, N, K, v, e / mref);
            else printf("shape %d x %d x %d: max error / max |ref| = %.2e\n", M, N, K, e / mref);
            worst = std::max(worst, std::isnan(e) ? 1e30 : e / mref);
            if (differ) {
                printf("  FAILED: %ld elements differ, bit for bit, between the %d calls (a race)\n", differ, kRepeats);
                worst = 1e30;
            }
            if (&candidate_dirty != nullptr) {   // a step with state across calls: back as the call found it
                const int dirty = candidate_dirty();
                if (dirty) {
                    printf("  FAILED: %d words of the step's state left non-zero by the call\n", dirty);
                    worst = 1e30;
                }
            }
        }
        g_var = 0;
        cudaFree(A); cudaFree(B); cudaFree(C);
    }
    printf("%s (tolerance 2e-3)\n", worst <= 2e-3 ? "OK" : "FAILED");
    return worst <= 2e-3 ? 0 : 1;
}
