// Checks a step (linked with it) on small shapes with edges in M against a CPU reference, for
// compute-sanitizer; or runs it once at one shape, next to one call of cuBLASLt, for ncu. A step with variants (it
// defines g_var and g_nvar, as step1.cu) is checked, or run, in each of them: v0, v1, ... A step that keeps state
// across calls (it defines candidate_dirty, as step6.cu) must leave it as it found it: the check fails a call after
// which candidate_dirty counts words of its state not back at zero. A call that has not finished after kHangSeconds,
// or past the run's budget (GEMM_LADDER_BUDGET, set by session.py; a grid barrier that never opens, for instance),
// ends the check: HANG, with its shape and variant, exit code 1. The output goes out line by line, so that a killed
// run keeps what it has checked.
//   ./verif                       shapes 16 x 256 x 64, 144 x 256 x 64, 128 x 512 x 128, 208 x 512 x 192,
//                                 272 x 5376 x 128, 1168 x 512 x 64, 144 x 512 x 256, 528 x 2560 x 64,
//                                 528 x 5376 x 64, 16 x 1024 x 1024, 16 x 6144 x 128
//   ./verif M N K [hex]           one call of each variant of the step, then one of cuBLASLt (fp16 output), at
//                                 M x N x K, on inputs drawn like the judge's (uniform in [-1, 1)); with hex (a
//                                 cublasLtMatmulAlgo_t as the judge stores it in its cache), cuBLASLt runs that exact
//                                 configuration (the best one the judge found), otherwise its heuristic's first choice
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

// A check's call, waited for on the host: past kHangSeconds, or past the run's budget, the call is taken as hung. The
// budget, GEMM_LADDER_BUDGET (seconds from the start of main), is set by modal/session.py for each run of the check, the
// plain run and each tool of compute-sanitizer: the session's time limit for that run on that card, less 60 s. So the
// HANG diagnostic always comes before the session cuts the run, on every card and under every tool (the rule is written
// in session.py too, next to its limits). Without it (a run by hand), kHangSeconds alone.
constexpr int kHangSeconds = 600;
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

// The judge's fill: uniform in [-1, 1) (judge/judge.cu, fill).
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
    // N is a multiple of 256 everywhere (the 128 x 256 tile). 16 x 256 x 64 and 144 x 256 x 64: a tile cut by M (16 rows
    // left); 128 x 512 x 128: four K32 slices, so each stage of a double buffer is filled twice; 208 x 512 x 192: an edge
    // past the first half tile of the output (80 rows left), three K64 slices; 272 x 5376 x 128: 63 tiles of 128 x 256
    // (126 of 128 x 128) for 40 blocks, so a persistent grid moves on to further tiles (the sanitizer sees that
    // transition), some of them in the edge row; 1168 x 512 x 64: 10 rows of tiles, so that step 3's groups of 8
    // rows (3c) end with a group of 2 rows, the second of them cut by M (16 rows left). Step 6's mechanisms (the
    // last five): 144 x 512 x 256, K cut in two (6d) at a tile cut by M; 528 x 2560 x 64, 50 tiles of 128 x 256 in
    // two waves (6b), one barrier of the whole grid per call; 528 x 5376 x 64, 105 tiles in three waves, two barriers
    // per call, the last row of tiles cut by M; 16 x 1024 x 1024, 4 bands of 256 columns, each in 12 slices of K
    // summed by the last block (6c); 16 x 6144 x 128, 48 bands of 128 columns without slices (6c). Of these,
    // 144 x 512 x 256 reaches 6d's counters (as 128 x 512 x 128 does), 528 x 2560 x 64 and 528 x 5376 x 64 6b's
    // barrier (1168 x 512 x 64 is one wave), 16 x 1024 x 1024 6c's counters; candidate_dirty checks them after each
    // call. A counter of the barrier left non-zero by a call is reported at 528 x 2560 x 64, where the call has one
    // barrier; where it has two, the second hangs (HANG).
    const int shapes[][3] = {{16, 256, 64}, {144, 256, 64}, {128, 512, 128}, {208, 512, 192}, {272, 5376, 128},
                             {1168, 512, 64}, {144, 512, 256}, {528, 2560, 64}, {528, 5376, 64}, {16, 1024, 1024},
                             {16, 6144, 128}};
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
            if (g_nvar > 1) printf("shape %d x %d x %d, v%d: max error / max |ref| = %.2e\n", M, N, K, v, e / mref);
            else printf("shape %d x %d x %d: max error / max |ref| = %.2e\n", M, N, K, e / mref);
            worst = std::max(worst, std::isnan(e) ? 1e30 : e / mref);
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
