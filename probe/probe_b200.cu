// probe_b200.cu (B200): the interleaved probe of the Blackwell rung (the B200 counterpart of probe/probe_t4.cu), with the rung's
// step variants as arms. Correctness, time, energy (NVML), clock and power per arm, in one process, in sustained
// interleaved blocks: CUDA graphs of about 20 ms alternating 3 input sets (the judge's law: uniform in [-1, 1)), 3 rounds in a
// random order, each block preceded by a warm-up of the same length; NVML sampled in a thread. The PDL edges between calls are
// neutralized in the graphs, as in the judge (cuBLAS chains its calls by PDL on the B200).
// Arms: 'ref<i>' = cuBLASLt configuration of rank i in the judge's cache (judge/reference/b200.txt), for this exact card line
// only (refused if the cache has none);
// 'defp' = cublasGemmEx's default, user handle; 'def64' = the same with 64 MB of workspace; 'cand' = candidate_gemm;
// 'v<i>' = variant i of the step (g_var, when the step defines variants).
// Correctness of each arm against cublasGemmEx with fp32 output (max error / max |ref|, the judge's tolerance 2e-3), on 2 calls,
// and again after the measurement on the measured graph's output.
// Usage: ./probe cache.txt M N K arm1,arm2,... [seconds]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s);
// The variant a call runs, and how many there are (a step without variants leaves the defaults); v<i> past g_nvar is refused.
__attribute__((weak)) int g_var = 0;
__attribute__((weak)) int g_nvar = 1;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); fflush(stdout); exit(3); } } while (0)
#define LT(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS %s:%d status %d\n", __FILE__, __LINE__, (int)s_); fflush(stdout); exit(3); } } while (0)

__global__ void fill(half* p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned x = ((unsigned)i * 2654435761u) ^ (seed * 0x9e3779b9u) ^ (unsigned)(i >> 32);
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        p[i] = __float2half((x >> 8) * (1.0f / 8388608.0f) - 1.0f);
    }
}
__global__ void gap(const half* __restrict__ out, const float* __restrict__ ref, size_t n, unsigned* res) {
    float e = 0.f, r = 0.f; bool bad = false;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float d = fabsf(__half2float(out[i]) - ref[i]);
        if (!isfinite(d)) bad = true; else e = fmaxf(e, d);
        r = fmaxf(r, fabsf(ref[i]));
    }
    for (int o = 16; o > 0; o >>= 1) {
        e = fmaxf(e, __shfl_xor_sync(0xffffffffu, e, o)); r = fmaxf(r, __shfl_xor_sync(0xffffffffu, r, o));
        bad |= __shfl_xor_sync(0xffffffffu, (int)bad, o);
    }
    if ((threadIdx.x & 31) == 0) { atomicMax(&res[0], __float_as_uint(e)); atomicMax(&res[1], __float_as_uint(r)); if (bad) atomicOr(&res[2], 1u); }
}
struct Nv {
    void* d = nullptr; bool ok = false;
    int (*en)(void*, unsigned long long*) = nullptr; int (*ck)(void*, int, unsigned*) = nullptr; int (*pw)(void*, unsigned*) = nullptr;
    int (*pl)(void*, unsigned*) = nullptr; int (*rs)(void*, unsigned long long*) = nullptr;
    void init() {
        void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW); if (!h) return;
        auto ini = (int (*)())dlsym(h, "nvmlInit_v2"); auto get = (int (*)(unsigned, void**))dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
        en = (int (*)(void*, unsigned long long*))dlsym(h, "nvmlDeviceGetTotalEnergyConsumption");
        ck = (int (*)(void*, int, unsigned*))dlsym(h, "nvmlDeviceGetClockInfo"); pw = (int (*)(void*, unsigned*))dlsym(h, "nvmlDeviceGetPowerUsage");
        pl = (int (*)(void*, unsigned*))dlsym(h, "nvmlDeviceGetEnforcedPowerLimit");
        rs = (int (*)(void*, unsigned long long*))dlsym(h, "nvmlDeviceGetCurrentClocksEventReasons");
        ok = ini && get && en && ck && pw && ini() == 0 && get(0, &d) == 0;
    }
};

int main(int argc, char** argv) {
    if (argc < 6) { printf("usage: ./probe cache.txt M N K arm,... [seconds]\n"); return 1; }
    const std::string cache = argv[1];
    const int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]);
    std::vector<std::string> arms; { std::stringstream ss(argv[5]); std::string t; while (std::getline(ss, t, ',')) arms.push_back(t); }
    const double block_s = argc > 6 ? atof(argv[6]) : 1.0;
    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
    half *A3[3], *B3[3], *C; CK(cudaMalloc(&C, nC * 2));
    for (int j = 0; j < 3; ++j) {
        CK(cudaMalloc(&A3[j], nA * 2)); CK(cudaMalloc(&B3[j], nB * 2));
        fill<<<1024, 256>>>(A3[j], nA, 11 + 2 * j); fill<<<1024, 256>>>(B3[j], nB, 12 + 2 * j);
    }
    CK(cudaDeviceSynchronize());
    const half *A = A3[0], *B = B3[0];
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cublasHandle_t hp, hb; LT(cublasCreate(&hp)); LT(cublasSetStream(hp, s));
    LT(cublasCreate(&hb)); LT(cublasSetStream(hb, s));
    void* ws; const size_t wsz = 64u << 20; CK(cudaMalloc(&ws, wsz)); LT(cublasSetWorkspace(hb, ws, wsz));
    cublasLtHandle_t lt; LT(cublasLtCreate(&lt));
    cublasLtMatmulDesc_t op; LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasLtMatrixLayout_t la, lb, lc; LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, N, K, N)); LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, M, K));
    LT(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, N, M, N));
    void* ws2; CK(cudaMalloc(&ws2, wsz));
    Nv nv; nv.init();
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); unsigned cap = 0; if (nv.ok && nv.pl) nv.pl(nv.d, &cap);
    int driver = 0; cudaDriverGetVersion(&driver);
    std::string card = p.name; for (char& c : card) if (c == ' ' || c == ',' || c == '|') c = '_';
    card += "|" + std::to_string(cap / 1000) + "W|lt" + std::to_string(cublasLtGetVersion()) + "|cuda" + std::to_string(driver);
    std::vector<std::pair<int, cublasLtMatmulAlgo_t>> memo_card, memo_other;
    { std::ifstream f(cache); std::string l;
      while (std::getline(f, l)) {
          std::stringstream ss(l); std::string word, label, c, h; int m_, n_, k_, rank; double trel;
          if (!(ss >> word >> label >> m_ >> n_ >> k_ >> c >> rank >> trel >> h) || word != "REFERENCE" || m_ != M || n_ != N || k_ != K) continue;
          cublasLtMatmulAlgo_t a; if (h.size() != 2 * sizeof a) continue;
          unsigned char* q = reinterpret_cast<unsigned char*>(&a);
          for (size_t i = 0; i < sizeof a; ++i) q[i] = (unsigned char)std::stoi(h.substr(2 * i, 2), nullptr, 16);
          (c == card ? memo_card : memo_other).push_back({rank, a});
      } }
    auto& memo = memo_card;   // the configurations of this exact card line only: ref<i> is refused without them
    std::stable_sort(memo.begin(), memo.end(), [](auto& a, auto& b) { return a.first < b.first; });
    printf("card %s; shape %d x %d x %d; bytes A+B+C %.1f MB; cache: %zu configurations of this card, %zu of others\n", card.c_str(),
           M, N, K, (nA + nB + nC) * 2 / 1e6, memo_card.size(), memo_other.size());
    const float one = 1.f, zero = 0.f;
    auto call = [&](const std::string& b) {
        if (b == "cand") candidate_gemm(A, B, C, M, N, K, s);
        else if (b == "defp") LT(cublasGemmEx(hp, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &zero, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        else if (b == "def64") LT(cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &zero, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        else if (b == "ref" || (b.rfind("ref", 0) == 0 && b.size() > 3 && std::all_of(b.begin() + 3, b.end(), [](char c) { return isdigit((unsigned char)c) != 0; }))) {
            const int i = b.size() > 3 ? atoi(b.c_str() + 3) : 0;
            if (i >= (int)memo.size()) { printf("arm %s: no configuration of this rank for the card line %s in the cache\n", b.c_str(), card.c_str()); exit(3); }
            LT(cublasLtMatmul(lt, op, &one, B, la, A, lb, &zero, C, lc, C, lc, &memo[i].second, ws2, wsz, s));
        }
        else if (b.size() >= 2 && b[0] == 'v' && std::all_of(b.begin() + 1, b.end(), [](char c) { return isdigit((unsigned char)c) != 0; })) {
            if (atoi(b.c_str() + 1) >= g_nvar) { printf("arm %s: the candidate has %d variant(s)\n", b.c_str(), g_nvar); exit(3); }
            g_var = atoi(b.c_str() + 1); candidate_gemm(A, B, C, M, N, K, s); g_var = 0;
        }
        else { printf("unknown arm %s\n", b.c_str()); exit(3); }
    };
    if (!nv.ok) { printf("NVML unavailable\n"); return 2; }
    float* R32; CK(cudaMalloc(&R32, nC * 4)); unsigned* dres; CK(cudaMalloc(&dres, 12));
    LT(cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &zero, R32, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CK(cudaStreamSynchronize(s));
    auto error_of = [&]() {
        CK(cudaMemsetAsync(dres, 0, 12, s)); gap<<<1024, 256, 0, s>>>(C, R32, nC, dres);
        unsigned h[3]; CK(cudaMemcpyAsync(h, dres, 12, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s));
        if (h[2]) return (double)INFINITY;
        float e, r; memcpy(&e, &h[0], 4); memcpy(&r, &h[1], 4); return (double)e / r;
    };
    std::vector<double> errs(arms.size(), -1);
    std::vector<cudaGraphExec_t> gx(arms.size()); std::vector<int> G(arms.size());
    for (size_t i = 0; i < arms.size(); ++i) {
        double e = 0;
        for (int r = 0; r < 2; ++r) {                                     // 2 calls: a state left by the first would show in the second
            CK(cudaMemsetAsync(C, 0xff, nC * 2, s));
            call(arms[i]);
            if (cudaStreamSynchronize(s) != cudaSuccess || cudaGetLastError() != cudaSuccess) { printf("arm %s: CUDA error\n", arms[i].c_str()); return 3; }
            e = std::max(e, error_of());
        }
        errs[i] = e;
        auto t0 = std::chrono::steady_clock::now(); for (int r = 0; r < 10; ++r) call(arms[i]); CK(cudaStreamSynchronize(s));
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 10;
        G[i] = std::max(3, std::min(600, (int)(0.02 / dt)));
        // A graph of G calls, the PDL edges between two calls NEUTRALIZED as in the judge (judge/judge.cu, Bench::graph): without
        // that, cuBLAS's references (PDL between calls on the B200) would be timed otherwise than in the judge.
        cudaGraph_t g; std::set<cudaGraphNode_t> boundaries;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int r = 0; r < G[i]; ++r) {
            A = A3[r % 3]; B = B3[r % 3]; call(arms[i]);
            if (r + 1 < G[i]) {
                cudaStreamCaptureStatus st; const cudaGraphNode_t* deps = nullptr; size_t nd = 0;
                if (cudaStreamGetCaptureInfo(s, &st, nullptr, nullptr, &deps, nullptr, &nd) == cudaSuccess)
                    for (size_t q = 0; q < nd; ++q) boundaries.insert(deps[q]);
            }
        }
        A = A3[0]; B = B3[0];
        CK(cudaStreamEndCapture(s, &g));
        {
            int npdl = 0; size_t ne = 0;
            cudaGraphGetEdges(g, nullptr, nullptr, nullptr, &ne);
            std::vector<cudaGraphNode_t> from(ne), to(ne); std::vector<cudaGraphEdgeData> edge_data(ne);
            if (ne && cudaGraphGetEdges(g, from.data(), to.data(), edge_data.data(), &ne) == cudaSuccess)
                for (size_t q = 0; q < ne; ++q)
                    if (boundaries.count(from[q]) && (edge_data[q].type != 0 || edge_data[q].from_port != 0)) {
                        ++npdl;
                        if (cudaGraphRemoveDependencies(g, &from[q], &to[q], &edge_data[q], 1) != cudaSuccess ||
                            cudaGraphAddDependencies(g, &from[q], &to[q], nullptr, 1) != cudaSuccess) {
                            printf("a PDL edge could not be neutralized (arm %s)\n", arms[i].c_str()); return 3;
                        }
                    }
            cudaGetLastError();
            if (npdl) printf("arm %s: %d PDL edge(s) between calls neutralized, as in the judge\n", arms[i].c_str(), npdl);
        }
        CK(cudaGraphInstantiate(&gx[i], g, 0)); cudaGraphDestroy(g);
    }
    std::atomic<int> active{-1}; std::atomic<bool> done{false}; std::mutex mx;
    std::vector<double> acc_mhz(arms.size()), acc_pw(arms.size()); std::vector<long> acc_n(arms.size()); std::vector<unsigned long long> acc_reasons(arms.size());
    std::thread th([&] { while (!done) { const int b = active; if (b >= 0) { unsigned c = 0, q = 0; unsigned long long r = 0; nv.ck(nv.d, 1, &c); nv.pw(nv.d, &q);
        if (nv.rs) nv.rs(nv.d, &r);
        std::lock_guard<std::mutex> l(mx); if (active == b) { acc_mhz[b] += c; acc_pw[b] += q / 1000.0; acc_n[b]++; acc_reasons[b] |= r; } }
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); } });
    std::vector<std::vector<double>> energy(arms.size()), t_us(arms.size());
    std::vector<size_t> order(arms.size()); for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    unsigned seed = 12345u;
    for (int round_i = 0; round_i < 3; ++round_i) {
        for (size_t i = order.size(); i > 1; --i) { seed = seed * 1103515245u + 12345u; std::swap(order[i - 1], order[(seed >> 8) % i]); }
        for (size_t i : order) {
            auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < block_s) { CK(cudaGraphLaunch(gx[i], s)); CK(cudaStreamSynchronize(s)); }
            active = (int)i;
            unsigned long long e0 = 0, e1 = 0; nv.en(nv.d, &e0); long ng = 0; t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < block_s) { for (int q = 0; q < 4; ++q) CK(cudaGraphLaunch(gx[i], s)); ng += 4; CK(cudaStreamSynchronize(s)); }
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); nv.en(nv.d, &e1);
            active = -1;
            energy[i].push_back((double)(e1 - e0) / (ng * G[i])); t_us[i].push_back(dt / (ng * G[i]) * 1e6);
        }
    }
    done = true; th.join();
    // Correctness AFTER the measurement, on the measured graph's output (its last call reads input set (G - 1) % 3), 3 times: an
    // arm whose split of the work depends on the run must be correct at each pass, not only in an isolated call.
    std::vector<double> errs2(arms.size(), -1);
    for (size_t i = 0; i < arms.size(); ++i) {
        const int j = (G[i] - 1) % 3;
        LT(cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B3[j], CUDA_R_16F, N, A3[j], CUDA_R_16F, K, &zero, R32, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        double e = 0;
        for (int r = 0; r < 3; ++r) { CK(cudaMemsetAsync(C, 0xff, nC * 2, s)); CK(cudaGraphLaunch(gx[i], s)); CK(cudaStreamSynchronize(s)); e = std::max(e, error_of()); }
        errs2[i] = e;
    }
    const double flop = 2.0 * M * N * K, bytes = (nA + nB + nC) * 2.0;
    for (size_t i = 0; i < arms.size(); ++i) {
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        const double n_ = acc_n[i] ? acc_n[i] : 1, t = med(t_us[i]);
        printf("ARM %-14s : %9.2f us | %7.1f TFLOP/s | %6.0f GB/s | %.3f mJ | SM %4.0f MHz | %5.1f W | 0x%llx | G %d | rounds %.2f %.2f %.2f | err %.2e / graph %.2e%s\n",
               arms[i].c_str(), t, flop / (t * 1e6), bytes / (t * 1e3), med(energy[i]), acc_mhz[i] / n_, acc_pw[i] / n_, acc_reasons[i], G[i],
               t_us[i][0], t_us[i][1], t_us[i][2], errs[i], errs2[i], (errs[i] <= 2e-3 && errs2[i] <= 2e-3) ? "" : "  WRONG");
    }
    return 0;
}
