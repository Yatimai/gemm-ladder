// probe.cu: the interleaved probe, one source for every card of its launcher (probe/modal_probe.py, which builds it
// with a candidate for the card it is given). In one process, each arm is timed in sustained blocks, the arms
// interleaved. As in the judge (judge/judge.cu): the inputs drawn by its fill (uniform in [-1, 1)), three input sets
// alternated in the CUDA graphs, the programmatic dependencies (PDL) between two calls neutralized in the graphs
// (Bench::graph), and the correctness checked against cublasGemmEx with an fp32 output, at its tolerance (2e-3: WRONG
// over it). Here: graphs of about 20 ms (3 to 400 calls); per round and per arm, a warm-up of a third of a block, then
// the timed block; the order of the arms drawn at each round; the number of rounds is an argument (7 by default); the
// correctness of each arm checked before the timing (2 calls) and after it (the timed graph, 3 times); an unknown arm
// refused. Printed per arm: the time per call (the median of the rounds, to 0.01 us, then each round's), the energy per
// call (NVML's energy counter, the median of the rounds), the mean SM clock and power (NVML samples during the timed
// blocks, at least 5 ms apart).
// Arms: 'v<i>' (the candidate's variant i, g_var), 'twin' (v0 timed again) and 'x:<hex>' (the exact configuration of a
// REFERENCE line, a cublasLtMatmulAlgo_t in hexadecimal, as <rung>/verif.cu takes it; printed "ref0").
// Usage: ./probe M N K energy|ncu [arm1,arm2,...] [seconds] [rounds]. In ncu mode, only the first input set is drawn:
// two fill kernels, then each arm's two calls on that set (without a list of arms, the fill kernels alone). The
// launcher profiles one arm per process with the two fill kernels skipped: the arm's second call is ID 1 of its report,
// or IDs 2 and 3 when a call launches two kernels.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cmath>
#include <dlfcn.h>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s);
__attribute__((weak)) int g_var = 0;   // arm v<i>: the candidate's variant i (it defines g_var when it has variants)
__attribute__((weak)) int g_nvar = 1;  // the number of its variants; v<i> with i >= g_nvar is refused
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(3); } } while (0)
#define LT(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS %s:%d status %d\n", __FILE__, __LINE__, (int)s_); exit(3); } } while (0)
// The judge's fill (judge/judge.cu, fill): uniform in [-1, 1).
__global__ void fill(half* p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned x = ((unsigned)i * 2654435761u) ^ (seed * 0x9e3779b9u) ^ (unsigned)(i >> 32);
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        p[i] = __float2half((x >> 8) * (1.0f / 8388608.0f) - 1.0f);
    }
}
// The judge's gap (judge/judge.cu, gap): max |out - ref|, max |ref| and a flag if an element is not finite.
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
struct Nv { void* d = nullptr; int (*en)(void*, unsigned long long*) = nullptr; int (*ck)(void*, int, unsigned*) = nullptr; int (*pw)(void*, unsigned*) = nullptr; bool ok = false;
    void init() { void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW); if (!h) return;
        auto ini = (int (*)())dlsym(h, "nvmlInit_v2"); auto get = (int (*)(unsigned, void**))dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
        en = (int (*)(void*, unsigned long long*))dlsym(h, "nvmlDeviceGetTotalEnergyConsumption");
        ck = (int (*)(void*, int, unsigned*))dlsym(h, "nvmlDeviceGetClockInfo"); pw = (int (*)(void*, unsigned*))dlsym(h, "nvmlDeviceGetPowerUsage");
        ok = ini && get && en && ini() == 0 && get(0, &d) == 0; } };
int main(int argc, char** argv) {
    if (argc < 5) { printf("usage: ./probe M N K energy|ncu [arm,...] [seconds] [rounds]\n"); return 1; }
    const int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]); const std::string mode = argv[4];
    std::vector<std::string> arms; if (argc > 5) { std::stringstream ss(argv[5]); std::string t; while (std::getline(ss, t, ',')) arms.push_back(t); }
    const double block_s = argc > 6 ? atof(argv[6]) : 1.0;
    const int rounds = argc > 7 ? std::max(1, atoi(argv[7])) : 7;
    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
    // Three input sets, drawn by the judge's fill (seeds 11 + 2j for A and 12 + 2j for B); a call reads set `cur`. In ncu
    // mode, only the first set (two fill kernels before the arm's calls, which the launcher skips: ncu --launch-skip 2).
    half *A3[3] = {nullptr, nullptr, nullptr}, *B3[3] = {nullptr, nullptr, nullptr}, *C; CK(cudaMalloc(&C, nC * 2));
    for (int j = 0; j < (mode == "ncu" ? 1 : 3); ++j) {
        CK(cudaMalloc(&A3[j], nA * 2)); CK(cudaMalloc(&B3[j], nB * 2));
        fill<<<1024, 256>>>(A3[j], nA, 11 + 2 * j); fill<<<1024, 256>>>(B3[j], nB, 12 + 2 * j);
    }
    CK(cudaDeviceSynchronize());
    int cur = 0;
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cublasHandle_t hb; LT(cublasCreate(&hb)); LT(cublasSetStream(hb, s));
    cublasLtHandle_t lt; LT(cublasLtCreate(&lt)); void* ws; size_t wsz = 64u << 20; CK(cudaMalloc(&ws, wsz));
    cublasLtMatmulDesc_t op; LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasLtMatrixLayout_t la, lb, lc; LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, N, K, N)); LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, M, K));
    LT(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, N, M, N));
    const float one = 1.f, zero = 0.f;
    auto digits = [](const std::string& b, size_t from) {
        return b.size() > from && std::all_of(b.begin() + from, b.end(), [](char c) { return isdigit((unsigned char)c) != 0; });
    };
    auto call = [&](const std::string& b) {
        const half* A = A3[cur]; const half* B = B3[cur];
        if (b.size() >= 2 && b[0] == 'v' && digits(b, 1)) {
            if (atoi(b.c_str() + 1) >= g_nvar) { printf("arm %s: the candidate has %d variant(s)\n", b.c_str(), g_nvar); exit(3); }
            g_var = atoi(b.c_str() + 1); candidate_gemm(A, B, C, M, N, K, s); g_var = 0;
        } else if (b == "twin") { g_var = 0; candidate_gemm(A, B, C, M, N, K, s); g_var = 0; }   // v0's call, an arm of its own
        else if (b.rfind("x:", 0) == 0) {                           // x:<hex>: the algorithm of a REFERENCE line, byte for byte
            static std::map<std::string, cublasLtMatmulAlgo_t> memx;
            auto it = memx.find(b);
            if (it == memx.end()) {
                cublasLtMatmulAlgo_t al; const char* x = b.c_str() + 2;
                if (strlen(x) != 2 * sizeof al || !std::all_of(x, x + strlen(x), [](char c) { return isdigit((unsigned char)c) || (c >= 'a' && c <= 'f'); })) {
                    printf("x: %zu lower-case hexadecimal digits expected\n", 2 * sizeof al); exit(3);
                }
                auto v = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
                unsigned char* q = reinterpret_cast<unsigned char*>(&al);
                for (size_t i = 0; i < sizeof al; ++i) q[i] = (unsigned char)(v(x[2 * i]) * 16 + v(x[2 * i + 1]));
                cublasLtMatmulHeuristicResult_t chk;
                if (cublasLtMatmulAlgoCheck(lt, op, la, lb, lc, lc, &al, &chk) != CUBLAS_STATUS_SUCCESS || chk.workspaceSize > wsz) { printf("the judge's configuration refused\n"); exit(3); }
                it = memx.emplace(b, al).first;
            }
            LT(cublasLtMatmul(lt, op, &one, B, la, A, lb, &zero, C, lc, C, lc, &it->second, ws, wsz, s));
        }
        else { printf("unknown arm %s\n", b.c_str()); exit(3); }
    };
    if (mode == "ncu") { for (auto& b : arms) { call(b); call(b); } CK(cudaStreamSynchronize(s)); printf("ncu: done\n"); return 0; }
    Nv nv; nv.init(); if (!nv.ok) { printf("NVML energy unavailable\n"); return 2; }
    printf("shape %d x %d x %d\n", M, N, K);
    // The correctness of each GEMM arm against cublasGemmEx with an fp32 output (the judge's reference), on the GPU (the
    // judge's gap): max |C - ref| / max |ref|, INFINITY if an element is not finite (C is set to NaN before each check).
    float* R32; CK(cudaMalloc(&R32, nC * 4)); unsigned* dres; CK(cudaMalloc(&dres, 12));
    int ref_set = -1;   // the input set whose reference is in R32
    auto error_of = [&](int j) {
        if (ref_set != j) {
            LT(cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B3[j], CUDA_R_16F, N, A3[j], CUDA_R_16F, K, &zero, R32, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            CK(cudaStreamSynchronize(s)); ref_set = j;
        }
        CK(cudaMemsetAsync(dres, 0, 12, s)); gap<<<1024, 256, 0, s>>>(C, R32, nC, dres);
        unsigned h[3]; CK(cudaMemcpyAsync(h, dres, 12, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s));
        if (h[2]) return (double)INFINITY;
        float e, r; memcpy(&e, &h[0], 4); memcpy(&r, &h[1], 4); return (double)e / r;
    };
    std::vector<double> errs(arms.size(), -1), errs2(arms.size(), -1);
    std::vector<cudaGraphExec_t> gx(arms.size()); std::vector<int> G(arms.size());
    for (size_t i = 0; i < arms.size(); ++i) {
        double e = 0;
        for (int r = 0; r < 2; ++r) {                                         // 2 calls: a state left by the first would show in the second
            CK(cudaMemsetAsync(C, 0xff, nC * 2, s));
            call(arms[i]);
            if (cudaStreamSynchronize(s) != cudaSuccess || cudaGetLastError() != cudaSuccess) { printf("arm %s: CUDA error\n", arms[i].c_str()); return 3; }
            e = std::max(e, error_of(0));
        }
        errs[i] = e;
        call(arms[i]); CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
        auto t0 = std::chrono::steady_clock::now(); for (int r = 0; r < 10; ++r) call(arms[i]); CK(cudaStreamSynchronize(s));
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 10;
        G[i] = std::max(3, std::min(400, (int)(0.02 / dt)));
        // A graph of G calls on the three input sets in turn, the PDL edges between two calls NEUTRALIZED as in the judge
        // (judge/judge.cu, Bench::graph): they become ordinary dependencies.
        cudaGraph_t g; std::set<cudaGraphNode_t> boundaries;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int r = 0; r < G[i]; ++r) {
            cur = r % 3; call(arms[i]);
            if (r + 1 < G[i]) {
                cudaStreamCaptureStatus st; const cudaGraphNode_t* deps = nullptr; size_t nd = 0;
                if (cudaStreamGetCaptureInfo(s, &st, nullptr, nullptr, &deps, nullptr, &nd) == cudaSuccess)
                    for (size_t q = 0; q < nd; ++q) boundaries.insert(deps[q]);
            }
        }
        cur = 0;
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
    std::vector<double> acc_sm(arms.size()), acc_pw(arms.size()); std::vector<long> acc_n(arms.size());
    std::thread th([&] { while (!done) { const int b = active; if (b >= 0) { unsigned c = 0, p = 0; nv.ck(nv.d, 1, &c); nv.pw(nv.d, &p);
        std::lock_guard<std::mutex> l(mx); if (active == b) { acc_sm[b] += c; acc_pw[b] += p / 1000.0; acc_n[b]++; } }
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); } });
    std::vector<std::vector<double>> energy(arms.size()), t_us(arms.size());
    std::vector<size_t> order(arms.size()); for (size_t q = 0; q < order.size(); ++q) order[q] = q;
    unsigned seed = 12345u;                                                    // the arms' order: a linear congruential generator
    for (int round_i = 0; round_i < rounds; ++round_i) {
        for (size_t q = order.size(); q > 1; --q) { seed = seed * 1103515245u + 12345u; std::swap(order[q - 1], order[(seed >> 8) % q]); }
        for (size_t ii = 0; ii < arms.size(); ++ii) {
            const size_t i = order[ii];
            auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < block_s / 3) { CK(cudaGraphLaunch(gx[i], s)); CK(cudaStreamSynchronize(s)); }
            active = (int)i;
            unsigned long long e0 = 0, e1 = 0; nv.en(nv.d, &e0); long ng = 0; t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < block_s) { for (int q = 0; q < 4; ++q) CK(cudaGraphLaunch(gx[i], s)); ng += 4; CK(cudaStreamSynchronize(s)); }
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); nv.en(nv.d, &e1);
            active = -1;
            energy[i].push_back((double)(e1 - e0) / (ng * G[i])); t_us[i].push_back(dt / (ng * G[i]) * 1e6);
        }
    }
    done = true; th.join();
    // The correctness AFTER the timing, on the timed graph's output (its last call reads input set (G - 1) % 3), 3 times.
    for (size_t i = 0; i < arms.size(); ++i) {
        double e = 0;
        for (int r = 0; r < 3; ++r) {
            CK(cudaMemsetAsync(C, 0xff, nC * 2, s)); CK(cudaGraphLaunch(gx[i], s)); CK(cudaStreamSynchronize(s));
            e = std::max(e, error_of((G[i] - 1) % 3));
        }
        errs2[i] = e;
    }
    for (size_t i = 0; i < arms.size(); ++i) {
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        const double n_ = acc_n[i] ? acc_n[i] : 1, t = med(t_us[i]);
        std::string tt; for (double x : t_us[i]) { char z[24]; snprintf(z, sizeof z, " %.2f", x); tt += z; }
        printf("ARM %-7s : %8.2f us | %.3f mJ per call | SM %4.0f MHz | %5.1f W | G %d | rounds%s us | ",
               arms[i].rfind("x:", 0) == 0 ? "ref0" : arms[i].c_str(), t, med(energy[i]), acc_sm[i] / n_, acc_pw[i] / n_, G[i], tt.c_str());
        printf("err %.2e / graph %.2e%s\n", errs[i], errs2[i], (errs[i] <= 2e-3 && errs2[i] <= 2e-3) ? "" : "  WRONG");
    }
    return 0;
}
