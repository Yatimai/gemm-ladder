// probe_t4.cu (T4): the ENERGY per GEMM (NVML, total energy counter) and the time, arm by arm, in one process, in sustained
// alternating blocks; plus a short mode for ncu. The method of the judge and of probe_b200.cu: the inputs drawn by the
// judge's fill (uniform in [-1, 1)), three input sets alternated in the CUDA graphs (about 20 ms each), the order of the arms
// drawn at each round (the same generator and seed as probe_b200.cu), the correctness of each GEMM arm checked against
// cublasGemmEx with an fp32 output before the timing (2 calls) and after it (the timed graph, 3 times), at the judge's
// tolerance (2e-3: WRONG past it), and an unknown arm refused. There is no programmatic dependent launch (PDL) on sm_75:
// nothing to neutralize in the graphs. Each block is preceded by a warm-up of a third of its length.
// Arms: 'lt' = cuBLASLt's heuristic's first configuration (h0), or 'lt<i>' = the heuristic's result i; 'def' =
// cublasGemmEx's default; 'cand' = candidate_gemm (linked at build time, as in the judge); 'v<i>' = the candidate's variant
// i; 'cpos' = the candidate on positive data; 'rd', 'rs<D>', 'rk<G>', 'w..' = pure reads of B (no correctness to check).
// Usage: ./probe M N K mode arm1,arm2,... [seconds] [rounds]; mode = energy (blocks of `seconds`, 3 rounds by default) | ncu
// (2 calls per arm).
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <dlfcn.h>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s);
// Arm v<i>: the candidate's variant i (an interleaved A/B in one process); the candidate defines g_var and g_nvar (the
// number of its variants) when it has variants; v<i> past g_nvar is refused.
__attribute__((weak)) int g_var = 0;
__attribute__((weak)) int g_nvar = 1;
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
// Arm 'cpos': "positive" data, fp16 in [0.125, 1), a fixed sign.
__global__ void fill_positive(half* p, size_t n, unsigned g) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned x = ((unsigned)i * 2654435761u) ^ (g * 0x9e3779b9u); x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        const unsigned short h = (unsigned short)((x & 0x3bffu) | 0x3000u); p[i] = *reinterpret_cast<const half*>(&h);
    }
}
struct Nv { void* d = nullptr; int (*en)(void*, unsigned long long*) = nullptr; int (*ck)(void*, int, unsigned*) = nullptr; int (*pw)(void*, unsigned*) = nullptr; bool ok = false;
    void init() { void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW); if (!h) return;
        auto ini = (int (*)())dlsym(h, "nvmlInit_v2"); auto get = (int (*)(unsigned, void**))dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
        en = (int (*)(void*, unsigned long long*))dlsym(h, "nvmlDeviceGetTotalEnergyConsumption");
        ck = (int (*)(void*, int, unsigned*))dlsym(h, "nvmlDeviceGetClockInfo"); pw = (int (*)(void*, unsigned*))dlsym(h, "nvmlDeviceGetPowerUsage");
        ok = ini && get && en && ini() == 0 && get(0, &d) == 0; } };
// Arm 'rd': a PURE READ of B (a bandwidth bound): each thread reads uint4 in a persistent grid, 4 in flight per turn.
__global__ void probe_read(const uint4* __restrict__ p, size_t n, unsigned* out) {
    unsigned x = 0; const size_t T = (size_t)gridDim.x * blockDim.x;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    for (; i + 3 * T < n; i += 4 * T) {
        uint4 a = __ldcs(p + i), b = __ldcs(p + i + T), c = __ldcs(p + i + 2 * T), d = __ldcs(p + i + 3 * T);
        x ^= a.x ^ b.y ^ c.z ^ d.w;
    }
    for (; i < n; i += T) x ^= __ldcs(p + i).x;
    if (x == 0x12345678u) out[0] = x;
}
// Arm rs<D>: a PURE READ of B BY BANDS, the access pattern of the M = 16 kernel without a split of K: N / 128 blocks of 256
// threads, a band of 128 columns (256 B per row) over all of K; per step of 32 rows, 2 LDG.128 cg per thread; D steps in
// flight per thread.
__device__ __forceinline__ uint4 ldg_cg(const uint8_t* p) {
    uint4 v; asm volatile("ld.global.cg.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
template <int D>
__global__ void __launch_bounds__(256) probe_read_band(const half* __restrict__ B, int N, int K, unsigned* out) {
    const int t = threadIdx.x, n0 = blockIdx.x * 128;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(B + (size_t)(t >> 4) * N + n0 + (t & 15) * 8);
    const size_t stride = (size_t)32 * N * 2, r16 = (size_t)16 * N * 2;
    const int KT = K / 32;
    uint4 b0[D], b1[D];
    unsigned x = 0;
    #pragma unroll
    for (int d = 0; d < D; ++d) { b0[d] = ldg_cg(p + d * stride); b1[d] = ldg_cg(p + d * stride + r16); }
    for (int k0 = 0; k0 < KT; k0 += D) {
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const int k = k0 + d;
            if (k < KT) {
                x ^= b0[d].x ^ b0[d].w ^ b1[d].y ^ b1[d].z;
                if (k + D < KT) { b0[d] = ldg_cg(p + (size_t)(k + D) * stride); b1[d] = ldg_cg(p + (size_t)(k + D) * stride + r16); }
            }
        }
    }
    if (x == 0x12345678u) out[0] = x;
}
// Arm rk<G>: a read by bands, SLICED: 32 blocks (a band of 128 columns), the 256 threads in G groups, group g reading the
// rows [g K / G, (g + 1) K / G) of the band; 8 loads of 16 B in flight per thread (an unrolled ring).
template <int G>
__global__ void __launch_bounds__(256) probe_read_band_k(const half* __restrict__ B, int N, int K, unsigned* out) {
    constexpr int TG = 256 / G, RPI = TG / 16, D = 8;             // threads per group, rows per instruction of the group
    const int t = threadIdx.x, g = t / TG, u = t % TG, n0 = blockIdx.x * 128;
    const int KG = K / G, NI = KG / RPI;                          // instructions per thread
    const uint8_t* p = reinterpret_cast<const uint8_t*>(B + ((size_t)g * KG + u / 16) * N + n0 + (u % 16) * 8);
    const size_t stride = (size_t)RPI * N * 2;
    uint4 b[D];
    unsigned x = 0;
    #pragma unroll
    for (int d = 0; d < D; ++d) b[d] = ldg_cg(p + d * stride);
    for (int i0 = 0; i0 < NI; i0 += D) {
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const int i = i0 + d;
            if (i < NI) { x ^= b[d].x ^ b[d].w; if (i + D < NI) b[d] = ldg_cg(p + (size_t)(i + D) * stride); }
        }
    }
    if (x == 0x12345678u) out[0] = x;
}
// Arm w<b><S>: a read by bands of BN = 128 b columns, S slices of K per band (a grid of (N / BN) x S, each block reading its
// slice of K of its band; a row of BN columns = BN / 8 threads); 8 loads of 16 B in flight per thread.
template <int BN, int S>
__global__ void __launch_bounds__(256) probe_read_bands(const half* __restrict__ B, int N, int K, unsigned* out) {
    constexpr int TPR = BN / 8, RPI = 256 / TPR, D = 8;           // threads per row, rows per instruction of the block
    const int t = threadIdx.x, band = blockIdx.x / S, tr = blockIdx.x % S, n0 = band * BN;
    const int KS = K / S, NI = KS / RPI;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(B + ((size_t)tr * KS + t / TPR) * N + n0 + (t % TPR) * 8);
    const size_t stride = (size_t)RPI * N * 2;
    uint4 b[D];
    unsigned x = 0;
    #pragma unroll
    for (int d = 0; d < D; ++d) b[d] = ldg_cg(p + d * stride);
    for (int i0 = 0; i0 < NI; i0 += D) {
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            const int i = i0 + d;
            if (i < NI) { x ^= b[d].x ^ b[d].w; if (i + D < NI) b[d] = ldg_cg(p + (size_t)(i + D) * stride); }
        }
    }
    if (x == 0x12345678u) out[0] = x;
}
int main(int argc, char** argv) {
    if (argc < 6) { printf("usage: ./probe M N K energy|ncu arm,... [seconds] [rounds]\n"); return 1; }
    const int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]); const std::string mode = argv[4];
    std::vector<std::string> arms; { std::stringstream ss(argv[5]); std::string t; while (std::getline(ss, t, ',')) arms.push_back(t); }
    const double block_s = argc > 6 ? atof(argv[6]) : 1.0;
    const int rounds = argc > 7 ? atoi(argv[7]) : 3;
    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
    // Three input sets, drawn by the judge's fill (seeds as in probe_b200.cu); a call reads set `cur`.
    half *A3[3], *B3[3], *C; CK(cudaMalloc(&C, nC * 2));
    for (int j = 0; j < 3; ++j) {
        CK(cudaMalloc(&A3[j], nA * 2)); CK(cudaMalloc(&B3[j], nB * 2));
        fill<<<1024, 256>>>(A3[j], nA, 11 + 2 * j); fill<<<1024, 256>>>(B3[j], nB, 12 + 2 * j);
    }
    CK(cudaDeviceSynchronize());
    half *AP[3] = {nullptr, nullptr, nullptr}, *BP[3] = {nullptr, nullptr, nullptr};   // the 'cpos' sets, drawn at its first call
    int cur = 0;
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cublasHandle_t hb; LT(cublasCreate(&hb)); LT(cublasSetStream(hb, s));
    cublasLtHandle_t lt; LT(cublasLtCreate(&lt)); void* ws; size_t wsz = 64u << 20; CK(cudaMalloc(&ws, wsz));
    cublasLtMatmulDesc_t op; LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasLtMatrixLayout_t la, lb, lc; LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, N, K, N)); LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, M, K));
    LT(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, N, M, N));
    cublasLtMatmulPreference_t pref; LT(cublasLtMatmulPreferenceCreate(&pref));
    LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
    std::vector<cublasLtMatmulHeuristicResult_t> res(16); int got = 0;
    LT(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 16, res.data(), &got));
    const float one = 1.f, zero = 0.f;
    auto digits = [](const std::string& b, size_t from) {
        return b.size() > from && std::all_of(b.begin() + from, b.end(), [](char c) { return isdigit((unsigned char)c) != 0; });
    };
    auto is_read = [&](const std::string& b) {
        return b == "rd" || b == "rs2" || b == "rs4" || b == "rs8" || b == "rk2" || b == "rk4" || b == "rk8" || (b.size() == 3 && b[0] == 'w');
    };
    auto call = [&](const std::string& b) {
        const half* A = A3[cur]; const half* B = B3[cur];
        if (b == "cand") candidate_gemm(A, B, C, M, N, K, s);
        else if (b == "cpos") {
            if (!AP[0]) {
                for (int j = 0; j < 3; ++j) {
                    CK(cudaMalloc(&AP[j], nA * 2)); CK(cudaMalloc(&BP[j], nB * 2));
                    fill_positive<<<1024, 256>>>(AP[j], nA, 7 + 2 * j); fill_positive<<<1024, 256>>>(BP[j], nB, 8 + 2 * j);
                }
                CK(cudaDeviceSynchronize());
            }
            candidate_gemm(AP[cur], BP[cur], C, M, N, K, s);
        }
        else if (b.size() == 2 && b[0] == 'v' && digits(b, 1)) {
            if (b[1] - '0' >= g_nvar) { printf("arm %s: the candidate has %d variant(s)\n", b.c_str(), g_nvar); exit(3); }
            g_var = b[1] - '0'; candidate_gemm(A, B, C, M, N, K, s); g_var = 0;
        }
        else if (b == "rs2") probe_read_band<2><<<N / 128, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "rs4") probe_read_band<4><<<N / 128, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "rs8") probe_read_band<8><<<N / 128, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "rk2") probe_read_band_k<2><<<N / 128, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "rk4") probe_read_band_k<4><<<N / 128, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "rk8") probe_read_band_k<8><<<N / 128, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "w11") probe_read_bands<128, 1><<<N / 128 * 1, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "w21") probe_read_bands<256, 1><<<N / 256 * 1, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "w22") probe_read_bands<256, 2><<<N / 256 * 2, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "w25") probe_read_bands<256, 5><<<N / 256 * 5, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "w42") probe_read_bands<512, 2><<<N / 512 * 2, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "w44") probe_read_bands<512, 4><<<N / 512 * 4, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "w23") probe_read_bands<256, 3><<<N / 256 * 3, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "w24") probe_read_bands<256, 4><<<N / 256 * 4, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "w26") probe_read_bands<256, 6><<<N / 256 * 6, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "w27") probe_read_bands<256, 7><<<N / 256 * 7, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "w28") probe_read_bands<256, 8><<<N / 256 * 8, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "wa5") probe_read_bands<512, 10><<<N / 512 * 10, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "w4a") probe_read_bands<512, 8><<<N / 512 * 8, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "wa0") probe_read_bands<512, 5><<<N / 512 * 5, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "wb5") probe_read_bands<1024, 20><<<N / 1024 * 20, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "wba") probe_read_bands<1024, 10><<<N / 1024 * 10, 256, 0, s>>>(B, N, K, reinterpret_cast<unsigned*>(C));
        else if (b == "rd") { static int P = 0; if (!P) { int d = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&P, cudaDevAttrMultiProcessorCount, d); }
            probe_read<<<P * 4, 256, 0, s>>>(reinterpret_cast<const uint4*>(B), nB / 8, reinterpret_cast<unsigned*>(C)); }
        else if (b == "def") cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &zero, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        else if (b == "lt" || (b.compare(0, 2, "lt") == 0 && digits(b, 2))) {
            const int i = b.size() > 2 ? atoi(b.c_str() + 2) : 0;
            if (i >= got) { printf("arm %s: cuBLASLt's heuristic gives %d result(s)\n", b.c_str(), got); exit(3); }
            cublasLtMatmul(lt, op, &one, B, la, A, lb, &zero, C, lc, C, lc, &res[i].algo, ws, wsz, s);
        }
        else { printf("unknown arm %s\n", b.c_str()); exit(3); }
    };
    if (mode == "ncu") { for (auto& b : arms) { call(b); call(b); } CK(cudaStreamSynchronize(s)); printf("ncu: done\n"); return 0; }
    Nv nv; nv.init(); if (!nv.ok) { printf("NVML energy unavailable\n"); return 2; }
    printf("shape %d x %d x %d; heuristic: %d results\n", M, N, K, got);
    // Timed by CUDA graphs (as in the judge), NVML sampled in a thread apart: direct calls, with NVML read on the host
    // thread, starve the GPU at the small shapes.
    // The correctness of each GEMM arm against cublasGemmEx with an fp32 output (the judge's reference), on the GPU (the
    // judge's gap): max |C - ref| / max |ref|, INFINITY if an element is not finite (C is set to NaN before each check).
    float* R32; CK(cudaMalloc(&R32, nC * 4)); unsigned* dres; CK(cudaMalloc(&dres, 12));
    int ref_pos = -1, ref_set = -1;   // the data (1: the 'cpos' sets) and the set whose reference is in R32
    auto error_of = [&](bool pos, int j) {
        if (ref_pos != (int)pos || ref_set != j) {
            const half* A = pos ? AP[j] : A3[j]; const half* B = pos ? BP[j] : B3[j];
            LT(cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &zero, R32, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            CK(cudaStreamSynchronize(s)); ref_pos = pos; ref_set = j;
        }
        CK(cudaMemsetAsync(dres, 0, 12, s)); gap<<<1024, 256, 0, s>>>(C, R32, nC, dres);
        unsigned h[3]; CK(cudaMemcpyAsync(h, dres, 12, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s));
        if (h[2]) return (double)INFINITY;
        float e, r; memcpy(&e, &h[0], 4); memcpy(&r, &h[1], 4); return (double)e / r;
    };
    std::vector<double> errs(arms.size(), -1), errs2(arms.size(), -1);
    std::vector<cudaGraphExec_t> gx(arms.size()); std::vector<int> G(arms.size());
    for (size_t i = 0; i < arms.size(); ++i) {
        if (!is_read(arms[i])) {
            double e = 0;
            for (int r = 0; r < 2; ++r) {                                     // 2 calls: a state left by the first would show in the second
                CK(cudaMemsetAsync(C, 0xff, nC * 2, s));
                call(arms[i]);
                if (cudaStreamSynchronize(s) != cudaSuccess || cudaGetLastError() != cudaSuccess) { printf("arm %s: CUDA error\n", arms[i].c_str()); return 3; }
                e = std::max(e, error_of(arms[i] == "cpos", 0));
            }
            errs[i] = e;
        } else {
            call(arms[i]); CK(cudaStreamSynchronize(s));
        }
        auto t0 = std::chrono::steady_clock::now(); for (int r = 0; r < 10; ++r) call(arms[i]); CK(cudaStreamSynchronize(s));
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 10;
        G[i] = std::max(3, std::min(400, (int)(0.02 / dt)));                 // ~20 ms per graph
        // A graph of G calls on the three input sets in turn, as in the judge (no PDL on sm_75: nothing to neutralize).
        cudaGraph_t g; CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int r = 0; r < G[i]; ++r) { cur = r % 3; call(arms[i]); }
        cur = 0;
        CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&gx[i], g, 0)); cudaGraphDestroy(g);
    }
    std::atomic<int> active{-1}; std::atomic<bool> done{false}; std::mutex mx;
    std::vector<double> acc_sm(arms.size()), acc_mem(arms.size()), acc_pw(arms.size()); std::vector<long> acc_n(arms.size());
    std::thread th([&] { while (!done) { const int b = active; if (b >= 0) { unsigned c = 0, cm = 0, p = 0; nv.ck(nv.d, 1, &c); nv.ck(nv.d, 2, &cm); nv.pw(nv.d, &p);
        std::lock_guard<std::mutex> l(mx); if (active == b) { acc_sm[b] += c; acc_mem[b] += cm; acc_pw[b] += p / 1000.0; acc_n[b]++; } }
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); } });
    std::vector<std::vector<double>> energy(arms.size()), t_us(arms.size());
    std::vector<size_t> order(arms.size()); for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    unsigned seed = 12345u;                                                    // probe_b200.cu's generator and seed
    for (int round_i = 0; round_i < rounds; ++round_i) {
        for (size_t i = order.size(); i > 1; --i) { seed = seed * 1103515245u + 12345u; std::swap(order[i - 1], order[(seed >> 8) % i]); }
        for (size_t i : order) {
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
        if (is_read(arms[i])) continue;
        double e = 0;
        for (int r = 0; r < 3; ++r) {
            CK(cudaMemsetAsync(C, 0xff, nC * 2, s)); CK(cudaGraphLaunch(gx[i], s)); CK(cudaStreamSynchronize(s));
            e = std::max(e, error_of(arms[i] == "cpos", (G[i] - 1) % 3));
        }
        errs2[i] = e;
    }
    for (size_t i = 0; i < arms.size(); ++i) {
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        const double n_ = acc_n[i] ? acc_n[i] : 1;
        printf("ARM %-6s : %.3f mJ per GEMM | %.1f us | SM %.0f MHz | MEM %.0f MHz | %.1f W | G %d | rounds", arms[i].c_str(), med(energy[i]), med(t_us[i]),
               acc_sm[i] / n_, acc_mem[i] / n_, acc_pw[i] / n_, G[i]);
        for (double x : t_us[i]) printf(" %.1f", x);
        if (is_read(arms[i])) printf(" us | err n/a\n");
        else printf(" us | err %.2e / graph %.2e%s\n", errs[i], errs2[i], (errs[i] <= 2e-3 && errs2[i] <= 2e-3) ? "" : "  WRONG");
    }
    return 0;
}
