// The judge of gemm-ladder, version 4. It times a candidate against the judge's reference, shape by shape, in a
// sustained regime: each arm runs alone for a block, as in service, so at its own clock (the T4, A100, H100 and B200 are
// held by their power cap during a GEMM).
//
// Convention: C = A B; A (M x K) and B (K x N) fp16 row-major; fp32 accumulation; C (M x N) fp16 row-major. The
// candidate provides, in a .cu compiled with this file:
//   void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s);
// It does not synchronize, does not allocate during a call (an allocation at the first call of a shape is allowed:
// that first call is never timed), can be captured in a CUDA graph, and works on the stream s (its own: the reference
// has another).
//
// Correctness (tolerance: max |C - ref| / max |ref| <= 2e-3, no element that is not finite, the whole output
// written; the gap is computed on the GPU against an fp32 reference from cublasGemmEx):
//   - before the timing, direct calls on 3 sets of inputs;
//   - during the timing, the output of the timed graph is checked after EVERY block of the candidate;
//   - after the timing: new contents at the same addresses, then the timed graph run again and checked; then new
//     contents again, then direct calls and a check (a kept result, a copy, or work skipped under capture fail);
//   - A and B must not be modified by the candidate (checksums);
//   - the GPU's state must not be modified by the candidate (limits, cache configuration, the L2 window of the
//     reference's stream), and the persisting L2 is reset before each block;
//   - the programmatic dependencies (PDL) between two consecutive calls are NEUTRALIZED in both timed graphs, the
//     reference's and the candidate's: they become ordinary dependencies, and both arms are timed as a chain of
//     dependent layers. PDL inside a call stays allowed. When the reference chains its calls by PDL (cuBLAS does on
//     the B200), the judge reports, for the record, what that PDL would have brought it.
//
// Timing: a CUDA graph of G >= 3 calls (alternating input sets) per arm; a block = an untimed warm-up, then the
// timing; P pairs of blocks (reference, candidate) in random order; ratio per pair = the reference's time over the
// candidate's; per shape: the median of the pairs; score: the geometric mean over the shapes. The clock and power of
// each arm are sampled through NVML.
//
// Reference: a short sort (blocks of 0.35 s) of cuBLASLt's configurations (the heuristic: its first 48 with 64 MB of
// workspace, then its first 8 not yet present with 32, 4, 1 and 0 MB; then, around the first of each family, each
// setting changed alone, split-K with each reduction scheme, clusters 2 to 51), then a FINAL of the first 5 in the
// regime of the pairs (2 interleaved rounds), which cuBLAS's default call (cublasGemmEx) also enters, twice: with the
// handle a user gets in a CUDA graph (cublasSetStream only: cuBLAS's pool workspace) and with the judge's 64 MB
// handle (cuBLAS picks another kernel depending on the workspace). The fastest is the reference. Labels: hN = result N
// of the heuristic at 64 MB, wMhN = result N at M MB, aID:settingV = a variation around family ID, cacheN = a cached
// configuration. The REFERENCE lines (cuBLASLt's finalists, with the card's identity and the whole algorithm in
// hexadecimal) feed a cache; with --reference FILE, the sort covers only the configurations kept for THIS card, the
// heuristic's first 3 at 64 MB and its first 8 not yet present at each other workspace cap, then the same final. The
// "reference time" of the SCORE line is the reference's: cuBLASLt's or cuBLAS's default call's. The final writes, per
// shape, the ratio of each default call's time to the best cuBLASLt configuration's; --default times the pool default
// (the ratio_default column then changes meaning). The NVML throttle reasons of each arm are written, and a thermal
// or hardware throttle raises a warning (such a pass does not count as an official pair). A CYCLES line follows the
// SCORE line (ratio in cycles, and clocks, at the shapes with M >= 512: under a power cap, a clock gain is an energy
// gain); the line of a refused timing is marked "# timing refused"; the REFERENCE lines carry the pass's token, which
// the launcher checks.
//
// Usage: ./judge shapes.txt [--pairs P] [--duration S] [--warmup S] [--quick] [--only a,b] [--default]
//        [--seed G] [--reference FILE]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef JUDGE_MD5
#define JUDGE_MD5 "unknown"
#endif

void candidate_gemm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t s);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); fflush(stdout); exit(3); } } while (0)
#define LT(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS %s:%d status %d\n", __FILE__, __LINE__, (int)s_); fflush(stdout); exit(3); } } while (0)

static const double kTol = 2e-3;
static const int kFinalists = 5;

// ---------------------------------------------------------------- NVML, loaded at run time
struct Nvml {
    void* dev = nullptr;
    int (*clock)(void*, int, unsigned*) = nullptr;
    int (*power)(void*, unsigned*) = nullptr;
    int (*reasons)(void*, unsigned long long*) = nullptr;
    int (*cap)(void*, unsigned*) = nullptr;
    bool ok = false;
    void init()
    {
        void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!h) return;
        auto ini = (int (*)())dlsym(h, "nvmlInit_v2");
        auto get = (int (*)(unsigned, void**))dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
        clock = (int (*)(void*, int, unsigned*))dlsym(h, "nvmlDeviceGetClockInfo");
        power = (int (*)(void*, unsigned*))dlsym(h, "nvmlDeviceGetPowerUsage");
        cap = (int (*)(void*, unsigned*))dlsym(h, "nvmlDeviceGetEnforcedPowerLimit");
        reasons = (int (*)(void*, unsigned long long*))dlsym(h, "nvmlDeviceGetCurrentClocksEventReasons");
        if (!reasons) reasons = (int (*)(void*, unsigned long long*))dlsym(h, "nvmlDeviceGetCurrentClocksThrottleReasons");
        ok = ini && get && clock && power && ini() == 0 && get(0, &dev) == 0;
    }
};

// Clock and power samples of the arm being timed (0 reference, 1 candidate, 2 default call).
struct Sample { double mhz = 0, w = 0; long n = 0; unsigned long long reasons = 0; };
static std::atomic<int> g_arm{-1};
static std::atomic<bool> g_done{false};
static Sample g_samples[3];
static std::mutex g_mx;

static void sampler(Nvml* nv)
{
    while (!g_done) {
        const int b = g_arm.load();
        if (b >= 0 && nv->ok) {
            unsigned c = 0, mw = 0; unsigned long long r = 0;
            nv->clock(nv->dev, 1 /* NVML_CLOCK_SM */, &c); nv->power(nv->dev, &mw);
            if (nv->reasons) nv->reasons(nv->dev, &r);
            std::lock_guard<std::mutex> l(g_mx);
            g_samples[b].mhz += c; g_samples[b].w += mw / 1000.0; g_samples[b].n++; g_samples[b].reasons |= r;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// ---------------------------------------------------------------- the judge's kernels
__global__ void fill(half* p, size_t n, unsigned seed)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned x = ((unsigned)i * 2654435761u) ^ (seed * 0x9e3779b9u) ^ (unsigned)(i >> 32);
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        p[i] = __float2half((x >> 8) * (1.0f / 8388608.0f) - 1.0f);   // [-1, 1)
    }
}

// The gap of an fp16 output to the fp32 reference, on the GPU: max |output - ref|, max |ref| (positive floats:
// their order is their bits') and a flag if an element is not finite (not written, NaN, infinity).
__global__ void gap(const half* __restrict__ out, const float* __restrict__ ref, size_t n, unsigned* res)
{
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
    if ((threadIdx.x & 31) == 0) {
        atomicMax(&res[0], __float_as_uint(e)); atomicMax(&res[1], __float_as_uint(r));
        if (bad) atomicOr(&res[2], 1u);
    }
}

// Checksum of an array of half words (its 16-bit words weighted by their position, summed modulo 2^64).
__global__ void checksum(const unsigned short* __restrict__ p, size_t n, unsigned long long* res)
{
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        s += (unsigned long long)p[i] * (2 * i + 1);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(res, s);
}

// ---------------------------------------------------------------- cuBLASLt's configurations
struct Cfg {
    std::string label;
    cublasLtMatmulAlgo_t algo;
    int id = -1, tile = -1, stages = -1, splitk = -1, red = -1, swz = -1, custom = -1, inner = -1, cluster = -1;
    double t = 1e30;
};

static int attr32(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoConfigAttributes_t at)
{
    int v = -1; size_t w = 0;
    return cublasLtMatmulAlgoConfigGetAttribute(&a, at, &v, sizeof(v), &w) == CUBLAS_STATUS_SUCCESS ? v : -1;
}
static int attr16(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoConfigAttributes_t at)
{
    uint16_t v = 0; size_t w = 0;   // INNER_SHAPE_ID and CLUSTER_SHAPE_ID are uint16_t
    return cublasLtMatmulAlgoConfigGetAttribute(&a, at, &v, sizeof(v), &w) == CUBLAS_STATUS_SUCCESS ? v : -1;
}
static void read_cfg(Cfg& c)
{
    c.id = attr32(c.algo, CUBLASLT_ALGO_CONFIG_ID); c.tile = attr32(c.algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
    c.stages = attr32(c.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID); c.splitk = attr32(c.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
    c.swz = attr32(c.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING); c.custom = attr32(c.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
    c.red = attr32(c.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME); c.inner = attr16(c.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID);
    c.cluster = attr16(c.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID);
}
static std::string describe(const Cfg& c)
{
    char b[200];
    snprintf(b, sizeof b, "%s algo %d tile %d stages %d splitk %d reduction %d swz %d option %d cluster %d", c.label.c_str(),
             c.id, c.tile, c.stages, c.splitk, c.red, c.swz, c.custom, c.cluster);
    return b;
}
// The whole algorithm (64 bytes, serializable per cublasLt.h for a given version of cuBLAS) in hexadecimal.
static std::string hex(const cublasLtMatmulAlgo_t& a)
{
    static const char* d = "0123456789abcdef";
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&a);
    std::string s;
    for (size_t i = 0; i < sizeof a; ++i) { s += d[p[i] >> 4]; s += d[p[i] & 15]; }
    return s;
}
static bool dehex(const std::string& s, cublasLtMatmulAlgo_t& a)
{
    if (s.size() != 2 * sizeof a) return false;
    unsigned char* p = reinterpret_cast<unsigned char*>(&a);
    for (size_t i = 0; i < sizeof a; ++i) {
        auto v = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; };
        const int h = v(s[2 * i]), l = v(s[2 * i + 1]);
        if (h < 0 || l < 0) return false;
        p[i] = (unsigned char)(h * 16 + l);
    }
    return true;
}

struct Lt {
    cublasLtHandle_t h;
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t la, lb, lc;
    void* ws; size_t wsMax = 64u << 20;
};

static bool check_algo(Lt& L, Cfg& c)
{
    cublasLtMatmulHeuristicResult_t chk;
    if (cublasLtMatmulAlgoCheck(L.h, L.op, L.la, L.lb, L.lc, L.lc, &c.algo, &chk) != CUBLAS_STATUS_SUCCESS) return false;
    return chk.workspaceSize <= L.wsMax;
}

// The heuristic's first n; with full, also, around the first of each family, each setting changed alone
// (split-K with each of the reduction schemes allowed).
static std::vector<Cfg> configurations(Lt& L, bool full, int n = 48)
{
    // The heuristic at 64 MB (its first n), then at 32, 4, 1 and 0 MB (its first 8 not yet present, in the full sort
    // as in cache mode: a cache written by an earlier version lacks these configurations); its choices change with
    // the workspace cap.
    std::vector<Cfg> cfgs;
    for (size_t cap : {L.wsMax, (size_t)32 << 20, (size_t)4 << 20, (size_t)1 << 20, (size_t)0}) {
        cublasLtMatmulPreference_t pref; LT(cublasLtMatmulPreferenceCreate(&pref));
        LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cap, sizeof(cap)));
        std::vector<cublasLtMatmulHeuristicResult_t> res(48); int got = 0;
        if (cublasLtMatmulAlgoGetHeuristic(L.h, L.op, L.la, L.lb, L.lc, L.lc, pref, 48, res.data(), &got) != CUBLAS_STATUS_SUCCESS) got = 0;
        cublasLtMatmulPreferenceDestroy(pref);
        const int wanted = cap == L.wsMax ? n : 8;
        int taken = 0;
        for (int i = 0; i < got && taken < wanted; ++i) {
            if (res[i].state != CUBLAS_STATUS_SUCCESS || res[i].workspaceSize > cap) continue;
            bool present = false;
            for (const Cfg& d : cfgs) if (!memcmp(&d.algo, &res[i].algo, sizeof d.algo)) { present = true; break; }
            if (present) continue;
            Cfg c; c.label = (cap == L.wsMax ? std::string("h") : "w" + std::to_string(cap >> 20) + "h") + std::to_string(i);
            c.algo = res[i].algo; read_cfg(c); cfgs.push_back(c); ++taken;
        }
    }
    if (!full) return cfgs;
    std::vector<int> families; const size_t nh = cfgs.size();
    for (size_t i = 0; i < nh; ++i) {
        if (std::find(families.begin(), families.end(), cfgs[i].id) != families.end()) continue;
        families.push_back(cfgs[i].id);
        const Cfg base = cfgs[i];
        auto set_attr = [](Cfg& c, cublasLtMatmulAlgoConfigAttributes_t at, int v, bool u16) {
            const uint16_t v16 = (uint16_t)v;
            return cublasLtMatmulAlgoConfigSetAttribute(&c.algo, at, u16 ? (const void*)&v16 : (const void*)&v,
                                                        u16 ? sizeof(v16) : sizeof(v)) == CUBLAS_STATUS_SUCCESS;
        };
        auto variation = [&](const std::string& tag, cublasLtMatmulAlgoConfigAttributes_t at, int v, bool u16 = false) {
            Cfg c = base; c.label = "a" + std::to_string(base.id) + ":" + tag + std::to_string(v);
            if (!set_attr(c, at, v, u16) || !check_algo(L, c)) return;
            read_cfg(c); cfgs.push_back(c);
        };
        for (int s = 0; s <= 1; ++s) if (s != base.swz) variation("swz", CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, s);
        int mask = 0; size_t sz = 0;
        cublasLtMatmulAlgoCapGetAttribute(&base.algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &mask, sizeof(mask), &sz);
        for (int k : {1, 2, 3, 4, 8}) {
            if (k == 1) { if (base.splitk != 1) variation("splitk", CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1); continue; }
            for (int r : {1, 2, 4}) {   // INPLACE, COMPUTE_TYPE, OUTPUT_TYPE, if allowed
                if (!(mask & r) || (k == base.splitk && r == base.red)) continue;
                Cfg c = base; c.label = "a" + std::to_string(base.id) + ":splitk" + std::to_string(k) + "r" + std::to_string(r);
                if (!set_attr(c, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, k, false) || !set_attr(c, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, r, false)
                    || !check_algo(L, c)) continue;
                read_cfg(c); cfgs.push_back(c);
            }
        }
        sz = 0;
        cublasLtMatmulAlgoCapGetAttribute(&base.algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz);
        std::vector<int> tiles(sz / sizeof(int));
        if (!tiles.empty()) cublasLtMatmulAlgoCapGetAttribute(&base.algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles.data(), sz, &sz);
        for (int t : tiles) if (t != base.tile) variation("tile", CUBLASLT_ALGO_CONFIG_TILE_ID, t);
        sz = 0;
        cublasLtMatmulAlgoCapGetAttribute(&base.algo, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &sz);
        std::vector<int> stage_ids(sz / sizeof(int));
        if (!stage_ids.empty()) cublasLtMatmulAlgoCapGetAttribute(&base.algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stage_ids.data(), sz, &sz);
        for (int s : stage_ids) if (s != base.stages) variation("stages", CUBLASLT_ALGO_CONFIG_STAGES_ID, s);
        int cmax = 0; sz = 0;
        cublasLtMatmulAlgoCapGetAttribute(&base.algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &cmax, sizeof(cmax), &sz);
        for (int o = 0; o <= cmax; ++o) if (o != base.custom) variation("custom", CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, o);
        if (base.cluster >= 0)
            for (int g = 2; g <= 51; ++g) if (g != base.cluster) variation("cluster", CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, g, true);
    }
    return cfgs;
}

// ---------------------------------------------------------------- the GPU state the candidate must not touch
struct GpuState {
    size_t persist = 0, fetch = 0; cudaFuncCache cache = cudaFuncCachePreferNone;
    cudaStreamAttrValue window;
    bool operator==(const GpuState& o) const
    {
        return persist == o.persist && fetch == o.fetch && cache == o.cache &&
               window.accessPolicyWindow.base_ptr == o.window.accessPolicyWindow.base_ptr &&
               window.accessPolicyWindow.num_bytes == o.window.accessPolicyWindow.num_bytes;
    }
};
static GpuState read_state(cudaStream_t s_ref)
{
    GpuState e; memset(&e.window, 0, sizeof e.window);
    cudaDeviceGetLimit(&e.persist, cudaLimitPersistingL2CacheSize);
    cudaDeviceGetLimit(&e.fetch, cudaLimitMaxL2FetchGranularity);
    cudaDeviceGetCacheConfig(&e.cache);
    cudaStreamGetAttribute(s_ref, cudaStreamAttributeAccessPolicyWindow, &e.window);
    cudaGetLastError();
    return e;
}

// ---------------------------------------------------------------- timing
struct Options {
    int pairs = 4; double duration = 1.5, warmup = 0.5; bool quick = false, with_default = false;
    unsigned seed = 0; std::vector<std::string> only; std::string reference;
};

struct Bench {
    cudaStream_t s; cudaEvent_t e0, e1;
    // A graph of G calls on the stream s; call(j) launches the call on input set j % 3. The edges that are not
    // ordinary (programmatic dependencies) from the end of a call to the next call are counted in *pdl and, if
    // neutralize, replaced by ordinary edges.
    cudaGraphExec_t graph(cudaStream_t s, const std::function<void(int)>& call, int G, bool* ok, int* pdl = nullptr,
                           bool neutralize = true)
    {
        cudaGraph_t g; *ok = true;
        std::set<cudaGraphNode_t> boundaries;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int j = 0; j < G; ++j) {
            call(j % 3);
            if (j + 1 < G) {
                cudaStreamCaptureStatus st; const cudaGraphNode_t* deps = nullptr; size_t nd = 0;
                if (cudaStreamGetCaptureInfo(s, &st, nullptr, nullptr, &deps, nullptr, &nd) == cudaSuccess)
                    for (size_t i = 0; i < nd; ++i) boundaries.insert(deps[i]);
            }
        }
        if (cudaStreamEndCapture(s, &g) != cudaSuccess) { cudaGetLastError(); *ok = false; return nullptr; }
        {
            int n = 0; size_t ne = 0;
            cudaGraphGetEdges(g, nullptr, nullptr, nullptr, &ne);
            std::vector<cudaGraphNode_t> from(ne), to(ne); std::vector<cudaGraphEdgeData> edge_data(ne);
            if (ne && cudaGraphGetEdges(g, from.data(), to.data(), edge_data.data(), &ne) == cudaSuccess)
                for (size_t i = 0; i < ne; ++i)
                    if (boundaries.count(from[i]) && (edge_data[i].type != 0 || edge_data[i].from_port != 0)) {
                        ++n;
                        if (neutralize && (cudaGraphRemoveDependencies(g, &from[i], &to[i], &edge_data[i], 1) != cudaSuccess ||
                                            cudaGraphAddDependencies(g, &from[i], &to[i], nullptr, 1) != cudaSuccess)) {
                            printf("FAILED (judge): a PDL edge could not be neutralized\n"); fflush(stdout);
                            cudaGetLastError(); cudaGraphDestroy(g); *ok = false; return nullptr;
                        }
                    }
            cudaGetLastError();
            if (pdl) *pdl = n;
        }
        cudaGraphExec_t e;
        if (cudaGraphInstantiate(&e, g, 0) != cudaSuccess) { cudaGetLastError(); cudaGraphDestroy(g); *ok = false; return nullptr; }
        cudaGraphDestroy(g);
        return e;
    }
    double one_graph_ms(cudaGraphExec_t e)
    {
        CK(cudaGraphLaunch(e, s)); CK(cudaStreamSynchronize(s));
        CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(e, s)); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); return ms;
    }
    // The graph in a loop for `duration` seconds, the first `warmup` of them untimed. Returns the time of one graph
    // in ms. The persisting L2 is reset before each block.
    double block(cudaGraphExec_t e, double t_ms, double duration, double warmup, int arm)
    {
        cudaCtxResetPersistingL2Cache(); cudaGetLastError();
        const long nc = std::max(1L, (long)(warmup * 1e3 / t_ms));
        const long nm = std::max(3L, (long)((duration - warmup) * 1e3 / t_ms));
        for (long i = 0; i < nc; ++i) CK(cudaGraphLaunch(e, s));
        CK(cudaStreamSynchronize(s));
        g_arm = arm;
        CK(cudaEventRecord(e0, s));
        for (long i = 0; i < nm; ++i) CK(cudaGraphLaunch(e, s));
        CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        g_arm = -1;
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); return ms / nm;
    }
};

static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char** argv)
{
    if (argc < 2) { printf("usage: ./judge shapes.txt [options]\n"); return 1; }
    Options o; o.seed = (unsigned)std::chrono::steady_clock::now().time_since_epoch().count();
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next_arg = [&]() -> std::string { if (i + 1 >= argc) { printf("option %s without a value\n", a.c_str()); exit(1); } return argv[++i]; };
        if (a == "--pairs") o.pairs = atoi(next_arg().c_str());
        else if (a == "--duration") o.duration = atof(next_arg().c_str());
        else if (a == "--warmup") o.warmup = atof(next_arg().c_str());
        else if (a == "--quick") { o.quick = true; o.pairs = 2; o.duration = 0.6; o.warmup = 0.2; }
        else if (a == "--default") o.with_default = true;
        else if (a == "--seed") o.seed = (unsigned)strtoul(next_arg().c_str(), nullptr, 10);
        else if (a == "--reference") o.reference = next_arg();
        else if (a == "--only") { std::stringstream ss(next_arg()); std::string t; while (std::getline(ss, t, ',')) o.only.push_back(t); }
        else { printf("unknown option %s\n", a.c_str()); return 1; }
    }
    if (o.pairs < 1 || o.pairs > 32 || !(o.duration > o.warmup) || !(o.warmup >= 0)) { printf("invalid timing options\n"); return 1; }
    struct Shape { int M, N, K; std::string tag; };
    std::vector<Shape> shapes;
    { std::ifstream f(argv[1]); std::string l;
      while (std::getline(f, l)) {
          if (l.empty() || l[0] == '#') continue;
          std::stringstream ss(l); Shape x; ss >> x.M >> x.N >> x.K >> x.tag;
          if (o.only.empty() || std::find(o.only.begin(), o.only.end(), x.tag) != o.only.end()) shapes.push_back(x);
      } }

    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    Nvml nv; nv.init();
    unsigned cap_mw = 0; if (nv.ok && nv.cap) nv.cap(nv.dev, &cap_mw);
    int driver = 0; cudaDriverGetVersion(&driver);
    // The card's identity, written in each REFERENCE line: a cache serves only the same card.
    std::string card = p.name; for (char& c : card) if (c == ' ' || c == ',' || c == '|') c = '_';
    card += "|" + std::to_string(cap_mw / 1000) + "W|lt" + std::to_string(cublasLtGetVersion()) + "|cuda" + std::to_string(driver);

    // The configurations kept for THIS card: "REFERENCE label M N K card rank t_rel algo_hex".
    struct Memo { std::string key; cublasLtMatmulAlgo_t algo; };
    std::vector<Memo> cache;
    if (!o.reference.empty()) {
        std::ifstream f(o.reference); std::string l;
        while (std::getline(f, l)) {
            std::stringstream ss(l); std::string word, tag, c, h; int M, N, K, rank; double trel;
            if (!(ss >> word >> tag >> M >> N >> K >> c >> rank >> trel >> h) || word != "REFERENCE" || c != card) continue;
            Memo m; m.key = tag + " " + std::to_string(M) + " " + std::to_string(N) + " " + std::to_string(K);
            if (dehex(h, m.algo)) cache.push_back(m);
        }
    }
    printf("judge v4 (md5 %s): %s, sm_%d%d, %d SMs, power cap %u W, CUDA driver %d; cuBLASLt %zu; seed %u; pairs %d, "
           "block %.2f s with a warm-up of %.2f s%s\n", JUDGE_MD5, p.name, p.major, p.minor, p.multiProcessorCount, cap_mw / 1000,
           driver, cublasLtGetVersion(), o.seed, o.pairs, o.duration, o.warmup, o.quick ? " (quick)" : "");
    printf("card: %s\n", card.c_str());
    // This pass's token signs the REFERENCE lines, and the launcher caches only the lines that carry it. It stops a
    // naive injection only: a hostile candidate could read it in the output buffer of its own process.
    std::random_device rd; const unsigned long long token = ((unsigned long long)rd() << 32) ^ rd();
    printf("token: %016llx\n", token);
    if (!o.reference.empty()) printf("reference: %s (%zu configurations kept for this card)\n", o.reference.c_str(), cache.size());
    std::mt19937 rng(o.seed);
    if (!nv.ok) printf("NVML unavailable: no clock per arm\n");
    std::thread th(sampler, &nv);

    Bench bench; CK(cudaStreamCreateWithFlags(&bench.s, cudaStreamNonBlocking));
    cudaStream_t s_cand; CK(cudaStreamCreateWithFlags(&s_cand, cudaStreamNonBlocking));
    CK(cudaEventCreate(&bench.e0)); CK(cudaEventCreate(&bench.e1));
    cublasHandle_t hb; LT(cublasCreate(&hb)); LT(cublasSetStream(hb, bench.s));
    void* wsb; CK(cudaMalloc(&wsb, 64u << 20)); LT(cublasSetWorkspace(hb, wsb, 64u << 20));
    // cuBLAS's default call as a user gets it: cublasCreate then cublasSetStream, no workspace set (cublasSetStream
    // restores cuBLAS's pool workspace). hb, above, serves the fp32 references and the final's "64 MB" default.
    cublasHandle_t hp; LT(cublasCreate(&hp)); LT(cublasSetStream(hp, bench.s));
    Lt L; LT(cublasLtCreate(&L.h)); CK(cudaMalloc(&L.ws, L.wsMax));
    const float one = 1.f, zero = 0.f;
    const GpuState state0 = read_state(bench.s);

    printf("shape,M,N,K,reference,t_ref_us,t_cand_us,ratio,ratio_min,ratio_max,mhz_ref,mhz_cand,w_ref,w_cand,"
           "ratio_cycles,err_before,err_after,throttle_ref,throttle_cand%s\n", o.with_default ? ",ratio_default" : "");
    fflush(stdout);
    std::vector<double> ratios; int won = 0, lost = 0; bool correct = true;
    bool card_throttled = false;   // a thermal or hardware throttle seen during a timing
    const unsigned long long kAbnormalThrottle = 0x8ull | 0x20ull | 0x40ull | 0x80ull;   // HW slowdown, thermal (sw, hw), power brake
    struct Compute { double ratio, rcyc, mr, mc, wr, wc; };
    std::vector<Compute> compute_bound;   // the shapes with M >= 512, bound by compute: for the CYCLES line
    auto fail = [&](const char* what, const std::string& shape, double e) {
        printf("FAILED %s: shape %s%s", what, shape.c_str(), e >= 0 ? "" : "\n");
        if (e >= 0) printf(", error %.3e (tolerance %.0e)\n", e, kTol);
        fflush(stdout); correct = false;
    };
    for (const Shape& F : shapes) {
        const int M = F.M, N = F.N, K = F.K;
        const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
        half *A[3], *B[3], *Cc, *Cl, *Cd; float *C32, *C32g;
        for (int j = 0; j < 3; ++j) { CK(cudaMalloc(&A[j], nA * 2)); CK(cudaMalloc(&B[j], nB * 2)); }
        CK(cudaMalloc(&Cc, nC * 2)); CK(cudaMalloc(&Cl, nC * 2)); CK(cudaMalloc(&Cd, nC * 2));
        CK(cudaMalloc(&C32, nC * 4)); CK(cudaMalloc(&C32g, nC * 4));
        int ref_set = -1;   // the input set whose reference is in C32
        unsigned* dres; CK(cudaMalloc(&dres, 3 * sizeof(unsigned)));
        unsigned long long* dsum; CK(cudaMalloc(&dsum, sizeof(unsigned long long)));
        auto seed_inputs = [&](unsigned base) {
            ref_set = -1;
            for (int j = 0; j < 3; ++j) {
                fill<<<1024, 256, 0, bench.s>>>(A[j], nA, base + 2 * j); fill<<<1024, 256, 0, bench.s>>>(B[j], nB, base + 2 * j + 1);
            }
            CK(cudaStreamSynchronize(bench.s));
        };
        auto input_checksum = [&]() {
            unsigned long long tot = 0;
            for (int j = 0; j < 3; ++j) for (half* q : {A[j], B[j]}) {
                CK(cudaMemsetAsync(dsum, 0, sizeof *dsum, bench.s));
                checksum<<<1024, 256, 0, bench.s>>>(reinterpret_cast<const unsigned short*>(q), q == A[j] ? nA : nB, dsum);
                unsigned long long h; CK(cudaMemcpyAsync(&h, dsum, sizeof h, cudaMemcpyDeviceToHost, bench.s)); CK(cudaStreamSynchronize(bench.s));
                tot = tot * 1000003ull + h;
            }
            return tot;
        };
        // C row-major M x N = (B^T A^T)^T: in column-major, C^T (N x M) = B^T (N x K) . A^T (K x M).
        auto ref32 = [&](int j, float* dst) {
            LT(cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B[j], CUDA_R_16F, N, A[j], CUDA_R_16F, K, &zero,
                            dst, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        };
        // The gap of the output dst to an fp32 reference; INFINITY if an element is not finite.
        auto gap_of = [&](const half* dst, const float* ref) {
            CK(cudaMemsetAsync(dres, 0, 3 * sizeof(unsigned), bench.s));
            gap<<<1024, 256, 0, bench.s>>>(dst, ref, nC, dres);
            unsigned h[3]; CK(cudaMemcpyAsync(h, dres, sizeof h, cudaMemcpyDeviceToHost, bench.s)); CK(cudaStreamSynchronize(bench.s));
            if (h[2]) return (double)INFINITY;
            float e, r; memcpy(&e, &h[0], 4); memcpy(&r, &h[1], 4);
            return (double)e / r;
        };
        // The error of a call (the candidate on its stream, or a cuBLASLt configuration) on input set j.
        auto error_of = [&](const std::function<void(int)>& call, cudaStream_t s, half* dst, int j) {
            if (ref_set != j) { ref32(j, C32); CK(cudaStreamSynchronize(bench.s)); ref_set = j; }
            CK(cudaMemsetAsync(dst, 0xff, nC * 2, bench.s)); CK(cudaStreamSynchronize(bench.s));   // 0xffff = NaN
            call(j);
            if (cudaStreamSynchronize(s) != cudaSuccess || cudaGetLastError() != cudaSuccess) return (double)INFINITY;
            return gap_of(dst, C32);
        };
        auto cand = [&](int j) { candidate_gemm(A[j], B[j], Cc, M, N, K, s_cand); };
        auto check_state = [&](const char* when) {
            if (!(read_state(bench.s) == state0)) { printf("FAILED: the candidate changed the GPU's state (%s), shape %s\n", when, F.tag.c_str()); fflush(stdout); return false; }
            return true;
        };

        seed_inputs(o.seed * 7 + 1);
        const unsigned long long sum0 = input_checksum();
        double err_before = 0;
        for (int j = 0; j < 3; ++j) err_before = std::max(err_before, error_of(cand, s_cand, Cc, j));
        if (!(err_before <= kTol)) { fail("CORRECTNESS before the timing", F.tag, err_before); break; }
        if (!check_state("before the timing")) { correct = false; break; }
        if (input_checksum() != sum0) { fail("the candidate changed A or B (before the timing)", F.tag, -1); break; }

        // The reference: the best cuBLASLt configuration of this shape.
        LT(cublasLtMatmulDescCreate(&L.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        LT(cublasLtMatrixLayoutCreate(&L.la, CUDA_R_16F, N, K, N));
        LT(cublasLtMatrixLayoutCreate(&L.lb, CUDA_R_16F, K, M, K));
        LT(cublasLtMatrixLayoutCreate(&L.lc, CUDA_R_16F, N, M, N));
        auto lt_call = [&](Cfg& c, half* dst, int j) {
            return cublasLtMatmul(L.h, L.op, &one, B[j], L.la, A[j], L.lb, &zero, dst, L.lc, dst, L.lc, &c.algo, L.ws, L.wsMax, bench.s);
        };
        std::vector<Cfg> cfgs;
        {
            const std::string key = F.tag + " " + std::to_string(M) + " " + std::to_string(N) + " " + std::to_string(K);
            std::vector<Cfg> memo;
            for (auto& m : cache) if (m.key == key) {
                Cfg c; c.algo = m.algo; c.label = "cache" + std::to_string(memo.size());
                if (check_algo(L, c)) { read_cfg(c); memo.push_back(c); }
            }
            if (!memo.empty()) { cfgs = configurations(L, false, 3); cfgs.insert(cfgs.end(), memo.begin(), memo.end()); }
            else {
                if (!o.reference.empty()) printf("reference missing or refused for %s: full sort\n", F.tag.c_str());
                cfgs = configurations(L, true);
            }
        }
        printf("info: %s: %zu cuBLASLt configurations to sort\n", F.tag.c_str(), cfgs.size());
        std::vector<Cfg> good;
        for (Cfg& c : cfgs) {
            const double e = error_of([&](int j) { lt_call(c, Cl, j); }, bench.s, Cl, 0);
            if (e <= kTol) good.push_back(c);
        }
        if (good.empty()) { printf("no correct cuBLASLt configuration for %s\n", F.tag.c_str()); return 3; }
        // G calls per graph, so that a graph lasts at least 2 ms, and at least 3 so that the three input sets run.
        bool ok;
        cudaGraphExec_t g1 = bench.graph(bench.s, [&](int j) { lt_call(good[0], Cl, j); }, 1, &ok);
        if (!ok) { printf("cuBLASLt cannot be captured\n"); return 3; }
        const int G = std::max(3, std::min(200, (int)std::ceil(2.0 / bench.one_graph_ms(g1))));
        cudaGraphExecDestroy(g1);
        std::vector<cudaGraphExec_t> gl(good.size());
        for (size_t i = 0; i < good.size(); ++i) {
            gl[i] = bench.graph(bench.s, [&](int j) { lt_call(good[i], Cl, j); }, G, &ok);
            if (!ok) { printf("configuration %s cannot be captured\n", good[i].label.c_str()); return 3; }
        }
        // Sort: one short block per configuration, in random order; then a final of the first ones in the regime of
        // the pairs.
        std::vector<int> order(good.size()); for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
        std::shuffle(order.begin(), order.end(), rng);
        for (int i : order) good[i].t = bench.block(gl[i], bench.one_graph_ms(gl[i]), 0.35, 0.1, -1);
        std::vector<int> rank = order;
        std::sort(rank.begin(), rank.end(), [&](int a, int b) { return good[a].t < good[b].t; });
        const int nf = std::min<int>(kFinalists, (int)rank.size());
        std::vector<std::vector<double>> tf(nf);
        // cuBLAS's default call (cublasGemmEx, the configuration cuBLAS picks itself) enters the final, with a user's
        // handle (hp, cuBLAS's pool) and with the 64 MB one (hb).
        auto default_hp = [&](int j) {
            cublasGemmEx(hp, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B[j], CUDA_R_16F, N, A[j], CUDA_R_16F, K, &zero,
                         Cl, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); };
        auto default_hb = [&](int j) {
            cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B[j], CUDA_R_16F, N, A[j], CUDA_R_16F, K, &zero,
                         Cl, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); };
        const std::function<void(int)> defaults[2] = {default_hp, default_hb};
        // No comma: desc_ref is a field of the CSV line.
        const char* default_names[2] = {"cuBLAS default pool (cublasGemmEx)", "cuBLAS default 64 MB (cublasGemmEx)"};
        cudaGraphExec_t gd[2] = {nullptr, nullptr}; std::vector<double> td[2]; std::string default_state[2];
        for (int d = 0; d < 2; ++d) {
            if (!(error_of(defaults[d], bench.s, Cl, 0) <= kTol)) { default_state[d] = "wrong"; continue; }
            bool okd; gd[d] = bench.graph(bench.s, defaults[d], G, &okd);
            if (!okd) { gd[d] = nullptr; default_state[d] = "not capturable"; }
        }
        for (int r = 0; r < (o.quick ? 1 : 2); ++r) {
            std::vector<int> oo; for (int i = 0; i < nf; ++i) oo.push_back(i);
            for (int d = 0; d < 2; ++d) if (gd[d]) oo.push_back(nf + d);
            std::shuffle(oo.begin(), oo.end(), rng);
            for (int i : oo) {
                if (i >= nf) td[i - nf].push_back(bench.block(gd[i - nf], bench.one_graph_ms(gd[i - nf]), o.duration, o.warmup, -1));
                else tf[i].push_back(bench.block(gl[rank[i]], bench.one_graph_ms(gl[rank[i]]), o.duration, o.warmup, -1));
            }
        }
        std::vector<int> standing(nf); for (int i = 0; i < nf; ++i) standing[i] = i;
        std::sort(standing.begin(), standing.end(), [&](int a, int b) { return median(tf[a]) < median(tf[b]); });
        const int best = rank[standing[0]];
        const double t_lt = median(tf[standing[0]]);
        int dref = -1;   // the default call that beats the best cuBLASLt configuration, if any (the faster of the two)
        for (int d = 0; d < 2; ++d) if (gd[d] && median(td[d]) < t_lt && (dref < 0 || median(td[d]) < median(td[dref]))) dref = d;
        const bool ref_is_default = dref >= 0;
        {
            char e[2][48];
            for (int d = 0; d < 2; ++d) {
                if (gd[d]) snprintf(e[d], sizeof e[d], "%.4f", median(td[d]) / t_lt);
                else snprintf(e[d], sizeof e[d], "%s", default_state[d].c_str());
            }
            printf("info: final of %s: cuBLAS default pool %s, 64 MB %s (time / best cuBLASLt configuration)\n",
                   F.tag.c_str(), e[0], e[1]);
        }
        for (int r = 0; r < nf; ++r) {   // the finalists, for the cache (the winner at rank 0)
            const Cfg& c = good[rank[standing[r]]];
            printf("REFERENCE %s %d %d %d %s %d %.4f %s token=%016llx\n", F.tag.c_str(), M, N, K, card.c_str(), r,
                   median(tf[standing[r]]) / median(tf[standing[0]]), hex(c.algo).c_str(), token);
        }
        fflush(stdout);
        for (size_t i = 0; i < gl.size(); ++i) if ((int)i != best || ref_is_default) cudaGraphExecDestroy(gl[i]);
        for (int d = 0; d < 2; ++d) if (gd[d] && d != dref) cudaGraphExecDestroy(gd[d]);
        cudaGraphExec_t gref = ref_is_default ? gd[dref] : gl[best]; const Cfg cref = good[best];
        const std::string desc_ref = ref_is_default ? std::string(default_names[dref]) : describe(cref);
        std::function<void(int)> ref_call = [&](int j) { lt_call(good[best], Cl, j); };
        if (ref_is_default) {
            ref_call = defaults[dref];
            printf("info: the reference of %s is the %s: it beats the best cuBLASLt configuration (time %.4f times its own)\n",
                   F.tag.c_str(), default_names[dref], median(td[dref]) / t_lt);
        }
        int pdl_ref = 0;
        { cudaGraphExec_t x = bench.graph(bench.s, ref_call, 2, &ok, &pdl_ref); if (ok) cudaGraphExecDestroy(x); }

        int pdl = 0;
        cudaGraphExec_t gcand = bench.graph(s_cand, cand, G, &ok, &pdl);
        if (!ok) { printf("FAILED: the candidate cannot be captured in a graph (shape %s)\n", F.tag.c_str()); return 2; }
        if (pdl) printf("info: %d PDL dependency(ies) between two calls of the candidate, neutralized in the timing (shape %s)\n", pdl, F.tag.c_str());
        if (pdl_ref) {   // for the record: what PDL between calls would have brought the reference (not in the score)
            bool ok2; cudaGraphExec_t gpdl = bench.graph(bench.s, ref_call, G, &ok2, nullptr, false);
            if (ok2) {
                const double tn = bench.one_graph_ms(gref), tp = bench.one_graph_ms(gpdl);
                std::vector<double> without, with_pdl;
                for (int r = 0; r < 2; ++r) { without.push_back(bench.block(gref, tn, o.duration, o.warmup, -1)); with_pdl.push_back(bench.block(gpdl, tp, o.duration, o.warmup, -1)); }
                printf("info: the reference at %s chains its calls by PDL (%d edge(s)), neutralized in the timing; with that PDL, "
                       "its time would be %.4f times the measured one\n", F.tag.c_str(), pdl_ref, median(with_pdl) / median(without));
                cudaGraphExecDestroy(gpdl);
            }
        }
        cudaGraphExec_t gdef = nullptr;
        if (o.with_default) {
            gdef = bench.graph(bench.s, [&](int j) {
                cublasGemmEx(hp, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B[j], CUDA_R_16F, N, A[j], CUDA_R_16F, K, &zero,
                             Cd, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); }, G, &ok);
            if (!ok) { printf("cublasGemmEx cannot be captured\n"); return 3; }
        }

        // The fp32 reference of the input set the candidate's graph writes last, to check each of its blocks.
        const int jg = (G - 1) % 3;
        ref32(jg, C32g); CK(cudaStreamSynchronize(bench.s));

        // The pairs of blocks.
        { std::lock_guard<std::mutex> l(g_mx); for (auto& e : g_samples) e = Sample(); }
        cudaGraphExec_t gs[3] = {gref, gcand, gdef};
        const int nb = o.with_default ? 3 : 2;
        double tg[3]; for (int b = 0; b < nb; ++b) tg[b] = bench.one_graph_ms(gs[b]);
        std::vector<double> tb[3];
        double err_blocks = 0;
        for (int r = 0; r < o.pairs; ++r) {
            std::vector<int> oo(nb); for (int b = 0; b < nb; ++b) oo[b] = b;
            std::shuffle(oo.begin(), oo.end(), rng);
            for (int b : oo) {
                tb[b].push_back(bench.block(gs[b], tg[b], o.duration, o.warmup, b));
                if (b == 1) err_blocks = std::max(err_blocks, gap_of(Cc, C32g));
            }
        }
        std::vector<double> rp(o.pairs);
        for (int r = 0; r < o.pairs; ++r) rp[r] = tb[0][r] / tb[1][r];
        const double ratio = median(rp);
        const double tref = median(tb[0]) / G * 1e3, tcand = median(tb[1]) / G * 1e3;
        Sample samples[3]; { std::lock_guard<std::mutex> l(g_mx); for (int b = 0; b < 3; ++b) samples[b] = g_samples[b]; }
        auto mhz = [&](int b) { return samples[b].n ? samples[b].mhz / samples[b].n : -1.0; };
        auto watts = [&](int b) { return samples[b].n ? samples[b].w / samples[b].n : -1.0; };
        const double rcyc = (mhz(0) > 0 && mhz(1) > 0) ? ratio * mhz(0) / mhz(1) : -1.0;

        // After the timing: new contents, the timed graph run again; new contents again, direct calls.
        double err_after = err_blocks;
        seed_inputs(o.seed * 7 + 1000);
        {
            ref32(jg, C32g); CK(cudaStreamSynchronize(bench.s));
            CK(cudaMemsetAsync(Cc, 0xff, nC * 2, bench.s)); CK(cudaStreamSynchronize(bench.s));
            CK(cudaGraphLaunch(gcand, bench.s));
            err_after = std::max(err_after, cudaStreamSynchronize(bench.s) == cudaSuccess ? gap_of(Cc, C32g) : (double)INFINITY);
        }
        seed_inputs(o.seed * 7 + 2000);
        const unsigned long long sum1 = input_checksum();
        for (int j = 0; j < 3; ++j) err_after = std::max(err_after, error_of(cand, s_cand, Cc, j));
        const bool inputs_intact = input_checksum() == sum1;
        const bool state_intact = check_state("after the timing");

        char buf[512];
        snprintf(buf, sizeof buf, "%s,%d,%d,%d,%s,%.2f,%.2f,%.4f,%.4f,%.4f,%.0f,%.0f,%.1f,%.1f,%.4f,%.2e,%.2e,0x%llx,0x%llx",
                 F.tag.c_str(), M, N, K, desc_ref.c_str(), tref, tcand, ratio,
                 *std::min_element(rp.begin(), rp.end()), *std::max_element(rp.begin(), rp.end()),
                 mhz(0), mhz(1), watts(0), watts(1), rcyc, err_before, err_after, samples[0].reasons, samples[1].reasons);
        std::string line = buf;
        if (o.with_default) { char d[32]; snprintf(d, sizeof d, ",%.4f", median(tb[0]) / median(tb[2])); line += d; }
        const bool valid = err_after <= kTol && inputs_intact && state_intact;
        printf("%s%s\n", valid ? "" : "# timing refused (see FAILED): ", line.c_str());
        if ((samples[0].reasons | samples[1].reasons) & kAbnormalThrottle) {
            card_throttled = true;
            printf("WARNING: thermal or hardware throttling while timing %s (NVML reasons: reference 0x%llx, candidate 0x%llx)\n",
                   F.tag.c_str(), samples[0].reasons, samples[1].reasons);
        }
        fflush(stdout);
        if (!(err_after <= kTol)) { fail("CORRECTNESS during or after the timing (timed blocks, graph run again, direct calls)", F.tag, err_after); break; }
        if (!inputs_intact) { fail("the candidate changed A or B (after the timing)", F.tag, -1); break; }
        if (!state_intact) { correct = false; break; }
        ratios.push_back(ratio);
        if (M >= 512 && rcyc > 0) compute_bound.push_back({ratio, rcyc, mhz(0), mhz(1), watts(0), watts(1)});
        if (ratio > 1.0 && *std::min_element(rp.begin(), rp.end()) > 1.0) ++won;
        if (ratio < 1.0 && *std::max_element(rp.begin(), rp.end()) < 1.0) ++lost;

        for (int b = 0; b < nb; ++b) cudaGraphExecDestroy(gs[b]);
        cublasLtMatmulDescDestroy(L.op); cublasLtMatrixLayoutDestroy(L.la); cublasLtMatrixLayoutDestroy(L.lb); cublasLtMatrixLayoutDestroy(L.lc);
        for (int j = 0; j < 3; ++j) { cudaFree(A[j]); cudaFree(B[j]); }
        cudaFree(Cc); cudaFree(Cl); cudaFree(Cd); cudaFree(C32); cudaFree(C32g); cudaFree(dres); cudaFree(dsum);
    }
    g_done = true; th.join();
    if (!correct) { printf("SCORE INVALID (correctness or rule)\n"); return 2; }
    if (ratios.empty()) { printf("no shape timed\n"); return 1; }
    double lg = 0; for (double r : ratios) lg += std::log(r);
    printf("SCORE %.4f (geometric mean of the ratios reference time / candidate time over %zu shapes); "
           "won %d, lost %d (all pairs on the same side); card %s\n", std::exp(lg / ratios.size()), ratios.size(),
           won, lost, card.c_str());
    if (!compute_bound.empty()) {
        double lr = 0, lc = 0, mr = 0, mc = 0, wr = 0, wc = 0;
        for (const Compute& c : compute_bound) { lr += std::log(c.ratio); lc += std::log(c.rcyc); mr += c.mr; mc += c.mc; wr += c.wr; wc += c.wc; }
        const double n = (double)compute_bound.size();
        printf("CYCLES %.4f (geometric mean of the ratios in cycles over the %zu shapes with M >= 512, bound by compute; "
               "in time over these shapes: %.4f); mean clock reference %.0f MHz, candidate %.0f MHz; mean power %.1f W "
               "and %.1f W, power cap %u W: under a power cap, time = energy / power, and a clock gain is an energy "
               "gain\n", std::exp(lc / n), compute_bound.size(), std::exp(lr / n), mr / n, mc / n, wr / n, wc / n, cap_mw / 1000);
    }
    if (card_throttled) printf("CARD THROTTLED (thermal or hardware): this pass does not count as an official pair\n");
    return 0;
}
