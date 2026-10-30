// The host emulator of the Hopper rung (README.md in this directory), the Ampere rung's (ampere/emu) cut down to what
// step 0 needs, then grown with the rung, each step adding only the engines it brings. It compiles the kernels' own
// source, nvidia_sample.cuh (step 0's compute_gemm, and simple_wmma_gemm, the reference point), step1.cuh and
// step2.cuh, as C++ against the headers in shim/, and runs each kernel on the CPU at the shapes of verif.cu, in three
// orders of its threads. It checks the result against a reference in double, every global and shared access, that each
// element of C is written exactly once, and the shared-memory wavefronts against the values the headers state.
// Step 1's engines: the warpgroup MMA (wgmma.mma_async: its descriptors and the 128-byte swizzle read as the PTX ISA
// states them, its accumulator layout, its groups), performed when issued in the forward order, as late as the program
// allows in the reverse order (when a wait_group retires its group, the accumulators NaN until then), and either way,
// chain by chain, in the random order; and the async proxy (a wgmma that reads a shared byte stored since its writer's
// last fence.proxy.async is an error). Step 2's: the mbarriers (arrivals, bytes, phases, waits by parity) and the TMA's
// 2D loads (the box, zeros outside the matrix, the 128-byte swizzle), landing once nothing else can go on in the
// forward order, when issued in the reverse order, and load by load at random in the random order: a wgmma that reads
// a stage before its load lands, and a load that overwrites a stage a wgmma still reads, both give a wrong C; threads
// waiting at mbarriers that nothing can complete are a DEADLOCK.
//   build (in hopper/emu): g++ -std=c++17 -O2 -march=native -I.. -Ishim emu.cpp -o emu
//   run:                   ./emu [step0|step1|step2|all]... [-x]   (default: all; -x: stop at the first failure)
// Exit code 0 only if every run passes; 1 if a run fails; 2 on a usage or internal error.
#define GEMM_LADDER_EMULATE   // step1.cuh: wgmma, the async proxy; step2.cuh: TMA, mbarriers; step3.cuh: elect.sync,
                              // named barriers; step4.cuh: clusters; step5.cuh: TMA stores; step6.cuh: transposed wgmma
                              // and stmatrix .trans; step7.cuh: setmaxnreg, mapa, the multicast TMA
#include <signal.h>
#include <sys/mman.h>
#include <ucontext.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "nvidia_sample.cuh"
#include "step1.cuh"
#include "step2.cuh"
#include "step3.cuh"
#include "step4.cuh"
#include "step5.cuh"
#include "step6.cuh"
#include "step7.cuh"

// ---------------------------------------------------------------- what the shim headers and the kernels refer to
emu_dim3 threadIdx, blockIdx, gridDim, blockDim;

// The kernels' dynamic shared memory (extern __shared__ half shmem[], one per namespace). A launch uses the first
// bytes; the rest of the 228 KB keeps the NaN fill written before each block, a canary checked after it.
constexpr int kShArrayBytes = 228 << 10;   // a block's 227 KB of shared memory on sm_90, and a canary
namespace nvidia_sample { alignas(4096) half shmem[kShArrayBytes / 2]; }
namespace step1 { alignas(4096) uint8_t shmem[kShArrayBytes]; }
namespace step2 { alignas(4096) uint8_t shmem[kShArrayBytes]; }

// ---------------------------------------------------------------- the launch under way
constexpr size_t kPage = 4096, kGuardBytes = 8 << 20;
struct Array {   // A, B or C: rows x cols halves, row-major, between two guard regions (PROT_NONE)
    const char* name;
    int rows = 0, cols = 0;
    bool writable = false;
    char* map = nullptr;   // the mapping, guard regions included
    size_t mapBytes = 0;
    half* p = nullptr;     // the array, which ends where the second guard region begins
    const char* lo() const { return (const char*)p; }
    const char* hi() const { return (const char*)(p + (size_t)rows * cols); }
};
static Array g_A{"A"}, g_B{"B"}, g_C{"C"};
static half* g_sh;                       // the launch's shared memory (the kernel's namespace's shmem)
static int g_shBytes, g_bStageBytes;     // its size; where B begins in a stage (a copy's store below it is to A)
static int g_stageBytes;                 // the size of a stage (a 16-byte store's class: its offset modulo it)
static int g_block;                      // the running block (linear index)
// Step 4: the blocks of a cluster run together. The cluster's blocks 0 to g_cluster - 1, g_tpb threads each (thread t
// of the run is thread t % g_tpb of block t / g_tpb); each block's shared memory, linear index and blockIdx; whether
// all its threads have left.
constexpr int kMaxCluster = 8;
static int g_cluster = 1, g_tpb;
static half* g_shBlocks[kMaxCluster];
static int g_blockLin[kMaxCluster];
static emu_dim3 g_blockIdx[kMaxCluster];
static bool g_blockLeft[kMaxCluster];
alignas(4096) static uint8_t g_clusterSh[kMaxCluster][kShArrayBytes];
static long g_errors;                    // errors found in the current run
constexpr long kShownErrors = 6;         // printed per run; the others are counted
static std::vector<uint8_t> g_cWrites;   // per element of C: the writes counted (up to 255)
static std::vector<int> g_cFirst;        // per element of C: the block of its first write
static long g_cAgain;                    // the first element of C written twice (-1: none), and by which blocks
static int g_cAgainBlocks[2];
struct Count { long instr = 0, wavefronts = 0; };
static std::map<std::string, Count> g_banks;   // per class of shared instruction: instructions, wavefronts
static char g_label[200];                // the current run, for the fault handler

static void error(const char* fmt, ...)
{
    if (++g_errors > kShownErrors) return;
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    printf("EMU ERROR: %s\n", msg);
}


// ---------------------------------------------------------------- threads: coroutines, barriers, warpgroup MMA
constexpr int kMaxThreads = 1536;
constexpr size_t kStackBytes = 256 << 10;
enum State { kRunnable, kAtWgmma, kAtMbar, kAtNamed, kAtShfl, kAtCluster, kAtStsm, kAtRegs, kAtLateWait, kExited };
struct Access { bool store; int bytes; long offset; };   // a shared access (offset in bytes)
struct Store { CUtensorMap map; int r; long src; int c0, c1; bool done; };   // step 5: a TMA store (r: its block)
struct Thread {
    ucontext_t ctx;
    State state;
    std::vector<Access> log;  // its shared accesses since the last barrier, in program order
    // Step 1: at a wgmma, its operands; its commits; whether it has run wgmma.fence since its warpgroup's accumulators
    // were last handed back; its shared stores not yet made visible to the async proxy (offset, bytes).
    float* acc;
    int n, scaleD;
    uint64_t da, db;
    int tA, tB;   // step 6: the transposes of its wgmma (tnspA, tnspB; steps 1 to 5: 0 and 1)
    long commits;
    bool fenced;
    std::vector<std::pair<long, int>> unfenced;
    // Step 1: at the wait of a late chain, the round of the scheduler from which it resumes.
    long lateRound;
    // Step 2: the mbarrier it waits for (shared offset) and the parity of the phase.
    long mbar;
    uint32_t parity;
    // Step 3: the named barrier it waits at (id, count of threads), its elect.sync so far. Step 7: its last setmaxnreg
    // (0: none) and its setmaxnreg so far; its registers (the launch's, then its setmaxnreg's), those it asks at a
    // setmaxnreg.inc.
    int named, namedCount, elects, maxnreg, maxnregs;
    int regs, regsWanted;
    int shflV, shflSrc, shflOut;   // at a __shfl_sync: its value, its source lane; its result
    // Step 4: its arrivals at and waits on the cluster barrier so far.
    int clArrived, clWaited;
    // Step 6: at a stmatrix .trans, its address and registers. Step 5: its TMA stores not yet performed: issued since
    // its last commit (open), and in committed groups, oldest first.
    uint32_t stsmAddr, stsmR[4];
    bool stsmTrans;   // step 6: stmatrix .trans
    std::vector<Store> bulkOpen;
    std::deque<std::vector<Store>> bulkGroups;
};
static Thread g_thr[kMaxThreads];
static ucontext_t g_sched;
static int g_cur = -1, g_threads;   // the running thread (linear index in the run); threads in the run (a cluster)
static long g_round;                // the scheduler's rounds so far in the run
static char* g_stacks;              // kMaxThreads x (a PROT_NONE page, then a stack of kStackBytes)
static void (*g_kernel)(const half*, const half*, half*, int, int, int);
static int g_M, g_N, g_K;
static bool g_abort;                // the block cannot go on (a warpgroup's threads at different wgmma)
static std::vector<int> g_unfenced; // per byte of each block's shared memory (block r at r * kShArrayBytes): 1 + the
                                    // thread that stored it, until that thread's fence

// Step 7: each block's pool of registers given back by setmaxnreg.dec and not yet taken by .inc (per thread, in
// registers); the registers per thread at the launch (Kernel::regs).
static long g_regPool[kMaxCluster];
static int g_launchRegs = 168;

static const char* who()
{
    static char s[64];
    if (g_cur >= 0) snprintf(s, sizeof s, "block %d, thread %d", g_block, g_cur % g_tpb);
    else snprintf(s, sizeof s, "block %d", g_block);
    return s;
}

void __syncthreads()   // the block barrier: named barrier 0 for all the block's threads
{
    Thread& th = g_thr[g_cur];
    th.named = 0;
    th.namedCount = g_tpb;
    th.state = kAtNamed;
    swapcontext(&th.ctx, &g_sched);
}

int emu_lane() { return g_cur % 32; }

static void wgmma_at_exit();
static void bulk_at_exit();

static void thread_main()
{
    g_kernel(g_A.p, g_B.p, g_C.p, g_M, g_N, g_K);
    wgmma_at_exit();
    bulk_at_exit();
    g_thr[g_cur].state = kExited;
}

static void resume(int t)   // runs thread t until it reaches a barrier, a wgmma or its end
{
    g_cur = t;
    const int r = t / g_tpb, local = t % g_tpb;
    threadIdx = {local % blockDim.x, local / blockDim.x, 0};
    blockIdx = g_blockIdx[r];
    g_block = g_blockLin[r];
    g_sh = g_shBlocks[r];
    swapcontext(&g_sched, &g_thr[t].ctx);
    g_cur = -1;
}

// ---------------------------------------------------------------- the bank model
// 32 banks of 4 bytes. A shared access of `bytes` per lane is served in passes of 128 / bytes lanes (a quarter warp
// for 16 bytes, the whole warp for 4); in a pass, distinct addresses that fall in one bank (4-byte accesses) or in one
// group of 4 banks (16-byte accesses: groups of 16 bytes in a 128-byte line) take one wavefront each, equal addresses
// one in all. An ldmatrix.x4 is served as four 16-byte accesses of 8 lanes, one per 8 x 8 matrix (its 8 rows).
static int wavefronts(const long* off, int lanes, int bytes)
{
    const int perPass = 128 / bytes;   // lanes per pass, and groups per 128-byte line
    int total = 0;
    for (int p0 = 0; p0 < lanes; p0 += perPass) {
        int worst = 0;
        for (int g = 0; g < perPass; g++) {
            long seen[32];
            int n = 0;
            for (int l = p0; l < std::min(lanes, p0 + perPass); l++)
                if (off[l] / bytes % perPass == g && std::find(seen, seen + n, off[l]) == seen + n) seen[n++] = off[l];
            worst = std::max(worst, n);
        }
        total += worst;
    }
    return total;
}

// At a barrier (or the end): the plain shared accesses of each warp since the previous one. The k-th access of each
// lane is one instruction of the warp (the kernels' accesses are converged: the 32 lanes make the same sequence). A
// 16-byte store is a copy of a slice, to A's stages or to B's by its place.
static void count_shared_accesses(const std::vector<bool>* only = nullptr)   // only: the warps to account (all if null)
{
    for (int w = 0; w < g_threads / 32; w++) {
        if (only && !(*only)[w]) continue;
        Thread* lane = &g_thr[32 * w];
        const std::vector<Access>& first = lane[0].log;
        bool converged = true;
        for (int l = 1; l < 32 && converged; l++) {
            converged = lane[l].log.size() == first.size();
            for (size_t k = 0; converged && k < first.size(); k++)
                converged = lane[l].log[k].store == first[k].store && lane[l].log[k].bytes == first[k].bytes;
        }
        if (!converged)
            error("warp %d of block %d: its lanes made different shared accesses between two barriers (the bank model "
                  "needs the 32 lanes to make the same sequence)", w, g_block);
        else
            for (size_t k = 0; k < first.size(); k++) {
                long off[32];
                for (int l = 0; l < 32; l++) off[l] = lane[l].log[k].offset;
                std::string cls = std::string(first[k].store ? "STS." : "LDS.") + std::to_string(8 * first[k].bytes);
                if (first[k].store && first[k].bytes == 16) cls += off[0] % g_stageBytes < g_bStageBytes ? " A" : " B";
                Count& c = g_banks[cls];
                c.instr++;
                c.wavefronts += wavefronts(off, 32, first[k].bytes);
            }
        for (int l = 0; l < 32; l++) lane[l].log.clear();
    }
}

// ---------------------------------------------------------------- step 1: the warpgroup MMA
// The PTX ISA's wgmma.mma_async m64nNk16 .f32.f16.f16, A and B from shared memory through their descriptors, each
// K-major or MN-major (tnspA, tnspB: steps 1 to 5 take A K-major and B MN-major; step 6's bands the other way). A chain
// is the run of MMA of a warpgroup from the first one issued while none is in flight to the wait_group that leaves
// none; its MMA may write several ranges of accumulators (step 6: two MMA of different widths per K16, or one per band
// of 64 columns), each one array per thread. While the chain lasts, its ranges live here (shadow, per thread of the
// warpgroup) and the threads' registers hold NaN when the chain is performed late; an early chain is performed when
// issued, on the registers. Early (forward order): each MMA when issued; late (reverse order): each group when a
// wait_group retires it, reading shared memory then; random order: each chain one way or the other at random.
struct WgOp { uint64_t da, db; int n, scaleD, range, tA, tB; };
struct AccRange {
    int n = 0;                               // its width
    float* regs[128];                        // each thread's accumulator array
    std::vector<float> shadow;               // late chains: [thread of the warpgroup][n / 2]
};
struct WarpGroup {
    std::vector<WgOp> open;                  // issued since the last commit (late chains only)
    std::deque<std::vector<WgOp>> groups;    // committed and not yet performed (late chains only), oldest first
    long created = 0, retired = 0;           // groups committed, groups retired
    bool inFlight = false, late = false;     // a chain in flight, performed late
    std::vector<AccRange> ranges;            // the chain's ranges of accumulators
};
static WarpGroup g_wg[kMaxThreads / 128 + 4];
// Warpgroups are counted within a block (four warps from a multiple of 128 threads of the block; a block's warps after
// its last multiple of 128 threads would make a partial one), each cluster block's from its own thread 0. wgIndex: a
// warpgroup's index in the run; wgFirst: its first thread.
static int wgPerBlock() { return (g_tpb + 127) / 128; }
static int wgIndex(int t) { return t / g_tpb * wgPerBlock() + t % g_tpb / 128; }
static int wgFirst(int t) { return t / g_tpb * g_tpb + t % g_tpb / 128 * 128; }
static int g_wgMode;                         // 0: chains early (forward order), 1: late (reverse), 2: at random
static std::mt19937 g_wgRng;

// The shared byte of a descriptor's operand at offset `off` from its start, with the 128-byte swizzle (bits 4-6 of the
// address XOR bits 7-9: chunk c of a 128-byte row r of an atom of 1024 bytes at chunk c ^ (r % 8)); checked inside the
// launch's shared memory and against the bytes not yet fenced for the async proxy.
static bool operand(int r, uint32_t start, uint32_t off, const char* which, half* v)   // r: the warpgroup's block
{
    uint32_t a = start + off;
    a ^= ((a >> 7) & 7) << 4;
    if (a + 2 > (uint32_t)g_shBytes) {
        error("wgmma: %s's element at shared byte %u, outside the %d bytes of the launch (%s)", which, a, g_shBytes, who());
        return false;
    }
    for (uint32_t b = a; b < a + 2; b++) {
        int& u = g_unfenced[(size_t)r * kShArrayBytes + b];
        if (u) {
            error("wgmma reads shared byte %u of %s, stored by thread %d without a fence.proxy.async since (%s)", b,
                  which, u - 1, who());
            u = 0;   // reported once
        }
    }
    memcpy(v, (const char*)g_shBlocks[r] + a, 2);
    return true;
}

static bool decode(uint64_t d, const char* which, uint32_t& start, uint32_t& lbo, uint32_t& sbo)
{
    start = (uint32_t)(d & 0x3FFF) << 4;
    lbo = (uint32_t)((d >> 16) & 0x3FFF) << 4;
    sbo = (uint32_t)((d >> 32) & 0x3FFF) << 4;
    const int base = (int)((d >> 49) & 7), swizzle = (int)(d >> 62);
    if (swizzle != 1 || base != 0) {
        error("wgmma: %s's descriptor has swizzle mode %d and base offset %d; only the 128-byte swizzle with base 0 is "
              "emulated (%s)", which, swizzle, base, who());
        return false;
    }
    return true;
}

// The byte of element (i, k) of an operand, i along M (A) or N (B), k along its K16, at the 128-byte swizzle: K-major,
// row i at (i / 8) * stride + (i % 8) * 128, its K16 in 32 bytes; MN-major, i at (i / 64) * leading + (i % 64) * 2,
// row k of K at (k / 8) * stride + (k % 8) * 128.
static uint32_t element(bool mnMajor, int i, int k, uint32_t lead, uint32_t stride)
{
    return mnMajor ? (i / 64) * lead + (k / 8) * stride + (k % 8) * 128 + 2 * (i % 64)
                   : (i / 8) * stride + (i % 8) * 128 + 2 * k;
}

// One MMA on the accumulators acc[thread][n / 2] of a warpgroup: D (64 x n) = A (64 x 16) B (16 x n) (+ D).
static void perform(const WgOp& op, float* const* acc, int r)   // r: the warpgroup's block
{
    uint32_t sa, la, ka, sb, lb, kb;
    if (!decode(op.da, "A", sa, la, ka) || !decode(op.db, "B", sb, lb, kb)) return;
    static float A[64][16], B[16][256];
    for (int m = 0; m < 64; m++)
        for (int k = 0; k < 16; k++) {
            half h = (half)NAN;
            operand(r, sa, element(op.tA, m, k, la, ka), "A", &h);
            A[m][k] = (float)h;
        }
    for (int k = 0; k < 16; k++)
        for (int j = 0; j < op.n; j++) {
            half h = (half)NAN;
            operand(r, sb, element(op.tB, j, k, lb, kb), "B", &h);
            B[k][j] = (float)h;
        }
    for (int T = 0; T < 128; T++) {       // lane l of warp w: rows 16 w + l / 4 (+ 8), columns 8 j + 2 (l % 4) (+ 1)
        const int w = T / 32, g = (T % 32) / 4, q = T % 4;
        float* d = acc[T];
        for (int j = 0; j < op.n / 8; j++)
            for (int e = 0; e < 4; e++) {
                const int m = 16 * w + g + 8 * (e / 2), c = 8 * j + 2 * q + e % 2;
                float s = op.scaleD ? d[4 * j + e] : 0.0f;
                for (int k = 0; k < 16; k++) s += A[m][k] * B[k][c];
                d[4 * j + e] = s;
            }
    }
}

// The chain's accumulators handed back to the threads' registers: the chain is over.
static void end_chain(WarpGroup& W, int first)
{
    if (W.late)
        for (AccRange& a : W.ranges)
            for (int T = 0; T < 128; T++) memcpy(a.regs[T], &a.shadow[(size_t)T * (a.n / 2)], a.n / 2 * sizeof(float));
    W.ranges.clear();
    W.inFlight = false;
    for (int T = 0; T < 128; T++) g_thr[first + T].fenced = false;   // the next chain needs a wgmma.fence
}

// When the 128 threads of warpgroup k wait at a wgmma: checks they give the same operands, issues the MMA and releases
// them. False if some are not there yet.
static bool complete_wgmma(int k)
{
    const int first = k / wgPerBlock() * g_tpb + k % wgPerBlock() * 128;
    if (first + 128 > (first / g_tpb + 1) * g_tpb) return false;   // a partial warpgroup (not a whole 128 threads)
    for (int T = 0; T < 128; T++)
        if (g_thr[first + T].state != kAtWgmma) return false;
    const Thread& t0 = g_thr[first];
    for (int T = 1; T < 128; T++) {
        const Thread& t = g_thr[first + T];
        if (t.da != t0.da || t.db != t0.db || t.n != t0.n || t.scaleD != t0.scaleD || t.tA != t0.tA || t.tB != t0.tB) {
            error("warpgroup %d of block %d: its threads issue a wgmma with different operands (thread %d against %d)",
                  k, g_block, first + T, first);
            g_abort = true;
            return false;
        }
    }
    for (int T = 0; T < 128; T++)
        if (!g_thr[first + T].fenced) {
            error("warpgroup %d of block %d: a wgmma with no wgmma.fence since the accumulators were last handed back "
                  "(thread %d)", k, g_block, first + T);
            break;
        }
    WarpGroup& W = g_wg[k];
    WgOp op = {t0.da, t0.db, t0.n, t0.scaleD, -1, t0.tA, t0.tB};
    if (!W.inFlight) {   // a new chain
        W.inFlight = true;
        W.late = g_wgMode == 1 || (g_wgMode == 2 && (g_wgRng() & 1));
        W.ranges.clear();
    }
    for (size_t i = 0; i < W.ranges.size(); i++)   // the range this MMA writes: one of the chain's, or a new one
        if (W.ranges[i].regs[0] == t0.acc) op.range = (int)i;
    if (op.range >= 0 && W.ranges[op.range].n != op.n) {
        error("warpgroup %d of block %d: a wgmma of width %d on a range of accumulators of width %d in its chain", k,
              g_block, op.n, W.ranges[op.range].n);
        g_abort = true;
        return false;
    }
    if (op.range < 0) {
        for (const AccRange& a : W.ranges)   // a new range may not overlap the chain's others
            if (t0.acc < a.regs[0] + a.n / 2 && a.regs[0] < t0.acc + op.n / 2) {
                error("warpgroup %d of block %d: a wgmma of width %d on accumulators that overlap a range of width %d "
                      "of its chain", k, g_block, op.n, a.n);
                g_abort = true;
                return false;
            }
        AccRange a;
        a.n = op.n;
        for (int T = 0; T < 128; T++) a.regs[T] = g_thr[first + T].acc;
        if (W.late) {
            a.shadow.assign((size_t)128 * (op.n / 2), 0.0f);
            for (int T = 0; T < 128; T++) {
                memcpy(&a.shadow[(size_t)T * (op.n / 2)], a.regs[T], op.n / 2 * sizeof(float));
                for (int i = 0; i < op.n / 2; i++) a.regs[T][i] = NAN;
            }
        }
        W.ranges.push_back(a);
        op.range = (int)W.ranges.size() - 1;
    }
    for (int T = 0; T < 128; T++)
        if (g_thr[first + T].acc != W.ranges[op.range].regs[T]) {
            error("warpgroup %d of block %d: thread %d's accumulators for this wgmma are not those of the range its "
                  "warpgroup's thread 0 names", k, g_block, first + T);
            break;
        }
    if (W.late) W.open.push_back(op);
    else perform(op, W.ranges[op.range].regs, first / g_tpb);
    for (int T = 0; T < 128; T++) g_thr[first + T].state = kRunnable;
    return true;
}

// ---------------------------------------------------------------- step 2: mbarriers and the TMA
// An mbarrier (PTX ISA): its expected arrivals, the arrivals and bytes still awaited in the current phase, and the
// number of phases completed; a phase completes when no arrival and no byte is awaited. A wait for parity P returns
// once the current phase's parity differs from P. A TMA load is checked when issued and lands (its box written in
// shared memory, its bytes completed on its barrier) when issued (early: reverse order), once nothing else can go on
// (late: forward order), or either way at random (random order).
struct MBar { bool init = false; int count = 0, pending = 0; long tx = 0, phase = 0; };
static std::map<long, MBar> g_mbar;          // per (block << 32 | shared offset), for the running cluster
static long mkey(int r, long off) { return ((long)r << 32) | off; }
struct Tma { CUtensorMap map; int r; long dst, bar; int c0, c1; int from; };   // r: the block it lands in; bar:
                                                                           // its key; from: the block that issued it
static std::deque<Tma> g_tma;                // issued and not landed, oldest first
static int g_tmaMode;                        // 0: late (forward order), 1: early (reverse), 2: at random
static std::mt19937 g_tmaRng;
constexpr uint64_t kMapMagic = 0x6d61705f68307070ull;

static MBar* mbar_at(const void* bar, const char* what)
{
    const long off = (const char*)bar - (const char*)g_sh;
    if (off < 0 || off + 8 > g_shBytes || off % 8) {
        error("%s on an mbarrier at shared byte %ld, outside the launch or misaligned (%s)", what, off, who());
        return nullptr;
    }
    MBar& b = g_mbar[mkey(g_cur / g_tpb, off)];
    if (!b.init) {
        error("%s on an mbarrier at shared byte %ld not initialized (%s)", what, off, who());
        return nullptr;
    }
    return &b;
}

// A phase completes when no arrival and no byte is awaited. Bytes may come before the arrival that announces them (a
// multicast load issued by another block of a cluster): the count of bytes goes below zero while an arrival is
// awaited; below zero with no arrival awaited, more bytes have come than were announced.
static void mbar_complete(MBar& b, long key)
{
    if (b.tx < 0 && b.pending == 0) {
        error("mbarrier at shared byte %ld of block %ld: %ld bytes completed over those announced (%s)",
              key & 0xffffffffL, key >> 32, -b.tx, who());
        b.tx = 0;
    }
    if (b.pending == 0 && b.tx == 0) {
        b.phase++;
        b.pending = b.count;
    }
}

// The box of a TMA load lands in block op.r: elements of the matrix inside it, zeros outside, at the destination with
// the 128-byte swizzle (bits 4-6 of the shared address XOR bits 7-9); its bytes complete on its barrier.
static void tma_land(const Tma& op)
{
    if (g_blockLeft[op.r]) {
        error("a TMA load lands in block %d, whose threads have all left", g_blockLin[op.r]);
        return;
    }
    const CUtensorMap& m = op.map;
    const half* base = (const half*)m.base;
    for (uint32_t y = 0; y < m.boxRows; y++)
        for (uint32_t x = 0; x < m.boxCols; x++) {
            const long gc = (long)op.c0 + x, gr = (long)op.c1 + y;
            half v = (half)0.0f;
            if (gc >= 0 && gc < (long)m.cols && gr >= 0 && gr < (long)m.rows) v = base[gr * m.cols + gc];
            long a = op.dst + (long)(y * m.boxCols + x) * 2;
            a ^= ((a >> 7) & 7) << 4;
            memcpy((char*)g_shBlocks[op.r] + a, &v, 2);
            const size_t u = (size_t)op.r * kShArrayBytes + a;
            g_unfenced[u] = g_unfenced[u + 1] = 0;   // written by the async proxy
        }
    auto it = g_mbar.find(op.bar);
    if (it == g_mbar.end() || !it->second.init) {
        error("a TMA load completes on an mbarrier at shared byte %ld of block %ld not initialized", op.bar & 0xffffffffL,
              op.bar >> 32);
        return;
    }
    it->second.tx -= (long)m.boxCols * m.boxRows * 2;
    mbar_complete(it->second, op.bar);
}

// Lands the oldest TMA load still in flight (late mode). False if none.
static bool tma_land_one()
{
    if (g_tma.empty()) return false;
    const Tma op = g_tma.front();
    g_tma.pop_front();
    tma_land(op);
    return true;
}

// Threads waiting at an mbarrier whose phase has completed resume.
// The threads at the wait of a late chain (emu_wgmma_wait) resume two rounds after they got there: in the round
// between, the threads that the barriers then complete wake, whatever the order. True while any thread is at such a wait.
static bool wake_late_waits()
{
    bool any = false;
    for (int t = 0; t < g_threads; t++)
        if (g_thr[t].state == kAtLateWait) {
            if (g_round >= g_thr[t].lateRound) g_thr[t].state = kRunnable;
            any = true;
        }
    return any;
}

static bool wake_mbar_waiters()
{
    bool any = false;
    for (int t = 0; t < g_threads; t++) {
        Thread& th = g_thr[t];
        if (th.state != kAtMbar) continue;
        const MBar& b = g_mbar[th.mbar];
        if ((uint32_t)(b.phase & 1) != th.parity) {
            th.state = kRunnable;
            any = true;
        }
    }
    return any;
}

// ---------------------------------------------------------------- step 3: __shfl_sync
// When the 32 lanes of warp w wait at a __shfl_sync: each gets its source lane's value, and they resume.
static bool complete_shfl(int w)
{
    Thread* lane = &g_thr[32 * w];
    for (int l = 0; l < 32; l++)
        if (lane[l].state != kAtShfl) return false;
    for (int l = 0; l < 32; l++) {
        lane[l].shflOut = lane[lane[l].shflSrc & 31].shflV;
        lane[l].state = kRunnable;
    }
    return true;
}

// ---------------------------------------------------------------- step 6: stmatrix .trans
// When the 32 lanes of warp w wait at a stmatrix.x4: row r of matrix i (16 bytes) goes to the address lane 8 i + r
// gives, its elements from register i of lanes 4 r to 4 r + 3 (two halves each); with .trans (step 6), row r of the
// stored matrix is column r of the fragment: element c from half r % 2 of register i of lane 4 c + r / 2. Each row is
// a store of the lane that gave its address (for the async proxy: visible to it after that lane's fence.proxy.async).
// Its wavefronts: each matrix's 8 rows as one access of 8 lanes of 16 bytes (class STSM; conflict-free: 4 per
// instruction).
static bool complete_stsm(int w)
{
    Thread* lane = &g_thr[32 * w];
    for (int l = 0; l < 32; l++)
        if (lane[l].state != kAtStsm) return false;
    const int r = 32 * w / g_tpb;
    char* sh = (char*)g_shBlocks[r];
    long wave = 0;
    bool ok = true;
    for (int l = 1; l < 32; l++)
        if (lane[l].stsmTrans != lane[0].stsmTrans) {
            error("stmatrix: lanes of warp %d of block %d with and without .trans", w % (g_tpb / 32), g_blockLin[r]);
            ok = false;
            break;
        }
    for (int i = 0; i < 4 && ok; i++) {
        long off[8];
        for (int row = 0; row < 8; row++) {
            const int a = 8 * i + row;
            const long o = lane[a].stsmAddr;
            off[row] = o;
            if (o % 16 || o + 16 > g_shBytes) {
                error("stmatrix: lane %d of warp %d of block %d gives shared byte %ld, %s (the launch has %d bytes)",
                      a, w % (g_tpb / 32), g_blockLin[r], o, o % 16 ? "not 16-byte aligned" : "over its end",
                      g_shBytes);
                ok = false;
                break;
            }
            if (lane[0].stsmTrans)
                for (int c = 0; c < 8; c++) {
                    const uint16_t v = (uint16_t)(lane[4 * c + row / 2].stsmR[i] >> (16 * (row % 2)));
                    memcpy(sh + o + 2 * c, &v, 2);
                }
            else   // without .trans: no kernel of the rung takes it since stmatrix (5e) left the code
                for (int e = 0; e < 4; e++) memcpy(sh + o + 4 * e, &lane[4 * row + e].stsmR[i], 4);
            const long base = (long)r * kShArrayBytes + o;
            for (long b = base; b < base + 16; b++) g_unfenced[b] = 32 * w + a + 1;
            lane[a].unfenced.push_back({base, 16});
        }
        if (ok) wave += wavefronts(off, 8, 16);
    }
    if (ok) {
        Count& c = g_banks["STSM"];
        c.instr++;
        c.wavefronts += wave;
    }
    for (int l = 0; l < 32; l++) lane[l].state = kRunnable;
    return true;
}

// ---------------------------------------------------------------- step 3: named barriers
// bar.sync id, count, per block: the threads of a block that wait at barrier id are released once `count` of them are
// there (whole warps); their warps' shared accesses since their last barrier are accounted then. __syncthreads is
// barrier 0 for all the block's threads. Threads at one id with different counts: an error.
static bool release_named()
{
    bool any = false;
    for (int r = 0; r < g_cluster; r++)
        for (int id = 0; id < 16; id++) {
            int n = 0, count = -1;
            bool mixed = false;
            for (int t = r * g_tpb; t < (r + 1) * g_tpb; t++)
                if (g_thr[t].state == kAtNamed && g_thr[t].named == id) {
                    n++;
                    if (count >= 0 && count != g_thr[t].namedCount) mixed = true;
                    count = g_thr[t].namedCount;
                }
            if (!n) continue;
            if (mixed) {
                error("named barrier %d of block %d: its threads give different counts", id, g_blockLin[r]);
                g_abort = true;
                return false;
            }
            if (n < count) continue;
            if (n > count) {
                error("named barrier %d of block %d: %d threads for a count of %d", id, g_blockLin[r], n, count);
                g_abort = true;
                return false;
            }
            std::vector<bool> warps(g_threads / 32, false);
            for (int t = r * g_tpb; t < (r + 1) * g_tpb; t++)
                if (g_thr[t].state == kAtNamed && g_thr[t].named == id) {
                    warps[t / 32] = true;
                    g_thr[t].state = kRunnable;
                }
            count_shared_accesses(&warps);
            any = true;
        }
    return any;
}

// ---------------------------------------------------------------- step 4: the cluster barrier
// barrier.cluster.arrive and .wait: a thread's k-th wait returns once every thread of the cluster has arrived k times
// (a thread that has left counts as arrived).
static bool cluster_ready(int k)
{
    for (int t = 0; t < g_threads; t++)
        if (g_thr[t].state != kExited && g_thr[t].clArrived < k) return false;
    return true;
}
static bool wake_cluster_waiters()
{
    bool any = false;
    for (int t = 0; t < g_threads; t++)
        if (g_thr[t].state == kAtCluster && cluster_ready(g_thr[t].clWaited + 1)) {
            g_thr[t].clWaited++;
            g_thr[t].state = kRunnable;
            any = true;
        }
    return any;
}

// Threads waiting at a setmaxnreg.inc take their registers from their block's pool, in thread order, when it has them.
static bool wake_regs()
{
    bool any = false;
    for (int t = 0; t < g_threads; t++) {
        Thread& th = g_thr[t];
        if (th.state != kAtRegs) continue;
        long& pool = g_regPool[t / g_tpb];
        if (pool >= th.regsWanted - th.regs) {
            pool -= th.regsWanted - th.regs;
            th.regs = th.regsWanted;
            th.state = kRunnable;
            any = true;
        }
    }
    return any;
}

enum Order { kForward, kReverse, kRandom };
static const char* const kOrderName[] = {"forward", "reverse", "random"};

static void start_thread(ucontext_t* ctx, char* stack)   // a context that starts thread_main on its own stack
{
    getcontext(ctx);
    ctx->uc_stack.ss_sp = stack;
    ctx->uc_stack.ss_size = kStackBytes;
    ctx->uc_link = &g_sched;
    makecontext(ctx, thread_main, 0);
}

// Runs the cluster's blocks together (one block, but for clusters). Forward and reverse: warp by warp (0 up, or the
// last one down), each lane in order (or reversed) until it blocks (a barrier, a wgmma its warpgroup has not all
// reached, an mbarrier's phase not complete, the cluster barrier, its end), then the warpgroup's wgmma if all its
// threads are there, over and over while something moves. Random: a thread drawn at random among those that can run.
// When nothing moves, a TMA load in flight lands, if any. False if the blocks cannot complete: a DEADLOCK when threads
// wait at barriers that nothing can complete.
static bool run_cluster(Order order, std::mt19937& rng)
{
    for (int t = 0; t < g_threads; t++) {
        start_thread(&g_thr[t].ctx, g_stacks + t * (kPage + kStackBytes) + kPage);
        g_thr[t].state = kRunnable;
        g_thr[t].log.clear();
        g_thr[t].commits = 0;
        g_thr[t].fenced = false;
        g_thr[t].unfenced.clear();
        g_thr[t].elects = 0;
        g_thr[t].maxnreg = 0;
        g_thr[t].maxnregs = 0;
        g_thr[t].clArrived = g_thr[t].clWaited = 0;
        g_thr[t].bulkOpen.clear();
        g_thr[t].bulkGroups.clear();
        g_thr[t].regs = g_launchRegs;
    }
    for (int r = 0; r < g_cluster; r++) g_regPool[r] = 0;
    for (WarpGroup& W : g_wg) W = WarpGroup();
    g_unfenced.assign((size_t)g_cluster * kShArrayBytes, 0);
    g_mbar.clear();
    g_tma.clear();
    for (int r = 0; r < g_cluster; r++) g_blockLeft[r] = false;
    const int warps = g_threads / 32;
    std::vector<int> ready;
    auto update_left = [&] {   // a block whose threads have all left; it may not leave a TMA load it issued in flight
        for (int r = 0; r < g_cluster; r++) {
            bool all = true;
            for (int t = r * g_tpb; t < (r + 1) * g_tpb && all; t++) all = g_thr[t].state == kExited;
            if (all && !g_blockLeft[r])
                for (const Tma& op : g_tma)
                    if (op.from == r) {
                        error("block %d leaves while a TMA load it issued has not landed in block %d (a block may "
                              "leave only once its loads into the cluster have landed)", g_blockLin[r],
                              g_blockLin[op.r]);
                        break;
                    }
            g_blockLeft[r] = all;
        }
    };
    for (bool progress = true; progress && !g_abort;) {
        g_round++;
        progress = wake_mbar_waiters() | release_named() | wake_cluster_waiters() | wake_regs() | wake_late_waits();
        for (int w = 0; w < warps; w++) progress |= complete_shfl(w) | complete_stsm(w);
        if (order == kRandom) {
            for (;;) {
                ready.clear();
                for (int t = 0; t < g_threads; t++)
                    if (g_thr[t].state == kRunnable) ready.push_back(t);
                if (ready.empty() || g_abort) break;
                const int t = ready[rng() % ready.size()];
                resume(t);
                progress = true;
                if (g_thr[t].state == kAtWgmma) complete_wgmma(wgIndex(t));
                if (g_thr[t].state == kAtShfl) complete_shfl(t / 32);
                if (g_thr[t].state == kAtStsm) complete_stsm(t / 32);
                if (g_thr[t].state == kExited) update_left();
                wake_mbar_waiters();
                release_named();
                wake_cluster_waiters();
                wake_regs();
            }
        } else {
            for (int k = 0; k < warps && !g_abort; k++) {
                const int w = order == kForward ? k : warps - 1 - k;
                for (int j = 0; j < 32; j++) {
                    const int t = 32 * w + (order == kForward ? j : 31 - j);
                    if (g_thr[t].state == kRunnable) {
                        resume(t);
                        progress = true;
                    }
                }
                if (complete_shfl(w)) progress = true;
                if (complete_stsm(w)) progress = true;
                if (complete_wgmma(wgIndex(32 * w))) progress = true;
            }
        }
        update_left();
        if (!progress && tma_land_one()) progress = true;   // nothing else moves: a TMA load lands
    }
    if (g_abort) return false;
    int atWgmma = 0, atMbar = 0, atNamed = 0, atCluster = 0, exited = 0;
    for (int t = 0; t < g_threads; t++) {
        atWgmma += g_thr[t].state == kAtWgmma;
        atMbar += g_thr[t].state == kAtMbar;
        atNamed += g_thr[t].state == kAtNamed;
        atCluster += g_thr[t].state == kAtCluster;
        exited += g_thr[t].state == kExited;
    }
    if (exited == g_threads) {
        count_shared_accesses();   // the accesses after the last barrier
        if (!g_tma.empty()) error("block %d ends with %zu TMA loads in flight", g_blockLin[0], g_tma.size());
        for (int t = 0; t < g_threads; t++)   // setmaxnreg: by all the warps of a warpgroup, or none
            if (g_thr[t].maxnregs != g_thr[wgFirst(t)].maxnregs) {
                error("block %d: setmaxnreg run %d times by thread %d, %d by thread %d of the same warpgroup (all its "
                      "warps run it)", g_blockLin[t / g_tpb], g_thr[t].maxnregs, t % g_tpb,
                      g_thr[wgFirst(t)].maxnregs, wgFirst(t) % g_tpb);
                break;
            }
        return true;
    }
    for (int t = 0; t < g_threads; t++)
        if (g_thr[t].state == kAtShfl || g_thr[t].state == kAtStsm) {
            error("block %d: thread %d waits at a %s that the rest of its warp does not reach", g_blockLin[t / g_tpb],
                  t % g_tpb, g_thr[t].state == kAtShfl ? "__shfl_sync" : "stmatrix");
            return false;
        }
    for (int t = 0; t < g_threads; t++)   // the block barrier while some of the block's threads have left
        if (g_thr[t].state == kAtNamed && g_thr[t].named == 0) {
            const int r = t / g_tpb;
            int at = 0, left = 0;
            for (int u = r * g_tpb; u < (r + 1) * g_tpb; u++) {
                at += g_thr[u].state == kAtNamed && g_thr[u].named == 0;
                left += g_thr[u].state == kExited;
            }
            if (left) {
                error("block %d: a barrier reached by %d threads while %d have exited", g_blockLin[r], at, left);
                return false;
            }
            break;
        }
    for (int t = 0; t < g_threads; t++)
        if (g_thr[t].state == kAtRegs) {
            error("DEADLOCK in block %d: thread %d waits at setmaxnreg.inc %d with %d registers; its block's pool has "
                  "%ld per thread, and nothing gives more back", g_blockLin[t / g_tpb], t % g_tpb, g_thr[t].regsWanted,
                  g_thr[t].regs, g_regPool[t / g_tpb]);
            return false;
        }
    if (atNamed) {
        int t0 = 0;
        while (g_thr[t0].state != kAtNamed) t0++;
        error("DEADLOCK in block %d: %d threads wait at named barriers that do not fill (thread %d at barrier %d for "
              "%d threads); %d at mbarriers, %d at the cluster barrier, %d exited", g_blockLin[t0 / g_tpb], atNamed,
              t0 % g_tpb, g_thr[t0].named, g_thr[t0].namedCount, atMbar, atCluster, exited);
        return false;
    }
    if (atMbar) {
        int t0 = 0;
        while (g_thr[t0].state != kAtMbar) t0++;
        const MBar& b = g_mbar[g_thr[t0].mbar];
        error("DEADLOCK in block %d: %d threads wait at mbarriers that nothing can complete (thread %d at shared byte "
              "%ld, parity %u; its phase %ld, %d arrivals and %ld bytes awaited); %d at a barrier, %d at the cluster "
              "barrier, %d at a wgmma, %d exited", g_blockLin[g_thr[t0].mbar >> 32], atMbar, t0 % g_tpb,
              g_thr[t0].mbar & 0xffffffffL, g_thr[t0].parity, b.phase, b.pending, b.tx, atNamed, atCluster, atWgmma,
              exited);
        return false;
    }
    if (atCluster) {
        int t0 = 0;
        while (g_thr[t0].state != kAtCluster) t0++;
        error("DEADLOCK in block %d: %d threads wait at the cluster barrier, which threads of the cluster never reach "
              "(thread %d, its wait %d)", g_blockLin[t0 / g_tpb], atCluster, t0 % g_tpb, g_thr[t0].clWaited + 1);
        return false;
    }
    if (atWgmma) {
        error("block %d: %d threads wait at a wgmma that the rest of their warpgroup does not reach (%d exited)",
              g_blockLin[0], atWgmma, exited);
        return false;
    }
    error("block %d: the run stops with threads that can neither go on nor wait (internal)", g_blockLin[0]);
    return false;
}

// ---------------------------------------------------------------- the checks of every access
static long floordiv(long a, long b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

// One access of `bytes` at p (align: the alignment it needs). In the launch's shared memory: inside it and aligned,
// and logged for the bank model if `log`. In A, B or C or one of their guard regions: inside the array and aligned, no
// store to A or B; a store to C counts its writes. Anywhere else (a variable of the thread): not checked. Returns
// whether the access may be performed.
static bool check(const void* ptr, int bytes, int align, bool store, bool log)
{
    const char* p = (const char*)ptr;
    const char* sh = (const char*)g_sh;
    const char* op = store ? "store" : "load";
    if (p >= sh && p < sh + kShArrayBytes) {
        const long off = p - sh;
        if (off + bytes > g_shBytes || off % align) {
            error("%s of %d bytes at byte %ld of shared memory, %s (the launch has %d bytes; %s)", op, bytes, off,
                  off + bytes > g_shBytes ? "over its end" : "misaligned", g_shBytes, who());
            return false;
        }
        if (log && g_cur >= 0) g_thr[g_cur].log.push_back({store, bytes, off});
        if (store && g_cur >= 0) {   // not yet visible to the async proxy (step 1's wgmma), until the thread's fence
            const long base = (long)(g_cur / g_tpb) * kShArrayBytes + off;   // this block's bytes
            for (long b = base; b < base + bytes; b++) g_unfenced[b] = g_cur + 1;
            g_thr[g_cur].unfenced.push_back({base, bytes});
        }
        return true;
    }
    for (Array* a : {&g_A, &g_B, &g_C}) {
        if (!a->p || p < a->lo() - kGuardBytes || p >= a->hi() + kGuardBytes) continue;
        const long e = floordiv(p - a->lo(), 2), row = floordiv(e, a->cols), col = e - row * a->cols;
        if (p < a->lo() || p + bytes > a->hi()) {
            error("%s of %d bytes at %s[%ld][%ld], outside %s (%d x %d; %s)", op, bytes, a->name, row, col, a->name,
                  a->rows, a->cols, who());
            return false;
        }
        if ((uintptr_t)p % align) {
            error("%s of %d bytes at %s[%ld][%ld], not aligned to %d bytes (%s)", op, bytes, a->name, row, col, align,
                  who());
            return false;
        }
        if (store && !a->writable) {
            error("store to %s[%ld][%ld], an input (%s)", a->name, row, col, who());
            return false;
        }
        if (store && a == &g_C)
            for (long i = e; i < e + bytes / 2; i++) {
                if (g_cWrites[i] == 0) {
                    g_cFirst[i] = g_block;
                } else if (g_cAgain < 0) {
                    g_cAgain = i;
                    g_cAgainBlocks[0] = g_cFirst[i];
                    g_cAgainBlocks[1] = g_block;
                }
                if (g_cWrites[i] < 255) g_cWrites[i]++;
            }
        return true;
    }
    return true;
}

void emu_copy(void* dst, const void* src, int bytes)
{
    const bool load = check(src, bytes, bytes, false, true);
    const bool store = check(dst, bytes, bytes, true, true);
    if (load && store) memcpy(dst, src, bytes);
}

// ---------------------------------------------------------------- step1.cuh's entry points
uint32_t step1::emu_smem_addr(const void* p)
{
    const long off = (const char*)p - (const char*)g_sh;
    if (off < 0 || off >= kShArrayBytes) error("cvta.to.shared of an address outside shared memory (%s)", who());
    return (uint32_t)off;
}

static void wgmma_at(float* d, int n, uint64_t da, uint64_t db, int scaleD, int tA, int tB)
{
    Thread& th = g_thr[g_cur];
    th.acc = d;
    th.n = n;
    th.da = da;
    th.db = db;
    th.scaleD = scaleD;
    th.tA = tA;
    th.tB = tB;
    th.state = kAtWgmma;
    swapcontext(&th.ctx, &g_sched);
}

void step1::emu_wgmma(float* d, int n, uint64_t da, uint64_t db, int scaleD) { wgmma_at(d, n, da, db, scaleD, 0, 1); }

void step1::emu_wgmma_fence() { g_thr[g_cur].fenced = true; }

// The first thread of the warpgroup to commit makes the group; the others find it made (the warpgroup's MMA are issued
// once all its threads have reached them, so the group is complete by then).
void step1::emu_wgmma_commit()
{
    Thread& th = g_thr[g_cur];
    WarpGroup& W = g_wg[wgIndex(g_cur)];
    if (++th.commits > W.created) {
        if (W.late) W.groups.push_back(W.open);
        W.open.clear();
        W.created++;
    }
}

// The first thread of the warpgroup to wait retires all its groups but the `pending` latest (a late chain performs them
// now); the chain ends when no MMA is left in flight. As on the GPU, where the other warps run on while the MMA are in
// flight, a wait with a group of a late chain still to retire first lets the other threads run for two rounds of the
// scheduler, and one more per warpgroup of the block before its own, so that the warpgroups of a block retire their
// groups at different times; the threads that a barrier completes meanwhile run (wake_late_waits): a stage released
// before the wait that retires its MMA, or loaded again before another warpgroup has released it, can then be
// overwritten before the MMA read it, whichever warps load and compute.
void step1::emu_wgmma_wait(int pending)
{
    Thread& th = g_thr[g_cur];
    WarpGroup& W = g_wg[wgIndex(g_cur)];
    if (W.late && th.commits - W.retired > pending) {
        th.state = kAtLateWait;
        th.lateRound = g_round + 2 + (g_cur % g_tpb) / 128;
        swapcontext(&th.ctx, &g_sched);
    }
    while (th.commits - W.retired > pending) {
        if (W.late && !W.groups.empty()) {
            std::vector<float*> acc(128);
            for (const WgOp& op : W.groups.front()) {
                AccRange& a = W.ranges[op.range];
                for (int T = 0; T < 128; T++) acc[T] = &a.shadow[(size_t)T * (a.n / 2)];
                perform(op, acc.data(), g_cur / g_tpb);
            }
            W.groups.pop_front();
        }
        W.retired++;
    }
    if (W.inFlight && W.retired == W.created && W.open.empty()) end_chain(W, wgFirst(g_cur));
}

void step1::emu_fence_proxy_async()
{
    Thread& th = g_thr[g_cur];
    for (const auto& s : th.unfenced)
        for (long b = s.first; b < s.first + s.second; b++)
            if (g_unfenced[b] == g_cur + 1) g_unfenced[b] = 0;
    th.unfenced.clear();
}

// ---------------------------------------------------------------- step2.cuh's entry points
void step2::emu_encode_map(CUtensorMap* map, const half* base, uint64_t cols, uint64_t rows, uint32_t boxCols,
                           uint32_t boxRows)
{
    if (boxCols * 2 > 128 || boxRows > 256 || boxCols == 0 || boxRows == 0)
        error("a tensor map with a box of %u x %u halves: the 128-byte swizzle takes at most 128 bytes of the fast "
              "dimension, a box at most 256 rows", boxCols, boxRows);
    *map = {base, cols, rows, boxCols, boxRows, kMapMagic};
}

void step2::emu_mbar_init(uint64_t* bar, int count)
{
    const long off = (const char*)bar - (const char*)g_sh;
    if (off < 0 || off + 8 > g_shBytes || off % 8) {
        error("mbarrier.init at shared byte %ld, outside the launch or misaligned (%s)", off, who());
        return;
    }
    const int r = g_cur / g_tpb;
    MBar& b = g_mbar[mkey(r, off)];
    for (long o = off - 7; o < off + 8; o++)   // another mbarrier's 8 bytes, or this one, already valid
        if (g_mbar.count(mkey(r, o)) && g_mbar[mkey(r, o)].init)
            error("mbarrier.init at shared byte %ld over the valid mbarrier at byte %ld (an mbarrier is initialized "
                  "once, or invalidated first; %s)", off, o, who());
    b = MBar();
    b.init = true;
    b.count = b.pending = count;
}

void step2::emu_mbar_arrive(uint64_t* bar, uint32_t bytes)
{
    MBar* b = mbar_at(bar, bytes ? "arrive.expect_tx" : "arrive");
    if (!b) return;
    b->tx += bytes;
    if (--b->pending < 0) {
        error("mbarrier at shared byte %ld: more arrivals than its count %d in a phase (%s)",
              (long)((const char*)bar - (const char*)g_sh), b->count, who());
        b->pending = 0;
    }
    mbar_complete(*b, (const char*)bar - (const char*)g_sh);
}

void step2::emu_mbar_wait(uint64_t* bar, uint32_t parity)
{
    MBar* b = mbar_at(bar, "try_wait.parity");
    if (!b) return;
    if ((uint32_t)(b->phase & 1) != parity) return;   // the phase of that parity has completed
    Thread& th = g_thr[g_cur];
    th.mbar = mkey(g_cur / g_tpb, (const char*)bar - (const char*)g_sh);
    th.parity = parity;
    th.state = kAtMbar;
    swapcontext(&th.ctx, &g_sched);
}

void step2::emu_tma_load_2d(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar)
{
    if (map->magic != kMapMagic) {
        error("a TMA load through a tensor map not encoded (%s)", who());
        return;
    }
    const long off = (const char*)dst - (const char*)g_sh, bytes = (long)map->boxCols * map->boxRows * 2;
    if (off < 0 || off + bytes > g_shBytes || off % 128) {
        error("a TMA box of %ld bytes to shared byte %ld: outside the launch or not 128-byte aligned (%s)", bytes, off,
              who());
        return;
    }
    if (!mbar_at(bar, "a TMA load's completion")) return;
    const int r = g_cur / g_tpb;
    const Tma op = {*map, r, off, mkey(r, (const char*)bar - (const char*)g_sh), c0, c1, r};
    const bool early = g_tmaMode == 1 || (g_tmaMode == 2 && (g_tmaRng() & 1));
    if (early) tma_land(op);
    else g_tma.push_back(op);
}

// ---------------------------------------------------------------- step3.cuh's entry points
int __shfl_sync(unsigned mask, int v, int srcLane)
{
    if (mask != 0xffffffffu) error("__shfl_sync with a mask of %08x: only the whole warp is emulated (%s)", mask, who());
    Thread& th = g_thr[g_cur];
    th.shflV = v;
    th.shflSrc = srcLane;
    th.state = kAtShfl;
    swapcontext(&th.ctx, &g_sched);
    return th.shflOut;
}

// elect.sync: the k-th call of each lane of a warp names the same lane: lane 0 in the forward order, 31 in the reverse,
// one drawn from the block, the warp and k in the random order.
bool step3::emu_elect()
{
    Thread& th = g_thr[g_cur];
    const int k = th.elects++, w = g_cur / 32;
    int lane = g_tmaMode == 0 ? 0 : g_tmaMode == 1 ? 31 : 0;
    if (g_tmaMode == 2) {
        uint32_t x = (uint32_t)(g_block * 7919 + w * 104729 + k * 1299709) ^ 0x9e3779b9u;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
        lane = (int)(x % 32);
    }
    return g_cur % 32 == lane;
}

void step3::emu_bar_named(int id, int count)
{
    if (id < 1 || id > 15 || count <= 0 || count % 32 || count > g_threads)
        error("bar.sync %d, %d: an id of 1 to 15 and a count of whole warps (%s)", id, count, who());
    Thread& th = g_thr[g_cur];
    th.named = id;
    th.namedCount = count;
    th.state = kAtNamed;
    swapcontext(&th.ctx, &g_sched);
}

// ---------------------------------------------------------------- step4.cuh's entry points
uint8_t* step4::emu_shared_base() { return (uint8_t*)g_shBlocks[g_cur / g_tpb]; }

uint32_t step4::emu_cluster_rank() { return (uint32_t)(g_cur / g_tpb); }

void step4::emu_cluster_arrive() { g_thr[g_cur].clArrived++; }

void step4::emu_cluster_wait()
{
    Thread& th = g_thr[g_cur];
    if (th.clArrived <= th.clWaited)
        error("barrier.cluster.wait with no arrival of the thread before it (%s)", who());
    if (cluster_ready(th.clWaited + 1)) {
        th.clWaited++;
        return;
    }
    th.state = kAtCluster;
    swapcontext(&th.ctx, &g_sched);
}

// ---------------------------------------------------------------- step5.cuh's entry points
// A TMA store (cp.async.bulk.tensor.2d, global from shared): checked when issued (its box inside the launch, 128-byte
// aligned, and each of its bytes stored by a thread made visible to the async proxy by that thread's fence before the
// store is issued); it reads its box (the 128-byte swizzle) and writes the matrix (elements outside it left out) when
// issued (forward order), as late as the thread's wait_group allows (reverse), or either way at random (random order).
static void store_perform(Store& op)
{
    if (op.done) return;
    op.done = true;
    const CUtensorMap& m = op.map;
    half* base = (half*)m.base;
    const char* sh = (const char*)g_shBlocks[op.r];
    for (uint32_t y = 0; y < m.boxRows; y++)
        for (uint32_t x = 0; x < m.boxCols; x++) {
            const long gc = (long)op.c0 + x, gr = (long)op.c1 + y;
            if (gc < 0 || gc >= (long)m.cols || gr < 0 || gr >= (long)m.rows) continue;
            long a = op.src + (long)(y * m.boxCols + x) * 2;
            a ^= ((a >> 7) & 7) << 4;
            half v;
            memcpy(&v, sh + a, 2);
            half* d = base + gr * (long)m.cols + gc;
            if (check(d, 2, 2, true, false)) memcpy(d, &v, 2);
        }
}

void step5::emu_tma_store_2d(const CUtensorMap* map, int c0, int c1, uint32_t src)
{
    if (map->magic != kMapMagic) {
        error("a TMA store through a tensor map not encoded (%s)", who());
        return;
    }
    const long bytes = (long)map->boxCols * map->boxRows * 2;
    if (src % 128 || src + bytes > g_shBytes) {
        error("a TMA store of a box of %ld bytes from shared byte %u: outside the launch or not 128-byte aligned (%s)",
              bytes, src, who());
        return;
    }
    const int r = g_cur / g_tpb;
    for (long b = src; b < src + bytes; b++) {
        int& u = g_unfenced[(size_t)r * kShArrayBytes + b];
        if (u) {
            error("a TMA store reads shared byte %ld, stored by thread %d without a fence.proxy.async before the store "
                  "was issued (%s)", b, (u - 1) % g_tpb, who());
            u = 0;   // reported once
            break;
        }
    }
    Thread& th = g_thr[g_cur];
    th.bulkOpen.push_back({*map, r, (long)src, c0, c1, false});
    const bool early = g_tmaMode == 0 || (g_tmaMode == 2 && (g_tmaRng() & 1));
    if (early) store_perform(th.bulkOpen.back());
}

void step5::emu_bulk_commit()
{
    Thread& th = g_thr[g_cur];
    th.bulkGroups.push_back(th.bulkOpen);
    th.bulkOpen.clear();
}

// wait_group(.read) N: the thread's committed groups but the N latest are performed (reads and writes at once here).
void step5::emu_bulk_wait(int pending, bool)
{
    Thread& th = g_thr[g_cur];
    while ((int)th.bulkGroups.size() > pending) {
        for (Store& op : th.bulkGroups.front()) store_perform(op);
        th.bulkGroups.pop_front();
    }
}

static void stmatrix_at(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, bool trans)
{
    Thread& th = g_thr[g_cur];
    th.stsmTrans = trans;
    th.stsmAddr = addr;
    th.stsmR[0] = r0;
    th.stsmR[1] = r1;
    th.stsmR[2] = r2;
    th.stsmR[3] = r3;
    th.state = kAtStsm;
    swapcontext(&th.ctx, &g_sched);
}

// ---------------------------------------------------------------- step6.cuh's entry points
void step6::emu_wgmma_t(float* d, int n, uint64_t da, uint64_t db, int scaleD, int tnspA, int tnspB)
{
    wgmma_at(d, n, da, db, scaleD, tnspA, tnspB);
}
void step6::emu_stmatrix_x4_trans(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3)
{
    stmatrix_at(addr, r0, r1, r2, r3, true);
}

// ---------------------------------------------------------------- step7.cuh's entry points
// mapa: an address in the cluster's shared window, block `rank`'s offset `addr`, coded as (rank + 1) << 24 | addr.
uint32_t step7::emu_mapa(uint32_t addr, uint32_t rank)
{
    if (rank >= (uint32_t)g_cluster || addr >= (1u << 24))
        error("mapa to rank %u of a cluster of %d, or of an address outside shared memory (%s)", rank, g_cluster,
              who());
    return ((rank + 1) << 24) | addr;
}

// An access to another block of the cluster: after the thread's first wait at the cluster barrier (the barriers it
// reaches were initialized before it), and while that block has not left.
static bool remote_ok(int rank, const char* what)
{
    if (rank < 0 || rank >= g_cluster) {
        error("%s: rank %d outside the cluster of %d (%s)", what, rank, g_cluster, who());
        return false;
    }
    if (rank != g_cur / g_tpb && g_thr[g_cur].clWaited == 0) {
        error("%s to block %d before any wait at the cluster barrier (its barriers may not be initialized; %s)", what,
              g_blockLin[rank], who());
        return false;
    }
    if (g_blockLeft[rank]) {
        error("%s to block %d, whose threads have all left (%s)", what, g_blockLin[rank], who());
        return false;
    }
    return true;
}

void step7::emu_mbar_arrive_remote(uint32_t addr)
{
    const int rank = (int)(addr >> 24) - 1;
    const long off = addr & 0xffffff;
    if (rank < 0) {
        error("an arrival through a shared::cluster address not made by mapa (%s)", who());
        return;
    }
    if (!remote_ok(rank, "mbarrier.arrive.shared::cluster")) return;
    auto it = g_mbar.find(mkey(rank, off));
    if (it == g_mbar.end() || !it->second.init) {
        error("an arrival on the mbarrier at shared byte %ld of block %d, not initialized (%s)", off, g_blockLin[rank],
              who());
        return;
    }
    MBar& b = it->second;
    if (--b.pending < 0) {
        error("mbarrier at shared byte %ld of block %d: more arrivals than its count %d in a phase (%s)", off,
              g_blockLin[rank], b.count, who());
        b.pending = 0;
    }
    mbar_complete(b, mkey(rank, off));
}

// The multicast TMA: the box lands at the same offset in each block of the mask, its bytes on each one's barrier.
void step7::emu_tma_load_2d_multicast(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar, uint16_t mask)
{
    if (map->magic != kMapMagic) {
        error("a TMA load through a tensor map not encoded (%s)", who());
        return;
    }
    const long off = (const char*)dst - (const char*)g_sh, bytes = (long)map->boxCols * map->boxRows * 2;
    const long barOff = (const char*)bar - (const char*)g_sh;
    if (off < 0 || off + bytes > g_shBytes || off % 128) {
        error("a multicast TMA box of %ld bytes to shared byte %ld: outside the launch or not 128-byte aligned (%s)",
              bytes, off, who());
        return;
    }
    if (mask == 0 || mask >> g_cluster)
        error("a multicast TMA with the mask %x in a cluster of %d (%s)", mask, g_cluster, who());
    for (int rank = 0; rank < g_cluster; rank++) {
        if (!(mask >> rank & 1) || !remote_ok(rank, "a multicast TMA load")) continue;
        auto it = g_mbar.find(mkey(rank, barOff));
        if (it == g_mbar.end() || !it->second.init) {
            error("a multicast TMA load completes on the mbarrier at shared byte %ld of block %d, not initialized (%s)",
                  barOff, g_blockLin[rank], who());
            continue;
        }
        const Tma op = {*map, rank, off, mkey(rank, barOff), c0, c1, g_cur / g_tpb};
        const bool early = g_tmaMode == 1 || (g_tmaMode == 2 && (g_tmaRng() & 1));
        if (early) tma_land(op);
        else g_tma.push_back(op);
    }
}

// setmaxnreg: between 24 and 256, a multiple of 8, and the same in the whole warpgroup.
// setmaxnreg.dec gives the thread's registers above `regs` back to its block's pool; .inc takes what it lacks from the
// pool, and waits (state kAtRegs) until the pool has it: a wait that nothing can satisfy is a DEADLOCK.
void step7::emu_setmaxnreg(int regs, bool inc)
{
    if (regs < 24 || regs > 256 || regs % 8)
        error("setmaxnreg.%s %d: 24 to 256 registers, a multiple of 8 (%s)", inc ? "inc" : "dec", regs, who());
    Thread& th = g_thr[g_cur];
    const int r = g_cur / g_tpb;
    if (!inc && regs > th.regs) error("setmaxnreg.dec %d with %d registers (%s)", regs, th.regs, who());
    if (inc && regs < th.regs) error("setmaxnreg.inc %d with %d registers (%s)", regs, th.regs, who());
    if (!inc && regs <= th.regs) {
        g_regPool[r] += th.regs - regs;
        th.regs = regs;
    } else if (inc && regs >= th.regs) {
        if (g_regPool[r] >= regs - th.regs) {
            g_regPool[r] -= regs - th.regs;
            th.regs = regs;
        } else {
            th.regsWanted = regs;
            th.state = kAtRegs;
            swapcontext(&th.ctx, &g_sched);   // resumed by wake_regs with its registers
        }
    }
    g_thr[g_cur].maxnreg = regs;
    g_thr[g_cur].maxnregs++;
    const int first = wgFirst(g_cur);
    for (int t = first; t < first + 128 && t < g_threads; t++)
        if (g_thr[t].maxnreg && g_thr[t].maxnreg != regs) {
            error("setmaxnreg %d in thread %d, %d in thread %d of the same warpgroup", regs, g_cur, g_thr[t].maxnreg,
                  t);
            break;
        }
}


// A thread that ends with TMA stores not waited for (their source, its block's shared memory, may still be read): an
// error; they are performed then.
static void bulk_at_exit()
{
    Thread& th = g_thr[g_cur];
    size_t n = th.bulkOpen.size();
    for (const auto& g : th.bulkGroups) n += g.size();
    if (!n) return;
    error("thread %d of block %d ends with %zu TMA stores not waited for (%zu not committed; a wait_group 0 after the "
          "last commit)", g_cur % g_tpb, g_block, n, th.bulkOpen.size());
    for (Store& op : th.bulkOpen) store_perform(op);
    for (auto& g : th.bulkGroups)
        for (Store& op : g) store_perform(op);
    th.bulkOpen.clear();
    th.bulkGroups.clear();
}

// A warpgroup that ends with MMA in flight (never waited for): an error, once per warpgroup.
static void wgmma_at_exit()
{
    WarpGroup& W = g_wg[wgIndex(g_cur)];
    if (g_cur == wgFirst(g_cur) && W.inFlight)
        error("warpgroup %d of block %d ends with wgmma in flight (no wait_group 0 after its last commit)", wgIndex(g_cur),
              g_block);
}

bool emu_tile(const void* p, unsigned ldm, int elemBytes, bool store)
{
    if ((uintptr_t)p % 32 || ldm * elemBytes % 16) {
        error("wmma %s at %p with a pitch of %u elements: wmma needs a 32-byte aligned pointer and a pitch that is a "
              "multiple of 16 bytes (%s)", store ? "store" : "load", p, ldm, who());
        return false;
    }
    for (int r = 0; r < 16; r++)
        if (!check((const char*)p + (size_t)r * ldm * elemBytes, 16 * elemBytes, 1, store, false)) return false;
    return true;
}

void emu_wmma_error(const char* what) { error("wmma: %s is not emulated (%s)", what, who()); }

// A memory fault (an access the checks above do not see, through a plain pointer): where it fell, then exit. Or an
// integer division by zero, which the CPU traps where the GPU gives an undefined value: where it happened, then exit.
static void on_fault(int sig, siginfo_t* si, void*)
{
    if (sig == SIGFPE) {
        char msg[600];
        const int n = snprintf(msg, sizeof msg, "EMU ERROR: an integer division by zero (%s; on the GPU, an undefined "
                               "value)\n%s  FAILED  arithmetic fault\nFAILED: stopped by an arithmetic fault\n", who(),
                               g_label);
        if (write(1, msg, std::min(n, (int)sizeof msg - 1)) < 0) _exit(2);
        _exit(1);
    }
    const char* a = (const char*)si->si_addr;
    char where[200];
    bool overflow = false;
    snprintf(where, sizeof where, "outside the arrays the emulator knows");
    for (const Array* r : {&g_A, &g_B, &g_C}) {
        if (!r->p) continue;
        if (a >= r->lo() - kGuardBytes && a < r->lo())
            snprintf(where, sizeof where, "in the guard region, %ld bytes before %s", (long)(r->lo() - a), r->name);
        else if (a >= r->hi() && a < r->hi() + kGuardBytes)
            snprintf(where, sizeof where, "in the guard region, %ld bytes after the end of %s", (long)(a - r->hi()),
                     r->name);
        else if (a >= r->lo() && a < r->hi())
            snprintf(where, sizeof where, "in %s, which the kernels may not write", r->name);
    }
    for (int t = 0; t < kMaxThreads; t++) {
        const char* guard = g_stacks + t * (kPage + kStackBytes);
        if (a >= guard && a < guard + kPage) {
            snprintf(where, sizeof where, "outside the stack of thread %d's coroutine (raise kStackBytes)", t);
            overflow = true;
        }
    }
    char msg[800];
    const int n = snprintf(msg, sizeof msg, "EMU ERROR: %s at %p, %s (block %d, thread %d)\n%s  FAILED  memory fault\n"
                           "FAILED: stopped by a memory fault\n", sig == SIGBUS ? "bus error" : "memory fault",
                           si->si_addr, where, g_block, g_cur, g_label);
    if (write(1, msg, std::min(n, (int)sizeof msg - 1)) < 0) _exit(2);
    _exit(overflow ? 2 : 1);
}

// ---------------------------------------------------------------- the kernels
enum { kStep0 = 1, kStep1 = 2, kStep2 = 4, kStep3 = 8, kStep4 = 16, kStep5 = 32, kStep6 = 64, kStep7 = 128 };
constexpr unsigned kAll = kStep0 | kStep1 | kStep2 | kStep3 | kStep4 | kStep5 | kStep6 | kStep7;

// Step 2's kernels take tensor maps: the maps step2.cu builds (step2::map_a, map_b), then the kernel.
template <int kStages>
static void step2_entry(const half* A, const half* B, half* C, int M, int N, int K)
{
    const CUtensorMap a = step2::map_a(A, M, K), b = step2::map_b(B, K, N);
    step2::gemm<kStages>(a, b, C, M, N, K);
}
// Step 2's v3 and v4: without 2d (wait_group 0).
template <int kStages>
static void step2_entry_w0(const half* A, const half* B, half* C, int M, int N, int K)
{
    const CUtensorMap a = step2::map_a(A, M, K), b = step2::map_b(B, K, N);
    step2::gemm<kStages, false>(a, b, C, M, N, K);
}
// Step 4's kernel as step4.cu launches it (step 2's kernel at 4 stages, in clusters of two): the clusters along M with
// two rows of tiles or more, along N with one; each block loads its own A and B.
static void step4_entry(const half* A, const half* B, half* C, int M, int N, int K)
{
    const CUtensorMap a = step2::map_a(A, M, K), b = step2::map_b(B, K, N);
    if (M > 128) step4::gemm<true>(a, b, C, M, N, K);
    else step4::gemm<false>(a, b, C, M, N, K);
}
// Step 5's tile, as step5.cu, step6.cu and step7.cu launch it (the clusters along M or N as step 4's, each block its
// own A and B): kGrouped: 5b.
template <bool kGrouped>
static void step5_entry(const half* A, const half* B, half* C, int M, int N, int K)
{
    const CUtensorMap a = step2::map_a(A, M, K), b = step2::map_b(B, K, N);
    if (M > 128) step5::gemm<true, kGrouped>(a, b, C, M, N, K);
    else step5::gemm<false, kGrouped>(a, b, C, M, N, K);
}
// Step 6's bands (6a), called directly with their J and stages (the rule takes J = 1 at the emulator's shapes), at
// M = 16.
template <int J, int NS>
static void step6_bands(const half* A, const half* B, half* C, int M, int N, int K)
{
    const CUtensorMap a = step6::map_a16(A, M, K), b = step2::map_b(B, K, N), c = step6::map_c16(C, M, N);
    step6::bands<J, NS>(a, b, c, C, M, N, K);
}
template <int J>
static emu_dim3 bandsGrid(int, int N)
{
    return emu_dim3{(unsigned)((N + 64 * J - 1) / (64 * J)), 1, 1};
}
// Step 6's persistent tiles (6b, 6c), as step6.cu and step7.cu's v6 launch them, at any shape (the rule picks them at
// the judge's).
template <int BM, int BN, int NS, step6::Pair P>
static void step6_tile(const half* A, const half* B, half* C, int M, int N, int K)
{
    const CUtensorMap a = step2::make_map(A, K, M, 64, BM), b = step2::map_b(B, K, N);
    step6::gemm<BM, BN, NS, P>(a, b, C, M, N, K);
}
template <int BM, int BN, step6::Pair P, int kRes>   // kRes: the clusters (or blocks) the GPU holds at once
static emu_dim3 tileGrid(int M, int N)
{
    int UM, UN;
    step6::units6(M, N, BM, BN, P, UM, UN);
    const int u = UM * UN < kRes ? UM * UN : kRes;
    return emu_dim3{(unsigned)(P == step6::kSolo ? u : 2 * u), 1, 1};
}
// Step 7's tiles with warp specialization (7f), as step7.cu launches them: step 5's tile in pairs side by side
// (step7::gemm5, at M <= 128, the entries' largest M, where step6::choose gives it), the persistent kernel at a tile
// (step7::gemm) and 6c in 7c's groups (step7::gemm_rows), with the switches of 7g (kUni), 7h (kShare), 7i (kTail) and
// 7j (kTma).
template <bool kTma, bool kShare, bool kTail, bool kUni>
static void step7_gemm5(const half* A, const half* B, half* C, int M, int N, int K)
{
    const CUtensorMap a = kShare ? step7::map_a64(A, M, K) : step2::map_a(A, M, K);
    const CUtensorMap b = step2::map_b(B, K, N), c = step7::map_c(C, M, N);
    step7::gemm5<kTma, kShare, kTail, kUni>(a, b, c, C, M, N, K);
}
template <int BM, int BN, int NS, step6::Pair P, int kInc, bool kTma, bool kShare, bool kTail, bool kUni>
static void step7_tile(const half* A, const half* B, half* C, int M, int N, int K)
{
    const CUtensorMap a = step2::make_map(A, K, M, 64, BM), b = step2::map_b(B, K, N), c = step7::map_c(C, M, N);
    step7::gemm<BM, BN, NS, P, kInc, kTma, kShare, kTail, kUni>(a, b, c, C, M, N, K);
}
// 7c: 6c's persistent kernel in groups of rows given by 7c's rule for kRes blocks held at once, on its grid
// (rows7cGrid); with warp specialization (step7::gemm_rows) or without (v8: step7::gemm_rows_noroles).
template <int kRes, bool kTma, bool kUni>
static void step7_rows(const half* A, const half* B, half* C, int M, int N, int K)
{
    int UM, UN, G, R;
    step6::units6(M, N, 128, 256, step6::kSolo, UM, UN);
    step7::rows7c(UM, UN, kRes, G, R);
    const CUtensorMap a = step2::make_map(A, K, M, 64, 128), b = step2::map_b(B, K, N), c = step7::map_c(C, M, N);
    step7::gemm_rows<128, 256, 4, 232, kTma, kUni>(a, b, c, C, M, N, K, G);
}
template <int kRes>
static void step7_rows_noroles(const half* A, const half* B, half* C, int M, int N, int K)
{
    int UM, UN, G, R;
    step6::units6(M, N, 128, 256, step6::kSolo, UM, UN);
    step7::rows7c(UM, UN, kRes, G, R);
    const CUtensorMap a = step2::make_map(A, K, M, 64, 128), b = step2::map_b(B, K, N);
    step7::gemm_rows_noroles<128, 256, 4>(a, b, C, M, N, K, G);
}
template <int kRes>
static emu_dim3 rows7cGrid(int M, int N)
{
    int UM, UN, G, R;
    step6::units6(M, N, 128, 256, step6::kSolo, UM, UN);
    step7::rows7c(UM, UN, kRes, G, R);
    return emu_dim3{(unsigned)R, 1, 1};
}
// Step 7's bands (7a, 7b, 7d), called directly with their rows, J, stages and mode: with warp specialization and the
// switches of 7l (kHint: the emulator loads the hinted boxes as the others), 7h, 7i and 7g; or without (v8).
template <int MM, int J, int NS, step7::Mode P, bool kHint, bool kShare, bool kTail, bool kUni>
static void step7_bands(const half* A, const half* B, half* C, int M, int N, int K)
{
    const CUtensorMap a = step2::make_map(A, K, M, 64, MM);
    const CUtensorMap b = step2::make_map(B, N, K, 64, 64), c = step2::make_map(C, N, M, 64, MM);
    step7::bands<MM, J, NS, P, kHint, kShare, kTail, kUni>(a, b, c, C, M, N, K);
}
template <int MM, int J, int NS, step7::Mode P>
static void step7_bands_noroles(const half* A, const half* B, half* C, int M, int N, int K)
{
    const CUtensorMap a = step2::make_map(A, K, M, 64, MM);
    const CUtensorMap b = step2::make_map(B, N, K, 64, 64), c = step2::make_map(C, N, M, 64, MM);
    step7::bands_noroles<MM, J, NS, P>(a, b, c, C, M, N, K);
}
template <int J, step7::Mode P>
static emu_dim3 bands7Grid(int, int N)
{
    int n = (N + 64 * J - 1) / (64 * J);
    if (P == step7::kAlongN) n = (n + 7) / 8 * 8;   // 7b's clusters of eight
    if (P == step7::kAlongM) n *= 2;
    return emu_dim3{(unsigned)n, 1, 1};
}
using KernelFn = void (*)(const half*, const half*, half*, int, int, int);
struct Expect { const char* cls; double perInstr; };
struct Kernel {
    const char* label;
    unsigned in;                    // the selections that run it
    KernelFn fn;
    emu_dim3 (*grid)(int M, int N);
    emu_dim3 block;
    half* shmem;
    int smemBytes, bStageBytes;     // the launch's dynamic shared memory; where B begins in a stage
    int stageBytes;                 // the size of a stage
    std::vector<Expect> expect;     // wavefronts per shared instruction, as the headers state them
    unsigned sameBits = 0;          // a class of kernels that all give the same C, bit for bit, at a shape; 0: none
    int cluster = 1;                // the blocks of a cluster, run together (step 4), each with its own shared memory
    int maxM = 1 << 30;             // the largest M it computes (step 6's bands: 16)
    int nMultiple = 64;             // N a multiple of it (step 5's tile in step 7: 256)
    int regs = 168;                 // registers per thread at the launch (384 threads: 65 536 / 384, rounded to 8)
};

static std::vector<Kernel> kernels()
{
    namespace s0 = nvidia_sample;
    using G256 = step1::Geometry<256>;
    using G128 = step1::Geometry<128>;
    auto sms = [](int, int) { return emu_dim3{132, 1, 1}; };   // a persistent grid on the H100's 132 SMs, as verif.cu
    auto three = [](int, int) { return emu_dim3{3, 1, 1}; };   // so that each block also goes through several tiles
    auto simple = [](int M, int N) { return emu_dim3{(unsigned)((M + 63) / 64), (unsigned)(N / 64), 1}; };
    auto tiles256 = [](int M, int N) { return emu_dim3{(unsigned)((M + 127) / 128 * (N / 256)), 1, 1}; };
    auto tiles128 = [](int M, int N) { return emu_dim3{(unsigned)((M + 127) / 128 * (N / 128)), 1, 1}; };
    const emu_dim3 b256 = {256, 1, 1}, b128x4 = {128, 4, 1};
    // The headers' bank patterns. Step 0: the copy's STS.128 of A and of B conflict-free, a quarter warp storing 8
    // chunks of one row, on rows of 160 bytes (K64 + 16 halves of padding); the epilogue's LDS.128 conflict-free; wmma
    // runs at the tile level, its LDSM not counted. Step 1: the copy's STS.128 conflict-free on the swizzled rows (a
    // quarter warp's 8 chunks of one row, permuted within their 128-byte line); the output's STS.32 (pitch 132 or 68
    // words, 4 mod 32) and LDS.128 conflict-free; the wgmma's reads of shared memory are not counted.
    const std::vector<Expect> pad160 = {{"STS.128 A", 4}, {"STS.128 B", 4}, {"LDS.128", 4}};
    const std::vector<Expect> swizzle = {{"STS.128 A", 4}, {"STS.128 B", 4}, {"STS.32", 1}, {"LDS.128", 4}};
    const int b0 = 128 * s0::kAStride * 2;
    half* sh1 = (half*)step1::shmem;
    half* sh2 = (half*)step2::shmem;
    const emu_dim3 b128x1 = {128, 1, 1};
    half* sh6 = (half*)g_clusterSh[0];   // step 6's kernels without clusters, in the first cluster block's memory
    using T64 = step6::Tile<64, 128, 8>;
    using T128 = step6::Tile<128, 128, 6>;
    using T320 = step6::Tile<128, 320, 3>;
    using T256 = step6::Tile<128, 256, 4>;
    using B7_16_1 = step7::Bands<16, 1, 8>;
    using B7_16_2 = step7::Bands<16, 2, 8>;
    using B7_16_8 = step7::Bands<16, 8, 3>;
    using B7_16_4 = step7::Bands<16, 4, 6>;
    using B7_16_7 = step7::Bands<16, 7, 3>;
    using B7_64 = step7::Bands<64, 2, 8>;
    using B7_128 = step7::Bands<128, 2, 6>;
    using R64 = step7::Tile<64, 128, 8>;   // the tiles with warp specialization (step 7)
    using R128 = step7::Tile<128, 128, 6>;
    using R256 = step7::Tile<128, 256, 4>;
    using R320 = step7::Tile<128, 320, 3>;
    const emu_dim3 b384 = {384, 1, 1}, b160 = {160, 1, 1};
    auto pairs = [](int M, int N) { return emu_dim3{(unsigned)step4::blocks(M, N, M > 128), 1, 1}; };   // step4.cu's grid
    // Step 5's persistent grids: 66 clusters at most (the H100's 132 SMs, as step5.cu finds them), and 2, so that each
    // cluster goes through several pairs of tiles even at the small shapes; and one cluster per pair (v1).
    auto gpu5 = [](int M, int N) { return emu_dim3{(unsigned)step5::blocks(M, N, M > 128, 66), 1, 1}; };
    auto two5 = [](int M, int N) { return emu_dim3{(unsigned)step5::blocks(M, N, M > 128, 2), 1, 1}; };
    auto all5 = [](int M, int N) { return emu_dim3{(unsigned)step5::blocks(M, N, M > 128, 1 << 30), 1, 1}; };
    // The bands' stmatrix .trans conflict-free (each 8 x 8 matrix's 8 rows of 16 bytes in 8 different chunks of the
    // swizzled 128-byte rows), the TMA's accesses not counted.
    const std::vector<Expect> out5 = {{"STSM", 4}};
    // Step 7's tiles with warp specialization and 7j: the output's STS.32 as below, the zones read by the TMA (not
    // counted); without 7j, the copy out of the zones too (out).
    const std::vector<Expect> outTma = {{"STS.32", 1}};
    // Step 2: the TMA writes the stages (not counted); the output's STS.32 and LDS.128 as step 1's. The tiles of
    // steps 5 to 7: each lane's pairs of halves, STS.32, conflict-free in the swizzled boxes (a warp's 8 rows send
    // their chunk to 8 different chunks of the 128-byte rows), then LDS.128.
    const std::vector<Expect> out = {{"STS.32", 1}, {"LDS.128", 4}};
    // Step 1's variants issue the same MMA on the same values, in the same order of K for each element of C (the
    // wait changes when, the tile how many columns a warpgroup holds, the copy which threads fill the stages): the
    // same C to the bit, in the emulator; step 2 too (the TMA changes how the stages are filled, not what they hold).
    const unsigned same1 = kStep1;
    return {
        {"ref simple_wmma_gemm", kStep0, s0::simple_wmma_gemm, simple, b128x4, s0::shmem, 0, 0, 1, {}},
        {"step0 compute_gemm, 132 blocks", kStep0 | kStep1, s0::compute_gemm, sms, b256, s0::shmem, s0::kSmemBytes, b0,
         s0::kSmemBytes, pad160},
        {"step0 compute_gemm, 3 blocks", kStep0, s0::compute_gemm, three, b256, s0::shmem, s0::kSmemBytes, b0,
         s0::kSmemBytes, pad160},
        {"step1 v0 (step 1: 1a + 1b)", kStep1 | kStep2, step1::gemm<256>, tiles256, b256, sh1, G256::kSmemBytes,
         G256::kABytes, G256::kStage, swizzle, same1},
        {"step1 v1 (a tile of 128 x 128)", kStep1, step1::gemm<128>, tiles128, b256, sh1, G128::kSmemBytes,
         G128::kABytes, G128::kStage, swizzle, same1},
        {"step1 v3 (without 1b)", kStep1, step1::gemm_split<256>, tiles256, b256, sh1, G256::kSmemBytes,
         G256::kABytes, G256::kStage, swizzle, same1},
        {"step2 v0 (step 2: 3 stages)", kStep2 | kStep3, step2_entry<3>, tiles256, b256, sh2, step2::Geometry<3>::kSmemBytes,
         step2::Geometry<3>::kABytes, step2::Geometry<3>::kStage, out, same1},
        {"step2 v1 (2 stages)", kStep2, step2_entry<2>, tiles256, b256, sh2, step2::Geometry<2>::kSmemBytes,
         step2::Geometry<2>::kABytes, step2::Geometry<2>::kStage, out, same1},
        {"step2 v3 (without 2d)", kStep2, step2_entry_w0<3>, tiles256, b256, sh2, step2::Geometry<3>::kSmemBytes,
         step2::Geometry<3>::kABytes, step2::Geometry<3>::kStage, out, same1},
        {"step2 v4 (2 stages without 2d)", kStep2, step2_entry_w0<2>, tiles256, b256, sh2, step2::Geometry<2>::kSmemBytes,
         step2::Geometry<2>::kABytes, step2::Geometry<2>::kStage, out, same1},
        {"step3 v0 (step 3: 3a, step 2's kernel at 4 stages)", kStep3 | kStep4, step2_entry<4>, tiles256, b256, sh2,
         step2::Geometry<4>::kSmemBytes, step2::Geometry<4>::kABytes, step2::Geometry<4>::kStage, out, same1},
        {"step4 v0 (step 4: 4a, 4b)", kStep4 | kStep5, step4_entry, pairs, b256, nullptr, step4::G::kSmemBytes,
         step4::G::kABytes, step4::G::kStage, out, same1, 2},
        // Step 5 (5a, 5b: 256 threads that all compute, warp 0 loading; each block its own A and B; the output by pairs
        // of halves, copied out by the threads: STS.32 and LDS.128, conflict-free)
        {"step5 v0 (step 5), 66 clusters", kStep5, step5_entry<true>, gpu5, b256, nullptr,
         step5::G::kSmemBytes, step5::G::kABytes, step5::G::kStage, out, same1, 2},
        {"step5 v0 (step 5), 2 clusters", kStep5, step5_entry<true>, two5, b256, nullptr,
         step5::G::kSmemBytes, step5::G::kABytes, step5::G::kStage, out, same1, 2},
        {"step5 v1 (without 5a)", kStep5, step5_entry<true>, all5, b256, nullptr, step5::G::kSmemBytes,
         step5::G::kABytes, step5::G::kStage, out, same1, 2},
        {"step5 v2 (without 5b), 2 clusters", kStep5, step5_entry<false>, two5, b256, nullptr,
         step5::G::kSmemBytes, step5::G::kABytes, step5::G::kStage, out, same1, 2},
        // Step 5's tile as steps 6 and 7 run it, at the shapes it takes (N a multiple of 256)
        {"step6 step 5's tile, 66 clusters", kStep6, step5_entry<true>, gpu5, b256, nullptr,
         step5::G::kSmemBytes, step5::G::kABytes, step5::G::kStage, out, same1, 2, 1 << 30, 256},
        {"step6 step 5's tile, 2 clusters", kStep6, step5_entry<true>, two5, b256, nullptr,
         step5::G::kSmemBytes, step5::G::kABytes, step5::G::kStage, out, same1, 2, 1 << 30, 256},
        // Step 6's bands (6a), J = 1 and 2 at 8 stages, J = 7 at 3 (the stage refilled at wait_group 0); step 7's v6
        // (step 6 itself) runs them. The stmatrix .trans conflict-free.
        {"step6 6a bands J=1, 8 stages", kStep6, step6_bands<1, 8>,
         bandsGrid<1>, b128x1, sh6, step6::Bands<1, 8>::kSmemBytes, step6::Bands<1, 8>::kABytes, step6::Bands<1,
         8>::kStage, out5, same1, 1, 16},
        {"step6 6a bands J=2, 8 stages", kStep6, step6_bands<2, 8>,
         bandsGrid<2>, b128x1, sh6, step6::Bands<2, 8>::kSmemBytes, step6::Bands<2, 8>::kABytes, step6::Bands<2,
         8>::kStage, out5, same1, 1, 16},
        {"step6 6a bands J=7, 3 stages", kStep6, step6_bands<7, 3>,
         bandsGrid<7>, b128x1, sh6, step6::Bands<7, 3>::kSmemBytes, step6::Bands<7, 3>::kABytes, step6::Bands<7,
         3>::kStage, out5, same1, 1, 16},
        // Step 6's tiles (6b, 6c), at the GPU's grids and at grids of 2 clusters or 3 blocks (several tiles per
        // cluster); in pairs, each block its own A and B. The output by pairs of halves, copied out by the threads.
        {"step6 6b 64x128 pairs, 66 clusters", kStep6, step6_tile<64, 128, 8, step6::kAlongM>,
         tileGrid<64, 128, step6::kAlongM, 66>, b256, nullptr, T64::kSmemBytes, T64::kABytes, T64::kStage, out,
         same1, 2},
        {"step6 6b 64x128 pairs, 2 clusters", kStep6, step6_tile<64, 128, 8, step6::kAlongM>,
         tileGrid<64, 128, step6::kAlongM, 2>, b256, nullptr, T64::kSmemBytes, T64::kABytes, T64::kStage, out, same1,
         2},
        {"step6 6b 64x128 alone, 3 blocks", kStep6, step6_tile<64, 128, 8, step6::kSolo>,
         tileGrid<64, 128, step6::kSolo, 3>, b256, sh6, T64::kSmemBytes, T64::kABytes, T64::kStage, out, same1},
        {"step6 6b 128x128, 132 blocks", kStep6, step6_tile<128, 128, 6, step6::kSolo>,
         tileGrid<128, 128, step6::kSolo, 132>, b256, sh6, T128::kSmemBytes, T128::kABytes, T128::kStage, out, same1},
        {"step6 6b 128x128, 3 blocks", kStep6, step6_tile<128, 128, 6, step6::kSolo>,
         tileGrid<128, 128, step6::kSolo, 3>, b256, sh6, T128::kSmemBytes, T128::kABytes, T128::kStage, out, same1},
        {"step6 6c 128x256 alone, 132 blocks", kStep6, step6_tile<128, 256, 4, step6::kSolo>,
         tileGrid<128, 256, step6::kSolo, 132>, b256, sh6, T256::kSmemBytes, T256::kABytes, T256::kStage, out, same1},
        {"step6 6c 128x256 alone, 3 blocks", kStep6, step6_tile<128, 256, 4, step6::kSolo>,
         tileGrid<128, 256, step6::kSolo, 3>, b256, sh6, T256::kSmemBytes, T256::kABytes, T256::kStage, out, same1},
        // Step 7's kernels, every one that its variants launch at the judge's shapes and verif.cu's: v0's at the GPU's
        // grids and at 2 clusters or 3 blocks (several tiles each), then those of the variants that change them (v2,
        // v7, v8, v9 to v14) at the small grids; step 5's tile along N only, at M <= 128, as step 7 runs it. Step 6's
        // kernels, which v6 and v8 launch too, run in step 6's runs, but its bands at J = 4 and 8. The tiles' output
        // STS.32 conflict-free; without 7j, the copy out of the zones too.
        {"step7 v0: step 5's tile in pairs, 66 clusters", kStep7, step7_gemm5<true, true, true, true>, gpu5, b384,
         nullptr, step5::G::kSmemBytes, step5::G::kABytes, step5::G::kStage, outTma, same1, 2, 128, 256},
        {"step7 v0: step 5's tile in pairs, 2 clusters", kStep7, step7_gemm5<true, true, true, true>, two5, b384,
         nullptr, step5::G::kSmemBytes, step5::G::kABytes, step5::G::kStage, outTma, same1, 2, 128, 256},
        {"step7 v9, 7f alone: step 5's tile in pairs, 2 clusters", kStep7, step7_gemm5<false, false, false, false>,
         two5, b384, nullptr, step5::G::kSmemBytes, step5::G::kABytes, step5::G::kStage, out, same1, 2, 128, 256},
        {"step7 v10, without 7g: step 5's tile in pairs, 2 clusters", kStep7, step7_gemm5<true, true, true, false>,
         two5, b384, nullptr, step5::G::kSmemBytes, step5::G::kABytes, step5::G::kStage, outTma, same1, 2, 128, 256},
        {"step7 v11, without 7h: step 5's tile in pairs, 2 clusters", kStep7, step7_gemm5<true, false, false, true>,
         two5, b384, nullptr, step5::G::kSmemBytes, step5::G::kABytes, step5::G::kStage, outTma, same1, 2, 128, 256},
        {"step7 v12, without 7i: step 5's tile in pairs, 2 clusters", kStep7, step7_gemm5<true, true, false, true>,
         two5, b384, nullptr, step5::G::kSmemBytes, step5::G::kABytes, step5::G::kStage, outTma, same1, 2, 128, 256},
        {"step7 v13, without 7j: step 5's tile in pairs, 2 clusters", kStep7, step7_gemm5<false, true, true, true>,
         two5, b384, nullptr, step5::G::kSmemBytes, step5::G::kABytes, step5::G::kStage, out, same1, 2, 128, 256},
        {"step7 v0: 64x128 pairs, 66 clusters", kStep7, step7_tile<64, 128, 8, step6::kAlongM, 0, true, true,
         true, true>, tileGrid<64, 128, step6::kAlongM, 66>, b384, nullptr, R64::kSmemBytes, R64::kABytes, R64::kStage,
         outTma, same1, 2},
        {"step7 v0: 64x128 pairs, 2 clusters", kStep7, step7_tile<64, 128, 8, step6::kAlongM, 0, true, true,
         true, true>, tileGrid<64, 128, step6::kAlongM, 2>, b384, nullptr, R64::kSmemBytes, R64::kABytes, R64::kStage,
         outTma, same1, 2},
        {"step7 v9, 7f alone: 64x128 pairs, 2 clusters", kStep7, step7_tile<64, 128, 8, step6::kAlongM, 0, false,
         false, false, false>, tileGrid<64, 128, step6::kAlongM, 2>, b384, nullptr, R64::kSmemBytes, R64::kABytes,
         R64::kStage, out, same1, 2},
        {"step7 v10, without 7g: 64x128 pairs, 2 clusters", kStep7, step7_tile<64, 128, 8, step6::kAlongM, 0,
         true, true, true, false>, tileGrid<64, 128, step6::kAlongM, 2>, b384, nullptr, R64::kSmemBytes, R64::kABytes,
         R64::kStage, outTma, same1, 2},
        {"step7 v11, without 7h: 64x128 pairs, 2 clusters", kStep7, step7_tile<64, 128, 8, step6::kAlongM, 0,
         true, false, false, true>, tileGrid<64, 128, step6::kAlongM, 2>, b384, nullptr, R64::kSmemBytes, R64::kABytes,
         R64::kStage, outTma, same1, 2},
        {"step7 v12, without 7i: 64x128 pairs, 2 clusters", kStep7, step7_tile<64, 128, 8, step6::kAlongM, 0,
         true, true, false, true>, tileGrid<64, 128, step6::kAlongM, 2>, b384, nullptr, R64::kSmemBytes, R64::kABytes,
         R64::kStage, outTma, same1, 2},
        {"step7 v13, without 7j: 64x128 pairs, 2 clusters", kStep7, step7_tile<64, 128, 8, step6::kAlongM, 0,
         false, true, true, true>, tileGrid<64, 128, step6::kAlongM, 2>, b384, nullptr, R64::kSmemBytes, R64::kABytes,
         R64::kStage, out, same1, 2},
        {"step7 v0: 64x128 alone, 3 blocks", kStep7, step7_tile<64, 128, 8, step6::kSolo, 0, true, false, false,
         true>, tileGrid<64, 128, step6::kSolo, 3>, b384, sh6, R64::kSmemBytes, R64::kABytes, R64::kStage, outTma,
         same1},
        {"step7 v9, 7f alone: 64x128 alone, 3 blocks", kStep7, step7_tile<64, 128, 8, step6::kSolo, 0, false,
         false, false, false>, tileGrid<64, 128, step6::kSolo, 3>, b384, sh6, R64::kSmemBytes, R64::kABytes,
         R64::kStage, out, same1},
        {"step7 v10, without 7g: 64x128 alone, 3 blocks", kStep7, step7_tile<64, 128, 8, step6::kSolo, 0, true,
         false, false, false>, tileGrid<64, 128, step6::kSolo, 3>, b384, sh6, R64::kSmemBytes, R64::kABytes,
         R64::kStage, outTma, same1},
        {"step7 v13, without 7j: 64x128 alone, 3 blocks", kStep7, step7_tile<64, 128, 8, step6::kSolo, 0, false,
         false, false, true>, tileGrid<64, 128, step6::kSolo, 3>, b384, sh6, R64::kSmemBytes, R64::kABytes, R64::kStage,
         out, same1},
        {"step7 v0: 128x128, 132 blocks", kStep7, step7_tile<128, 128, 6, step6::kSolo, 0, true, false, false,
         true>, tileGrid<128, 128, step6::kSolo, 132>, b384, sh6, R128::kSmemBytes, R128::kABytes, R128::kStage, outTma,
         same1},
        {"step7 v0: 128x128, 3 blocks", kStep7, step7_tile<128, 128, 6, step6::kSolo, 0, true, false, false,
         true>, tileGrid<128, 128, step6::kSolo, 3>, b384, sh6, R128::kSmemBytes, R128::kABytes, R128::kStage, outTma,
         same1},
        {"step7 v9, 7f alone: 128x128, 3 blocks", kStep7, step7_tile<128, 128, 6, step6::kSolo, 0, false, false,
         false, false>, tileGrid<128, 128, step6::kSolo, 3>, b384, sh6, R128::kSmemBytes, R128::kABytes, R128::kStage,
         out, same1},
        {"step7 v10, without 7g: 128x128, 3 blocks", kStep7, step7_tile<128, 128, 6, step6::kSolo, 0, true,
         false, false, false>, tileGrid<128, 128, step6::kSolo, 3>, b384, sh6, R128::kSmemBytes, R128::kABytes,
         R128::kStage, outTma, same1},
        {"step7 v13, without 7j: 128x128, 3 blocks", kStep7, step7_tile<128, 128, 6, step6::kSolo, 0, false,
         false, false, true>, tileGrid<128, 128, step6::kSolo, 3>, b384, sh6, R128::kSmemBytes, R128::kABytes,
         R128::kStage, out, same1},
        {"step7 v0: 7c 6c, 132 blocks", kStep7, step7_rows<132, true, true>, rows7cGrid<132>, b384, sh6,
         R256::kSmemBytes, R256::kABytes, R256::kStage, outTma, same1},
        {"step7 v0: 7c 6c, 3 blocks", kStep7, step7_rows<3, true, true>, rows7cGrid<3>, b384, sh6, R256::kSmemBytes,
         R256::kABytes, R256::kStage, outTma, same1},
        {"step7 v9, 7f alone: 7c 6c, 3 blocks", kStep7, step7_rows<3, false, false>, rows7cGrid<3>, b384, sh6,
         R256::kSmemBytes, R256::kABytes, R256::kStage, out, same1},
        {"step7 v10, without 7g: 7c 6c, 3 blocks", kStep7, step7_rows<3, true, false>, rows7cGrid<3>, b384, sh6,
         R256::kSmemBytes, R256::kABytes, R256::kStage, outTma, same1},
        {"step7 v13, without 7j: 7c 6c, 3 blocks", kStep7, step7_rows<3, false, true>, rows7cGrid<3>, b384, sh6,
         R256::kSmemBytes, R256::kABytes, R256::kStage, out, same1},
        {"step7 v7, without 7c: 6c, 132 blocks", kStep7, step7_tile<128, 256, 4, step6::kSolo, 232, true, false,
         false, true>, tileGrid<128, 256, step6::kSolo, 132>, b384, sh6, R256::kSmemBytes, R256::kABytes, R256::kStage,
         outTma, same1},
        {"step7 v7, without 7c: 6c, 3 blocks", kStep7, step7_tile<128, 256, 4, step6::kSolo, 232, true, false,
         false, true>, tileGrid<128, 256, step6::kSolo, 3>, b384, sh6, R256::kSmemBytes, R256::kABytes, R256::kStage,
         outTma, same1},
        {"step7 v0: 7e 128x320 pairs, 66 clusters", kStep7, step7_tile<128, 320, 3, step6::kAlongM, 232, true, true,
         true, true>, tileGrid<128, 320, step6::kAlongM, 66>, b384, nullptr, R320::kSmemBytes, R320::kABytes,
         R320::kStage, outTma, same1, 2},
        {"step7 v0: 7e 128x320 pairs, 2 clusters", kStep7, step7_tile<128, 320, 3, step6::kAlongM, 232, true, true,
         true, true>, tileGrid<128, 320, step6::kAlongM, 2>, b384, nullptr, R320::kSmemBytes, R320::kABytes,
         R320::kStage, outTma, same1, 2},
        {"step7 v10, without 7g: 7e 128x320 pairs, 2 clusters", kStep7, step7_tile<128, 320, 3, step6::kAlongM, 232,
         true, true, true, false>, tileGrid<128, 320, step6::kAlongM, 2>, b384, nullptr, R320::kSmemBytes,
         R320::kABytes, R320::kStage, outTma, same1, 2},
        {"step7 v11, without 7h: 7e 128x320 pairs, 2 clusters", kStep7, step7_tile<128, 320, 3, step6::kAlongM, 232,
         true, false, false, true>, tileGrid<128, 320, step6::kAlongM, 2>, b384, nullptr, R320::kSmemBytes,
         R320::kABytes, R320::kStage, outTma, same1, 2},
        {"step7 v12, without 7i: 7e 128x320 pairs, 2 clusters", kStep7, step7_tile<128, 320, 3, step6::kAlongM, 232,
         true, true, false, true>, tileGrid<128, 320, step6::kAlongM, 2>, b384, nullptr, R320::kSmemBytes,
         R320::kABytes, R320::kStage, outTma, same1, 2},
        {"step7 v13, without 7j: 7e 128x320 pairs, 2 clusters", kStep7, step7_tile<128, 320, 3, step6::kAlongM, 232,
         false, true, true, true>, tileGrid<128, 320, step6::kAlongM, 2>, b384, nullptr, R320::kSmemBytes,
         R320::kABytes, R320::kStage, out, same1, 2},
        {"step7 v0: 7e 128x320 alone, 132 blocks", kStep7, step7_tile<128, 320, 3, step6::kSolo, 232, true, false,
         false, true>, tileGrid<128, 320, step6::kSolo, 132>, b384, sh6, R320::kSmemBytes, R320::kABytes,
         R320::kStage, outTma, same1},
        {"step7 v0: 7e 128x320 alone, 3 blocks", kStep7, step7_tile<128, 320, 3, step6::kSolo, 232, true, false,
         false, true>, tileGrid<128, 320, step6::kSolo, 3>, b384, sh6, R320::kSmemBytes, R320::kABytes, R320::kStage,
         outTma, same1},
        {"step7 v10, without 7g: 7e 128x320 alone, 3 blocks", kStep7, step7_tile<128, 320, 3, step6::kSolo, 232, true,
         false, false, false>, tileGrid<128, 320, step6::kSolo, 3>, b384, sh6, R320::kSmemBytes, R320::kABytes,
         R320::kStage, outTma, same1},
        {"step7 v13, without 7j: 7e 128x320 alone, 3 blocks", kStep7, step7_tile<128, 320, 3, step6::kSolo, 232, false,
         false, false, true>, tileGrid<128, 320, step6::kSolo, 3>, b384, sh6, R320::kSmemBytes, R320::kABytes,
         R320::kStage, out, same1},
        {"step7 v9, 7f alone: 7e 128x320 pairs, 2 clusters", kStep7, step7_tile<128, 320, 3, step6::kAlongM, 232,
         false, false, false, false>, tileGrid<128, 320, step6::kAlongM, 2>, b384, nullptr, R320::kSmemBytes,
         R320::kABytes, R320::kStage, out, same1, 2},
        {"step7 v9, 7f alone: 7e 128x320 alone, 3 blocks", kStep7, step7_tile<128, 320, 3, step6::kSolo, 232, false,
         false, false, false>, tileGrid<128, 320, step6::kSolo, 3>, b384, sh6, R320::kSmemBytes, R320::kABytes,
         R320::kStage, out, same1},
        {"step7 v0: 7b bands 16 rows J=2 in clusters of 8", kStep7, step7_bands<16, 2, 8, step7::kAlongN, true, true,
         true, true>, bands7Grid<2, step7::kAlongN>, b160, nullptr, B7_16_2::kSmemBytes, B7_16_2::kABytes,
         B7_16_2::kStage, out5, same1, 8, 16},
        {"step7 v0: 7b bands 16 rows J=4 in clusters of 8", kStep7, step7_bands<16, 4, 6, step7::kAlongN, true, true,
         true, true>, bands7Grid<4, step7::kAlongN>, b160, nullptr, B7_16_4::kSmemBytes, B7_16_4::kABytes,
         B7_16_4::kStage, out5, same1, 8, 16},
        {"step7 v0: 7b bands 16 rows J=8 in clusters of 8", kStep7, step7_bands<16, 8, 3, step7::kAlongN, true, true,
         true, true>, bands7Grid<8, step7::kAlongN>, b160, nullptr, B7_16_8::kSmemBytes, B7_16_8::kABytes,
         B7_16_8::kStage, out5, same1, 8, 16},
        {"step7 v9, 7f alone: 7b bands 16 rows J=2 in clusters of 8", kStep7, step7_bands<16, 2, 8, step7::kAlongN,
         false, false, false, false>, bands7Grid<2, step7::kAlongN>, b160, nullptr, B7_16_2::kSmemBytes,
         B7_16_2::kABytes, B7_16_2::kStage, out5, same1, 8, 16},
        {"step7 v9, 7f alone: 7b bands 16 rows J=4 in clusters of 8", kStep7, step7_bands<16, 4, 6, step7::kAlongN,
         false, false, false, false>, bands7Grid<4, step7::kAlongN>, b160, nullptr, B7_16_4::kSmemBytes,
         B7_16_4::kABytes, B7_16_4::kStage, out5, same1, 8, 16},
        {"step7 v9, 7f alone: 7b bands 16 rows J=8 in clusters of 8", kStep7, step7_bands<16, 8, 3, step7::kAlongN,
         false, false, false, false>, bands7Grid<8, step7::kAlongN>, b160, nullptr, B7_16_8::kSmemBytes,
         B7_16_8::kABytes, B7_16_8::kStage, out5, same1, 8, 16},
        {"step7 v10, without 7g: 7b bands 16 rows J=2 in clusters of 8", kStep7, step7_bands<16, 2, 8, step7::kAlongN,
         true, true, true, false>, bands7Grid<2, step7::kAlongN>, b160, nullptr, B7_16_2::kSmemBytes, B7_16_2::kABytes,
         B7_16_2::kStage, out5, same1, 8, 16},
        {"step7 v10, without 7g: 7b bands 16 rows J=4 in clusters of 8", kStep7, step7_bands<16, 4, 6, step7::kAlongN,
         true, true, true, false>, bands7Grid<4, step7::kAlongN>, b160, nullptr, B7_16_4::kSmemBytes, B7_16_4::kABytes,
         B7_16_4::kStage, out5, same1, 8, 16},
        {"step7 v10, without 7g: 7b bands 16 rows J=8 in clusters of 8", kStep7, step7_bands<16, 8, 3, step7::kAlongN,
         true, true, true, false>, bands7Grid<8, step7::kAlongN>, b160, nullptr, B7_16_8::kSmemBytes, B7_16_8::kABytes,
         B7_16_8::kStage, out5, same1, 8, 16},
        {"step7 v11, without 7h: 7b bands 16 rows J=2 in clusters of 8", kStep7, step7_bands<16, 2, 8, step7::kAlongN,
         true, false, false, true>, bands7Grid<2, step7::kAlongN>, b160, nullptr, B7_16_2::kSmemBytes, B7_16_2::kABytes,
         B7_16_2::kStage, out5, same1, 8, 16},
        {"step7 v11, without 7h: 7b bands 16 rows J=4 in clusters of 8", kStep7, step7_bands<16, 4, 6, step7::kAlongN,
         true, false, false, true>, bands7Grid<4, step7::kAlongN>, b160, nullptr, B7_16_4::kSmemBytes, B7_16_4::kABytes,
         B7_16_4::kStage, out5, same1, 8, 16},
        {"step7 v11, without 7h: 7b bands 16 rows J=8 in clusters of 8", kStep7, step7_bands<16, 8, 3, step7::kAlongN,
         true, false, false, true>, bands7Grid<8, step7::kAlongN>, b160, nullptr, B7_16_8::kSmemBytes, B7_16_8::kABytes,
         B7_16_8::kStage, out5, same1, 8, 16},
        {"step7 v12, without 7i: 7b bands 16 rows J=2 in clusters of 8", kStep7, step7_bands<16, 2, 8, step7::kAlongN,
         true, true, false, true>, bands7Grid<2, step7::kAlongN>, b160, nullptr, B7_16_2::kSmemBytes, B7_16_2::kABytes,
         B7_16_2::kStage, out5, same1, 8, 16},
        {"step7 v12, without 7i: 7b bands 16 rows J=4 in clusters of 8", kStep7, step7_bands<16, 4, 6, step7::kAlongN,
         true, true, false, true>, bands7Grid<4, step7::kAlongN>, b160, nullptr, B7_16_4::kSmemBytes, B7_16_4::kABytes,
         B7_16_4::kStage, out5, same1, 8, 16},
        {"step7 v12, without 7i: 7b bands 16 rows J=8 in clusters of 8", kStep7, step7_bands<16, 8, 3, step7::kAlongN,
         true, true, false, true>, bands7Grid<8, step7::kAlongN>, b160, nullptr, B7_16_8::kSmemBytes, B7_16_8::kABytes,
         B7_16_8::kStage, out5, same1, 8, 16},
        {"step7 v14, without 7l: 7b bands 16 rows J=2 in clusters of 8", kStep7, step7_bands<16, 2, 8, step7::kAlongN,
         false, true, true, true>, bands7Grid<2, step7::kAlongN>, b160, nullptr, B7_16_2::kSmemBytes, B7_16_2::kABytes,
         B7_16_2::kStage, out5, same1, 8, 16},
        {"step7 v14, without 7l: 7b bands 16 rows J=4 in clusters of 8", kStep7, step7_bands<16, 4, 6, step7::kAlongN,
         false, true, true, true>, bands7Grid<4, step7::kAlongN>, b160, nullptr, B7_16_4::kSmemBytes, B7_16_4::kABytes,
         B7_16_4::kStage, out5, same1, 8, 16},
        {"step7 v14, without 7l: 7b bands 16 rows J=8 in clusters of 8", kStep7, step7_bands<16, 8, 3, step7::kAlongN,
         false, true, true, true>, bands7Grid<8, step7::kAlongN>, b160, nullptr, B7_16_8::kSmemBytes, B7_16_8::kABytes,
         B7_16_8::kStage, out5, same1, 8, 16},
        {"step7 v0: bands 16 rows J=1 alone", kStep7, step7_bands<16, 1, 8, step7::kSolo, true, false, false, true>,
         bands7Grid<1, step7::kSolo>, b160, sh6, B7_16_1::kSmemBytes, B7_16_1::kABytes, B7_16_1::kStage, out5, same1, 1,
         16},
        {"step7 v0: bands 16 rows J=7 alone", kStep7, step7_bands<16, 7, 3, step7::kSolo, true, false, false, true>,
         bands7Grid<7, step7::kSolo>, b160, sh6, B7_16_7::kSmemBytes, B7_16_7::kABytes, B7_16_7::kStage, out5, same1, 1,
         16},
        {"step7 v0: bands 16 rows J=8 alone", kStep7, step7_bands<16, 8, 3, step7::kSolo, true, false, false, true>,
         bands7Grid<8, step7::kSolo>, b160, sh6, B7_16_8::kSmemBytes, B7_16_8::kABytes, B7_16_8::kStage, out5, same1, 1,
         16},
        {"step7 v2, without 7b: bands 16 rows J=2 alone", kStep7, step7_bands<16, 2, 8, step7::kSolo, true, false,
         false, true>, bands7Grid<2, step7::kSolo>, b160, sh6, B7_16_2::kSmemBytes, B7_16_2::kABytes, B7_16_2::kStage,
         out5, same1, 1, 16},
        {"step7 v2, without 7b: bands 16 rows J=4 alone", kStep7, step7_bands<16, 4, 6, step7::kSolo, true, false,
         false, true>, bands7Grid<4, step7::kSolo>, b160, sh6, B7_16_4::kSmemBytes, B7_16_4::kABytes, B7_16_4::kStage,
         out5, same1, 1, 16},
        {"step7 v9, 7f alone: bands 16 rows J=7 alone", kStep7, step7_bands<16, 7, 3, step7::kSolo, false, false, false,
         false>, bands7Grid<7, step7::kSolo>, b160, sh6, B7_16_7::kSmemBytes, B7_16_7::kABytes, B7_16_7::kStage, out5,
         same1, 1, 16},
        {"step7 v9, 7f alone: bands 16 rows J=8 alone", kStep7, step7_bands<16, 8, 3, step7::kSolo, false, false, false,
         false>, bands7Grid<8, step7::kSolo>, b160, sh6, B7_16_8::kSmemBytes, B7_16_8::kABytes, B7_16_8::kStage, out5,
         same1, 1, 16},
        {"step7 v10, without 7g: bands 16 rows J=7 alone", kStep7, step7_bands<16, 7, 3, step7::kSolo, true, false,
         false, false>, bands7Grid<7, step7::kSolo>, b160, sh6, B7_16_7::kSmemBytes, B7_16_7::kABytes, B7_16_7::kStage,
         out5, same1, 1, 16},
        {"step7 v10, without 7g: bands 16 rows J=8 alone", kStep7, step7_bands<16, 8, 3, step7::kSolo, true, false,
         false, false>, bands7Grid<8, step7::kSolo>, b160, sh6, B7_16_8::kSmemBytes, B7_16_8::kABytes, B7_16_8::kStage,
         out5, same1, 1, 16},
        {"step7 v14, without 7l: bands 16 rows J=7 alone", kStep7, step7_bands<16, 7, 3, step7::kSolo, false, false,
         false, true>, bands7Grid<7, step7::kSolo>, b160, sh6, B7_16_7::kSmemBytes, B7_16_7::kABytes, B7_16_7::kStage,
         out5, same1, 1, 16},
        {"step7 v14, without 7l: bands 16 rows J=8 alone", kStep7, step7_bands<16, 8, 3, step7::kSolo, false, false,
         false, true>, bands7Grid<8, step7::kSolo>, b160, sh6, B7_16_8::kSmemBytes, B7_16_8::kABytes, B7_16_8::kStage,
         out5, same1, 1, 16},
        {"step7 v0: 7a bands 64 rows in pairs", kStep7, step7_bands<64, 2, 8, step7::kAlongM, true, true, true, true>,
         bands7Grid<2, step7::kAlongM>, b160, nullptr, B7_64::kSmemBytes, B7_64::kABytes, B7_64::kStage, out5, same1, 2,
         128},
        {"step7 v9, 7f alone: 7a bands 64 rows in pairs", kStep7, step7_bands<64, 2, 8, step7::kAlongM, false, false,
         false, false>, bands7Grid<2, step7::kAlongM>, b160, nullptr, B7_64::kSmemBytes, B7_64::kABytes, B7_64::kStage,
         out5, same1, 2, 128},
        {"step7 v10, without 7g: 7a bands 64 rows in pairs", kStep7, step7_bands<64, 2, 8, step7::kAlongM, true, true,
         true, false>, bands7Grid<2, step7::kAlongM>, b160, nullptr, B7_64::kSmemBytes, B7_64::kABytes, B7_64::kStage,
         out5, same1, 2, 128},
        {"step7 v11, without 7h: 7a bands 64 rows in pairs", kStep7, step7_bands<64, 2, 8, step7::kAlongM, true, false,
         false, true>, bands7Grid<2, step7::kAlongM>, b160, nullptr, B7_64::kSmemBytes, B7_64::kABytes, B7_64::kStage,
         out5, same1, 2, 128},
        {"step7 v12, without 7i: 7a bands 64 rows in pairs", kStep7, step7_bands<64, 2, 8, step7::kAlongM, true, true,
         false, true>, bands7Grid<2, step7::kAlongM>, b160, nullptr, B7_64::kSmemBytes, B7_64::kABytes, B7_64::kStage,
         out5, same1, 2, 128},
        {"step7 v14, without 7l: 7a bands 64 rows in pairs", kStep7, step7_bands<64, 2, 8, step7::kAlongM, false, true,
         true, true>, bands7Grid<2, step7::kAlongM>, b160, nullptr, B7_64::kSmemBytes, B7_64::kABytes, B7_64::kStage,
         out5, same1, 2, 128},
        {"step7 v0: 7d bands 128 rows", kStep7, step7_bands<128, 2, 6, step7::kSolo, true, false, false, true>,
         bands7Grid<2, step7::kSolo>, b160, sh6, B7_128::kSmemBytes, B7_128::kABytes, B7_128::kStage, out5, same1, 1,
         128},
        {"step7 v9, 7f alone: 7d bands 128 rows", kStep7, step7_bands<128, 2, 6, step7::kSolo, false, false, false,
         false>, bands7Grid<2, step7::kSolo>, b160, sh6, B7_128::kSmemBytes, B7_128::kABytes, B7_128::kStage, out5,
         same1, 1, 128},
        {"step7 v10, without 7g: 7d bands 128 rows", kStep7, step7_bands<128, 2, 6, step7::kSolo, true, false, false,
         false>, bands7Grid<2, step7::kSolo>, b160, sh6, B7_128::kSmemBytes, B7_128::kABytes, B7_128::kStage, out5,
         same1, 1, 128},
        {"step7 v14, without 7l: 7d bands 128 rows", kStep7, step7_bands<128, 2, 6, step7::kSolo, false, false, false,
         true>, bands7Grid<2, step7::kSolo>, b160, sh6, B7_128::kSmemBytes, B7_128::kABytes, B7_128::kStage, out5,
         same1, 1, 128},
        {"step7 v8: 7b bands 16 rows J=2 in clusters of 8", kStep7, step7_bands_noroles<16, 2, 8, step7::kAlongN>,
         bands7Grid<2, step7::kAlongN>, b128x1, nullptr, B7_16_2::kSmemBytes, B7_16_2::kABytes, B7_16_2::kStage, out5,
         same1, 8, 16},
        {"step7 v8: 7b bands 16 rows J=4 in clusters of 8", kStep7, step7_bands_noroles<16, 4, 6, step7::kAlongN>,
         bands7Grid<4, step7::kAlongN>, b128x1, nullptr, B7_16_4::kSmemBytes, B7_16_4::kABytes, B7_16_4::kStage, out5,
         same1, 8, 16},
        {"step7 v8: 7b bands 16 rows J=8 in clusters of 8", kStep7, step7_bands_noroles<16, 8, 3, step7::kAlongN>,
         bands7Grid<8, step7::kAlongN>, b128x1, nullptr, B7_16_8::kSmemBytes, B7_16_8::kABytes, B7_16_8::kStage, out5,
         same1, 8, 16},
        {"step7 v8: bands 16 rows J=1 alone", kStep7, step7_bands_noroles<16, 1, 8, step7::kSolo>, bands7Grid<1,
         step7::kSolo>, b128x1, sh6, B7_16_1::kSmemBytes, B7_16_1::kABytes, B7_16_1::kStage, out5, same1, 1, 16},
        {"step7 v8: bands 16 rows J=7 alone", kStep7, step7_bands_noroles<16, 7, 3, step7::kSolo>, bands7Grid<7,
         step7::kSolo>, b128x1, sh6, B7_16_7::kSmemBytes, B7_16_7::kABytes, B7_16_7::kStage, out5, same1, 1, 16},
        {"step7 v8: bands 16 rows J=8 alone", kStep7, step7_bands_noroles<16, 8, 3, step7::kSolo>, bands7Grid<8,
         step7::kSolo>, b128x1, sh6, B7_16_8::kSmemBytes, B7_16_8::kABytes, B7_16_8::kStage, out5, same1, 1, 16},
        {"step7 v8: 7a bands 64 rows in pairs", kStep7, step7_bands_noroles<64, 2, 8, step7::kAlongM>, bands7Grid<2,
         step7::kAlongM>, b128x1, nullptr, B7_64::kSmemBytes, B7_64::kABytes, B7_64::kStage, out5, same1, 2, 128},
        {"step7 v8: 7d bands 128 rows", kStep7, step7_bands_noroles<128, 2, 6, step7::kSolo>, bands7Grid<2,
         step7::kSolo>, b128x1, sh6, B7_128::kSmemBytes, B7_128::kABytes, B7_128::kStage, out5, same1, 1, 128},
        {"step7 v8: 7e 128x320 pairs, 66 clusters", kStep7, step6_tile<128, 320, 3, step6::kAlongM>, tileGrid<128, 320,
         step6::kAlongM, 66>, b256, nullptr, T320::kSmemBytes, T320::kABytes, T320::kStage, out, same1, 2},
        {"step7 v8: 7e 128x320 pairs, 2 clusters", kStep7, step6_tile<128, 320, 3, step6::kAlongM>, tileGrid<128, 320,
         step6::kAlongM, 2>, b256, nullptr, T320::kSmemBytes, T320::kABytes, T320::kStage, out, same1, 2},
        {"step7 v8: 7e 128x320 alone, 132 blocks", kStep7, step6_tile<128, 320, 3, step6::kSolo>, tileGrid<128, 320,
         step6::kSolo, 132>, b256, sh6, T320::kSmemBytes, T320::kABytes, T320::kStage, out, same1},
        {"step7 v8: 7e 128x320 alone, 3 blocks", kStep7, step6_tile<128, 320, 3, step6::kSolo>, tileGrid<128, 320,
         step6::kSolo, 3>, b256, sh6, T320::kSmemBytes, T320::kABytes, T320::kStage, out, same1},
        {"step7 v8: 7c 6c, 132 blocks", kStep7, step7_rows_noroles<132>, rows7cGrid<132>, b256, sh6, T256::kSmemBytes,
         T256::kABytes, T256::kStage, out, same1},
        {"step7 v8: 7c 6c, 3 blocks", kStep7, step7_rows_noroles<3>, rows7cGrid<3>, b256, sh6, T256::kSmemBytes,
         T256::kABytes, T256::kStage, out, same1},
    };
}

// ---------------------------------------------------------------- the arrays and the runs
static void allocate(Array& a, int rows, int cols, bool writable)
{
    const size_t bytes = (size_t)rows * cols * 2, body = (bytes + kPage - 1) / kPage * kPage;
    a.rows = rows;
    a.cols = cols;
    a.writable = writable;
    a.mapBytes = kGuardBytes + body + kGuardBytes;
    a.map = (char*)mmap(nullptr, a.mapBytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (a.map == MAP_FAILED || mprotect(a.map + kGuardBytes, body, PROT_READ | PROT_WRITE)) {
        perror("EMU INTERNAL ERROR: mmap");
        exit(2);
    }
    a.p = (half*)(a.map + kGuardBytes + body - bytes);
    if ((uintptr_t)a.p % 256) {   // cudaMalloc's alignment; it holds at the shapes here (sizes multiples of 256 bytes)
        printf("EMU INTERNAL ERROR: %s is not 256-byte aligned\n", a.name);
        exit(2);
    }
}

static void release(Array& a)
{
    munmap(a.map, a.mapBytes);
    a.map = nullptr;
    a.p = nullptr;
}

static float rnd(unsigned& s)   // verif.cu's inputs
{
    s = s * 1664525u + 1013904223u;
    return ((s >> 9) & 0xffff) / 32768.0f - 1.0f;
}

constexpr double kTolerance = 2e-3;   // verif.cu's: max error / max |ref|

struct Totals { int runs = 0, failed = 0; double worst = 0; std::vector<std::string> failures; };

// One run: kernel k at M x N x K in one order. Prints its line; returns whether it passed. forward: k's C in the
// forward order; same: per class of kernels that give the same bits, the first of them to run at this shape and its C
// in the forward order (both set by the run that produces them, compared by the others).
struct SameBits { const Kernel* first; std::vector<uint16_t> c; };   // a class's first kernel to run at a shape, its C

static bool run(const Kernel& k, int M, int N, int K, Order order, std::vector<uint16_t>& forward,
                std::map<unsigned, SameBits>& same, const std::vector<double>& ref, double mref, Totals& totals)
{
    snprintf(g_label, sizeof g_label, "%-33s %4d x %4d x %3d  %-7s", k.label, M, N, K, kOrderName[order]);
    const size_t nC = (size_t)M * N;
    g_errors = 0;
    g_abort = false;
    g_banks.clear();
    g_cWrites.assign(nC, 0);
    g_cFirst.assign(nC, -1);
    g_cAgain = -1;
    memset(g_C.p, 0xff, nC * 2);   // NaN: an element left unwritten shows
    g_cluster = k.cluster;
    g_sh = k.shmem;
    g_shBytes = k.smemBytes;
    if (g_shBytes > kShArrayBytes) {
        printf("EMU INTERNAL ERROR: %s needs %d bytes of shared memory, the emulator has %d\n", k.label, g_shBytes,
               kShArrayBytes);
        exit(2);
    }
    g_bStageBytes = k.bStageBytes;
    g_stageBytes = k.stageBytes;
    g_kernel = k.fn;
    g_M = M, g_N = N, g_K = K;
    gridDim = k.grid(M, N);
    blockDim = k.block;
    g_tpb = k.block.x * k.block.y;
    g_threads = g_tpb * g_cluster;
    g_launchRegs = k.regs;
    std::mt19937 rng(1234 + order);
    g_wgMode = order == kForward ? 0 : order == kReverse ? 1 : 2;   // step 1's wgmma: early, late, mixed
    g_wgRng.seed(77 + order);
    g_tmaMode = order == kForward ? 0 : order == kReverse ? 1 : 2;   // step 2's TMA: late, early, mixed
    g_tmaRng.seed(55 + order);
    // The clusters one at a time (a cluster of one block for the kernels before step 4): in the grid's order (forward),
    // backwards (reverse), shuffled (random). No cluster of these kernels waits for another, so each order is a
    // schedule the GPU could run.
    const unsigned nBlocks = gridDim.x * gridDim.y;
    if (nBlocks % g_cluster) {
        printf("EMU INTERNAL ERROR: %s: %u blocks for clusters of %d\n", k.label, nBlocks, g_cluster);
        exit(2);
    }
    std::vector<unsigned> clusters(nBlocks / g_cluster);
    std::iota(clusters.begin(), clusters.end(), 0u);
    if (order == kReverse) std::reverse(clusters.begin(), clusters.end());
    if (order == kRandom) std::shuffle(clusters.begin(), clusters.end(), rng);
    bool complete = true;
    for (size_t n = 0; n < clusters.size() && complete; n++) {
        for (int r = 0; r < g_cluster; r++) {
            const unsigned b = clusters[n] * g_cluster + r;
            g_blockLin[r] = b;
            g_blockIdx[r] = {b % gridDim.x, b / gridDim.x, 0};
            g_shBlocks[r] = g_cluster == 1 ? k.shmem : (half*)g_clusterSh[r];
            memset(g_shBlocks[r], 0xff, kShArrayBytes);   // NaN: a read of shared memory not yet written shows
        }
        blockIdx = g_blockIdx[0];
        g_block = g_blockLin[0];
        g_sh = g_shBlocks[0];
        complete = run_cluster(order, rng);
        for (int r = 0; r < g_cluster; r++) {
            const unsigned char* canary = (const unsigned char*)g_shBlocks[r];
            for (int i = g_shBytes; i < kShArrayBytes; i++)
                if (canary[i] != 0xff) {
                    error("shared memory written at byte %d, outside the %d bytes of the launch (block %d)", i, g_shBytes,
                          g_blockLin[r]);
                    break;
                }
        }
    }
    // The result: the error against the reference, NaN, the order, the other variants, the writes to C, the wavefronts.
    std::vector<uint16_t> out(nC);
    memcpy(out.data(), g_C.p, nC * 2);
    double err = 0;
    long nan = 0, differ = 0, never = 0, again = 0, uncounted = 0, firstNever = -1;
    for (size_t i = 0; i < nC; i++) {
        half h;
        memcpy(&h, &out[i], 2);
        const double v = (double)h;
        if (std::isnan(v)) nan++;
        else err = std::max(err, std::fabs(v - ref[i]) / mref);
        if (g_cWrites[i] == 0) {
            never++;
            if (firstNever < 0) firstNever = i;
            if (out[i] != 0xffff) uncounted++;
        } else if (g_cWrites[i] > 1) again++;
    }
    if (order == kForward) forward = out;
    else
        for (size_t i = 0; i < nC; i++) differ += out[i] != forward[i];
    const Kernel* first = nullptr;   // another kernel of k's class that ran before it at this shape, if any
    long differFirst = 0;
    if (k.sameBits) {
        const auto it = same.find(k.sameBits);
        if (it == same.end()) same[k.sameBits] = {&k, out};   // k's forward order, its first run
        else if (it->second.first != &k) {
            first = it->second.first;
            for (size_t i = 0; i < nC; i++) differFirst += out[i] != it->second.c[i];
        }
    }
    std::string why;
    auto add = [&](const std::string& s) { why += (why.empty() ? "  [" : "; ") + s; };
    char s[200];
    if (!complete) add("block " + std::to_string(g_block) + " could not complete");
    if (g_errors) add(std::to_string(g_errors) + (g_errors == 1 ? " error" : " errors"));
    if (nan) add(std::to_string(nan) + " NaN in C");
    if (err > kTolerance) add("error above the tolerance");
    if (differ) add(std::to_string(differ) + " elements differ from the forward order");
    if (differFirst)
        add(std::to_string(differFirst) + " elements differ from " + first->label + " (the same bits expected)");
    if (never) {
        snprintf(s, sizeof s, "%ld elements of C never written, the first C[%ld][%ld]", never, firstNever / N,
                 firstNever % N);
        add(s);
    }
    if (again) {
        snprintf(s, sizeof s, "%ld elements of C written more than once, the first C[%ld][%ld] by blocks %d and %d",
                 again, g_cAgain / N, g_cAgain % N, g_cAgainBlocks[0], g_cAgainBlocks[1]);
        add(s);
    }
    if (uncounted) add(std::to_string(uncounted) + " elements of C written by an uninstrumented store");
    std::string banks;   // "STS.128 A 4.00 B 4.00  STS.32 1.00  LDS.128 4.00", then any other class
    std::vector<std::string> classes = {"STS.128 A", "STS.128 B", "STS.32", "LDS.128"};
    for (const auto& b : g_banks)
        if (std::find(classes.begin(), classes.end(), b.first) == classes.end()) classes.push_back(b.first);
    for (const std::string& cls : classes) {
        const auto it = g_banks.find(cls);
        if (it == g_banks.end() || !it->second.instr) continue;
        const double v = (double)it->second.wavefronts / it->second.instr;
        if (cls == "STS.128 B" && g_banks.count("STS.128 A"))
            snprintf(s, sizeof s, " B %.2f", v);
        else snprintf(s, sizeof s, "%s%s %.2f", banks.empty() ? "" : "  ", cls.c_str(), v);
        banks += s;
    }
    for (const Expect& e : k.expect) {   // meaningful only for a complete grid
        if (!complete) break;
        const auto it = g_banks.find(e.cls);
        const bool counted = it != g_banks.end() && it->second.instr;
        const double got = counted ? (double)it->second.wavefronts / it->second.instr : -1;
        if (got < 0) snprintf(s, sizeof s, "no %s counted", e.cls);
        else snprintf(s, sizeof s, "%s %.2f wavefronts per instruction, %.2f expected", e.cls, got, e.perInstr);
        if (std::fabs(got - e.perInstr) > 1e-9) add(s);
    }
    if (!why.empty()) why += "]";
    const bool ok = why.empty();
    if (g_errors > kShownErrors) printf("EMU ERROR: ... %ld more errors in this run\n", g_errors - kShownErrors);
    char errText[32];
    if (nan) snprintf(errText, sizeof errText, "nan     ");
    else snprintf(errText, sizeof errText, "%.2e", err);
    printf("%s  %-6s  err %s%s%s%s\n", g_label, ok ? "ok" : "FAILED", errText, banks.empty() ? "" : "  ", banks.c_str(),
           why.c_str());
    totals.runs++;
    if (ok) totals.worst = std::max(totals.worst, err);
    else {
        totals.failed++;
        snprintf(s, sizeof s, "%s at %d x %d x %d, %s", k.label, M, N, K, kOrderName[order]);
        totals.failures.push_back(s);
    }
    return ok;
}

// Step 6's rule (step6::choose without 7e, on the H100: 132 SMs, 66 clusters of two at once): at the judge's 17
// shapes, the kernel the rule names (the table `want` below); at every M (a multiple of 16) up to 4096 and N (a
// multiple of 64) up to 65536, a kernel that can take the shape (the bands at M <= 16 with J <= 8; step 5's tile,
// alone or in pairs, only with N a multiple of 256, alone only at M > 128; the pairs of 64 x 128 only at 64 < M <=
// 128) and never 7e's tile.
static bool check_rule(Totals& totals)
{
    struct Want { int M, N; step6::Kind kind; int J; };
    const Want want[] = {{16, 10240, step6::kBandsK, 2}, {16, 8192, step6::kBandsK, 1}, {16, 57344, step6::kBandsK, 7},
                         {128, 10240, step6::kT128, 0}, {128, 8192, step6::kT64Pair, 0}, {128, 57344, step6::kT256, 0},
                         {512, 10240, step6::kT256Solo, 0}, {512, 8192, step6::kT256Solo, 0},
                         {512, 57344, step6::kT256Solo, 0}, {2048, 10240, step6::kT256Solo, 0},
                         {2048, 8192, step6::kT256Solo, 0}, {2048, 57344, step6::kT256Solo, 0},
                         {4608, 5632, step6::kT256Solo, 0}};
    long bad = 0, n = 0;
    char first[200] = "";
    for (const Want& w : want) {
        const step6::Choice c = step6::choose(w.M, w.N, 132, 66, false);
        n++;
        if (c.kind != w.kind || (w.kind == step6::kBandsK && c.J != w.J))
            if (!bad++) snprintf(first, sizeof first, "%d x %d: kernel %d (J %d), %d (J %d) expected", w.M, w.N, c.kind,
                                 c.J, w.kind, w.J);
    }
    for (int M = 16; M <= 4096; M += 16)
        for (int N = 64; N <= 65536; N += 64) {
            const step6::Choice c = step6::choose(M, N, 132, 66, false);
            n++;
            const bool ok = c.kind == step6::kBandsK ? M <= 16 && c.J >= 1 && c.J <= 8
                          : c.kind == step6::kT256 ? N % 256 == 0
                          : c.kind == step6::kT256Solo ? N % 256 == 0 && M > 128
                          : c.kind == step6::kT320 || c.kind == step6::kT320Solo ? false
                          : c.kind == step6::kT64Pair ? M > 64 && M <= 128
                          : c.kind == step6::kT64Solo ? M <= 64 : true;
            if (!ok && !bad++) snprintf(first, sizeof first, "%d x %d: kernel %d (J %d) cannot take it", M, N, c.kind, c.J);
        }
    totals.runs++;
    if (bad) {
        totals.failed++;
        totals.failures.push_back("step 6's rule");
        printf("step6 rule, %ld shapes  FAILED  [%ld faults, the first: %s]\n", n, bad, first);
        return false;
    }
    printf("step6 rule, %ld shapes  ok  (the judge's shapes as the table wants; a kernel that can take every shape)\n", n);
    return true;
}

// Step 5's order of the tiles (5b's groups, and v2's rows): at every M (a multiple of 16) up to 8192 and N (a
// multiple of 256) up to 12288, and at the judge's shapes, the units 0 to UM UN - 1 take each pair of tiles once, and
// the pairs take each tile once (a pair's second tile may lie outside the matrix: its block writes nothing); and, for
// step 7's 7c, in groups of 1 to 16 rows, each of up to 40 x 100 units once.
static bool check_order(Totals& totals)
{
    std::vector<std::pair<int, int>> mn;
    for (int M = 16; M <= 8192; M += 16)
        for (int N = 256; N <= 12288; N += 256) mn.push_back({M, N});
    for (const int M : {16, 128, 512, 2048})
        for (const int N : {10240, 8192, 57344}) mn.push_back({M, N});
    mn.push_back({4608, 5632});
    long bad = 0;
    char first[200] = "";
    std::vector<int> seen, cover;
    for (const auto& x : mn) {
        const int M = x.first, N = x.second, tilesM = (M + 127) / 128, tilesN = N / 256;
        const bool alongM = M > 128;
        int UM, UN;
        step5::units(M, N, alongM, UM, UN);
        for (const bool grouped : {true, false}) {
            seen.assign((size_t)UM * UN, 0);
            cover.assign((size_t)tilesM * tilesN, 0);
            const int rows = alongM ? step5::kGroupRows / 2 : step5::kGroupRows;
            for (int u = 0; u < UM * UN; u++) {
                int um, un;
                step5::place(u, UM, UN, rows, grouped, um, un);
                if (um < 0 || um >= UM || un < 0 || un >= UN) {
                    if (!bad++) snprintf(first, sizeof first, "%d x %d, %s: unit %d at (%d, %d), outside %d x %d", M, N,
                                         grouped ? "groups" : "rows", u, um, un, UM, UN);
                    continue;
                }
                seen[(size_t)um * UN + un]++;
                for (int rank = 0; rank < 2; rank++) {
                    const int tm = alongM ? 2 * um + rank : um, tn = alongM ? un : 2 * un + rank;
                    if (tm < tilesM && tn < tilesN) cover[(size_t)tm * tilesN + tn]++;
                }
            }
            for (size_t i = 0; i < seen.size(); i++)
                if (seen[i] != 1 && !bad++)
                    snprintf(first, sizeof first, "%d x %d, %s: the pair at (%zu, %zu) taken %d times", M, N,
                             grouped ? "groups" : "rows", i / UN, i % UN, seen[i]);
            for (size_t i = 0; i < cover.size(); i++)
                if (cover[i] != 1 && !bad++)
                    snprintf(first, sizeof first, "%d x %d, %s: the tile at (%zu, %zu) taken %d times", M, N,
                             grouped ? "groups" : "rows", i / tilesN, i % tilesN, cover[i]);
        }
    }
    // step 7's 7c: groups of 1 to 16 rows, up to 40 x 100 units, each unit once
    for (int rows = 1; rows <= 16; rows++)
        for (int UM = 1; UM <= 40; UM++)
            for (int UN = 1; UN <= 100; UN++) {
                seen.assign((size_t)UM * UN, 0);
                for (int u = 0; u < UM * UN; u++) {
                    int um, un;
                    step5::place(u, UM, UN, rows, true, um, un);
                    if (um < 0 || um >= UM || un < 0 || un >= UN) {
                        if (!bad++)
                            snprintf(first, sizeof first, "groups of %d rows, %d x %d units: unit %d outside", rows, UM,
                                     UN, u);
                        continue;
                    }
                    seen[(size_t)um * UN + un]++;
                }
                for (size_t i = 0; i < seen.size(); i++)
                    if (seen[i] != 1 && !bad++)
                        snprintf(first, sizeof first, "groups of %d rows, %d x %d units: unit (%zu, %zu) taken %d "
                                 "times", rows, UM, UN, i / UN, i % UN, seen[i]);
            }
    totals.runs++;
    if (bad) {
        totals.failed++;
        totals.failures.push_back("step 5's order of the tiles");
        printf("step5 order of the tiles, %zu shapes  FAILED  [%ld faults, the first: %s]\n", mn.size(), bad, first);
        return false;
    }
    printf("step5 order of the tiles, %zu shapes  ok  (each pair of tiles to one unit, each tile once, in groups and "
           "in rows)\n", mn.size());
    return true;
}

static int summary(const Totals& t, bool stopped)
{
    printf("%d runs: %d ok, %d failed%s; worst error / max |ref| of the runs that passed %.2e (tolerance %.0e)\n",
           t.runs, t.runs - t.failed, t.failed, stopped ? " (-x: stopped at the first failure)" : "", t.worst,
           kTolerance);
    if (!t.failed) {
        printf("ALL OK\n");
        return 0;
    }
    for (size_t i = 0; i < t.failures.size() && i < 10; i++) printf("  failed: %s\n", t.failures[i].c_str());
    if (t.failures.size() > 10) printf("  failed: ... and %zu more\n", t.failures.size() - 10);
    printf("FAILED\n");
    return 1;
}

// Step 7's rule, on the H100 (132 SMs, 15 clusters of eight held at once, 8 GPCs): at the judge's 17 shapes,
// step7::choose's kernel of the table `want` below, 7b at 7, 8 and 9 clusters (16 x 28672, 16 x 32768, 16 x 36864),
// and with 4 clusters of eight held (16 x 10240); at every M (a multiple of 16) up to 4096 and N (a multiple of 64)
// up to 65536, with 4, 13 and 15 clusters of eight held, a kernel that can take the shape (7b's clusters one per
// GPC);
// 7e's rule (step6::choose with 7e, as step7.cu asks it): at the judge's shapes M > 128, the kernel of the table
// `want6` below, and at every M > 128 up to 4096 and N up to 65536, a kernel that can take the shape (step 5's tile
// alone only with N a multiple of 256); and 7c's rule (below).
static bool check_rule7(Totals& totals)
{
    struct Want { int M, N, clusters8; step7::Kind kind; int J; };
    const Want want[] = {{16, 10240, 15, step7::kBands16Wide, 2}, {16, 8192, 15, step7::kBands16Wide, 1},
                         {16, 57344, 15, step7::kBands16, 7},     {16, 32768, 15, step7::kBands16Wide, 4},
                         {16, 36864, 15, step7::kBands16, 5},     {16, 28672, 15, step7::kBands16Wide, 4},
                         {16, 10240, 4, step7::kBands16, 2},      {128, 10240, 15, step7::kBands128, 2},
                         {128, 8192, 15, step7::kBands64Pairs, 2}, {128, 57344, 15, step7::kStep6, 0},
                         {512, 10240, 15, step7::kStep6, 0},      {512, 8192, 15, step7::kStep6, 0},
                         {512, 57344, 15, step7::kStep6, 0},      {2048, 10240, 15, step7::kStep6, 0},
                         {2048, 8192, 15, step7::kStep6, 0},      {2048, 57344, 15, step7::kStep6, 0},
                         {4608, 5632, 15, step7::kStep6, 0}};
    long bad = 0, n = 0;
    char first[200] = "";
    for (const Want& w : want) {
        n++;
        const step7::Choice c = step7::choose(w.M, w.N, 132, w.clusters8);
        if ((c.kind != w.kind || c.J != w.J) && !bad++)
            snprintf(first, sizeof first, "%d x %d, %d clusters of eight: kernel %d (J %d), expected %d (J %d)", w.M,
                     w.N, w.clusters8, c.kind, c.J, w.kind, w.J);
    }
    for (const int clusters8 : {4, 13, 15})
        for (int M = 16; M <= 4096; M += 16)
            for (int N = 64; N <= 65536; N += 64) {
                n++;
                const step7::Choice c = step7::choose(M, N, 132, clusters8);
                const int J7b = 2 * c.J < 8 ? 2 * c.J : 8, bands7b = (N + 64 * J7b - 1) / (64 * J7b);
                const int gpcs = (clusters8 + 1) / 2;
                const bool bands16 = M <= 16 && c.J >= 1 && c.J <= 8;
                const bool ok = c.kind == step7::kBands16 ? bands16
                              : c.kind == step7::kBands16Wide ? bands16 && (bands7b + 7) / 8 <= gpcs
                              : c.kind == step7::kBands64Pairs || c.kind == step7::kBands128 ? M <= 128 && c.J == 2
                              : true;
                if (!ok && !bad++)
                    snprintf(first, sizeof first, "%d x %d, %d clusters of eight: kernel %d (J %d) cannot take it", M,
                             N, clusters8, c.kind, c.J);
            }
    // 7e's rule
    struct Want6 { int M, N; step6::Kind kind; };
    const Want6 want6[] = {{512, 10240, step6::kT320Solo}, {512, 8192, step6::kT256Solo},
                           {512, 57344, step6::kT256Solo}, {2048, 10240, step6::kT256Solo},
                           {2048, 8192, step6::kT256Solo}, {2048, 57344, step6::kT320}, {4608, 5632, step6::kT256Solo}};
    for (const Want6& w : want6) {
        n++;
        const step6::Choice c = step6::choose(w.M, w.N, 132, 66, true);
        if (c.kind != w.kind && !bad++)
            snprintf(first, sizeof first, "7e's rule at %d x %d: kernel %d, expected %d", w.M, w.N, c.kind, w.kind);
    }
    for (int M = 144; M <= 4096; M += 16)
        for (int N = 64; N <= 65536; N += 64) {
            n++;
            const step6::Choice c = step6::choose(M, N, 132, 66, true);
            const bool ok = c.kind == step6::kT320 || c.kind == step6::kT320Solo ||
                            (c.kind == step6::kT256Solo && N % 256 == 0);
            if (!ok && !bad++)
                snprintf(first, sizeof first, "7e's rule at %d x %d: kernel %d cannot take it", M, N, c.kind);
        }
    // 7c's rule: at the judge's 6c shapes (table below, 132 blocks); and at every 1 to 40 x 1 to 100 tiles, with 132
    // and 3 blocks held: G at most 16, the rows and the blocks, R at most the blocks and the tiles, R a whole number of
    // groups unless cut to the tiles, the rounds kept, and no larger G that keeps them
    struct Rows { int UM, UN, G, R; };
    const Rows rows[] = {{36, 22, 12, 132}, {16, 40, 16, 128}, {16, 32, 16, 128}, {4, 32, 4, 128}, {4, 224, 4, 132}};
    for (const Rows& w : rows) {
        n++;
        int G, R;
        step7::rows7c(w.UM, w.UN, 132, G, R);
        if ((G != w.G || R != w.R) && !bad++)
            snprintf(first, sizeof first, "7c at %d x %d tiles: G %d R %d, expected G %d R %d", w.UM, w.UN, G, R, w.G,
                     w.R);
    }
    for (const int res : {132, 3})
        for (int UM = 1; UM <= 40; UM++)
            for (int UN = 1; UN <= 100; UN++) {
                n++;
                int G, R;
                step7::rows7c(UM, UN, res, G, R);
                const int tiles = UM * UN, rounds = (tiles + res - 1) / res, top = UM < 16 ? UM : 16;
                bool ok = G >= 1 && G <= top && G <= res && R >= 1 && R <= res && R <= tiles &&
                          (R == tiles || R % G == 0) && (tiles + R - 1) / R == rounds;
                for (int g = G + 1; g <= top; g++) {
                    const int r = g * (res / g);
                    if (r >= 1 && (tiles + r - 1) / r == rounds) ok = false;
                }
                if (!ok && !bad++)
                    snprintf(first, sizeof first, "7c at %d x %d tiles, %d blocks: G %d R %d", UM, UN, res, G, R);
            }
    totals.runs++;
    if (bad) {
        totals.failed++;
        totals.failures.push_back("step 7's rule");
        printf("step7 rule, %ld checks  FAILED  [%ld faults, the first: %s]\n", n, bad, first);
        return false;
    }
    printf("step7 rule, %ld checks  ok  (the judge's shapes as the table wants; a kernel that can take every "
           "shape)\n", n);
    return true;
}

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);   // whole lines reach the output even if a fault ends the program
    unsigned sel = 0;
    bool stopAtFirst = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-x") stopAtFirst = true;
        else if (a == "step0") sel |= kStep0;
        else if (a == "step1") sel |= kStep1;
        else if (a == "step2") sel |= kStep2;
        else if (a == "step3") sel |= kStep3;
        else if (a == "step4") sel |= kStep4;
        else if (a == "step5") sel |= kStep5;
        else if (a == "step6") sel |= kStep6;
        else if (a == "step7") sel |= kStep7;
        else if (a == "all") sel |= kAll;
        else {
            printf("usage: %s [step0|step1|step2|step3|step4|step5|step6|step7|all]... [-x]\n"
                   "  step0  step 0, nvidia_sample.cuh's compute_gemm (132 and 3 blocks), and its simple_wmma_gemm\n"
                   "  step1  step 1's v0, v1 and v3 (step1.cuh; v2 is step 0's compute_gemm), with the same bits\n"
                   "  step2  step 2's v0, v1, v3 and v4 (step2.cuh; v2 is step 1's v0), with step 1's bits\n"
                   "  step3  step 3's v0 (step 2's kernel at 4 stages; v1 is step 2's v0), with step 1's bits\n"
                   "  step4  step 4's v0 (step4.cuh; clusters of two blocks run together; v1 is step 3's v0)\n"
                   "  step5  step 5's v0 to v2 (step5.cuh; v3 is step 4's v0), and the order of its tiles\n"
                   "  step6  step 6's bands (6a) and tiles (6b, 6c), step 5's tile, and rule (step6.cuh)\n"
                   "  step7  step 7's kernels, with warp specialization and its switches (7f to 7l) or without:\n"
                   "         the bands (7a, 7b, 7d), 7e's tile, step 6's and step 5's tiles, 7c's order and the rule\n"
                   "         (step7.cuh, step6.cuh)\n"
                   "  all    every kernel once (the default)\n"
                   "  -x     stop at the first failed run\n", argv[0]);
            return 2;
        }
    }
    if (!sel) sel = kAll;

    // The threads' stacks, each above a PROT_NONE page; a stack for the fault handler.
    g_stacks = (char*)mmap(nullptr, kMaxThreads * (kPage + kStackBytes), PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (g_stacks == MAP_FAILED) { perror("EMU INTERNAL ERROR: mmap"); return 2; }
    for (int t = 0; t < kMaxThreads; t++) mprotect(g_stacks + t * (kPage + kStackBytes), kPage, PROT_NONE);
    static char altStack[64 << 10];
    stack_t ss = {};
    ss.ss_sp = altStack;
    ss.ss_size = sizeof altStack;
    sigaltstack(&ss, nullptr);
    struct sigaction sa = {};
    sa.sa_sigaction = on_fault;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    sigaction(SIGFPE, &sa, nullptr);

    const std::vector<Kernel> all = kernels();
    printf("gemm-ladder emulator, Hopper rung:%s%s%s%s%s%s%s%s; the shapes of verif.cu; each run in forward, reverse and "
           "random order (wavefronts per shared instruction after err; step 1's wgmma early in the forward order, late "
           "in the reverse, chain by chain at random in the random; step 2's TMA loads late, early, load by load at "
           "random; step 5's TMA stores early, late, store by store at random)\n",
           sel & kStep0 ? " step0" : "", sel & kStep1 ? " step1" : "", sel & kStep2 ? " step2" : "",
           sel & kStep3 ? " step3" : "", sel & kStep4 ? " step4" : "", sel & kStep5 ? " step5" : "",
           sel & kStep6 ? " step6" : "", sel & kStep7 ? " step7" : "");
    Totals totals;
    if ((sel & kStep5) && !check_order(totals) && stopAtFirst) return summary(totals, true);
    if ((sel & kStep6) && !check_rule(totals) && stopAtFirst) return summary(totals, true);
    if ((sel & kStep7) && !check_rule7(totals) && stopAtFirst) return summary(totals, true);
    // The first six shapes of verif.cu: M = 16 (one tile cut by M), a cut tile after a full one, tiles in 1 to 3 rows,
    // 42 and 84 tiles (the persistent grid of 3 blocks goes through many), 10 rows of tiles; and, for the rings of
    // steps 1 and 2 (2 and 3 stages), three more with 4, 10 and 16 slices of K64 (with 1 to 3 slices, the slices of
    // the shapes above, the stages never come round): 10 slices is 3 turns of 3 stages and one more. Step 5: one more,
    // 2320 x 768 x 64, 19 rows of tiles (10 rows of pairs, the last pair cut by M): two groups of 5b's order, the
    // second one from the last column back.
    constexpr unsigned kRings = kStep1 | kStep2 | kStep3 | kStep4 | kStep5 | kStep6 | kStep7;
    struct Shape { int M, N, K; unsigned in; };
    const Shape shapes[] = {{16, 256, 64, kAll},    {144, 256, 64, kAll},    {128, 512, 128, kAll},
                            {144, 512, 256, kRings}, {208, 512, 192, kAll},  {144, 512, 640, kRings},
                            {144, 512, 320, kStep3 | kStep4 | kStep5},  {144, 512, 832, kStep3 | kStep4 | kStep5},
                            {16, 1024, 1024, kRings}, {272, 5376, 128, kAll}, {1168, 512, 64, kAll},
                            {2320, 768, 64, kStep5},
                            {16, 512, 448, kStep6 | kStep7},  {16, 512, 512, kStep6 | kStep7},  {16, 512, 576, kStep6 | kStep7},
                            {16, 512, 1600, kStep6 | kStep7}, {16, 960, 128, kStep6 | kStep7},  {16, 960, 192, kStep6 | kStep7},
                            {16, 960, 256, kStep6 | kStep7},  {16, 960, 640, kStep6 | kStep7},
                            {128, 256, 448, kStep6 | kStep7}, {128, 256, 512, kStep6 | kStep7}, {128, 256, 576, kStep6 | kStep7},
                            {128, 256, 1600, kStep6 | kStep7}, {128, 256, 320, kStep6 | kStep7}, {128, 256, 384, kStep6 | kStep7},
                            {128, 256, 1216, kStep6 | kStep7}, {256, 384, 128, kStep6}, {256, 384, 192, kStep6},
                            {256, 384, 256, kStep6}, {272, 704, 640, kStep6},
                            // step 7: the bands' rings at M = 16 and 128 (the kStep6 | kStep7 shapes above), the
                            // 64-row pairs and the 128-row bands cut by M; step 5's tile along N with a cluster
                            // through two units (2 clusters, 4 units of 5 slices: as 46 of 66 at gateup_m128)
                            {80, 256, 448, kStep7}, {80, 512, 192, kStep7}, {112, 256, 448, kStep7},
                            {16, 2048, 320, kStep7}};
    for (const Shape& shape : shapes) {
        if (!(shape.in & sel)) continue;
        const int M = shape.M, N = shape.N, K = shape.K;
        allocate(g_A, M, K, false);
        allocate(g_B, K, N, false);
        allocate(g_C, M, N, true);
        unsigned seed = 7;
        for (size_t i = 0; i < (size_t)M * K; i++) g_A.p[i] = (half)rnd(seed);
        for (size_t i = 0; i < (size_t)K * N; i++) g_B.p[i] = (half)rnd(seed);
        std::vector<double> ref((size_t)M * N, 0.0);   // in double, from the fp16 inputs, as verif.cu
        for (int i = 0; i < M; i++)
            for (int k = 0; k < K; k++) {
                const double a = (double)g_A.p[(size_t)i * K + k];
                for (int j = 0; j < N; j++) ref[(size_t)i * N + j] += a * (double)g_B.p[(size_t)k * N + j];
            }
        double mref = 0;
        for (const double r : ref) mref = std::max(mref, std::fabs(r));
        for (Array* in : {&g_A, &g_B})   // the inputs are read-only from here on
            mprotect(in->map + kGuardBytes, in->mapBytes - 2 * kGuardBytes, PROT_READ);
        std::map<unsigned, SameBits> same;   // per class of kernels with the same bits: its first one here, its C
        for (const Kernel& k : all) {
            if (!(k.in & sel & shape.in) || M > k.maxM || N % k.nMultiple) continue;
            std::vector<uint16_t> forward;
            for (const Order order : {kForward, kReverse, kRandom})
                if (!run(k, M, N, K, order, forward, same, ref, mref, totals) && stopAtFirst)
                    return summary(totals, true);
        }
        release(g_A);
        release(g_B);
        release(g_C);
    }
    return summary(totals, false);
}
