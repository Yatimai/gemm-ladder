// The host emulator of the Turing rung (README.md in this directory). It compiles the kernels' own source,
// nvidia_sample.cuh (step 0's compute_gemm, and simple_wmma_gemm, the reference point), step1.cuh to step6.cuh, as
// C++ against the headers in shim/, and runs each kernel on the CPU at the shapes of verif.cu, in three orders of its
// threads. It checks the result against a reference in double, every global and shared access, that each element of C
// is written exactly once, the shared-memory wavefronts against the values the headers state, that the variants of
// steps 3 to 6 give the same bits as step 2 where their headers say so, and that step 6's counters are back at zero
// after each run; the order of the tiles of steps 3 and 5 on its own, at every shape up to 8192 x 12288; step 5's
// choice of tile and step 6's plan at the judge's 17 shapes.
//   build (in turing/emu): g++ -std=c++17 -O2 -march=native -I.. -Ishim emu.cpp -o emu
//   run:                   ./emu [step0|...|step6|all]... [-x]   (default: all; -x: stop at the first failure)
// Exit code 0 only if every run passes; 1 if a run fails; 2 on a usage or internal error.
#define GEMM_LADDER_EMULATE   // step2.cuh: ldmatrix and mma.sync as C++ branches; step3.cuh: plain loads and stores
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

// ---------------------------------------------------------------- what the shim headers and the kernels refer to
emu_dim3 threadIdx, blockIdx, gridDim, blockDim;

// The kernels' dynamic shared memory (extern __shared__ half shmem[], one per namespace). A launch uses the first
// bytes; the rest of the 128 KB keeps the NaN fill written before each block, a canary checked after it.
constexpr int kShArrayBytes = 128 << 10;
namespace nvidia_sample { alignas(4096) half shmem[kShArrayBytes / 2]; }
namespace step1 { alignas(4096) half shmem[kShArrayBytes / 2]; }
namespace step2 { alignas(4096) half shmem[kShArrayBytes / 2]; }
namespace step3 { alignas(4096) half shmem[kShArrayBytes / 2]; }
namespace step4 { alignas(4096) half shmem[kShArrayBytes / 2]; }
namespace step5 { alignas(4096) half shmem[kShArrayBytes / 2]; }
namespace step6 { alignas(4096) half shmem[kShArrayBytes / 2]; }

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
static Array g_P{"parts"};               // step 6's parts (6c, 6d), sized to the launch (see emu6 below)
static half* g_sh;                       // the launch's shared memory (the kernel's namespace's shmem)
static int g_shBytes, g_bStageBytes;     // its size; where B's stages begin (a copy's store below it is to A)
static int g_block;                      // the running block (linear index)
static std::vector<int> g_gridBarriers;  // per block of the run: its arrivals at a grid barrier (step 6's waves)
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

// step6.cuh's grid barrier, emulated: the block's thread 0 counts the block's arrival (between two block barriers).
namespace step6 { void emu_grid_barrier() { g_gridBarriers[g_block]++; } }

// ---------------------------------------------------------------- threads: coroutines, barriers, warp exchanges
constexpr int kMaxThreads = 512;
constexpr size_t kStackBytes = 256 << 10;
enum State { kRunnable, kAtBarrier, kAtExchange, kExited };
struct Access { bool store; int bytes; long offset; };   // a shared access (offset in bytes)
struct Thread {
    ucontext_t ctx;
    State state;
    int xbytes;               // at a warp exchange: the size of its payload
    unsigned char xin[64];    // the payload
    std::vector<Access> log;  // its shared accesses since the last barrier, in program order
};
static Thread g_thr[kMaxThreads];
static ucontext_t g_sched;
static int g_cur = -1, g_threads;   // the running thread (linear index in its block); threads per block
static char* g_stacks;              // kMaxThreads x (a PROT_NONE page, then a stack of kStackBytes)
static unsigned char g_xall[kMaxThreads / 32][32 * 64];   // per warp: the 32 payloads of its last exchange
static void (*g_kernel)(const half*, const half*, half*, int, int, int);
static int g_M, g_N, g_K;
static bool g_abort;                // the block cannot go on (its lanes wait at different warp-wide instructions)

static const char* who()
{
    static char s[64];
    if (g_cur >= 0) snprintf(s, sizeof s, "block %d, thread %d", g_block, g_cur);
    else snprintf(s, sizeof s, "block %d", g_block);
    return s;
}

void __syncthreads()
{
    g_thr[g_cur].state = kAtBarrier;
    swapcontext(&g_thr[g_cur].ctx, &g_sched);
}

int emu_lane() { return g_cur % 32; }

// The meeting point of a warp-wide instruction (step2.cuh): the lane leaves its payload and waits; once the 32 lanes
// are there, the scheduler gathers the payloads (and accounts for the instruction), then each lane resumes with all 32.
void step2::emu_warp_exchange(const void* mine, void* all, int bytes)
{
    const int t = g_cur;
    Thread& th = g_thr[t];
    if (bytes > (int)sizeof th.xin) { printf("EMU INTERNAL ERROR: a warp exchange of %d bytes\n", bytes); exit(2); }
    memcpy(th.xin, mine, bytes);
    th.xbytes = bytes;
    th.state = kAtExchange;
    swapcontext(&th.ctx, &g_sched);
    memcpy(all, g_xall[t / 32], 32 * (size_t)bytes);
}

static void thread_main()
{
    g_kernel(g_A.p, g_B.p, g_C.p, g_M, g_N, g_K);
    g_thr[g_cur].state = kExited;
}

static void resume(int t)   // runs thread t until it reaches a barrier, a warp exchange or its end
{
    g_cur = t;
    threadIdx = {t % blockDim.x, t / blockDim.x, 0};
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
static void count_shared_accesses()
{
    for (int w = 0; w < g_threads / 32; w++) {
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
                if (first[k].store && first[k].bytes == 16) cls += off[0] < g_bStageBytes ? " A" : " B";
                Count& c = g_banks[cls];
                c.instr++;
                c.wavefronts += wavefronts(off, 32, first[k].bytes);
            }
        for (int l = 0; l < 32; l++) lane[l].log.clear();
    }
}

alignas(16) static half g_nanRow[8];   // an invalid ldmatrix row is read from here (NaN), once reported

// An ldmatrix.x4: its 32 row addresses (lane l gives row l % 8 of matrix l / 8), each 16-byte aligned inside the
// launch's shared memory; their wavefronts.
static void ldmatrix_rows(int w)
{
    long off[32];
    for (int l = 0; l < 32; l++) {
        const char* p;
        memcpy(&p, g_xall[w] + l * sizeof p, sizeof p);
        off[l] = p - (const char*)g_sh;
        const bool inside = off[l] >= 0 && off[l] + 16 <= g_shBytes;
        if (!inside || off[l] % 16) {
            error("ldmatrix in warp %d of block %d: the row address of lane %d is %s", w, g_block, l,
                  inside ? "not 16-byte aligned" : "outside the launch's shared memory");
            p = (const char*)g_nanRow;
            memcpy(g_xall[w] + l * sizeof p, &p, sizeof p);
        }
    }
    Count& c = g_banks["LDSM.x4"];
    c.instr++;
    for (int q = 0; q < 4; q++) c.wavefronts += wavefronts(off + 8 * q, 8, 16);
}

// When the 32 lanes of warp w wait at a warp exchange: performs it and releases them. An exchange of a pointer is an
// ldmatrix (each lane gives its row address); the other one of step2.cuh, three registers per lane, an mma.sync.
static bool complete_exchange(int w)
{
    Thread* lane = &g_thr[32 * w];
    for (int l = 0; l < 32; l++)
        if (lane[l].state != kAtExchange) return false;
    const int bytes = lane[0].xbytes;
    for (int l = 0; l < 32; l++) {
        if (lane[l].xbytes != bytes) {
            error("warp %d of block %d: its lanes wait at different warp-wide instructions", w, g_block);
            g_abort = true;
            return false;
        }
        memcpy(g_xall[w] + l * bytes, lane[l].xin, bytes);
    }
    if (bytes == sizeof(const half*)) ldmatrix_rows(w);
    for (int l = 0; l < 32; l++) lane[l].state = kRunnable;
    return true;
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

// Runs block g_block. A phase runs every thread from the barrier it waited at (or its start) to the next barrier (or
// its end). Forward and reverse: warp by warp (0 up, or the last one down), each through its whole phase, its lanes in
// order (or reversed), a warp exchange completing as soon as its 32 lanes have reached it. Random: a thread drawn at
// random runs until it blocks. False if the block cannot complete.
static bool run_block(Order order, std::mt19937& rng)
{
    for (int t = 0; t < g_threads; t++) {
        start_thread(&g_thr[t].ctx, g_stacks + t * (kPage + kStackBytes) + kPage);
        g_thr[t].state = kRunnable;
        g_thr[t].log.clear();
    }
    const int warps = g_threads / 32;
    std::vector<int> ready;
    for (;;) {
        if (order == kRandom) {
            ready.clear();
            for (int t = 0; t < g_threads; t++) ready.push_back(t);
            while (!ready.empty() && !g_abort) {
                const size_t i = rng() % ready.size();
                const int t = ready[i];
                ready[i] = ready.back();
                ready.pop_back();
                resume(t);
                if (g_thr[t].state == kAtExchange && complete_exchange(t / 32))
                    for (int l = 0; l < 32; l++) ready.push_back(t / 32 * 32 + l);
            }
        } else {
            for (int k = 0; k < warps && !g_abort; k++) {
                const int w = order == kForward ? k : warps - 1 - k;
                do
                    for (int j = 0; j < 32; j++) {
                        const int t = 32 * w + (order == kForward ? j : 31 - j);
                        if (g_thr[t].state == kRunnable) resume(t);
                    }
                while (complete_exchange(w));
            }
        }
        if (g_abort) return false;
        int atBarrier = 0, atExchange = 0, exited = 0;
        for (int t = 0; t < g_threads; t++) {
            atBarrier += g_thr[t].state == kAtBarrier;
            atExchange += g_thr[t].state == kAtExchange;
            exited += g_thr[t].state == kExited;
        }
        if (atExchange) {
            error("block %d: %d threads wait at a warp-wide instruction that the rest of their warp does not reach "
                  "(%d at the barrier, %d exited)", g_block, atExchange, atBarrier, exited);
            return false;
        }
        if (exited && exited < g_threads) {
            error("block %d: a barrier reached by %d threads while %d have exited", g_block, atBarrier, exited);
            return false;
        }
        count_shared_accesses();
        if (exited == g_threads) return true;
        for (int t = 0; t < g_threads; t++) g_thr[t].state = kRunnable;
    }
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
                  off + bytes > g_shBytes ? "past its end" : "misaligned", g_shBytes, who());
            return false;
        }
        if (log && g_cur >= 0) g_thr[g_cur].log.push_back({store, bytes, off});
        return true;
    }
    for (Array* a : {&g_A, &g_B, &g_C, &g_P}) {
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
    for (const Array* r : {&g_A, &g_B, &g_C, &g_P}) {
        if (!r->p) continue;
        if (a >= r->lo() - kGuardBytes && a < r->lo())
            snprintf(where, sizeof where, "in the guard region, %ld bytes before %s", (long)(r->lo() - a), r->name);
        else if (a >= r->hi() && a < r->hi() + kGuardBytes)
            snprintf(where, sizeof where, "in the guard region, %ld bytes past the end of %s", (long)(a - r->hi()),
                     r->name);
        else if (a >= r->lo() && a < r->hi())
            snprintf(where, sizeof where, "in %s, which the kernels may not write", r->name);
    }
    for (int t = 0; t < kMaxThreads; t++) {
        const char* guard = g_stacks + t * (kPage + kStackBytes);
        if (a >= guard && a < guard + kPage) {
            snprintf(where, sizeof where, "past the stack of thread %d's coroutine (raise kStackBytes)", t);
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

// ---------------------------------------------------------------- step 6's workspace
// step6.cu's, at the sizes of the shapes here: 6c's and 6d's parts in g_P, an array between guard regions like A, B
// and C, sized to the launch (6c: 16 x BN halves per block; 6d: 2 x M x N) and filled with NaN before each run, so that
// a part read before it is written shows in C and one written or read out of place is reported; the counters of 6c's
// bands and 6d's tiles, and 6b's barrier words, zero at the start, which each run must leave so (Kernel::dirty).
static void allocate(Array& a, int rows, int cols, bool writable);
static void release(Array& a);
namespace emu6 {
constexpr int kCounters = 128;
unsigned g_bandCounts[kCounters], g_splitCounts[kCounters], g_sync[2];
template <int BM, int BN> void waves(const half* A, const half* B, half* C, int M, int N, int K)
{
    step6::gemm_waves<BM, BN, true>(A, B, C, M, N, K, g_sync);
}
void counters(int needed)   // the counters a launch indexes: within the arrays (they are not guarded)
{
    if (needed > kCounters) {
        printf("EMU INTERNAL ERROR: %d counters needed, %d available (raise emu6::kCounters)\n", needed, kCounters);
        exit(2);
    }
}
template <int BM, int BN> void split(const half* A, const half* B, half* C, int M, int N, int K)
{
    counters((M + BM - 1) / BM * (N / BN));
    step6::gemm_split<BM, BN, true>(A, B, C, M, N, K, g_P.p, g_splitCounts);
}
template <int BN, int S> void bands(const half* A, const half* B, half* C, int M, int N, int K)
{
    counters(N / BN);
    const int steps = K / step6::BandGeometry<BN>::kBK;
    step6::gemm_bands<BN, (S > 1)>(A, B, C, M, N, K, steps / S, steps % S, g_P.p, g_bandCounts);
}
template <int BM, int BN> emu_dim3 grid_split(int M, int N)   // two blocks per tile
{
    return emu_dim3{(unsigned)((M + BM - 1) / BM * (N / BN) * 2), 1, 1};
}
template <int BN, int S> emu_dim3 grid_bands(int, int N) { return emu_dim3{(unsigned)S, (unsigned)(N / BN), 1}; }
// The shapes each kernel runs at: the rung's (M a multiple of 16), 6d's (K of 128 too), 6c's band (M <= 16, N a
// multiple of its width, at least 3 steps per slice: the kernel's contract).
bool rung(int M, int, int) { return M % 16 == 0; }
bool halves(int M, int, int K) { return M % 16 == 0 && K % 128 == 0; }
template <int BN, int S> bool band(int M, int N, int K)
{
    const int bk = step6::BandGeometry<BN>::kBK;
    return M <= 16 && N % BN == 0 && K % bk == 0 && K / bk / S >= 3;
}
void parts(int rows, int cols)   // the launch's parts, NaN
{
    if (g_P.p) release(g_P);
    allocate(g_P, rows, cols, true);
    memset(g_P.p, 0xff, (size_t)rows * cols * 2);
}
void parts_split() { parts(2 * g_M, g_N); }                                              // 6d: 2 x M x N
template <int BN> void parts_bands() { parts(gridDim.x * gridDim.y * 16, BN); }        // 6c: 16 x BN per block
int dirty()   // the counters left non-zero by a run, then put back to zero for the next one
{
    int n = g_sync[0] != 0;
    for (unsigned& c : g_bandCounts) n += c != 0, c = 0;
    for (unsigned& c : g_splitCounts) n += c != 0, c = 0;
    g_sync[0] = 0;
    return n;
}
}  // namespace emu6

// ---------------------------------------------------------------- the kernels
enum { kStep0 = 1, kStep1 = 2, kStep2 = 4, kStep3 = 8, kStep4 = 16, kStep5 = 32, kStep6 = 64 };
constexpr unsigned kAll = kStep0 | kStep1 | kStep2 | kStep3 | kStep4 | kStep5 | kStep6;
using KernelFn = void (*)(const half*, const half*, half*, int, int, int);
struct Expect { const char* cls; double perInstr; };
struct Kernel {
    const char* label;
    unsigned in;                    // the selections that run it (step0 to step6)
    KernelFn fn;
    emu_dim3 (*grid)(int M, int N);
    emu_dim3 block;
    half* shmem;
    int smemBytes, bStageBytes;     // the launch's dynamic shared memory; where B's stages begin in it
    std::vector<Expect> expect;     // wavefronts per shared instruction, as the headers state them
    unsigned sameBits = 0;          // a class of kernels that all give the same C, bit for bit, at a shape; 0: none
    bool (*accepts)(int M, int N, int K) = nullptr;   // the shapes it runs at (nullptr: every shape of its selections)
    void (*before)() = nullptr;     // before each run (step 6: its parts filled with NaN)
    int (*dirty)() = nullptr;       // after each run, the words of its state left non-zero (step 6: its counters)
};

static std::vector<Kernel> kernels()
{
    namespace s0 = nvidia_sample;
    using G1a = step1::Geometry<256, true>;    // v0, v1
    using G1b = step1::Geometry<256, false>;   // v2
    using G1c = step1::Geometry<128, true>;    // v3
    using G1d = step1::Geometry<128, false>;   // v4
    using G2s = step2::Geometry<true>;         // v0
    using G2p = step2::Geometry<false>;        // v1
    using G3 = step3::G;                       // step 3's v0 to v3: step 2's v0 stages
    using G4 = step4::G;                       // step 4's v0, v1: step 2's v0 stages
    using G5a = step5::Geometry<64, 128>;      // step 5's four tiles
    using G5b = step5::Geometry<64, 256>;
    using G5c = step5::Geometry<128, 128>;
    using G5d = step5::Geometry<128, 256>;
    using GB128 = step6::BandGeometry<128>;    // 6c's two widths
    using GB256 = step6::BandGeometry<256>;
    auto tiles256 = [](int M, int N) { return emu_dim3{(unsigned)((M + 127) / 128 * (N / 256)), 1, 1}; };
    auto tiles128 = [](int M, int N) { return emu_dim3{(unsigned)((M + 127) / 128 * (N / 128)), 1, 1}; };
    auto tiles64x128 = [](int M, int N) { return emu_dim3{(unsigned)((M + 63) / 64 * (N / 128)), 1, 1}; };
    auto tiles64x256 = [](int M, int N) { return emu_dim3{(unsigned)((M + 63) / 64 * (N / 256)), 1, 1}; };
    auto sms = [](int, int) { return emu_dim3{40, 1, 1}; };    // a persistent grid on the T4's 40 SMs, as verif.cu
    auto three = [](int, int) { return emu_dim3{3, 1, 1}; };   // so that each block also goes through several tiles
    auto simple = [](int M, int N) { return emu_dim3{(unsigned)((M + 63) / 64), (unsigned)(N / 64), 1}; };
    const emu_dim3 b256 = {256, 1, 1}, b128 = {128, 1, 1}, b64 = {64, 1, 1}, b128x4 = {128, 4, 1};
    // The headers' bank patterns. The copy's STS.128 of A: two-way on rows of 96 bytes (K32 + 16 halves of padding,
    // step 1's double buffer: a quarter warp stores into two rows whose groups overlap), conflict-free on rows of 160
    // bytes (K64 + padding: step 0, v2, v4) and on step 2's swizzled rows of 64 bytes; B's stores and the epilogue's
    // LDS.128 and STS.32: conflict-free; LDSM.x4: two-way with the padding (v1), conflict-free with the swizzle (v0).
    // Steps 3 and 4 keep step 2's swizzled stages, copy and epilogue: step 3 changes no shared access, and each matrix
    // of 4a's LDSM is, as in step 2, 8 rows of one chunk (the first row a multiple of 8): step 2's v0 pattern in all
    // their variants. Step 5's four tiles keep that pattern: the rows of B's stages and of the output buffer change
    // length (256 or 512 bytes; 272 or 528 for the output), not the swizzle nor what a quarter warp touches. Step 6
    // keeps step 5's tiles; 6c's bands apply step 2's swizzle rule to their rows of 32 or 64 bytes of A and 256 or 512
    // of B, and pad the output's rows by 8 halves: the same counts (step6.cuh).
    const std::vector<Expect> pad160 = {{"STS.128 A", 4}, {"STS.128 B", 4}, {"LDS.128", 4}};
    const std::vector<Expect> pad96 = {{"STS.128 A", 8}, {"STS.128 B", 4}, {"LDS.128", 4}};
    const std::vector<Expect> swizzle = {{"LDSM.x4", 4}, {"STS.128 A", 4}, {"STS.128 B", 4}, {"STS.32", 1},
                                         {"LDS.128", 4}};
    const std::vector<Expect> padLdsm = {{"LDSM.x4", 8}, {"STS.128 A", 8}, {"STS.128 B", 4}, {"STS.32", 1},
                                         {"LDS.128", 4}};
    const int b0 = 128 * s0::kAStride * 2;
    // No variant of steps 3 to 5 changes the arithmetic (step3.cuh to step5.cuh): each accumulator sums the same
    // products in the same order as in step 2's v0, whatever the tile, hence the same C to the bit. They form one class
    // with step 2's v0, the first of it in the table, which is the reference of the others at each shape (step4 and
    // step5 run it as well).
    const unsigned same = kStep3 | kStep4 | kStep5;
    return {
        {"ref simple_wmma_gemm", kStep0, s0::simple_wmma_gemm, simple, b128x4, s0::shmem, 0, 0, {}},
        {"step0 compute_gemm, 40 blocks", kStep0 | kStep1, s0::compute_gemm, sms, b256, s0::shmem, s0::kSmemBytes, b0,
         pad160},
        {"step0 compute_gemm, 3 blocks", kStep0 | kStep1, s0::compute_gemm, three, b256, s0::shmem, s0::kSmemBytes, b0,
         pad160},
        {"step1 v0 (step 1)", kStep1 | kStep2, step1::gemm<true, true, 256>, tiles256, b256, step1::shmem,
         G1a::kSmemBytes, G1a::kStages * G1a::kAStage * 2, pad96},
        {"step1 v1 (without 1a), 40 blocks", kStep1, step1::gemm<false, true, 256>, sms, b256, step1::shmem,
         G1a::kSmemBytes, G1a::kStages * G1a::kAStage * 2, pad96},
        {"step1 v1 (without 1a), 3 blocks", kStep1, step1::gemm<false, true, 256>, three, b256, step1::shmem,
         G1a::kSmemBytes, G1a::kStages * G1a::kAStage * 2, pad96},
        {"step1 v2 (without 1b)", kStep1, step1::gemm<true, false, 256>, tiles256, b256, step1::shmem, G1b::kSmemBytes,
         G1b::kStages * G1b::kAStage * 2, pad160},
        {"step1 v3 (without 1c)", kStep1, step1::gemm<true, true, 128>, tiles128, b256, step1::shmem, G1c::kSmemBytes,
         G1c::kStages * G1c::kAStage * 2, pad96},
        {"step1 v4 (1d alone), 40 blocks", kStep1, step1::gemm<false, false, 128>, sms, b256, step1::shmem,
         G1d::kSmemBytes, G1d::kStages * G1d::kAStage * 2, pad160},
        {"step1 v4 (1d alone), 3 blocks", kStep1, step1::gemm<false, false, 128>, three, b256, step1::shmem,
         G1d::kSmemBytes, G1d::kStages * G1d::kAStage * 2, pad160},
        {"step2 v0 (step 2: 2a + 2b)", kStep2 | kStep3 | kStep4 | kStep5 | kStep6, step2::gemm<true>, tiles256, b256,
         step2::shmem, G2s::kSmemBytes, 2 * G2s::kAStage * 2, swizzle, same, emu6::rung},
        {"step2 v1 (without 2b)", kStep2, step2::gemm<false>, tiles256, b256, step2::shmem, G2p::kSmemBytes,
         2 * G2p::kAStage * 2, padLdsm},
        // Step 3, the memory hierarchy: its v4 is step 2's v0, above.
        {"step3 v0 (step 3: 3a + 3b + 3c)", kStep3 | kStep4, step3::gemm<true, true, true>, tiles256, b256,
         step3::shmem, G3::kSmemBytes, 2 * G3::kAStage * 2, swizzle, same},
        {"step3 v1 (without 3a)", kStep3, step3::gemm<false, true, true>, tiles256, b256, step3::shmem,
         G3::kSmemBytes, 2 * G3::kAStage * 2, swizzle, same},
        {"step3 v2 (without 3b)", kStep3, step3::gemm<true, false, true>, tiles256, b256, step3::shmem,
         G3::kSmemBytes, 2 * G3::kAStage * 2, swizzle, same},
        {"step3 v3 (without 3c)", kStep3, step3::gemm<true, true, false>, tiles256, b256, step3::shmem,
         G3::kSmemBytes, 2 * G3::kAStage * 2, swizzle, same},
        // Step 4, the issue: its v2 is step 3's v0, above.
        {"step4 v0 (step 4: 4a + 4b)", kStep4 | kStep5, step4::gemm<true>, tiles256, b256, step4::shmem, G4::kSmemBytes,
         2 * G4::kAStage * 2, swizzle, same},
        {"step4 v1 (without 4b)", kStep4, step4::gemm<false>, tiles256, b256, step4::shmem, G4::kSmemBytes,
         2 * G4::kAStage * 2, swizzle, same},
        // Step 5, cuBLASLt's choice of tile: its four tiles, each at every shape (its variants choose among them by
        // shape; its v3 is step 4's v0, above).
        {"step5 64 x 128", kStep5, step5::gemm<64, 128>, tiles64x128, b64, step5::shmem, G5a::kSmemBytes,
         2 * G5a::kAStage * 2, swizzle, same},
        {"step5 64 x 256", kStep5, step5::gemm<64, 256>, tiles64x256, b128, step5::shmem, G5b::kSmemBytes,
         2 * G5b::kAStage * 2, swizzle, same},
        {"step5 128 x 128", kStep5, step5::gemm<128, 128>, tiles128, b128, step5::shmem, G5c::kSmemBytes,
         2 * G5c::kAStage * 2, swizzle, same},
        {"step5 128 x 256", kStep5, step5::gemm<128, 256>, tiles256, b256, step5::shmem, G5d::kSmemBytes,
         2 * G5d::kAStage * 2, swizzle, same},
        // Step 6, beyond cuBLASLt (its kernels, each at every shape of step6 it accepts; step6.cu chooses among them by
        // shape). 6a's loads are plain copies here: step 5's tiles with 6a, and 6b's waves (40 and 3 blocks; the grid
        // barrier is not emulated), give the bits of step 2's v0. 6d (K in two halves) and 6c (bands of B, S slices
        // of K) sum fp16 parts: other bits, within the tolerance; their counters must be back at zero after each run.
        {"step6 64 x 128 (6a)", kStep6, step6::gemm<64, 128, true>, tiles64x128, b64, step6::shmem, G5a::kSmemBytes,
         2 * G5a::kAStage * 2, swizzle, same, emu6::rung},
        {"step6 64 x 256 (6a)", kStep6, step6::gemm<64, 256, true>, tiles64x256, b128, step6::shmem, G5b::kSmemBytes,
         2 * G5b::kAStage * 2, swizzle, same, emu6::rung},
        {"step6 128 x 128 (6a)", kStep6, step6::gemm<128, 128, true>, tiles128, b128, step6::shmem, G5c::kSmemBytes,
         2 * G5c::kAStage * 2, swizzle, same, emu6::rung},
        {"step6 128 x 256 (6a)", kStep6, step6::gemm<128, 256, true>, tiles256, b256, step6::shmem, G5d::kSmemBytes,
         2 * G5d::kAStage * 2, swizzle, same, emu6::rung},
        {"step6 6b waves, 40 blocks", kStep6, emu6::waves<128, 256>, sms, b256, step6::shmem, G5d::kSmemBytes,
         2 * G5d::kAStage * 2, swizzle, same, emu6::rung, nullptr, emu6::dirty},
        {"step6 6b waves, 3 blocks", kStep6, emu6::waves<128, 256>, three, b256, step6::shmem, G5d::kSmemBytes,
         2 * G5d::kAStage * 2, swizzle, same, emu6::rung, nullptr, emu6::dirty},
        {"step6 6d K in 2, 128 x 256", kStep6, emu6::split<128, 256>, emu6::grid_split<128, 256>, b256, step6::shmem,
         G5d::kSmemBytes + 16, 2 * G5d::kAStage * 2, swizzle, 0, emu6::halves, emu6::parts_split, emu6::dirty},
        {"step6 6c band 128, 1 slice", kStep6, emu6::bands<128, 1>, emu6::grid_bands<128, 1>, b256, step6::shmem,
         GB128::kSmemBytes, 2 * GB128::kAStage * 2, swizzle, 0,
         emu6::band<128, 1>, nullptr, emu6::dirty},
        {"step6 6c band 128, 2 slices", kStep6, emu6::bands<128, 2>, emu6::grid_bands<128, 2>, b256, step6::shmem,
         GB128::kSmemBytes, 2 * GB128::kAStage * 2, swizzle, 0,
         emu6::band<128, 2>, emu6::parts_bands<128>, emu6::dirty},
        {"step6 6c band 128, 5 slices", kStep6, emu6::bands<128, 5>, emu6::grid_bands<128, 5>, b256, step6::shmem,
         GB128::kSmemBytes, 2 * GB128::kAStage * 2, swizzle, 0,
         emu6::band<128, 5>, emu6::parts_bands<128>, emu6::dirty},
        {"step6 6c band 256, 1 slice", kStep6, emu6::bands<256, 1>, emu6::grid_bands<256, 1>, b256, step6::shmem,
         GB256::kSmemBytes, 2 * GB256::kAStage * 2, swizzle, 0,
         emu6::band<256, 1>, nullptr, emu6::dirty},
        {"step6 6c band 256, 2 slices", kStep6, emu6::bands<256, 2>, emu6::grid_bands<256, 2>, b256, step6::shmem,
         GB256::kSmemBytes, 2 * GB256::kAStage * 2, swizzle, 0,
         emu6::band<256, 2>, emu6::parts_bands<256>, emu6::dirty},
        {"step6 6c band 256, 3 slices", kStep6, emu6::bands<256, 3>, emu6::grid_bands<256, 3>, b256, step6::shmem,
         GB256::kSmemBytes, 2 * GB256::kAStage * 2, swizzle, 0,
         emu6::band<256, 3>, emu6::parts_bands<256>, emu6::dirty},
        {"step6 6c band 256, 5 slices", kStep6, emu6::bands<256, 5>, emu6::grid_bands<256, 5>, b256, step6::shmem,
         GB256::kSmemBytes, 2 * GB256::kAStage * 2, swizzle, 0,
         emu6::band<256, 5>, emu6::parts_bands<256>, emu6::dirty},
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
struct SameBits { const Kernel* first; std::vector<uint16_t> c; };   // a class's first kernel to run at a shape, its C

// One run: kernel k at M x N x K in one order. Prints its line; returns whether it passed. forward: k's C in the
// forward order; same: per class of kernels that give the same bits, the first of them to run at this shape and its C
// in the forward order (both set by the run that produces them, compared by the others).
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
    g_sh = k.shmem;
    g_shBytes = k.smemBytes;
    g_bStageBytes = k.bStageBytes;
    g_kernel = k.fn;
    g_M = M, g_N = N, g_K = K;
    gridDim = k.grid(M, N);
    blockDim = k.block;
    g_threads = k.block.x * k.block.y;
    if (k.before) k.before();
    std::mt19937 rng(1234 + order);
    // The blocks one at a time: in the grid's order (forward), backwards (reverse), shuffled (random). No block of
    // these kernels waits for another, so each order is a schedule the GPU could run; the last block of 6c's bands
    // and 6d's tiles to arrive changes with it.
    std::vector<unsigned> blocks(gridDim.x * gridDim.y);
    std::iota(blocks.begin(), blocks.end(), 0u);
    if (order == kReverse) std::reverse(blocks.begin(), blocks.end());
    if (order == kRandom) std::shuffle(blocks.begin(), blocks.end(), rng);
    g_gridBarriers.assign(blocks.size(), 0);
    bool complete = true;
    for (size_t n = 0; n < blocks.size() && complete; n++) {
        const unsigned bx = blocks[n] % gridDim.x, by = blocks[n] / gridDim.x;
        blockIdx = {bx, by, 0};
        g_block = by * gridDim.x + bx;
        memset(g_sh, 0xff, kShArrayBytes);   // NaN: a read of shared memory not yet written shows
        complete = run_block(order, rng);
        const unsigned char* canary = (const unsigned char*)g_sh;
        for (int i = g_shBytes; i < kShArrayBytes; i++)
            if (canary[i] != 0xff) {
                error("shared memory written at byte %d, past the %d bytes of the launch (block %d)", i, g_shBytes,
                      g_block);
                break;
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
    if (k.dirty) {
        const int dirty = k.dirty();
        if (dirty) add(std::to_string(dirty) + " counters left non-zero (each call must put them back to zero)");
    }
    for (size_t b = 1; b < g_gridBarriers.size() && complete; b++)   // on the GPU, a block that arrives less hangs
        if (g_gridBarriers[b] != g_gridBarriers[0]) {
            snprintf(s, sizeof s, "blocks reach the grid barrier different numbers of times (block 0 %d, block %zu %d)",
                     g_gridBarriers[0], b, g_gridBarriers[b]);
            add(s);
            break;
        }
    std::string banks;   // "LDSM.x4 4.00  STS.128 A 4.00 B 4.00  STS.32 1.00  LDS.128 4.00", then any other class
    std::vector<std::string> classes = {"LDSM.x4", "STS.128 A", "STS.128 B", "STS.32", "LDS.128"};
    for (const auto& b : g_banks)
        if (std::find(classes.begin(), classes.end(), b.first) == classes.end()) classes.push_back(b.first);
    for (const std::string& cls : classes) {
        const auto it = g_banks.find(cls);
        if (it == g_banks.end() || !it->second.instr) continue;
        const double v = (double)it->second.wavefronts / it->second.instr;
        if (cls == "STS.128 B" && g_banks.count("STS.128 A")) snprintf(s, sizeof s, " B %.2f", v);
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

// ---------------------------------------------------------------- the order of the tiles
// Steps 3 and 4 find their tile with step3::tile_of, step 5 with step5::tile_of, the same order for any tile size: with
// 3c, the blocks of a group cover 8 rows of tiles (the last group fewer, when fewer are left), column by column;
// without it (step 3's v3), step 0's order. The runs see two groups at one shape of verif.cu only, 1168 x 512 x 64 (10
// rows of 128-row tiles: a group of 8, then one of 2), the others having 1 to 3 rows of tiles; so each order is also
// checked on its own, at every M multiple of 16 up to kTileMaxM (64 or 128 rows of tiles; the last row of tiles cut by
// M or not) and every N multiple of the tile's width up to kTileMaxN (48 or 96 columns of tiles). The grid has one
// block per tile, as step3.cu to step5.cu launch it: no block may get a tile outside the grid, nor two blocks the same
// tile; there being as many blocks as tiles, each tile then has its block. One line per order, counted as a run.
constexpr int kTileMaxM = 8192, kTileMaxN = 12288;
struct TileOrder {
    const char* label;
    unsigned in;
    void (*fn)(int b, int M, int N, int& tm, int& tn);
    int bm = 128, bn = 256;   // the tile
};

static bool check_tiles(const TileOrder& o, Totals& totals)
{
    g_errors = 0;
    g_cur = -1;
    long shapes = 0, failed = 0;
    std::vector<int> owner;   // per tile: the block that got it, -1 if none yet
    for (int M = 16; M <= kTileMaxM; M += 16)
        for (int N = o.bn; N <= kTileMaxN; N += o.bn) {
            snprintf(g_label, sizeof g_label, "%-33s at %d x %d", o.label, M, N);   // for the fault handler
            const int tilesM = (M + o.bm - 1) / o.bm, tilesN = N / o.bn;
            owner.assign((size_t)tilesM * tilesN, -1);
            bool ok = true;
            for (int b = 0; b < tilesM * tilesN && ok; b++) {
                int tm = -1, tn = -1;
                g_block = b;
                o.fn(b, M, N, tm, tn);
                if (tm < 0 || tm >= tilesM || tn < 0 || tn >= tilesN) {
                    error("at %d x %d (%d x %d tiles), block %d gets tile (%d, %d), outside the grid", M, N, tilesM,
                          tilesN, b, tm, tn);
                    ok = false;
                } else if (owner[(size_t)tm * tilesN + tn] >= 0) {
                    error("at %d x %d (%d x %d tiles), blocks %d and %d get the same tile (%d, %d)", M, N, tilesM,
                          tilesN, owner[(size_t)tm * tilesN + tn], b, tm, tn);
                    ok = false;
                } else {
                    owner[(size_t)tm * tilesN + tn] = b;
                }
            }
            shapes++;
            failed += !ok;
        }
    char range[64];
    snprintf(range, sizeof range, "every M <= %d, N <= %d", kTileMaxM, kTileMaxN);
    snprintf(g_label, sizeof g_label, "%-33s %-27s", o.label, range);
    if (g_errors > kShownErrors) printf("EMU ERROR: ... %ld more errors in this check\n", g_errors - kShownErrors);
    totals.runs++;
    if (!failed) {
        printf("%s  ok      %ld shapes, each tile to one block\n", g_label, shapes);
        return true;
    }
    printf("%s  FAILED  [%ld of the %ld shapes: a tile outside the grid, or to two blocks]\n", g_label, failed, shapes);
    totals.failed++;
    totals.failures.push_back(std::string(o.label) + ", " + range);
    return false;
}

// ---------------------------------------------------------------- step 5's choice of tile
// step5::choose must give, on the T4's 40 SMs, the tile of cuBLASLt's best configuration at each of the judge's 17
// shapes, in the order the judge keeps for this card (its reference file judge/reference/t4.txt, the lines of the T4
// with cuBLASLt 13.2.1 and the driver for CUDA 13.3, first rank; cuBLAS's tiles read with its M and N swapped, as it
// computes C^T): 64 x 256 at qkv_m16 and gateup_m16, 64 x 128 at o_m16 and down_m16, 128 x 128 at o_m128 and down_m128,
// 128 x 256 elsewhere. The judge sorts again at each run, and the best configuration of a shape can change from one run
// to the next: this checks the rule against that table, not against every run. One line, counted as a run.
static bool check_choice(Totals& totals)
{
    struct Row { const char* name; int M, N, bm, bn; };
    const Row rows[] = {{"qkv_m16", 16, 6144, 64, 256},       {"qkv_m128", 128, 6144, 128, 256},
                        {"qkv_m512", 512, 6144, 128, 256},    {"qkv_m2048", 2048, 6144, 128, 256},
                        {"o_m16", 16, 4096, 64, 128},         {"o_m128", 128, 4096, 128, 128},
                        {"o_m512", 512, 4096, 128, 256},      {"o_m2048", 2048, 4096, 128, 256},
                        {"gateup_m16", 16, 28672, 64, 256},   {"gateup_m128", 128, 28672, 128, 256},
                        {"gateup_m512", 512, 28672, 128, 256}, {"gateup_m2048", 2048, 28672, 128, 256},
                        {"down_m16", 16, 4096, 64, 128},      {"down_m128", 128, 4096, 128, 128},
                        {"down_m512", 512, 4096, 128, 256},   {"down_m2048", 2048, 4096, 128, 256},
                        {"ladder", 2048, 2560, 128, 256}};
    int wrong = 0;
    for (const Row& r : rows) {
        const step5::Choice c = step5::choose(r.M, r.N, 40, true, true);
        if (c.bm != r.bm || c.bn != r.bn) {
            printf("EMU ERROR: step5::choose at %s (%d x %d): %d x %d, cuBLASLt's tile is %d x %d\n", r.name, r.M, r.N,
                   c.bm, c.bn, r.bm, r.bn);
            wrong++;
        }
    }
    snprintf(g_label, sizeof g_label, "%-33s %-27s", "step5 choose, 40 SMs", "the judge's 17 shapes");
    totals.runs++;
    if (!wrong) {
        printf("%s  ok      cuBLASLt's tile at each of them\n", g_label);
        return true;
    }
    printf("%s  FAILED  [%d of the 17 shapes]\n", g_label, wrong);
    totals.failed++;
    totals.failures.push_back("step5 choose at the judge's 17 shapes");
    return false;
}

// ---------------------------------------------------------------- step 6's plan
// step6::plan must give, on the T4's 40 SMs, at the judge's 17 shapes: 6c's bands at the four M = 16 shapes, with the
// widths and slices of step6.cuh (qkv_m16 128 columns x 1 slice, o_m16 256 x 3, gateup_m16 256 x 1, down_m16 256 x 5);
// 6d at o_m128 and down_m128; step 5's 128 x 256 grid at qkv_m128 and gateup_m128; 6b's waves at M >= 512 and ladder.
// With 6b, 6c and 6d switched off, step 5's choice at every shape; with one of them off (the variants v2 to v4 of
// step6.cu), step 5's choice where it acted and the whole plan elsewhere. One line, counted as a run.
static bool check_plan(Totals& totals)
{
    struct Row { const char* name; int M, N, K; step6::Kind kind; int bn, slices; };
    using step6::kBands, step6::kGrid, step6::kSplit, step6::kWaves;
    const Row rows[] = {
        {"qkv_m16", 16, 6144, 4096, kBands, 128, 1},
        {"qkv_m128", 128, 6144, 4096, kGrid, 256, 1},
        {"qkv_m512", 512, 6144, 4096, kWaves, 256, 1},
        {"qkv_m2048", 2048, 6144, 4096, kWaves, 256, 1},
        {"o_m16", 16, 4096, 4096, kBands, 256, 3},
        {"o_m128", 128, 4096, 4096, kSplit, 256, 2},
        {"o_m512", 512, 4096, 4096, kWaves, 256, 1},
        {"o_m2048", 2048, 4096, 4096, kWaves, 256, 1},
        {"gateup_m16", 16, 28672, 4096, kBands, 256, 1},
        {"gateup_m128", 128, 28672, 4096, kGrid, 256, 1},
        {"gateup_m512", 512, 28672, 4096, kWaves, 256, 1},
        {"gateup_m2048", 2048, 28672, 4096, kWaves, 256, 1},
        {"down_m16", 16, 4096, 14336, kBands, 256, 5},
        {"down_m128", 128, 4096, 14336, kSplit, 256, 2},
        {"down_m512", 512, 4096, 14336, kWaves, 256, 1},
        {"down_m2048", 2048, 4096, 14336, kWaves, 256, 1},
        {"ladder", 2048, 2560, 2048, kWaves, 256, 1},
    };
    const char* kinds[] = {"step 5's grid", "6b's waves", "6d's split", "6c's bands"};
    int wrong = 0;
    for (const Row& r : rows) {
        const step6::Plan p = step6::plan(r.M, r.N, r.K, 40, true, true, true);
        if (p.kind != r.kind || p.bn != r.bn || p.slices != r.slices) {
            printf("EMU ERROR: step6::plan at %s (%d x %d x %d): %s, %d columns, %d slices; expected %s, %d, %d\n",
                   r.name, r.M, r.N, r.K, kinds[p.kind], p.bn, p.slices, kinds[r.kind], r.bn, r.slices);
            wrong++;
        }
        const step6::Plan off = step6::plan(r.M, r.N, r.K, 40, false, false, false);
        const step5::Choice c = step5::choose(r.M, r.N, 40, true, true);
        if (off.kind != kGrid || off.bm != c.bm || off.bn != c.bn) {
            printf("EMU ERROR: step6::plan at %s without 6b, 6c, 6d: %s, %d x %d; step 5's is %d x %d\n", r.name,
                   kinds[off.kind], off.bm, off.bn, c.bm, c.bn);
            wrong++;
        }
        // One mechanism off (v2, v3, v4): step 5's tile where it acted, the whole plan elsewhere.
        const char* names[] = {"6b", "6c", "6d"};
        const step6::Kind kindOf[] = {kWaves, kBands, kSplit};
        for (int m = 0; m < 3; m++) {
            const step6::Plan q = step6::plan(r.M, r.N, r.K, 40, m != 0, m != 1, m != 2);
            const bool acted = p.kind == kindOf[m];
            const bool ok = acted ? q.kind == kGrid && q.bm == c.bm && q.bn == c.bn
                                  : q.kind == p.kind && q.bm == p.bm && q.bn == p.bn && q.slices == p.slices;
            if (!ok) {
                printf("EMU ERROR: step6::plan at %s without %s: %s, %d x %d, %d slices; expected %s\n", r.name,
                       names[m], kinds[q.kind], q.bm, q.bn, q.slices, acted ? "step 5's tile" : "the whole plan");
                wrong++;
            }
        }
    }
    snprintf(g_label, sizeof g_label, "%-33s %-27s", "step6 plan, 40 SMs", "the judge's 17 shapes");
    totals.runs++;
    if (!wrong) {
        printf("%s  ok      its mechanism at each, step 5's tile without it\n", g_label);
        return true;
    }
    printf("%s  FAILED  [%d errors]\n", g_label, wrong);
    totals.failed++;
    totals.failures.push_back("step6 plan at the judge's 17 shapes");
    return false;
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
        else if (a == "all") sel |= kAll;
        else {
            printf("usage: %s [step0|step1|step2|step3|step4|step5|step6|all]... [-x]\n"
                   "  step0  step 0, nvidia_sample.cuh's compute_gemm (40 and 3 blocks), and its simple_wmma_gemm\n"
                   "  step1  step 1's v0 to v5 (step1.cuh; v5 is step 0's compute_gemm)\n"
                   "  step2  step 2's v0 to v2 (step2.cuh; v2 is step 1's v0)\n"
                   "  step3  step 3's v0 to v4 (step3.cuh, the memory hierarchy; v4 is step 2's v0), and the two\n"
                   "         orders of the tiles of step3::tile_of\n"
                   "  step4  step 4's v0 to v2 (step4.cuh, the issue; v2 is step 3's v0), step 2's v0 (the reference\n"
                   "         of the same bits), and 3c's order of the tiles\n"
                   "  step5  step 5's four tiles (step5.cuh, cuBLASLt's choice of tile), step 4's v0 and step 2's v0,\n"
                   "         the four orders of the tiles of step5::tile_of, and step5::choose at the judge's 17\n"
                   "         shapes\n"
                   "  step6  step 6's kernels (step6.cuh, beyond cuBLASLt): step 5's four tiles with 6a, 6b's\n"
                   "         waves (40 and 3 blocks), 6d's K in two halves, 6c's bands (128 and 256 columns, 1 to 5\n"
                   "         slices), step 2's v0, the four orders of the tiles of step5::tile_of, and step6::plan at\n"
                   "         the judge's 17 shapes; at the shapes of verif.cu and two more with M < 16 (6c's rows\n"
                   "         past M)\n"
                   "         (every variant of steps 3 to 5, and step 6's tiles and waves, must give the bits of step\n"
                   "         2's v0)\n"
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
    memset(g_nanRow, 0xff, sizeof g_nanRow);

    const std::vector<Kernel> all = kernels();
    printf("gemm-ladder emulator:%s%s%s%s%s%s%s; the shapes of verif.cu; each run in forward, reverse and random "
           "order\n",
           sel & kStep0 ? " step0" : "", sel & kStep1 ? " step1" : "", sel & kStep2 ? " step2" : "",
           sel & kStep3 ? " step3" : "", sel & kStep4 ? " step4" : "", sel & kStep5 ? " step5" : "",
           sel & kStep6 ? " step6" : "");
    printf("(step 1's v5 is step 0's compute_gemm, step 2's v2 is step 1's v0, step 3's v4 is step 2's v0, step 4's "
           "v2 is step 3's v0 and step 5's v3 is step 4's v0: each kernel runs once, under its own name; wavefronts "
           "per shared instruction after err; the variants of steps 3 to 5, and step 6's tiles and waves, must give "
           "the bits of step 2's v0, and the orders of the tiles are also checked on their own, at every shape up to "
           "%d x %d)\n", kTileMaxM, kTileMaxN);
    Totals totals;
    const TileOrder orders[] = {{"step3 tile_of<true>, 3c's order", kStep3 | kStep4, step3::tile_of<true>},
                                {"step3 tile_of<false>, step 0's", kStep3, step3::tile_of<false>},
                                {"step5 tile_of<64, 128>", kStep5 | kStep6, step5::tile_of<64, 128>, 64, 128},
                                {"step5 tile_of<64, 256>", kStep5 | kStep6, step5::tile_of<64, 256>, 64, 256},
                                {"step5 tile_of<128, 128>", kStep5 | kStep6, step5::tile_of<128, 128>, 128, 128},
                                {"step5 tile_of<128, 256>", kStep5 | kStep6, step5::tile_of<128, 256>, 128, 256}};
    for (const TileOrder& o : orders)
        if ((o.in & sel) && !check_tiles(o, totals) && stopAtFirst) return summary(totals, true);
    if ((sel & kStep5) && !check_choice(totals) && stopAtFirst) return summary(totals, true);
    if ((sel & kStep6) && !check_plan(totals) && stopAtFirst) return summary(totals, true);
    // The shapes of verif.cu: its first six for every step, its last five for step 6 (6c in 12 slices of 4 bands, and
    // in 48 bands of 128 columns; 6d at a tile cut by M; 6b in two and three waves); and, for step 6 only, two with
    // M < 16 (6c's bands take 1 to 16 rows; the host passes M multiples of 16): its rows of A past M, its rows of C not
    // written. Step 6's small shapes come early, so that -x finds a bug of 6c or 6d before the large shapes.
    struct Shape { int M, N, K; unsigned in; };
    const Shape shapes[] = {{16, 256, 64, kAll},      {16, 1024, 1024, kStep6}, {16, 6144, 128, kStep6},
                            {11, 768, 2048, kStep6},  {1, 256, 1280, kStep6},   {144, 256, 64, kAll},
                            {128, 512, 128, kAll},    {144, 512, 256, kStep6},  {208, 512, 192, kAll},
                            {272, 5376, 128, kAll},   {528, 2560, 64, kStep6},  {528, 5376, 64, kStep6},
                            {1168, 512, 64, kAll}};
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
            if (!(k.in & sel & shape.in) || (k.accepts && !k.accepts(M, N, K))) continue;
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
