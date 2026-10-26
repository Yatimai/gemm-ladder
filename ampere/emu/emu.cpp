// The host emulator of the Ampere rung (README.md in this directory), adapted from the Turing rung's (turing/emu): it
// compiles the kernels' own source, nvidia_sample.cuh (step 0's compute_gemm, and simple_wmma_gemm, the reference
// point), step1.cuh to step7.cuh, as C++ against the headers in shim/, and runs each kernel on the CPU at shapes of
// verif.cu, in three orders of its threads. It checks the result against a reference in double, its 16- and 4-byte
// copies, wmma and ldmatrix accesses, that each element of C is written once (7a and 7c in place: once per part; 7b and
// 7g: once per call), the shared-memory wavefronts against the values the headers state, and that the variants of steps
// 2 to 6 give the same bits where K is not split; the orders of the tiles of steps 4 and 6 on their own, at
// every shape up to 8192 x 12288; step 6's choice of tile and the rules of steps 6 and 7 at the judge's 17 shapes.
// Step 3's cp.async copies land as late as the program
// allows (at their wait_group) in the forward order, as early as possible (when issued) in the reverse order, and
// either way, copy by copy, in the random order: a missing wait and a copy into a stage still being read both show. The
// blocks of a kernel run one at a time, or interleaved where they wait on each other or arrive on a count or a flag of
// the grid (7a and 7c in place, 6b's stream, 7b and 7g, 6c, 7e's tail): a block is suspended after each arrival, and
// a block that arrives before its threads' writes are done shows; a read of another block's data (C in place, the
// parts) before the block's acquisition for its tile (its wait on the flag, its arrival or its wait on a count) shows.
//   build (in ampere/emu): g++ -std=c++17 -O2 -march=native -I.. -Ishim emu.cpp -o emu
//   run:                   ./emu [step0|step1|step2|step3|step4|step5|step6|step7|all]... [-x]   (default: all; -x: stop at the first failure)
// Exit code 0 only if every run passes; 1 if a run fails; 2 on a usage or internal error.
#define GEMM_LADDER_EMULATE   // step2.cuh: ldmatrix and mma.sync; step3.cuh: cp.async, deferred; step4.cuh: plain stores
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
#include "step7.cuh"

// ---------------------------------------------------------------- what the shim headers and the kernels refer to
emu_dim3 threadIdx, blockIdx, gridDim, blockDim;

// The kernels' dynamic shared memory (extern __shared__ half shmem[], one per namespace): a block's 163 KB on sm_80,
// and more. A launch uses the first bytes (run() refuses one above the array); the rest keeps the NaN fill written
// before each block, a canary checked after it.
constexpr int kShArrayBytes = 164 << 10;
namespace nvidia_sample { alignas(4096) half shmem[kShArrayBytes / 2]; }
namespace step1 { alignas(4096) half shmem[kShArrayBytes / 2]; }
namespace step2 { alignas(4096) half shmem[kShArrayBytes / 2]; }
namespace step3 { alignas(4096) half shmem[kShArrayBytes / 2]; }
namespace step4 { alignas(4096) half shmem[kShArrayBytes / 2]; }
namespace step5 { alignas(4096) half shmem[kShArrayBytes / 2]; }
namespace step6 { alignas(4096) half shmem[kShArrayBytes / 2]; }
namespace step7 { alignas(4096) half shmem[kShArrayBytes / 2]; }
static const half* const kShArrays[] = {nvidia_sample::shmem, step1::shmem, step2::shmem, step3::shmem, step4::shmem,
                                        step5::shmem, step6::shmem, step7::shmem};

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
static Array g_P{"parts"}, g_F{"flags"};   // 7a's workspace and the others', sized to the launch (see emu6 below)
static half* g_sh;                       // the launch's shared memory (the kernel's namespace's shmem)
static int g_shBytes, g_bStageBytes;     // its size; where B's stages begin (a copy's store below it is to A)
static int g_block;                      // the running block (linear index)
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


// ---------------------------------------------------------------- threads: coroutines, barriers, warp exchanges
constexpr int kMaxThreads = 512;
constexpr size_t kStackBytes = 256 << 10;
enum State { kRunnable, kAtBarrier, kAtExchange, kExited, kAtGridWait, kAtArrival };
struct Access { bool store; int bytes; long offset; bool async; };   // a shared access (offset in bytes; async: cp.async)
struct Copy { char* dst; const char* src; };            // a cp.async of 16 bytes, not yet performed
struct Thread {
    ucontext_t ctx;
    State state;
    int xbytes;               // at a warp exchange: the size of its payload
    unsigned char xin[64];    // the payload
    std::vector<Access> log;  // its shared accesses since the last barrier, in program order
    std::vector<Copy> open;                  // its cp.async issued since the last commit
    std::vector<std::vector<Copy>> groups;   // its committed groups not yet performed, oldest first
    const unsigned* waitCount = nullptr;     // at a wait of the grid (7b): the count it waits on, and for which value
    unsigned waitValue = 0;
    int tileSeq = 0;                         // the tiles it has begun (step6.cuh's compute_tile)
};
static Thread g_thr0[kMaxThreads];
static Thread* g_thr = g_thr0;   // the running block's threads (g_thr0, or a slot's: see the blocks that wait)
static ucontext_t g_sched;
static int g_cur = -1, g_threads;   // the running thread (linear index in its block); threads per block
static char* g_stacks;              // kMaxThreads x (a PROT_NONE page, then a stack of kStackBytes)
static char* g_stacks0;             // the stacks of the blocks run one at a time
static unsigned char g_xall0[kMaxThreads / 32][32 * 64];   // per warp: the 32 payloads of its last exchange
static unsigned char (*g_xall)[32 * 64] = g_xall0;
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

// The reads of another block's data (step6.cuh's ld16_cg: C in place, the parts of K) come after the block's
// acquisition for the tile its threads compute: its wait on the tile's flag (7a and 7c in place), its arrival on a
// count (6b's stream, 6c) or its wait on a count (7b). compute_tile numbers the tiles each thread begins; an
// acquisition records the number of the acquiring thread's tile; a read by a thread whose tile has another number, or
// before any acquisition, is an error (on the GPU, it could read the data before the other block has written it).
static std::vector<int> g_acq;   // per block of the grid: the tile number of its last acquisition, -1 if none
static void acquired() { g_acq[g_block] = g_thr[g_cur].tileSeq; }
void step6::emu_tile_begin() { g_thr[g_cur].tileSeq++; }
void step6::emu_read_shared(const void* p)
{
    if (g_acq[g_block] != g_thr[g_cur].tileSeq)
        error("a read of another block's data (at %p) before the block's acquisition for its tile (%s)", p, who());
}

// step6.cuh's wait of the grid (7b): the thread waits until the count reaches `value`; if it has not, the block is
// suspended there (the grid's scheduler runs other blocks and resumes this one once the count has reached the value).
void step6::emu_wait_count(const unsigned* count, unsigned value)
{
    while (*count < value) {
        Thread& th = g_thr[g_cur];
        th.waitCount = count;
        th.waitValue = value;
        th.state = kAtGridWait;
        swapcontext(&th.ctx, &g_sched);
    }
    acquired();
}

// An arrival on a count or a flag of the grid (step6.cuh's arrive and set_flag): the block is suspended right
// after it, its other threads where they are, and the grid's scheduler may run other blocks before resuming it. A block
// that arrives before its other threads' writes are done shows: a block that reads them may run in between.
static void suspend_at_arrival()
{
    Thread& th = g_thr[g_cur];
    th.state = kAtArrival;
    swapcontext(&th.ctx, &g_sched);
}
unsigned step6::emu_arrive(unsigned* count)
{
    const unsigned before = (*count)++;
    suspend_at_arrival();
    acquired();
    return before;
}

static void cp_async_at_exit();

static void thread_main()
{
    g_kernel(g_A.p, g_B.p, g_C.p, g_M, g_N, g_K);
    cp_async_at_exit();
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
                converged = lane[l].log[k].store == first[k].store && lane[l].log[k].bytes == first[k].bytes &&
                            lane[l].log[k].async == first[k].async;
        }
        if (!converged)
            error("warp %d of block %d: its lanes made different shared accesses between two barriers (the bank model "
                  "needs the 32 lanes to make the same sequence)", w, g_block);
        else
            for (size_t k = 0; k < first.size(); k++) {
                long off[32];
                for (int l = 0; l < 32; l++) off[l] = lane[l].log[k].offset;
                std::string cls = std::string(first[k].async ? "LDGSTS." : first[k].store ? "STS." : "LDS.") +
                                  std::to_string(8 * first[k].bytes);
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
// ldmatrix (each lane gives its row address); the other one of step2.cuh, six registers per lane, an mma.sync.
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
static void start_block()
{
    for (int t = 0; t < g_threads; t++) {
        start_thread(&g_thr[t].ctx, g_stacks + t * (kPage + kStackBytes) + kPage);
        g_thr[t].state = kRunnable;
        g_thr[t].log.clear();
        g_thr[t].open.clear();
        g_thr[t].groups.clear();
        g_thr[t].waitCount = nullptr;
        g_thr[t].tileSeq = 0;
    }
}

// A block runs until it ends (kBlockDone), until a thread waits on a count of the grid not yet reached (kBlockWaiting:
// the block can be resumed later, its other threads where they were), until a thread arrives on a count or a flag of
// the grid (kBlockArrived: the same, at once), or until it cannot complete (kBlockFailed).
enum BlockRun { kBlockDone, kBlockWaiting, kBlockArrived, kBlockFailed };
static BlockRun continue_block(Order order, std::mt19937& rng)
{
    for (int t = 0; t < g_threads; t++) {   // a resumed block: its waits met (the scheduler checked), its arrival done
        if (g_thr[t].state == kAtGridWait && *g_thr[t].waitCount >= g_thr[t].waitValue) g_thr[t].state = kRunnable;
        if (g_thr[t].state == kAtArrival) g_thr[t].state = kRunnable;
    }
    const int warps = g_threads / 32;
    std::vector<int> ready;
    for (;;) {
        if (order == kRandom) {
            ready.clear();
            for (int t = 0; t < g_threads; t++)   // all at a phase's start; a resumed block: those whose wait is met
                if (g_thr[t].state == kRunnable) ready.push_back(t);
            while (!ready.empty() && !g_abort) {
                const size_t i = rng() % ready.size();
                const int t = ready[i];
                ready[i] = ready.back();
                ready.pop_back();
                resume(t);
                if (g_thr[t].state == kAtArrival) return kBlockArrived;
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
                        if (g_thr[t].state == kAtArrival) return kBlockArrived;
                    }
                while (complete_exchange(w));
            }
        }
        if (g_abort) return kBlockFailed;
        int atBarrier = 0, atExchange = 0, exited = 0, atWait = 0;
        for (int t = 0; t < g_threads; t++) {
            atBarrier += g_thr[t].state == kAtBarrier;
            atExchange += g_thr[t].state == kAtExchange;
            exited += g_thr[t].state == kExited;
            atWait += g_thr[t].state == kAtGridWait;
        }
        if (atExchange) {
            error("block %d: %d threads wait at a warp-wide instruction that the rest of their warp does not reach "
                  "(%d at the barrier, %d exited)", g_block, atExchange, atBarrier, exited);
            return kBlockFailed;
        }
        if (atWait) {   // waiting on the grid: met now (another thread of the block made it), or the block is suspended
            bool met = false;
            for (int t = 0; t < g_threads; t++)
                if (g_thr[t].state == kAtGridWait && *g_thr[t].waitCount >= g_thr[t].waitValue) {
                    g_thr[t].state = kRunnable;
                    met = true;
                }
            if (met) continue;
            return kBlockWaiting;
        }
        if (exited && exited < g_threads) {
            error("block %d: a barrier reached by %d threads while %d have exited", g_block, atBarrier, exited);
            return kBlockFailed;
        }
        count_shared_accesses();
        if (exited == g_threads) return kBlockDone;
        for (int t = 0; t < g_threads; t++) g_thr[t].state = kRunnable;
    }
}

static bool run_block(Order order, std::mt19937& rng)   // a block that waits on no other, and arrives on nothing
{
    start_block();
    const BlockRun r = continue_block(order, rng);
    if (r == kBlockWaiting) error("block %d waits on a count of the grid, in a kernel whose blocks run one at a time", g_block);
    if (r == kBlockArrived) {
        printf("EMU INTERNAL ERROR: block %d arrives on a count or a flag of the grid, in a kernel whose blocks run one "
               "at a time (its entry must be marked interleaved)\n", g_block);
        exit(2);
    }
    return r == kBlockDone;
}

// ---------------------------------------------------------------- blocks that wait on each other or arrive (6, 7)
// A kernel whose blocks wait on counts of the grid (7b, 7g), or arrive on counts or flags of the grid (6b's stream, 7b,
// 7g, 6c, 7a and 7c in place), runs with up to kSlots blocks alive at once, each in a slot of its own (its
// threads, their stacks, its warps' exchanges, its shared memory while suspended). The scheduler resumes a suspended block whose wait is met,
// starts the next block in the order of the run, or resumes a block suspended at an arrival, the last one first (the
// order of these depends on the order of the run, below); when none of these is left: every alive block waits on a
// count that will not reach its value, an EMU ERROR (on the GPU, a hang).
constexpr int kSlots = 16;
struct Slot {
    Thread thr[kMaxThreads];
    unsigned char xall[kMaxThreads / 32][32 * 64];
    char* stacks = nullptr;
    std::vector<unsigned char> sh;   // its shared memory, while another block runs
    unsigned bx = 0, by = 0;
    int block = -1;
    long arrival = 0;   // suspended at an arrival: when (the scheduler's count of suspensions)
};
static Slot* g_slots[kSlots];
static Slot* slot(int i)
{
    if (!g_slots[i]) {
        g_slots[i] = new Slot;
        g_slots[i]->stacks = (char*)mmap(nullptr, kMaxThreads * (kPage + kStackBytes), PROT_READ | PROT_WRITE,
                                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (g_slots[i]->stacks == MAP_FAILED) { perror("EMU INTERNAL ERROR: mmap (slot)"); exit(2); }
        for (int t = 0; t < kMaxThreads; t++) mprotect(g_slots[i]->stacks + t * (kPage + kStackBytes), kPage, PROT_NONE);
        g_slots[i]->sh.resize(kShArrayBytes);
    }
    return g_slots[i];
}
static void use_slot(Slot* sl)   // the running block's resources are this slot's
{
    g_thr = sl->thr;
    g_xall = sl->xall;
    g_stacks = sl->stacks;
    blockIdx = {sl->bx, sl->by, 0};
    g_block = sl->block;
}
static void use_main()   // back to the resources of the blocks run one at a time
{
    g_thr = g_thr0;
    g_xall = g_xall0;
    g_stacks = g_stacks0;
}
static bool resumable(const Slot* sl)   // one of its threads waits on a count that has reached its value
{
    for (int t = 0; t < g_threads; t++)
        if (sl->thr[t].state == kAtGridWait && *sl->thr[t].waitCount >= sl->thr[t].waitValue) return true;
    return false;
}
static bool arrived(const Slot* sl)   // one of its threads has just arrived on a count or a flag
{
    for (int t = 0; t < g_threads; t++)
        if (sl->thr[t].state == kAtArrival) return true;
    return false;
}
static bool check_canary(const char* sh);

// The grid in `blocks`' order of dispatch. Forward: a block whose wait is met resumes before the next block starts;
// reverse: the next block starts while a slot is free, then the last block suspended among those whose wait is met;
// both: a block suspended at an arrival resumes only once no other can start or resume, the last one suspended first
// (a block that arrives early stays suspended while the later ones, the last to arrive among them, go on); random: any
// of these, drawn. Returns whether every block completed.
static bool run_waiting(const std::vector<unsigned>& blocks, Order order, std::mt19937& rng)
{
    std::vector<int> alive, freeSlots;   // the slots of the started blocks not done, in the order they started
    for (int i = kSlots - 1; i >= 0; i--) freeSlots.push_back(i);
    size_t next = 0;
    long suspensions = 0;
    for (;;) {
        std::vector<int> met, arrivals;
        for (const int i : alive)
            if (resumable(slot(i))) met.push_back(i);
            else if (arrived(slot(i))) arrivals.push_back(i);
        std::sort(arrivals.begin(), arrivals.end(), [](int a, int b) { return slot(a)->arrival < slot(b)->arrival; });
        const bool canStart = next < blocks.size() && !freeSlots.empty();
        if (met.empty() && arrivals.empty() && !canStart) {
            if (alive.empty()) return true;
            const Slot* w = slot(alive.front());
            int t = 0;
            while (w->thr[t].state != kAtGridWait) t++;
            g_block = w->block;
            if (next < blocks.size())
                error("the %zu blocks alive (the emulator's %d SMs) all wait on counts of the grid, and the grid's "
                      "block %u cannot start: more blocks must be resident together than there are SMs (on the GPU, "
                      "a hang); block %d, thread %d waits for %u on a count at %u", alive.size(), kSlots, blocks[next],
                      w->block, t, w->thr[t].waitValue, *w->thr[t].waitCount);
            else
                error("the %zu blocks alive all wait on counts of the grid that will not reach their values (on the GPU, "
                      "a hang); block %d, thread %d waits for %u on a count at %u", alive.size(), w->block, t,
                      w->thr[t].waitValue, *w->thr[t].waitCount);
            return false;
        }
        bool start = false;
        int pick = -1;
        if (order == kRandom) {
            const size_t r = rng() % (met.size() + arrivals.size() + canStart);
            if (r < met.size()) pick = met[r];
            else if (r < met.size() + arrivals.size()) pick = arrivals[r - met.size()];
            else start = true;
        } else if (order == kForward && !met.empty()) {
            pick = met.front();
        } else if (canStart) {
            start = true;
        } else {
            pick = !met.empty() ? met.back() : arrivals.back();
        }
        Slot* sl;
        if (start) {
            pick = freeSlots.back();
            freeSlots.pop_back();
            alive.push_back(pick);
            sl = slot(pick);
            sl->bx = blocks[next] % gridDim.x;
            sl->by = blocks[next] / gridDim.x;
            sl->block = sl->by * gridDim.x + sl->bx;
            next++;
            use_slot(sl);
            memset(g_sh, 0xff, kShArrayBytes);   // NaN: a read of shared memory not yet written shows
            start_block();
        } else {
            sl = slot(pick);
            use_slot(sl);
            memcpy(g_sh, sl->sh.data(), kShArrayBytes);
        }
        const BlockRun r = continue_block(order, rng);
        if (r == kBlockFailed) return false;
        if (r == kBlockWaiting || r == kBlockArrived) {
            if (r == kBlockArrived) sl->arrival = ++suspensions;
            memcpy(sl->sh.data(), g_sh, kShArrayBytes);
            continue;
        }
        check_canary((const char*)g_sh);
        alive.erase(std::find(alive.begin(), alive.end(), pick));
        freeSlots.push_back(pick);
    }
}

// ---------------------------------------------------------------- the checks of the accesses
static long floordiv(long a, long b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

// One access of `bytes` at p (align: the alignment it needs). In the launch's shared memory: inside it and aligned, and
// logged for the bank model if `log`; in another step's shared array: an error (an access outside the launch's own). In
// A, B or C or one of their guard regions: inside the array and aligned, no store to A or B; a store to C counts its
// writes. Anywhere else (a variable of the thread): not checked. Returns whether the access may be performed.
static bool check(const void* ptr, int bytes, int align, bool store, bool log, bool async = false)
{
    const char* p = (const char*)ptr;
    const char* sh = (const char*)g_sh;
    const char* op = store ? "store" : "load";
    if (p >= sh && p < sh + kShArrayBytes) {
        const long off = p - sh;
        if (off + bytes > g_shBytes || off % align) {
            error("%s of %d bytes at byte %ld of shared memory, %s (the launch has %d bytes; %s)", op, bytes, off,
                  off + bytes > g_shBytes ? "after its end" : "misaligned", g_shBytes, who());
            return false;
        }
        if (log && g_cur >= 0) g_thr[g_cur].log.push_back({store, bytes, off, async});
        return true;
    }
    for (const half* other : kShArrays)   // another step's array: an access outside the launch's own array
        if (p >= (const char*)other && p < (const char*)other + kShArrayBytes) {
            error("%s of %d bytes in another step's shared array, %ld bytes from the start of the launch's (%s)", op,
                  bytes, (long)(p - sh), who());
            return false;
        }
    for (Array* a : {&g_A, &g_B, &g_C, &g_P, &g_F}) {
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

// step3.cuh's cp.async. A copy is checked when issued (its source in A or B, its destination in the launch's shared
// memory, both 16-byte aligned) and logged then for the bank model (class LDGSTS.128); it is performed according to
// g_cpEarly: when issued (early), or when a cp.async.wait_group leaves its group out of the groups still in flight
// (late). A copy never waited for, at the thread's end, is an error (its 16 bytes may land after the kernel reads them).
static int g_cpMode;                 // 0: every copy late (forward order), 1: early (reverse), 2: each at random (random)
static std::mt19937 g_cpRng;

void step3::emu_cp_async(void* smem, const void* gmem)
{
    const bool load = check(gmem, 16, 16, false, false);
    const bool store = check(smem, 16, 16, true, true, true);
    if (!load || !store) return;
    const bool early = g_cpMode == 1 || (g_cpMode == 2 && (g_cpRng() & 1));
    if (early) memcpy(smem, gmem, 16);
    else g_thr[g_cur].open.push_back({(char*)smem, (const char*)gmem});
}

void step3::emu_cp_async_commit()
{
    Thread& th = g_thr[g_cur];
    th.groups.push_back(th.open);
    th.open.clear();
}

void step3::emu_cp_async_wait(int pending)
{
    Thread& th = g_thr[g_cur];
    while ((int)th.groups.size() > pending) {
        for (const Copy& c : th.groups.front()) memcpy(c.dst, c.src, 16);
        th.groups.erase(th.groups.begin());
    }
}

// step6.cuh's discard of a 128-byte line from the L2 (7d): the line must lie in the parts (g_P), 128-byte aligned; its
// content is undefined from then on: NaN, so that a read of it after the discard shows.
void step6::emu_discard(const void* line)
{
    const char* p = (const char*)line;
    if (!g_P.p || p < g_P.lo() || p + 128 > g_P.hi()) {
        error("discard of a line outside the parts (%s)", who());
        return;
    }
    if (check(line, 128, 128, true, false)) memset((void*)line, 0xff, 128);
}

// step6.cuh's wait on a tile's flag: the blocks run one at a time, in an order where every earlier part of the tile has
// run; the flag must read the value the waiting part expects (otherwise, on the GPU, the block would wait forever or
// add to a C that is not yet the sum of the earlier parts).
static int g_waitBlock = -1;          // the block that last waited on a flag, and on which flag and value
static const unsigned* g_waitFlag;
static unsigned g_waitValue;
void step6::emu_wait_flag(const unsigned* flag, unsigned value)
{
    if (*flag != value)
        error("a part of K waits on its tile's flag for %u, which reads %u (%s)", value, *flag, who());
    g_waitBlock = g_block;
    g_waitFlag = flag;
    g_waitValue = value;
    acquired();
}
void step6::emu_set_flag(unsigned* flag, unsigned value)
{
    if (g_waitBlock != g_block || g_waitFlag != flag)
        error("a part of K sets its tile's flag without having waited on it (%s)", who());
    else if (value != 0 && value != g_waitValue + 1)
        error("a part of K sets its tile's flag to %u after waiting for %u (%s)", value, g_waitValue, who());
    *flag = value;
    suspend_at_arrival();
}

static void cp_async_at_exit()
{
    Thread& th = g_thr[g_cur];
    size_t left = th.open.size();
    for (const auto& gr : th.groups) left += gr.size();
    if (left) error("%zu cp.async copies never waited for at the end of the kernel (%s)", left, who());
    th.open.clear();
    th.groups.clear();
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
    for (const Array* r : {&g_A, &g_B, &g_C, &g_P, &g_F}) {
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
            snprintf(where, sizeof where, "in the guard page below thread %d's coroutine stack (raise kStackBytes)", t);
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

// ---------------------------------------------------------------- the workspaces of step6.cuh's kernels
// step7.cu's, at the sizes of the shapes here: 7a's parts in g_P and its in-place flags in g_F, arrays between guard
// regions like A, B and C (the flags' rows padded to 64, so that the array keeps cudaMalloc's 256-byte alignment); the
// parts filled with NaN before each run (a part read before it is written shows), the flags zero, and after each run
// every flag must be back at zero (the next call finds them so).
static void allocate(Array& a, int rows, int cols, bool writable);
static void release(Array& a);
namespace emu6 {
template <int BM, int BN, int WM, int SK, int ST, int kSplit, int S, bool kTogether = false>
void split(const half* A, const half* B, half* C, int M, int N, int K)
{
    step6::gemm_split<BM, BN, WM, SK, ST, kSplit, kTogether>(A, B, C, M, N, K, S, g_P.p, (unsigned*)g_F.p);
}
template <int BM, int BN, int S>
void reduce(const half*, const half*, half* C, int M, int N, int)
{
    step6::reduce_parts<BM, BN>(g_P.p, C, M, N, S);
}
template <int BM, int BN, int S> emu_dim3 grid_split(int M, int N)
{
    return emu_dim3{(unsigned)(S * ((M + BM - 1) / BM) * (N / BN)), 1, 1};
}
template <int BM, int BN> emu_dim3 grid_tiles(int M, int N)
{
    return emu_dim3{(unsigned)((M + BM - 1) / BM * (N / BN)), 1, 1};
}
template <int BM, int BN, int S> void before_parts()
{
    if (g_P.p) release(g_P);
    allocate(g_P, S * ((g_M + BM - 1) / BM) * (g_N / BN), BM * BN, true);
    memset(g_P.p, 0xff, (size_t)g_P.rows * g_P.cols * 2);
}
template <int BM, int BN> void before_flags()
{
    if (g_F.p) release(g_F);
    const int tiles = (g_M + BM - 1) / BM * (g_N / BN);
    allocate(g_F, (tiles + 63) / 64 * 64, 2, true);   // one unsigned (two halves) per tile
    memset(g_F.p, 0, (size_t)g_F.rows * g_F.cols * 2);
}
int dirty_flags()
{
    int n = 0;
    const unsigned* f = (const unsigned*)g_F.p;
    for (int i = 0; i < g_F.rows; i++) n += f[i] != 0;
    memset(g_F.p, 0, (size_t)g_F.rows * g_F.cols * 2);
    return n;
}
// step6.cu's 6b stream, at P blocks (108, the A100's SMs, or fewer than the units): its parts and counters in g_P and g_F.
// kHint: with 7d's hints, as step 7 runs it (step 6 without).
inline int blocks(int P, int N, int K) { const long U = (long)(N / 256) * (K / 64); return U < P ? (int)U : P; }
inline int max_segs(int P, int N, int K)
{
    const long steps = K / 64, U = (long)(N / 256) * steps, per = U / blocks(P, N, K);
    return (int)((steps + per - 1) / per) + 1;
}
template <int P, bool kHint> void stream(const half* A, const half* B, half* C, int M, int N, int K)
{
    step6::gemm_stream<256, 4, kHint>(A, B, C, M, N, K, g_P.p, (unsigned*)g_F.p, max_segs(P, N, K));
}
template <int P> emu_dim3 grid(int, int N) { return emu_dim3{(unsigned)blocks(P, N, g_K), 1, 1}; }
template <int P> void before()
{
    if (g_P.p) release(g_P);
    allocate(g_P, (g_N / 256) * max_segs(P, g_N, g_K), 16 * 256, true);
    memset(g_P.p, 0xff, (size_t)g_P.rows * g_P.cols * 2);
    if (g_F.p) release(g_F);
    allocate(g_F, (g_N / 256 + 63) / 64 * 64, 2, true);
    memset(g_F.p, 0, (size_t)g_F.rows * g_F.cols * 2);
}
bool m16(int M, int, int) { return M <= 16; }

// 7b: 7a's parts in g_P, two counters per tile in g_F; at the shapes whose S x tiles blocks fit on the emulator's
// kSlots SMs (7b's condition, there); kHint: with 7d's hints (step 7's).
template <int BM, int BN, int WM, int SK, int ST, int S, bool kHint>
void parallel(const half* A, const half* B, half* C, int M, int N, int K)
{
    step6::gemm_parallel<BM, BN, WM, SK, ST, kHint>(A, B, C, M, N, K, S, g_P.p, (unsigned*)g_F.p);
}
template <int BM, int BN, int S> void before_parallel()
{
    before_parts<BM, BN, S>();
    if (g_F.p) release(g_F);
    const int tiles = (g_M + BM - 1) / BM * (g_N / BN);
    allocate(g_F, (2 * tiles + 63) / 64 * 64, 2, true);   // two unsigned (four halves) per tile
    memset(g_F.p, 0, (size_t)g_F.rows * g_F.cols * 2);
}
template <int BM, int BN, int S> bool fits(int M, int N, int)
{
    return (long)S * ((M + BM - 1) / BM) * (N / BN) <= kSlots;
}

// 6c: the stream in phase at P blocks rounded down to a multiple of MT (at most one per unit, as step6.cu; at least one
// here): 108 as on the A100, and 7, where few groups cut many tiles. Its parts (two slots per block) in g_P, one
// counter per tile in g_F; kHint: with 7d's hints (step 7's).
inline int phase_blocks(int P, int M)
{
    const long MT = (M + 127) / 128, U = (long)(g_N / 256) * (g_K / 64);
    long groups = P / MT > 0 ? P / MT : 1;
    if (groups > U) groups = U;
    return (int)(groups * MT);
}
template <bool kHint, bool kTogether = false> void phase(const half* A, const half* B, half* C, int M, int N, int K)
{
    if constexpr (kTogether)
        step6::gemm_phase_sums<128, 256, 64, 64, 3, kHint>(A, B, C, M, N, K, g_P.p, (unsigned*)g_F.p);
    else
        step6::gemm_phase<128, 256, 64, 64, 3, kHint>(A, B, C, M, N, K, g_P.p, (unsigned*)g_F.p);
}
template <int P> emu_dim3 grid_phase(int M, int) { return emu_dim3{(unsigned)phase_blocks(P, M), 1, 1}; }
template <int P> void before_phase()
{
    if (g_P.p) release(g_P);
    allocate(g_P, 2 * phase_blocks(P, g_M), 128 * 256, true);
    memset(g_P.p, 0xff, (size_t)g_P.rows * g_P.cols * 2);
    if (g_F.p) release(g_F);
    const int tiles = (g_M + 127) / 128 * (g_N / 256);
    allocate(g_F, (tiles + 63) / 64 * 64, 2, true);
    memset(g_F.p, 0, (size_t)g_F.rows * g_F.cols * 2);
}

}  // namespace emu6

namespace emu7 {
// 7g: 7b's parts reduced by their own blocks, each keeping its own share in registers (7b's workspace and shapes);
// kHint: with 7d's hints.
template <int BM, int BN, int WM, int SK, int ST, int S, bool kHint>
void parallel_own(const half* A, const half* B, half* C, int M, int N, int K)
{
    step7::gemm_parallel_own<BM, BN, WM, SK, ST, kHint>(A, B, C, M, N, K, S, g_P.p, (unsigned*)g_F.p);
}

// 7e: the tail as on P SMs (step7.cu takes the card's): 108, where every tile of verif.cu's shapes is in the tail, and
// 6, where whole tiles come first; its parts in g_P (S R of them), one counter per tile of the tail in g_F; kHint:
// with 7d's hints.
inline int tiles256(int M, int N) { return (M + 127) / 128 * (N / 256); }
template <int P, int S, bool kHint> void tail(const half* A, const half* B, half* C, int M, int N, int K)
{
    const int tiles = tiles256(M, N);
    step6::gemm_tail<128, 256, 64, 64, 3, kHint>(A, B, C, M, N, K, tiles - tiles % P, S, g_P.p, (unsigned*)g_F.p);
}
template <int P, int S> emu_dim3 grid_tail(int M, int N)
{
    const int tiles = tiles256(M, N), R = tiles % P;
    return emu_dim3{(unsigned)(tiles - R + R * S), 1, 1};
}
template <int P> bool has_tail(int M, int N, int) { return tiles256(M, N) % P != 0; }
template <int P, int S> void before_tail()
{
    const int R = tiles256(g_M, g_N) % P;
    if (g_P.p) release(g_P);
    allocate(g_P, S * R, 128 * 256, true);
    memset(g_P.p, 0xff, (size_t)g_P.rows * g_P.cols * 2);
    if (g_F.p) release(g_F);
    allocate(g_F, (R + 63) / 64 * 64, 2, true);
    memset(g_F.p, 0, (size_t)g_F.rows * g_F.cols * 2);
}
}  // namespace emu7

// ---------------------------------------------------------------- the kernels
enum { kStep0 = 1, kStep1 = 2, kStep2 = 4, kStep3 = 8, kStep4 = 16, kStep5 = 32, kStep6 = 64, kStep7 = 128 };
constexpr unsigned kAll = kStep0 | kStep1 | kStep2 | kStep3 | kStep4 | kStep5 | kStep6 | kStep7;
using KernelFn = void (*)(const half*, const half*, half*, int, int, int);
struct Expect { const char* cls; double perInstr; };
struct Kernel {
    const char* label;
    unsigned in;                    // the selections that run it (step0 to step7)
    KernelFn fn;
    emu_dim3 (*grid)(int M, int N);
    emu_dim3 block;
    half* shmem;
    int smemBytes, bStageBytes;     // the launch's dynamic shared memory; where B's stage begins in it
    std::vector<Expect> expect;     // wavefronts per shared instruction, as the headers state them
    unsigned sameBits = 0;          // a class of kernels that all give the same C, bit for bit, at a shape; 0: none
    bool (*accepts)(int M, int N, int K) = nullptr;   // the shapes it runs at (nullptr: every shape of its selections)
    void (*before)() = nullptr;     // before each run
    int (*dirty)() = nullptr;       // after each run, the words of its state left non-zero
    int levels = 0;                 // > 0: the grid is `levels` rows of blocks, each row one part of K of every tile,
                                    // run row by row (7a in place: a part waits for the previous part of its tile)
    KernelFn fn2 = nullptr;         // a second kernel run after the first (7a's reduction of the parts)
    emu_dim3 (*grid2)(int M, int N) = nullptr;
    emu_dim3 block2 = {0, 0, 0};
    int cWrites = 1;                // the writes of each element of C (7a in place: one per part; 7b: one per call)
    bool interleaved = false;       // its blocks wait on counts of the grid (7b) or arrive on counts or flags of the grid
                                    // (7a in place, 6b's stream, 7b, 6c): up to kSlots of them alive at once
    int calls = 1;                  // calls in a row, each on the state the previous one left (7b: two)
};

static std::vector<Kernel> kernels()
{
    namespace s0 = nvidia_sample;
    using G1a = step1::Geometry<256, false>;   // step 1's v0, v1
    using G1b = step1::Geometry<128, false>;   // v2, v3
    using G2s = step2::Geometry<true>;         // step 2's v0
    using G2p = step2::Geometry<false>;        // v1
    using S3a = step3::Smem<3>;                // step 3's v0 to v2: 3, 2 and 1 stages
    using S3b = step3::Smem<2>;
    using S3c = step3::Smem<1>;
    using G6a = step6::Geometry<128, 128, 64, 32, 5>;   // step 6's 128 x 128 (6a)
    using G6s = step6::Geometry<128, 256, 64, 64, 3>;   // step 6's 128 x 256, for 7a's parts
    using GBands = step6::BandGeometry<128, 6>;         // 6b's bands
    using GStream = step6::BandGeometry<256, 4>;        // 6b's stream
    auto bands128 = [](int, int N) { return emu_dim3{(unsigned)(N / 128), 1, 1}; };
    // 6b's band of 16 rows: A's 16 rows of K64 swizzled as step 2's (STS.128 replaced by LDGSTS.128), B's rows of 128 or
    // 256 columns as step 2's; the output's rows of BN + 8 halves; the conflict-free pattern of steps 3 to 6.
    auto tiles256 = [](int M, int N) { return emu_dim3{(unsigned)((M + 127) / 128 * (N / 256)), 1, 1}; };
    auto tiles128 = [](int M, int N) { return emu_dim3{(unsigned)((M + 127) / 128 * (N / 128)), 1, 1}; };
    auto sms = [](int, int) { return emu_dim3{108, 1, 1}; };   // a persistent grid on the A100's 108 SMs, as verif.cu
    auto three = [](int, int) { return emu_dim3{3, 1, 1}; };   // so that each block also goes through several tiles
    auto simple = [](int M, int N) { return emu_dim3{(unsigned)((M + 63) / 64), (unsigned)(N / 64), 1}; };
    const emu_dim3 b256 = {256, 1, 1}, b128 = {128, 1, 1}, b128x4 = {128, 4, 1};
    // The headers' bank patterns. The copy's STS.128 of A and of B: conflict-free, a quarter warp storing 8 chunks of
    // one row, on rows of 160 bytes (K64 + 16 halves of padding: step 0, step 1, step 2's v1) as on step 2's swizzled
    // rows of 128 bytes; the epilogue's LDS.128 and STS.32: conflict-free; LDSM.x4: two-way with the padding (v1),
    // conflict-free with the swizzle (v0). wmma (steps 0 and 1) is emulated at the tile level: its LDSM are not counted.
    const std::vector<Expect> pad160 = {{"STS.128 A", 4}, {"STS.128 B", 4}, {"LDS.128", 4}};
    const std::vector<Expect> swizzle = {{"LDSM.x4", 4}, {"STS.128 A", 4}, {"STS.128 B", 4}, {"STS.32", 1},
                                         {"LDS.128", 4}};
    const std::vector<Expect> padLdsm = {{"LDSM.x4", 8}, {"STS.128 A", 4}, {"STS.128 B", 4}, {"STS.32", 1},
                                         {"LDS.128", 4}};
    // Steps 3 and 4 keep step 2's swizzled stages; their copies are cp.async (LDGSTS.128), conflict-free as the STS.128
    // were; step 4 changes only its stores of C and its order of the tiles.
    const std::vector<Expect> async = {{"LDSM.x4", 4}, {"LDGSTS.128 A", 4}, {"LDGSTS.128 B", 4}, {"STS.32", 1},
                                       {"LDS.128", 4}};
    const int b0 = 128 * s0::kAStride * 2;
    // Steps 2 to 6, K unsplit, issue the same ldmatrix and mma.sync on the same values (2b moves the data in shared
    // memory, step 3 changes how it gets there, step 4 where C goes and in which order, step 5 when the fragments load
    // and in which order the independent HMMA run, step 6 how the tile and its slices are cut): each element of C sums
    // the same products in the same order of K, the same C to the bit.
    const unsigned same2 = kStep2;
    // 7b (and 7g) add the parts as 7a's reduction kernel does, in the order of K, in fp32 rounded once: 7a's bits at
    // its tile.
    const unsigned parts256 = 1u << 21;
    return {
        {"ref simple_wmma_gemm", kStep0, s0::simple_wmma_gemm, simple, b128x4, s0::shmem, 0, 0, {}},
        {"step0 compute_gemm, 108 blocks", kStep0 | kStep1, s0::compute_gemm, sms, b256, s0::shmem, s0::kSmemBytes,
         b0, pad160},
        {"step0 compute_gemm, 3 blocks", kStep0 | kStep1, s0::compute_gemm, three, b256, s0::shmem, s0::kSmemBytes, b0,
         pad160},
        {"step1 v0 (step 1)", kStep1 | kStep2, step1::gemm<true, false, 256>, tiles256, b256, step1::shmem,
         G1a::kSmemBytes, G1a::kAStage * 2, pad160},
        {"step1 v1 (without 1a), 108 blocks", kStep1, step1::gemm<false, false, 256>, sms, b256, step1::shmem,
         G1a::kSmemBytes, G1a::kAStage * 2, pad160},
        {"step1 v1 (without 1a), 3 blocks", kStep1, step1::gemm<false, false, 256>, three, b256, step1::shmem,
         G1a::kSmemBytes, G1a::kAStage * 2, pad160},
        {"step1 v2 (without 1b)", kStep1, step1::gemm<true, false, 128>, tiles128, b256, step1::shmem, G1b::kSmemBytes,
         G1b::kAStage * 2, pad160},
        {"step1 v3 (1c alone), 108 blocks", kStep1, step1::gemm<false, false, 128>, sms, b256, step1::shmem,
         G1b::kSmemBytes, G1b::kAStage * 2, pad160},
        {"step1 v3 (1c alone), 3 blocks", kStep1, step1::gemm<false, false, 128>, three, b256, step1::shmem,
         G1b::kSmemBytes, G1b::kAStage * 2, pad160},
        {"step2 v0 (step 2: 2a + 2b)", kStep2 | kStep3 | kStep7, step2::gemm<true>, tiles256, b256, step2::shmem, G2s::kSmemBytes,
         G2s::kAStage * 2, swizzle, same2},
        {"step2 v1 (without 2b)", kStep2, step2::gemm<false>, tiles256, b256, step2::shmem, G2p::kSmemBytes,
         G2p::kAStage * 2, padLdsm, same2},
        // Step 3 (cp.async, three stages): its v3 is step 2's v0, above.
        {"step3 v0 (step 3: 3 stages)", kStep3 | kStep4, step3::gemm<3>, tiles256, b256, step3::shmem, S3a::kBytes,
         S3a::kAAll * 2, async, same2},
        {"step3 v1 (2 stages)", kStep3, step3::gemm<2>, tiles256, b256, step3::shmem, S3b::kBytes, S3b::kAAll * 2,
         async, same2},
        {"step3 v2 (3a alone, 1 stage)", kStep3, step3::gemm<1>, tiles256, b256, step3::shmem, S3c::kBytes,
         S3c::kAAll * 2, async, same2},
        // Step 4 (evict-first stores, a tile swizzle): its v3 is step 3's v0, above.
        {"step4 v0 (step 4: 4a + 4b)", kStep4 | kStep5, step4::gemm<true, true>, tiles256, b256, step4::shmem, S3a::kBytes,
         S3a::kAAll * 2, async, same2},
        {"step4 v1 (without 4a)", kStep4, step4::gemm<false, true>, tiles256, b256, step4::shmem, S3a::kBytes,
         S3a::kAAll * 2, async, same2},
        {"step4 v2 (without 4b)", kStep4, step4::gemm<true, false>, tiles256, b256, step4::shmem, S3a::kBytes,
         S3a::kAAll * 2, async, same2},
        // Step 5 (fragment double buffering, a serpentine HMMA order): its v3 is step 4's v0, above.
        {"step5 v0 (step 5: 5a + 5b)", kStep5 | kStep6, step5::gemm<true, true>, tiles256, b256, step5::shmem, S3a::kBytes,
         S3a::kAAll * 2, async, same2},
        {"step5 v1 (without 5a)", kStep5, step5::gemm<false, true>, tiles256, b256, step5::shmem, S3a::kBytes,
         S3a::kAAll * 2, async, same2},
        {"step5 v2 (without 5b)", kStep5, step5::gemm<true, false>, tiles256, b256, step5::shmem, S3a::kBytes,
         S3a::kAAll * 2, async, same2},
        // Step 6 (the tile per shape, 16-row bands, a stream in phase): its new tile (6a), at every shape (step6.cu
        // chooses between it and step 5's 128 x 256 by shape; its v4 is step 5's v0, above).
        {"step6 128 x 128, K32 x 5 (6a)", kStep6, step6::gemm<128, 128, 64, 32, 5>, tiles128, b128, step6::shmem,
         G6a::kSmemBytes, G6a::kAAll * 2, async, same2},
        // 6b at M <= 16: the bands (each element of C sums the same products in the same order of K as step 2's: the
        // same bits); the stream, its parts rounded to fp16 (other bits), at 108 blocks as on the A100 and at 7, where
        // ranges cross bands in their middle. Step 6 runs 6b's and 6c's kernels without 7d's hints (step 7 with them,
        // below).
        {"step6 bands 128, 6 stages (6b)", kStep6, step6::gemm_bands<128, 6, false>, bands128, b128, step6::shmem,
         GBands::kSmemBytes, 6 * GBands::kAStage * 2, async, same2, emu6::m16},
        {"step6 stream 256, 108 blocks (6b)", kStep6, emu6::stream<108, false>, emu6::grid<108>, b128, step6::shmem,
         GStream::kSmemBytes, 4 * GStream::kAStage * 2, async, 0, emu6::m16, emu6::before<108>, emu6::dirty_flags, 0,
         nullptr, nullptr, {0, 0, 0}, 1, true},
        {"step6 stream 256, 7 blocks (6b)", kStep6, emu6::stream<7, false>, emu6::grid<7>, b128, step6::shmem,
         GStream::kSmemBytes, 4 * GStream::kAStage * 2, async, 0, emu6::m16, emu6::before<7>, emu6::dirty_flags, 0,
         nullptr, nullptr, {0, 0, 0}, 1, true},
        // 6c: the stream in phase, at 108 blocks as on the A100 and at 7 (groups of MT blocks, MT the rows of tiles);
        // its cut tiles' parts rounded to fp16 (other bits than step 2's).
        {"step6 phase, 108 blocks (6c)", kStep6, emu6::phase<false>, emu6::grid_phase<108>, b256, step6::shmem,
         G6s::kSmemBytes + 16, G6s::kAAll * 2, async, 0, nullptr, emu6::before_phase<108>, emu6::dirty_flags, 0, nullptr,
         nullptr, {0, 0, 0}, 1, true},
        {"step6 phase, 7 blocks (6c)", kStep6, emu6::phase<false>, emu6::grid_phase<7>, b256, step6::shmem,
         G6s::kSmemBytes + 16, G6s::kAAll * 2, async, 0, nullptr, emu6::before_phase<7>, emu6::dirty_flags, 0, nullptr,
         nullptr, {0, 0, 0}, 1, true},
        // Step 7 (split-K, parts reduced by their own blocks, L2 hints). 7a: K in parts. Each part is rounded to fp16
        // before the parts are added: other bits than step 2's, within the tolerance. Parts and reduction: two kernels,
        // the second without shared memory; in place: the parts of a tile in the order of K, each writing C (2 writes
        // of each element).
        {"step7 128 x 256, K in 3 parts (7a)", kStep7, emu6::split<128, 256, 64, 64, 3, step6::kParts, 3>,
         emu6::grid_split<128, 256, 3>, b256, step6::shmem, G6s::kSmemBytes, G6s::kAAll * 2, async, parts256, nullptr,
         emu6::before_parts<128, 256, 3>, nullptr, 0, emu6::reduce<128, 256, 3>, emu6::grid_tiles<128, 256>, b256},
        {"step7 128 x 128, K in 2 in place (7a)", kStep7, emu6::split<128, 128, 64, 32, 5, step6::kInPlace, 2>,
         emu6::grid_split<128, 128, 2>, b128, step6::shmem, G6a::kSmemBytes, G6a::kAAll * 2, async, 0, nullptr,
         emu6::before_flags<128, 128>, emu6::dirty_flags, 2, nullptr, nullptr, {0, 0, 0}, 2, true},
        // 7c: 7a's in-place parts on step 5's 128 x 256 tile, in 5 parts (its rule at down_m512) and in 2 (step7.cu's
        // v7), row by row of the grid as 7a's in place (S writes of each element; other bits than step 2's).
        {"step7 128 x 256, K in 5 in place (7c)", kStep7, emu6::split<128, 256, 64, 64, 3, step6::kInPlace, 5>,
         emu6::grid_split<128, 256, 5>, b256, step6::shmem, G6s::kSmemBytes, G6s::kAAll * 2, async, 0, nullptr,
         emu6::before_flags<128, 256>, emu6::dirty_flags, 5, nullptr, nullptr, {0, 0, 0}, 5, true},
        {"step7 128 x 256, K in 2 in place (7c)", kStep7, emu6::split<128, 256, 64, 64, 3, step6::kInPlace, 2>,
         emu6::grid_split<128, 256, 2>, b256, step6::shmem, G6s::kSmemBytes, G6s::kAAll * 2, async, 0, nullptr,
         emu6::before_flags<128, 256>, emu6::dirty_flags, 2, nullptr, nullptr, {0, 0, 0}, 2, true},
        // 7g: 7b without the block's own part (kept in registers): the same sums in the same order, 7a's bits.
        {"step7 128 x 256, K in 3 parts (7g)", kStep7, emu7::parallel_own<128, 256, 64, 64, 3, 3, true>,
         emu6::grid_split<128, 256, 3>, b256, step7::shmem, G6s::kSmemBytes, G6s::kAAll * 2, async, parts256,
         emu6::fits<128, 256, 3>, emu6::before_parallel<128, 256, 3>, emu6::dirty_flags, 0, nullptr, nullptr, {0, 0, 0},
         2, true, 2},
        // 7g without 7d's hints (step7.cu's v4 and v10).
        {"step7 128 x 256, K in 3 parts (7g, no 7d)", kStep7, emu7::parallel_own<128, 256, 64, 64, 3, 3, false>,
         emu6::grid_split<128, 256, 3>, b256, step7::shmem, G6s::kSmemBytes, G6s::kAAll * 2, async, parts256,
         emu6::fits<128, 256, 3>, emu6::before_parallel<128, 256, 3>, emu6::dirty_flags, 0, nullptr, nullptr, {0, 0, 0},
         2, true, 2},
        // 7d: the kernels of 6b, 7b and 6c (step6.cuh) with their L2 cache hints, as step 7 runs them: the hints as
        // plain copies, each line of the parts discarded once added filled with NaN (a part read after its discard
        // shows).
        {"step7 bands 128, 6 stages (6b, 7d)", kStep7, step6::gemm_bands<128, 6, true>, bands128, b128, step6::shmem,
         GBands::kSmemBytes, 6 * GBands::kAStage * 2, async, same2, emu6::m16},
        {"step7 stream 256, 108 blocks (6b, 7d)", kStep7, emu6::stream<108, true>, emu6::grid<108>, b128, step6::shmem,
         GStream::kSmemBytes, 4 * GStream::kAStage * 2, async, 0, emu6::m16, emu6::before<108>, emu6::dirty_flags, 0,
         nullptr, nullptr, {0, 0, 0}, 1, true},
        {"step7 stream 256, 7 blocks (6b, 7d)", kStep7, emu6::stream<7, true>, emu6::grid<7>, b128, step6::shmem,
         GStream::kSmemBytes, 4 * GStream::kAStage * 2, async, 0, emu6::m16, emu6::before<7>, emu6::dirty_flags, 0,
         nullptr, nullptr, {0, 0, 0}, 1, true},
        // 7b: 7a's parts, reduced by their own blocks once all S have arrived (7a's bits at 3 parts); two calls in a
        // row, the second on the counters the first put back to zero.
        {"step7 128 x 256, K in 3 parts (7b, 7d)", kStep7, emu6::parallel<128, 256, 64, 64, 3, 3, true>,
         emu6::grid_split<128, 256, 3>, b256, step6::shmem, G6s::kSmemBytes, G6s::kAAll * 2, async, parts256,
         emu6::fits<128, 256, 3>, emu6::before_parallel<128, 256, 3>, emu6::dirty_flags, 0, nullptr, nullptr, {0, 0, 0},
         2, true, 2},
        {"step7 phase, 108 blocks (6c, 7d)", kStep7, emu6::phase<true>, emu6::grid_phase<108>, b256, step6::shmem,
         G6s::kSmemBytes + 16, G6s::kAAll * 2, async, 0, nullptr, emu6::before_phase<108>, emu6::dirty_flags, 0, nullptr,
         nullptr, {0, 0, 0}, 1, true},
        {"step7 phase, 7 blocks (6c, 7d)", kStep7, emu6::phase<true>, emu6::grid_phase<7>, b256, step6::shmem,
         G6s::kSmemBytes + 16, G6s::kAAll * 2, async, 0, nullptr, emu6::before_phase<7>, emu6::dirty_flags, 0, nullptr,
         nullptr, {0, 0, 0}, 1, true},
        // 7h: the reads of an addition issued together, in the parts in place of 7a and 7c and in 6c's kernel as step
        // 7 runs it (the same bits as without 7h).
        {"step7 128 x 128, K in 2 in place (7a, 7h)", kStep7,
         emu6::split<128, 128, 64, 32, 5, step6::kInPlace, 2, true>, emu6::grid_split<128, 128, 2>, b128, step6::shmem,
         G6a::kSmemBytes, G6a::kAAll * 2, async, 0, nullptr, emu6::before_flags<128, 128>, emu6::dirty_flags, 2,
         nullptr, nullptr, {0, 0, 0}, 2, true},
        {"step7 128 x 256, K in 5 in place (7c, 7h)", kStep7,
         emu6::split<128, 256, 64, 64, 3, step6::kInPlace, 5, true>, emu6::grid_split<128, 256, 5>, b256, step6::shmem,
         G6s::kSmemBytes, G6s::kAAll * 2, async, 0, nullptr, emu6::before_flags<128, 256>, emu6::dirty_flags, 5,
         nullptr, nullptr, {0, 0, 0}, 5, true},
        {"step7 128 x 256, K in 2 in place (7c, 7h)", kStep7,
         emu6::split<128, 256, 64, 64, 3, step6::kInPlace, 2, true>, emu6::grid_split<128, 256, 2>, b256, step6::shmem,
         G6s::kSmemBytes, G6s::kAAll * 2, async, 0, nullptr, emu6::before_flags<128, 256>, emu6::dirty_flags, 2,
         nullptr, nullptr, {0, 0, 0}, 2, true},
        {"step7 phase, 108 blocks (6c, 7d, 7h)", kStep7, emu6::phase<true, true>, emu6::grid_phase<108>, b256,
         step6::shmem, G6s::kSmemBytes + 16, G6s::kAAll * 2, async, 0, nullptr, emu6::before_phase<108>,
         emu6::dirty_flags, 0, nullptr, nullptr, {0, 0, 0}, 1, true},
        {"step7 phase, 7 blocks (6c, 7d, 7h)", kStep7, emu6::phase<true, true>, emu6::grid_phase<7>, b256,
         step6::shmem, G6s::kSmemBytes + 16, G6s::kAAll * 2, async, 0, nullptr, emu6::before_phase<7>,
         emu6::dirty_flags, 0, nullptr, nullptr, {0, 0, 0}, 1, true},
        // 6c's kernel with 7h, without 7d's hints (step7.cu's v4 and v10).
        {"step7 phase, 108 blocks (6c, 7h, no 7d)", kStep7, emu6::phase<false, true>, emu6::grid_phase<108>, b256,
         step6::shmem, G6s::kSmemBytes + 16, G6s::kAAll * 2, async, 0, nullptr, emu6::before_phase<108>,
         emu6::dirty_flags, 0, nullptr, nullptr, {0, 0, 0}, 1, true},
        {"step7 phase, 7 blocks (6c, 7h, no 7d)", kStep7, emu6::phase<false, true>, emu6::grid_phase<7>, b256,
         step6::shmem, G6s::kSmemBytes + 16, G6s::kAAll * 2, async, 0, nullptr, emu6::before_phase<7>,
         emu6::dirty_flags, 0, nullptr, nullptr, {0, 0, 0}, 1, true},
        // 7e: the tail of the waves in parts of K, as on 108 SMs (every tile in the tail, 3 parts) and on 6 (whole
        // tiles first, then the tail in 2 parts), with 7d's hints; its parts rounded to fp16 (other bits than step 5's).
        {"step7 tail, 108 SMs, 3 parts (7e, 7d)", kStep7, emu7::tail<108, 3, true>, emu7::grid_tail<108, 3>, b256,
         step6::shmem, G6s::kSmemBytes + 16, G6s::kAAll * 2, async, 0, emu7::has_tail<108>, emu7::before_tail<108, 3>,
         emu6::dirty_flags, 0, nullptr, nullptr, {0, 0, 0}, 1, true},
        {"step7 tail, 6 SMs, 2 parts (7e, 7d)", kStep7, emu7::tail<6, 2, true>, emu7::grid_tail<6, 2>, b256,
         step6::shmem, G6s::kSmemBytes + 16, G6s::kAAll * 2, async, 0, emu7::has_tail<6>, emu7::before_tail<6, 2>,
         emu6::dirty_flags, 0, nullptr, nullptr, {0, 0, 0}, 1, true},
        // 7e without 7d's hints (step7.cu's v4 and v10).
        {"step7 tail, 108 SMs, 3 parts (7e, no 7d)", kStep7, emu7::tail<108, 3, false>, emu7::grid_tail<108, 3>,
         b256, step6::shmem, G6s::kSmemBytes + 16, G6s::kAAll * 2, async, 0, emu7::has_tail<108>,
         emu7::before_tail<108, 3>, emu6::dirty_flags, 0, nullptr, nullptr, {0, 0, 0}, 1, true},
        {"step7 tail, 6 SMs, 2 parts (7e, no 7d)", kStep7, emu7::tail<6, 2, false>, emu7::grid_tail<6, 2>, b256,
         step6::shmem, G6s::kSmemBytes + 16, G6s::kAAll * 2, async, 0, emu7::has_tail<6>, emu7::before_tail<6, 2>,
         emu6::dirty_flags, 0, nullptr, nullptr, {0, 0, 0}, 1, true},
    };
}

// ---------------------------------------------------------------- the arrays and the runs
static bool check_canary(const char* sh)   // the shared array after the launch's bytes still NaN: not written
{
    for (int i = g_shBytes; i < kShArrayBytes; i++)
        if ((unsigned char)sh[i] != 0xff) {
            error("shared memory written at byte %d, outside the %d bytes of the launch (block %d)", i, g_shBytes,
                  g_block);
            return false;
        }
    return true;
}

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
    if (k.smemBytes > kShArrayBytes - 16) {   // outside the array, its accesses would go unchecked
        printf("EMU INTERNAL ERROR: %s: %d bytes of shared memory, above the emulator's %d\n", k.label, k.smemBytes,
               kShArrayBytes - 16);
        exit(2);
    }
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
    g_cpMode = order == kForward ? 0 : order == kReverse ? 1 : 2;   // step 3's cp.async: late, early, mixed
    g_cpRng.seed(99 + order);
    // The blocks in the grid's order (forward), backwards (reverse), shuffled (random): one at a time, or interleaved
    // (run_waiting) in the kernels whose blocks wait or arrive. No block of the others waits for another, nor arrives,
    // so each order is a schedule the GPU could run.
    // With levels (7a in place), the rows of blocks one after the other, each row in that order: a part of K runs after
    // the earlier parts of its tile, the order of dispatch that step6.cuh assumes (CUDA does not promise it).
    auto order_of = [&](unsigned count, int levels) {
        std::vector<unsigned> blocks(count);
        const unsigned row = levels > 0 ? count / levels : count;
        for (unsigned r0 = 0; r0 < count; r0 += row) {
            std::iota(blocks.begin() + r0, blocks.begin() + r0 + row, r0);
            if (order == kReverse) std::reverse(blocks.begin() + r0, blocks.begin() + r0 + row);
            if (order == kRandom) std::shuffle(blocks.begin() + r0, blocks.begin() + r0 + row, rng);
        }
        return blocks;
    };
    bool complete = true;
    auto run_grid = [&](const std::vector<unsigned>& blocks) {
        for (size_t n = 0; n < blocks.size() && complete; n++) {
            const unsigned bx = blocks[n] % gridDim.x, by = blocks[n] / gridDim.x;
            blockIdx = {bx, by, 0};
            g_block = by * gridDim.x + bx;
            memset(g_sh, 0xff, kShArrayBytes);   // NaN: a read of shared memory not yet written shows
            complete = run_block(order, rng);
            check_canary((const char*)g_sh);
        }
    };
    // 7b and 7g: two calls, the second on the state the first left (its counters, its parts). C is set to NaN again
    // before the second and the two C compared: an error of the first call would otherwise be written over.
    std::vector<uint16_t> firstCall;
    for (int call = 0; call < k.calls && complete; call++) {
        g_acq.assign((size_t)gridDim.x * gridDim.y, -1);
        if (call == 1) {
            firstCall.assign((const uint16_t*)g_C.p, (const uint16_t*)g_C.p + nC);
            memset(g_C.p, 0xff, nC * 2);
        }
        const std::vector<unsigned> blocks = order_of(gridDim.x * gridDim.y, k.levels);
        if (!k.interleaved) run_grid(blocks);
        else {
            complete = run_waiting(blocks, order, rng);
            use_main();
        }
    }
    if (k.fn2 && complete) {   // the second kernel (7a's reduction): no shared memory
        g_kernel = k.fn2;
        gridDim = k.grid2(M, N);
        blockDim = k.block2;
        g_threads = k.block2.x * k.block2.y;
        g_shBytes = 0;
        run_grid(order_of(gridDim.x * gridDim.y, 0));
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
        } else if (g_cWrites[i] > k.cWrites) again++;
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
    long differCalls = 0;
    for (size_t i = 0; i < firstCall.size(); i++) differCalls += firstCall[i] != out[i];
    if (differCalls) add(std::to_string(differCalls) + " elements of C differ between the first call and the second");
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
    std::string banks;   // "LDSM.x4 4.00  STS.128 A 4.00 B 4.00  STS.32 1.00  LDS.128 4.00", then any other class
    std::vector<std::string> classes = {"LDSM.x4", "STS.128 A", "STS.128 B", "LDGSTS.128 A", "LDGSTS.128 B", "STS.32",
                                        "LDS.128"};
    for (const auto& b : g_banks)
        if (std::find(classes.begin(), classes.end(), b.first) == classes.end()) classes.push_back(b.first);
    for (const std::string& cls : classes) {
        const auto it = g_banks.find(cls);
        if (it == g_banks.end() || !it->second.instr) continue;
        const double v = (double)it->second.wavefronts / it->second.instr;
        if ((cls == "STS.128 B" && g_banks.count("STS.128 A")) || (cls == "LDGSTS.128 B" && g_banks.count("LDGSTS.128 A")))
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

// ---------------------------------------------------------------- the order of the tiles
// Step 4 finds its tile with step4::tile_of: with 4b, the blocks of a group cover R = 16 rows of tiles (the last group
// fewer, when fewer are left), column by column; without it (step 4's v2), step 0's order. Step 4's runs have at most
// 10 rows of tiles, so they see one group only; each order is checked on its own, at every M multiple of 16 up to
// kTileMaxM (64 rows of tiles: four groups; the last row of tiles cut by M or not) and every N multiple of 256 up to
// kTileMaxN (48 columns of tiles). The grid has one block per tile, as step4.cu launches it: no block may get a tile
// outside the grid, nor two blocks the same tile; there being as many blocks as tiles, each tile then has its block.
// One line per order, counted as a run.
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

// ---------------------------------------------------------------- step 6's choice of tile
// step6::choose must give, on the A100's 108 SMs, at the judge's 17 shapes: 128 x 128 where 128 x 256 would give fewer
// than three waves of tiles (6a), except 128 x 256 in 3 parts at o_m16, down_m16, o_m128 and down_m128 (7a), and in 2
// parts in place at o_m512 and down_m512 (7a); 128 x 256, step 5's, elsewhere. Without 7a (step 6), the same tiles, K
// whole; without 6a either (step 6's v1), 128 x 256 at every shape. One line, counted as a run.
static bool check_choice(Totals& totals)
{
    // The expected plan (tile, parts, split) with 6a and 7a (step 7), then without 6a and 7a (step 6's v1), without 7a
    // (step 6).
    using step6::k128x128, step6::k128x256, step6::kWhole, step6::kParts, step6::kInPlace;
    struct P { step6::Tile tile; int S; step6::Split split; };
    struct Row { const char* name; int M, N, K; P all, no6a, no7a; };
    const P W256 = {k128x256, 1, kWhole}, W128 = {k128x128, 1, kWhole};
    const P P256 = {k128x256, 3, kParts}, I128 = {k128x128, 2, kInPlace};
    const Row rows[] = {
        {"qkv_m16", 16, 10240, 8192, W128, W256, W128},        {"qkv_m128", 128, 10240, 8192, W128, W256, W128},
        {"qkv_m512", 512, 10240, 8192, W128, W256, W128},      {"qkv_m2048", 2048, 10240, 8192, W256, W256, W256},
        {"o_m16", 16, 8192, 8192, P256, W256, W128},           {"o_m128", 128, 8192, 8192, P256, W256, W128},
        {"o_m512", 512, 8192, 8192, I128, W256, W128},         {"o_m2048", 2048, 8192, 8192, W256, W256, W256},
        {"gateup_m16", 16, 57344, 8192, W128, W256, W128},     {"gateup_m128", 128, 57344, 8192, W128, W256, W128},
        {"gateup_m512", 512, 57344, 8192, W256, W256, W256},   {"gateup_m2048", 2048, 57344, 8192, W256, W256, W256},
        {"down_m16", 16, 8192, 28672, P256, W256, W128},       {"down_m128", 128, 8192, 28672, P256, W256, W128},
        {"down_m512", 512, 8192, 28672, I128, W256, W128},     {"down_m2048", 2048, 8192, 28672, W256, W256, W256},
        {"ladder", 3456, 4096, 2048, W256, W256, W256}};
    const char* tiles[] = {"128 x 256", "128 x 128"};
    const char* splits[] = {"whole", "parts", "in place"};
    auto same = [](const step6::Plan& a, const P& b) { return a.tile == b.tile && a.S == b.S && a.split == b.split; };
    int wrong = 0;
    for (const Row& r : rows) {
        const step6::Plan got[3] = {step6::choose(r.M, r.N, r.K, 108, true, true),
                                    step6::choose(r.M, r.N, r.K, 108, false, false),
                                    step6::choose(r.M, r.N, r.K, 108, true, false)};
        const P want[3] = {r.all, r.no6a, r.no7a};
        const char* which[3] = {"", " without 6a and 7a", " without 7a"};
        for (int v = 0; v < 3; v++)
            if (!same(got[v], want[v])) {
                printf("EMU ERROR: step6::choose at %s%s: %s, K in %d (%s); expected %s, K in %d (%s)\n", r.name,
                       which[v], tiles[got[v].tile], got[v].S, splits[got[v].split], tiles[want[v].tile], want[v].S,
                       splits[want[v].split]);
                wrong++;
            }
    }
    snprintf(g_label, sizeof g_label, "%-33s %-27s", "step6 choose, 108 SMs", "the judge's 17 shapes");
    totals.runs++;
    if (!wrong) {
        printf("%s  ok      the expected plan at each, with and without 6a and 7a\n", g_label);
        return true;
    }
    printf("%s  FAILED  [%d of the 17 shapes]\n", g_label, wrong);
    totals.failed++;
    totals.failures.push_back("step6 choose at the judge's 17 shapes");
    return false;
}

// ---------------------------------------------------------------- 6b's rule for M <= 16
// step6::m16 must give, on the A100's 108 SMs: bands at qkv_m16 (80 bands: at least 2/3 of the SMs) and o_m16 (64
// bands, short K: 16 x 128 x 8192 < 2^25); the stream at gateup_m16 (448 bands, more than a wave) and down_m16 (64
// bands, long K); no band at M > 16 (checked at qkv_m128, down_m512 and the ladder). One line, counted as a run.
static bool check_m16(Totals& totals)
{
    struct Row { const char* name; int M, N, K; step6::M16 want; };
    const Row rows[] = {{"qkv_m16", 16, 10240, 8192, step6::kBands},    {"o_m16", 16, 8192, 8192, step6::kBands},
                        {"gateup_m16", 16, 57344, 8192, step6::kStream}, {"down_m16", 16, 8192, 28672, step6::kStream},
                        {"qkv_m128", 128, 10240, 8192, step6::kNoBands}, {"down_m512", 512, 8192, 28672, step6::kNoBands},
                        {"ladder", 3456, 4096, 2048, step6::kNoBands}};
    const char* names[] = {"no band", "bands", "the stream"};
    int wrong = 0;
    for (const Row& r : rows) {
        const step6::M16 got = step6::m16(r.M, r.N, r.K, 108);
        if (got != r.want) {
            printf("EMU ERROR: step6::m16 at %s: %s, expected %s\n", r.name, names[got], names[r.want]);
            wrong++;
        }
    }
    snprintf(g_label, sizeof g_label, "%-33s %-27s", "step6 m16, 108 SMs", "7 of the judge's shapes");
    totals.runs++;
    if (!wrong) {
        printf("%s  ok      bands or the stream as expected\n", g_label);
        return true;
    }
    printf("%s  FAILED  [%d shapes]\n", g_label, wrong);
    totals.failed++;
    totals.failures.push_back("step6 m16 at the judge's shapes");
    return false;
}

// ---------------------------------------------------------------- 7b's condition: its blocks resident together
// 7b's blocks wait on each other: the S parts of every tile must be resident together, S x tiles blocks on the SMs
// (one block per SM at these tiles), or the first blocks wait forever on parts that cannot start. On the A100's 108
// SMs, at the judge's 17 shapes, with step6::choose's plan and with K in 3 parts at every shape (step7.cu's v6): 7b
// (step6::parallel_fits) exactly where S x tiles <= 108, counted here apart from the header; with the plan, at o_m16,
// down_m16, o_m128 and down_m128 (32 tiles of 128 x 256 in 3 parts, 96 blocks; at M = 16, 6b runs first). One line,
// counted as a run.
static bool check_parallel(Totals& totals)
{
    struct Row { const char* name; int M, N, K; bool want; };
    const Row rows[] = {
        {"qkv_m16", 16, 10240, 8192, false},   {"qkv_m128", 128, 10240, 8192, false},
        {"qkv_m512", 512, 10240, 8192, false}, {"qkv_m2048", 2048, 10240, 8192, false},
        {"o_m16", 16, 8192, 8192, true},       {"o_m128", 128, 8192, 8192, true},
        {"o_m512", 512, 8192, 8192, false},    {"o_m2048", 2048, 8192, 8192, false},
        {"gateup_m16", 16, 57344, 8192, false}, {"gateup_m128", 128, 57344, 8192, false},
        {"gateup_m512", 512, 57344, 8192, false}, {"gateup_m2048", 2048, 57344, 8192, false},
        {"down_m16", 16, 8192, 28672, true},   {"down_m128", 128, 8192, 28672, true},
        {"down_m512", 512, 8192, 28672, false}, {"down_m2048", 2048, 8192, 28672, false},
        {"ladder", 3456, 4096, 2048, false}};
    int wrong = 0;
    for (const Row& r : rows)
        for (const bool forced : {false, true}) {
            step6::Plan p = step6::choose(r.M, r.N, r.K, 108, true, true);
            if (forced) {   // step7.cu's v6: 3 parts, the tile 128 x 256
                p.S = 3;
                p.split = step6::kParts;
                if (p.tile == step6::k128x128) p.tile = step6::k128x256;
            }
            if (p.split != step6::kParts) {
                if (!forced && r.want) {
                    printf("EMU ERROR: 7b at %s: step6::choose gives no parts, 7b expected\n", r.name);
                    wrong++;
                }
                continue;
            }
            const int BN = p.tile == step6::k128x256 ? 256 : 128;
            const int tiles = (r.M + 127) / 128 * (r.N / BN);
            const bool got = step6::parallel_fits(tiles, p.S, 108), resident = (long)p.S * tiles <= 108;
            if (got != resident || (!forced && got != r.want)) {
                printf("EMU ERROR: 7b at %s%s: %d tiles of 128 x %d in %d parts, %d blocks on 108 SMs; parallel_fits "
                       "gives %s, expected %s\n", r.name, forced ? " (3 parts forced)" : "", tiles, BN, p.S,
                       p.S * tiles, got ? "7b" : "no 7b", resident && (forced || r.want) ? "7b" : "no 7b");
                wrong++;
            }
        }
    snprintf(g_label, sizeof g_label, "%-33s %-27s", "step6 parallel_fits, 108 SMs", "the judge's 17 shapes");
    totals.runs++;
    if (!wrong) {
        printf("%s  ok      7b where its blocks fit on the SMs, at o_m16, down_m16, o_m128 and down_m128 with the plan\n",
               g_label);
        return true;
    }
    printf("%s  FAILED  [%d cases]\n", g_label, wrong);
    totals.failed++;
    totals.failures.push_back("step6 parallel_fits at the judge's shapes");
    return false;
}

// ---------------------------------------------------------------- 6c's rule
// step6::phase must give, on the A100's 108 SMs, at the judge's 17 shapes: the stream in phase at gateup_m128 (448
// tiles of 128 x 128, the fifth wave 16 tiles) and gateup_m512 (896 tiles of 128 x 256, the ninth wave 32 tiles), and
// nowhere else: qkv_m512's third wave is nearly full (104 of 108), qkv_m128 is one wave, 7a's rule cuts K at the other
// M = 128 / 512 shapes (step 7; 6c's rule reads its plan), and M = 16 and 2048 are outside its class. One line, counted
// as a run.
static bool check_phase(Totals& totals)
{
    struct Row { const char* name; int M, N, K; bool want; };
    const Row rows[] = {
        {"qkv_m16", 16, 10240, 8192, false},    {"qkv_m128", 128, 10240, 8192, false},
        {"qkv_m512", 512, 10240, 8192, false},  {"qkv_m2048", 2048, 10240, 8192, false},
        {"o_m16", 16, 8192, 8192, false},       {"o_m128", 128, 8192, 8192, false},
        {"o_m512", 512, 8192, 8192, false},     {"o_m2048", 2048, 8192, 8192, false},
        {"gateup_m16", 16, 57344, 8192, false}, {"gateup_m128", 128, 57344, 8192, true},
        {"gateup_m512", 512, 57344, 8192, true}, {"gateup_m2048", 2048, 57344, 8192, false},
        {"down_m16", 16, 8192, 28672, false},   {"down_m128", 128, 8192, 28672, false},
        {"down_m512", 512, 8192, 28672, false}, {"down_m2048", 2048, 8192, 28672, false},
        {"ladder", 3456, 4096, 2048, false}};
    int wrong = 0;
    for (const Row& r : rows) {
        const bool got = step6::phase(r.M, r.N, r.K, 108);
        if (got != r.want) {
            printf("EMU ERROR: step6::phase at %s: %s, expected %s\n", r.name, got ? "the stream in phase" : "none",
                   r.want ? "the stream in phase" : "none");
            wrong++;
        }
    }
    snprintf(g_label, sizeof g_label, "%-33s %-27s", "step6 phase, 108 SMs", "the judge's 17 shapes");
    totals.runs++;
    if (!wrong) {
        printf("%s  ok      the stream in phase at gateup_m128 and gateup_m512 only\n", g_label);
        return true;
    }
    printf("%s  FAILED  [%d shapes]\n", g_label, wrong);
    totals.failed++;
    totals.failures.push_back("step6 phase at the judge's shapes");
    return false;
}

// ---------------------------------------------------------------- the rules of 7c, 7e and 7f
// On the A100's 108 SMs, at the judge's 17 shapes: step7::inplace_parts (7c) gives 5 parts at down_m512 (128 tiles of
// 128 x 256 in 5 parts: 640 blocks, 6 waves of 108, 99 % full) and none elsewhere (at o_m512, the other shape where
// 7a adds parts in place, K = 8192 allows 2 parts of 64 slices at most: 256 blocks, 3 waves, 79 % full);
// step6::tail_parts (7e) gives 4 parts at down_m2048 (512 tiles, a tail of 80: three rounds of quarter tiles, 0.75 of a
// tile) and 2 at gateup_m2048 (3584 tiles, a tail of 20: 0.5); no tail at o_m2048 (a tail of 80, but K = 8192 allows 2
// parts of 64 slices at most: still one tile), qkv_m2048 (a tail of 100: no S below one tile), the ladder (432 tiles,
// 4 full waves), nor at M <= 512; step7::phase_in_place (7f) gives the stream in phase at o_m512 only (256 tiles of
// 128 x 128, the third wave 40 tiles). One line, counted as a run.
static bool check_rules_7(Totals& totals)
{
    struct Row { const char* name; int M, N, K; int f; int e; bool g; };
    const Row rows[] = {
        {"qkv_m16", 16, 10240, 8192, 0, 0, false},      {"qkv_m128", 128, 10240, 8192, 0, 0, false},
        {"qkv_m512", 512, 10240, 8192, 0, 0, false},    {"qkv_m2048", 2048, 10240, 8192, 0, 0, false},
        {"o_m16", 16, 8192, 8192, 0, 0, false},         {"o_m128", 128, 8192, 8192, 0, 0, false},
        {"o_m512", 512, 8192, 8192, 0, 0, true},        {"o_m2048", 2048, 8192, 8192, 0, 0, false},
        {"gateup_m16", 16, 57344, 8192, 0, 0, false},   {"gateup_m128", 128, 57344, 8192, 0, 0, false},
        {"gateup_m512", 512, 57344, 8192, 0, 0, false}, {"gateup_m2048", 2048, 57344, 8192, 0, 2, false},
        {"down_m16", 16, 8192, 28672, 0, 0, false},     {"down_m128", 128, 8192, 28672, 0, 0, false},
        {"down_m512", 512, 8192, 28672, 5, 0, false},   {"down_m2048", 2048, 8192, 28672, 0, 4, false},
        {"ladder", 3456, 4096, 2048, 0, 0, false}};
    int wrong = 0;
    for (const Row& r : rows) {
        const int f = step7::inplace_parts(r.M, r.N, r.K, 108);
        const int e = step6::tail_parts(r.M, r.N, r.K, 108);
        const bool g = step7::phase_in_place(r.M, r.N, r.K, 108);
        if (f != r.f || e != r.e || g != r.g) {
            printf("EMU ERROR: step 7's rules at %s: 7c %d parts, 7e %d, 7f %s; expected 7c %d, 7e %d, 7f %s\n", r.name,
                   f, e, g ? "yes" : "no", r.f, r.e, r.g ? "yes" : "no");
            wrong++;
        }
    }
    snprintf(g_label, sizeof g_label, "%-33s %-27s", "step7 rules of 7c 7e 7f, 108 SMs", "the judge's 17 shapes");
    totals.runs++;
    if (!wrong) {
        printf("%s  ok      7c at down_m512 (5 parts), 7e at down_m2048 (4) and gateup_m2048 (2), 7f at o_m512\n",
               g_label);
        return true;
    }
    printf("%s  FAILED  [%d shapes]\n", g_label, wrong);
    totals.failed++;
    totals.failures.push_back("step7 rules of 7c, 7e and 7f at the judge's shapes");
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
        else if (a == "step7") sel |= kStep7;
        else if (a == "all") sel |= kAll;
        else {
            printf("usage: %s [step0|step1|step2|step3|step4|step5|step6|step7|all]... [-x]\n"
                   "  step0  step 0, nvidia_sample.cuh's compute_gemm (108 and 3 blocks), and its simple_wmma_gemm\n"
                   "  step1  step 1's v0 to v3 (step1.cuh; v4 is step 0's compute_gemm)\n"
                   "  step2  step 2's v0 to v2 (step2.cuh; v2 is step 1's v0); v0 and v1 must give the same bits\n"
                   "  step3  step 3's v0 to v3 (step3.cuh; v3 is step 2's v0); all must give step 2's bits\n"
                   "  step4  step 4's v0 to v3 (step4.cuh; v3 is step 3's v0), all with step 2's bits, and the two\n"
                   "         orders of the tiles of step4::tile_of at every shape up to 8192 x 12288\n"
                   "  step5  step 5's v0 to v3 (step5.cuh; v3 is step 4's v0), all with step 2's bits\n"
                   "  step6  step 6's new tile (step6.cuh) and step 5's v0, all with step 2's bits; 6b's bands and\n"
                   "         stream (M <= 16); 6c's stream in phase (108 and 7 blocks), all without 7d's hints; the\n"
                   "         orders of their tiles, step6::choose at the judge's 17 shapes, step6::m16 at seven of\n"
                   "         them, phase at the 17\n"
                   "  step7  step 7's kernels: 7a's parts and its in-place split (step6.cuh; other bits); 7c's\n"
                   "         parts in place (5 and 2); 7g's parts reduced by their own blocks (step7.cuh), with and\n"
                   "         without 7d's hints, with 7a's parts (the same bits); 6b's, 7b's and 6c's kernels with\n"
                   "         7d's hints; 7h's reads together in the parts in place, and in 6c's kernel with and\n"
                   "         without 7d's hints (the same bits); 7e's tail in parts (108 and 6 SMs, with and without\n"
                   "         7d's hints; other bits); step6::choose, parallel_fits, 7c's, 7e's and 7f's rules at the\n"
                   "         judge's 17 shapes\n"
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
    g_stacks0 = g_stacks;
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
    printf("gemm-ladder emulator, Ampere rung:%s%s%s%s%s%s%s%s; shapes of verif.cu; each run in forward, reverse and "
           "random order\n", sel & kStep0 ? " step0" : "", sel & kStep1 ? " step1" : "", sel & kStep2 ? " step2" : "",
           sel & kStep3 ? " step3" : "", sel & kStep4 ? " step4" : "", sel & kStep5 ? " step5" : "",
           sel & kStep6 ? " step6" : "", sel & kStep7 ? " step7" : "");
    printf("(step 1's v4 is step 0's compute_gemm, step 2's v2 is step 1's v0, step 3's v3 is step 2's v0, step 4's v3 "
           "is step 3's v0 and step 5's v3 is step 4's v0: each "
           "kernel runs once, under its own name; wavefronts per shared instruction after err; step 3's cp.async "
           "late in the forward order, early in the reverse, mixed in the random)\n");
    Totals totals;
    const TileOrder orders[] = {{"step4 tile_of<true>, 4b's order", kStep4, step4::tile_of<true>},
                                {"step4 tile_of<false>, step 0's", kStep4, step4::tile_of<false>},
                                {"step6 tile_of<128, 128>", kStep6, step6::tile_of<128, 128>, 128, 128}};
    for (const TileOrder& o : orders)
        if ((o.in & sel) && !check_tiles(o, totals) && stopAtFirst) return summary(totals, true);
    if ((sel & (kStep6 | kStep7)) && !check_choice(totals) && stopAtFirst) return summary(totals, true);
    if ((sel & kStep6) && !check_m16(totals) && stopAtFirst) return summary(totals, true);
    if ((sel & kStep7) && !check_parallel(totals) && stopAtFirst) return summary(totals, true);
    if ((sel & kStep6) && !check_phase(totals) && stopAtFirst) return summary(totals, true);
    if ((sel & kStep7) && !check_rules_7(totals) && stopAtFirst) return summary(totals, true);
    // The first six shapes of verif.cu: M = 16 (one tile cut by M), a cut tile after a full one, tiles of 128 x 256 and
    // 128 x 128 in 1 to 3 rows, 63 and 126 tiles (the persistent grid of 3 blocks goes through many), 10 rows of tiles;
    // for steps 3 to 7, two more of verif.cu with 4 and 16 slices of K64: with at most 3 slices (K <= 192), the three
    // stages never come round, and a stage index that does not turn would pass; for 6b's stream (and step 7),
    // 16 x 6144 x 128.
    constexpr unsigned kStages3 = kStep3 | kStep4 | kStep5 | kStep6 | kStep7;
    struct Shape { int M, N, K; unsigned in; };
    const Shape shapes[] = {{16, 256, 64, kAll},       {144, 256, 64, kAll},   {128, 512, 128, kAll},
                            {144, 512, 256, kStages3}, {208, 512, 192, kAll}, {16, 1024, 1024, kStages3},
                            {272, 5376, 128, kAll},    {1168, 512, 64, kAll},  {16, 6144, 128, kStep6 | kStep7}};
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
