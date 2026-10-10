// Host shim of cuda_runtime.h: just enough for the kernel headers of turing/ to compile as C++ in the emulator
// (../emu.cpp). The qualifiers vanish; the built-in indices are globals that the emulator sets before it resumes a
// thread; __syncthreads() suspends the calling thread until the whole block has reached it; int4, the type of every
// 16-byte load and store of the kernels, routes them through emu_copy.
#pragma once
#include <cstddef>
#include <cstdint>

#define __device__
#define __host__
#define __forceinline__ inline
#define __global__
#define __shared__
#define __align__(n)

struct emu_dim3 { unsigned x, y, z; };
extern emu_dim3 threadIdx, blockIdx, gridDim, blockDim;
constexpr int warpSize = 32;
typedef struct CUstream_st* cudaStream_t;

inline int min(int a, int b) { return a < b ? a : b; }
void __syncthreads();
int emu_lane();   // the running thread's lane (its linear index in the block, modulo 32)

// A load or store of `bytes` bytes from src to dst by the running thread, checked first (alignment; bounds of the
// launch's shared memory and of A, B and C; no store to A or B), recorded (a shared access for the bank model, a write
// to C for the exactly-once check), then performed; an access found invalid is reported and skipped.
void emu_copy(void* dst, const void* src, int bytes);

struct alignas(16) int4 {
    int x, y, z, w;
    int4() = default;
    int4(const int4& o) { emu_copy(this, &o, sizeof o); }
    int4& operator=(const int4& o) { emu_copy(this, &o, sizeof o); return *this; }
};
