// Host shim of cuda_fp16.h: half is the compiler's _Float16 (its conversion from float rounds to nearest even, as
// __float2half); __half2, the type of the 4-byte stores of step 2's epilogue, routes its loads and stores through
// emu_copy, as int4 does.
#pragma once
#include "cuda_runtime.h"

typedef _Float16 half;
inline half __float2half(float f) { return (half)f; }
inline float __half2float(half h) { return (float)h; }

struct alignas(4) __half2 {
    half x, y;
    __half2() = default;
    __half2(const __half2& o) { emu_copy(this, &o, sizeof o); }
    __half2& operator=(const __half2& o) { emu_copy(this, &o, sizeof o); return *this; }
};
inline __half2 __floats2half2_rn(float a, float b)
{
    __half2 r;
    r.x = (half)a;
    r.y = (half)b;
    return r;
}
