// Host shim of mma.h (wmma), for step 0, step 1 and simple_wmma_gemm. A warp's collective operation is performed on
// whole 16 x 16 tiles by its lane 0, when lane 0 runs; the other lanes skip it (their fragments are never read). A
// fragment holds its whole tile, row-major, so the element-wise conversion of an fp32 accumulator into an fp16 one is
// exact here whatever wmma's per-lane layout: the emulator checks how the kernels use wmma at the tile level (pointer,
// pitch, alignment, data in place when the load runs), not that layout. Loads and stores go through emu_tile.
#pragma once
#include <type_traits>
#include "cuda_runtime.h"
#include "cuda_fp16.h"

// The 16 rows of a 16 x 16 tile at p, at a pitch of ldm elements of elemBytes bytes, read or written by a wmma load or
// store: checked as emu_copy's accesses, and against wmma's rules (p 32-byte aligned, the pitch a multiple of 16
// bytes); a store to C counts its writes. Returns false if the access is invalid; it is then skipped.
bool emu_tile(const void* p, unsigned ldm, int elemBytes, bool store);
void emu_wmma_error(const char* what);   // a use of wmma that this shim does not emulate

namespace nvcuda { namespace wmma {
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
enum layout_t { mem_row_major, mem_col_major };

template <class Use, int m, int n, int k, class T, class Layout = void>
struct fragment {
    static_assert(m == 16 && n == 16 && k == 16, "only 16 x 16 x 16 fragments are emulated");
    static_assert(!std::is_same<Layout, col_major>::value, "only row-major fragments are emulated");
    static constexpr int num_elements = 256;
    T x[256];   // the whole tile, row-major (a real fragment holds 8 elements per lane)
};

template <class Use, class T, class L>
inline void load_matrix_sync(fragment<Use, 16, 16, 16, T, L>& f, const T* p, unsigned ldm)
{
    if (emu_lane() != 0 || !emu_tile(p, ldm, sizeof(T), false)) return;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) f.x[r * 16 + c] = p[(size_t)r * ldm + c];
}

// D = A B + C, fp32 accumulator: each product of two halves is exact in fp32, each sum rounded to fp32.
template <class LA, class LB>
inline void mma_sync(fragment<accumulator, 16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, half, LA>& a,
                     const fragment<matrix_b, 16, 16, 16, half, LB>& b,
                     const fragment<accumulator, 16, 16, 16, float>& c)
{
    if (emu_lane() != 0) return;
    float t[256];
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            float s = c.x[i * 16 + j];
            for (int k = 0; k < 16; k++) s += (float)a.x[i * 16 + k] * (float)b.x[k * 16 + j];
            t[i * 16 + j] = s;
        }
    for (int e = 0; e < 256; e++) d.x[e] = t[e];
}

template <class Use, int m, int n, int k, class T, class L, class V>
inline void fill_fragment(fragment<Use, m, n, k, T, L>& f, V v)
{
    for (int e = 0; e < 256; e++) f.x[e] = (T)v;
}

template <class T>
inline void store_matrix_sync(T* p, const fragment<accumulator, 16, 16, 16, T>& f, unsigned ldm, layout_t layout)
{
    if (emu_lane() != 0) return;
    if (layout != mem_row_major) { emu_wmma_error("store_matrix_sync with mem_col_major"); return; }
    if (!emu_tile(p, ldm, sizeof(T), true)) return;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) p[(size_t)r * ldm + c] = f.x[r * 16 + c];
}
}}  // namespace nvcuda::wmma
