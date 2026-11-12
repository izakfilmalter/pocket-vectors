// pocket-vectors: brute-force scoring kernels (NEON on arm64, scalar elsewhere). MIT licence.
#include "pv_kernels.h"

#include <stdatomic.h>
#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// The ARMv8.2 dot-product instruction (sdot) is on every Apple chip that runs iOS 27 / macOS 27, but
// not on every ARMv8 CPU, so it is checked once at run time and a widening-multiply kernel covers the
// rest. Atomic: several threads may search at once; racing first callers store the same value.
static atomic_int pv_dotprod_state = -1;  // -1 = not checked yet

int pv_has_dotprod(void) {
    int s = atomic_load_explicit(&pv_dotprod_state, memory_order_acquire);
    if (s < 0) {
        int v = 0;
#if defined(__APPLE__) && defined(__aarch64__)
        size_t n = sizeof v;
        if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &n, NULL, 0) != 0) v = 0;
#endif
        s = v ? 1 : 0;
        atomic_store_explicit(&pv_dotprod_state, s, memory_order_release);
    }
    return s;
}

// Tests only: overrides the check for every thread (don't call it while searches run).
void pv_force_dotprod(int on) { atomic_store_explicit(&pv_dotprod_state, on ? 1 : 0, memory_order_release); }

void pv_dot_f16(const float *query, const uint16_t *rows, size_t n, size_t dim, float *scores) {
#if defined(__aarch64__)
    for (size_t i = 0; i < n; i++) {
        const float16_t *r = (const float16_t *)(rows + i * dim);
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        for (size_t d = 0; d < dim; d += 16) {
            float16x8_t h0 = vld1q_f16(r + d), h1 = vld1q_f16(r + d + 8);
            a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(h0)), vld1q_f32(query + d));
            a1 = vfmaq_f32(a1, vcvt_high_f32_f16(h0), vld1q_f32(query + d + 4));
            a2 = vfmaq_f32(a2, vcvt_f32_f16(vget_low_f16(h1)), vld1q_f32(query + d + 8));
            a3 = vfmaq_f32(a3, vcvt_high_f32_f16(h1), vld1q_f32(query + d + 12));
        }
        scores[i] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    }
#else
    for (size_t i = 0; i < n; i++) {
        const _Float16 *r = (const _Float16 *)(rows + i * dim);
        float s = 0;
        for (size_t d = 0; d < dim; d++) s += (float)r[d] * query[d];
        scores[i] = s;
    }
#endif
}

#if defined(__aarch64__)
__attribute__((target("dotprod")))
static void dot_i8_sdot(const int8_t *query, float qscale, const int8_t *rows, const float *rowscale,
                        size_t n, size_t dim, float *scores) {
    for (size_t i = 0; i < n; i++) {
        const int8_t *r = rows + i * dim;
        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        size_t d = 0;
        for (; d + 32 <= dim; d += 32) {
            a0 = vdotq_s32(a0, vld1q_s8(r + d), vld1q_s8(query + d));
            a1 = vdotq_s32(a1, vld1q_s8(r + d + 16), vld1q_s8(query + d + 16));
        }
        for (; d < dim; d += 16) a0 = vdotq_s32(a0, vld1q_s8(r + d), vld1q_s8(query + d));
        scores[i] = (float)vaddvq_s32(vaddq_s32(a0, a1)) * qscale * rowscale[i];
    }
}
#endif

#if defined(__aarch64__)
// Without sdot: widening multiplies (int8 x int8 fits int16) accumulated pairwise into int32.
static void dot_i8_widen(const int8_t *query, float qscale, const int8_t *rows, const float *rowscale,
                         size_t n, size_t dim, float *scores) {
    for (size_t i = 0; i < n; i++) {
        const int8_t *r = rows + i * dim;
        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        for (size_t d = 0; d < dim; d += 16) {
            int8x16_t x = vld1q_s8(r + d), q = vld1q_s8(query + d);
            a0 = vpadalq_s16(a0, vmull_s8(vget_low_s8(x), vget_low_s8(q)));
            a1 = vpadalq_s16(a1, vmull_high_s8(x, q));
        }
        scores[i] = (float)vaddvq_s32(vaddq_s32(a0, a1)) * qscale * rowscale[i];
    }
}
#endif

void pv_dot_i8(const int8_t *query, float qscale, const int8_t *rows, const float *rowscale, size_t n,
               size_t dim, float *scores) {
#if defined(__aarch64__)
    if (pv_has_dotprod()) dot_i8_sdot(query, qscale, rows, rowscale, n, dim, scores);
    else dot_i8_widen(query, qscale, rows, rowscale, n, dim, scores);
#else
    for (size_t i = 0; i < n; i++) {
        const int8_t *r = rows + i * dim;
        int32_t s = 0;
        for (size_t d = 0; d < dim; d++) s += (int32_t)r[d] * query[d];
        scores[i] = (float)s * qscale * rowscale[i];
    }
#endif
}

size_t pv_topk(const float *scores, size_t n, size_t k, uint32_t *outIdx, float *outScore) {
    if (k > 1024) k = 1024;
    if (k > n) k = n;
    if (k == 0) return 0;
    // sorted insertion into a small array (best first); most scores fail the threshold test
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        float s = scores[i];
        if (m == k && s <= outScore[k - 1]) continue;
        size_t j = (m < k) ? m++ : k - 1;
        while (j > 0 && outScore[j - 1] < s) {
            outScore[j] = outScore[j - 1];
            outIdx[j] = outIdx[j - 1];
            j--;
        }
        outScore[j] = s;
        outIdx[j] = (uint32_t)i;
    }
    return m;
}
