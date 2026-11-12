// pocket-vectors: brute-force scoring kernels (NEON on arm64, scalar elsewhere). MIT licence.
// Rows are contiguous, `dim` values each; scores[i] = dot(query, row i).
#ifndef PV_KERNELS_H
#define PV_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// fp16 rows, fp32 query, fp32 accumulate. dim % 16 == 0.
void pv_dot_f16(const float *query, const uint16_t *rows, size_t n, size_t dim, float *scores);

/// int8 rows (per-row scale), int8 query (one scale). scores[i] = dot * qscale * rowscale[i].
/// Uses the ARMv8.2 dot-product instruction when available. dim % 16 == 0.
void pv_dot_i8(const int8_t *query, float qscale, const int8_t *rows, const float *rowscale, size_t n,
               size_t dim, float *scores);

/// 1 if the CPU has the ARMv8.2 dot-product instruction (checked once via sysctl); pv_dot_i8 uses it
/// when present and a widening-multiply kernel otherwise. pv_force_dotprod is for tests.
int pv_has_dotprod(void);
void pv_force_dotprod(int on);

/// Indexes of the k largest scores, best first (k <= 1024). Returns how many were written.
size_t pv_topk(const float *scores, size_t n, size_t k, uint32_t *outIdx, float *outScore);

#ifdef __cplusplus
}
#endif
#endif
