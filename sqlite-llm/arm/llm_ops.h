/*
 * llm_ops.h - SQLite LLM Operations Plugin: shared types and helpers
 *
 * Binary blob layout
 * ------------------
 * Vector (1-D): [n:int32][v0:float32 ... v(n-1):float32]
 * Matrix (2-D): [rows:int32][cols:int32][m_00:float32 ... m_(R-1)(C-1):float32]
 * (row-major) N-D Tensor:
 * [LLM_ND_MAGIC:int32][ndim:int32][d0:int32]...[d(ndim-1):int32][data:float32...]
 *
 * The LLM_ND_MAGIC marker (INT32_MIN = 0x80000000) distinguishes N-D blobs
 * from legacy 1-D / 2-D blobs, which always start with a non-negative int32.
 * All N-D functions accept legacy vector/matrix blobs transparently (via the
 * llm_to_nd() helper that promotes them to the unified TensorNDF32 view).
 *
 * All floating-point data is stored as IEEE 754 single precision (float32),
 * row-major (C-contiguous) order.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LLM_OPS_H
#define LLM_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── SIMD availability ──────────────────────────────────────────────────────
 *
 * ARM-specific kernels must gate vectorized paths through LLM_HAVE_NEON.
 * Any code that is not explicitly written for NEON falls back to the scalar
 * reference implementation in the owning source file.
 */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define LLM_HAVE_NEON 1
#else
#define LLM_HAVE_NEON 0
#endif

/* ── Compiler hints ─────────────────────────────────────────────────────────
 */
#if defined(__GNUC__) || defined(__clang__)
#define LLM_LIKELY(x) __builtin_expect(!!(x), 1)
#define LLM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define LLM_INLINE static inline __attribute__((always_inline))
#else
#define LLM_LIKELY(x) (x)
#define LLM_UNLIKELY(x) (x)
#define LLM_INLINE static inline
#endif

#if LLM_HAVE_NEON
LLM_INLINE float llm_hsumq_f32(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t pair = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    pair = vpadd_f32(pair, pair);
    return vget_lane_f32(pair, 0);
#endif
}

LLM_INLINE float llm_hmaxq_f32(float32x4_t v) {
#if defined(__aarch64__)
    return vmaxvq_f32(v);
#else
    float32x2_t pair = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
    pair = vpmax_f32(pair, pair);
    return vget_lane_f32(pair, 0);
#endif
}

LLM_INLINE float llm_hminq_f32(float32x4_t v) {
#if defined(__aarch64__)
    return vminvq_f32(v);
#else
    float32x2_t pair = vpmin_f32(vget_low_f32(v), vget_high_f32(v));
    pair = vpmin_f32(pair, pair);
    return vget_lane_f32(pair, 0);
#endif
}

LLM_INLINE void
llm_store_dequantize_i8x8(float *out, const int8_t *src, float scale) {
    int8x8_t vi8 = vld1_s8(src);
    int16x8_t vi16 = vmovl_s8(vi8);
    int32x4_t lo = vmovl_s16(vget_low_s16(vi16));
    int32x4_t hi = vmovl_s16(vget_high_s16(vi16));
    float32x4_t vscale = vdupq_n_f32(scale);
    vst1q_f32(out, vmulq_f32(vcvtq_f32_s32(lo), vscale));
    vst1q_f32(out + 4, vmulq_f32(vcvtq_f32_s32(hi), vscale));
}

LLM_INLINE float
llm_dot_i8f32_neon(const float *x, const int8_t *w, int32_t n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t vi8 = vld1_s8(w + i);
        int16x8_t vi16 = vmovl_s8(vi8);
        int32x4_t lo = vmovl_s16(vget_low_s16(vi16));
        int32x4_t hi = vmovl_s16(vget_high_s16(vi16));
        acc0 = vmlaq_f32(acc0, vld1q_f32(x + i), vcvtq_f32_s32(lo));
        acc1 = vmlaq_f32(acc1, vld1q_f32(x + i + 4), vcvtq_f32_s32(hi));
    }
    float acc = llm_hsumq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++)
        acc += x[i] * (float)w[i];
    return acc;
}
#endif

/* ── Blob size calculations ─────────────────────────────────────────────────
 */
#define LLM_VEC_BLOB_SIZE(n) (4 + (int)(n)*4)
#define LLM_MAT_BLOB_SIZE(r, c) (8 + (int)(r) * (int)(c)*4)

/* ── N-D tensor constants ───────────────────────────────────────────────────
 */
/* Magic marker stored in the first 4 bytes of every N-D blob.               */
/* INT32_MIN (0x80000000) is safe: legacy vec/mat blobs always start with a  */
/* non-negative int32 (element count ≥ 0).                                   */
#define LLM_ND_MAGIC ((int32_t)0x80000000)
#define LLM_MAX_NDIM 8 /* maximum supported number of dimensions          */
#define LLM_MAX_CAT_TENSORS 64 /* max tensors for cat_multi / stack         */

/* N-D header size: 4 bytes magic + 4 bytes ndim + ndim * 4 bytes shape      */
#define LLM_ND_HDR(ndim) (8 + (int)(ndim)*4)

/* ── INT8 quantized tensor constants ─────────────────────────────────────── */
/*
 * INT8 quantized N-D blob layout (per-row symmetric quantization):
 *
 *   [LLM_INT8_MAGIC:int32][ndim:int32][d0:int32]...[d(ndim-1):int32]
 *   [scale_0:float32]...[scale_(nrows-1):float32]
 *   [q_0:int8]...[q_(total-1):int8]
 *
 * where nrows = product(shape[0..ndim-2])  (all dims except the last one)
 * For 1-D tensors, nrows = 1.
 * scale_i = max(|row_i|) / 127.0  (absmax symmetric quantization)
 * q_ij = round(data_ij / scale_i)  clamped to [-127, 127]
 */
#define LLM_INT8_MAGIC ((int32_t)0x80000008)

/* INT8 N-D header size: same as FP32 N-D header                             */
#define LLM_INT8_HDR(ndim) (8 + (int)(ndim)*4)

/* INT8 quantized tensor view (points into blob memory).                     */
typedef struct {
    int32_t ndim;
    int32_t shape[LLM_MAX_NDIM];
    int32_t total;   /* product of all shape elements                     */
    int32_t nrows;   /* product of shape[0..ndim-2]                       */
    int32_t row_len; /* shape[ndim-1] — elements per row                 */
    float *scales;   /* one scale per row (nrows floats)                  */
    int8_t *data;    /* quantized int8 payload                            */
} TensorNDInt8;

/* Parse an INT8 quantized N-D tensor blob; returns 0 on success.            */
LLM_INLINE int llm_parse_int8(const void *blob, int sz, TensorNDInt8 *t) {
    if (LLM_UNLIKELY(!blob || sz < 8))
        return -1;
    const int32_t *h = (const int32_t *)blob;
    if (h[0] != LLM_INT8_MAGIC)
        return -1;
    t->ndim = h[1];
    if (LLM_UNLIKELY(t->ndim < 1 || t->ndim > LLM_MAX_NDIM))
        return -1;
    int hdr = LLM_INT8_HDR(t->ndim);
    if (LLM_UNLIKELY(sz < hdr))
        return -1;
    t->total = 1;
    for (int i = 0; i < t->ndim; i++) {
        t->shape[i] = h[2 + i];
        if (LLM_UNLIKELY(t->shape[i] < 0))
            return -1;
        t->total *= t->shape[i];
    }
    t->row_len = t->shape[t->ndim - 1];
    t->nrows = (t->row_len > 0) ? t->total / t->row_len : 0;
    int scales_bytes = t->nrows * 4; /* float32 per row */
    int data_bytes = t->total;       /* int8 per element */
    if (LLM_UNLIKELY(sz < hdr + scales_bytes + data_bytes))
        return -1;
    t->scales = (float *)((const char *)blob + hdr);
    t->data = (int8_t *)((const char *)blob + hdr + scales_bytes);
    return 0;
}

/* Allocate and fill a new INT8 quantized tensor blob.
 * scales: float32 array of nrows entries
 * data: int8 array of total entries
 * Returns NULL on allocation failure.                                       */
LLM_INLINE void *llm_build_int8(int ndim,
                                const int32_t *shape,
                                const float *scales,
                                const int8_t *data,
                                int *sz_out) {
    if (ndim < 1 || ndim > LLM_MAX_NDIM)
        return NULL;
    int32_t total = 1;
    for (int i = 0; i < ndim; i++)
        total *= shape[i];
    int32_t row_len = shape[ndim - 1];
    int32_t nrows = (row_len > 0) ? total / row_len : 0;
    int hdr = LLM_INT8_HDR(ndim);
    int scales_bytes = nrows * 4;
    int data_bytes = total;
    int sz = hdr + scales_bytes + data_bytes;
    char *b = (char *)malloc((size_t)sz);
    if (!b)
        return NULL;
    *(int32_t *)b = LLM_INT8_MAGIC;
    *((int32_t *)b + 1) = ndim;
    for (int i = 0; i < ndim; i++)
        ((int32_t *)b)[2 + i] = shape[i];
    if (scales)
        memcpy(b + hdr, scales, (size_t)scales_bytes);
    else
        memset(b + hdr, 0, (size_t)scales_bytes);
    if (data)
        memcpy(b + hdr + scales_bytes, data, (size_t)data_bytes);
    else
        memset(b + hdr + scales_bytes, 0, (size_t)data_bytes);
    *sz_out = sz;
    return b;
}

/* ── Lightweight view structs (point into existing blob memory) ─────────────
 */
typedef struct {
    int32_t n;
    float *data;
} VecF32;
typedef struct {
    int32_t rows, cols;
    float *data;
} MatF32;

/* N-D tensor view (points into blob memory; does NOT own the buffer).       */
typedef struct {
    int32_t ndim;
    int32_t shape[LLM_MAX_NDIM];
    int32_t total; /* product of all shape elements                       */
    float *data;   /* points into the blob payload                        */
} TensorNDF32;

/* Parse a vector from a raw blob; returns 0 on success, -1 on error. */
LLM_INLINE int llm_parse_vec(const void *blob, int sz, VecF32 *v) {
    if (LLM_UNLIKELY(!blob || sz < 4))
        return -1;
    v->n = *(const int32_t *)blob;
    if (LLM_UNLIKELY(v->n < 0 || sz < LLM_VEC_BLOB_SIZE(v->n)))
        return -1;
    v->data = (float *)((const char *)blob + 4);
    return 0;
}

/* Parse a matrix from a raw blob; returns 0 on success, -1 on error. */
LLM_INLINE int llm_parse_mat(const void *blob, int sz, MatF32 *m) {
    if (LLM_UNLIKELY(!blob || sz < 8))
        return -1;
    m->rows = *(const int32_t *)blob;
    m->cols = *((const int32_t *)blob + 1);
    if (LLM_UNLIKELY(m->rows < 0 || m->cols < 0 ||
                     sz < LLM_MAT_BLOB_SIZE(m->rows, m->cols)))
        return -1;
    m->data = (float *)((const char *)blob + 8);
    return 0;
}

/* Parse an N-D tensor blob; returns 0 on success, -1 on error.
 * Requires the first int32 == LLM_ND_MAGIC.                               */
LLM_INLINE int llm_parse_nd(const void *blob, int sz, TensorNDF32 *t) {
    if (LLM_UNLIKELY(!blob || sz < 8))
        return -1;
    const int32_t *h = (const int32_t *)blob;
    if (h[0] != LLM_ND_MAGIC)
        return -1;
    t->ndim = h[1];
    if (LLM_UNLIKELY(t->ndim < 1 || t->ndim > LLM_MAX_NDIM))
        return -1;
    int hdr = LLM_ND_HDR(t->ndim);
    if (LLM_UNLIKELY(sz < hdr))
        return -1;
    t->total = 1;
    for (int i = 0; i < t->ndim; i++) {
        t->shape[i] = h[2 + i];
        if (LLM_UNLIKELY(t->shape[i] < 0))
            return -1;
        t->total *= t->shape[i];
    }
    if (LLM_UNLIKELY(sz < hdr + t->total * 4))
        return -1;
    t->data = (float *)((const char *)blob + hdr);
    return 0;
}

/* Promote any supported blob format (vec, mat, or nd) to TensorNDF32.
 * Returns 0 on success, -1 on error.  The TensorNDF32 view points directly
 * into the original blob memory — no copy is made.
 * Vec detection uses an exact-size check to avoid mistaking matrix blobs
 * (which always have more bytes than a 1-D vector of equal first element).  */
LLM_INLINE int llm_to_nd(const void *blob, int sz, TensorNDF32 *t) {
    if (llm_parse_nd(blob, sz, t) == 0)
        return 0;
    VecF32 v;
    /* Exact-size check: a matrix blob [rows][cols][data] has sz =
     * 8+rows*cols*4, which is always larger than the 4+rows*4 bytes of a vec
     * blob of length rows.  */
    if (llm_parse_vec(blob, sz, &v) == 0 && sz == LLM_VEC_BLOB_SIZE(v.n)) {
        t->ndim = 1;
        t->shape[0] = v.n;
        t->total = v.n;
        t->data = v.data;
        return 0;
    }
    MatF32 m;
    if (llm_parse_mat(blob, sz, &m) == 0) {
        t->ndim = 2;
        t->shape[0] = m.rows;
        t->shape[1] = m.cols;
        t->total = m.rows * m.cols;
        t->data = m.data;
        return 0;
    }
    return -1;
}

/* Allocate and fill a new N-D tensor blob.  The caller owns the returned
 * buffer and must free() it.  If data == NULL the payload is zeroed.
 * Returns NULL on allocation failure.                                       */
LLM_INLINE void *
llm_build_nd(int ndim, const int32_t *shape, const float *data, int *sz_out) {
    if (ndim < 1 || ndim > LLM_MAX_NDIM)
        return NULL;
    int32_t total = 1;
    for (int i = 0; i < ndim; i++)
        total *= shape[i];
    int hdr = LLM_ND_HDR(ndim);
    int sz = hdr + total * 4;
    char *b = (char *)malloc((size_t)sz);
    if (!b)
        return NULL;
    *(int32_t *)b = LLM_ND_MAGIC;
    *((int32_t *)b + 1) = ndim;
    for (int i = 0; i < ndim; i++)
        ((int32_t *)b)[2 + i] = shape[i];
    if (data)
        memcpy(b + hdr, data, (size_t)total * 4);
    else
        memset(b + hdr, 0, (size_t)total * 4);
    *sz_out = sz;
    return b;
}

/* ── Scalar math helpers ─────────────────────────────────────────────────── */
LLM_INLINE float llm_gelu(float x) {
    /* GELU ≈ x · 0.5 · (1 + tanh(sqrt(2/π) · (x + 0.044715·x³))) */
    static const float c0 = 0.7978845608028654f; /* sqrt(2/π) */
    static const float c1 = 0.044715f;
    float cube = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c0 * (x + c1 * cube)));
}

LLM_INLINE float llm_silu(float x) {
    /* SiLU / Swish: x · σ(x) */
    return x / (1.0f + expf(-x));
}

LLM_INLINE float llm_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

#endif /* LLM_OPS_H */
