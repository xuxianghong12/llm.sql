/*
 * llm_ops_x86_common.c - Shared x86 helper kernels and blob allocators
 *
 * This file holds cross-cutting helpers that multiple operator groups depend
 * on: base SIMD kernels, legacy blob allocators, and shared N-D shape helpers.
 *
 * Maintenance guidance:
 *   - Keep only reusable helpers here. SQL-facing wrappers belong in more
 *     specific files so registration can stay decoupled from implementation.
 *   - Any helper added here should already have more than one caller.
 *   - Preserve blob/header layout compatibility when touching alloc helpers.
 */

#include "llm_ops_x86_internal.h"
SQLITE_EXTENSION_INIT3

void k_add_simd(const float *a, const float *b, float *c, int n) {
#if LLM_HAVE_AVX2
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(
            c + i,
            _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++)
        c[i] = a[i] + b[i];
#else
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
#endif
}

float k_dot_simd(const float *a, const float *b, int n) {
#if LLM_HAVE_AVX2
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8)
        acc = _mm256_fmadd_ps(
            _mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    float sum = llm_hsum256(acc);
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
#endif
}

void *alloc_vec_blob(int n, const float *data, int *sz_out) {
    int sz = LLM_VEC_BLOB_SIZE(n);
    void *blob = sqlite3_malloc(sz);
    if (!blob)
        return NULL;
    *(int32_t *)blob = n;
    if (data)
        memcpy((char *)blob + 4, data, (size_t)n * sizeof(float));
    else
        memset((char *)blob + 4, 0, (size_t)n * sizeof(float));
    *sz_out = sz;
    return blob;
}

void *alloc_mat_blob(int rows, int cols, const float *data, int *sz_out) {
    int sz = LLM_MAT_BLOB_SIZE(rows, cols);
    void *blob = sqlite3_malloc(sz);
    if (!blob)
        return NULL;
    *(int32_t *)blob = rows;
    *((int32_t *)blob + 1) = cols;
    int nel = rows * cols;
    if (data)
        memcpy((char *)blob + 8, data, (size_t)nel * sizeof(float));
    else
        memset((char *)blob + 8, 0, (size_t)nel * sizeof(float));
    *sz_out = sz;
    return blob;
}

void nd_strides(const int32_t *shape, int ndim, int32_t *strides) {
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        strides[i] = strides[i + 1] * shape[i + 1];
}

void *nd_result_alloc(int ndim,
                      const int32_t *shape,
                      int32_t total,
                      int *sz_out) {
    int hdr = LLM_ND_HDR(ndim);
    int sz = hdr + total * (int)sizeof(float);
    char *blob = (char *)sqlite3_malloc(sz);
    if (!blob)
        return NULL;
    *(int32_t *)blob = LLM_ND_MAGIC;
    *((int32_t *)blob + 1) = ndim;
    for (int i = 0; i < ndim; i++)
        ((int32_t *)blob)[2 + i] = shape[i];
    *sz_out = sz;
    return blob;
}

int32_t broadcast_shape(const TensorNDF32 *ta,
                        const TensorNDF32 *tb,
                        int32_t *out_shape,
                        int *out_ndim) {
    int ndim = ta->ndim > tb->ndim ? ta->ndim : tb->ndim;
    *out_ndim = ndim;
    int off_a = ndim - ta->ndim;
    int off_b = ndim - tb->ndim;
    int32_t total = 1;
    for (int i = 0; i < ndim; i++) {
        int32_t da = (i >= off_a) ? ta->shape[i - off_a] : 1;
        int32_t db = (i >= off_b) ? tb->shape[i - off_b] : 1;
        if (da != db && da != 1 && db != 1)
            return -1;
        out_shape[i] = da > db ? da : db;
        total *= out_shape[i];
    }
    return total;
}