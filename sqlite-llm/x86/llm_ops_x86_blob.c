/*
 * llm_ops_x86_blob.c - Legacy vector/matrix blob SQL wrappers
 *
 * This file owns the pre-N-D SQL surface for constructing, inspecting, and
 * doing simple vector arithmetic on legacy vector/matrix blob formats.
 *
 * Maintenance guidance:
 *   - Keep legacy blob parsing and formatting localized here so future changes
 *     to the compatibility layer do not leak into N-D or model-specific files.
 *   - Reuse alloc_vec_blob/alloc_mat_blob and shared kernels from the common
 *     layer instead of duplicating buffer management.
 */

#include "llm_ops_x86_internal.h"
SQLITE_EXTENSION_INIT3

static void k_sub_simd(const float *a, const float *b, float *c, int n) {
#if LLM_HAVE_AVX2
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(
            c + i,
            _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++)
        c[i] = a[i] - b[i];
#else
    for (int i = 0; i < n; i++)
        c[i] = a[i] - b[i];
#endif
}

static void k_mul_simd(const float *a, const float *b, float *c, int n) {
#if LLM_HAVE_AVX2
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(
            c + i,
            _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++)
        c[i] = a[i] * b[i];
#else
    for (int i = 0; i < n; i++)
        c[i] = a[i] * b[i];
#endif
}

static void k_div_simd(const float *a, const float *b, float *c, int n) {
#if LLM_HAVE_AVX2
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(
            c + i,
            _mm256_div_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++)
        c[i] = a[i] / b[i];
#else
    for (int i = 0; i < n; i++)
        c[i] = a[i] / b[i];
#endif
}

static void k_scale_simd(const float *a, float alpha, float *b, int n) {
#if LLM_HAVE_AVX2
    __m256 va = _mm256_set1_ps(alpha);
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(b + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), va));
    for (; i < n; i++)
        b[i] = alpha * a[i];
#else
    for (int i = 0; i < n; i++)
        b[i] = alpha * a[i];
#endif
}

static void k_add_blas(const float *a, const float *b, float *c, int n) {
    memcpy(c, b, (size_t)n * sizeof(float));
    cblas_saxpy(n, 1.0f, a, 1, c, 1);
}

static void k_mul_blas(const float *a, const float *b, float *c, int n) {
    k_mul_simd(a, b, c, n);
}

static void k_scale_blas(const float *a, float alpha, float *b, int n) {
    memcpy(b, a, (size_t)n * sizeof(float));
    cblas_sscal(n, alpha, b, 1);
}

static float k_dot_blas(const float *a, const float *b, int n) {
    return cblas_sdot(n, a, 1, b, 1);
}

void sql_vec(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    int n = sqlite3_value_int(argv[0]);
    float val = (float)sqlite3_value_double(argv[1]);
    if (n <= 0)
        ERR(ctx, "n must be positive");
    int sz = LLM_VEC_BLOB_SIZE(n);
    void *blob = sqlite3_malloc(sz);
    if (!blob) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    *(int32_t *)blob = n;
    float *data = (float *)((char *)blob + 4);
    for (int i = 0; i < n; i++)
        data[i] = val;
    sqlite3_result_blob(ctx, blob, sz, sqlite3_free);
}

void sql_mat(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    int rows = sqlite3_value_int(argv[0]);
    int cols = sqlite3_value_int(argv[1]);
    float val = (float)sqlite3_value_double(argv[2]);
    if (rows <= 0 || cols <= 0)
        ERR(ctx, "rows and cols must be positive");
    int sz = LLM_MAT_BLOB_SIZE(rows, cols);
    void *blob = sqlite3_malloc(sz);
    if (!blob) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    *(int32_t *)blob = rows;
    *((int32_t *)blob + 1) = cols;
    float *data = (float *)((char *)blob + 8);
    for (int i = 0; i < rows * cols; i++)
        data[i] = val;
    sqlite3_result_blob(ctx, blob, sz, sqlite3_free);
}

void sql_vget(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 v;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &v) < 0)
        ERR(ctx, "Invalid vector blob");
    int idx = sqlite3_value_int(argv[1]);
    if (idx < 0 || idx >= v.n)
        ERR(ctx, "Index out of range");
    sqlite3_result_double(ctx, (double)v.data[idx]);
}

void sql_mget(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 m;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &m) < 0)
        ERR(ctx, "Invalid matrix blob");
    int r = sqlite3_value_int(argv[1]);
    int c = sqlite3_value_int(argv[2]);
    if (r < 0 || r >= m.rows || c < 0 || c >= m.cols)
        ERR(ctx, "Index out of range");
    sqlite3_result_double(ctx, (double)m.data[(size_t)r * m.cols + c]);
}

void sql_len(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 v;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &v) < 0)
        ERR(ctx, "Invalid vector blob");
    sqlite3_result_int(ctx, v.n);
}

void sql_rows(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 m;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &m) < 0)
        ERR(ctx, "Invalid matrix blob");
    sqlite3_result_int(ctx, m.rows);
}

void sql_cols(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 m;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &m) < 0)
        ERR(ctx, "Invalid matrix blob");
    sqlite3_result_int(ctx, m.cols);
}

void sql_str(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int bsz = sqlite3_value_bytes(argv[0]);
    if (!blob || bsz < 4)
        ERR(ctx, "Invalid blob");

    VecF32 v;
    MatF32 m;
    int is_vec = (llm_parse_vec(blob, bsz, &v) == 0 &&
                  bsz == (int)(4 + (size_t)v.n * 4));
    int is_mat = (!is_vec && llm_parse_mat(blob, bsz, &m) == 0);
    if (!is_vec && !is_mat)
        ERR(ctx, "Cannot parse blob as vector or matrix");

    if (is_vec) {
        int cap = v.n * 18 + 4;
        char *buf = sqlite3_malloc(cap);
        if (!buf) {
            sqlite3_result_error_nomem(ctx);
            return;
        }
        int pos = 0;
        buf[pos++] = '[';
        for (int i = 0; i < v.n; i++) {
            if (i) {
                buf[pos++] = ',';
                buf[pos++] = ' ';
            }
            int rem = cap - pos;
            if (rem <= 0)
                break;
            int w = snprintf(buf + pos, (size_t)rem, "%.6g", (double)v.data[i]);
            if (w > 0 && w < rem)
                pos += w;
        }
        buf[pos++] = ']';
        buf[pos] = '\0';
        sqlite3_result_text(ctx, buf, pos, sqlite3_free);
        return;
    }

    int nel = m.rows * m.cols;
    int cap = nel * 18 + m.rows * 6 + 8;
    char *buf = sqlite3_malloc(cap);
    if (!buf) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    int pos = 0;
    buf[pos++] = '[';
    for (int i = 0; i < m.rows; i++) {
        if (i) {
            buf[pos++] = ',';
            buf[pos++] = ' ';
        }
        buf[pos++] = '[';
        for (int j = 0; j < m.cols; j++) {
            if (j) {
                buf[pos++] = ',';
                buf[pos++] = ' ';
            }
            int rem = cap - pos;
            if (rem <= 0)
                break;
            int w = snprintf(buf + pos,
                             (size_t)rem,
                             "%.6g",
                             (double)m.data[(size_t)i * m.cols + j]);
            if (w > 0 && w < rem)
                pos += w;
        }
        buf[pos++] = ']';
    }
    buf[pos++] = ']';
    buf[pos] = '\0';
    sqlite3_result_text(ctx, buf, pos, sqlite3_free);
}

void sql_add_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    EMIT_VEC_BINOP(ctx, va, k_add_simd(va.data, vb.data, tmp, va.n));
}

void sql_add_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    EMIT_VEC_BINOP(ctx, va, k_add_blas(va.data, vb.data, tmp, va.n));
}

void sql_sub_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    EMIT_VEC_BINOP(ctx, va, k_sub_simd(va.data, vb.data, tmp, va.n));
}

void sql_mul_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    EMIT_VEC_BINOP(ctx, va, k_mul_simd(va.data, vb.data, tmp, va.n));
}

void sql_mul_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    EMIT_VEC_BINOP(ctx, va, k_mul_blas(va.data, vb.data, tmp, va.n));
}

void sql_div_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    EMIT_VEC_BINOP(ctx, va, k_div_simd(va.data, vb.data, tmp, va.n));
}

void sql_scale_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    float alpha = (float)sqlite3_value_double(argv[1]);
    EMIT_VEC_BINOP(ctx, va, k_scale_simd(va.data, alpha, tmp, va.n));
}

void sql_scale_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    float alpha = (float)sqlite3_value_double(argv[1]);
    EMIT_VEC_BINOP(ctx, va, k_scale_blas(va.data, alpha, tmp, va.n));
}

void sql_dot_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    sqlite3_result_double(ctx, (double)k_dot_simd(va.data, vb.data, va.n));
}

void sql_dot_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    sqlite3_result_double(ctx, (double)k_dot_blas(va.data, vb.data, va.n));
}