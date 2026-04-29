/*
 * llm_ops_arm_int8.c - INT8 quantization and parameter-backed inference wrappers
 *
 * This file owns the quantized tensor path: fp32<->int8 conversion, int8
 * linear layers, int8 embeddings, and parameter-backed lookup wrappers that
 * dispatch between fp32 and int8 weights.
 *
 * Maintenance guidance:
 *   - Keep the thread-pool, dot-product kernels, and wrapper entry points in
 *     one file so performance-sensitive int8 paths can evolve together.
 *   - Parameter-backed operators should reuse the same core kernels as direct
 *     blob-backed operators instead of forking behavior.
 *   - Runtime thread count is shared through g_llm_num_threads; do not hide a
 *     second threading knob inside this file.
 */

#include "llm_ops_arm_internal.h"
SQLITE_EXTENSION_INIT3

typedef struct {
    void (*func)(void *arg, int32_t row_start, int32_t row_end);
    void *arg;
    int32_t total_rows;
    int32_t chunk;
    int tid;
} ThreadArg;

static void *thread_worker(void *raw) {
    ThreadArg *ta = (ThreadArg *)raw;
    int32_t start = ta->tid * ta->chunk;
    int32_t end = start + ta->chunk;
    if (end > ta->total_rows)
        end = ta->total_rows;
    if (start < end)
        ta->func(ta->arg, start, end);
    return NULL;
}

static void parallel_rows(void (*func)(void *, int32_t, int32_t),
                          void *arg,
                          int32_t nrows) {
    int nthreads = g_llm_num_threads;
    if (nrows < nthreads)
        nthreads = nrows;
    if (nthreads <= 1) {
        func(arg, 0, nrows);
        return;
    }

    int32_t chunk = (nrows + nthreads - 1) / nthreads;
    pthread_t threads[LLM_NUM_THREADS];
    ThreadArg args[LLM_NUM_THREADS];
    for (int i = 1; i < nthreads; i++) {
        args[i].func = func;
        args[i].arg = arg;
        args[i].total_rows = nrows;
        args[i].chunk = chunk;
        args[i].tid = i;
        pthread_create(&threads[i], NULL, thread_worker, &args[i]);
    }

    int32_t end0 = chunk < nrows ? chunk : nrows;
    func(arg, 0, end0);
    for (int i = 1; i < nthreads; i++)
        pthread_join(threads[i], NULL);
}

static void *int8_result_alloc(int ndim,
                               const int32_t *shape,
                               int32_t total,
                               int32_t nrows,
                               int *sz_out) {
    int hdr = LLM_INT8_HDR(ndim);
    int scales_bytes = nrows * 4;
    int data_bytes = total;
    int sz = hdr + scales_bytes + data_bytes;
    char *blob = (char *)sqlite3_malloc(sz);
    if (!blob)
        return NULL;
    *(int32_t *)blob = LLM_INT8_MAGIC;
    *((int32_t *)blob + 1) = ndim;
    for (int i = 0; i < ndim; i++)
        ((int32_t *)blob)[2 + i] = shape[i];
    *sz_out = sz;
    return blob;
}

typedef struct {
    const float *src;
    float *scales;
    int8_t *dst;
    int32_t row_len;
} QuantizeArg;

static void quantize_rows(void *raw, int32_t r0, int32_t r1) {
    QuantizeArg *arg = (QuantizeArg *)raw;
    int32_t row_len = arg->row_len;
    for (int32_t row_idx = r0; row_idx < r1; row_idx++) {
        const float *row = arg->src + (size_t)row_idx * row_len;
        int8_t *qrow = arg->dst + (size_t)row_idx * row_len;
        float absmax = 0.0f;
        for (int32_t col_idx = 0; col_idx < row_len; col_idx++) {
            float v = row[col_idx] < 0 ? -row[col_idx] : row[col_idx];
            if (v > absmax)
                absmax = v;
        }

        float scale = absmax / 127.0f;
        arg->scales[row_idx] = scale;
        if (scale < 1e-10f) {
            memset(qrow, 0, (size_t)row_len);
            continue;
        }

        float inv_scale = 127.0f / absmax;
        for (int32_t col_idx = 0; col_idx < row_len; col_idx++) {
            float v = row[col_idx] * inv_scale;
            int q = (int)roundf(v);
            if (q < -127)
                q = -127;
            if (q > 127)
                q = 127;
            qrow[col_idx] = (int8_t)q;
        }
    }
}

void sql_quantize_int8_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx;
    if (llm_to_nd(sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_quantize_int8_nd: invalid tensor");

    int32_t row_len = tx.shape[tx.ndim - 1];
    int32_t nrows = (row_len > 0) ? tx.total / row_len : 0;

    int out_sz;
    void *out = int8_result_alloc(tx.ndim, tx.shape, tx.total, nrows, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    int hdr = LLM_INT8_HDR(tx.ndim);
    float *scales = (float *)((char *)out + hdr);
    int8_t *qdata = (int8_t *)((char *)out + hdr + nrows * 4);

    QuantizeArg arg = {tx.data, scales, qdata, row_len};
    parallel_rows(quantize_rows, &arg, nrows);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

typedef struct {
    const float *scales;
    const int8_t *src;
    float *dst;
    int32_t row_len;
} DequantizeArg;

static void dequantize_rows(void *raw, int32_t r0, int32_t r1) {
    DequantizeArg *arg = (DequantizeArg *)raw;
    int32_t row_len = arg->row_len;
    for (int32_t row_idx = r0; row_idx < r1; row_idx++) {
        const int8_t *qrow = arg->src + (size_t)row_idx * row_len;
        float *out_row = arg->dst + (size_t)row_idx * row_len;
        float scale = arg->scales[row_idx];
        for (int32_t col_idx = 0; col_idx < row_len; col_idx++)
            out_row[col_idx] = (float)qrow[col_idx] * scale;
    }
}

void sql_dequantize_int8_nd(sqlite3_context *ctx,
                            int argc,
                            sqlite3_value **argv) {
    (void)argc;
    TensorNDInt8 tq;
    if (llm_parse_int8(sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tq) < 0)
        ERR(ctx, "llm_dequantize_int8_nd: invalid int8 tensor");

    int out_sz;
    void *out = nd_result_alloc(tq.ndim, tq.shape, tq.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tq.ndim));

    DequantizeArg arg = {tq.scales, tq.data, dst, tq.row_len};
    parallel_rows(dequantize_rows, &arg, tq.nrows);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

LLM_INLINE float dot_i8f32(const float *x, const int8_t *w, int32_t K) {
    float acc = 0.0f;
    for (int32_t k = 0; k < K; k++)
        acc += x[k] * (float)w[k];
    return acc;
}

typedef struct {
    const float *x;
    const int8_t *wq;
    const float *scales;
    float *dst;
    int32_t M, N, K;
} LinearInt8Arg;

static void linear_int8_rows_m(void *raw, int32_t m0, int32_t m1) {
    LinearInt8Arg *arg = (LinearInt8Arg *)raw;
    for (int32_t m = m0; m < m1; m++) {
        const float *xrow = arg->x + (size_t)m * arg->K;
        float *out_row = arg->dst + (size_t)m * arg->N;
        for (int32_t n = 0; n < arg->N; n++) {
            const int8_t *wrow = arg->wq + (size_t)n * arg->K;
            out_row[n] = dot_i8f32(xrow, wrow, arg->K) * arg->scales[n];
        }
    }
}

static void linear_int8_rows_n(void *raw, int32_t n0, int32_t n1) {
    LinearInt8Arg *arg = (LinearInt8Arg *)raw;
    const float *xrow = arg->x;
    for (int32_t n = n0; n < n1; n++) {
        const int8_t *wrow = arg->wq + (size_t)n * arg->K;
        arg->dst[n] = dot_i8f32(xrow, wrow, arg->K) * arg->scales[n];
    }
}

void sql_linear_int8_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx;
    TensorNDInt8 tw;
    if (llm_to_nd(sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_linear_int8_nd: invalid x");
    if (llm_parse_int8(sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tw) < 0)
        ERR(ctx, "llm_linear_int8_nd: invalid int8 weight");
    if (tx.ndim < 1)
        ERR(ctx, "llm_linear_int8_nd: x must be at least 1-D");
    if (tw.ndim != 2)
        ERR(ctx, "llm_linear_int8_nd: weight must be 2-D");

    int32_t K = tx.shape[tx.ndim - 1];
    if (tw.shape[1] != K)
        ERR(ctx, "llm_linear_int8_nd: in_features mismatch");

    int32_t M = tx.total / K;
    int32_t N = tw.shape[0];
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < tx.ndim; i++)
        out_shape[i] = tx.shape[i];
    out_shape[tx.ndim - 1] = N;
    int32_t out_total = M * N;

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    LinearInt8Arg arg = {tx.data, tw.data, tw.scales, dst, M, N, K};
    if (M == 1) {
        if ((int64_t)N * K >= LLM_INT8_THREAD_OPS_MIN)
            parallel_rows(linear_int8_rows_n, &arg, N);
        else
            linear_int8_rows_n(&arg, 0, N);
    } else {
        parallel_rows(linear_int8_rows_m, &arg, M);
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_linear_int8_bias_nd(sqlite3_context *ctx,
                             int argc,
                             sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, tb;
    TensorNDInt8 tw;
    if (llm_to_nd(sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_linear_int8_bias_nd: invalid x");
    if (llm_parse_int8(sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tw) < 0)
        ERR(ctx, "llm_linear_int8_bias_nd: invalid int8 weight");
    if (llm_to_nd(sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &tb) < 0)
        ERR(ctx, "llm_linear_int8_bias_nd: invalid bias");
    if (tx.ndim < 1)
        ERR(ctx, "llm_linear_int8_bias_nd: x must be at least 1-D");
    if (tw.ndim != 2)
        ERR(ctx, "llm_linear_int8_bias_nd: weight must be 2-D");

    int32_t K = tx.shape[tx.ndim - 1];
    if (tw.shape[1] != K)
        ERR(ctx, "llm_linear_int8_bias_nd: in_features mismatch");

    int32_t M = tx.total / K;
    int32_t N = tw.shape[0];
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < tx.ndim; i++)
        out_shape[i] = tx.shape[i];
    out_shape[tx.ndim - 1] = N;
    int32_t out_total = M * N;

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    LinearInt8Arg arg = {tx.data, tw.data, tw.scales, dst, M, N, K};
    if (M == 1) {
        if ((int64_t)N * K >= LLM_INT8_THREAD_OPS_MIN)
            parallel_rows(linear_int8_rows_n, &arg, N);
        else
            linear_int8_rows_n(&arg, 0, N);
    } else {
        parallel_rows(linear_int8_rows_m, &arg, M);
    }

    const float *bias = tb.data;
    for (int32_t row_idx = 0; row_idx < M; row_idx++) {
        float *row = dst + (size_t)row_idx * N;
        for (int32_t col_idx = 0; col_idx < N; col_idx++)
            row[col_idx] += bias[col_idx];
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_linear_param_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, tw;
    TensorNDInt8 twq;
    llm_param_lookup_t *lookup = (llm_param_lookup_t *)sqlite3_user_data(ctx);
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    const unsigned char *weight_name = sqlite3_value_text(argv[1]);
    const void *weight_blob = NULL;
    int weight_size = 0;

    if (llm_to_nd(sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_linear_param_nd: invalid x");
    if (llm_param_lookup_step_blob(lookup,
                                   db,
                                   ctx,
                                   (const char *)weight_name,
                                   &weight_blob,
                                   &weight_size) != SQLITE_OK) {
        return;
    }

    if (llm_parse_int8(weight_blob, weight_size, &twq) == 0) {
        if (tx.ndim < 1) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_linear_param_nd: x must be at least 1-D");
        }
        if (twq.ndim != 2) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_linear_param_nd: weight must be 2-D");
        }
        int32_t K = tx.shape[tx.ndim - 1];
        if (twq.shape[1] != K) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_linear_param_nd: in_features mismatch");
        }

        int32_t M = tx.total / K;
        int32_t N = twq.shape[0];
        int32_t out_shape[LLM_MAX_NDIM];
        for (int i = 0; i < tx.ndim; i++)
            out_shape[i] = tx.shape[i];
        out_shape[tx.ndim - 1] = N;
        int32_t out_total = M * N;

        int out_sz;
        void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
        if (!out) {
            llm_param_lookup_release(lookup);
            sqlite3_result_error_nomem(ctx);
            return;
        }
        float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

        LinearInt8Arg arg = {tx.data, twq.data, twq.scales, dst, M, N, K};
        if (M == 1) {
            if ((int64_t)N * K >= LLM_INT8_THREAD_OPS_MIN)
                parallel_rows(linear_int8_rows_n, &arg, N);
            else
                linear_int8_rows_n(&arg, 0, N);
        } else {
            parallel_rows(linear_int8_rows_m, &arg, M);
        }

        llm_param_lookup_release(lookup);
        sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
        return;
    }

    if (llm_to_nd(weight_blob, weight_size, &tw) < 0) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_nd: invalid weight");
    }
    if (tx.ndim < 1) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_nd: x must be at least 1-D");
    }
    if (tw.ndim != 2) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_nd: weight must be 2-D");
    }

    int32_t k = tx.shape[tx.ndim - 1];
    if (tw.shape[1] != k) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_nd: in_features mismatch");
    }

    int32_t m = tx.total / k;
    int32_t n = tw.shape[0];
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < tx.ndim; i++)
        out_shape[i] = tx.shape[i];
    out_shape[tx.ndim - 1] = n;
    int32_t out_total = m * n;

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        llm_param_lookup_release(lookup);
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    if (m == 1) {
        cblas_sgemv(CblasRowMajor,
                    CblasNoTrans,
                    n,
                    k,
                    1.0f,
                    tw.data,
                    k,
                    tx.data,
                    1,
                    0.0f,
                    dst,
                    1);
    } else {
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    m,
                    n,
                    k,
                    1.0f,
                    tx.data,
                    k,
                    tw.data,
                    k,
                    0.0f,
                    dst,
                    n);
    }

    llm_param_lookup_release(lookup);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_linear_param_bias_nd(sqlite3_context *ctx,
                              int argc,
                              sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, tb, tw;
    TensorNDInt8 twq;
    llm_param_lookup_t *lookup = (llm_param_lookup_t *)sqlite3_user_data(ctx);
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    const unsigned char *weight_name = sqlite3_value_text(argv[1]);
    const void *weight_blob = NULL;
    int weight_size = 0;

    if (llm_to_nd(sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_linear_param_bias_nd: invalid x");
    if (llm_to_nd(sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &tb) < 0)
        ERR(ctx, "llm_linear_param_bias_nd: invalid bias");
    if (llm_param_lookup_step_blob(lookup,
                                   db,
                                   ctx,
                                   (const char *)weight_name,
                                   &weight_blob,
                                   &weight_size) != SQLITE_OK) {
        return;
    }

    if (llm_parse_int8(weight_blob, weight_size, &twq) == 0) {
        if (tx.ndim < 1) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_linear_param_bias_nd: x must be at least 1-D");
        }
        if (twq.ndim != 2) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_linear_param_bias_nd: weight must be 2-D");
        }
        int32_t K = tx.shape[tx.ndim - 1];
        if (twq.shape[1] != K) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_linear_param_bias_nd: in_features mismatch");
        }

        int32_t M = tx.total / K;
        int32_t N = twq.shape[0];
        int32_t out_shape[LLM_MAX_NDIM];
        for (int i = 0; i < tx.ndim; i++)
            out_shape[i] = tx.shape[i];
        out_shape[tx.ndim - 1] = N;
        int32_t out_total = M * N;

        int out_sz;
        void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
        if (!out) {
            llm_param_lookup_release(lookup);
            sqlite3_result_error_nomem(ctx);
            return;
        }
        float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

        LinearInt8Arg arg = {tx.data, twq.data, twq.scales, dst, M, N, K};
        if (M == 1) {
            if ((int64_t)N * K >= LLM_INT8_THREAD_OPS_MIN)
                parallel_rows(linear_int8_rows_n, &arg, N);
            else
                linear_int8_rows_n(&arg, 0, N);
        } else {
            parallel_rows(linear_int8_rows_m, &arg, M);
        }

        const float *bias = tb.data;
        for (int32_t row_idx = 0; row_idx < M; row_idx++) {
            float *row = dst + (size_t)row_idx * N;
            for (int32_t col_idx = 0; col_idx < N; col_idx++)
                row[col_idx] += bias[col_idx];
        }

        llm_param_lookup_release(lookup);
        sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
        return;
    }

    if (llm_to_nd(weight_blob, weight_size, &tw) < 0) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_bias_nd: invalid weight");
    }
    if (tx.ndim < 1) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_bias_nd: x must be at least 1-D");
    }
    if (tw.ndim != 2) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_bias_nd: weight must be 2-D");
    }

    int32_t k = tx.shape[tx.ndim - 1];
    if (tw.shape[1] != k) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_bias_nd: in_features mismatch");
    }

    int32_t m = tx.total / k;
    int32_t n = tw.shape[0];
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < tx.ndim; i++)
        out_shape[i] = tx.shape[i];
    out_shape[tx.ndim - 1] = n;
    int32_t out_total = m * n;

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        llm_param_lookup_release(lookup);
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    if (m == 1) {
        cblas_sgemv(CblasRowMajor,
                    CblasNoTrans,
                    n,
                    k,
                    1.0f,
                    tw.data,
                    k,
                    tx.data,
                    1,
                    0.0f,
                    dst,
                    1);
    } else {
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    m,
                    n,
                    k,
                    1.0f,
                    tx.data,
                    k,
                    tw.data,
                    k,
                    0.0f,
                    dst,
                    n);
    }

    const float *bias = tb.data;
    for (int32_t row_idx = 0; row_idx < m; row_idx++) {
        float *row = dst + (size_t)row_idx * n;
        for (int32_t col_idx = 0; col_idx < n; col_idx++)
            row[col_idx] += bias[col_idx];
    }

    llm_param_lookup_release(lookup);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_embedding_int8_nd(sqlite3_context *ctx,
                           int argc,
                           sqlite3_value **argv) {
    (void)argc;
    TensorNDInt8 tw;
    TensorNDF32 ti;
    if (llm_parse_int8(sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tw) < 0)
        ERR(ctx, "llm_embedding_int8_nd: invalid int8 weight");
    if (llm_to_nd(sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &ti) < 0)
        ERR(ctx, "llm_embedding_int8_nd: invalid indices");
    if (tw.ndim != 2)
        ERR(ctx, "llm_embedding_int8_nd: weight must be 2-D");

    int32_t vocab = tw.shape[0];
    int32_t embed = tw.shape[1];
    int out_ndim = ti.ndim + 1;
    if (out_ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_embedding_int8_nd: too many dims");

    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < ti.ndim; i++)
        out_shape[i] = ti.shape[i];
    out_shape[ti.ndim] = embed;
    int32_t out_total = ti.total * embed;

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    for (int32_t i = 0; i < ti.total; i++) {
        int idx = (int)roundf(ti.data[i]);
        if (idx < 0 || idx >= vocab) {
            sqlite3_free(out);
            ERR(ctx, "llm_embedding_int8_nd: index out of range");
        }
        float scale = tw.scales[idx];
        const int8_t *qrow = tw.data + (size_t)idx * embed;
        float *drow = dst + (size_t)i * embed;
        for (int32_t j = 0; j < embed; j++)
            drow[j] = (float)qrow[j] * scale;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_embedding_param_nd(sqlite3_context *ctx,
                            int argc,
                            sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tw, ti;
    TensorNDInt8 twq;
    llm_param_lookup_t *lookup = (llm_param_lookup_t *)sqlite3_user_data(ctx);
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    const unsigned char *weight_name = sqlite3_value_text(argv[0]);
    const void *weight_blob = NULL;
    int weight_size = 0;

    if (llm_to_nd(sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &ti) < 0)
        ERR(ctx, "llm_embedding_param_nd: invalid indices");
    if (llm_param_lookup_step_blob(lookup,
                                   db,
                                   ctx,
                                   (const char *)weight_name,
                                   &weight_blob,
                                   &weight_size) != SQLITE_OK) {
        return;
    }

    if (llm_parse_int8(weight_blob, weight_size, &twq) == 0) {
        if (twq.ndim != 2) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_embedding_param_nd: weight must be 2-D");
        }
        int32_t vocab = twq.shape[0];
        int32_t embed = twq.shape[1];
        int out_ndim = ti.ndim + 1;
        if (out_ndim > LLM_MAX_NDIM) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_embedding_param_nd: too many dims");
        }
        int32_t out_shape[LLM_MAX_NDIM];
        for (int i = 0; i < ti.ndim; i++)
            out_shape[i] = ti.shape[i];
        out_shape[ti.ndim] = embed;
        int32_t out_total = ti.total * embed;

        int out_sz;
        void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
        if (!out) {
            llm_param_lookup_release(lookup);
            sqlite3_result_error_nomem(ctx);
            return;
        }
        float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

        for (int32_t i = 0; i < ti.total; i++) {
            int idx = (int)roundf(ti.data[i]);
            if (idx < 0 || idx >= vocab) {
                llm_param_lookup_release(lookup);
                sqlite3_free(out);
                ERR(ctx, "llm_embedding_param_nd: index out of range");
            }
            float scale = twq.scales[idx];
            const int8_t *qrow = twq.data + (size_t)idx * embed;
            float *drow = dst + (size_t)i * embed;
            for (int32_t j = 0; j < embed; j++)
                drow[j] = (float)qrow[j] * scale;
        }

        llm_param_lookup_release(lookup);
        sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
        return;
    }

    if (llm_to_nd(weight_blob, weight_size, &tw) < 0) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_embedding_param_nd: invalid weight blob");
    }
    if (tw.ndim != 2) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_embedding_param_nd: weight must be 2-D");
    }

    int32_t vocab = tw.shape[0];
    int32_t embed = tw.shape[1];
    int out_ndim = ti.ndim + 1;
    if (out_ndim > LLM_MAX_NDIM) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_embedding_param_nd: too many dims");
    }
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < ti.ndim; i++)
        out_shape[i] = ti.shape[i];
    out_shape[ti.ndim] = embed;
    int32_t out_total = ti.total * embed;

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
    if (!out) {
        llm_param_lookup_release(lookup);
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    for (int32_t i = 0; i < ti.total; i++) {
        int idx = (int)roundf(ti.data[i]);
        if (idx < 0 || idx >= vocab) {
            llm_param_lookup_release(lookup);
            sqlite3_free(out);
            ERR(ctx, "llm_embedding_param_nd: index out of range");
        }
        memcpy(dst + (size_t)i * embed,
               tw.data + (size_t)idx * embed,
               (size_t)embed * sizeof(float));
    }

    llm_param_lookup_release(lookup);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_is_int8_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    if (blob && sz >= 4) {
        int32_t magic = *(const int32_t *)blob;
        sqlite3_result_int(ctx, magic == LLM_INT8_MAGIC ? 1 : 0);
    } else {
        sqlite3_result_int(ctx, 0);
    }
}