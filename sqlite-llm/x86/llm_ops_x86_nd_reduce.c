/*
 * llm_ops_x86_nd_reduce.c - N-D reduction and value-selection wrappers
 *
 * This file owns N-D SQL wrappers that collapse tensor data into scalars or
 * lower-rank outputs: global reductions, flat-item access, and dim-wise max or
 * argmax selection. Add code here when the operator's primary job is choosing
 * or aggregating values rather than preserving full tensor shape.
 *
 * Maintenance guidance:
 *   - Keep scalar/global reduction semantics together so the SQL surface stays
 *     consistent across sum, mean, min, max, and argmax variants.
 *   - Reuse nd_result_alloc() for dim-wise outputs and preserve keepdim logic
 *     at the wrapper boundary where shape decisions are easiest to audit.
 *   - If a new operator keeps full shape and only rewrites elements, it likely
 *     belongs in llm_ops_x86_nd_pointwise.c instead of here.
 */

#include "llm_ops_x86_internal.h"
SQLITE_EXTENSION_INIT3

void sql_reduce_sum_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_reduce_sum_nd: invalid blob");
    double acc = 0.0;
    for (int32_t i = 0; i < t.total; i++)
        acc += (double)t.data[i];
    int32_t shape1 = 1;
    int out_sz;
    void *out = nd_result_alloc(1, &shape1, 1, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(1));
    dst[0] = (float)acc;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_reduce_mean_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_reduce_mean_nd: invalid blob");
    double acc = 0.0;
    for (int32_t i = 0; i < t.total; i++)
        acc += (double)t.data[i];
    int32_t shape1 = 1;
    int out_sz;
    void *out = nd_result_alloc(1, &shape1, 1, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(1));
    dst[0] = (float)(acc / (double)t.total);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_reduce_max_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_reduce_max_nd: invalid blob");
    float mx = -FLT_MAX;
    for (int32_t i = 0; i < t.total; i++)
        if (t.data[i] > mx)
            mx = t.data[i];
    int32_t shape1 = 1;
    int out_sz;
    void *out = nd_result_alloc(1, &shape1, 1, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(1));
    dst[0] = mx;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_reduce_min_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_reduce_min_nd: invalid blob");
    float mn = FLT_MAX;
    for (int32_t i = 0; i < t.total; i++)
        if (t.data[i] < mn)
            mn = t.data[i];
    int32_t shape1 = 1;
    int out_sz;
    void *out = nd_result_alloc(1, &shape1, 1, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(1));
    dst[0] = mn;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_item_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_item_nd: invalid blob");
    int32_t idx = sqlite3_value_int(argv[1]);
    if (idx < 0)
        idx += t.total;
    if (idx < 0 || idx >= t.total)
        ERR(ctx, "llm_item_nd: index out of range");
    sqlite3_result_double(ctx, (double)t.data[idx]);
}

void sql_reduce_argmax_nd(sqlite3_context *ctx,
                          int argc,
                          sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_reduce_argmax_nd: invalid blob");
    float mx = -FLT_MAX;
    int32_t idx = 0;
    for (int32_t i = 0; i < t.total; i++) {
        if (t.data[i] > mx) {
            mx = t.data[i];
            idx = i;
        }
    }
    sqlite3_result_int(ctx, idx);
}

void sql_max_dim_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_max_dim_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    int keepdim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_max_dim_nd: dim out of range");

    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim = 0;
    for (int i = 0; i < t.ndim; i++) {
        if (i == dim) {
            if (keepdim)
                out_shape[out_ndim++] = 1;
        } else {
            out_shape[out_ndim++] = t.shape[i];
        }
    }
    if (out_ndim == 0) {
        out_shape[0] = 1;
        out_ndim = 1;
    }
    int32_t out_total = outer * inner;

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    for (int32_t o = 0; o < outer; o++) {
        for (int32_t n = 0; n < inner; n++) {
            float mx = -FLT_MAX;
            for (int32_t d = 0; d < dimsize; d++) {
                float v = t.data[(o * dimsize + d) * inner + n];
                if (v > mx)
                    mx = v;
            }
            dst[o * inner + n] = mx;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_argmax_dim_nd(sqlite3_context *ctx,
                       int argc,
                       sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_argmax_dim_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    int keepdim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_argmax_dim_nd: dim out of range");

    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim = 0;
    for (int i = 0; i < t.ndim; i++) {
        if (i == dim) {
            if (keepdim)
                out_shape[out_ndim++] = 1;
        } else {
            out_shape[out_ndim++] = t.shape[i];
        }
    }
    if (out_ndim == 0) {
        out_shape[0] = 1;
        out_ndim = 1;
    }
    int32_t out_total = outer * inner;

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    for (int32_t o = 0; o < outer; o++) {
        for (int32_t n = 0; n < inner; n++) {
            float mx = -FLT_MAX;
            int32_t best = 0;
            for (int32_t d = 0; d < dimsize; d++) {
                float v = t.data[(o * dimsize + d) * inner + n];
                if (v > mx) {
                    mx = v;
                    best = d;
                }
            }
            dst[o * inner + n] = (float)best;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}