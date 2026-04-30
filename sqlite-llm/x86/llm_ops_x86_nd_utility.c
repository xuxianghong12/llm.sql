/*
 * llm_ops_x86_nd_utility.c - N-D masking, factory, and utility wrappers
 *
 * This file owns N-D SQL wrappers that do not fit neatly into layout,
 * broadcast algebra, pointwise math, pure reductions, or model-specific math:
 * triangular masking, logical/bitwise helpers, remainder-style arithmetic,
 * tensor factories, inspection helpers, and argmax-style utility selection.
 * Add code here when the operator is tensor-oriented but is best described as
 * a utility or construction helper rather than a core math primitive.
 *
 * Maintenance guidance:
 *   - Keep broadcast-aware utility logic here so layout and algebra files stay
 *     focused on their primary responsibilities.
 *   - Reuse broadcast_shape() and nd_strides() for mask/where-style operators
 *     instead of duplicating tensor iteration logic elsewhere.
 *   - If an operator primarily aggregates or collapses dimensions, move it to
 *     llm_ops_x86_nd_reduce.c instead of growing this utility bucket.
 */

#include "llm_ops_x86_internal.h"
SQLITE_EXTENSION_INIT3

void sql_triu_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_triu_nd: invalid blob");
    if (t.ndim != 2)
        ERR(ctx, "llm_triu_nd: only 2-D tensors supported");
    int k = sqlite3_value_int(argv[1]);
    int32_t rows = t.shape[0], cols = t.shape[1];

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t r = 0; r < rows; r++)
        for (int32_t c = 0; c < cols; c++)
            dst[r * cols + c] = (c >= r + k) ? t.data[r * cols + c] : 0.0f;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_where_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tc, tx, ty;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tc) < 0)
        ERR(ctx, "llm_where_nd: invalid cond blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tx) < 0)
        ERR(ctx, "llm_where_nd: invalid x blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &ty) < 0)
        ERR(ctx, "llm_where_nd: invalid y blob");

    int32_t xy_shape[LLM_MAX_NDIM];
    int xy_ndim;
    int32_t xy_total = broadcast_shape(&tx, &ty, xy_shape, &xy_ndim);
    if (xy_total < 0)
        ERR(ctx, "llm_where_nd: x/y shapes incompatible");

    TensorNDF32 txy;
    txy.ndim = xy_ndim;
    for (int i = 0; i < xy_ndim; i++)
        txy.shape[i] = xy_shape[i];
    txy.total = xy_total;

    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim;
    int32_t total = broadcast_shape(&tc, &txy, out_shape, &out_ndim);
    if (total < 0)
        ERR(ctx, "llm_where_nd: cond shape incompatible");

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    int off_c = out_ndim - tc.ndim;
    int off_x = out_ndim - tx.ndim;
    int off_y = out_ndim - ty.ndim;

    int32_t pc[LLM_MAX_NDIM], px[LLM_MAX_NDIM], py[LLM_MAX_NDIM];
    int32_t sc[LLM_MAX_NDIM], sx[LLM_MAX_NDIM], sy[LLM_MAX_NDIM];
    for (int i = 0; i < off_c; i++)
        pc[i] = 1;
    for (int i = 0; i < tc.ndim; i++)
        pc[i + off_c] = tc.shape[i];
    for (int i = 0; i < off_x; i++)
        px[i] = 1;
    for (int i = 0; i < tx.ndim; i++)
        px[i + off_x] = tx.shape[i];
    for (int i = 0; i < off_y; i++)
        py[i] = 1;
    for (int i = 0; i < ty.ndim; i++)
        py[i + off_y] = ty.shape[i];

    nd_strides(pc, out_ndim, sc);
    nd_strides(px, out_ndim, sx);
    nd_strides(py, out_ndim, sy);
    for (int i = 0; i < out_ndim; i++) {
        if (pc[i] == 1)
            sc[i] = 0;
        if (px[i] == 1)
            sx[i] = 0;
        if (py[i] == 1)
            sy[i] = 0;
    }

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < total; flat++) {
        int32_t oc = 0, ox = 0, oy = 0;
        for (int d = 0; d < out_ndim; d++) {
            oc += idx[d] * sc[d];
            ox += idx[d] * sx[d];
            oy += idx[d] * sy[d];
        }
        dst[flat] = (tc.data[oc] > 0.5f) ? tx.data[ox] : ty.data[oy];
        for (int d = out_ndim - 1; d >= 0; d--) {
            if (++idx[d] < out_shape[d])
                break;
            idx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_masked_fill_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, tm;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_masked_fill_nd: invalid x blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tm) < 0)
        ERR(ctx, "llm_masked_fill_nd: invalid mask blob");
    float val = (float)sqlite3_value_double(argv[2]);

    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim;
    int32_t total = broadcast_shape(&tx, &tm, out_shape, &out_ndim);
    if (total < 0)
        ERR(ctx, "llm_masked_fill_nd: incompatible shapes");

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    int off_x = out_ndim - tx.ndim;
    int off_m = out_ndim - tm.ndim;
    int32_t px[LLM_MAX_NDIM], pm[LLM_MAX_NDIM];
    int32_t sx[LLM_MAX_NDIM], sm[LLM_MAX_NDIM];
    for (int i = 0; i < off_x; i++)
        px[i] = 1;
    for (int i = 0; i < tx.ndim; i++)
        px[i + off_x] = tx.shape[i];
    for (int i = 0; i < off_m; i++)
        pm[i] = 1;
    for (int i = 0; i < tm.ndim; i++)
        pm[i + off_m] = tm.shape[i];

    nd_strides(px, out_ndim, sx);
    nd_strides(pm, out_ndim, sm);
    for (int i = 0; i < out_ndim; i++) {
        if (px[i] == 1)
            sx[i] = 0;
        if (pm[i] == 1)
            sm[i] = 0;
    }

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < total; flat++) {
        int32_t ox = 0, om = 0;
        for (int d = 0; d < out_ndim; d++) {
            ox += idx[d] * sx[d];
            om += idx[d] * sm[d];
        }
        dst[flat] = (tm.data[om] > 0.5f) ? val : tx.data[ox];
        for (int d = out_ndim - 1; d >= 0; d--) {
            if (++idx[d] < out_shape[d])
                break;
            idx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_arange_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    int32_t n = sqlite3_value_int(argv[0]);
    if (n <= 0)
        ERR(ctx, "llm_arange_nd: n must be > 0");
    int32_t shape[1] = {n};
    int out_sz;
    void *out = nd_result_alloc(1, shape, n, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(1));
    for (int32_t i = 0; i < n; i++)
        dst[i] = (float)i;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_full_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2)
        ERR(ctx, "llm_full_nd: need value + at least 1 dim");
    float val = (float)sqlite3_value_double(argv[0]);
    int ndim = argc - 1;
    if (ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_full_nd: too many dims");
    int32_t shape[LLM_MAX_NDIM];
    int32_t total = 1;
    for (int i = 0; i < ndim; i++) {
        shape[i] = sqlite3_value_int(argv[i + 1]);
        if (shape[i] <= 0)
            ERR(ctx, "llm_full_nd: dims must be > 0");
        total *= shape[i];
    }
    int out_sz;
    void *out = nd_result_alloc(ndim, shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ndim));
    for (int32_t i = 0; i < total; i++)
        dst[i] = val;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_argmax_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_argmax_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_argmax_nd: dim out of range");

    int32_t new_shape[LLM_MAX_NDIM];
    int new_ndim = t.ndim - 1;
    if (new_ndim < 1) {
        new_ndim = 1;
        new_shape[0] = 1;
    } else {
        int j = 0;
        for (int i = 0; i < t.ndim; i++)
            if (i != dim)
                new_shape[j++] = t.shape[i];
    }
    int32_t new_total = 1;
    for (int i = 0; i < new_ndim; i++)
        new_total *= new_shape[i];

    int out_sz;
    void *out = nd_result_alloc(new_ndim, new_shape, new_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(new_ndim));

    int32_t strides[LLM_MAX_NDIM];
    nd_strides(t.shape, t.ndim, strides);
    int32_t dim_sz = t.shape[dim];
    int32_t dim_str = strides[dim];

    int32_t other_shape[LLM_MAX_NDIM];
    int32_t other_strides[LLM_MAX_NDIM];
    int on = 0;
    for (int d = 0; d < t.ndim; d++) {
        if (d != dim) {
            other_shape[on] = t.shape[d];
            other_strides[on] = strides[d];
            on++;
        }
    }

    int32_t outer = t.total / dim_sz;
    int32_t oidx[LLM_MAX_NDIM];
    memset(oidx, 0, sizeof(oidx));
    for (int32_t slice = 0; slice < outer; slice++) {
        int32_t base = 0;
        for (int i = 0; i < on; i++)
            base += oidx[i] * other_strides[i];
        float best = t.data[base];
        int32_t best_idx = 0;
        for (int32_t k = 1; k < dim_sz; k++) {
            float v = t.data[base + k * dim_str];
            if (v > best) {
                best = v;
                best_idx = k;
            }
        }
        dst[slice] = (float)best_idx;
        for (int i = on - 1; i >= 0; i--) {
            if (++oidx[i] < other_shape[i])
                break;
            oidx[i] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_tril_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_tril_nd: invalid blob");
    if (t.ndim < 2)
        ERR(ctx, "llm_tril_nd: need at least 2-D tensor");
    int diag = sqlite3_value_int(argv[1]);
    int rows = t.shape[t.ndim - 2];
    int cols = t.shape[t.ndim - 1];
    int mat_size = rows * cols;
    int batch = t.total / mat_size;

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int b = 0; b < batch; b++) {
        const float *src_mat = t.data + b * mat_size;
        float *dst_mat = dst + b * mat_size;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++)
                dst_mat[r * cols + c] =
                    (c <= r + diag) ? src_mat[r * cols + c] : 0.0f;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_logical_not_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_logical_not_nd: invalid blob");

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = (t.data[i] == 0.0f) ? 1.0f : 0.0f;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

static void sql_binary_mask_nd(sqlite3_context *ctx,
                               sqlite3_value **argv,
                               const char *name,
                               int use_or) {
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, name);
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, name);
    if (ta.total != tb.total)
        ERR(ctx, "tensor logic op: size mismatch");

    int out_sz;
    void *out = nd_result_alloc(ta.ndim, ta.shape, ta.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ta.ndim));
    for (int32_t i = 0; i < ta.total; i++) {
        int lhs = ta.data[i] != 0.0f;
        int rhs = tb.data[i] != 0.0f;
        dst[i] = (use_or ? (lhs || rhs) : (lhs && rhs)) ? 1.0f : 0.0f;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_logical_and_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_binary_mask_nd(ctx, argv, "llm_logical_and_nd: invalid blob", 0);
}

void sql_logical_or_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_binary_mask_nd(ctx, argv, "llm_logical_or_nd: invalid blob", 1);
}

void sql_bitwise_not_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_logical_not_nd(ctx, argc, argv);
}

void sql_bitwise_and_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_binary_mask_nd(ctx, argv, "llm_bitwise_and_nd: invalid blob", 0);
}

void sql_bitwise_or_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_binary_mask_nd(ctx, argv, "llm_bitwise_or_nd: invalid blob", 1);
}

void sql_bitwise_and_scalar_nd(sqlite3_context *ctx,
                               int argc,
                               sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_bitwise_and_scalar_nd: invalid blob");
    int64_t scalar = sqlite3_value_int64(argv[1]);

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = (float)((int64_t)t.data[i] & scalar);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_pow_scalar_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    float base = (float)sqlite3_value_double(argv[0]);
    const void *blob = sqlite3_value_blob(argv[1]);
    int sz = sqlite3_value_bytes(argv[1]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_pow_scalar_nd: invalid blob");

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = powf(base, t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_floor_div_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_floor_div_nd: invalid blob a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_floor_div_nd: invalid blob b");
    if (ta.total != tb.total)
        ERR(ctx, "llm_floor_div_nd: size mismatch");

    int out_sz;
    void *out = nd_result_alloc(ta.ndim, ta.shape, ta.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ta.ndim));
    for (int32_t i = 0; i < ta.total; i++)
        dst[i] = floorf(ta.data[i] / tb.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_remainder_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_remainder_nd: invalid blob");
    float scalar = (float)sqlite3_value_double(argv[1]);

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++) {
        float remainder = fmodf(t.data[i], scalar);
        if (remainder != 0.0f && ((t.data[i] < 0) != (scalar < 0)))
            remainder += scalar;
        dst[i] = remainder;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_remainder_tensor_nd(sqlite3_context *ctx,
                             int argc,
                             sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_remainder_tensor_nd: invalid blob a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_remainder_tensor_nd: invalid blob b");
    if (ta.total != tb.total)
        ERR(ctx, "llm_remainder_tensor_nd: size mismatch");

    int out_sz;
    void *out = nd_result_alloc(ta.ndim, ta.shape, ta.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ta.ndim));
    for (int32_t i = 0; i < ta.total; i++) {
        float remainder = fmodf(ta.data[i], tb.data[i]);
        if (remainder != 0.0f && ((ta.data[i] < 0) != (tb.data[i] < 0)))
            remainder += tb.data[i];
        dst[i] = remainder;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_numel_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_numel_nd: invalid blob");
    sqlite3_result_int(ctx, t.total);
}