/*
 * llm_ops_x86_nd_sequence.c - N-D sequence, index, and structural reorder
 * wrappers
 *
 * This file owns N-D SQL wrappers that traverse tensor sequences along a
 * dimension: cumulative scans such as cumsum/cumprod and index-driven reads
 * such as gather/index_select. It also owns indexed updates and
 * order-preserving structural transforms like repeat, flatten, concatenate, and
 * stack. Keep these operators here when they are defined by ordered traversal,
 * explicit index tensors, or deterministic data rearrangement rather than by
 * broadcast, pointwise math, or full reductions.
 *
 * Maintenance guidance:
 *   - Reuse nd_strides() and nd_result_alloc() for dimension-aware iteration.
 *   - Keep simple traversal kernels file-local; only promote helpers when a
 *     second translation unit needs them.
 *   - Extend this file with related scan/scatter/repeat style operators before
 *     adding more special-purpose sections back into llm_ops_x86.c.
 */

#include "llm_ops_x86_internal.h"
SQLITE_EXTENSION_INIT3

void sql_cumsum_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_cumsum_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_cumsum_nd: dim out of range");

    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    for (int32_t outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (int32_t inner_idx = 0; inner_idx < inner; inner_idx++) {
            double acc = 0.0;
            for (int32_t dim_idx = 0; dim_idx < dimsize; dim_idx++) {
                int32_t idx =
                    (outer_idx * dimsize + dim_idx) * inner + inner_idx;
                acc += (double)t.data[idx];
                dst[idx] = (float)acc;
            }
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_cumprod_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_cumprod_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_cumprod_nd: dim out of range");

    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    for (int32_t outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (int32_t inner_idx = 0; inner_idx < inner; inner_idx++) {
            double acc = 1.0;
            for (int32_t dim_idx = 0; dim_idx < dimsize; dim_idx++) {
                int32_t idx =
                    (outer_idx * dimsize + dim_idx) * inner + inner_idx;
                acc *= (double)t.data[idx];
                dst[idx] = (float)acc;
            }
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_gather_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, ti;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_gather_nd: invalid x blob");
    int dim = sqlite3_value_int(argv[1]);
    if (llm_to_nd(
            sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &ti) < 0)
        ERR(ctx, "llm_gather_nd: invalid index blob");
    if (dim < 0)
        dim += tx.ndim;
    if (dim < 0 || dim >= tx.ndim)
        ERR(ctx, "llm_gather_nd: dim out of range");
    if (tx.ndim != ti.ndim)
        ERR(ctx, "llm_gather_nd: ndim mismatch");

    int out_sz;
    void *out = nd_result_alloc(ti.ndim, ti.shape, ti.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ti.ndim));

    int32_t x_strides[LLM_MAX_NDIM], i_strides[LLM_MAX_NDIM];
    nd_strides(tx.shape, tx.ndim, x_strides);
    nd_strides(ti.shape, ti.ndim, i_strides);

    for (int32_t flat = 0; flat < ti.total; flat++) {
        int32_t rem = flat;
        int32_t x_off = 0;
        for (int d = 0; d < ti.ndim; d++) {
            int32_t coord = rem / i_strides[d];
            rem %= i_strides[d];
            if (d == dim) {
                int32_t idx_val = (int32_t)ti.data[flat];
                x_off += idx_val * x_strides[d];
            } else {
                x_off += coord * x_strides[d];
            }
        }
        dst[flat] = tx.data[x_off];
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_index_select_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, ti;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_index_select_nd: invalid x blob");
    int dim = sqlite3_value_int(argv[1]);
    if (llm_to_nd(
            sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &ti) < 0)
        ERR(ctx, "llm_index_select_nd: invalid index blob");
    if (dim < 0)
        dim += tx.ndim;
    if (dim < 0 || dim >= tx.ndim)
        ERR(ctx, "llm_index_select_nd: dim out of range");

    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < tx.ndim; i++)
        out_shape[i] = tx.shape[i];
    out_shape[dim] = ti.total;

    int32_t out_total = 1;
    for (int i = 0; i < tx.ndim; i++)
        out_total *= out_shape[i];

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    int32_t outer = 1, inner = 1;
    for (int i = 0; i < dim; i++)
        outer *= tx.shape[i];
    for (int i = dim + 1; i < tx.ndim; i++)
        inner *= tx.shape[i];
    int32_t dim_sz = tx.shape[dim];

    for (int32_t outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (int32_t index_idx = 0; index_idx < ti.total; index_idx++) {
            int32_t idx = (int32_t)ti.data[index_idx];
            if (idx < 0)
                idx += dim_sz;
            const float *src = tx.data + (outer_idx * dim_sz + idx) * inner;
            float *dst_ptr = dst + (outer_idx * ti.total + index_idx) * inner;
            memcpy(dst_ptr, src, (size_t)inner * sizeof(float));
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_scatter_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, ti;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_scatter_nd: invalid x blob");
    int dim = sqlite3_value_int(argv[1]);
    if (llm_to_nd(
            sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &ti) < 0)
        ERR(ctx, "llm_scatter_nd: invalid index blob");
    float val = (float)sqlite3_value_double(argv[3]);
    if (dim < 0)
        dim += tx.ndim;
    if (dim < 0 || dim >= tx.ndim)
        ERR(ctx, "llm_scatter_nd: dim out of range");

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, tx.shape, tx.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));
    memcpy(dst, tx.data, (size_t)tx.total * sizeof(float));

    int32_t x_strides[LLM_MAX_NDIM], i_strides[LLM_MAX_NDIM];
    nd_strides(tx.shape, tx.ndim, x_strides);
    nd_strides(ti.shape, ti.ndim, i_strides);

    for (int32_t flat = 0; flat < ti.total; flat++) {
        int32_t rem = flat;
        int32_t x_off = 0;
        for (int d = 0; d < ti.ndim; d++) {
            int32_t coord = rem / i_strides[d];
            rem %= i_strides[d];
            if (d == dim) {
                int32_t idx_val = (int32_t)ti.data[flat];
                x_off += idx_val * x_strides[d];
            } else {
                x_off += coord * x_strides[d];
            }
        }
        dst[x_off] = val;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_repeat_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2)
        ERR(ctx, "llm_repeat_nd: need blob + at least 1 repeat");
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_repeat_nd: invalid blob");
    int nrep = argc - 1;
    int32_t reps[LLM_MAX_NDIM];
    for (int i = 0; i < nrep; i++)
        reps[i] = sqlite3_value_int(argv[i + 1]);

    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim = nrep > t.ndim ? nrep : t.ndim;
    if (out_ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_repeat_nd: too many dims");
    int32_t out_total = 1;
    for (int i = 0; i < out_ndim; i++) {
        int si = i - (out_ndim - t.ndim);
        int ri = i - (out_ndim - nrep);
        int32_t shape_dim = (si >= 0 && si < t.ndim) ? t.shape[si] : 1;
        int32_t rep_dim = (ri >= 0 && ri < nrep) ? reps[ri] : 1;
        out_shape[i] = shape_dim * rep_dim;
        out_total *= out_shape[i];
    }

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    int32_t out_strides[LLM_MAX_NDIM], src_strides[LLM_MAX_NDIM];
    nd_strides(out_shape, out_ndim, out_strides);
    nd_strides(t.shape, t.ndim, src_strides);

    for (int32_t flat = 0; flat < out_total; flat++) {
        int32_t rem = flat;
        int32_t src_off = 0;
        for (int d = 0; d < out_ndim; d++) {
            int32_t coord = rem / out_strides[d];
            rem %= out_strides[d];
            int si = d - (out_ndim - t.ndim);
            if (si >= 0 && si < t.ndim)
                src_off += (coord % t.shape[si]) * src_strides[si];
        }
        dst[flat] = t.data[src_off];
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_repeat_kv_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_repeat_kv_nd: invalid blob");
    int num_repeats = sqlite3_value_int(argv[1]);
    if (num_repeats <= 0)
        ERR(ctx, "llm_repeat_kv_nd: num_repeats must be > 0");
    if (num_repeats == 1) {
        sqlite3_result_blob(ctx, blob, sz, SQLITE_TRANSIENT);
        return;
    }
    if (t.ndim != 4)
        ERR(ctx, "llm_repeat_kv_nd: expected 4-D tensor");

    int32_t batch = t.shape[0];
    int32_t kv_heads = t.shape[1];
    int32_t seq_len = t.shape[2];
    int32_t head_dim = t.shape[3];

    int32_t out_shape[4] = {batch, kv_heads * num_repeats, seq_len, head_dim};
    int32_t out_total = batch * kv_heads * num_repeats * seq_len * head_dim;

    int out_sz;
    void *out = nd_result_alloc(4, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(4));

    int32_t head_elems = seq_len * head_dim;
    int32_t head_bytes = head_elems * (int32_t)sizeof(float);

    for (int32_t batch_idx = 0; batch_idx < batch; batch_idx++) {
        for (int32_t head_idx = 0; head_idx < kv_heads; head_idx++) {
            const float *src_head =
                t.data + (batch_idx * kv_heads + head_idx) * head_elems;
            for (int repeat_idx = 0; repeat_idx < num_repeats; repeat_idx++) {
                memcpy(dst, src_head, head_bytes);
                dst += head_elems;
            }
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_flip_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_flip_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_flip_nd: dim out of range");

    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    for (int32_t outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (int32_t dim_idx = 0; dim_idx < dimsize; dim_idx++) {
            const float *src =
                t.data +
                (outer_idx * dimsize + (dimsize - 1 - dim_idx)) * inner;
            float *dst_ptr = dst + (outer_idx * dimsize + dim_idx) * inner;
            memcpy(dst_ptr, src, (size_t)inner * sizeof(float));
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_roll_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_roll_nd: invalid blob");
    int shift = sqlite3_value_int(argv[1]);
    int dim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_roll_nd: dim out of range");

    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    shift = ((shift % dimsize) + dimsize) % dimsize;

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    for (int32_t outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (int32_t dim_idx = 0; dim_idx < dimsize; dim_idx++) {
            int32_t src_dim = (dim_idx - shift + dimsize) % dimsize;
            const float *src = t.data + (outer_idx * dimsize + src_dim) * inner;
            float *dst_ptr = dst + (outer_idx * dimsize + dim_idx) * inner;
            memcpy(dst_ptr, src, (size_t)inner * sizeof(float));
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_repeat_interleave_nd(sqlite3_context *ctx,
                              int argc,
                              sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_repeat_interleave_nd: invalid blob");
    int32_t reps = sqlite3_value_int(argv[1]);
    int dim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_repeat_interleave_nd: dim out of range");

    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < t.ndim; i++)
        out_shape[i] = t.shape[i];
    out_shape[dim] = dimsize * reps;

    int32_t out_total = 1;
    for (int i = 0; i < t.ndim; i++)
        out_total *= out_shape[i];

    int out_sz;
    void *out = nd_result_alloc(t.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    for (int32_t outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (int32_t dim_idx = 0; dim_idx < dimsize; dim_idx++) {
            const float *src = t.data + (outer_idx * dimsize + dim_idx) * inner;
            for (int32_t repeat_idx = 0; repeat_idx < reps; repeat_idx++) {
                float *dst_ptr = dst + (outer_idx * dimsize * reps +
                                        dim_idx * reps + repeat_idx) *
                                           inner;
                memcpy(dst_ptr, src, (size_t)inner * sizeof(float));
            }
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_index_put_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 4)
        ERR(ctx, "llm_index_put_nd: need x, values, accumulate, idx...");
    TensorNDF32 tx, tv;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_index_put_nd: invalid x blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tv) < 0)
        ERR(ctx, "llm_index_put_nd: invalid values blob");
    int accumulate = sqlite3_value_int(argv[2]);

    int n_idx_dims = argc - 3;
    if (n_idx_dims > tx.ndim)
        ERR(ctx, "llm_index_put_nd: more indices than x dims");

    TensorNDF32 idx_t[LLM_MAX_NDIM];
    int has_idx[LLM_MAX_NDIM];
    for (int d = 0; d < n_idx_dims; d++) {
        if (sqlite3_value_type(argv[3 + d]) == SQLITE_NULL) {
            has_idx[d] = 0;
        } else {
            if (llm_to_nd(sqlite3_value_blob(argv[3 + d]),
                          sqlite3_value_bytes(argv[3 + d]),
                          &idx_t[d]) < 0)
                ERR(ctx, "llm_index_put_nd: invalid index blob");
            has_idx[d] = 1;
        }
    }

    int32_t n_positions = 0;
    for (int d = 0; d < n_idx_dims; d++) {
        if (has_idx[d]) {
            n_positions = idx_t[d].total;
            break;
        }
    }
    if (n_positions == 0) {
        sqlite3_result_blob(ctx,
                            sqlite3_value_blob(argv[0]),
                            sqlite3_value_bytes(argv[0]),
                            SQLITE_TRANSIENT);
        return;
    }

    int32_t x_strides[LLM_MAX_NDIM];
    nd_strides(tx.shape, tx.ndim, x_strides);

    int32_t tail_size = 1;
    for (int d = n_idx_dims; d < tx.ndim; d++)
        tail_size *= tx.shape[d];

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, tx.shape, tx.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));
    memcpy(dst, tx.data, (size_t)tx.total * sizeof(float));

    for (int32_t pos = 0; pos < n_positions; pos++) {
        int32_t flat_offset = 0;
        for (int d = 0; d < n_idx_dims; d++) {
            int32_t dim_idx;
            if (has_idx[d]) {
                dim_idx = (int32_t)idx_t[d].data[pos];
                if (dim_idx < 0)
                    dim_idx += tx.shape[d];
                if (dim_idx < 0)
                    dim_idx = 0;
                if (dim_idx >= tx.shape[d])
                    dim_idx = tx.shape[d] - 1;
            } else {
                dim_idx = 0;
            }
            flat_offset += dim_idx * x_strides[d];
        }

        int32_t val_offset = pos * tail_size;
        if (val_offset + tail_size > tv.total)
            break;
        if (flat_offset + tail_size > tx.total)
            break;
        if (accumulate) {
            for (int32_t k = 0; k < tail_size; k++)
                dst[flat_offset + k] += tv.data[val_offset + k];
        } else {
            memcpy(dst + flat_offset,
                   tv.data + val_offset,
                   (size_t)tail_size * sizeof(float));
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_repeat_interleave_tensor_nd(sqlite3_context *ctx,
                                     int argc,
                                     sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 t, tr;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &t) < 0)
        ERR(ctx, "llm_repeat_interleave_tensor_nd: invalid x blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tr) < 0)
        ERR(ctx, "llm_repeat_interleave_tensor_nd: invalid repeats blob");
    int dim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_repeat_interleave_tensor_nd: dim out of range");

    int32_t dimsize = t.shape[dim];
    if (tr.total != dimsize)
        ERR(ctx,
            "llm_repeat_interleave_tensor_nd: repeats length != shape[dim]");

    int32_t out_dimsize = 0;
    for (int32_t i = 0; i < dimsize; i++)
        out_dimsize += (int32_t)tr.data[i];

    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < t.ndim; i++)
        out_shape[i] = t.shape[i];
    out_shape[dim] = out_dimsize;

    int32_t out_total = 1;
    for (int i = 0; i < t.ndim; i++)
        out_total *= out_shape[i];

    int out_sz;
    void *out = nd_result_alloc(t.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    int32_t outer = 1, inner = 1;
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    for (int32_t outer_idx = 0; outer_idx < outer; outer_idx++) {
        int32_t dst_dim = 0;
        for (int32_t dim_idx = 0; dim_idx < dimsize; dim_idx++) {
            int32_t repeat_count = (int32_t)tr.data[dim_idx];
            const float *src = t.data + (outer_idx * dimsize + dim_idx) * inner;
            for (int32_t rep = 0; rep < repeat_count; rep++) {
                float *dst_ptr =
                    dst + (outer_idx * out_dimsize + dst_dim) * inner;
                memcpy(dst_ptr, src, (size_t)inner * sizeof(float));
                dst_dim++;
            }
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_squeeze_all_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_squeeze_all_nd: invalid blob");

    int32_t new_shape[LLM_MAX_NDIM];
    int new_ndim = 0;
    for (int i = 0; i < t.ndim; i++) {
        if (t.shape[i] != 1)
            new_shape[new_ndim++] = t.shape[i];
    }
    if (new_ndim == 0) {
        new_shape[0] = 1;
        new_ndim = 1;
    }

    int out_sz;
    void *out = nd_result_alloc(new_ndim, new_shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    memcpy((char *)out + LLM_ND_HDR(new_ndim),
           t.data,
           (size_t)t.total * sizeof(float));
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_flatten_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_flatten_nd: invalid blob");
    int start_dim = sqlite3_value_int(argv[1]);
    int end_dim = sqlite3_value_int(argv[2]);
    if (start_dim < 0)
        start_dim += t.ndim;
    if (end_dim < 0)
        end_dim += t.ndim;
    if (start_dim < 0)
        start_dim = 0;
    if (end_dim >= t.ndim)
        end_dim = t.ndim - 1;
    if (start_dim > end_dim)
        ERR(ctx, "llm_flatten_nd: start_dim > end_dim");

    int32_t new_shape[LLM_MAX_NDIM];
    int new_ndim = 0;
    for (int i = 0; i < start_dim; i++)
        new_shape[new_ndim++] = t.shape[i];
    int32_t flat = 1;
    for (int i = start_dim; i <= end_dim; i++)
        flat *= t.shape[i];
    new_shape[new_ndim++] = flat;
    for (int i = end_dim + 1; i < t.ndim; i++)
        new_shape[new_ndim++] = t.shape[i];

    int out_sz;
    void *out = nd_result_alloc(new_ndim, new_shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    memcpy((char *)out + LLM_ND_HDR(new_ndim),
           t.data,
           (size_t)t.total * sizeof(float));
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_cat_multi_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2)
        ERR(ctx, "llm_cat_multi_nd: need dim + at least 1 tensor");
    int dim = sqlite3_value_int(argv[0]);
    int n_tensors = argc - 1;

    TensorNDF32 ts[LLM_MAX_CAT_TENSORS];
    if (n_tensors > LLM_MAX_CAT_TENSORS)
        ERR(ctx, "llm_cat_multi_nd: too many tensors");
    for (int i = 0; i < n_tensors; i++) {
        if (llm_to_nd(sqlite3_value_blob(argv[1 + i]),
                      sqlite3_value_bytes(argv[1 + i]),
                      &ts[i]) < 0)
            ERR(ctx, "llm_cat_multi_nd: invalid tensor blob");
    }

    int ndim = ts[0].ndim;
    if (dim < 0)
        dim += ndim;
    if (dim < 0 || dim >= ndim)
        ERR(ctx, "llm_cat_multi_nd: dim out of range");

    int32_t out_shape[LLM_MAX_NDIM];
    for (int d = 0; d < ndim; d++)
        out_shape[d] = ts[0].shape[d];
    int32_t cat_dim_total = ts[0].shape[dim];
    for (int i = 1; i < n_tensors; i++)
        cat_dim_total += ts[i].shape[dim];
    out_shape[dim] = cat_dim_total;

    int32_t out_total = 1;
    for (int d = 0; d < ndim; d++)
        out_total *= out_shape[d];

    int out_sz;
    void *out = nd_result_alloc(ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ndim));

    int32_t outer = 1, inner = 1;
    for (int d = 0; d < dim; d++)
        outer *= out_shape[d];
    for (int d = dim + 1; d < ndim; d++)
        inner *= out_shape[d];

    for (int32_t outer_idx = 0; outer_idx < outer; outer_idx++) {
        int32_t dst_dim = 0;
        for (int i = 0; i < n_tensors; i++) {
            int32_t tensor_dim = ts[i].shape[dim];
            const float *src = ts[i].data + outer_idx * tensor_dim * inner;
            float *dst_ptr =
                dst + (outer_idx * cat_dim_total + dst_dim) * inner;
            memcpy(dst_ptr, src, (size_t)(tensor_dim * inner) * sizeof(float));
            dst_dim += tensor_dim;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_stack_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2)
        ERR(ctx, "llm_stack_nd: need dim + at least 1 tensor");
    int dim = sqlite3_value_int(argv[0]);
    int n_tensors = argc - 1;

    TensorNDF32 ts[LLM_MAX_CAT_TENSORS];
    if (n_tensors > LLM_MAX_CAT_TENSORS)
        ERR(ctx, "llm_stack_nd: too many tensors");
    for (int i = 0; i < n_tensors; i++) {
        if (llm_to_nd(sqlite3_value_blob(argv[1 + i]),
                      sqlite3_value_bytes(argv[1 + i]),
                      &ts[i]) < 0)
            ERR(ctx, "llm_stack_nd: invalid tensor blob");
    }

    int src_ndim = ts[0].ndim;
    int out_ndim = src_ndim + 1;
    if (dim < 0)
        dim += out_ndim;
    if (dim < 0 || dim > src_ndim)
        ERR(ctx, "llm_stack_nd: dim out of range");
    if (out_ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_stack_nd: too many dims");

    int32_t out_shape[LLM_MAX_NDIM];
    int src_idx = 0;
    for (int d = 0; d < out_ndim; d++) {
        if (d == dim)
            out_shape[d] = n_tensors;
        else
            out_shape[d] = ts[0].shape[src_idx++];
    }

    int32_t out_total = 1;
    for (int d = 0; d < out_ndim; d++)
        out_total *= out_shape[d];

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    int32_t outer = 1, inner = 1;
    for (int d = 0; d < dim; d++)
        outer *= ts[0].shape[d];
    for (int d = dim; d < src_ndim; d++)
        inner *= ts[0].shape[d];

    for (int32_t outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (int i = 0; i < n_tensors; i++) {
            const float *src = ts[i].data + outer_idx * inner;
            float *dst_ptr = dst + (outer_idx * n_tensors + i) * inner;
            memcpy(dst_ptr, src, (size_t)inner * sizeof(float));
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}