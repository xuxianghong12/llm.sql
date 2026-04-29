/*
 * llm_ops_x86_nd_layout.c - N-D tensor layout and indexing operators
 *
 * This file owns N-D SQL wrappers whose primary job is to reshape tensors,
 * reorder axes, slice ranges, select indices, concatenate inputs, or expand
 * data across broadcast dimensions. When adding a new operation, place it here
 * if the behavior is mostly about tensor layout and data movement rather than
 * model-specific math kernels.
 *
 * Maintenance guidance:
 *   - Reuse nd_strides() and nd_result_alloc() for shared shape logic.
 *   - Keep SQL validation close to the wrapper that owns the behavior.
 *   - If a helper is reused by other N-D categories, declare it in
 *     llm_ops_x86_internal.h instead of duplicating it here.
 */

#include "llm_ops_x86_internal.h"
SQLITE_EXTENSION_INIT3

void sql_ndim(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0) {
        ERR(ctx, "Invalid tensor blob");
    }
    sqlite3_result_int(ctx, t.ndim);
}

void sql_shape(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0) {
        ERR(ctx, "Invalid tensor blob");
    }
    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_shape: dim out of range");
    sqlite3_result_int(ctx, t.shape[dim]);
}

void sql_view_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2)
        ERR(ctx, "llm_view_nd: need blob + at least 1 dim");
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_view_nd: invalid blob");

    int new_ndim = argc - 1;
    if (new_ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_view_nd: too many dims");

    int32_t new_shape[LLM_MAX_NDIM];
    int32_t total = 1;
    int infer_idx = -1;
    for (int i = 0; i < new_ndim; i++) {
        new_shape[i] = sqlite3_value_int(argv[i + 1]);
        if (new_shape[i] == -1) {
            if (infer_idx >= 0)
                ERR(ctx, "llm_view_nd: only one -1 allowed");
            infer_idx = i;
        } else {
            if (new_shape[i] < 0)
                ERR(ctx, "llm_view_nd: negative dim");
            total *= new_shape[i];
        }
    }
    if (infer_idx >= 0) {
        if (total == 0 || t.total % total != 0)
            ERR(ctx, "llm_view_nd: cannot infer dim");
        new_shape[infer_idx] = t.total / total;
        total = t.total;
    }
    if (total != t.total)
        ERR(ctx, "llm_view_nd: element count mismatch");

    int out_sz;
    void *out = nd_result_alloc(new_ndim, new_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    memcpy((char *)out + LLM_ND_HDR(new_ndim),
           t.data,
           (size_t)total * sizeof(float));
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_permute_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2)
        ERR(ctx, "llm_permute_nd: need blob + perm axes");
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_permute_nd: invalid blob");
    int ndim = t.ndim;
    if (argc - 1 != ndim)
        ERR(ctx, "llm_permute_nd: wrong number of perm axes");

    int32_t perm[LLM_MAX_NDIM];
    int seen[LLM_MAX_NDIM] = {0};
    for (int i = 0; i < ndim; i++) {
        perm[i] = sqlite3_value_int(argv[i + 1]);
        if (perm[i] < 0 || perm[i] >= ndim || seen[perm[i]])
            ERR(ctx, "llm_permute_nd: invalid permutation");
        seen[perm[i]] = 1;
    }

    int32_t new_shape[LLM_MAX_NDIM];
    for (int i = 0; i < ndim; i++)
        new_shape[i] = t.shape[perm[i]];

    int out_sz;
    void *out = nd_result_alloc(ndim, new_shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ndim));

    int32_t src_strides[LLM_MAX_NDIM];
    nd_strides(t.shape, ndim, src_strides);

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < t.total; flat++) {
        int32_t src_off = 0;
        for (int d = 0; d < ndim; d++)
            src_off += idx[d] * src_strides[perm[d]];
        dst[flat] = t.data[src_off];
        for (int d = ndim - 1; d >= 0; d--) {
            if (++idx[d] < new_shape[d])
                break;
            idx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_transpose_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_transpose_nd: invalid blob");
    int d0 = sqlite3_value_int(argv[1]);
    int d1 = sqlite3_value_int(argv[2]);
    if (d0 < 0)
        d0 += t.ndim;
    if (d1 < 0)
        d1 += t.ndim;
    if (d0 < 0 || d0 >= t.ndim || d1 < 0 || d1 >= t.ndim)
        ERR(ctx, "llm_transpose_nd: dim out of range");

    int32_t perm[LLM_MAX_NDIM];
    for (int i = 0; i < t.ndim; i++)
        perm[i] = i;
    perm[d0] = d1;
    perm[d1] = d0;

    int32_t new_shape[LLM_MAX_NDIM];
    for (int i = 0; i < t.ndim; i++)
        new_shape[i] = t.shape[perm[i]];
    int out_sz;
    void *out = nd_result_alloc(t.ndim, new_shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    int32_t src_strides[LLM_MAX_NDIM];
    nd_strides(t.shape, t.ndim, src_strides);

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < t.total; flat++) {
        int32_t src_off = 0;
        for (int d = 0; d < t.ndim; d++)
            src_off += idx[d] * src_strides[perm[d]];
        dst[flat] = t.data[src_off];
        for (int d = t.ndim - 1; d >= 0; d--) {
            if (++idx[d] < new_shape[d])
                break;
            idx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_unsqueeze_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_unsqueeze_nd: invalid blob");
    if (t.ndim >= LLM_MAX_NDIM)
        ERR(ctx, "llm_unsqueeze_nd: already at max dims");

    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim + 1;
    if (dim < 0 || dim > t.ndim)
        ERR(ctx, "llm_unsqueeze_nd: dim out of range");

    int32_t new_shape[LLM_MAX_NDIM];
    int new_ndim = t.ndim + 1;
    for (int i = 0; i < dim; i++)
        new_shape[i] = t.shape[i];
    new_shape[dim] = 1;
    for (int i = dim; i < t.ndim; i++)
        new_shape[i + 1] = t.shape[i];

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

void sql_view_t12_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 3)
        ERR(ctx, "llm_view_t12_nd: need blob + at least 2 shape dims");
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_view_t12_nd: invalid blob");

    int new_ndim = argc - 1;
    if (new_ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_view_t12_nd: too many dims");
    int32_t view_shape[LLM_MAX_NDIM];
    int neg_idx = -1;
    int32_t known = 1;
    for (int i = 0; i < new_ndim; i++) {
        view_shape[i] = sqlite3_value_int(argv[i + 1]);
        if (view_shape[i] == -1)
            neg_idx = i;
        else
            known *= view_shape[i];
    }
    if (neg_idx >= 0)
        view_shape[neg_idx] = t.total / known;
    if (new_ndim < 3)
        ERR(ctx, "llm_view_t12_nd: need at least 3 dims for transpose(1,2)");

    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < new_ndim; i++)
        out_shape[i] = view_shape[i];
    out_shape[1] = view_shape[2];
    out_shape[2] = view_shape[1];

    int out_sz;
    void *out = nd_result_alloc(new_ndim, out_shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(new_ndim));

    if (view_shape[1] == 1 || view_shape[2] == 1) {
        memcpy(dst, t.data, (size_t)t.total * sizeof(float));
    } else {
        int32_t perm[LLM_MAX_NDIM];
        for (int i = 0; i < new_ndim; i++)
            perm[i] = i;
        perm[1] = 2;
        perm[2] = 1;

        int32_t src_strides[LLM_MAX_NDIM];
        nd_strides(view_shape, new_ndim, src_strides);

        int32_t idx[LLM_MAX_NDIM];
        memset(idx, 0, sizeof(idx));
        for (int32_t flat = 0; flat < t.total; flat++) {
            int32_t src_off = 0;
            for (int d = 0; d < new_ndim; d++)
                src_off += idx[d] * src_strides[perm[d]];
            dst[flat] = t.data[src_off];
            for (int d = new_ndim - 1; d >= 0; d--) {
                if (++idx[d] < out_shape[d])
                    break;
                idx[d] = 0;
            }
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_squeeze_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_squeeze_nd: invalid blob");

    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_squeeze_nd: dim out of range");
    if (t.shape[dim] != 1)
        ERR(ctx, "llm_squeeze_nd: dim is not size 1");
    if (t.ndim <= 1)
        ERR(ctx, "llm_squeeze_nd: cannot reduce to 0 dims");

    int32_t new_shape[LLM_MAX_NDIM];
    int new_ndim = t.ndim - 1;
    int j = 0;
    for (int i = 0; i < t.ndim; i++)
        if (i != dim)
            new_shape[j++] = t.shape[i];

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

void sql_cat_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_cat_nd: invalid first blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_cat_nd: invalid second blob");
    if (ta.ndim != tb.ndim)
        ERR(ctx, "llm_cat_nd: ndim mismatch");

    int dim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += ta.ndim;
    if (dim < 0 || dim >= ta.ndim)
        ERR(ctx, "llm_cat_nd: dim out of range");

    for (int i = 0; i < ta.ndim; i++)
        if (i != dim && ta.shape[i] != tb.shape[i])
            ERR(ctx, "llm_cat_nd: shape mismatch on non-cat dim");

    int32_t new_shape[LLM_MAX_NDIM];
    for (int i = 0; i < ta.ndim; i++)
        new_shape[i] = ta.shape[i];
    new_shape[dim] = ta.shape[dim] + tb.shape[dim];
    int32_t new_total = 1;
    for (int i = 0; i < ta.ndim; i++)
        new_total *= new_shape[i];

    int out_sz;
    void *out = nd_result_alloc(ta.ndim, new_shape, new_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ta.ndim));

    int32_t outer = 1, inner = 1;
    for (int i = 0; i < dim; i++)
        outer *= ta.shape[i];
    for (int i = dim + 1; i < ta.ndim; i++)
        inner *= ta.shape[i];

    int32_t a_block = ta.shape[dim] * inner;
    int32_t b_block = tb.shape[dim] * inner;
    int32_t o_block = a_block + b_block;

    for (int32_t i = 0; i < outer; i++) {
        memcpy(dst + (size_t)i * o_block,
               ta.data + (size_t)i * a_block,
               (size_t)a_block * sizeof(float));
        memcpy(dst + (size_t)i * o_block + a_block,
               tb.data + (size_t)i * b_block,
               (size_t)b_block * sizeof(float));
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_slice_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc != 4 && argc != 5)
        ERR(ctx, "llm_slice_nd: need 4 or 5 arguments");
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_slice_nd: invalid blob");

    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_slice_nd: dim out of range");

    int s = sqlite3_value_int(argv[2]);
    int e = sqlite3_value_int(argv[3]);
    int step = argc >= 5 ? sqlite3_value_int(argv[4]) : 1;
    if (step <= 0)
        ERR(ctx, "llm_slice_nd: step must be > 0");
    int d_sz = t.shape[dim];
    if (s < 0)
        s += d_sz;
    if (e < 0)
        e += d_sz;
    if (s < 0)
        s = 0;
    if (s > d_sz)
        s = d_sz;
    if (e < 0)
        e = 0;
    if (e > d_sz)
        e = d_sz;
    if (s > e)
        s = e;

    int32_t new_shape[LLM_MAX_NDIM];
    for (int i = 0; i < t.ndim; i++)
        new_shape[i] = t.shape[i];
    int32_t span = e - s;
    int32_t sliced = span <= 0 ? 0 : (span + step - 1) / step;
    new_shape[dim] = sliced;
    int32_t new_total = 1;
    for (int i = 0; i < t.ndim; i++)
        new_total *= new_shape[i];

    int out_sz;
    void *out = nd_result_alloc(t.ndim, new_shape, new_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    int32_t strides[LLM_MAX_NDIM];
    nd_strides(t.shape, t.ndim, strides);

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < new_total; flat++) {
        int32_t src_off = 0;
        for (int d = 0; d < t.ndim; d++) {
            if (d == dim)
                src_off += (s + idx[d] * step) * strides[d];
            else
                src_off += idx[d] * strides[d];
        }
        dst[flat] = t.data[src_off];
        for (int d = t.ndim - 1; d >= 0; d--) {
            if (++idx[d] < new_shape[d])
                break;
            idx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_select_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_select_nd: invalid blob");
    if (t.ndim < 2)
        ERR(ctx, "llm_select_nd: need at least 2-D tensor");

    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_select_nd: dim out of range");

    int idx = sqlite3_value_int(argv[2]);
    if (idx < 0)
        idx += t.shape[dim];
    if (idx < 0 || idx >= t.shape[dim])
        ERR(ctx, "llm_select_nd: index out of range");

    int new_ndim = t.ndim - 1;
    int32_t new_shape[LLM_MAX_NDIM];
    int ni = 0;
    for (int i = 0; i < t.ndim; i++) {
        if (i != dim)
            new_shape[ni++] = t.shape[i];
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

    int32_t oidx[LLM_MAX_NDIM];
    memset(oidx, 0, sizeof(oidx));
    for (int32_t flat = 0; flat < new_total; flat++) {
        int32_t src_off = idx * strides[dim];
        int oi = 0;
        for (int d = 0; d < t.ndim; d++) {
            if (d != dim) {
                src_off += oidx[oi] * strides[d];
                oi++;
            }
        }
        dst[flat] = t.data[src_off];
        for (int d = new_ndim - 1; d >= 0; d--) {
            if (++oidx[d] < new_shape[d])
                break;
            oidx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_expand_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2)
        ERR(ctx, "llm_expand_nd: need blob + shape dims");
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_expand_nd: invalid blob");

    int new_ndim = argc - 1;
    if (new_ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_expand_nd: too many dims");
    if (new_ndim < t.ndim)
        ERR(ctx, "llm_expand_nd: cannot reduce ndim via expand");

    int32_t new_shape[LLM_MAX_NDIM];
    for (int i = 0; i < new_ndim; i++)
        new_shape[i] = sqlite3_value_int(argv[i + 1]);

    int offset = new_ndim - t.ndim;
    for (int i = 0; i < new_ndim; i++) {
        if (new_shape[i] == -1)
            new_shape[i] = (i < offset) ? 1 : t.shape[i - offset];
    }

    for (int i = 0; i < t.ndim; i++) {
        int src_d = t.shape[i];
        int dst_d = new_shape[i + offset];
        if (src_d != 1 && src_d != dst_d)
            ERR(ctx, "llm_expand_nd: incompatible shapes for expand");
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

    int32_t src_shape_padded[LLM_MAX_NDIM];
    for (int i = 0; i < offset; i++)
        src_shape_padded[i] = 1;
    for (int i = 0; i < t.ndim; i++)
        src_shape_padded[i + offset] = t.shape[i];

    int32_t src_strides[LLM_MAX_NDIM];
    nd_strides(src_shape_padded, new_ndim, src_strides);
    for (int i = 0; i < new_ndim; i++)
        if (src_shape_padded[i] == 1)
            src_strides[i] = 0;

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < new_total; flat++) {
        int32_t src_off = 0;
        for (int d = 0; d < new_ndim; d++)
            src_off += idx[d] * src_strides[d];
        dst[flat] = t.data[src_off];
        for (int d = new_ndim - 1; d >= 0; d--) {
            if (++idx[d] < new_shape[d])
                break;
            idx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}