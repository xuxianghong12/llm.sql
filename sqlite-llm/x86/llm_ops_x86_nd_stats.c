/*
 * llm_ops_x86_nd_stats.c - N-D softmax, aggregate, and string helpers
 *
 * This file groups the remaining N-D wrappers that compute statistics or
 * provide debug-friendly tensor summaries without fitting the existing point-
 * wise or reduction buckets cleanly.
 *
 * Maintenance guidance:
 *   - Keep dim-reduction shape math aligned with nd_reduce helpers.
 *   - Keep softmax numerically stable and shape-preserving.
 *   - String formatting here is for diagnostics only; avoid coupling it to
 *     binary tensor layout changes beyond llm_to_nd().
 */

#include "llm_ops_x86_internal.h"
SQLITE_EXTENSION_INIT3

void sql_softmax_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_softmax_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_softmax_nd: dim out of range");

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    memcpy(dst, t.data, (size_t)t.total * sizeof(float));

    int32_t strides[LLM_MAX_NDIM];
    nd_strides(t.shape, t.ndim, strides);
    int32_t dim_sz = t.shape[dim];
    int32_t dim_str = strides[dim];
    int32_t outer = t.total / dim_sz;
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

    int32_t oidx[LLM_MAX_NDIM];
    memset(oidx, 0, sizeof(oidx));
    for (int32_t slice = 0; slice < outer; slice++) {
        int32_t base = 0;
        for (int i = 0; i < on; i++)
            base += oidx[i] * other_strides[i];
        float mx = -FLT_MAX;
        for (int32_t k = 0; k < dim_sz; k++) {
            float v = dst[base + k * dim_str];
            if (v > mx)
                mx = v;
        }
        float sum = 0.0f;
        for (int32_t k = 0; k < dim_sz; k++) {
            float e = expf(dst[base + k * dim_str] - mx);
            dst[base + k * dim_str] = e;
            sum += e;
        }
        for (int32_t k = 0; k < dim_sz; k++)
            dst[base + k * dim_str] /= sum;
        for (int i = on - 1; i >= 0; i--) {
            if (++oidx[i] < other_shape[i])
                break;
            oidx[i] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_mean_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_mean_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    int keepdim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_mean_nd: dim out of range");

    int32_t new_shape[LLM_MAX_NDIM];
    int new_ndim;
    if (keepdim) {
        new_ndim = t.ndim;
        for (int i = 0; i < t.ndim; i++)
            new_shape[i] = t.shape[i];
        new_shape[dim] = 1;
    } else {
        new_ndim = t.ndim - 1;
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
    memset(dst, 0, (size_t)new_total * sizeof(float));

    int32_t dim_sz = t.shape[dim];
    int32_t out_strides[LLM_MAX_NDIM];
    nd_strides(new_shape, new_ndim, out_strides);

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < t.total; flat++) {
        int32_t out_off = 0;
        if (keepdim) {
            for (int d = 0; d < t.ndim; d++) {
                int32_t coord = d == dim ? 0 : idx[d];
                out_off += coord * out_strides[d];
            }
        } else {
            int j = 0;
            for (int d = 0; d < t.ndim; d++) {
                if (d == dim)
                    continue;
                out_off += idx[d] * out_strides[j++];
            }
        }
        dst[out_off] += t.data[flat];
        for (int d = t.ndim - 1; d >= 0; d--) {
            if (++idx[d] < t.shape[d])
                break;
            idx[d] = 0;
        }
    }
    for (int32_t i = 0; i < new_total; i++)
        dst[i] /= (float)dim_sz;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_sum_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_sum_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    int keepdim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_sum_nd: dim out of range");

    int32_t new_shape[LLM_MAX_NDIM];
    int new_ndim;
    if (keepdim) {
        new_ndim = t.ndim;
        for (int i = 0; i < t.ndim; i++)
            new_shape[i] = t.shape[i];
        new_shape[dim] = 1;
    } else {
        new_ndim = t.ndim - 1;
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
    memset(dst, 0, (size_t)new_total * sizeof(float));

    int32_t out_strides[LLM_MAX_NDIM];
    nd_strides(new_shape, new_ndim, out_strides);
    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < t.total; flat++) {
        int32_t out_off = 0;
        if (keepdim) {
            for (int d = 0; d < t.ndim; d++) {
                int32_t coord = d == dim ? 0 : idx[d];
                out_off += coord * out_strides[d];
            }
        } else {
            int j = 0;
            for (int d = 0; d < t.ndim; d++) {
                if (d == dim)
                    continue;
                out_off += idx[d] * out_strides[j++];
            }
        }
        dst[out_off] += t.data[flat];
        for (int d = t.ndim - 1; d >= 0; d--) {
            if (++idx[d] < t.shape[d])
                break;
            idx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_str_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0) {
        sqlite3_result_text(ctx, "<invalid blob>", -1, SQLITE_STATIC);
        return;
    }
    char buf[512];
    int off = snprintf(buf, sizeof(buf), "tensor(shape=[");
    if (off < 0)
        off = 0;
    if ((size_t)off >= sizeof(buf))
        off = (int)(sizeof(buf) - 1);
    for (int i = 0; i < t.ndim; i++) {
        int n = snprintf(buf + off,
                         sizeof(buf) - (size_t)off,
                         "%d%s",
                         t.shape[i],
                         i < t.ndim - 1 ? "," : "");
        if (n > 0)
            off += n;
        if ((size_t)off >= sizeof(buf)) {
            off = (int)(sizeof(buf) - 1);
            break;
        }
    }
    if ((size_t)off < sizeof(buf) - 1) {
        int n = snprintf(buf + off, sizeof(buf) - (size_t)off, "] data=[");
        if (n > 0)
            off += n;
        if ((size_t)off >= sizeof(buf))
            off = (int)(sizeof(buf) - 1);
    }
    int show = t.total < 6 ? t.total : 6;
    for (int i = 0; i < show; i++) {
        if ((size_t)off >= sizeof(buf) - 1) {
            off = (int)(sizeof(buf) - 1);
            break;
        }
        int n = snprintf(buf + off,
                         sizeof(buf) - (size_t)off,
                         "%.4g%s",
                         t.data[i],
                         i < show - 1 ? "," : "");
        if (n > 0)
            off += n;
        if ((size_t)off >= sizeof(buf)) {
            off = (int)(sizeof(buf) - 1);
            break;
        }
    }
    if (t.total > 6 && (size_t)off < sizeof(buf) - 1) {
        int n = snprintf(buf + off, sizeof(buf) - (size_t)off, ",...");
        if (n > 0)
            off += n;
        if ((size_t)off >= sizeof(buf))
            off = (int)(sizeof(buf) - 1);
    }
    if ((size_t)off < sizeof(buf) - 1)
        snprintf(buf + off, sizeof(buf) - (size_t)off, "])");
    sqlite3_result_text(ctx, buf, -1, SQLITE_TRANSIENT);
}