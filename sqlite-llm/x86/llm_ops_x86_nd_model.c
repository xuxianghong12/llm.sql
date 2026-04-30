/*
 * llm_ops_x86_nd_model.c - N-D normalization, embedding, and attention wrappers
 *
 * This file owns higher-level N-D operators that encode common model building
 * blocks rather than generic tensor algebra: normalization layers, embedding
 * lookup, mask conversion, and fused scaled dot-product attention.
 *
 * Maintenance guidance:
 *   - Keep model-specific invariants close to the wrapper boundary so shape
 *     assumptions stay explicit.
 *   - Reuse generic N-D helpers for allocation and shape traversal, but keep
 *     fused model logic here instead of scattering it across utility files.
 *   - If an operator becomes generic tensor math rather than model plumbing,
 *     move it into the relevant algebra/reduce/utility split instead.
 */

#include "llm_ops_x86_internal.h"
SQLITE_EXTENSION_INIT3

void sql_layernorm_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, tw;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_layernorm_nd: invalid x blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tw) < 0)
        ERR(ctx, "llm_layernorm_nd: invalid weight blob");
    if (tw.ndim != 1)
        ERR(ctx, "llm_layernorm_nd: weight must be 1-D");

    int has_bias = (sqlite3_value_type(argv[2]) != SQLITE_NULL);
    TensorNDF32 tb;
    memset(&tb, 0, sizeof(tb));
    if (has_bias) {
        if (llm_to_nd(sqlite3_value_blob(argv[2]),
                      sqlite3_value_bytes(argv[2]),
                      &tb) < 0)
            ERR(ctx, "llm_layernorm_nd: invalid bias blob");
    }

    float eps = (float)sqlite3_value_double(argv[3]);
    int32_t norm_size = tw.shape[0];
    if (tx.total % norm_size != 0)
        ERR(ctx, "llm_layernorm_nd: size mismatch");
    int32_t outer = tx.total / norm_size;

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, tx.shape, tx.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    for (int32_t row_idx = 0; row_idx < outer; row_idx++) {
        const float *src = tx.data + row_idx * norm_size;
        float *dst_row = dst + row_idx * norm_size;
        double sum = 0.0;
        for (int32_t col_idx = 0; col_idx < norm_size; col_idx++)
            sum += (double)src[col_idx];
        double mean = sum / (double)norm_size;

        double var = 0.0;
        for (int32_t col_idx = 0; col_idx < norm_size; col_idx++) {
            double diff = (double)src[col_idx] - mean;
            var += diff * diff;
        }
        var /= (double)norm_size;
        double rstd = 1.0 / sqrt(var + (double)eps);

        for (int32_t col_idx = 0; col_idx < norm_size; col_idx++) {
            double normed = ((double)src[col_idx] - mean) * rstd;
            dst_row[col_idx] = (float)(normed * (double)tw.data[col_idx]);
            if (has_bias)
                dst_row[col_idx] += tb.data[col_idx];
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_embedding_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tw, ti;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tw) < 0)
        ERR(ctx, "llm_embedding_nd: invalid weight blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &ti) < 0)
        ERR(ctx, "llm_embedding_nd: invalid indices blob");
    if (tw.ndim != 2)
        ERR(ctx, "llm_embedding_nd: weight must be 2-D");

    int32_t vocab = tw.shape[0];
    int32_t embed = tw.shape[1];
    int out_ndim = ti.ndim + 1;
    if (out_ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_embedding_nd: too many dims");

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
        if (idx < 0 || idx >= vocab)
            ERR(ctx, "llm_embedding_nd: index out of range");
        memcpy(dst + (size_t)i * embed,
               tw.data + (size_t)idx * embed,
               (size_t)embed * sizeof(float));
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_rmsnorm_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, tw;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_rmsnorm_nd: invalid x blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tw) < 0)
        ERR(ctx, "llm_rmsnorm_nd: invalid weight blob");
    if (tw.ndim != 1)
        ERR(ctx, "llm_rmsnorm_nd: weight must be 1-D");
    float eps = (float)sqlite3_value_double(argv[2]);

    int32_t last = tx.shape[tx.ndim - 1];
    if (tw.shape[0] != last)
        ERR(ctx, "llm_rmsnorm_nd: weight size mismatch");

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, tx.shape, tx.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    int32_t nrows = tx.total / last;
    for (int32_t row_idx = 0; row_idx < nrows; row_idx++) {
        const float *row = tx.data + (size_t)row_idx * last;
        float *out_row = dst + (size_t)row_idx * last;
        float sumsq = 0.0f;
        for (int32_t col_idx = 0; col_idx < last; col_idx++)
            sumsq += row[col_idx] * row[col_idx];
        float rms = 1.0f / sqrtf(sumsq / (float)last + eps);
        for (int32_t col_idx = 0; col_idx < last; col_idx++)
            out_row[col_idx] = row[col_idx] * rms * tw.data[col_idx];
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_norm_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_norm_nd: invalid blob");
    float p = (float)sqlite3_value_double(argv[1]);
    int dim = sqlite3_value_int(argv[2]);
    int keepdim = sqlite3_value_int(argv[3]);

    if (dim == -999) {
        double acc = 0.0;
        for (int32_t i = 0; i < t.total; i++)
            acc += pow(fabs((double)t.data[i]), (double)p);
        float result = (float)pow(acc, 1.0 / (double)p);
        int32_t shape1 = 1;
        int out_sz;
        void *out = nd_result_alloc(1, &shape1, 1, &out_sz);
        if (!out) {
            sqlite3_result_error_nomem(ctx);
            return;
        }
        float *dst = (float *)((char *)out + LLM_ND_HDR(1));
        dst[0] = result;
        sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
        return;
    }

    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_norm_nd: dim out of range");

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

    for (int32_t outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (int32_t inner_idx = 0; inner_idx < inner; inner_idx++) {
            double acc = 0.0;
            for (int32_t dim_idx = 0; dim_idx < dimsize; dim_idx++) {
                float value =
                    t.data[(outer_idx * dimsize + dim_idx) * inner + inner_idx];
                acc += pow(fabs((double)value), (double)p);
            }
            dst[outer_idx * inner + inner_idx] =
                (float)pow(acc, 1.0 / (double)p);
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_bool_to_additive_mask_nd(sqlite3_context *ctx,
                                  int argc,
                                  sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_bool_to_additive_mask_nd: invalid blob");
    float scale = (float)sqlite3_value_double(argv[1]);

    if (t.total == 0 || t.data[0] <= 0.5f) {
        sqlite3_result_blob(ctx, blob, sz, SQLITE_TRANSIENT);
        return;
    }

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = (t.data[i] - 1.0f) * scale;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_sdpa_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc != 5 && argc != 6)
        ERR(ctx, "llm_sdpa_nd: need 5 or 6 arguments");
    TensorNDF32 tq, tk, tv;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tq) < 0)
        ERR(ctx, "llm_sdpa_nd: invalid Q blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tk) < 0)
        ERR(ctx, "llm_sdpa_nd: invalid K blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &tv) < 0)
        ERR(ctx, "llm_sdpa_nd: invalid V blob");
    float scale = (float)sqlite3_value_double(argv[3]);
    int is_causal = sqlite3_value_int(argv[4]);

    if (tq.ndim < 2 || tk.ndim < 2 || tv.ndim < 2)
        ERR(ctx, "llm_sdpa_nd: need at least 2-D tensors");
    if (tq.ndim != tk.ndim || tq.ndim != tv.ndim)
        ERR(ctx, "llm_sdpa_nd: Q, K, V must have same ndim");

    int nd = tq.ndim;
    int seq_q = tq.shape[nd - 2];
    int d_k = tq.shape[nd - 1];
    int seq_k = tk.shape[nd - 2];
    int d_kk = tk.shape[nd - 1];
    int d_v = tv.shape[nd - 1];
    int seq_kv = tv.shape[nd - 2];

    if (d_k != d_kk)
        ERR(ctx, "llm_sdpa_nd: Q head_dim != K head_dim");
    if (seq_k != seq_kv)
        ERR(ctx, "llm_sdpa_nd: K seq_len != V seq_len");

    int32_t batch = 1;
    for (int i = 0; i < nd - 2; i++) {
        if (tq.shape[i] != tk.shape[i] || tq.shape[i] != tv.shape[i])
            ERR(ctx, "llm_sdpa_nd: batch dims mismatch");
        batch *= tq.shape[i];
    }

    int32_t scores_n = batch * seq_q * seq_k;
    float *scores = (float *)sqlite3_malloc64((size_t)scores_n * sizeof(float));
    if (!scores) {
        sqlite3_result_error_nomem(ctx);
        return;
    }

    for (int b = 0; b < batch; b++) {
        const float *q_ptr = tq.data + (size_t)b * seq_q * d_k;
        const float *k_ptr = tk.data + (size_t)b * seq_k * d_k;
        float *s_ptr = scores + (size_t)b * seq_q * seq_k;
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    seq_q,
                    seq_k,
                    d_k,
                    scale,
                    q_ptr,
                    d_k,
                    k_ptr,
                    d_k,
                    0.0f,
                    s_ptr,
                    seq_k);
    }

    if (is_causal) {
        int offset = seq_k - seq_q;
        for (int b = 0; b < batch; b++) {
            float *s_ptr = scores + (size_t)b * seq_q * seq_k;
            for (int i = 0; i < seq_q; i++) {
                for (int j = i + 1 + offset; j < seq_k; j++)
                    s_ptr[i * seq_k + j] = -1e9f;
            }
        }
    }

    if (argc == 6 && sqlite3_value_type(argv[5]) != SQLITE_NULL) {
        TensorNDF32 tm;
        if (llm_to_nd(sqlite3_value_blob(argv[5]),
                      sqlite3_value_bytes(argv[5]),
                      &tm) < 0) {
            sqlite3_free(scores);
            ERR(ctx, "llm_sdpa_nd: invalid mask blob");
        }
        int32_t mask_seq_q = tm.shape[tm.ndim - 2];
        int32_t mask_seq_k = tm.shape[tm.ndim - 1];
        int32_t mask_batch = tm.total / (mask_seq_q * mask_seq_k);
        for (int b = 0; b < batch; b++) {
            float *s_ptr = scores + (size_t)b * seq_q * seq_k;
            int mb = b % mask_batch;
            const float *m_ptr = tm.data + (size_t)mb * mask_seq_q * mask_seq_k;
            for (int i = 0; i < seq_q; i++) {
                int mi = i < mask_seq_q ? i : mask_seq_q - 1;
                for (int j = 0; j < seq_k; j++) {
                    int mj = j < mask_seq_k ? j : mask_seq_k - 1;
                    if (m_ptr[mi * mask_seq_k + mj] < 0.5f)
                        s_ptr[i * seq_k + j] = -1e9f;
                }
            }
        }
    }

    for (int b = 0; b < batch; b++) {
        float *s_ptr = scores + (size_t)b * seq_q * seq_k;
        for (int i = 0; i < seq_q; i++) {
            float *row = s_ptr + i * seq_k;
            float mx = -FLT_MAX;
            for (int j = 0; j < seq_k; j++) {
                if (row[j] > mx)
                    mx = row[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < seq_k; j++) {
                row[j] = expf(row[j] - mx);
                sum += row[j];
            }
            for (int j = 0; j < seq_k; j++)
                row[j] /= sum;
        }
    }

    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < nd - 2; i++)
        out_shape[i] = tq.shape[i];
    out_shape[nd - 2] = seq_q;
    out_shape[nd - 1] = d_v;
    int32_t out_total = batch * seq_q * d_v;

    int out_sz;
    void *out = nd_result_alloc(nd, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_free(scores);
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(nd));

    for (int b = 0; b < batch; b++) {
        const float *s_ptr = scores + (size_t)b * seq_q * seq_k;
        const float *v_ptr = tv.data + (size_t)b * seq_k * d_v;
        float *o_ptr = dst + (size_t)b * seq_q * d_v;
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    seq_q,
                    d_v,
                    seq_k,
                    1.0f,
                    s_ptr,
                    seq_k,
                    v_ptr,
                    d_v,
                    0.0f,
                    o_ptr,
                    d_v);
    }

    sqlite3_free(scores);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}