/*
 * llm_ops_x86_nd_algebra.c - N-D batch matmul and broadcast algebra wrappers
 *
 * This file owns N-D SQL wrappers that perform numeric work across tensors:
 * batch matrix multiply, linear projection, broadcasted arithmetic, and
 * broadcasted comparisons. Add new code here when the operation combines
 * tensor values mathematically rather than only reshaping or indexing them.
 *
 * Maintenance guidance:
 *   - Keep broadcast-specific helper logic local unless another N-D category
 *     reuses it broadly.
 *   - Reuse broadcast_shape() from the shared internal layer so later N-D
 *     operators agree on shape inference and broadcasting rules.
 *   - Prefer BLAS fast paths for GEMM-like work and keep wrapper validation
 *     close to the SQL entry point.
 */

#include "llm_ops_x86_internal.h"
SQLITE_EXTENSION_INIT3

static float llm_cmp_mask_to_fp32(int cmp) {
    return cmp ? 1.0f : 0.0f;
}

static void broadcast_op(const TensorNDF32 *ta,
                         const TensorNDF32 *tb,
                         float *dst,
                         int32_t total,
                         const int32_t *out_shape,
                         int ndim,
                         int op) {
    int off_a = ndim - ta->ndim;
    int off_b = ndim - tb->ndim;

    int32_t sa[LLM_MAX_NDIM], sb[LLM_MAX_NDIM];
    int32_t pa[LLM_MAX_NDIM], pb[LLM_MAX_NDIM];
    for (int i = 0; i < off_a; i++)
        pa[i] = 1;
    for (int i = 0; i < ta->ndim; i++)
        pa[i + off_a] = ta->shape[i];
    for (int i = 0; i < off_b; i++)
        pb[i] = 1;
    for (int i = 0; i < tb->ndim; i++)
        pb[i + off_b] = tb->shape[i];

    nd_strides(pa, ndim, sa);
    nd_strides(pb, ndim, sb);
    for (int i = 0; i < ndim; i++) {
        if (pa[i] == 1)
            sa[i] = 0;
        if (pb[i] == 1)
            sb[i] = 0;
    }

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < total; flat++) {
        int32_t oa = 0, ob = 0;
        for (int d = 0; d < ndim; d++) {
            oa += idx[d] * sa[d];
            ob += idx[d] * sb[d];
        }
        float va = ta->data[oa], vb = tb->data[ob];
        if (op == 0)
            dst[flat] = va + vb;
        else if (op == 1)
            dst[flat] = va * vb;
        else if (op == 2)
            dst[flat] = va - vb;
        else if (op == 3)
            dst[flat] = (vb != 0.0f) ? va / vb : 0.0f;
        else if (op == 4)
            dst[flat] = llm_cmp_mask_to_fp32(va > vb);
        else if (op == 5)
            dst[flat] = llm_cmp_mask_to_fp32(va < vb);
        else if (op == 6)
            dst[flat] = llm_cmp_mask_to_fp32(va >= vb);
        else if (op == 7)
            dst[flat] = llm_cmp_mask_to_fp32(va <= vb);
        else if (op == 8)
            dst[flat] = llm_cmp_mask_to_fp32(va == vb);
        else if (op == 9)
            dst[flat] = llm_cmp_mask_to_fp32(va != vb);
        for (int d = ndim - 1; d >= 0; d--) {
            if (++idx[d] < out_shape[d])
                break;
            idx[d] = 0;
        }
    }
}

static void sql_cmp_nd_impl(sqlite3_context *ctx,
                            sqlite3_value **argv,
                            int op) {
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_cmp_nd: invalid a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_cmp_nd: invalid b");
    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim;
    int32_t total = broadcast_shape(&ta, &tb, out_shape, &out_ndim);
    if (total < 0)
        ERR(ctx, "llm_cmp_nd: incompatible shapes");
    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    broadcast_op(&ta,
                 &tb,
                 (float *)((char *)out + LLM_ND_HDR(out_ndim)),
                 total,
                 out_shape,
                 out_ndim,
                 op);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_bmm(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_bmm: invalid A blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_bmm: invalid B blob");
    if (ta.ndim < 2 || tb.ndim < 2)
        ERR(ctx, "llm_bmm: need at least 2-D tensors");
    if (ta.ndim != tb.ndim)
        ERR(ctx, "llm_bmm: A and B must have same ndim");

    int nd = ta.ndim;
    int M = ta.shape[nd - 2];
    int K = ta.shape[nd - 1];
    int K2 = tb.shape[nd - 2];
    int N = tb.shape[nd - 1];
    if (K != K2)
        ERR(ctx, "llm_bmm: inner dims mismatch");

    int32_t batch = 1;
    int32_t new_shape[LLM_MAX_NDIM];
    for (int i = 0; i < nd - 2; i++) {
        if (ta.shape[i] != tb.shape[i])
            ERR(ctx, "llm_bmm: batch dims mismatch");
        new_shape[i] = ta.shape[i];
        batch *= ta.shape[i];
    }
    new_shape[nd - 2] = M;
    new_shape[nd - 1] = N;
    int32_t new_total = batch * M * N;

    int out_sz;
    void *out = nd_result_alloc(nd, new_shape, new_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(nd));

    if (M == 1) {
        for (int b = 0; b < batch; b++) {
            const float *a_ptr = ta.data + (size_t)b * K;
            const float *b_ptr = tb.data + (size_t)b * K * N;
            float *c_ptr = dst + (size_t)b * N;
            cblas_sgemv(CblasRowMajor,
                        CblasTrans,
                        K,
                        N,
                        1.0f,
                        b_ptr,
                        N,
                        a_ptr,
                        1,
                        0.0f,
                        c_ptr,
                        1);
        }
    } else {
        for (int b = 0; b < batch; b++) {
            const float *a_ptr = ta.data + (size_t)b * M * K;
            const float *b_ptr = tb.data + (size_t)b * K * N;
            float *c_ptr = dst + (size_t)b * M * N;
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        M,
                        N,
                        K,
                        1.0f,
                        a_ptr,
                        K,
                        b_ptr,
                        N,
                        0.0f,
                        c_ptr,
                        N);
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_linear_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, tw;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_linear_nd: invalid x");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tw) < 0)
        ERR(ctx, "llm_linear_nd: invalid weight");
    if (tx.ndim < 1)
        ERR(ctx, "llm_linear_nd: x must be at least 1-D");
    if (tw.ndim != 2)
        ERR(ctx, "llm_linear_nd: weight must be 2-D");

    int32_t k = tx.shape[tx.ndim - 1];
    if (tw.shape[1] != k)
        ERR(ctx, "llm_linear_nd: in_features mismatch");

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
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_add_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_add_nd: invalid a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_add_nd: invalid b");
    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim;
    int32_t total = broadcast_shape(&ta, &tb, out_shape, &out_ndim);
    if (total < 0)
        ERR(ctx, "llm_add_nd: incompatible shapes");
    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    broadcast_op(&ta,
                 &tb,
                 (float *)((char *)out + LLM_ND_HDR(out_ndim)),
                 total,
                 out_shape,
                 out_ndim,
                 0);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_mul_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_mul_nd: invalid a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_mul_nd: invalid b");
    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim;
    int32_t total = broadcast_shape(&ta, &tb, out_shape, &out_ndim);
    if (total < 0)
        ERR(ctx, "llm_mul_nd: incompatible shapes");
    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    broadcast_op(&ta,
                 &tb,
                 (float *)((char *)out + LLM_ND_HDR(out_ndim)),
                 total,
                 out_shape,
                 out_ndim,
                 1);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_sub_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_sub_nd: invalid a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_sub_nd: invalid b");
    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim;
    int32_t total = broadcast_shape(&ta, &tb, out_shape, &out_ndim);
    if (total < 0)
        ERR(ctx, "llm_sub_nd: incompatible shapes");
    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    broadcast_op(&ta,
                 &tb,
                 (float *)((char *)out + LLM_ND_HDR(out_ndim)),
                 total,
                 out_shape,
                 out_ndim,
                 2);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_div_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_div_nd: invalid a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_div_nd: invalid b");
    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim;
    int32_t total = broadcast_shape(&ta, &tb, out_shape, &out_ndim);
    if (total < 0)
        ERR(ctx, "llm_div_nd: incompatible shapes");
    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    broadcast_op(&ta,
                 &tb,
                 (float *)((char *)out + LLM_ND_HDR(out_ndim)),
                 total,
                 out_shape,
                 out_ndim,
                 3);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_gt_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_cmp_nd_impl(ctx, argv, 4);
}

void sql_lt_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_cmp_nd_impl(ctx, argv, 5);
}

void sql_ge_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_cmp_nd_impl(ctx, argv, 6);
}

void sql_le_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_cmp_nd_impl(ctx, argv, 7);
}

void sql_eq_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_cmp_nd_impl(ctx, argv, 8);
}

void sql_ne_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_cmp_nd_impl(ctx, argv, 9);
}