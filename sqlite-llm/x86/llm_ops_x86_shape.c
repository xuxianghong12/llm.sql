/*
 * llm_ops_x86_shape.c - Legacy vector and matrix shape/index wrappers
 *
 * This file owns reshape-like and indexing-style operations for the pre-N-D
 * VecF32 and MatF32 APIs: reshape, flatten, slicing, concatenation, row
 * access, and gather. Keep legacy shape manipulation here so it stays separate
 * from the newer N-D tensor layout code.
 *
 * Maintenance guidance:
 *   - Preserve the historical VecF32/MatF32 blob contracts when editing this
 *     file; do not mix in TensorNDF32-specific logic unless migrating the API.
 *   - Centralize bounds checks near the SQL entry point so invalid indexing
 *     behavior stays predictable.
 *   - New rank-changing N-D operators belong in llm_ops_x86_nd_layout.c.
 */
#include "llm_ops_x86_internal.h"
SQLITE_EXTENSION_INIT3

void sql_reshape(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    int new_rows = sqlite3_value_int(argv[1]),
        new_cols = sqlite3_value_int(argv[2]);
    if (new_rows <= 0 || new_cols <= 0)
        ERR(ctx, "rows and cols must be positive");
    const void *blob = sqlite3_value_blob(argv[0]);
    int bsz = sqlite3_value_bytes(argv[0]);
    VecF32 v;
    MatF32 m;
    int nel = 0;
    const float *src = NULL;
    if (llm_parse_vec(blob, bsz, &v) == 0) {
        nel = v.n;
        src = v.data;
    } else if (llm_parse_mat(blob, bsz, &m) == 0) {
        nel = m.rows * m.cols;
        src = m.data;
    } else
        ERR(ctx, "Invalid blob");
    if (nel != new_rows * new_cols)
        ERR(ctx, "Total element count mismatch for reshape");
    int out_sz;
    void *out = alloc_mat_blob(new_rows, new_cols, src, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_flatten(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 m;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &m) < 0)
        ERR(ctx, "Invalid matrix blob");
    int nel = m.rows * m.cols;
    int out_sz;
    void *out = alloc_vec_blob(nel, m.data, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_slice(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 v;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &v) < 0)
        ERR(ctx, "Invalid vector blob");
    int start = sqlite3_value_int(argv[1]), end = sqlite3_value_int(argv[2]);
    if (end < -v.n)
        end = 0;
    if (end < 0)
        end = v.n + end;
    if (start < 0)
        start = 0;
    if (end > v.n)
        end = v.n;
    if (start >= end)
        ERR(ctx, "Empty or invalid slice range");
    int len = end - start;
    int out_sz;
    void *out = alloc_vec_blob(len, v.data + start, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_concat(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va, vb;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob (arg 0)");
    if (llm_parse_vec(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &vb) < 0)
        ERR(ctx, "Invalid vector blob (arg 1)");
    int n = va.n + vb.n;
    int out_sz = LLM_VEC_BLOB_SIZE(n);
    void *out = sqlite3_malloc(out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    *(int32_t *)out = n;
    float *data = (float *)((char *)out + 4);
    memcpy(data, va.data, (size_t)va.n * sizeof(float));
    memcpy(data + va.n, vb.data, (size_t)vb.n * sizeof(float));
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_mrow(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 m;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &m) < 0)
        ERR(ctx, "Invalid matrix blob");
    int r = sqlite3_value_int(argv[1]);
    if (r < 0 || r >= m.rows)
        ERR(ctx, "Row index out of range");
    int out_sz;
    void *out = alloc_vec_blob(m.cols, m.data + (size_t)r * m.cols, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_gather(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 m;
    VecF32 idx;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &m) < 0)
        ERR(ctx, "Invalid matrix blob");
    if (llm_parse_vec(sqlite3_value_blob(argv[1]),
                      sqlite3_value_bytes(argv[1]),
                      &idx) < 0)
        ERR(ctx, "Invalid index vector blob");
    int out_rows = idx.n, out_cols = m.cols;
    float *tmp =
        sqlite3_malloc((int)((size_t)out_rows * out_cols * sizeof(float)));
    if (!tmp) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    for (int i = 0; i < out_rows; i++) {
        int r = (int)roundf(idx.data[i]);
        if (r < 0 || r >= m.rows) {
            sqlite3_free(tmp);
            ERR(ctx, "Gather index out of range");
        }
        memcpy(tmp + (size_t)i * out_cols,
               m.data + (size_t)r * out_cols,
               (size_t)out_cols * sizeof(float));
    }
    int out_sz;
    void *out = alloc_mat_blob(out_rows, out_cols, tmp, &out_sz);
    sqlite3_free(tmp);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}