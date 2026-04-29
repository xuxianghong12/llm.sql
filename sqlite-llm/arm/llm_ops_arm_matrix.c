/*
 * llm_ops_arm_matrix.c - Legacy matrix algebra wrappers
 *
 * This file owns 2-D matrix operations for the MatF32 API: GEMM, matrix add,
 * transpose, and matrix-vector products. Add code here when the operator is
 * fundamentally matrix-shaped and does not need N-D broadcasting semantics.
 *
 * Maintenance guidance:
 *   - Keep SIMD and BLAS variants close together so the SQL wrappers expose a
 *     stable contract regardless of backend choice.
 *   - Reuse alloc_mat_blob or alloc_vec_blob for outputs instead of manually
 *     laying out legacy matrix/vector blobs.
 *   - Batch or tensorized matrix work belongs in the N-D algebra file.
 */
#include "llm_ops_arm_internal.h"
SQLITE_EXTENSION_INIT3

static void
k_gemm_simd(const float *A, const float *B, float *C, int M, int K, int N) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[(size_t)i * K + k];
            for (int j = 0; j < N; j++)
                C[(size_t)i * N + j] += a_ik * B[(size_t)k * N + j];
        }
    }
}

static void k_transpose(const float *A, float *B, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            B[(size_t)j * rows + i] = A[(size_t)i * cols + j];
}

static void
k_matvec_simd(const float *A, const float *x, float *y, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        y[i] = k_dot_simd(A + (size_t)i * cols, x, cols);
}

static void
k_gemm_blas(const float *A, const float *B, float *C, int M, int K, int N) {
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                M,
                N,
                K,
                1.0f,
                A,
                K,
                B,
                N,
                0.0f,
                C,
                N);
}

static void
k_matvec_blas(const float *A, const float *x, float *y, int rows, int cols) {
    cblas_sgemv(CblasRowMajor,
                CblasNoTrans,
                rows,
                cols,
                1.0f,
                A,
                cols,
                x,
                1,
                0.0f,
                y,
                1);
}

void sql_gemm_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2MAT(ctx, argv, ma, mb);
    if (ma.cols != mb.rows)
        ERR(ctx, "Matrix inner dimension mismatch");
    int M = ma.rows, K = ma.cols, N = mb.cols;
    size_t nel = (size_t)M * N;
    float *tmp = (float *)sqlite3_malloc64(nel * sizeof(float));
    if (!tmp) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    k_gemm_simd(ma.data, mb.data, tmp, M, K, N);
    int out_sz;
    void *out = alloc_mat_blob(M, N, tmp, &out_sz);
    sqlite3_free(tmp);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_gemm_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2MAT(ctx, argv, ma, mb);
    if (ma.cols != mb.rows)
        ERR(ctx, "Matrix inner dimension mismatch");
    int M = ma.rows, K = ma.cols, N = mb.cols;
    size_t nel = (size_t)M * N;
    float *tmp = (float *)sqlite3_malloc64(nel * sizeof(float));
    if (!tmp) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    k_gemm_blas(ma.data, mb.data, tmp, M, K, N);
    int out_sz;
    void *out = alloc_mat_blob(M, N, tmp, &out_sz);
    sqlite3_free(tmp);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_matadd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2MAT(ctx, argv, ma, mb);
    if (ma.rows != mb.rows || ma.cols != mb.cols)
        ERR(ctx, "Matrix shape mismatch");
    int nel = ma.rows * ma.cols;
    float *tmp = sqlite3_malloc((int)(nel * sizeof(float)));
    if (!tmp) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    k_add_simd(ma.data, mb.data, tmp, nel);
    int out_sz;
    void *out = alloc_mat_blob(ma.rows, ma.cols, tmp, &out_sz);
    sqlite3_free(tmp);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_transpose(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 ma;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ma) < 0)
        ERR(ctx, "Invalid matrix blob");
    int nel = ma.rows * ma.cols;
    float *tmp = sqlite3_malloc((int)(nel * sizeof(float)));
    if (!tmp) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    k_transpose(ma.data, tmp, ma.rows, ma.cols);
    int out_sz;
    void *out = alloc_mat_blob(ma.cols, ma.rows, tmp, &out_sz);
    sqlite3_free(tmp);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_matvec_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 ma;
    VecF32 vx;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ma) < 0)
        ERR(ctx, "Invalid matrix blob");
    if (llm_parse_vec(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &vx) < 0)
        ERR(ctx, "Invalid vector blob");
    if (ma.cols != vx.n)
        ERR(ctx, "Matrix column count != vector size");
    float *tmp = sqlite3_malloc((int)(ma.rows * sizeof(float)));
    if (!tmp) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    k_matvec_simd(ma.data, vx.data, tmp, ma.rows, ma.cols);
    int out_sz;
    void *out = alloc_vec_blob(ma.rows, tmp, &out_sz);
    sqlite3_free(tmp);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_matvec_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 ma;
    VecF32 vx;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ma) < 0)
        ERR(ctx, "Invalid matrix blob");
    if (llm_parse_vec(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &vx) < 0)
        ERR(ctx, "Invalid vector blob");
    if (ma.cols != vx.n)
        ERR(ctx, "Matrix column count != vector size");
    float *tmp = sqlite3_malloc((int)(ma.rows * sizeof(float)));
    if (!tmp) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    k_matvec_blas(ma.data, vx.data, tmp, ma.rows, ma.cols);
    int out_sz;
    void *out = alloc_vec_blob(ma.rows, tmp, &out_sz);
    sqlite3_free(tmp);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}