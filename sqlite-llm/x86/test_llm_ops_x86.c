/*
 * @SPDX-License-Identifier: Apache-2.0
 * @Author: xuxianghong12
 * @Date: 2026-04-21 22:27:40
 * @LastEditTime: 2026-04-21 23:39:56
 * @LastEditors: xuxianghong12
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sqlite3.h>

#include "llm_ops.h"

typedef enum {
    ARG_BLOB,
    ARG_INT,
    ARG_DOUBLE,
    ARG_TEXT,
} SqlArgType;

typedef struct {
    SqlArgType type;
    const void *blob;
    int blob_size;
    int64_t int_value;
    double double_value;
    const char *text_value;
} SqlArg;

typedef struct {
    const char *name;
    void (*fn)(sqlite3 *db);
} TestCase;

typedef struct {
    const char *name;
    const char *sql;
    int iterations;
    int argc;
    const SqlArg *args;
} BenchmarkCase;

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define ARRAY_LEN(a) ((int)(sizeof(a) / sizeof((a)[0])))

#define FAIL(...)                                                              \
    do {                                                                       \
        fprintf(stderr, "FAIL: " __VA_ARGS__);                                 \
        fprintf(stderr, "\n");                                                 \
        exit(1);                                                               \
    } while (0)

#define ASSERT_TRUE(cond, ...)                                                 \
    do {                                                                       \
        if (!(cond))                                                           \
            FAIL(__VA_ARGS__);                                                 \
    } while (0)

static double now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        FAIL("clock_gettime failed");
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void bind_args(sqlite3_stmt *stmt, int argc, const SqlArg *args) {
    for (int i = 0; i < argc; i++) {
        int rc = SQLITE_OK;
        switch (args[i].type) {
        case ARG_BLOB:
            rc = sqlite3_bind_blob(
                stmt, i + 1, args[i].blob, args[i].blob_size, SQLITE_STATIC);
            break;
        case ARG_INT:
            rc = sqlite3_bind_int64(stmt, i + 1, args[i].int_value);
            break;
        case ARG_DOUBLE:
            rc = sqlite3_bind_double(stmt, i + 1, args[i].double_value);
            break;
        case ARG_TEXT:
            rc = sqlite3_bind_text(
                stmt, i + 1, args[i].text_value, -1, SQLITE_STATIC);
            break;
        }
        if (rc != SQLITE_OK)
            FAIL("sqlite bind failed: %s", sqlite3_errstr(rc));
    }
}

static sqlite3_stmt *prepare_stmt(sqlite3 *db, const char *sql) {
    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
        FAIL("prepare failed for [%s]: %s", sql, sqlite3_errmsg(db));
    return stmt;
}

static void exec_sql(sqlite3 *db, const char *sql) {
    char *errmsg = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &errmsg);
    if (rc != SQLITE_OK) {
        const char *msg = errmsg ? errmsg : sqlite3_errmsg(db);
        FAIL("exec failed for [%s]: %s", sql, msg);
    }
    sqlite3_free(errmsg);
}

static int
query_int(sqlite3 *db, const char *sql, int argc, const SqlArg *args) {
    sqlite3_stmt *stmt = prepare_stmt(db, sql);
    bind_args(stmt, argc, args);
    int rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW)
        FAIL("query_int failed for [%s]: %s", sql, sqlite3_errmsg(db));
    int value = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    return value;
}

static double
query_double(sqlite3 *db, const char *sql, int argc, const SqlArg *args) {
    sqlite3_stmt *stmt = prepare_stmt(db, sql);
    bind_args(stmt, argc, args);
    int rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW)
        FAIL("query_double failed for [%s]: %s", sql, sqlite3_errmsg(db));
    double value = sqlite3_column_double(stmt, 0);
    sqlite3_finalize(stmt);
    return value;
}

static char *
query_text(sqlite3 *db, const char *sql, int argc, const SqlArg *args) {
    sqlite3_stmt *stmt = prepare_stmt(db, sql);
    bind_args(stmt, argc, args);
    int rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW)
        FAIL("query_text failed for [%s]: %s", sql, sqlite3_errmsg(db));
    const unsigned char *text = sqlite3_column_text(stmt, 0);
    char *copy =
        strdup((const char *)(text ? text : (const unsigned char *)""));
    sqlite3_finalize(stmt);
    if (!copy)
        FAIL("strdup failed");
    return copy;
}

static void *query_blob(
    sqlite3 *db, const char *sql, int argc, const SqlArg *args, int *size_out) {
    sqlite3_stmt *stmt = prepare_stmt(db, sql);
    bind_args(stmt, argc, args);
    int rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW)
        FAIL("query_blob failed for [%s]: %s", sql, sqlite3_errmsg(db));
    int size = sqlite3_column_bytes(stmt, 0);
    const void *blob = sqlite3_column_blob(stmt, 0);
    void *copy = malloc((size_t)size);
    if (!copy)
        FAIL("malloc failed");
    memcpy(copy, blob, (size_t)size);
    sqlite3_finalize(stmt);
    *size_out = size;
    return copy;
}

static void run_single_row_sql(sqlite3 *db, const char *name, const char *sql) {
    sqlite3_stmt *stmt = prepare_stmt(db, sql);
    int rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW)
        FAIL("%s failed for [%s]: %s", name, sql, sqlite3_errmsg(db));
    sqlite3_finalize(stmt);
}

static void
assert_closef(float actual, float expected, float tol, const char *msg) {
    if (fabsf(actual - expected) > tol)
        FAIL("%s: expected %.7f got %.7f", msg, expected, actual);
}

static void assert_int_eq(int actual, int expected, const char *msg) {
    if (actual != expected)
        FAIL("%s: expected %d got %d", msg, expected, actual);
}

static void
assert_text_eq(const char *actual, const char *expected, const char *msg) {
    if (strcmp(actual, expected) != 0)
        FAIL("%s: expected [%s] got [%s]", msg, expected, actual);
}

static void assert_vec_blob_eq(const void *blob,
                               int blob_size,
                               const float *expected,
                               int n,
                               float tol,
                               const char *msg) {
    VecF32 vec;
    if (llm_parse_vec(blob, blob_size, &vec) != 0)
        FAIL("%s: blob is not a vector", msg);
    if (vec.n != n)
        FAIL("%s: expected length %d got %d", msg, n, vec.n);
    for (int i = 0; i < n; i++) {
        char item_msg[128];
        snprintf(item_msg, sizeof(item_msg), "%s[%d]", msg, i);
        assert_closef(vec.data[i], expected[i], tol, item_msg);
    }
}

static void assert_mat_blob_eq(const void *blob,
                               int blob_size,
                               const float *expected,
                               int rows,
                               int cols,
                               float tol,
                               const char *msg) {
    MatF32 mat;
    if (llm_parse_mat(blob, blob_size, &mat) != 0)
        FAIL("%s: blob is not a matrix", msg);
    if (mat.rows != rows || mat.cols != cols)
        FAIL("%s: expected shape (%d,%d) got (%d,%d)",
             msg,
             rows,
             cols,
             mat.rows,
             mat.cols);
    for (int i = 0; i < rows * cols; i++) {
        char item_msg[128];
        snprintf(item_msg, sizeof(item_msg), "%s[%d]", msg, i);
        assert_closef(mat.data[i], expected[i], tol, item_msg);
    }
}

static void
assert_softmax_vec(const void *blob, int blob_size, int n, const char *msg) {
    VecF32 vec;
    if (llm_parse_vec(blob, blob_size, &vec) != 0)
        FAIL("%s: blob is not a vector", msg);
    if (vec.n != n)
        FAIL("%s: expected length %d got %d", msg, n, vec.n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(
            vec.data[i] >= 0.0f, "%s: softmax[%d] must be >= 0", msg, i);
        sum += vec.data[i];
    }
    assert_closef(sum, 1.0f, 1e-5f, "softmax sum");
}

static void *build_vec_blob(const float *data, int n, int *size_out) {
    int size = LLM_VEC_BLOB_SIZE(n);
    char *blob = (char *)malloc((size_t)size);
    if (!blob)
        FAIL("malloc failed");
    *(int32_t *)blob = n;
    memcpy(blob + 4, data, (size_t)n * sizeof(float));
    *size_out = size;
    return blob;
}

static void *
build_mat_blob(const float *data, int rows, int cols, int *size_out) {
    int size = LLM_MAT_BLOB_SIZE(rows, cols);
    char *blob = (char *)malloc((size_t)size);
    if (!blob)
        FAIL("malloc failed");
    ((int32_t *)blob)[0] = rows;
    ((int32_t *)blob)[1] = cols;
    memcpy(blob + 8, data, (size_t)(rows * cols) * sizeof(float));
    *size_out = size;
    return blob;
}

static void *build_nd_blob(const int32_t *shape,
                           int ndim,
                           const float *data,
                           int *size_out) {
    void *blob = llm_build_nd(ndim, shape, data, size_out);
    if (!blob)
        FAIL("llm_build_nd failed");
    return blob;
}

static void assert_nd_blob_eq(const void *blob,
                              int blob_size,
                              const int32_t *expected_shape,
                              int expected_ndim,
                              const float *expected_data,
                              int expected_total,
                              float tol,
                              const char *msg) {
    TensorNDF32 tensor;
    if (llm_to_nd(blob, blob_size, &tensor) != 0)
        FAIL("%s: blob is not an ND tensor", msg);
    if (tensor.ndim != expected_ndim)
        FAIL("%s: expected ndim %d got %d", msg, expected_ndim, tensor.ndim);
    if (tensor.total != expected_total)
        FAIL("%s: expected total %d got %d", msg, expected_total, tensor.total);
    for (int i = 0; i < expected_ndim; i++) {
        if (tensor.shape[i] != expected_shape[i])
            FAIL("%s: shape[%d] expected %d got %d",
                 msg,
                 i,
                 expected_shape[i],
                 tensor.shape[i]);
    }
    for (int i = 0; i < expected_total; i++) {
        char item_msg[128];
        snprintf(item_msg, sizeof(item_msg), "%s[%d]", msg, i);
        assert_closef(tensor.data[i], expected_data[i], tol, item_msg);
    }
}

static sqlite3 *open_test_db_x86(void) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK)
        FAIL("sqlite3_open failed: %s", sqlite3_errmsg(db));

    rc = sqlite3_enable_load_extension(db, 1);
    if (rc != SQLITE_OK)
        FAIL("sqlite3_enable_load_extension failed: %s", sqlite3_errmsg(db));

    char *errmsg = NULL;
    rc = sqlite3_load_extension(
        db, "./llm_ops.so", "sqlite3_llm_ops_init", &errmsg);
    if (rc != SQLITE_OK) {
        const char *msg = errmsg ? errmsg : sqlite3_errmsg(db);
        FAIL("sqlite3_load_extension failed: %s", msg);
    }
    sqlite3_free(errmsg);
    return db;
}

static sqlite3 *open_file_test_db_x86(const char *path) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open(path, &db);
    if (rc != SQLITE_OK)
        FAIL("sqlite3_open(%s) failed: %s", path, sqlite3_errmsg(db));

    rc = sqlite3_enable_load_extension(db, 1);
    if (rc != SQLITE_OK)
        FAIL("sqlite3_enable_load_extension(file db) failed: %s",
             sqlite3_errmsg(db));

    char *errmsg = NULL;
    rc = sqlite3_load_extension(
        db, "./llm_ops.so", "sqlite3_llm_ops_init", &errmsg);
    if (rc != SQLITE_OK) {
        const char *msg = errmsg ? errmsg : sqlite3_errmsg(db);
        FAIL("sqlite3_load_extension(file db) failed: %s", msg);
    }
    sqlite3_free(errmsg);
    return db;
}

static void test_runtime_threads_x86(sqlite3 *db) {
    int initial = query_int(db, "SELECT llm_get_threads()", 0, NULL);
    ASSERT_TRUE(initial >= 1, "llm_get_threads should be >= 1");

    int set_two = query_int(db, "SELECT llm_set_threads(2)", 0, NULL);
    assert_int_eq(set_two, 2, "llm_set_threads(2)");

    int clamped_low = query_int(db, "SELECT llm_set_threads(0)", 0, NULL);
    assert_int_eq(clamped_low, 1, "llm_set_threads low clamp");

    int clamped_high = query_int(db, "SELECT llm_set_threads(999)", 0, NULL);
    assert_int_eq(clamped_high, 256, "llm_set_threads high clamp");

    int reset = query_int(db, "SELECT llm_set_threads(4)", 0, NULL);
    ASSERT_TRUE(reset >= 1, "llm_set_threads reset");
}

static void test_construct_and_inspect_x86(sqlite3 *db) {
    int size = 0;
    void *vec_blob = query_blob(db, "SELECT llm_vec(4, 1.5)", 0, NULL, &size);
    float expected_vec[] = {1.5f, 1.5f, 1.5f, 1.5f};
    assert_vec_blob_eq(vec_blob, size, expected_vec, 4, 1e-6f, "llm_vec");
    free(vec_blob);

    void *mat_blob =
        query_blob(db, "SELECT llm_mat(2, 3, -2.0)", 0, NULL, &size);
    float expected_mat[] = {-2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f};
    assert_mat_blob_eq(mat_blob, size, expected_mat, 2, 3, 1e-6f, "llm_mat");
    free(mat_blob);

    float vec_data[] = {1.0f, 3.0f, 5.0f, 7.0f};
    int vec_size = 0;
    void *manual_vec = build_vec_blob(vec_data, 4, &vec_size);
    SqlArg vec_arg[] = {
        {.type = ARG_BLOB, .blob = manual_vec, .blob_size = vec_size}};

    assert_int_eq(query_int(db, "SELECT llm_len(?)", 1, vec_arg), 4, "llm_len");
    assert_closef((float)query_double(db, "SELECT llm_vget(?, 2)", 1, vec_arg),
                  5.0f,
                  1e-6f,
                  "llm_vget");

    char *vec_str = query_text(db, "SELECT llm_str(?)", 1, vec_arg);
    assert_text_eq(vec_str, "[1, 3, 5, 7]", "llm_str vector");
    free(vec_str);
    free(manual_vec);

    float mat_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int mat_size = 0;
    void *manual_mat = build_mat_blob(mat_data, 2, 3, &mat_size);
    SqlArg mat_arg[] = {
        {.type = ARG_BLOB, .blob = manual_mat, .blob_size = mat_size}};

    assert_int_eq(
        query_int(db, "SELECT llm_rows(?)", 1, mat_arg), 2, "llm_rows");
    assert_int_eq(
        query_int(db, "SELECT llm_cols(?)", 1, mat_arg), 3, "llm_cols");
    assert_closef(
        (float)query_double(db, "SELECT llm_mget(?, 1, 2)", 1, mat_arg),
        6.0f,
        1e-6f,
        "llm_mget");

    char *mat_str = query_text(db, "SELECT llm_str(?)", 1, mat_arg);
    assert_text_eq(mat_str, "[[1, 2, 3], [4, 5, 6]]", "llm_str matrix");
    free(mat_str);
    free(manual_mat);
}

static void test_vector_arithmetic_x86(sqlite3 *db) {
    float a_data[] = {1.0f, -2.0f, 3.0f, -4.0f};
    float b_data[] = {0.5f, 4.0f, -1.0f, 2.0f};
    int a_size = 0;
    int b_size = 0;
    void *a_blob = build_vec_blob(a_data, 4, &a_size);
    void *b_blob = build_vec_blob(b_data, 4, &b_size);

    SqlArg args2[] = {
        {.type = ARG_BLOB, .blob = a_blob, .blob_size = a_size},
        {.type = ARG_BLOB, .blob = b_blob, .blob_size = b_size},
    };

    int out_size = 0;
    void *out =
        query_blob(db, "SELECT llm_add_simd(?, ?)", 2, args2, &out_size);
    float add_expected[] = {1.5f, 2.0f, 2.0f, -2.0f};
    assert_vec_blob_eq(out, out_size, add_expected, 4, 1e-6f, "llm_add_simd");
    free(out);

    out = query_blob(db, "SELECT llm_add_blas(?, ?)", 2, args2, &out_size);
    assert_vec_blob_eq(out, out_size, add_expected, 4, 1e-6f, "llm_add_blas");
    free(out);

    out = query_blob(db, "SELECT llm_sub_simd(?, ?)", 2, args2, &out_size);
    float sub_expected[] = {0.5f, -6.0f, 4.0f, -6.0f};
    assert_vec_blob_eq(out, out_size, sub_expected, 4, 1e-6f, "llm_sub_simd");
    free(out);

    out = query_blob(db, "SELECT llm_mul_simd(?, ?)", 2, args2, &out_size);
    float mul_expected[] = {0.5f, -8.0f, -3.0f, -8.0f};
    assert_vec_blob_eq(out, out_size, mul_expected, 4, 1e-6f, "llm_mul_simd");
    free(out);

    out = query_blob(db, "SELECT llm_mul_blas(?, ?)", 2, args2, &out_size);
    assert_vec_blob_eq(out, out_size, mul_expected, 4, 1e-6f, "llm_mul_blas");
    free(out);

    out = query_blob(db, "SELECT llm_div_simd(?, ?)", 2, args2, &out_size);
    float div_expected[] = {2.0f, -0.5f, -3.0f, -2.0f};
    assert_vec_blob_eq(out, out_size, div_expected, 4, 1e-6f, "llm_div_simd");
    free(out);

    SqlArg scale_args[] = {
        {.type = ARG_BLOB, .blob = a_blob, .blob_size = a_size},
        {.type = ARG_DOUBLE, .double_value = -0.25},
    };
    out =
        query_blob(db, "SELECT llm_scale_simd(?, ?)", 2, scale_args, &out_size);
    float scale_expected[] = {-0.25f, 0.5f, -0.75f, 1.0f};
    assert_vec_blob_eq(
        out, out_size, scale_expected, 4, 1e-6f, "llm_scale_simd");
    free(out);

    out =
        query_blob(db, "SELECT llm_scale_blas(?, ?)", 2, scale_args, &out_size);
    assert_vec_blob_eq(
        out, out_size, scale_expected, 4, 1e-6f, "llm_scale_blas");
    free(out);

    assert_closef(
        (float)query_double(db, "SELECT llm_dot_simd(?, ?)", 2, args2),
        -18.5f,
        1e-5f,
        "llm_dot_simd");
    assert_closef(
        (float)query_double(db, "SELECT llm_dot_blas(?, ?)", 2, args2),
        -18.5f,
        1e-5f,
        "llm_dot_blas");

    free(a_blob);
    free(b_blob);
}

static void test_matrix_core_ops_x86(sqlite3 *db) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float m_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float v_data[] = {1.0f, -1.0f, 2.0f};
    int a_size = 0;
    int b_size = 0;
    int m_size = 0;
    int v_size = 0;
    void *a_blob = build_mat_blob(a_data, 2, 2, &a_size);
    void *b_blob = build_mat_blob(b_data, 2, 2, &b_size);
    void *m_blob = build_mat_blob(m_data, 2, 3, &m_size);
    void *v_blob = build_vec_blob(v_data, 3, &v_size);

    SqlArg mm_args[] = {
        {.type = ARG_BLOB, .blob = a_blob, .blob_size = a_size},
        {.type = ARG_BLOB, .blob = b_blob, .blob_size = b_size},
    };
    int out_size = 0;
    void *out =
        query_blob(db, "SELECT llm_matadd(?, ?)", 2, mm_args, &out_size);
    float matadd_expected[] = {6.0f, 8.0f, 10.0f, 12.0f};
    assert_mat_blob_eq(
        out, out_size, matadd_expected, 2, 2, 1e-6f, "llm_matadd");
    free(out);

    out = query_blob(db, "SELECT llm_transpose(?)", 1, mm_args, &out_size);
    float transpose_expected[] = {1.0f, 3.0f, 2.0f, 4.0f};
    assert_mat_blob_eq(
        out, out_size, transpose_expected, 2, 2, 1e-6f, "llm_transpose");
    free(out);

    out = query_blob(db, "SELECT llm_gemm_simd(?, ?)", 2, mm_args, &out_size);
    float gemm_expected[] = {19.0f, 22.0f, 43.0f, 50.0f};
    assert_mat_blob_eq(
        out, out_size, gemm_expected, 2, 2, 1e-5f, "llm_gemm_simd");
    free(out);

    out = query_blob(db, "SELECT llm_gemm_blas(?, ?)", 2, mm_args, &out_size);
    assert_mat_blob_eq(
        out, out_size, gemm_expected, 2, 2, 1e-5f, "llm_gemm_blas");
    free(out);

    SqlArg mv_args[] = {
        {.type = ARG_BLOB, .blob = m_blob, .blob_size = m_size},
        {.type = ARG_BLOB, .blob = v_blob, .blob_size = v_size},
    };
    out = query_blob(db, "SELECT llm_matvec_simd(?, ?)", 2, mv_args, &out_size);
    float matvec_expected[] = {5.0f, 11.0f};
    assert_vec_blob_eq(
        out, out_size, matvec_expected, 2, 1e-5f, "llm_matvec_simd");
    free(out);

    out = query_blob(db, "SELECT llm_matvec_blas(?, ?)", 2, mv_args, &out_size);
    assert_vec_blob_eq(
        out, out_size, matvec_expected, 2, 1e-5f, "llm_matvec_blas");
    free(out);

    free(a_blob);
    free(b_blob);
    free(m_blob);
    free(v_blob);
}

static void test_unary_activation_reduction_x86(sqlite3 *db) {
    float x_data[] = {-1.0f, 0.0f, 1.0f, 4.0f};
    int x_size = 0;
    void *x_blob = build_vec_blob(x_data, 4, &x_size);
    SqlArg x_arg[] = {{.type = ARG_BLOB, .blob = x_blob, .blob_size = x_size}};

    int out_size = 0;
    void *out = query_blob(db, "SELECT llm_neg(?)", 1, x_arg, &out_size);
    float neg_expected[] = {1.0f, -0.0f, -1.0f, -4.0f};
    assert_vec_blob_eq(out, out_size, neg_expected, 4, 1e-6f, "llm_neg");
    free(out);

    out = query_blob(db, "SELECT llm_abs(?)", 1, x_arg, &out_size);
    float abs_expected[] = {1.0f, 0.0f, 1.0f, 4.0f};
    assert_vec_blob_eq(out, out_size, abs_expected, 4, 1e-6f, "llm_abs");
    free(out);

    out = query_blob(db, "SELECT llm_relu(?)", 1, x_arg, &out_size);
    float relu_expected[] = {0.0f, 0.0f, 1.0f, 4.0f};
    assert_vec_blob_eq(out, out_size, relu_expected, 4, 1e-6f, "llm_relu");
    free(out);

    out = query_blob(db, "SELECT llm_sqrt(?)", 1, x_arg, &out_size);
    VecF32 sqrt_vec;
    if (llm_parse_vec(out, out_size, &sqrt_vec) != 0)
        FAIL("llm_sqrt output is not vec");
    ASSERT_TRUE(isnan(sqrt_vec.data[0]), "llm_sqrt(-1) should be NaN");
    assert_closef(sqrt_vec.data[1], 0.0f, 1e-6f, "llm_sqrt(0)");
    assert_closef(sqrt_vec.data[2], 1.0f, 1e-6f, "llm_sqrt(1)");
    assert_closef(sqrt_vec.data[3], 2.0f, 1e-6f, "llm_sqrt(4)");
    free(out);

    out = query_blob(db, "SELECT llm_clip(?, -0.5, 2.0)", 1, x_arg, &out_size);
    float clip_expected[] = {-0.5f, 0.0f, 1.0f, 2.0f};
    assert_vec_blob_eq(out, out_size, clip_expected, 4, 1e-6f, "llm_clip");
    free(out);

    float pos_data[] = {0.25f, 1.0f, 2.0f, 4.0f};
    int pos_size = 0;
    void *pos_blob = build_vec_blob(pos_data, 4, &pos_size);
    SqlArg pos_arg[] = {
        {.type = ARG_BLOB, .blob = pos_blob, .blob_size = pos_size}};

    out = query_blob(db, "SELECT llm_log(?)", 1, pos_arg, &out_size);
    void *exp_out = query_blob(db, "SELECT llm_exp(?)", 1, pos_arg, &out_size);
    VecF32 log_vec;
    VecF32 exp_vec;
    if (llm_parse_vec(out, out_size, &log_vec) != 0 ||
        llm_parse_vec(exp_out, out_size, &exp_vec) != 0)
        FAIL("llm_log/llm_exp output parse failed");
    assert_closef(log_vec.data[0], logf(0.25f), 1e-6f, "llm_log");
    assert_closef(exp_vec.data[2], expf(2.0f), 1e-5f, "llm_exp");
    free(out);
    free(exp_out);

    out = query_blob(db, "SELECT llm_gelu(?)", 1, x_arg, &out_size);
    VecF32 gelu_vec;
    if (llm_parse_vec(out, out_size, &gelu_vec) != 0)
        FAIL("llm_gelu parse failed");
    assert_closef(gelu_vec.data[1], 0.0f, 1e-6f, "llm_gelu(0)");
    free(out);

    out = query_blob(db, "SELECT llm_silu(?)", 1, x_arg, &out_size);
    VecF32 silu_vec;
    if (llm_parse_vec(out, out_size, &silu_vec) != 0)
        FAIL("llm_silu parse failed");
    assert_closef(silu_vec.data[1], 0.0f, 1e-6f, "llm_silu(0)");
    free(out);

    out = query_blob(db, "SELECT llm_sigmoid(?)", 1, x_arg, &out_size);
    VecF32 sigmoid_vec;
    if (llm_parse_vec(out, out_size, &sigmoid_vec) != 0)
        FAIL("llm_sigmoid parse failed");
    assert_closef(sigmoid_vec.data[1], 0.5f, 1e-6f, "llm_sigmoid(0)");
    free(out);

    out = query_blob(db, "SELECT llm_softmax(?)", 1, x_arg, &out_size);
    assert_softmax_vec(out, out_size, 4, "llm_softmax");
    free(out);

    assert_closef((float)query_double(db, "SELECT llm_sum(?)", 1, x_arg),
                  4.0f,
                  1e-6f,
                  "llm_sum");
    assert_closef((float)query_double(db, "SELECT llm_mean(?)", 1, x_arg),
                  1.0f,
                  1e-6f,
                  "llm_mean");
    assert_closef((float)query_double(db, "SELECT llm_max(?)", 1, x_arg),
                  4.0f,
                  1e-6f,
                  "llm_max");
    assert_closef((float)query_double(db, "SELECT llm_min(?)", 1, x_arg),
                  -1.0f,
                  1e-6f,
                  "llm_min");

    float w_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float b_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    int w_size = 0;
    int b_size = 0;
    void *w_blob = build_vec_blob(w_data, 4, &w_size);
    void *b_blob = build_vec_blob(b_data, 4, &b_size);
    SqlArg rms_args[] = {
        {.type = ARG_BLOB, .blob = x_blob, .blob_size = x_size},
        {.type = ARG_BLOB, .blob = w_blob, .blob_size = w_size},
        {.type = ARG_DOUBLE, .double_value = 1e-5},
    };
    out = query_blob(db, "SELECT llm_rmsnorm(?, ?, ?)", 3, rms_args, &out_size);
    VecF32 rms_vec;
    if (llm_parse_vec(out, out_size, &rms_vec) != 0)
        FAIL("llm_rmsnorm parse failed");
    float rms_scale = 1.0f / sqrtf((1.0f + 0.0f + 1.0f + 16.0f) / 4.0f + 1e-5f);
    assert_closef(rms_vec.data[0], -1.0f * rms_scale, 1e-5f, "llm_rmsnorm[0]");
    free(out);

    SqlArg ln_args[] = {
        {.type = ARG_BLOB, .blob = x_blob, .blob_size = x_size},
        {.type = ARG_BLOB, .blob = w_blob, .blob_size = w_size},
        {.type = ARG_BLOB, .blob = b_blob, .blob_size = b_size},
        {.type = ARG_DOUBLE, .double_value = 1e-5},
    };
    out = query_blob(
        db, "SELECT llm_layernorm(?, ?, ?, ?)", 4, ln_args, &out_size);
    VecF32 ln_vec;
    if (llm_parse_vec(out, out_size, &ln_vec) != 0)
        FAIL("llm_layernorm parse failed");
    float mean = (x_data[0] + x_data[1] + x_data[2] + x_data[3]) / 4.0f;
    float var = 0.0f;
    for (int i = 0; i < 4; i++) {
        float d = x_data[i] - mean;
        var += d * d;
    }
    var /= 4.0f;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    assert_closef(ln_vec.data[3],
                  (x_data[3] - mean) * inv_std,
                  1e-5f,
                  "llm_layernorm[3]");
    free(out);

    free(w_blob);
    free(b_blob);
    free(pos_blob);
    free(x_blob);
}

static void test_shape_index_and_rope_x86(sqlite3 *db) {
    float vec_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
    int vec_size = 0;
    void *vec_blob = build_vec_blob(vec_data, 4, &vec_size);
    SqlArg vec_arg[] = {
        {.type = ARG_BLOB, .blob = vec_blob, .blob_size = vec_size}};

    int out_size = 0;
    void *out =
        query_blob(db, "SELECT llm_slice(?, 1, 3)", 1, vec_arg, &out_size);
    float slice_expected[] = {20.0f, 30.0f};
    assert_vec_blob_eq(out, out_size, slice_expected, 2, 1e-6f, "llm_slice");
    free(out);

    float tail_data[] = {50.0f, 60.0f};
    int tail_size = 0;
    void *tail_blob = build_vec_blob(tail_data, 2, &tail_size);
    SqlArg concat_args[] = {
        {.type = ARG_BLOB, .blob = vec_blob, .blob_size = vec_size},
        {.type = ARG_BLOB, .blob = tail_blob, .blob_size = tail_size},
    };
    out = query_blob(db, "SELECT llm_concat(?, ?)", 2, concat_args, &out_size);
    float concat_expected[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    assert_vec_blob_eq(out, out_size, concat_expected, 6, 1e-6f, "llm_concat");
    free(out);

    out = query_blob(db, "SELECT llm_reshape(?, 2, 2)", 1, vec_arg, &out_size);
    float reshape_expected[] = {10.0f, 20.0f, 30.0f, 40.0f};
    assert_mat_blob_eq(
        out, out_size, reshape_expected, 2, 2, 1e-6f, "llm_reshape");

    SqlArg reshaped_arg[] = {
        {.type = ARG_BLOB, .blob = out, .blob_size = out_size}};
    void *flat =
        query_blob(db, "SELECT llm_flatten(?)", 1, reshaped_arg, &out_size);
    assert_vec_blob_eq(
        flat, out_size, reshape_expected, 4, 1e-6f, "llm_flatten");
    free(out);
    free(flat);

    float mat_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int mat_size = 0;
    void *mat_blob = build_mat_blob(mat_data, 2, 3, &mat_size);
    SqlArg mat_arg[] = {
        {.type = ARG_BLOB, .blob = mat_blob, .blob_size = mat_size}};
    out = query_blob(db, "SELECT llm_mrow(?, 1)", 1, mat_arg, &out_size);
    float row_expected[] = {4.0f, 5.0f, 6.0f};
    assert_vec_blob_eq(out, out_size, row_expected, 3, 1e-6f, "llm_mrow");
    free(out);

    float idx_data[] = {1.0f, 0.0f};
    int idx_size = 0;
    void *idx_blob = build_vec_blob(idx_data, 2, &idx_size);
    SqlArg gather_args[] = {
        {.type = ARG_BLOB, .blob = mat_blob, .blob_size = mat_size},
        {.type = ARG_BLOB, .blob = idx_blob, .blob_size = idx_size},
    };
    out = query_blob(db, "SELECT llm_gather(?, ?)", 2, gather_args, &out_size);
    float gather_expected[] = {4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 3.0f};
    assert_mat_blob_eq(
        out, out_size, gather_expected, 2, 3, 1e-6f, "llm_gather");
    free(out);

    float rope_x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float rope_c[] = {1.0f, 1.0f};
    float rope_s[] = {0.0f, 0.0f};
    int rope_x_sz = 0;
    int rope_c_sz = 0;
    int rope_s_sz = 0;
    void *rope_x_blob = build_vec_blob(rope_x, 4, &rope_x_sz);
    void *rope_c_blob = build_vec_blob(rope_c, 2, &rope_c_sz);
    void *rope_s_blob = build_vec_blob(rope_s, 2, &rope_s_sz);
    SqlArg rope_args[] = {
        {.type = ARG_BLOB, .blob = rope_x_blob, .blob_size = rope_x_sz},
        {.type = ARG_BLOB, .blob = rope_c_blob, .blob_size = rope_c_sz},
        {.type = ARG_BLOB, .blob = rope_s_blob, .blob_size = rope_s_sz},
        {.type = ARG_INT, .int_value = 4},
    };
    out =
        query_blob(db, "SELECT llm_rope(?, ?, ?, ?)", 4, rope_args, &out_size);
    assert_vec_blob_eq(out, out_size, rope_x, 4, 1e-6f, "llm_rope identity");
    free(out);

    free(rope_x_blob);
    free(rope_c_blob);
    free(rope_s_blob);
    free(idx_blob);
    free(mat_blob);
    free(tail_blob);
    free(vec_blob);
}

static void test_nd_and_int8_smoke_x86(sqlite3 *db) {
    static const char *smoke_sqls[][2] = {
        {"llm_ndim_x86", "SELECT llm_ndim(llm_vec(3, 1.0))"},
        {"llm_shape_x86",
         "SELECT llm_shape(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1)"},
        {"llm_view_nd_x86", "SELECT llm_view_nd(llm_vec(6, 1.0), 2, 3)"},
        {"llm_view_t12_nd_x86",
         "SELECT llm_view_t12_nd(llm_vec(8, 1.0), 1, 2, 4)"},
        {"llm_permute_nd_x86",
         "SELECT llm_permute_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1, 0)"},
        {"llm_transpose_nd_x86",
         "SELECT llm_transpose_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 0, 1)"},
        {"llm_unsqueeze_nd_x86", "SELECT llm_unsqueeze_nd(llm_vec(4, 1.0), 0)"},
        {"llm_squeeze_nd_x86",
         "SELECT llm_squeeze_nd(llm_unsqueeze_nd(llm_vec(4, 1.0), 0), 0)"},
        {"llm_cat_nd_x86",
         "SELECT llm_cat_nd(llm_view_nd(llm_vec(4, 1.0), 2, 2), "
         "llm_view_nd(llm_vec(4, 2.0), 2, 2), 1)"},
        {"llm_slice_nd_x86",
         "SELECT llm_slice_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1, 0, 2)"},
        {"llm_select_nd_x86",
         "SELECT llm_select_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1, 1)"},
        {"llm_expand_nd_x86",
         "SELECT llm_expand_nd(llm_view_nd(llm_vec(3, 1.0), 1, 3), 2, 3)"},
        {"llm_bmm_x86",
         "SELECT llm_bmm(llm_view_nd(llm_vec(8, 1.0), 1, 2, 4), "
         "llm_view_nd(llm_vec(12, 1.0), 1, 4, 3))"},
        {"llm_linear_nd_x86",
         "SELECT llm_linear_nd(llm_view_nd(llm_vec(4, 1.0), 1, 4), "
         "llm_view_nd(llm_vec(8, 1.0), 2, 4))"},
        {"llm_add_nd_x86",
         "SELECT llm_add_nd(llm_vec(4, 1.0), llm_vec(4, 2.0))"},
        {"llm_mul_nd_x86",
         "SELECT llm_mul_nd(llm_vec(4, 1.0), llm_vec(4, 2.0))"},
        {"llm_sub_nd_x86",
         "SELECT llm_sub_nd(llm_vec(4, 1.0), llm_vec(4, 2.0))"},
        {"llm_gt_nd_x86", "SELECT llm_gt_nd(llm_vec(4, 2.0), llm_vec(4, 1.0))"},
        {"llm_lt_nd_x86", "SELECT llm_lt_nd(llm_vec(4, 2.0), llm_vec(4, 1.0))"},
        {"llm_ge_nd_x86", "SELECT llm_ge_nd(llm_vec(4, 2.0), llm_vec(4, 1.0))"},
        {"llm_le_nd_x86", "SELECT llm_le_nd(llm_vec(4, 2.0), llm_vec(4, 1.0))"},
        {"llm_eq_nd_x86", "SELECT llm_eq_nd(llm_vec(4, 2.0), llm_vec(4, 1.0))"},
        {"llm_ne_nd_x86", "SELECT llm_ne_nd(llm_vec(4, 2.0), llm_vec(4, 1.0))"},
        {"llm_softmax_nd_x86",
         "SELECT llm_softmax_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1)"},
        {"llm_mean_nd_x86",
         "SELECT llm_mean_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1, 1)"},
        {"llm_sum_nd_x86",
         "SELECT llm_sum_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1, 0)"},
        {"llm_str_nd_x86",
         "SELECT llm_str_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3))"},
        {"llm_div_nd_x86",
         "SELECT llm_div_nd(llm_vec(4, 2.0), llm_vec(4, 1.0))"},
        {"llm_neg_nd_x86", "SELECT llm_neg_nd(llm_vec(4, 2.0))"},
        {"llm_scale_nd_x86", "SELECT llm_scale_nd(llm_vec(4, 2.0), 0.5)"},
        {"llm_rsqrt_nd_x86", "SELECT llm_rsqrt_nd(llm_vec(4, 4.0))"},
        {"llm_silu_nd_x86", "SELECT llm_silu_nd(llm_vec(4, 1.0))"},
        {"llm_silu_mul_nd_x86",
         "SELECT llm_silu_mul_nd(llm_vec(4, 1.0), llm_vec(4, 2.0))"},
        {"llm_rope_nd_x86",
         "SELECT llm_rope_nd(llm_view_nd(llm_vec(4, 1.0), 1, 1, 1, 4), "
         "llm_view_nd(llm_vec(4, 1.0), 1, 1, 1, 4), llm_view_nd(llm_vec(4, "
         "0.0), 1, 1, 1, 4))"},
        {"llm_exp_nd_x86", "SELECT llm_exp_nd(llm_vec(4, 1.0))"},
        {"llm_cos_nd_x86", "SELECT llm_cos_nd(llm_vec(4, 1.0))"},
        {"llm_sin_nd_x86", "SELECT llm_sin_nd(llm_vec(4, 1.0))"},
        {"llm_tanh_nd_x86", "SELECT llm_tanh_nd(llm_vec(4, 1.0))"},
        {"llm_pow_nd_x86", "SELECT llm_pow_nd(llm_vec(4, 2.0), 2.0)"},
        {"llm_abs_nd_x86", "SELECT llm_abs_nd(llm_vec(4, -2.0))"},
        {"llm_sqrt_nd_x86", "SELECT llm_sqrt_nd(llm_vec(4, 4.0))"},
        {"llm_embedding_nd_x86",
         "SELECT llm_embedding_nd(llm_view_nd(llm_vec(12, 1.0), 3, 4), "
         "llm_view_nd(llm_vec(2, 1.0), 2))"},
        {"llm_rmsnorm_nd_x86",
         "SELECT llm_rmsnorm_nd(llm_view_nd(llm_vec(4, 1.0), 1, 4), llm_vec(4, "
         "1.0), 1e-5)"},
        {"llm_triu_nd_x86",
         "SELECT llm_triu_nd(llm_view_nd(llm_vec(9, 1.0), 3, 3), 0)"},
        {"llm_where_nd_x86",
         "SELECT llm_where_nd(llm_vec(4, 1.0), llm_vec(4, 2.0), llm_vec(4, "
         "3.0))"},
        {"llm_masked_fill_nd_x86",
         "SELECT llm_masked_fill_nd(llm_vec(4, 2.0), llm_vec(4, 1.0), -1.0)"},
        {"llm_arange_nd_x86", "SELECT llm_arange_nd(8)"},
        {"llm_full_nd_x86", "SELECT llm_full_nd(1.5, 2, 3)"},
        {"llm_argmax_nd_x86",
         "SELECT llm_argmax_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1)"},
        {"llm_log_nd_x86", "SELECT llm_log_nd(llm_vec(4, 2.0))"},
        {"llm_sigmoid_nd_x86", "SELECT llm_sigmoid_nd(llm_vec(4, 2.0))"},
        {"llm_relu_nd_x86", "SELECT llm_relu_nd(llm_vec(4, -2.0))"},
        {"llm_gelu_nd_x86", "SELECT llm_gelu_nd(llm_vec(4, 2.0))"},
        {"llm_floor_nd_x86", "SELECT llm_floor_nd(llm_vec(4, 2.0))"},
        {"llm_ceil_nd_x86", "SELECT llm_ceil_nd(llm_vec(4, 2.0))"},
        {"llm_round_nd_x86", "SELECT llm_round_nd(llm_vec(4, 2.0))"},
        {"llm_trunc_nd_x86", "SELECT llm_trunc_nd(llm_vec(4, 2.0))"},
        {"llm_sign_nd_x86", "SELECT llm_sign_nd(llm_vec(4, -2.0))"},
        {"llm_clamp_nd_x86", "SELECT llm_clamp_nd(llm_vec(4, 2.0), -1.0, 1.0)"},
        {"llm_tril_nd_x86",
         "SELECT llm_tril_nd(llm_view_nd(llm_vec(9, 1.0), 3, 3), 0)"},
        {"llm_logical_not_nd_x86",
         "SELECT llm_logical_not_nd(llm_vec(4, 0.0))"},
        {"llm_logical_and_nd_x86",
         "SELECT llm_logical_and_nd(llm_vec(4, 1.0), llm_vec(4, 1.0))"},
        {"llm_logical_or_nd_x86",
         "SELECT llm_logical_or_nd(llm_vec(4, 0.0), llm_vec(4, 1.0))"},
        {"llm_bitwise_not_nd_x86",
         "SELECT llm_bitwise_not_nd(llm_vec(4, 0.0))"},
        {"llm_bitwise_and_nd_x86",
         "SELECT llm_bitwise_and_nd(llm_vec(4, 1.0), llm_vec(4, 1.0))"},
        {"llm_bitwise_or_nd_x86",
         "SELECT llm_bitwise_or_nd(llm_vec(4, 0.0), llm_vec(4, 1.0))"},
        {"llm_bitwise_and_scalar_nd_x86",
         "SELECT llm_bitwise_and_scalar_nd(llm_vec(4, 3.0), 1)"},
        {"llm_pow_scalar_nd_x86",
         "SELECT llm_pow_scalar_nd(2.0, llm_vec(4, 3.0))"},
        {"llm_floor_div_nd_x86",
         "SELECT llm_floor_div_nd(llm_vec(4, 5.0), llm_vec(4, 2.0))"},
        {"llm_remainder_nd_x86",
         "SELECT llm_remainder_nd(llm_vec(4, 5.0), 2.0)"},
        {"llm_remainder_tensor_nd_x86",
         "SELECT llm_remainder_tensor_nd(llm_vec(4, 5.0), llm_vec(4, 2.0))"},
        {"llm_reduce_sum_nd_x86", "SELECT llm_reduce_sum_nd(llm_vec(4, 2.0))"},
        {"llm_reduce_mean_nd_x86",
         "SELECT llm_reduce_mean_nd(llm_vec(4, 2.0))"},
        {"llm_reduce_max_nd_x86", "SELECT llm_reduce_max_nd(llm_vec(4, 2.0))"},
        {"llm_reduce_min_nd_x86", "SELECT llm_reduce_min_nd(llm_vec(4, 2.0))"},
        {"llm_reduce_argmax_nd_x86",
         "SELECT llm_reduce_argmax_nd(llm_vec(4, 2.0))"},
        {"llm_item_nd_x86", "SELECT llm_item_nd(llm_vec(4, 2.0), 2)"},
        {"llm_max_dim_nd_x86",
         "SELECT llm_max_dim_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1, 0)"},
        {"llm_argmax_dim_nd_x86",
         "SELECT llm_argmax_dim_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1, 0)"},
        {"llm_layernorm_nd_x86",
         "SELECT llm_layernorm_nd(llm_view_nd(llm_vec(4, 1.0), 1, 4), "
         "llm_vec(4, 1.0), llm_vec(4, 0.0), 1e-5)"},
        {"llm_cumsum_nd_x86",
         "SELECT llm_cumsum_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1)"},
        {"llm_cumprod_nd_x86",
         "SELECT llm_cumprod_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1)"},
        {"llm_gather_nd_x86",
         "SELECT llm_gather_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1, "
         "llm_view_nd(llm_vec(6, 0.0), 2, 3))"},
        {"llm_index_select_nd_x86",
         "SELECT llm_index_select_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 0, "
         "llm_view_nd(llm_vec(2, 1.0), 2))"},
        {"llm_scatter_nd_x86",
         "SELECT llm_scatter_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1, "
         "llm_view_nd(llm_vec(6, 0.0), 2, 3), 7.0)"},
        {"llm_repeat_nd_x86",
         "SELECT llm_repeat_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 2, 1)"},
        {"llm_repeat_kv_nd_x86",
         "SELECT llm_repeat_kv_nd(llm_view_nd(llm_vec(8, 1.0), 1, 1, 2, 4), "
         "2)"},
        {"llm_flip_nd_x86",
         "SELECT llm_flip_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1)"},
        {"llm_roll_nd_x86",
         "SELECT llm_roll_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 1, 1)"},
        {"llm_repeat_interleave_nd_x86",
         "SELECT llm_repeat_interleave_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), "
         "2, 1)"},
        {"llm_norm_nd_x86",
         "SELECT llm_norm_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), 2.0, 1, 0)"},
        {"llm_index_put_nd_x86",
         "SELECT llm_index_put_nd(llm_view_nd(llm_vec(6, 1.0), 2, 3), "
         "llm_view_nd(llm_vec(6, 2.0), 2, 3), 0, NULL)"},
        {"llm_repeat_interleave_tensor_nd_x86",
         "SELECT llm_repeat_interleave_tensor_nd(llm_view_nd(llm_vec(6, 1.0), "
         "2, 3), llm_view_nd(llm_vec(3, 2.0), 3), 1)"},
        {"llm_squeeze_all_nd_x86",
         "SELECT llm_squeeze_all_nd(llm_view_nd(llm_vec(4, 1.0), 1, 1, 4))"},
        {"llm_flatten_nd_x86",
         "SELECT llm_flatten_nd(llm_view_nd(llm_vec(8, 1.0), 2, 2, 2), 1, 2)"},
        {"llm_cat_multi_nd_x86",
         "SELECT llm_cat_multi_nd(1, llm_view_nd(llm_vec(4, 1.0), 2, 2), "
         "llm_view_nd(llm_vec(4, 2.0), 2, 2), llm_view_nd(llm_vec(4, 3.0), 2, "
         "2))"},
        {"llm_stack_nd_x86",
         "SELECT llm_stack_nd(0, llm_view_nd(llm_vec(4, 1.0), 2, 2), "
         "llm_view_nd(llm_vec(4, 2.0), 2, 2))"},
        {"llm_bool_to_additive_mask_nd_x86",
         "SELECT llm_bool_to_additive_mask_nd(llm_view_nd(llm_vec(4, 1.0), 2, "
         "2), 10000.0)"},
        {"llm_numel_nd_x86",
         "SELECT llm_numel_nd(llm_view_nd(llm_vec(8, 1.0), 2, 2, 2))"},
        {"llm_sdpa_nd_x86",
         "SELECT llm_sdpa_nd(llm_view_nd(llm_vec(4, 1.0), 1, 1, 2, 2), "
         "llm_view_nd(llm_vec(4, 1.0), 1, 1, 2, 2), llm_view_nd(llm_vec(4, "
         "1.0), 1, 1, 2, 2), 1.0, 0)"},
        {"llm_quantize_int8_nd_x86",
         "SELECT llm_quantize_int8_nd(llm_view_nd(llm_vec(12, 1.0), 3, 4))"},
        {"llm_dequantize_int8_nd_x86",
         "SELECT "
         "llm_dequantize_int8_nd(llm_quantize_int8_nd(llm_view_nd(llm_vec(12, "
         "1.0), 3, 4)))"},
        {"llm_linear_int8_nd_x86",
         "SELECT llm_linear_int8_nd(llm_view_nd(llm_vec(4, 1.0), 1, 4), "
         "llm_quantize_int8_nd(llm_view_nd(llm_vec(8, 1.0), 2, 4)))"},
        {"llm_linear_int8_bias_nd_x86",
         "SELECT llm_linear_int8_bias_nd(llm_view_nd(llm_vec(4, 1.0), 1, 4), "
         "llm_quantize_int8_nd(llm_view_nd(llm_vec(8, 1.0), 2, 4)), llm_vec(2, "
         "0.1))"},
        {"llm_embedding_int8_nd_x86",
         "SELECT "
         "llm_embedding_int8_nd(llm_quantize_int8_nd(llm_view_nd(llm_vec(12, "
         "1.0), 3, 4)), llm_view_nd(llm_vec(2, 1.0), 2))"},
        {"llm_is_int8_nd_x86",
         "SELECT llm_is_int8_nd(llm_quantize_int8_nd(llm_view_nd(llm_vec(12, "
         "1.0), 3, 4)))"},
    };

    for (int i = 0; i < ARRAY_LEN(smoke_sqls); i++)
        run_single_row_sql(db, smoke_sqls[i][0], smoke_sqls[i][1]);
}

static void test_nd_broadcast_reduce_x86(sqlite3 *db) {
    int32_t a_shape[2] = {2, 2};
    int32_t b_shape[2] = {1, 2};
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {10.0f, 20.0f};
    int a_size = 0;
    int b_size = 0;
    void *a_blob = build_nd_blob(a_shape, 2, a_data, &a_size);
    void *b_blob = build_nd_blob(b_shape, 2, b_data, &b_size);
    SqlArg args2[] = {
        {.type = ARG_BLOB, .blob = a_blob, .blob_size = a_size},
        {.type = ARG_BLOB, .blob = b_blob, .blob_size = b_size},
    };

    int out_size = 0;
    void *out = query_blob(db, "SELECT llm_add_nd(?, ?)", 2, args2, &out_size);
    float add_expected[] = {11.0f, 22.0f, 13.0f, 24.0f};
    assert_nd_blob_eq(
        out, out_size, a_shape, 2, add_expected, 4, 1e-6f, "llm_add_nd");
    free(out);

    out = query_blob(db, "SELECT llm_mul_nd(?, ?)", 2, args2, &out_size);
    float mul_expected[] = {10.0f, 40.0f, 30.0f, 80.0f};
    assert_nd_blob_eq(
        out, out_size, a_shape, 2, mul_expected, 4, 1e-6f, "llm_mul_nd");
    free(out);

    out = query_blob(db, "SELECT llm_sub_nd(?, ?)", 2, args2, &out_size);
    float sub_expected[] = {-9.0f, -18.0f, -7.0f, -16.0f};
    assert_nd_blob_eq(
        out, out_size, a_shape, 2, sub_expected, 4, 1e-6f, "llm_sub_nd");
    free(out);

    float d_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int d_size = 0;
    void *d_blob = build_nd_blob(a_shape, 2, d_data, &d_size);
    SqlArg div_args[] = {
        {.type = ARG_BLOB, .blob = a_blob, .blob_size = a_size},
        {.type = ARG_BLOB, .blob = d_blob, .blob_size = d_size},
    };
    out = query_blob(db, "SELECT llm_div_nd(?, ?)", 2, div_args, &out_size);
    float div_expected[] = {1.0f, 1.0f, 1.0f, 1.0f};
    assert_nd_blob_eq(
        out, out_size, a_shape, 2, div_expected, 4, 1e-6f, "llm_div_nd");
    free(out);

    float softmax_data[] = {0.0f, 1.0f, 1.0f, 0.0f};
    int softmax_size = 0;
    void *softmax_blob = build_nd_blob(a_shape, 2, softmax_data, &softmax_size);
    SqlArg softmax_args[] = {
        {.type = ARG_BLOB, .blob = softmax_blob, .blob_size = softmax_size},
        {.type = ARG_INT, .int_value = 1},
    };
    out = query_blob(
        db, "SELECT llm_softmax_nd(?, ?)", 2, softmax_args, &out_size);
    float softmax_expected[] = {
        0.26894143f, 0.73105860f, 0.73105860f, 0.26894143f};
    assert_nd_blob_eq(out,
                      out_size,
                      a_shape,
                      2,
                      softmax_expected,
                      4,
                      1e-5f,
                      "llm_softmax_nd");
    free(out);

    SqlArg mean_args[] = {
        {.type = ARG_BLOB, .blob = a_blob, .blob_size = a_size},
        {.type = ARG_INT, .int_value = 1},
        {.type = ARG_INT, .int_value = 0},
    };
    int32_t mean_shape[1] = {2};
    out =
        query_blob(db, "SELECT llm_mean_nd(?, ?, ?)", 3, mean_args, &out_size);
    float mean_expected[] = {1.5f, 3.5f};
    assert_nd_blob_eq(
        out, out_size, mean_shape, 1, mean_expected, 2, 1e-6f, "llm_mean_nd");
    free(out);

    SqlArg sum_args[] = {
        {.type = ARG_BLOB, .blob = a_blob, .blob_size = a_size},
        {.type = ARG_INT, .int_value = 0},
        {.type = ARG_INT, .int_value = 1},
    };
    int32_t sum_shape[2] = {1, 2};
    out = query_blob(db, "SELECT llm_sum_nd(?, ?, ?)", 3, sum_args, &out_size);
    float sum_expected[] = {4.0f, 6.0f};
    assert_nd_blob_eq(
        out, out_size, sum_shape, 2, sum_expected, 2, 1e-6f, "llm_sum_nd");
    free(out);

    assert_int_eq(query_int(db, "SELECT llm_reduce_argmax_nd(?)", 1, args2),
                  3,
                  "llm_reduce_argmax_nd");
    assert_closef((float)query_double(db, "SELECT llm_item_nd(?, 3)", 1, args2),
                  4.0f,
                  1e-6f,
                  "llm_item_nd");
    assert_int_eq(
        query_int(db, "SELECT llm_numel_nd(?)", 1, args2), 4, "llm_numel_nd");

    out = query_blob(db, "SELECT llm_max_dim_nd(?, 1, 0)", 1, args2, &out_size);
    float max_expected[] = {2.0f, 4.0f};
    assert_nd_blob_eq(
        out, out_size, mean_shape, 1, max_expected, 2, 1e-6f, "llm_max_dim_nd");
    free(out);

    out = query_blob(
        db, "SELECT llm_argmax_dim_nd(?, 1, 0)", 1, args2, &out_size);
    float argmax_expected[] = {1.0f, 1.0f};
    assert_nd_blob_eq(out,
                      out_size,
                      mean_shape,
                      1,
                      argmax_expected,
                      2,
                      1e-6f,
                      "llm_argmax_dim_nd");
    free(out);

    free(softmax_blob);
    free(d_blob);
    free(b_blob);
    free(a_blob);
}

static void test_nd_transform_index_x86(sqlite3 *db) {
    int32_t base_shape[2] = {2, 3};
    float base_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int base_size = 0;
    void *base_blob = build_nd_blob(base_shape, 2, base_data, &base_size);
    SqlArg base_arg[] = {
        {.type = ARG_BLOB, .blob = base_blob, .blob_size = base_size}};
    int out_size = 0;

    char *shape_text = query_text(
        db,
        "SELECT llm_shape(?, 0) || ',' || llm_shape(?, 1)",
        2,
        (SqlArg[]){
            {.type = ARG_BLOB, .blob = base_blob, .blob_size = base_size},
            {.type = ARG_BLOB, .blob = base_blob, .blob_size = base_size}});
    assert_text_eq(shape_text, "2,3", "llm_shape");
    free(shape_text);
    assert_int_eq(
        query_int(db, "SELECT llm_ndim(?)", 1, base_arg), 2, "llm_ndim");

    float vt_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int32_t vt_shape[1] = {8};
    int vt_size = 0;
    void *vt_blob = build_nd_blob(vt_shape, 1, vt_data, &vt_size);
    SqlArg vt_args[] = {
        {.type = ARG_BLOB, .blob = vt_blob, .blob_size = vt_size},
        {.type = ARG_INT, .int_value = 1},
        {.type = ARG_INT, .int_value = 2},
        {.type = ARG_INT, .int_value = 4},
    };
    void *out = query_blob(
        db, "SELECT llm_view_t12_nd(?, ?, ?, ?)", 4, vt_args, &out_size);
    int32_t vt_expected_shape[3] = {1, 4, 2};
    float vt_expected[] = {1, 5, 2, 6, 3, 7, 4, 8};
    assert_nd_blob_eq(out,
                      out_size,
                      vt_expected_shape,
                      3,
                      vt_expected,
                      8,
                      1e-6f,
                      "llm_view_t12_nd");
    free(out);

    out = query_blob(
        db, "SELECT llm_permute_nd(?, 1, 0)", 1, base_arg, &out_size);
    int32_t perm_shape[2] = {3, 2};
    float perm_expected[] = {1, 4, 2, 5, 3, 6};
    assert_nd_blob_eq(out,
                      out_size,
                      perm_shape,
                      2,
                      perm_expected,
                      6,
                      1e-6f,
                      "llm_permute_nd");
    free(out);

    out = query_blob(
        db, "SELECT llm_transpose_nd(?, 0, 1)", 1, base_arg, &out_size);
    assert_nd_blob_eq(out,
                      out_size,
                      perm_shape,
                      2,
                      perm_expected,
                      6,
                      1e-6f,
                      "llm_transpose_nd");
    free(out);

    out =
        query_blob(db, "SELECT llm_unsqueeze_nd(?, 0)", 1, base_arg, &out_size);
    int32_t unsq_shape[3] = {1, 2, 3};
    assert_nd_blob_eq(
        out, out_size, unsq_shape, 3, base_data, 6, 1e-6f, "llm_unsqueeze_nd");
    SqlArg unsq_arg[] = {
        {.type = ARG_BLOB, .blob = out, .blob_size = out_size}};
    void *squeezed =
        query_blob(db, "SELECT llm_squeeze_nd(?, 0)", 1, unsq_arg, &out_size);
    assert_nd_blob_eq(squeezed,
                      out_size,
                      base_shape,
                      2,
                      base_data,
                      6,
                      1e-6f,
                      "llm_squeeze_nd");
    free(out);
    free(squeezed);

    float right_data[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    int right_size = 0;
    void *right_blob = build_nd_blob(base_shape, 2, right_data, &right_size);
    SqlArg cat_args[] = {
        {.type = ARG_BLOB, .blob = base_blob, .blob_size = base_size},
        {.type = ARG_BLOB, .blob = right_blob, .blob_size = right_size},
        {.type = ARG_INT, .int_value = 1},
    };
    out = query_blob(db, "SELECT llm_cat_nd(?, ?, ?)", 3, cat_args, &out_size);
    int32_t cat_shape[2] = {2, 6};
    float cat_expected[] = {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12};
    assert_nd_blob_eq(
        out, out_size, cat_shape, 2, cat_expected, 12, 1e-6f, "llm_cat_nd");
    free(out);

    SqlArg slice_args[] = {
        {.type = ARG_BLOB, .blob = base_blob, .blob_size = base_size},
        {.type = ARG_INT, .int_value = 1},
        {.type = ARG_INT, .int_value = 1},
        {.type = ARG_INT, .int_value = 3},
    };
    out = query_blob(
        db, "SELECT llm_slice_nd(?, ?, ?, ?)", 4, slice_args, &out_size);
    int32_t slice_shape[2] = {2, 2};
    float slice_expected[] = {2, 3, 5, 6};
    assert_nd_blob_eq(out,
                      out_size,
                      slice_shape,
                      2,
                      slice_expected,
                      4,
                      1e-6f,
                      "llm_slice_nd");
    free(out);

    SqlArg select_args[] = {
        {.type = ARG_BLOB, .blob = base_blob, .blob_size = base_size},
        {.type = ARG_INT, .int_value = 0},
        {.type = ARG_INT, .int_value = 1},
    };
    out = query_blob(
        db, "SELECT llm_select_nd(?, ?, ?)", 3, select_args, &out_size);
    int32_t select_shape[1] = {3};
    float select_expected[] = {4, 5, 6};
    assert_nd_blob_eq(out,
                      out_size,
                      select_shape,
                      1,
                      select_expected,
                      3,
                      1e-6f,
                      "llm_select_nd");
    free(out);

    int32_t expand_src_shape[2] = {1, 3};
    float expand_src_data[] = {7, 8, 9};
    int expand_src_size = 0;
    void *expand_src_blob =
        build_nd_blob(expand_src_shape, 2, expand_src_data, &expand_src_size);
    SqlArg expand_args[] = {
        {.type = ARG_BLOB,
         .blob = expand_src_blob,
         .blob_size = expand_src_size},
        {.type = ARG_INT, .int_value = 2},
        {.type = ARG_INT, .int_value = 3},
    };
    out = query_blob(
        db, "SELECT llm_expand_nd(?, ?, ?)", 3, expand_args, &out_size);
    float expand_expected[] = {7, 8, 9, 7, 8, 9};
    assert_nd_blob_eq(out,
                      out_size,
                      base_shape,
                      2,
                      expand_expected,
                      6,
                      1e-6f,
                      "llm_expand_nd");
    free(out);

    out = query_blob(db, "SELECT llm_cumsum_nd(?, 1)", 1, base_arg, &out_size);
    float cumsum_expected[] = {1, 3, 6, 4, 9, 15};
    assert_nd_blob_eq(out,
                      out_size,
                      base_shape,
                      2,
                      cumsum_expected,
                      6,
                      1e-6f,
                      "llm_cumsum_nd");
    free(out);

    out = query_blob(db, "SELECT llm_cumprod_nd(?, 1)", 1, base_arg, &out_size);
    float cumprod_expected[] = {1, 2, 6, 4, 20, 120};
    assert_nd_blob_eq(out,
                      out_size,
                      base_shape,
                      2,
                      cumprod_expected,
                      6,
                      1e-6f,
                      "llm_cumprod_nd");
    free(out);

    float gather_index_data[] = {2, 1, 0, 0, 0, 2};
    int gather_index_size = 0;
    void *gather_index_blob =
        build_nd_blob(base_shape, 2, gather_index_data, &gather_index_size);
    SqlArg gather_args[] = {
        {.type = ARG_BLOB, .blob = base_blob, .blob_size = base_size},
        {.type = ARG_INT, .int_value = 1},
        {.type = ARG_BLOB,
         .blob = gather_index_blob,
         .blob_size = gather_index_size},
    };
    out = query_blob(
        db, "SELECT llm_gather_nd(?, ?, ?)", 3, gather_args, &out_size);
    float gather_expected[] = {3, 2, 1, 4, 4, 6};
    assert_nd_blob_eq(out,
                      out_size,
                      base_shape,
                      2,
                      gather_expected,
                      6,
                      1e-6f,
                      "llm_gather_nd");
    free(out);

    int32_t index_shape[1] = {2};
    float index_data[] = {1, 0};
    int index_size = 0;
    void *index_blob = build_nd_blob(index_shape, 1, index_data, &index_size);
    SqlArg index_select_args[] = {
        {.type = ARG_BLOB, .blob = base_blob, .blob_size = base_size},
        {.type = ARG_INT, .int_value = 0},
        {.type = ARG_BLOB, .blob = index_blob, .blob_size = index_size},
    };
    out = query_blob(db,
                     "SELECT llm_index_select_nd(?, ?, ?)",
                     3,
                     index_select_args,
                     &out_size);
    float index_select_expected[] = {4, 5, 6, 1, 2, 3};
    assert_nd_blob_eq(out,
                      out_size,
                      base_shape,
                      2,
                      index_select_expected,
                      6,
                      1e-6f,
                      "llm_index_select_nd");
    free(out);

    float zeros_data[] = {0, 0, 0, 0, 0, 0};
    int zeros_size = 0;
    void *zeros_blob = build_nd_blob(base_shape, 2, zeros_data, &zeros_size);
    SqlArg scatter_args[] = {
        {.type = ARG_BLOB, .blob = zeros_blob, .blob_size = zeros_size},
        {.type = ARG_INT, .int_value = 1},
        {.type = ARG_BLOB,
         .blob = gather_index_blob,
         .blob_size = gather_index_size},
        {.type = ARG_DOUBLE, .double_value = 9.0},
    };
    out = query_blob(
        db, "SELECT llm_scatter_nd(?, ?, ?, ?)", 4, scatter_args, &out_size);
    float scatter_expected[] = {9, 9, 9, 9, 0, 9};
    assert_nd_blob_eq(out,
                      out_size,
                      base_shape,
                      2,
                      scatter_expected,
                      6,
                      1e-6f,
                      "llm_scatter_nd");
    free(out);

    int32_t repeat_shape[2] = {2, 2};
    float repeat_data[] = {1, 2, 3, 4};
    int repeat_size = 0;
    void *repeat_blob =
        build_nd_blob(repeat_shape, 2, repeat_data, &repeat_size);
    SqlArg repeat_args[] = {
        {.type = ARG_BLOB, .blob = repeat_blob, .blob_size = repeat_size},
        {.type = ARG_INT, .int_value = 2},
        {.type = ARG_INT, .int_value = 1},
    };
    out = query_blob(
        db, "SELECT llm_repeat_nd(?, ?, ?)", 3, repeat_args, &out_size);
    int32_t repeat_out_shape[2] = {4, 2};
    float repeat_expected[] = {1, 2, 3, 4, 1, 2, 3, 4};
    assert_nd_blob_eq(out,
                      out_size,
                      repeat_out_shape,
                      2,
                      repeat_expected,
                      8,
                      1e-6f,
                      "llm_repeat_nd");
    free(out);

    out = query_blob(db, "SELECT llm_flip_nd(?, 1)", 1, base_arg, &out_size);
    float flip_expected[] = {3, 2, 1, 6, 5, 4};
    assert_nd_blob_eq(
        out, out_size, base_shape, 2, flip_expected, 6, 1e-6f, "llm_flip_nd");
    free(out);

    SqlArg roll_args[] = {
        {.type = ARG_BLOB, .blob = base_blob, .blob_size = base_size},
        {.type = ARG_INT, .int_value = 1},
        {.type = ARG_INT, .int_value = 1},
    };
    out =
        query_blob(db, "SELECT llm_roll_nd(?, ?, ?)", 3, roll_args, &out_size);
    float roll_expected[] = {3, 1, 2, 6, 4, 5};
    assert_nd_blob_eq(
        out, out_size, base_shape, 2, roll_expected, 6, 1e-6f, "llm_roll_nd");
    free(out);

    SqlArg repeat_interleave_args[] = {
        {.type = ARG_BLOB, .blob = repeat_blob, .blob_size = repeat_size},
        {.type = ARG_INT, .int_value = 2},
        {.type = ARG_INT, .int_value = 1},
    };
    out = query_blob(db,
                     "SELECT llm_repeat_interleave_nd(?, ?, ?)",
                     3,
                     repeat_interleave_args,
                     &out_size);
    int32_t repeat_interleave_shape[2] = {2, 4};
    float repeat_interleave_expected[] = {1, 1, 2, 2, 3, 3, 4, 4};
    assert_nd_blob_eq(out,
                      out_size,
                      repeat_interleave_shape,
                      2,
                      repeat_interleave_expected,
                      8,
                      1e-6f,
                      "llm_repeat_interleave_nd");
    free(out);

    int32_t repeats_tensor_shape[1] = {3};
    float repeats_tensor_data[] = {1, 2, 1};
    int repeats_tensor_size = 0;
    void *repeats_tensor_blob = build_nd_blob(
        repeats_tensor_shape, 1, repeats_tensor_data, &repeats_tensor_size);
    SqlArg repeat_tensor_args[] = {
        {.type = ARG_BLOB, .blob = base_blob, .blob_size = base_size},
        {.type = ARG_BLOB,
         .blob = repeats_tensor_blob,
         .blob_size = repeats_tensor_size},
        {.type = ARG_INT, .int_value = 1},
    };
    out = query_blob(db,
                     "SELECT llm_repeat_interleave_tensor_nd(?, ?, ?)",
                     3,
                     repeat_tensor_args,
                     &out_size);
    int32_t repeat_tensor_shape_out[2] = {2, 4};
    float repeat_tensor_expected[] = {1, 2, 2, 3, 4, 5, 5, 6};
    assert_nd_blob_eq(out,
                      out_size,
                      repeat_tensor_shape_out,
                      2,
                      repeat_tensor_expected,
                      8,
                      1e-6f,
                      "llm_repeat_interleave_tensor_nd");
    free(out);

    int32_t sq_shape[3] = {1, 1, 4};
    float sq_data[] = {1, 2, 3, 4};
    int sq_size = 0;
    void *sq_blob = build_nd_blob(sq_shape, 3, sq_data, &sq_size);
    SqlArg sq_arg[] = {
        {.type = ARG_BLOB, .blob = sq_blob, .blob_size = sq_size}};
    out = query_blob(db, "SELECT llm_squeeze_all_nd(?)", 1, sq_arg, &out_size);
    int32_t sq_out_shape[1] = {4};
    assert_nd_blob_eq(out,
                      out_size,
                      sq_out_shape,
                      1,
                      sq_data,
                      4,
                      1e-6f,
                      "llm_squeeze_all_nd");
    free(out);

    int32_t flat_shape_src[3] = {2, 2, 2};
    float flat_src_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int flat_src_size = 0;
    void *flat_src_blob =
        build_nd_blob(flat_shape_src, 3, flat_src_data, &flat_src_size);
    SqlArg flat_args[] = {
        {.type = ARG_BLOB, .blob = flat_src_blob, .blob_size = flat_src_size},
        {.type = ARG_INT, .int_value = 1},
        {.type = ARG_INT, .int_value = 2},
    };
    out = query_blob(
        db, "SELECT llm_flatten_nd(?, ?, ?)", 3, flat_args, &out_size);
    int32_t flat_out_shape[2] = {2, 4};
    assert_nd_blob_eq(out,
                      out_size,
                      flat_out_shape,
                      2,
                      flat_src_data,
                      8,
                      1e-6f,
                      "llm_flatten_nd");
    free(out);

    SqlArg cat_multi_args[] = {
        {.type = ARG_INT, .int_value = 1},
        {.type = ARG_BLOB, .blob = base_blob, .blob_size = base_size},
        {.type = ARG_BLOB, .blob = right_blob, .blob_size = right_size},
    };
    out = query_blob(
        db, "SELECT llm_cat_multi_nd(?, ?, ?)", 3, cat_multi_args, &out_size);
    assert_nd_blob_eq(out,
                      out_size,
                      cat_shape,
                      2,
                      cat_expected,
                      12,
                      1e-6f,
                      "llm_cat_multi_nd");
    free(out);

    SqlArg stack_args[] = {
        {.type = ARG_INT, .int_value = 0},
        {.type = ARG_BLOB, .blob = base_blob, .blob_size = base_size},
        {.type = ARG_BLOB, .blob = right_blob, .blob_size = right_size},
    };
    out = query_blob(
        db, "SELECT llm_stack_nd(?, ?, ?)", 3, stack_args, &out_size);
    int32_t stack_shape[3] = {2, 2, 3};
    float stack_expected[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    assert_nd_blob_eq(out,
                      out_size,
                      stack_shape,
                      3,
                      stack_expected,
                      12,
                      1e-6f,
                      "llm_stack_nd");
    free(out);

    free(flat_src_blob);
    free(sq_blob);
    free(repeats_tensor_blob);
    free(repeat_blob);
    free(zeros_blob);
    free(index_blob);
    free(gather_index_blob);
    free(expand_src_blob);
    free(right_blob);
    free(vt_blob);
    free(base_blob);
}

static void test_nd_model_ops_x86(sqlite3 *db) {
    int out_size = 0;
    int32_t base_shape[2] = {2, 3};

    int32_t linear_x_shape[2] = {1, 2};
    int32_t linear_w_shape[2] = {2, 2};
    float linear_x_data[] = {1, 2};
    float linear_w_data[] = {1, 0, 0, 1};
    int linear_x_size = 0;
    int linear_w_size = 0;
    void *linear_x_blob =
        build_nd_blob(linear_x_shape, 2, linear_x_data, &linear_x_size);
    void *linear_w_blob =
        build_nd_blob(linear_w_shape, 2, linear_w_data, &linear_w_size);
    SqlArg linear_args[] = {
        {.type = ARG_BLOB, .blob = linear_x_blob, .blob_size = linear_x_size},
        {.type = ARG_BLOB, .blob = linear_w_blob, .blob_size = linear_w_size},
    };
    void *out =
        query_blob(db, "SELECT llm_linear_nd(?, ?)", 2, linear_args, &out_size);
    float linear_expected[] = {1, 2};
    assert_nd_blob_eq(out,
                      out_size,
                      linear_x_shape,
                      2,
                      linear_expected,
                      2,
                      1e-6f,
                      "llm_linear_nd");
    free(out);

    int32_t emb_shape[2] = {3, 2};
    int32_t idx_shape[1] = {2};
    float emb_data[] = {0, 1, 2, 3, 4, 5};
    float idx_data[] = {2, 0};
    int emb_size = 0;
    int idx_size = 0;
    void *emb_blob = build_nd_blob(emb_shape, 2, emb_data, &emb_size);
    void *idx_blob = build_nd_blob(idx_shape, 1, idx_data, &idx_size);
    SqlArg emb_args[] = {
        {.type = ARG_BLOB, .blob = emb_blob, .blob_size = emb_size},
        {.type = ARG_BLOB, .blob = idx_blob, .blob_size = idx_size},
    };
    out =
        query_blob(db, "SELECT llm_embedding_nd(?, ?)", 2, emb_args, &out_size);
    int32_t emb_out_shape[2] = {2, 2};
    float emb_expected[] = {4, 5, 0, 1};
    assert_nd_blob_eq(out,
                      out_size,
                      emb_out_shape,
                      2,
                      emb_expected,
                      4,
                      1e-6f,
                      "llm_embedding_nd");
    free(out);

    int32_t norm_shape[2] = {1, 4};
    int32_t weight_shape[1] = {4};
    float norm_x_data[] = {1, 2, 3, 4};
    float weight_data[] = {1, 1, 1, 1};
    float bias_data[] = {0, 0, 0, 0};
    int norm_x_size = 0;
    int weight_size = 0;
    int bias_size = 0;
    void *norm_x_blob = build_nd_blob(norm_shape, 2, norm_x_data, &norm_x_size);
    void *weight_blob =
        build_nd_blob(weight_shape, 1, weight_data, &weight_size);
    void *bias_blob = build_nd_blob(weight_shape, 1, bias_data, &bias_size);

    SqlArg rms_args[] = {
        {.type = ARG_BLOB, .blob = norm_x_blob, .blob_size = norm_x_size},
        {.type = ARG_BLOB, .blob = weight_blob, .blob_size = weight_size},
        {.type = ARG_DOUBLE, .double_value = 1e-5},
    };
    out = query_blob(
        db, "SELECT llm_rmsnorm_nd(?, ?, ?)", 3, rms_args, &out_size);
    float rms_scale = 1.0f / sqrtf((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + 1e-5f);
    float rms_expected[] = {
        1 * rms_scale, 2 * rms_scale, 3 * rms_scale, 4 * rms_scale};
    assert_nd_blob_eq(
        out, out_size, norm_shape, 2, rms_expected, 4, 1e-5f, "llm_rmsnorm_nd");
    free(out);

    SqlArg layernorm_args[] = {
        {.type = ARG_BLOB, .blob = norm_x_blob, .blob_size = norm_x_size},
        {.type = ARG_BLOB, .blob = weight_blob, .blob_size = weight_size},
        {.type = ARG_BLOB, .blob = bias_blob, .blob_size = bias_size},
        {.type = ARG_DOUBLE, .double_value = 1e-5},
    };
    out = query_blob(db,
                     "SELECT llm_layernorm_nd(?, ?, ?, ?)",
                     4,
                     layernorm_args,
                     &out_size);
    float mean = 2.5f;
    float variance = 1.25f;
    float inv_std = 1.0f / sqrtf(variance + 1e-5f);
    float layernorm_expected[] = {
        (1 - mean) * inv_std,
        (2 - mean) * inv_std,
        (3 - mean) * inv_std,
        (4 - mean) * inv_std,
    };
    assert_nd_blob_eq(out,
                      out_size,
                      norm_shape,
                      2,
                      layernorm_expected,
                      4,
                      1e-5f,
                      "llm_layernorm_nd");
    free(out);

    int32_t mask_shape[2] = {2, 2};
    float cond_data[] = {1, 0, 0, 1};
    float where_x_data[] = {1, 2, 3, 4};
    float where_y_data[] = {5, 6, 7, 8};
    int cond_size = 0;
    int where_x_size = 0;
    int where_y_size = 0;
    void *cond_blob = build_nd_blob(mask_shape, 2, cond_data, &cond_size);
    void *where_x_blob =
        build_nd_blob(mask_shape, 2, where_x_data, &where_x_size);
    void *where_y_blob =
        build_nd_blob(mask_shape, 2, where_y_data, &where_y_size);
    SqlArg where_args[] = {
        {.type = ARG_BLOB, .blob = cond_blob, .blob_size = cond_size},
        {.type = ARG_BLOB, .blob = where_x_blob, .blob_size = where_x_size},
        {.type = ARG_BLOB, .blob = where_y_blob, .blob_size = where_y_size},
    };
    out = query_blob(
        db, "SELECT llm_where_nd(?, ?, ?)", 3, where_args, &out_size);
    float where_expected[] = {1, 6, 7, 4};
    assert_nd_blob_eq(
        out, out_size, mask_shape, 2, where_expected, 4, 1e-6f, "llm_where_nd");
    free(out);

    SqlArg masked_fill_args[] = {
        {.type = ARG_BLOB, .blob = where_x_blob, .blob_size = where_x_size},
        {.type = ARG_BLOB, .blob = cond_blob, .blob_size = cond_size},
        {.type = ARG_DOUBLE, .double_value = -9.0},
    };
    out = query_blob(db,
                     "SELECT llm_masked_fill_nd(?, ?, ?)",
                     3,
                     masked_fill_args,
                     &out_size);
    float masked_fill_expected[] = {-9, 2, 3, -9};
    assert_nd_blob_eq(out,
                      out_size,
                      mask_shape,
                      2,
                      masked_fill_expected,
                      4,
                      1e-6f,
                      "llm_masked_fill_nd");
    free(out);

    SqlArg norm_args[] = {
        {.type = ARG_BLOB, .blob = where_x_blob, .blob_size = where_x_size},
        {.type = ARG_DOUBLE, .double_value = 2.0},
        {.type = ARG_INT, .int_value = 1},
        {.type = ARG_INT, .int_value = 0},
    };
    int32_t norm_out_shape[1] = {2};
    out = query_blob(
        db, "SELECT llm_norm_nd(?, ?, ?, ?)", 4, norm_args, &out_size);
    float norm_expected[] = {sqrtf(5.0f), 5.0f};
    assert_nd_blob_eq(out,
                      out_size,
                      norm_out_shape,
                      1,
                      norm_expected,
                      2,
                      1e-5f,
                      "llm_norm_nd");
    free(out);

    SqlArg bool_mask_args[] = {
        {.type = ARG_BLOB, .blob = cond_blob, .blob_size = cond_size},
        {.type = ARG_DOUBLE, .double_value = 100.0},
    };
    out = query_blob(db,
                     "SELECT llm_bool_to_additive_mask_nd(?, ?)",
                     2,
                     bool_mask_args,
                     &out_size);
    float additive_expected[] = {0, -100, -100, 0};
    assert_nd_blob_eq(out,
                      out_size,
                      mask_shape,
                      2,
                      additive_expected,
                      4,
                      1e-6f,
                      "llm_bool_to_additive_mask_nd");
    free(out);

    int32_t kv_shape[4] = {1, 1, 2, 2};
    float kv_data[] = {1, 2, 3, 4};
    int kv_size = 0;
    void *kv_blob = build_nd_blob(kv_shape, 4, kv_data, &kv_size);
    SqlArg repeat_kv_args[] = {
        {.type = ARG_BLOB, .blob = kv_blob, .blob_size = kv_size},
        {.type = ARG_INT, .int_value = 2},
    };
    out = query_blob(
        db, "SELECT llm_repeat_kv_nd(?, ?)", 2, repeat_kv_args, &out_size);
    int32_t repeat_kv_shape[4] = {1, 2, 2, 2};
    float repeat_kv_expected[] = {1, 2, 3, 4, 1, 2, 3, 4};
    assert_nd_blob_eq(out,
                      out_size,
                      repeat_kv_shape,
                      4,
                      repeat_kv_expected,
                      8,
                      1e-6f,
                      "llm_repeat_kv_nd");
    free(out);

    float index_put_x_data[] = {0, 0, 0, 0, 0, 0};
    float index_put_values_data[] = {10, 11, 12, 20, 21, 22};
    float index_put_idx_data[] = {1, 0};
    int index_put_x_size = 0;
    int index_put_values_size = 0;
    int index_put_idx_size = 0;
    void *index_put_x_blob =
        build_nd_blob(base_shape, 2, index_put_x_data, &index_put_x_size);
    void *index_put_values_blob = build_nd_blob(
        base_shape, 2, index_put_values_data, &index_put_values_size);
    void *index_put_idx_blob =
        build_nd_blob(idx_shape, 1, index_put_idx_data, &index_put_idx_size);
    SqlArg index_put_args[] = {
        {.type = ARG_BLOB,
         .blob = index_put_x_blob,
         .blob_size = index_put_x_size},
        {.type = ARG_BLOB,
         .blob = index_put_values_blob,
         .blob_size = index_put_values_size},
        {.type = ARG_INT, .int_value = 0},
        {.type = ARG_BLOB,
         .blob = index_put_idx_blob,
         .blob_size = index_put_idx_size},
    };
    out = query_blob(db,
                     "SELECT llm_index_put_nd(?, ?, ?, ?)",
                     4,
                     index_put_args,
                     &out_size);
    float index_put_expected[] = {20, 21, 22, 10, 11, 12};
    assert_nd_blob_eq(out,
                      out_size,
                      base_shape,
                      2,
                      index_put_expected,
                      6,
                      1e-6f,
                      "llm_index_put_nd");
    free(out);

    float qk_zero_data[] = {0, 0, 0, 0};
    float v_data[] = {1, 2, 3, 4};
    int q_size = 0;
    int k_size = 0;
    int v_size = 0;
    void *q_blob = build_nd_blob(kv_shape, 4, qk_zero_data, &q_size);
    void *k_blob = build_nd_blob(kv_shape, 4, qk_zero_data, &k_size);
    void *v_blob = build_nd_blob(kv_shape, 4, v_data, &v_size);
    SqlArg sdpa_args[] = {
        {.type = ARG_BLOB, .blob = q_blob, .blob_size = q_size},
        {.type = ARG_BLOB, .blob = k_blob, .blob_size = k_size},
        {.type = ARG_BLOB, .blob = v_blob, .blob_size = v_size},
        {.type = ARG_DOUBLE, .double_value = 1.0},
        {.type = ARG_INT, .int_value = 0},
    };
    out = query_blob(
        db, "SELECT llm_sdpa_nd(?, ?, ?, ?, ?)", 5, sdpa_args, &out_size);
    float sdpa_expected[] = {2, 3, 2, 3};
    assert_nd_blob_eq(
        out, out_size, kv_shape, 4, sdpa_expected, 4, 1e-5f, "llm_sdpa_nd");
    free(out);

    sdpa_args[4].int_value = 1;
    out = query_blob(
        db, "SELECT llm_sdpa_nd(?, ?, ?, ?, ?)", 5, sdpa_args, &out_size);
    float sdpa_causal_expected[] = {1, 2, 2, 3};
    assert_nd_blob_eq(out,
                      out_size,
                      kv_shape,
                      4,
                      sdpa_causal_expected,
                      4,
                      1e-5f,
                      "llm_sdpa_nd causal");
    free(out);

    free(v_blob);
    free(k_blob);
    free(q_blob);
    free(index_put_idx_blob);
    free(index_put_values_blob);
    free(index_put_x_blob);
    free(kv_blob);
    free(where_y_blob);
    free(where_x_blob);
    free(cond_blob);
    free(bias_blob);
    free(weight_blob);
    free(norm_x_blob);
    free(idx_blob);
    free(emb_blob);
    free(linear_w_blob);
    free(linear_x_blob);
}

static void test_int8_ops_x86(sqlite3 *db) {
    int32_t shape[2] = {2, 4};
    float src_data[] = {1.0f, -2.0f, 3.0f, -4.0f, 0.5f, 1.5f, -0.5f, 2.0f};
    int src_size = 0;
    void *src_blob = build_nd_blob(shape, 2, src_data, &src_size);
    SqlArg src_arg[] = {
        {.type = ARG_BLOB, .blob = src_blob, .blob_size = src_size}};

    int quant_size = 0;
    void *quant_blob = query_blob(
        db, "SELECT llm_quantize_int8_nd(?)", 1, src_arg, &quant_size);
    assert_int_eq(query_int(db,
                            "SELECT llm_is_int8_nd(?)",
                            1,
                            (SqlArg[]){{.type = ARG_BLOB,
                                        .blob = quant_blob,
                                        .blob_size = quant_size}}),
                  1,
                  "llm_is_int8_nd");

    SqlArg quant_arg[] = {
        {.type = ARG_BLOB, .blob = quant_blob, .blob_size = quant_size}};
    int out_size = 0;
    void *out = query_blob(
        db, "SELECT llm_dequantize_int8_nd(?)", 1, quant_arg, &out_size);
    assert_nd_blob_eq(
        out, out_size, shape, 2, src_data, 8, 0.05f, "llm_dequantize_int8_nd");
    free(out);

    int32_t x_shape[2] = {1, 4};
    int32_t w_shape[2] = {2, 4};
    float x_data[] = {1, 2, 3, 4};
    float w_data[] = {1, 0, 0, 0, 0, 1, 0, 0};
    float bias_data[] = {0.5f, -1.0f};
    int x_size = 0;
    int w_size = 0;
    int bias_size = 0;
    void *x_blob = build_nd_blob(x_shape, 2, x_data, &x_size);
    void *w_blob = build_nd_blob(w_shape, 2, w_data, &w_size);
    void *bias_blob = build_nd_blob((int32_t[]){2}, 1, bias_data, &bias_size);
    SqlArg w_arg[] = {{.type = ARG_BLOB, .blob = w_blob, .blob_size = w_size}};
    int w_quant_size = 0;
    void *w_quant_blob = query_blob(
        db, "SELECT llm_quantize_int8_nd(?)", 1, w_arg, &w_quant_size);

    SqlArg linear_int8_args[] = {
        {.type = ARG_BLOB, .blob = x_blob, .blob_size = x_size},
        {.type = ARG_BLOB, .blob = w_quant_blob, .blob_size = w_quant_size},
    };
    out = query_blob(
        db, "SELECT llm_linear_int8_nd(?, ?)", 2, linear_int8_args, &out_size);
    float linear_expected[] = {1.0f, 2.0f};
    assert_nd_blob_eq(out,
                      out_size,
                      (int32_t[]){1, 2},
                      2,
                      linear_expected,
                      2,
                      0.05f,
                      "llm_linear_int8_nd");
    free(out);

    SqlArg linear_int8_bias_args[] = {
        {.type = ARG_BLOB, .blob = x_blob, .blob_size = x_size},
        {.type = ARG_BLOB, .blob = w_quant_blob, .blob_size = w_quant_size},
        {.type = ARG_BLOB, .blob = bias_blob, .blob_size = bias_size},
    };
    out = query_blob(db,
                     "SELECT llm_linear_int8_bias_nd(?, ?, ?)",
                     3,
                     linear_int8_bias_args,
                     &out_size);
    float linear_bias_expected[] = {1.5f, 1.0f};
    assert_nd_blob_eq(out,
                      out_size,
                      (int32_t[]){1, 2},
                      2,
                      linear_bias_expected,
                      2,
                      0.05f,
                      "llm_linear_int8_bias_nd");
    free(out);

    int32_t emb_shape[2] = {3, 2};
    float emb_data[] = {0, 1, 2, 3, 4, 5};
    int emb_size = 0;
    void *emb_blob = build_nd_blob(emb_shape, 2, emb_data, &emb_size);
    SqlArg emb_src_arg[] = {
        {.type = ARG_BLOB, .blob = emb_blob, .blob_size = emb_size}};
    int emb_quant_size = 0;
    void *emb_quant_blob = query_blob(
        db, "SELECT llm_quantize_int8_nd(?)", 1, emb_src_arg, &emb_quant_size);
    int idx_size = 0;
    void *idx_blob =
        build_nd_blob((int32_t[]){2}, 1, (float[]){2, 0}, &idx_size);
    SqlArg emb_int8_args[] = {
        {.type = ARG_BLOB, .blob = emb_quant_blob, .blob_size = emb_quant_size},
        {.type = ARG_BLOB, .blob = idx_blob, .blob_size = idx_size},
    };
    out = query_blob(
        db, "SELECT llm_embedding_int8_nd(?, ?)", 2, emb_int8_args, &out_size);
    float emb_expected[] = {4, 5, 0, 1};
    assert_nd_blob_eq(out,
                      out_size,
                      (int32_t[]){2, 2},
                      2,
                      emb_expected,
                      4,
                      0.05f,
                      "llm_embedding_int8_nd");
    free(out);

    free(idx_blob);
    free(emb_quant_blob);
    free(emb_blob);
    free(w_quant_blob);
    free(bias_blob);
    free(w_blob);
    free(x_blob);
    free(quant_blob);
    free(src_blob);
}

static void test_nd_misc_ops_x86(sqlite3 *db) {
    int out_size = 0;
    int32_t vec_shape[1] = {4};
    float a_data[] = {1, 2, 2, 4};
    float b_data[] = {2, 2, 1, 5};
    int a_size = 0;
    int b_size = 0;
    void *a_blob = build_nd_blob(vec_shape, 1, a_data, &a_size);
    void *b_blob = build_nd_blob(vec_shape, 1, b_data, &b_size);
    SqlArg cmp_args[] = {
        {.type = ARG_BLOB, .blob = a_blob, .blob_size = a_size},
        {.type = ARG_BLOB, .blob = b_blob, .blob_size = b_size},
    };

    void *out =
        query_blob(db, "SELECT llm_gt_nd(?, ?)", 2, cmp_args, &out_size);
    float gt_expected[] = {0, 0, 1, 0};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, gt_expected, 4, 1e-6f, "llm_gt_nd");
    free(out);

    out = query_blob(db, "SELECT llm_lt_nd(?, ?)", 2, cmp_args, &out_size);
    float lt_expected[] = {1, 0, 0, 1};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, lt_expected, 4, 1e-6f, "llm_lt_nd");
    free(out);

    out = query_blob(db, "SELECT llm_ge_nd(?, ?)", 2, cmp_args, &out_size);
    float ge_expected[] = {0, 1, 1, 0};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, ge_expected, 4, 1e-6f, "llm_ge_nd");
    free(out);

    out = query_blob(db, "SELECT llm_le_nd(?, ?)", 2, cmp_args, &out_size);
    float le_expected[] = {1, 1, 0, 1};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, le_expected, 4, 1e-6f, "llm_le_nd");
    free(out);

    out = query_blob(db, "SELECT llm_eq_nd(?, ?)", 2, cmp_args, &out_size);
    float eq_expected[] = {0, 1, 0, 0};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, eq_expected, 4, 1e-6f, "llm_eq_nd");
    free(out);

    out = query_blob(db, "SELECT llm_ne_nd(?, ?)", 2, cmp_args, &out_size);
    float ne_expected[] = {1, 0, 1, 1};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, ne_expected, 4, 1e-6f, "llm_ne_nd");
    free(out);

    float unary_data[] = {-1.5f, -0.5f, 0.5f, 1.5f};
    int unary_size = 0;
    void *unary_blob = build_nd_blob(vec_shape, 1, unary_data, &unary_size);
    SqlArg unary_arg[] = {
        {.type = ARG_BLOB, .blob = unary_blob, .blob_size = unary_size}};

    out = query_blob(db, "SELECT llm_neg_nd(?)", 1, unary_arg, &out_size);
    float neg_expected[] = {1.5f, 0.5f, -0.5f, -1.5f};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, neg_expected, 4, 1e-6f, "llm_neg_nd");
    free(out);

    out =
        query_blob(db, "SELECT llm_scale_nd(?, 2.0)", 1, unary_arg, &out_size);
    float scale_expected[] = {-3.0f, -1.0f, 1.0f, 3.0f};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, scale_expected, 4, 1e-6f, "llm_scale_nd");
    free(out);

    int32_t pos_shape[1] = {2};
    float pos_data[] = {1.0f, 4.0f};
    int pos_size = 0;
    void *pos_blob = build_nd_blob(pos_shape, 1, pos_data, &pos_size);
    SqlArg pos_arg[] = {
        {.type = ARG_BLOB, .blob = pos_blob, .blob_size = pos_size}};
    out = query_blob(db, "SELECT llm_rsqrt_nd(?)", 1, pos_arg, &out_size);
    float rsqrt_expected[] = {1.0f, 0.5f};
    assert_nd_blob_eq(
        out, out_size, pos_shape, 1, rsqrt_expected, 2, 1e-6f, "llm_rsqrt_nd");
    free(out);

    out = query_blob(db, "SELECT llm_silu_nd(?)", 1, unary_arg, &out_size);
    TensorNDF32 silu_out;
    if (llm_to_nd(out, out_size, &silu_out) != 0)
        FAIL("llm_silu_nd parse failed");
    assert_closef(silu_out.data[2], 0.31122968f, 1e-5f, "llm_silu_nd[2]");
    free(out);

    SqlArg silu_mul_args[] = {
        {.type = ARG_BLOB, .blob = unary_blob, .blob_size = unary_size},
        {.type = ARG_BLOB, .blob = a_blob, .blob_size = a_size},
    };
    out = query_blob(
        db, "SELECT llm_silu_mul_nd(?, ?)", 2, silu_mul_args, &out_size);
    TensorNDF32 silu_mul_out;
    if (llm_to_nd(out, out_size, &silu_mul_out) != 0)
        FAIL("llm_silu_mul_nd parse failed");
    assert_closef(
        silu_mul_out.data[0], -0.27363828f, 1e-5f, "llm_silu_mul_nd[0]");
    assert_closef(
        silu_mul_out.data[3], 4.90544653f, 1e-5f, "llm_silu_mul_nd[3]");
    free(out);

    int32_t trig_shape[1] = {2};
    float trig_data[] = {0.0f, 1.0f};
    int trig_size = 0;
    void *trig_blob = build_nd_blob(trig_shape, 1, trig_data, &trig_size);
    SqlArg trig_arg[] = {
        {.type = ARG_BLOB, .blob = trig_blob, .blob_size = trig_size}};
    out = query_blob(db, "SELECT llm_exp_nd(?)", 1, trig_arg, &out_size);
    float exp_expected[] = {1.0f, expf(1.0f)};
    assert_nd_blob_eq(
        out, out_size, trig_shape, 1, exp_expected, 2, 1e-5f, "llm_exp_nd");
    free(out);
    out = query_blob(db, "SELECT llm_cos_nd(?)", 1, trig_arg, &out_size);
    float cos_expected[] = {1.0f, cosf(1.0f)};
    assert_nd_blob_eq(
        out, out_size, trig_shape, 1, cos_expected, 2, 1e-5f, "llm_cos_nd");
    free(out);
    out = query_blob(db, "SELECT llm_sin_nd(?)", 1, trig_arg, &out_size);
    float sin_expected[] = {0.0f, sinf(1.0f)};
    assert_nd_blob_eq(
        out, out_size, trig_shape, 1, sin_expected, 2, 1e-5f, "llm_sin_nd");
    free(out);
    out = query_blob(db, "SELECT llm_tanh_nd(?)", 1, trig_arg, &out_size);
    float tanh_expected[] = {0.0f, tanhf(1.0f)};
    assert_nd_blob_eq(
        out, out_size, trig_shape, 1, tanh_expected, 2, 1e-5f, "llm_tanh_nd");
    free(out);
    out = query_blob(db, "SELECT llm_pow_nd(?, 2.0)", 1, trig_arg, &out_size);
    float pow_expected[] = {0.0f, 1.0f};
    assert_nd_blob_eq(
        out, out_size, trig_shape, 1, pow_expected, 2, 1e-6f, "llm_pow_nd");
    free(out);
    out = query_blob(db, "SELECT llm_abs_nd(?)", 1, unary_arg, &out_size);
    float abs_expected[] = {1.5f, 0.5f, 0.5f, 1.5f};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, abs_expected, 4, 1e-6f, "llm_abs_nd");
    free(out);
    out = query_blob(db, "SELECT llm_sqrt_nd(?)", 1, pos_arg, &out_size);
    float sqrt_expected[] = {1.0f, 2.0f};
    assert_nd_blob_eq(
        out, out_size, pos_shape, 1, sqrt_expected, 2, 1e-6f, "llm_sqrt_nd");
    free(out);
    out = query_blob(db, "SELECT llm_log_nd(?)", 1, pos_arg, &out_size);
    float log_expected[] = {0.0f, logf(4.0f)};
    assert_nd_blob_eq(
        out, out_size, pos_shape, 1, log_expected, 2, 1e-6f, "llm_log_nd");
    free(out);
    out = query_blob(db, "SELECT llm_sigmoid_nd(?)", 1, trig_arg, &out_size);
    float sigmoid_expected[] = {0.5f, 0.73105860f};
    assert_nd_blob_eq(out,
                      out_size,
                      trig_shape,
                      1,
                      sigmoid_expected,
                      2,
                      1e-5f,
                      "llm_sigmoid_nd");
    free(out);
    out = query_blob(db, "SELECT llm_relu_nd(?)", 1, unary_arg, &out_size);
    float relu_expected[] = {0.0f, 0.0f, 0.5f, 1.5f};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, relu_expected, 4, 1e-6f, "llm_relu_nd");
    free(out);
    out = query_blob(db, "SELECT llm_gelu_nd(?)", 1, trig_arg, &out_size);
    TensorNDF32 gelu_out;
    if (llm_to_nd(out, out_size, &gelu_out) != 0)
        FAIL("llm_gelu_nd parse failed");
    assert_closef(gelu_out.data[0], 0.0f, 1e-6f, "llm_gelu_nd[0]");
    free(out);

    float round_data[] = {-1.7f, -0.2f, 0.2f, 1.7f};
    int round_size = 0;
    void *round_blob = build_nd_blob(vec_shape, 1, round_data, &round_size);
    SqlArg round_arg[] = {
        {.type = ARG_BLOB, .blob = round_blob, .blob_size = round_size}};
    out = query_blob(db, "SELECT llm_floor_nd(?)", 1, round_arg, &out_size);
    float floor_expected[] = {-2, -1, 0, 1};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, floor_expected, 4, 1e-6f, "llm_floor_nd");
    free(out);
    out = query_blob(db, "SELECT llm_ceil_nd(?)", 1, round_arg, &out_size);
    float ceil_expected[] = {-1, 0, 1, 2};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, ceil_expected, 4, 1e-6f, "llm_ceil_nd");
    free(out);
    out = query_blob(db, "SELECT llm_round_nd(?)", 1, round_arg, &out_size);
    float round_expected[] = {-2, 0, 0, 2};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, round_expected, 4, 1e-6f, "llm_round_nd");
    free(out);
    out = query_blob(db, "SELECT llm_trunc_nd(?)", 1, round_arg, &out_size);
    float trunc_expected[] = {-1, 0, 0, 1};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, trunc_expected, 4, 1e-6f, "llm_trunc_nd");
    free(out);
    out = query_blob(db, "SELECT llm_sign_nd(?)", 1, round_arg, &out_size);
    float sign_expected[] = {-1, -1, 1, 1};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, sign_expected, 4, 1e-6f, "llm_sign_nd");
    free(out);
    out = query_blob(
        db, "SELECT llm_clamp_nd(?, -1.0, 1.0)", 1, round_arg, &out_size);
    float clamp_expected[] = {-1, -0.2f, 0.2f, 1};
    assert_nd_blob_eq(
        out, out_size, vec_shape, 1, clamp_expected, 4, 1e-6f, "llm_clamp_nd");
    free(out);

    int32_t mat_shape[2] = {3, 3};
    float mat_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int mat_size = 0;
    void *mat_blob = build_nd_blob(mat_shape, 2, mat_data, &mat_size);
    SqlArg mat_arg[] = {
        {.type = ARG_BLOB, .blob = mat_blob, .blob_size = mat_size}};
    out = query_blob(db, "SELECT llm_tril_nd(?, 0)", 1, mat_arg, &out_size);
    float tril_expected[] = {1, 0, 0, 4, 5, 0, 7, 8, 9};
    assert_nd_blob_eq(
        out, out_size, mat_shape, 2, tril_expected, 9, 1e-6f, "llm_tril_nd");
    free(out);
    out = query_blob(db, "SELECT llm_triu_nd(?, 0)", 1, mat_arg, &out_size);
    float triu_expected[] = {1, 2, 3, 0, 5, 6, 0, 0, 9};
    assert_nd_blob_eq(
        out, out_size, mat_shape, 2, triu_expected, 9, 1e-6f, "llm_triu_nd");
    free(out);

    float logic_a[] = {0, 1, 1, 0};
    float logic_b[] = {0, 0, 1, 1};
    int logic_a_size = 0;
    int logic_b_size = 0;
    void *logic_a_blob = build_nd_blob(vec_shape, 1, logic_a, &logic_a_size);
    void *logic_b_blob = build_nd_blob(vec_shape, 1, logic_b, &logic_b_size);
    SqlArg logic_args[] = {
        {.type = ARG_BLOB, .blob = logic_a_blob, .blob_size = logic_a_size},
        {.type = ARG_BLOB, .blob = logic_b_blob, .blob_size = logic_b_size},
    };
    out = query_blob(db,
                     "SELECT llm_logical_not_nd(?)",
                     1,
                     (SqlArg[]){{.type = ARG_BLOB,
                                 .blob = logic_a_blob,
                                 .blob_size = logic_a_size}},
                     &out_size);
    float logical_not_expected[] = {1, 0, 0, 1};
    assert_nd_blob_eq(out,
                      out_size,
                      vec_shape,
                      1,
                      logical_not_expected,
                      4,
                      1e-6f,
                      "llm_logical_not_nd");
    free(out);
    out = query_blob(
        db, "SELECT llm_logical_and_nd(?, ?)", 2, logic_args, &out_size);
    float logical_and_expected[] = {0, 0, 1, 0};
    assert_nd_blob_eq(out,
                      out_size,
                      vec_shape,
                      1,
                      logical_and_expected,
                      4,
                      1e-6f,
                      "llm_logical_and_nd");
    free(out);
    out = query_blob(
        db, "SELECT llm_logical_or_nd(?, ?)", 2, logic_args, &out_size);
    float logical_or_expected[] = {0, 1, 1, 1};
    assert_nd_blob_eq(out,
                      out_size,
                      vec_shape,
                      1,
                      logical_or_expected,
                      4,
                      1e-6f,
                      "llm_logical_or_nd");
    free(out);
    out = query_blob(db,
                     "SELECT llm_bitwise_not_nd(?)",
                     1,
                     (SqlArg[]){{.type = ARG_BLOB,
                                 .blob = logic_a_blob,
                                 .blob_size = logic_a_size}},
                     &out_size);
    assert_nd_blob_eq(out,
                      out_size,
                      vec_shape,
                      1,
                      logical_not_expected,
                      4,
                      1e-6f,
                      "llm_bitwise_not_nd");
    free(out);
    out = query_blob(
        db, "SELECT llm_bitwise_and_nd(?, ?)", 2, logic_args, &out_size);
    assert_nd_blob_eq(out,
                      out_size,
                      vec_shape,
                      1,
                      logical_and_expected,
                      4,
                      1e-6f,
                      "llm_bitwise_and_nd");
    free(out);
    out = query_blob(
        db, "SELECT llm_bitwise_or_nd(?, ?)", 2, logic_args, &out_size);
    assert_nd_blob_eq(out,
                      out_size,
                      vec_shape,
                      1,
                      logical_or_expected,
                      4,
                      1e-6f,
                      "llm_bitwise_or_nd");
    free(out);
    out = query_blob(
        db,
        "SELECT llm_bitwise_and_scalar_nd(?, 1)",
        1,
        (SqlArg[]){{.type = ARG_BLOB, .blob = a_blob, .blob_size = a_size}},
        &out_size);
    float bitwise_scalar_expected[] = {1, 0, 0, 0};
    assert_nd_blob_eq(out,
                      out_size,
                      vec_shape,
                      1,
                      bitwise_scalar_expected,
                      4,
                      1e-6f,
                      "llm_bitwise_and_scalar_nd");
    free(out);

    out = query_blob(
        db,
        "SELECT llm_pow_scalar_nd(2.0, ?)",
        1,
        (SqlArg[]){{.type = ARG_BLOB, .blob = b_blob, .blob_size = b_size}},
        &out_size);
    float pow_scalar_expected[] = {4, 4, 2, 32};
    assert_nd_blob_eq(out,
                      out_size,
                      vec_shape,
                      1,
                      pow_scalar_expected,
                      4,
                      1e-6f,
                      "llm_pow_scalar_nd");
    free(out);
    out =
        query_blob(db, "SELECT llm_floor_div_nd(?, ?)", 2, cmp_args, &out_size);
    float floor_div_expected[] = {0, 1, 2, 0};
    assert_nd_blob_eq(out,
                      out_size,
                      vec_shape,
                      1,
                      floor_div_expected,
                      4,
                      1e-6f,
                      "llm_floor_div_nd");
    free(out);
    out = query_blob(
        db,
        "SELECT llm_remainder_nd(?, 2.0)",
        1,
        (SqlArg[]){{.type = ARG_BLOB, .blob = a_blob, .blob_size = a_size}},
        &out_size);
    float remainder_expected[] = {1, 0, 0, 0};
    assert_nd_blob_eq(out,
                      out_size,
                      vec_shape,
                      1,
                      remainder_expected,
                      4,
                      1e-6f,
                      "llm_remainder_nd");
    free(out);
    out = query_blob(
        db, "SELECT llm_remainder_tensor_nd(?, ?)", 2, cmp_args, &out_size);
    float remainder_tensor_expected[] = {1, 0, 0, 4};
    assert_nd_blob_eq(out,
                      out_size,
                      vec_shape,
                      1,
                      remainder_tensor_expected,
                      4,
                      1e-6f,
                      "llm_remainder_tensor_nd");
    free(out);

    out = query_blob(
        db,
        "SELECT llm_reduce_sum_nd(?)",
        1,
        (SqlArg[]){{.type = ARG_BLOB, .blob = a_blob, .blob_size = a_size}},
        &out_size);
    assert_nd_blob_eq(out,
                      out_size,
                      (int32_t[]){1},
                      1,
                      (float[]){9.0f},
                      1,
                      1e-6f,
                      "llm_reduce_sum_nd");
    free(out);
    out = query_blob(
        db,
        "SELECT llm_reduce_mean_nd(?)",
        1,
        (SqlArg[]){{.type = ARG_BLOB, .blob = a_blob, .blob_size = a_size}},
        &out_size);
    assert_nd_blob_eq(out,
                      out_size,
                      (int32_t[]){1},
                      1,
                      (float[]){2.25f},
                      1,
                      1e-6f,
                      "llm_reduce_mean_nd");
    free(out);
    out = query_blob(
        db,
        "SELECT llm_reduce_max_nd(?)",
        1,
        (SqlArg[]){{.type = ARG_BLOB, .blob = a_blob, .blob_size = a_size}},
        &out_size);
    assert_nd_blob_eq(out,
                      out_size,
                      (int32_t[]){1},
                      1,
                      (float[]){4.0f},
                      1,
                      1e-6f,
                      "llm_reduce_max_nd");
    free(out);
    out = query_blob(
        db,
        "SELECT llm_reduce_min_nd(?)",
        1,
        (SqlArg[]){
            {.type = ARG_BLOB, .blob = unary_blob, .blob_size = unary_size}},
        &out_size);
    assert_nd_blob_eq(out,
                      out_size,
                      (int32_t[]){1},
                      1,
                      (float[]){-1.5f},
                      1,
                      1e-6f,
                      "llm_reduce_min_nd");
    free(out);

    out = query_blob(db, "SELECT llm_arange_nd(4)", 0, NULL, &out_size);
    float arange_expected[] = {0, 1, 2, 3};
    assert_nd_blob_eq(out,
                      out_size,
                      vec_shape,
                      1,
                      arange_expected,
                      4,
                      1e-6f,
                      "llm_arange_nd");
    free(out);
    out = query_blob(db, "SELECT llm_full_nd(1.5, 2, 2)", 0, NULL, &out_size);
    int32_t full_shape[2] = {2, 2};
    float full_expected[] = {1.5f, 1.5f, 1.5f, 1.5f};
    assert_nd_blob_eq(
        out, out_size, full_shape, 2, full_expected, 4, 1e-6f, "llm_full_nd");
    free(out);

    free(logic_b_blob);
    free(logic_a_blob);
    free(mat_blob);
    free(round_blob);
    free(trig_blob);
    free(pos_blob);
    free(unary_blob);
    free(b_blob);
    free(a_blob);
}

static void test_param_lookup_ops_x86(void) {
    const char *db_path = "/tmp/llm_ops_param_test_x86.sqlite";
    (void)unlink(db_path);
    sqlite3 *db = open_file_test_db_x86(db_path);

    exec_sql(db,
             "CREATE TABLE IF NOT EXISTS model_params ("
             "name TEXT PRIMARY KEY, "
             "data BLOB NOT NULL)");

    int32_t w_shape[2] = {2, 2};
    float w_data[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    int w_sz = 0;
    void *w_blob = llm_build_nd(2, w_shape, w_data, &w_sz);
    if (!w_blob)
        FAIL("llm_build_nd(w_blob) failed");

    sqlite3_stmt *ins = prepare_stmt(
        db, "INSERT OR REPLACE INTO model_params(name, data) VALUES (?1, ?2)");
    sqlite3_bind_text(ins, 1, "weight_fp32", -1, SQLITE_STATIC);
    sqlite3_bind_blob(ins, 2, w_blob, w_sz, SQLITE_STATIC);
    if (sqlite3_step(ins) != SQLITE_DONE)
        FAIL("insert weight_fp32 failed: %s", sqlite3_errmsg(db));
    sqlite3_finalize(ins);

    SqlArg w_arg[] = {{.type = ARG_BLOB, .blob = w_blob, .blob_size = w_sz}};
    int wq_sz = 0;
    void *wq_blob =
        query_blob(db, "SELECT llm_quantize_int8_nd(?)", 1, w_arg, &wq_sz);
    ins = prepare_stmt(
        db, "INSERT OR REPLACE INTO model_params(name, data) VALUES (?1, ?2)");
    sqlite3_bind_text(ins, 1, "weight_i8", -1, SQLITE_STATIC);
    sqlite3_bind_blob(ins, 2, wq_blob, wq_sz, SQLITE_STATIC);
    if (sqlite3_step(ins) != SQLITE_DONE)
        FAIL("insert weight_i8 failed: %s", sqlite3_errmsg(db));
    sqlite3_finalize(ins);

    int32_t emb_shape[2] = {3, 2};
    float emb_data[6] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int emb_sz = 0;
    void *emb_blob = llm_build_nd(2, emb_shape, emb_data, &emb_sz);
    if (!emb_blob)
        FAIL("llm_build_nd(emb_blob) failed");
    ins = prepare_stmt(
        db, "INSERT OR REPLACE INTO model_params(name, data) VALUES (?1, ?2)");
    sqlite3_bind_text(ins, 1, "emb_fp32", -1, SQLITE_STATIC);
    sqlite3_bind_blob(ins, 2, emb_blob, emb_sz, SQLITE_STATIC);
    if (sqlite3_step(ins) != SQLITE_DONE)
        FAIL("insert emb_fp32 failed: %s", sqlite3_errmsg(db));
    sqlite3_finalize(ins);

    int x_sz = 0;
    float x_data[2] = {7.0f, 9.0f};
    void *x_blob = build_vec_blob(x_data, 2, &x_sz);
    SqlArg x_text_args[] = {
        {.type = ARG_BLOB, .blob = x_blob, .blob_size = x_sz},
        {.type = ARG_TEXT, .text_value = "weight_fp32"},
    };
    int out_sz = 0;
    void *out = query_blob(
        db, "SELECT llm_linear_param_nd(?, ?)", 2, x_text_args, &out_sz);
    TensorNDF32 lin_out;
    if (llm_to_nd(out, out_sz, &lin_out) != 0)
        FAIL("llm_linear_param_nd fp32 parse failed");
    ASSERT_TRUE(lin_out.total == 2, "llm_linear_param_nd fp32 shape");
    assert_closef(lin_out.data[0], 7.0f, 1e-5f, "llm_linear_param_nd fp32[0]");
    assert_closef(lin_out.data[1], 9.0f, 1e-5f, "llm_linear_param_nd fp32[1]");
    free(out);

    x_text_args[1].text_value = "weight_i8";
    out = query_blob(
        db, "SELECT llm_linear_param_nd(?, ?)", 2, x_text_args, &out_sz);
    if (llm_to_nd(out, out_sz, &lin_out) != 0)
        FAIL("llm_linear_param_nd int8 parse failed");
    assert_closef(lin_out.data[0], 7.0f, 0.2f, "llm_linear_param_nd int8[0]");
    assert_closef(lin_out.data[1], 9.0f, 0.2f, "llm_linear_param_nd int8[1]");
    free(out);

    int b_sz = 0;
    float b_data[2] = {0.5f, -0.5f};
    void *b_blob = build_vec_blob(b_data, 2, &b_sz);
    SqlArg x_text_bias_args[] = {
        {.type = ARG_BLOB, .blob = x_blob, .blob_size = x_sz},
        {.type = ARG_TEXT, .text_value = "weight_fp32"},
        {.type = ARG_BLOB, .blob = b_blob, .blob_size = b_sz},
    };
    out = query_blob(db,
                     "SELECT llm_linear_param_bias_nd(?, ?, ?)",
                     3,
                     x_text_bias_args,
                     &out_sz);
    if (llm_to_nd(out, out_sz, &lin_out) != 0)
        FAIL("llm_linear_param_bias_nd parse failed");
    ASSERT_TRUE(lin_out.total == 2, "llm_linear_param_bias_nd shape");
    assert_closef(lin_out.data[0], 7.5f, 1e-5f, "llm_linear_param_bias_nd[0]");
    assert_closef(lin_out.data[1], 8.5f, 1e-5f, "llm_linear_param_bias_nd[1]");
    free(out);

    int idx_sz = 0;
    float idx_data[2] = {2.0f, 0.0f};
    void *idx_blob = build_vec_blob(idx_data, 2, &idx_sz);
    SqlArg emb_param_args[] = {
        {.type = ARG_TEXT, .text_value = "emb_fp32"},
        {.type = ARG_BLOB, .blob = idx_blob, .blob_size = idx_sz},
    };
    out = query_blob(
        db, "SELECT llm_embedding_param_nd(?, ?)", 2, emb_param_args, &out_sz);
    TensorNDF32 emb_out;
    if (llm_to_nd(out, out_sz, &emb_out) != 0)
        FAIL("llm_embedding_param_nd output parse failed");
    ASSERT_TRUE(emb_out.ndim == 2 && emb_out.shape[0] == 2 &&
                    emb_out.shape[1] == 2,
                "llm_embedding_param_nd output shape");
    assert_closef(emb_out.data[0], 4.0f, 1e-6f, "llm_embedding_param_nd[0]");
    assert_closef(emb_out.data[1], 5.0f, 1e-6f, "llm_embedding_param_nd[1]");
    assert_closef(emb_out.data[2], 0.0f, 1e-6f, "llm_embedding_param_nd[2]");
    assert_closef(emb_out.data[3], 1.0f, 1e-6f, "llm_embedding_param_nd[3]");
    free(out);

    SqlArg get_param_args[] = {{.type = ARG_TEXT, .text_value = "weight_fp32"}};
    out = query_blob(db, "SELECT llm_get_param(?)", 1, get_param_args, &out_sz);
    TensorNDF32 got_param;
    if (llm_to_nd(out, out_sz, &got_param) != 0)
        FAIL("llm_get_param output parse failed");
    ASSERT_TRUE(got_param.ndim == 2 && got_param.shape[0] == 2 &&
                    got_param.shape[1] == 2,
                "llm_get_param shape");
    free(out);

    free(idx_blob);
    free(b_blob);
    free(x_blob);
    free(emb_blob);
    free(wq_blob);
    free(w_blob);
    sqlite3_close(db);
    (void)unlink(db_path);
}

static void benchmark_nd_and_int8_smoke_x86(sqlite3 *db) {
    static const char *bench_sqls[][2] = {
        {"bench_llm_view_nd_x86",
         "SELECT llm_view_nd(llm_vec(1024, 1.0), 256, 4)"},
        {"bench_llm_add_nd_x86",
         "SELECT llm_add_nd(llm_vec(1024, 1.0), llm_vec(1024, 2.0))"},
        {"bench_llm_softmax_nd_x86",
         "SELECT llm_softmax_nd(llm_view_nd(llm_vec(1024, 1.0), 32, 32), 1)"},
        {"bench_llm_bmm_x86",
         "SELECT llm_bmm(llm_view_nd(llm_vec(4096, 1.0), 4, 16, 64), "
         "llm_view_nd(llm_vec(4096, 1.0), 4, 64, 16))"},
        {"bench_llm_sdpa_nd_x86",
         "SELECT llm_sdpa_nd(llm_view_nd(llm_vec(512, 1.0), 1, 2, 16, 16), "
         "llm_view_nd(llm_vec(512, 1.0), 1, 2, 16, 16), "
         "llm_view_nd(llm_vec(512, 1.0), 1, 2, 16, 16), 0.25, 0)"},
        {"bench_llm_quantize_int8_nd_x86",
         "SELECT llm_quantize_int8_nd(llm_view_nd(llm_vec(4096, 1.0), 64, "
         "64))"},
        {"bench_llm_linear_int8_nd_x86",
         "SELECT llm_linear_int8_nd(llm_view_nd(llm_vec(256, 1.0), 1, 256), "
         "llm_quantize_int8_nd(llm_view_nd(llm_vec(16384, 1.0), 64, 256)))"},
    };

    for (int i = 0; i < ARRAY_LEN(bench_sqls); i++) {
        sqlite3_stmt *stmt = prepare_stmt(db, bench_sqls[i][1]);
        int iterations = 50;
        double start_ms = now_ms();
        for (int it = 0; it < iterations; it++) {
            sqlite3_reset(stmt);
            int rc = sqlite3_step(stmt);
            if (rc != SQLITE_ROW)
                FAIL("%s benchmark failed: %s",
                     bench_sqls[i][0],
                     sqlite3_errmsg(db));
        }
        double elapsed_ms = now_ms() - start_ms;
        printf("[bench] %s iterations=%d total=%.3f ms avg=%.6f ms\n",
               bench_sqls[i][0],
               iterations,
               elapsed_ms,
               elapsed_ms / (double)iterations);
        sqlite3_finalize(stmt);
    }
}

static void run_test_case(sqlite3 *db, const TestCase *test_case) {
    double start_ms = now_ms();
    test_case->fn(db);
    double elapsed_ms = now_ms() - start_ms;
    g_tests_run += 1;
    printf("[pass] %s (%.3f ms)\n", test_case->name, elapsed_ms);
}

static void run_benchmark_case(sqlite3 *db, const BenchmarkCase *bench_case) {
    sqlite3_stmt *stmt = prepare_stmt(db, bench_case->sql);
    bind_args(stmt, bench_case->argc, bench_case->args);

    double start_ms = now_ms();
    for (int i = 0; i < bench_case->iterations; i++) {
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
        bind_args(stmt, bench_case->argc, bench_case->args);
        int rc = sqlite3_step(stmt);
        if (rc != SQLITE_ROW)
            FAIL("benchmark failed for [%s]: %s",
                 bench_case->sql,
                 sqlite3_errmsg(db));
    }
    double elapsed_ms = now_ms() - start_ms;
    sqlite3_finalize(stmt);

    printf("[bench] %s iterations=%d total=%.3f ms avg=%.6f ms\n",
           bench_case->name,
           bench_case->iterations,
           elapsed_ms,
           elapsed_ms / (double)bench_case->iterations);
}

int main(void) {
    sqlite3 *db = open_test_db_x86();

    TestCase tests[] = {
        {"test_runtime_threads_x86", test_runtime_threads_x86},
        {"test_construct_and_inspect_x86", test_construct_and_inspect_x86},
        {"test_vector_arithmetic_x86", test_vector_arithmetic_x86},
        {"test_matrix_core_ops_x86", test_matrix_core_ops_x86},
        {"test_unary_activation_reduction_x86",
         test_unary_activation_reduction_x86},
        {"test_shape_index_and_rope_x86", test_shape_index_and_rope_x86},
        {"test_nd_and_int8_smoke_x86", test_nd_and_int8_smoke_x86},
        {"test_nd_broadcast_reduce_x86", test_nd_broadcast_reduce_x86},
        {"test_nd_transform_index_x86", test_nd_transform_index_x86},
        {"test_nd_model_ops_x86", test_nd_model_ops_x86},
        {"test_int8_ops_x86", test_int8_ops_x86},
        {"test_nd_misc_ops_x86", test_nd_misc_ops_x86},
    };

    for (int i = 0; i < ARRAY_LEN(tests); i++)
        run_test_case(db, &tests[i]);

    float bench_a[1024];
    float bench_b[1024];
    for (int i = 0; i < 1024; i++) {
        bench_a[i] = (float)(i % 17) * 0.25f;
        bench_b[i] = (float)((i % 13) - 6) * 0.5f;
    }
    int bench_a_size = 0;
    int bench_b_size = 0;
    void *bench_a_blob = build_vec_blob(bench_a, 1024, &bench_a_size);
    void *bench_b_blob = build_vec_blob(bench_b, 1024, &bench_b_size);

    SqlArg bench_args[] = {
        {.type = ARG_BLOB, .blob = bench_a_blob, .blob_size = bench_a_size},
        {.type = ARG_BLOB, .blob = bench_b_blob, .blob_size = bench_b_size},
    };
    BenchmarkCase benches[] = {
        {"bench_llm_add_simd_x86",
         "SELECT llm_add_simd(?, ?)",
         500,
         2,
         bench_args},
        {"bench_llm_dot_simd_x86",
         "SELECT llm_dot_simd(?, ?)",
         1000,
         2,
         bench_args},
        {"bench_llm_mul_simd_x86",
         "SELECT llm_mul_simd(?, ?)",
         500,
         2,
         bench_args},
        {"bench_llm_div_simd_x86",
         "SELECT llm_div_simd(?, ?)",
         500,
         2,
         bench_args},
    };

    for (int i = 0; i < ARRAY_LEN(benches); i++)
        run_benchmark_case(db, &benches[i]);

    benchmark_nd_and_int8_smoke_x86(db);

    free(bench_a_blob);
    free(bench_b_blob);
    sqlite3_close(db);

    test_param_lookup_ops_x86();
    printf("[pass] test_param_lookup_ops_x86\n");

    if (g_tests_failed != 0)
        return 1;
    printf("x86 operator test batch complete: %d tests passed\n", g_tests_run);
    return 0;
}