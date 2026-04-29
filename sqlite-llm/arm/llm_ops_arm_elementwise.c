/*
 * llm_ops_arm_elementwise.c - Vector unary math and activation wrappers
 *
 * This file owns 1-D vector operators that apply one output formula per input
 * element: basic unary math, clipping, activations, and the vector softmax
 * wrapper. Add code here when the SQL surface stays in legacy VecF32 form and
 * does not need matrix, N-D, or reduction-specific shape handling.
 *
 * Maintenance guidance:
 *   - Keep low-level vector kernels private to this file unless another split
 *     unit needs to reuse them.
 *   - Reuse DEFINE_UNARY_VEC or the existing vector allocation helpers when a
 *     new operator is a direct elementwise map over VecF32.
 *   - Move anything that changes rank or depends on tensor broadcasting into
 *     the dedicated shape or N-D files instead of growing this file sideways.
 */
#include "llm_ops_arm_internal.h"
SQLITE_EXTENSION_INIT3

static void k_neg_simd(const float *a, float *b, int n) {
    for (int i = 0; i < n; i++)
        b[i] = -a[i];
}

static void k_abs_simd(const float *a, float *b, int n) {
    for (int i = 0; i < n; i++)
        b[i] = fabsf(a[i]);
}

static void k_sqrt_simd(const float *a, float *b, int n) {
    for (int i = 0; i < n; i++)
        b[i] = sqrtf(a[i]);
}

static void k_exp_simd(const float *a, float *b, int n) {
    for (int i = 0; i < n; i++)
        b[i] = expf(a[i]);
}

static void k_log_simd(const float *a, float *b, int n) {
    for (int i = 0; i < n; i++)
        b[i] = logf(a[i]);
}

static void k_clip_simd(const float *a, float lo, float hi, float *b, int n) {
    for (int i = 0; i < n; i++) {
        float v = a[i];
        b[i] = v < lo ? lo : (v > hi ? hi : v);
    }
}

static void k_relu_simd(const float *a, float *b, int n) {
    for (int i = 0; i < n; i++)
        b[i] = a[i] > 0.0f ? a[i] : 0.0f;
}

static void k_gelu_simd(const float *a, float *b, int n) {
    for (int i = 0; i < n; i++)
        b[i] = llm_gelu(a[i]);
}

static void k_silu_simd(const float *a, float *b, int n) {
    for (int i = 0; i < n; i++)
        b[i] = llm_silu(a[i]);
}

static void k_sigmoid_simd(const float *a, float *b, int n) {
    for (int i = 0; i < n; i++)
        b[i] = llm_sigmoid(a[i]);
}

static void k_softmax_simd(const float *a, float *b, int n) {
    float mx = a[0];
    for (int i = 1; i < n; i++)
        if (a[i] > mx)
            mx = a[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        b[i] = expf(a[i] - mx);
        sum += b[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++)
        b[i] *= inv;
}

void sql_neg(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    EMIT_VEC_BINOP(ctx, va, k_neg_simd(va.data, tmp, va.n));
}

void sql_abs(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    EMIT_VEC_BINOP(ctx, va, k_abs_simd(va.data, tmp, va.n));
}

void sql_sqrt(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    EMIT_VEC_BINOP(ctx, va, k_sqrt_simd(va.data, tmp, va.n));
}

void sql_exp(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    EMIT_VEC_BINOP(ctx, va, k_exp_simd(va.data, tmp, va.n));
}

void sql_log(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    EMIT_VEC_BINOP(ctx, va, k_log_simd(va.data, tmp, va.n));
}

void sql_clip(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    float lo = (float)sqlite3_value_double(argv[1]);
    float hi = (float)sqlite3_value_double(argv[2]);
    EMIT_VEC_BINOP(ctx, va, k_clip_simd(va.data, lo, hi, tmp, va.n));
}

void sql_relu(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    EMIT_VEC_BINOP(ctx, va, k_relu_simd(va.data, tmp, va.n));
}

void sql_gelu(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    EMIT_VEC_BINOP(ctx, va, k_gelu_simd(va.data, tmp, va.n));
}

void sql_silu(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    EMIT_VEC_BINOP(ctx, va, k_silu_simd(va.data, tmp, va.n));
}

void sql_sigmoid(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    EMIT_VEC_BINOP(ctx, va, k_sigmoid_simd(va.data, tmp, va.n));
}

void sql_softmax(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    EMIT_VEC_BINOP(ctx, va, k_softmax_simd(va.data, tmp, va.n));
}