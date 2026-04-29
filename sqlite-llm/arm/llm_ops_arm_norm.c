/*
 * llm_ops_arm_norm.c - Legacy vector normalization wrappers
 *
 * This file owns normalization operators over VecF32 inputs, including RMSNorm
 * and LayerNorm. Add code here when the SQL function consumes vector-shaped
 * blobs and the semantics are normalization-specific rather than generic
 * elementwise math.
 *
 * Maintenance guidance:
 *   - Keep statistical passes and affine application together so numerical
 *     behavior is easy to audit.
 *   - Reuse the shared vector allocation helpers instead of hand-rolling blob
 *     headers in each wrapper.
 *   - Put tensor-rank-aware normalization in the N-D section, not here.
 */
#include "llm_ops_arm_internal.h"
SQLITE_EXTENSION_INIT3

static void
k_rmsnorm(const float *a, const float *w, float *b, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++)
        ss += a[i] * a[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++)
        b[i] = a[i] * scale * w[i];
}

static void k_layernorm(const float *a,
                        const float *w,
                        const float *bias,
                        float *b,
                        int n,
                        float eps) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++)
        mean += a[i];
    mean /= (float)n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    float scale = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++)
        b[i] = (a[i] - mean) * scale * w[i] + bias[i];
}

void sql_rmsnorm(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va, vw;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid input vector blob");
    if (llm_parse_vec(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &vw) < 0)
        ERR(ctx, "Invalid weight vector blob");
    if (va.n != vw.n)
        ERR(ctx, "Vector size mismatch");
    float eps = (float)sqlite3_value_double(argv[2]);
    EMIT_VEC_BINOP(ctx, va, k_rmsnorm(va.data, vw.data, tmp, va.n, eps));
}

void sql_layernorm(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va, vw, vb;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid input vector blob");
    if (llm_parse_vec(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &vw) < 0)
        ERR(ctx, "Invalid weight vector blob");
    if (llm_parse_vec(
            sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &vb) < 0)
        ERR(ctx, "Invalid bias vector blob");
    if (va.n != vw.n || va.n != vb.n)
        ERR(ctx, "Vector size mismatch");
    float eps = (float)sqlite3_value_double(argv[3]);
    EMIT_VEC_BINOP(
        ctx, va, k_layernorm(va.data, vw.data, vb.data, tmp, va.n, eps));
}