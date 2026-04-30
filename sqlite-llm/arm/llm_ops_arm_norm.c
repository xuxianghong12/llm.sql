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
#if LLM_HAVE_NEON
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        acc = vmlaq_f32(acc, va, va);
    }
    ss = llm_hsumq_f32(acc);
    for (; i < n; i++)
        ss += a[i] * a[i];
#else
    for (int i = 0; i < n; i++)
        ss += a[i] * a[i];
#endif
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
#if LLM_HAVE_NEON
    float32x4_t vscale = vdupq_n_f32(scale);
    int i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vw = vld1q_f32(w + i);
        vst1q_f32(b + i, vmulq_f32(vmulq_f32(va, vscale), vw));
    }
    for (; i < n; i++)
        b[i] = a[i] * scale * w[i];
#else
    for (int i = 0; i < n; i++)
        b[i] = a[i] * scale * w[i];
#endif
}

static void k_layernorm(const float *a,
                        const float *w,
                        const float *bias,
                        float *b,
                        int n,
                        float eps) {
    float mean = 0.0f;
#if LLM_HAVE_NEON
    float32x4_t acc_mean = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= n - 4; i += 4)
        acc_mean = vaddq_f32(acc_mean, vld1q_f32(a + i));
    mean = llm_hsumq_f32(acc_mean);
    for (; i < n; i++)
        mean += a[i];
#else
    for (int i = 0; i < n; i++)
        mean += a[i];
#endif
    mean /= (float)n;
    float var = 0.0f;
#if LLM_HAVE_NEON
    float32x4_t vmean = vdupq_n_f32(mean);
    float32x4_t acc_var = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(a + i), vmean);
        acc_var = vmlaq_f32(acc_var, d, d);
    }
    var = llm_hsumq_f32(acc_var);
    for (; i < n; i++) {
        float d = a[i] - mean;
        var += d * d;
    }
#else
    for (int i = 0; i < n; i++) {
        float d = a[i] - mean;
        var += d * d;
    }
#endif
    var /= (float)n;
    float scale = 1.0f / sqrtf(var + eps);
#if LLM_HAVE_NEON
    float32x4_t vscale = vdupq_n_f32(scale);
    int i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t norm =
            vmulq_f32(vsubq_f32(vld1q_f32(a + i), vmean), vscale);
        vst1q_f32(
            b + i,
            vaddq_f32(vmulq_f32(norm, vld1q_f32(w + i)), vld1q_f32(bias + i)));
    }
    for (; i < n; i++)
        b[i] = (a[i] - mean) * scale * w[i] + bias[i];
#else
    for (int i = 0; i < n; i++)
        b[i] = (a[i] - mean) * scale * w[i] + bias[i];
#endif
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