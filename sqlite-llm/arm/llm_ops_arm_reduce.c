/*
 * llm_ops_arm_reduce.c - Vector reductions and reduction-style wrappers
 *
 * This file owns the legacy vector reduction surface: sum, mean, max, min,
 * and any helper that collapses a VecF32 to a scalar or reduction-shaped
 * result. Use this file when the operator's main job is aggregation rather
 * than elementwise transformation.
 *
 * Maintenance guidance:
 *   - Keep SIMD accumulation helpers here so reduction accuracy and fallback
 *     behavior stay aligned across wrappers.
 *   - When adding a reduction, decide first whether it belongs to legacy 1-D
 *     vectors here or to the N-D reduction block in the main tensor section.
 *   - Preserve scalar return semantics for the legacy SQL functions instead of
 *     silently switching them to blob outputs.
 */
#include "llm_ops_arm_internal.h"
SQLITE_EXTENSION_INIT3

static float k_reduce_sum(const float *a, int n) {
#if LLM_HAVE_NEON
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= n - 4; i += 4)
        acc = vaddq_f32(acc, vld1q_f32(a + i));
    float s = llm_hsumq_f32(acc);
    for (; i < n; i++)
        s += a[i];
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++)
        s += a[i];
    return s;
#endif
}

static float k_reduce_max(const float *a, int n) {
#if LLM_HAVE_NEON
    if (n >= 4) {
        float32x4_t vmx = vld1q_f32(a);
        int i = 4;
        for (; i <= n - 4; i += 4)
            vmx = vmaxq_f32(vmx, vld1q_f32(a + i));
        float mx = llm_hmaxq_f32(vmx);
        for (; i < n; i++)
            if (a[i] > mx)
                mx = a[i];
        return mx;
    }
#endif
    float mx = a[0];
    for (int i = 1; i < n; i++)
        if (a[i] > mx)
            mx = a[i];
    return mx;
}

static float k_reduce_min(const float *a, int n) {
#if LLM_HAVE_NEON
    if (n >= 4) {
        float32x4_t vmn = vld1q_f32(a);
        int i = 4;
        for (; i <= n - 4; i += 4)
            vmn = vminq_f32(vmn, vld1q_f32(a + i));
        float mn = llm_hminq_f32(vmn);
        for (; i < n; i++)
            if (a[i] < mn)
                mn = a[i];
        return mn;
    }
#endif
    float mn = a[0];
    for (int i = 1; i < n; i++)
        if (a[i] < mn)
            mn = a[i];
    return mn;
}

void sql_sum(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    if (va.n == 0)
        ERR(ctx, "Empty vector");
    sqlite3_result_double(ctx, (double)k_reduce_sum(va.data, va.n));
}

void sql_mean(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    if (va.n == 0)
        ERR(ctx, "Empty vector");
    sqlite3_result_double(ctx,
                          (double)k_reduce_sum(va.data, va.n) / (double)va.n);
}

void sql_max(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    if (va.n == 0)
        ERR(ctx, "Empty vector");
    sqlite3_result_double(ctx, (double)k_reduce_max(va.data, va.n));
}

void sql_min(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    if (va.n == 0)
        ERR(ctx, "Empty vector");
    sqlite3_result_double(ctx, (double)k_reduce_min(va.data, va.n));
}