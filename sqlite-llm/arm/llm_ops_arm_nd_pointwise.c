/*
 * llm_ops_arm_nd_pointwise.c - N-D pointwise math, logic, and scalar wrappers
 *
 * This file owns SQL entry points that transform tensor elements independently
 * or pairwise without changing reduction structure: unary math, simple binary
 * pointwise logic, scalar-vs-tensor operations, and RoPE-style per-row fused
 * transforms. If an operator walks tensor storage once and writes one output
 * element per input position, it usually belongs here.
 *
 * Maintenance guidance:
 *   - Keep shape-preserving elementwise operators here; move broadcasted
 *     arithmetic to llm_ops_arm_nd_algebra.c and dim-changing reductions to
 *     the file that owns reduction semantics.
 *   - Reuse nd_result_alloc() for any shape-preserving output blob so the
 *     tensor header stays consistent with the source tensor.
 *   - When adding a new operator, keep validation next to the SQL wrapper and
 *     prefer obvious scalar code unless profiling shows a real need for SIMD.
 */

#include "llm_ops_arm_internal.h"
SQLITE_EXTENSION_INIT3

void sql_neg_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_neg_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = -t.data[i];
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_scale_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_scale_nd: invalid blob");
    float s = (float)sqlite3_value_double(argv[1]);
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = t.data[i] * s;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_rsqrt_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_rsqrt_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = 1.0f / sqrtf(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_silu_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_silu_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = llm_silu(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_silu_mul_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *b_gate = sqlite3_value_blob(argv[0]);
    int sz_gate = sqlite3_value_bytes(argv[0]);
    const void *b_up = sqlite3_value_blob(argv[1]);
    int sz_up = sqlite3_value_bytes(argv[1]);
    TensorNDF32 tg, tu;
    if (llm_to_nd(b_gate, sz_gate, &tg) < 0)
        ERR(ctx, "llm_silu_mul_nd: invalid gate blob");
    if (llm_to_nd(b_up, sz_up, &tu) < 0)
        ERR(ctx, "llm_silu_mul_nd: invalid up blob");
    if (tg.total != tu.total)
        ERR(ctx,
            "llm_silu_mul_nd: gate and up must have the same number of elements");
    int out_sz;
    void *out = nd_result_alloc(tg.ndim, tg.shape, tg.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tg.ndim));
    for (int32_t i = 0; i < tg.total; i++)
        dst[i] = llm_silu(tg.data[i]) * tu.data[i];
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_rope_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *bx = sqlite3_value_blob(argv[0]);
    int szx = sqlite3_value_bytes(argv[0]);
    const void *bc = sqlite3_value_blob(argv[1]);
    int szc = sqlite3_value_bytes(argv[1]);
    const void *bs = sqlite3_value_blob(argv[2]);
    int szs = sqlite3_value_bytes(argv[2]);
    TensorNDF32 tx, tc, ts;
    if (llm_to_nd(bx, szx, &tx) < 0)
        ERR(ctx, "llm_rope_nd: invalid x blob");
    if (llm_to_nd(bc, szc, &tc) < 0)
        ERR(ctx, "llm_rope_nd: invalid cos blob");
    if (llm_to_nd(bs, szs, &ts) < 0)
        ERR(ctx, "llm_rope_nd: invalid sin blob");

    int32_t D = tx.shape[tx.ndim - 1];
    if (tc.shape[tc.ndim - 1] != D || ts.shape[ts.ndim - 1] != D)
        ERR(ctx, "llm_rope_nd: last dim of x, cos, sin must match");
    if (D < 2 || D % 2 != 0)
        ERR(ctx, "llm_rope_nd: head_dim must be a positive even integer");
    if (tc.total != ts.total)
        ERR(ctx, "llm_rope_nd: cos and sin must have the same shape");
    if (tx.total % tc.total != 0)
        ERR(ctx, "llm_rope_nd: x.total must be a multiple of cos.total");

    int32_t n_rows_x = tx.total / D;
    int32_t n_rows_cs = tc.total / D;
    int32_t ratio = n_rows_x / n_rows_cs;
    int32_t half = D / 2;

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, tx.shape, tx.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    for (int32_t r = 0; r < n_rows_x; r++) {
        const float *xr = tx.data + r * D;
        const float *cr = tc.data + (r / ratio) * D;
        const float *sr = ts.data + (r / ratio) * D;
        float *dr = dst + r * D;
        for (int32_t j = 0; j < half; j++) {
            float xj = xr[j];
            float xj_half = xr[j + half];
            dr[j] = xj * cr[j] - xj_half * sr[j];
            dr[j + half] = xj_half * cr[j + half] + xj * sr[j + half];
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_exp_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_exp_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = expf(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_cos_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_cos_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = cosf(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_sin_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_sin_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = sinf(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_tanh_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_tanh_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = tanhf(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_pow_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_pow_nd: invalid blob");
    float e = (float)sqlite3_value_double(argv[1]);
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = powf(t.data[i], e);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_abs_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_abs_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = fabsf(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_sqrt_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_sqrt_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = sqrtf(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_log_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_log_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = logf(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_sigmoid_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_sigmoid_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = llm_sigmoid(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_relu_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_relu_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = t.data[i] > 0.0f ? t.data[i] : 0.0f;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_gelu_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_gelu_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = llm_gelu(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_floor_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_floor_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = floorf(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_ceil_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_ceil_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = ceilf(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_round_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_round_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = roundf(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_trunc_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_trunc_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = truncf(t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_sign_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_sign_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = (t.data[i] > 0.0f) ? 1.0f : (t.data[i] < 0.0f) ? -1.0f : 0.0f;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

void sql_clamp_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_clamp_nd: invalid blob");
    int has_min = sqlite3_value_type(argv[1]) != SQLITE_NULL;
    int has_max = sqlite3_value_type(argv[2]) != SQLITE_NULL;
    float lo = has_min ? (float)sqlite3_value_double(argv[1]) : 0.0f;
    float hi = has_max ? (float)sqlite3_value_double(argv[2]) : 0.0f;
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++) {
        float v = t.data[i];
        if (has_min && v < lo)
            v = lo;
        if (has_max && v > hi)
            v = hi;
        dst[i] = v;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}