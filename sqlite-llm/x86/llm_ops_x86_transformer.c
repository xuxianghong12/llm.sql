/*
 * llm_ops_x86_transformer.c - Transformer-specific legacy vector wrappers
 *
 * This file owns specialized transformer math that does not fit the generic
 * elementwise or matrix categories. Today that mainly means the legacy vector
 * RoPE wrapper used by the older SQL surface.
 *
 * Maintenance guidance:
 *   - Keep model-specific assumptions documented near the wrapper because this
 *     file is the exception bucket for transformer-specific kernels.
 *   - If a transformer helper grows N-D tensor semantics, move it into an N-D
 *     split file instead of duplicating logic across both APIs.
 *   - Validate dimensional constraints before allocating output blobs.
 */
#include "llm_ops_x86_internal.h"
SQLITE_EXTENSION_INIT3

static void k_rope_simd(
    const float *x, const float *cv, const float *sv, float *out, int hd) {
    int half = hd / 2;
#if LLM_HAVE_AVX2
    int i = 0;
    for (; i <= half - 4; i += 4) {
        __m128 xlo = _mm_loadu_ps(x + 2 * i), xhi = _mm_loadu_ps(x + 2 * i + 4);
        __m128 xe = _mm_shuffle_ps(xlo, xhi, _MM_SHUFFLE(2, 0, 2, 0));
        __m128 xo = _mm_shuffle_ps(xlo, xhi, _MM_SHUFFLE(3, 1, 3, 1));
        __m128 vc = _mm_loadu_ps(cv + i), vs = _mm_loadu_ps(sv + i);
        __m128 re = _mm_sub_ps(_mm_mul_ps(xe, vc), _mm_mul_ps(xo, vs));
        __m128 ro = _mm_add_ps(_mm_mul_ps(xe, vs), _mm_mul_ps(xo, vc));
        _mm_storeu_ps(out + 2 * i, _mm_unpacklo_ps(re, ro));
        _mm_storeu_ps(out + 2 * i + 4, _mm_unpackhi_ps(re, ro));
    }
    for (; i < half; i++) {
        float xe = x[2 * i], xo = x[2 * i + 1];
        out[2 * i] = xe * cv[i] - xo * sv[i];
        out[2 * i + 1] = xe * sv[i] + xo * cv[i];
    }
#else
    for (int i = 0; i < half; i++) {
        float xe = x[2 * i], xo = x[2 * i + 1];
        out[2 * i] = xe * cv[i] - xo * sv[i];
        out[2 * i + 1] = xe * sv[i] + xo * cv[i];
    }
#endif
}

void sql_rope(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 vx, vc, vs;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &vx) < 0)
        ERR(ctx, "Invalid x blob");
    if (llm_parse_vec(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &vc) < 0)
        ERR(ctx, "Invalid cos_row blob");
    if (llm_parse_vec(
            sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &vs) < 0)
        ERR(ctx, "Invalid sin_row blob");
    int hd = sqlite3_value_int(argv[3]);
    if (hd <= 0 || hd % 2 != 0)
        ERR(ctx, "head_dim must be a positive even integer");
    if (vx.n != hd)
        ERR(ctx, "x.n != head_dim");
    if (vc.n != hd / 2)
        ERR(ctx, "cos_row.n != head_dim/2");
    if (vs.n != hd / 2)
        ERR(ctx, "sin_row.n != head_dim/2");
    float *tmp = sqlite3_malloc((int)(hd * sizeof(float)));
    if (!tmp) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    k_rope_simd(vx.data, vc.data, vs.data, tmp, hd);
    int out_sz;
    void *out = alloc_vec_blob(hd, tmp, &out_sz);
    sqlite3_free(tmp);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}