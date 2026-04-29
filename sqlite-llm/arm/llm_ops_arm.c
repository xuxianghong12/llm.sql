/*
 * llm_ops_arm.c - SQLite Extension: LLM Tensor Operations (ARM baseline)
 *
 * Implements fundamental tensor operations required for Large Language Model
 * (e.g. Qwen2.5, LLaMA, GPT-2) forward-pass inference.
 *
 * Naming convention
 * -----------------
 *   llm_<op>            single implementation (SIMD auto-selected)
 *   llm_<op>_simd       compatibility entry point that currently maps to the
 *                      portable scalar/reference path in the initial ARM port
 * available) llm_<op>_blas       compatibility entry points backed by the local
 *                      CBLAS shim in this repository
 *
 * SQL function catalogue
 * ----------------------
 *  Construct/Inspect: llm_vec(n,val)  llm_mat(R,C,val)
 *                     llm_len(v)  llm_rows(m)  llm_cols(m)
 *                     llm_vget(v,i)  llm_mget(m,r,c)  llm_str(blob)
 *  Arithmetic:        llm_add_{simd,blas}(a,b)
 *                     llm_sub_simd(a,b)  llm_div_simd(a,b)
 *                     llm_mul_{simd,blas}(a,b)  [Hadamard]
 *                     llm_scale_{simd,blas}(a,alpha)
 *                     llm_dot_{simd,blas}(a,b)
 *  Element-wise math: llm_neg(x)  llm_abs(x)  llm_sqrt(x)
 *                     llm_exp(x)  llm_log(x)  llm_clip(x,lo,hi)
 *  Activations:       llm_relu(x)  llm_gelu(x)  llm_silu(x)
 *                     llm_sigmoid(x)  llm_softmax(x)
 *  Reductions:        llm_sum(x)  llm_mean(x)  llm_max(x)  llm_min(x)
 *  Normalization:     llm_rmsnorm(x,w,eps)  llm_layernorm(x,w,bias,eps)
 *  Matrix ops:        llm_gemm_{simd,blas}(A,B)
 *                     llm_matadd(A,B)  llm_transpose(A)
 *                     llm_matvec_{simd,blas}(A,x)
 *  Shape/indexing:    llm_reshape(blob,rows,cols)  llm_flatten(m)
 *                     llm_slice(v,start,end)  llm_concat(a,b)
 *                     llm_mrow(m,r)  llm_gather(m,idx_vec)
 *  Transformer:       llm_rope(x,cos_row,sin_row,head_dim)
 *
 * Build:  make arm                 -> llm_ops_arm.so
 *         make test-arm            -> run ARM parity test suite
 *         make LLM_NUM_THREADS=8 arm -> override thread count (default 4)
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_ops_arm_internal.h"
SQLITE_EXTENSION_INIT1

/* Default thread count for int8 parallel ops (overridden via
 * -DLLM_NUM_THREADS). Can be changed at runtime via llm_set_threads(n) SQL
 * function. */
int g_llm_num_threads = LLM_NUM_THREADS;

/* Minimum N*K work (int8×fp32 multiply-adds) before spawning threads for
 * the decode (M=1) path.  Thread creation/join costs ~20-100 µs per thread;
 * single-threaded scalar code can still handle smaller decode workloads, so
 * threading only pays off when total work >> 1M.  2M ≈ hidden_size ≥ 1024. */

/* Shared helper kernels and legacy blob wrappers live in split files:
 *   - llm_ops_arm_common.c
 *   - llm_ops_arm_blob.c
 */

/* ===========================================================================
 * Section 7: Element-wise math + activations
 * =========================================================================*/

/* ===========================================================================
 * Section 8: Reductions
 * =========================================================================*/

/* ===========================================================================
 * Section 9: Normalization
 * =========================================================================*/

/* ===========================================================================
 * Section 10: Matrix operations
 * =========================================================================*/

/* ===========================================================================
 * Section 11: Shape / Indexing
 * =========================================================================*/

/* ===========================================================================
 * Section 12: Transformer -- RoPE
 * =========================================================================*/

/* ===========================================================================
 * Section 13: N-D Tensor operations
 *
 * All functions in this section accept any supported blob format (legacy 1-D
 * vector, legacy 2-D matrix, or N-D) via the llm_to_nd() helper in llm_ops.h.
 * Output blobs always use the new N-D format (LLM_ND_MAGIC header).
 *
 * Supported SQL functions:
 *   llm_ndim(blob)                  → INT    number of dimensions
 *   llm_shape(blob, dim)            → INT    size of dimension dim
 *   llm_view_nd(blob, d0[,d1,...])  → BLOB   reshape (variadic, ≤8 dims)
 *   llm_permute_nd(blob, p0,p1,...) → BLOB   permute axes (variadic, ≤8)
 *   llm_transpose_nd(blob, d0, d1) → BLOB   swap two axes
 *   llm_unsqueeze_nd(blob, dim)     → BLOB   insert unit axis
 *   llm_squeeze_nd(blob, dim)       → BLOB   remove unit axis
 *   llm_cat_nd(a, b, dim)          → BLOB   concatenate along axis
 *   llm_slice_nd(blob, dim, s, e [, step]) → BLOB   slice [s,e) along axis
 *   llm_expand_nd(blob, d0[,...])  → BLOB   broadcast-expand to shape
 *   llm_bmm(A, B)                  → BLOB   batch matrix multiply
 *   llm_add_nd(a, b)               → BLOB   element-wise add (broadcast)
 *   llm_mul_nd(a, b)               → BLOB   element-wise mul (broadcast)
 *   llm_sub_nd(a, b)               → BLOB   element-wise sub (broadcast)
 *   llm_softmax_nd(blob, dim)      → BLOB   softmax along axis
 *   llm_mean_nd(blob, dim, keepdim)→ BLOB   mean along axis
 *   llm_sum_nd(blob, dim, keepdim) → BLOB   sum along axis
 *   llm_str_nd(blob)               → TEXT   human-readable shape + excerpt
 *   llm_div_nd(a, b)               → BLOB   element-wise div (broadcast)
 *   llm_neg_nd(blob)               → BLOB   element-wise negation
 *   llm_scale_nd(blob, scalar)     → BLOB   multiply all by scalar
 *   llm_rsqrt_nd(blob)             → BLOB   element-wise 1/sqrt(x)
 *   llm_silu_nd(blob)              → BLOB   element-wise SiLU activation
 *   llm_silu_mul_nd(gate, up)      → BLOB   fused silu(gate) * up
 *   llm_rope_nd(x, cos, sin)      → BLOB   fused RoPE: x*cos +
 * rotate_half(x)*sin llm_exp_nd(blob)               → BLOB   element-wise exp
 *   llm_cos_nd(blob)               → BLOB   element-wise cosine
 *   llm_sin_nd(blob)               → BLOB   element-wise sine
 *   llm_tanh_nd(blob)              → BLOB   element-wise tanh
 *   llm_pow_nd(blob, exponent)     → BLOB   element-wise power
 *   llm_abs_nd(blob)               → BLOB   element-wise absolute value
 *   llm_sqrt_nd(blob)              → BLOB   element-wise square root
 *   llm_embedding_nd(weight, idx)  → BLOB   embedding lookup
 *   llm_rmsnorm_nd(x, w, eps)      → BLOB   RMS normalization
 *   llm_triu_nd(blob, diagonal)    → BLOB   upper triangular mask (2-D)
 *   llm_where_nd(cond, x, y)       → BLOB   element-wise conditional
 *   llm_masked_fill_nd(x, m, val)  → BLOB   masked fill with scalar
 *   llm_arange_nd(n)               → BLOB   1-D tensor [0..n-1]
 *   llm_full_nd(val, d0, ...)      → BLOB   tensor filled with val
 *   llm_argmax_nd(blob, dim)       → BLOB   argmax along dimension
 * =========================================================================*/

/* N-D layout and indexing wrappers live in llm_ops_arm_nd_layout.c. */
/* N-D algebra and broadcast wrappers live in llm_ops_arm_nd_algebra.c. */
/* Shared N-D helpers live in llm_ops_arm_common.c. */
/* N-D softmax, aggregate, and debug-string wrappers live in llm_ops_arm_nd_stats.c. */

/* N-D divide and comparison wrappers live in llm_ops_arm_nd_algebra.c. */

/* N-D unary and pointwise transforms live in llm_ops_arm_nd_pointwise.c. */

/* N-D logic, remainder, and inspection wrappers live in llm_ops_arm_nd_utility.c. */

/* N-D reduction and value-selection wrappers live in llm_ops_arm_nd_reduce.c. */

/* N-D normalization and model wrappers live in llm_ops_arm_nd_model.c. */

/* N-D masking, factory, and utility wrappers live in llm_ops_arm_nd_utility.c. */

/* ===========================================================================
 * Section 13b: Additional N-D ops (cumsum, gather, index_select, repeat, etc.)
 * =========================================================================*/

/* N-D cumulative scan and indexed selection wrappers live in llm_ops_arm_nd_sequence.c. */

/* N-D scatter, repeat, and reorder wrappers live in llm_ops_arm_nd_sequence.c. */

/* N-D indexed update and structural reorder wrappers live in llm_ops_arm_nd_sequence.c. */

/* N-D norms, masks, and attention wrappers live in llm_ops_arm_nd_model.c. */

/* INT8 quantization and parameter-backed inference wrappers live in llm_ops_arm_int8.c. */

#define REG(name, narg, fn)                                                    \
    do {                                                                       \
        int rc = sqlite3_create_function(db,                                   \
                                         name,                                 \
                                         narg,                                 \
                                         SQLITE_UTF8 | SQLITE_DETERMINISTIC,   \
                                         NULL,                                 \
                                         fn,                                   \
                                         NULL,                                 \
                                         NULL);                                \
        if (rc != SQLITE_OK)                                                   \
            return rc;                                                         \
    } while (0)

#ifdef _WIN32
__declspec(dllexport)
#elif defined(__GNUC__)
__attribute__((visibility("default")))
#endif
int sqlite3_llm_ops_init(sqlite3 *db,
                         char **pzErrMsg,
                         const sqlite3_api_routines *pApi) {
    (void)pzErrMsg;
    SQLITE_EXTENSION_INIT2(pApi);

    int rc =
        llm_register_lookup_function(db, "llm_get_param", 1, sql_get_param);
    if (rc != SQLITE_OK)
        return rc;
    rc = llm_register_lookup_function(
        db, "llm_linear_param_nd", 2, sql_linear_param_nd);
    if (rc != SQLITE_OK)
        return rc;
    rc = llm_register_lookup_function(
        db, "llm_linear_param_bias_nd", 3, sql_linear_param_bias_nd);
    if (rc != SQLITE_OK)
        return rc;
    rc = llm_register_lookup_function(
        db, "llm_embedding_param_nd", 2, sql_embedding_param_nd);
    if (rc != SQLITE_OK)
        return rc;

    /* §5 Construct / Inspect */
    REG("llm_vec", 2, sql_vec);
    REG("llm_mat", 3, sql_mat);
    REG("llm_vget", 2, sql_vget);
    REG("llm_mget", 3, sql_mget);
    REG("llm_len", 1, sql_len);
    REG("llm_rows", 1, sql_rows);
    REG("llm_cols", 1, sql_cols);
    REG("llm_str", 1, sql_str);

    /* §6 Arithmetic */
    REG("llm_add_simd", 2, sql_add_simd);
    REG("llm_add_blas", 2, sql_add_blas);
    REG("llm_sub_simd", 2, sql_sub_simd);
    REG("llm_mul_simd", 2, sql_mul_simd);
    REG("llm_mul_blas", 2, sql_mul_blas);
    REG("llm_div_simd", 2, sql_div_simd);
    REG("llm_scale_simd", 2, sql_scale_simd);
    REG("llm_scale_blas", 2, sql_scale_blas);
    REG("llm_dot_simd", 2, sql_dot_simd);
    REG("llm_dot_blas", 2, sql_dot_blas);

    /* §7 Element-wise math + activations */
    REG("llm_neg", 1, sql_neg);
    REG("llm_abs", 1, sql_abs);
    REG("llm_sqrt", 1, sql_sqrt);
    REG("llm_exp", 1, sql_exp);
    REG("llm_log", 1, sql_log);
    REG("llm_clip", 3, sql_clip);
    REG("llm_relu", 1, sql_relu);
    REG("llm_gelu", 1, sql_gelu);
    REG("llm_silu", 1, sql_silu);
    REG("llm_sigmoid", 1, sql_sigmoid);
    REG("llm_softmax", 1, sql_softmax);

    /* §8 Reductions */
    REG("llm_sum", 1, sql_sum);
    REG("llm_mean", 1, sql_mean);
    REG("llm_max", 1, sql_max);
    REG("llm_min", 1, sql_min);

    /* §9 Normalization */
    REG("llm_rmsnorm", 3, sql_rmsnorm);
    REG("llm_layernorm", 4, sql_layernorm);

    /* §10 Matrix */
    REG("llm_gemm_simd", 2, sql_gemm_simd);
    REG("llm_gemm_blas", 2, sql_gemm_blas);
    REG("llm_matadd", 2, sql_matadd);
    REG("llm_transpose", 1, sql_transpose);
    REG("llm_matvec_simd", 2, sql_matvec_simd);
    REG("llm_matvec_blas", 2, sql_matvec_blas);

    /* §11 Shape / Indexing */
    REG("llm_reshape", 3, sql_reshape);
    REG("llm_flatten", 1, sql_flatten);
    REG("llm_slice", 3, sql_slice);
    REG("llm_concat", 2, sql_concat);
    REG("llm_mrow", 2, sql_mrow);
    REG("llm_gather", 2, sql_gather);

    /* §12 Transformer */
    REG("llm_rope", 4, sql_rope);

    /* §13 N-D tensor operations */
    REG("llm_ndim", 1, sql_ndim);
    REG("llm_shape", 2, sql_shape);
    REG("llm_view_nd", -1, sql_view_nd);
    REG("llm_view_t12_nd", -1, sql_view_t12_nd);
    REG("llm_permute_nd", -1, sql_permute_nd);
    REG("llm_transpose_nd", 3, sql_transpose_nd);
    REG("llm_unsqueeze_nd", 2, sql_unsqueeze_nd);
    REG("llm_squeeze_nd", 2, sql_squeeze_nd);
    REG("llm_cat_nd", 3, sql_cat_nd);
    REG("llm_slice_nd", -1, sql_slice_nd);
    REG("llm_select_nd", 3, sql_select_nd);
    REG("llm_expand_nd", -1, sql_expand_nd);
    REG("llm_bmm", 2, sql_bmm);
    REG("llm_linear_nd", 2, sql_linear_nd);
    REG("llm_add_nd", 2, sql_add_nd);
    REG("llm_mul_nd", 2, sql_mul_nd);
    REG("llm_sub_nd", 2, sql_sub_nd);
    REG("llm_gt_nd", 2, sql_gt_nd);
    REG("llm_lt_nd", 2, sql_lt_nd);
    REG("llm_ge_nd", 2, sql_ge_nd);
    REG("llm_le_nd", 2, sql_le_nd);
    REG("llm_eq_nd", 2, sql_eq_nd);
    REG("llm_ne_nd", 2, sql_ne_nd);
    REG("llm_softmax_nd", 2, sql_softmax_nd);
    REG("llm_mean_nd", 3, sql_mean_nd);
    REG("llm_sum_nd", 3, sql_sum_nd);
    REG("llm_str_nd", 1, sql_str_nd);
    REG("llm_div_nd", 2, sql_div_nd);
    REG("llm_neg_nd", 1, sql_neg_nd);
    REG("llm_scale_nd", 2, sql_scale_nd);
    REG("llm_rsqrt_nd", 1, sql_rsqrt_nd);
    REG("llm_silu_nd", 1, sql_silu_nd);
    REG("llm_silu_mul_nd", 2, sql_silu_mul_nd);
    REG("llm_rope_nd", 3, sql_rope_nd);
    REG("llm_exp_nd", 1, sql_exp_nd);
    REG("llm_cos_nd", 1, sql_cos_nd);
    REG("llm_sin_nd", 1, sql_sin_nd);
    REG("llm_tanh_nd", 1, sql_tanh_nd);
    REG("llm_pow_nd", 2, sql_pow_nd);
    REG("llm_abs_nd", 1, sql_abs_nd);
    REG("llm_sqrt_nd", 1, sql_sqrt_nd);
    REG("llm_embedding_nd", 2, sql_embedding_nd);
    REG("llm_rmsnorm_nd", 3, sql_rmsnorm_nd);
    REG("llm_triu_nd", 2, sql_triu_nd);
    REG("llm_where_nd", 3, sql_where_nd);
    REG("llm_masked_fill_nd", 3, sql_masked_fill_nd);
    REG("llm_arange_nd", 1, sql_arange_nd);
    REG("llm_full_nd", -1, sql_full_nd);
    REG("llm_argmax_nd", 2, sql_argmax_nd);

    /* ── New N-D element-wise / reduction / logical functions ─────────── */
    REG("llm_log_nd", 1, sql_log_nd);
    REG("llm_sigmoid_nd", 1, sql_sigmoid_nd);
    REG("llm_relu_nd", 1, sql_relu_nd);
    REG("llm_gelu_nd", 1, sql_gelu_nd);
    REG("llm_floor_nd", 1, sql_floor_nd);
    REG("llm_ceil_nd", 1, sql_ceil_nd);
    REG("llm_round_nd", 1, sql_round_nd);
    REG("llm_trunc_nd", 1, sql_trunc_nd);
    REG("llm_sign_nd", 1, sql_sign_nd);
    REG("llm_clamp_nd", 3, sql_clamp_nd);
    REG("llm_tril_nd", 2, sql_tril_nd);
    REG("llm_logical_not_nd", 1, sql_logical_not_nd);
    REG("llm_logical_and_nd", 2, sql_logical_and_nd);
    REG("llm_logical_or_nd", 2, sql_logical_or_nd);
    REG("llm_bitwise_not_nd", 1, sql_bitwise_not_nd);
    REG("llm_bitwise_and_nd", 2, sql_bitwise_and_nd);
    REG("llm_bitwise_or_nd", 2, sql_bitwise_or_nd);
    REG("llm_bitwise_and_scalar_nd", 2, sql_bitwise_and_scalar_nd);
    REG("llm_pow_scalar_nd", 2, sql_pow_scalar_nd);
    REG("llm_floor_div_nd", 2, sql_floor_div_nd);
    REG("llm_remainder_nd", 2, sql_remainder_nd);
    REG("llm_remainder_tensor_nd", 2, sql_remainder_tensor_nd);
    REG("llm_reduce_sum_nd", 1, sql_reduce_sum_nd);
    REG("llm_reduce_mean_nd", 1, sql_reduce_mean_nd);
    REG("llm_reduce_max_nd", 1, sql_reduce_max_nd);
    REG("llm_reduce_min_nd", 1, sql_reduce_min_nd);
    REG("llm_reduce_argmax_nd", 1, sql_reduce_argmax_nd);
    REG("llm_item_nd", 2, sql_item_nd);
    REG("llm_max_dim_nd", 3, sql_max_dim_nd);
    REG("llm_argmax_dim_nd", 3, sql_argmax_dim_nd);
    REG("llm_layernorm_nd", 4, sql_layernorm_nd);
    REG("llm_cumsum_nd", 2, sql_cumsum_nd);
    REG("llm_cumprod_nd", 2, sql_cumprod_nd);
    REG("llm_gather_nd", 3, sql_gather_nd);
    REG("llm_index_select_nd", 3, sql_index_select_nd);
    REG("llm_scatter_nd", 4, sql_scatter_nd);
    REG("llm_repeat_nd", -1, sql_repeat_nd);
    REG("llm_repeat_kv_nd", 2, sql_repeat_kv_nd);
    REG("llm_flip_nd", 2, sql_flip_nd);
    REG("llm_roll_nd", 3, sql_roll_nd);
    REG("llm_repeat_interleave_nd", 3, sql_repeat_interleave_nd);
    REG("llm_norm_nd", 4, sql_norm_nd);
    REG("llm_index_put_nd", -1, sql_index_put_nd);
    REG("llm_repeat_interleave_tensor_nd", 3, sql_repeat_interleave_tensor_nd);
    REG("llm_squeeze_all_nd", 1, sql_squeeze_all_nd);
    REG("llm_flatten_nd", 3, sql_flatten_nd);
    REG("llm_cat_multi_nd", -1, sql_cat_multi_nd);
    REG("llm_stack_nd", -1, sql_stack_nd);
    REG("llm_bool_to_additive_mask_nd", 2, sql_bool_to_additive_mask_nd);
    REG("llm_numel_nd", 1, sql_numel_nd);
    REG("llm_sdpa_nd", -1, sql_sdpa_nd);

    /* §14 INT8 quantization */
    REG("llm_quantize_int8_nd", 1, sql_quantize_int8_nd);
    REG("llm_dequantize_int8_nd", 1, sql_dequantize_int8_nd);
    REG("llm_linear_int8_nd", 2, sql_linear_int8_nd);
    REG("llm_linear_int8_bias_nd", 3, sql_linear_int8_bias_nd);
    REG("llm_embedding_int8_nd", 2, sql_embedding_int8_nd);
    REG("llm_is_int8_nd", 1, sql_is_int8_nd);

    /* §15 Runtime configuration */
    REG("llm_set_threads", 1, sql_set_threads);
    REG("llm_get_threads", 0, sql_get_threads);

    return SQLITE_OK;
}

/*
 * sqlite3_llmops_init — alias for sqlite3_llm_ops_init.
 *
 * When Python's sqlite3 module calls load_extension() WITHOUT an explicit
 * entrypoint (Python < 3.12), SQLite derives the entry-point name from the
 * shared-library filename by stripping underscores:
 *   llm_ops.so  →  sqlite3_llmops_init   (underscores stripped from "llm_ops")
 *
 * Exporting this alias makes the extension load correctly on Python 3.10/3.11
 * without requiring the caller to specify entrypoint='sqlite3_llm_ops_init'.
 */
#ifdef _WIN32
__declspec(dllexport)
#elif defined(__GNUC__)
__attribute__((visibility("default")))
#endif
int sqlite3_llmops_init(sqlite3 *db,
                        char **pzErrMsg,
                        const sqlite3_api_routines *pApi) {
    return sqlite3_llm_ops_init(db, pzErrMsg, pApi);
}
