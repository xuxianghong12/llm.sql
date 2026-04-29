/*
 * llm_ops_arm_internal.h - Shared internal interfaces for split arm ops
 *
 * This header centralizes the private macros, helper declarations, and
 * cross-file SQL wrapper prototypes used by the arm operator implementation.
 * Add declarations here only when a symbol is intentionally shared across
 * multiple translation units; keep file-local helpers static inside their
 * owning source file when no reuse is needed.
 */
#ifndef LLM_OPS_ARM_INTERNAL_H
#define LLM_OPS_ARM_INTERNAL_H

#include "sqlite3ext.h"
#include "llm_ops.h"

#include <assert.h>
#include <float.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cblas.h"

#ifndef LLM_NUM_THREADS
#define LLM_NUM_THREADS 4
#endif

#define LLM_INT8_THREAD_OPS_MIN (2 * 1024 * 1024)

#define ERR(ctx, msg)                                                          \
    do {                                                                       \
        sqlite3_result_error((ctx), (msg), -1);                                \
        return;                                                                \
    } while (0)

#define PARSE_2VEC(ctx, argv, va, vb)                                          \
    VecF32 va, vb;                                                             \
    do {                                                                       \
        if (sqlite3_value_type((argv)[0]) == SQLITE_NULL ||                    \
            sqlite3_value_type((argv)[1]) == SQLITE_NULL)                      \
            ERR((ctx), "NULL argument");                                       \
        if (llm_parse_vec(sqlite3_value_blob((argv)[0]),                       \
                          sqlite3_value_bytes((argv)[0]),                      \
                          &(va)) < 0)                                          \
            ERR((ctx), "Invalid vector blob (arg 0)");                         \
        if (llm_parse_vec(sqlite3_value_blob((argv)[1]),                       \
                          sqlite3_value_bytes((argv)[1]),                      \
                          &(vb)) < 0)                                          \
            ERR((ctx), "Invalid vector blob (arg 1)");                         \
        if ((va).n != (vb).n)                                                  \
            ERR((ctx), "Vector size mismatch");                                \
    } while (0)

#define PARSE_2MAT(ctx, argv, ma, mb)                                          \
    MatF32 ma, mb;                                                             \
    do {                                                                       \
        if (llm_parse_mat(sqlite3_value_blob((argv)[0]),                       \
                          sqlite3_value_bytes((argv)[0]),                      \
                          &(ma)) < 0)                                          \
            ERR((ctx), "Invalid matrix blob (arg 0)");                         \
        if (llm_parse_mat(sqlite3_value_blob((argv)[1]),                       \
                          sqlite3_value_bytes((argv)[1]),                      \
                          &(mb)) < 0)                                          \
            ERR((ctx), "Invalid matrix blob (arg 1)");                         \
    } while (0)

#define EMIT_VEC_BINOP(ctx, va, kernel_call)                                   \
    do {                                                                       \
        float *tmp = sqlite3_malloc((int)((va).n * sizeof(float)));            \
        if (!tmp) {                                                            \
            sqlite3_result_error_nomem(ctx);                                   \
            return;                                                            \
        }                                                                      \
        kernel_call;                                                           \
        int out_sz;                                                            \
        void *out = alloc_vec_blob((va).n, tmp, &out_sz);                      \
        sqlite3_free(tmp);                                                     \
        if (!out) {                                                            \
            sqlite3_result_error_nomem(ctx);                                   \
            return;                                                            \
        }                                                                      \
        sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);                   \
    } while (0)

#define DEFINE_UNARY_VEC(sql_name, kernel)                                     \
    static void sql_name(                                                      \
        sqlite3_context *ctx, int argc, sqlite3_value **argv) {                \
        (void)argc;                                                            \
        VecF32 va;                                                             \
        if (llm_parse_vec(sqlite3_value_blob(argv[0]),                         \
                          sqlite3_value_bytes(argv[0]),                        \
                          &va) < 0)                                            \
            ERR(ctx, "Invalid vector blob");                                   \
        EMIT_VEC_BINOP(ctx, va, kernel(va.data, tmp, va.n));                   \
    }

typedef struct llm_param_lookup_t llm_param_lookup_t;

void *alloc_vec_blob(int n, const float *data, int *sz_out);
void *alloc_mat_blob(int rows, int cols, const float *data, int *sz_out);
void nd_strides(const int32_t *shape, int ndim, int32_t *strides);
void *nd_result_alloc(int ndim,
                      const int32_t *shape,
                      int32_t total,
                      int *sz_out);
int32_t broadcast_shape(const TensorNDF32 *ta,
                        const TensorNDF32 *tb,
                        int32_t *out_shape,
                        int *out_ndim);

extern int g_llm_num_threads;

void k_add_simd(const float *a, const float *b, float *c, int n);
float k_dot_simd(const float *a, const float *b, int n);

void sql_vec(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_mat(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_vget(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_mget(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_len(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_rows(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_cols(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_str(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_add_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_add_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_sub_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_mul_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_mul_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_div_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_scale_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_scale_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_dot_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_dot_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv);

void sql_neg(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_abs(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_sqrt(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_exp(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_log(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_clip(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_relu(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_gelu(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_silu(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_sigmoid(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_softmax(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_sum(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_mean(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_max(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_min(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_rmsnorm(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_layernorm(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_reshape(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_flatten(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_slice(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_concat(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_mrow(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_gather(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_gemm_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_gemm_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_matadd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_transpose(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_matvec_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_matvec_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_rope(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_set_threads(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_get_threads(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_get_param(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_ndim(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_shape(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_view_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_view_t12_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_permute_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_transpose_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_unsqueeze_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_squeeze_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_cat_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_slice_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_select_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_expand_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_softmax_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_mean_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_sum_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_str_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_bmm(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_linear_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_add_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_mul_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_sub_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_div_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_gt_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_lt_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_ge_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_le_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_eq_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_ne_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_neg_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_scale_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_rsqrt_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_silu_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_silu_mul_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_rope_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_exp_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_cos_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_sin_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_tanh_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_pow_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_abs_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_sqrt_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_log_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_sigmoid_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_relu_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_gelu_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_floor_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_ceil_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_round_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_trunc_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_sign_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_clamp_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_reduce_sum_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_reduce_mean_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_reduce_max_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_reduce_min_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_item_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_reduce_argmax_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_max_dim_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_argmax_dim_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_triu_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_where_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_masked_fill_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_arange_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_full_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_argmax_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_tril_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_logical_not_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_logical_and_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_logical_or_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_bitwise_not_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_bitwise_and_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_bitwise_or_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_bitwise_and_scalar_nd(sqlite3_context *ctx,
                               int argc,
                               sqlite3_value **argv);
void sql_pow_scalar_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_floor_div_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_remainder_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_remainder_tensor_nd(sqlite3_context *ctx,
                             int argc,
                             sqlite3_value **argv);
void sql_numel_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_cumsum_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_cumprod_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_gather_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_index_select_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_scatter_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_repeat_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_repeat_kv_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_flip_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_roll_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_repeat_interleave_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_index_put_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_repeat_interleave_tensor_nd(sqlite3_context *ctx,
                                     int argc,
                                     sqlite3_value **argv);
void sql_squeeze_all_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_flatten_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_cat_multi_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_stack_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_layernorm_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_embedding_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_rmsnorm_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_norm_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_bool_to_additive_mask_nd(sqlite3_context *ctx,
                                  int argc,
                                  sqlite3_value **argv);
void sql_sdpa_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_quantize_int8_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_dequantize_int8_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_linear_int8_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_linear_int8_bias_nd(sqlite3_context *ctx,
                             int argc,
                             sqlite3_value **argv);
void sql_linear_param_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);
void sql_linear_param_bias_nd(sqlite3_context *ctx,
                              int argc,
                              sqlite3_value **argv);
void sql_embedding_int8_nd(sqlite3_context *ctx,
                           int argc,
                           sqlite3_value **argv);
void sql_embedding_param_nd(sqlite3_context *ctx,
                            int argc,
                            sqlite3_value **argv);
void sql_is_int8_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv);

int llm_register_lookup_function(sqlite3 *db,
                                 const char *name,
                                 int narg,
                                 void (*fn)(sqlite3_context *,
                                            int,
                                            sqlite3_value **));
int llm_param_lookup_step_blob(llm_param_lookup_t *lookup,
                               sqlite3 *db,
                               sqlite3_context *ctx,
                               const char *name,
                               const void **out_blob,
                               int *out_size);
void llm_param_lookup_release(llm_param_lookup_t *lookup);

#endif