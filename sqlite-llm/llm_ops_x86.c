/*
 * llm_ops_x86.c - SQLite Extension: LLM Tensor Operations (x86 / AVX2+FMA)
 *
 * Implements fundamental tensor operations required for Large Language Model
 * (e.g. Qwen2.5, LLaMA, GPT-2) forward-pass inference.
 *
 * Naming convention
 * -----------------
 *   llm_<op>            single implementation (SIMD auto-selected)
 *   llm_<op>_simd       explicit AVX2+FMA path (scalar fallback if not
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
 * Build:  make                     -> llm_ops.so
 *         make LLM_NUM_THREADS=8  -> override thread count (default 4)
 *         make test                -> run test suite
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include "llm_ops.h"

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>

#include "cblas.h"

/* Default thread count for int8 parallel ops (overridden via
 * -DLLM_NUM_THREADS). Can be changed at runtime via llm_set_threads(n) SQL
 * function. */
#ifndef LLM_NUM_THREADS
#define LLM_NUM_THREADS 4
#endif
static int g_llm_num_threads = LLM_NUM_THREADS;

/* Minimum N*K work (int8×fp32 multiply-adds) before spawning threads for
 * the decode (M=1) path.  Thread creation/join costs ~20-100 µs per thread;
 * single-threaded AVX2 SIMD can process ~1M int8×fp32 ops in ~100 µs, so
 * threading only pays off when total work >> 1M.  2M ≈ hidden_size ≥ 1024. */
#define LLM_INT8_THREAD_OPS_MIN (2 * 1024 * 1024)

/* ===========================================================================
 * Section 1: SIMD kernel implementations
 * =========================================================================*/

static void k_add_simd(const float *a, const float *b, float *c, int n) {
#if LLM_HAVE_AVX2
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(
            c + i,
            _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++)
        c[i] = a[i] + b[i];
#else
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
#endif
}

static void k_sub_simd(const float *a, const float *b, float *c, int n) {
#if LLM_HAVE_AVX2
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(
            c + i,
            _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++)
        c[i] = a[i] - b[i];
#else
    for (int i = 0; i < n; i++)
        c[i] = a[i] - b[i];
#endif
}

static void k_mul_simd(const float *a, const float *b, float *c, int n) {
#if LLM_HAVE_AVX2
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(
            c + i,
            _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++)
        c[i] = a[i] * b[i];
#else
    for (int i = 0; i < n; i++)
        c[i] = a[i] * b[i];
#endif
}

static void k_div_simd(const float *a, const float *b, float *c, int n) {
#if LLM_HAVE_AVX2
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(
            c + i,
            _mm256_div_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++)
        c[i] = a[i] / b[i];
#else
    for (int i = 0; i < n; i++)
        c[i] = a[i] / b[i];
#endif
}

static float llm_cmp_mask_to_fp32(int cmp) {
    return cmp ? 1.0f : 0.0f;
}

static void k_scale_simd(const float *a, float alpha, float *b, int n) {
#if LLM_HAVE_AVX2
    __m256 va = _mm256_set1_ps(alpha);
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(b + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), va));
    for (; i < n; i++)
        b[i] = alpha * a[i];
#else
    for (int i = 0; i < n; i++)
        b[i] = alpha * a[i];
#endif
}

static float k_dot_simd(const float *a, const float *b, int n) {
#if LLM_HAVE_AVX2
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8)
        acc = _mm256_fmadd_ps(
            _mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    float s = llm_hsum256(acc);
    for (; i < n; i++)
        s += a[i] * b[i];
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
#endif
}

/* Unary element-wise */
static void k_neg_simd(const float *a, float *b, int n) {
#if LLM_HAVE_AVX2
    __m256 sign = _mm256_set1_ps(-0.0f);
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(b + i, _mm256_xor_ps(_mm256_loadu_ps(a + i), sign));
    for (; i < n; i++)
        b[i] = -a[i];
#else
    for (int i = 0; i < n; i++)
        b[i] = -a[i];
#endif
}

static void k_abs_simd(const float *a, float *b, int n) {
#if LLM_HAVE_AVX2
    __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(b + i, _mm256_and_ps(_mm256_loadu_ps(a + i), mask));
    for (; i < n; i++)
        b[i] = fabsf(a[i]);
#else
    for (int i = 0; i < n; i++)
        b[i] = fabsf(a[i]);
#endif
}

static void k_sqrt_simd(const float *a, float *b, int n) {
#if LLM_HAVE_AVX2
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(b + i, _mm256_sqrt_ps(_mm256_loadu_ps(a + i)));
    for (; i < n; i++)
        b[i] = sqrtf(a[i]);
#else
    for (int i = 0; i < n; i++)
        b[i] = sqrtf(a[i]);
#endif
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
#if LLM_HAVE_AVX2
    __m256 vlo = _mm256_set1_ps(lo), vhi = _mm256_set1_ps(hi);
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(
            b + i,
            _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(a + i), vlo), vhi));
    for (; i < n; i++) {
        float v = a[i];
        b[i] = v < lo ? lo : (v > hi ? hi : v);
    }
#else
    for (int i = 0; i < n; i++) {
        float v = a[i];
        b[i] = v < lo ? lo : (v > hi ? hi : v);
    }
#endif
}

static void k_relu_simd(const float *a, float *b, int n) {
#if LLM_HAVE_AVX2
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(b + i, _mm256_max_ps(_mm256_loadu_ps(a + i), zero));
    for (; i < n; i++)
        b[i] = a[i] > 0.0f ? a[i] : 0.0f;
#else
    for (int i = 0; i < n; i++)
        b[i] = a[i] > 0.0f ? a[i] : 0.0f;
#endif
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
#if LLM_HAVE_AVX2
    {
        __m256 vmx = _mm256_set1_ps(-FLT_MAX);
        int i = 0;
        for (; i <= n - 8; i += 8)
            vmx = _mm256_max_ps(vmx, _mm256_loadu_ps(a + i));
        __m128 lo = _mm256_castps256_ps128(vmx);
        __m128 hi = _mm256_extractf128_ps(vmx, 1);
        lo = _mm_max_ps(lo, hi);
        lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1)));
        lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 3, 2)));
        mx = _mm_cvtss_f32(lo);
        for (; i < n; i++)
            if (a[i] > mx)
                mx = a[i];
    }
#else
    for (int i = 1; i < n; i++)
        if (a[i] > mx)
            mx = a[i];
#endif
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        b[i] = expf(a[i] - mx);
        sum += b[i];
    }
    float inv = 1.0f / sum;
#if LLM_HAVE_AVX2
    {
        __m256 vinv = _mm256_set1_ps(inv);
        int i = 0;
        for (; i <= n - 8; i += 8)
            _mm256_storeu_ps(b + i,
                             _mm256_mul_ps(_mm256_loadu_ps(b + i), vinv));
        for (; i < n; i++)
            b[i] *= inv;
    }
#else
    for (int i = 0; i < n; i++)
        b[i] *= inv;
#endif
}

/* Reductions */
static float k_reduce_sum(const float *a, int n) {
#if LLM_HAVE_AVX2
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8)
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(a + i));
    float s = llm_hsum256(acc);
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
#if LLM_HAVE_AVX2
    __m256 vmx = _mm256_set1_ps(-FLT_MAX);
    int i = 0;
    for (; i <= n - 8; i += 8)
        vmx = _mm256_max_ps(vmx, _mm256_loadu_ps(a + i));
    __m128 lo = _mm256_castps256_ps128(vmx), hi = _mm256_extractf128_ps(vmx, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1)));
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 3, 2)));
    float mx = _mm_cvtss_f32(lo);
    for (; i < n; i++)
        if (a[i] > mx)
            mx = a[i];
    return mx;
#else
    float mx = a[0];
    for (int i = 1; i < n; i++)
        if (a[i] > mx)
            mx = a[i];
    return mx;
#endif
}

static float k_reduce_min(const float *a, int n) {
#if LLM_HAVE_AVX2
    __m256 vmn = _mm256_set1_ps(FLT_MAX);
    int i = 0;
    for (; i <= n - 8; i += 8)
        vmn = _mm256_min_ps(vmn, _mm256_loadu_ps(a + i));
    __m128 lo = _mm256_castps256_ps128(vmn), hi = _mm256_extractf128_ps(vmn, 1);
    lo = _mm_min_ps(lo, hi);
    lo = _mm_min_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1)));
    lo = _mm_min_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 3, 2)));
    float mn = _mm_cvtss_f32(lo);
    for (; i < n; i++)
        if (a[i] < mn)
            mn = a[i];
    return mn;
#else
    float mn = a[0];
    for (int i = 1; i < n; i++)
        if (a[i] < mn)
            mn = a[i];
    return mn;
#endif
}

/* Normalization */
static void
k_rmsnorm(const float *a, const float *w, float *b, int n, float eps) {
    float ss = 0.0f;
#if LLM_HAVE_AVX2
    {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i <= n - 8; i += 8)
            acc = _mm256_fmadd_ps(
                _mm256_loadu_ps(a + i), _mm256_loadu_ps(a + i), acc);
        ss = llm_hsum256(acc);
        for (; i < n; i++)
            ss += a[i] * a[i];
    }
#else
    for (int i = 0; i < n; i++)
        ss += a[i] * a[i];
#endif
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
#if LLM_HAVE_AVX2
    {
        __m256 vs = _mm256_set1_ps(scale);
        int i = 0;
        for (; i <= n - 8; i += 8)
            _mm256_storeu_ps(
                b + i,
                _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(a + i), vs),
                              _mm256_loadu_ps(w + i)));
        for (; i < n; i++)
            b[i] = a[i] * scale * w[i];
    }
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
#if LLM_HAVE_AVX2
    {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i <= n - 8; i += 8)
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(a + i));
        mean = llm_hsum256(acc);
        for (; i < n; i++)
            mean += a[i];
    }
#else
    for (int i = 0; i < n; i++)
        mean += a[i];
#endif
    mean /= (float)n;
    float var = 0.0f;
#if LLM_HAVE_AVX2
    {
        __m256 vmean = _mm256_set1_ps(mean), acc = _mm256_setzero_ps();
        int i = 0;
        for (; i <= n - 8; i += 8) {
            __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), vmean);
            acc = _mm256_fmadd_ps(d, d, acc);
        }
        var = llm_hsum256(acc);
        for (; i < n; i++) {
            float d = a[i] - mean;
            var += d * d;
        }
    }
#else
    for (int i = 0; i < n; i++) {
        float d = a[i] - mean;
        var += d * d;
    }
#endif
    var /= (float)n;
    float scale = 1.0f / sqrtf(var + eps);
#if LLM_HAVE_AVX2
    {
        __m256 vmean = _mm256_set1_ps(mean), vscale = _mm256_set1_ps(scale);
        int i = 0;
        for (; i <= n - 8; i += 8) {
            __m256 norm = _mm256_mul_ps(
                _mm256_sub_ps(_mm256_loadu_ps(a + i), vmean), vscale);
            _mm256_storeu_ps(b + i,
                             _mm256_fmadd_ps(norm,
                                             _mm256_loadu_ps(w + i),
                                             _mm256_loadu_ps(bias + i)));
        }
        for (; i < n; i++)
            b[i] = (a[i] - mean) * scale * w[i] + bias[i];
    }
#else
    for (int i = 0; i < n; i++)
        b[i] = (a[i] - mean) * scale * w[i] + bias[i];
#endif
}

/* Matrix operations */
static void
k_gemm_simd(const float *A, const float *B, float *C, int M, int K, int N) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[(size_t)i * K + k];
#if LLM_HAVE_AVX2
            __m256 va = _mm256_set1_ps(a_ik);
            int j = 0;
            for (; j <= N - 8; j += 8)
                _mm256_storeu_ps(
                    C + (size_t)i * N + j,
                    _mm256_fmadd_ps(va,
                                    _mm256_loadu_ps(B + (size_t)k * N + j),
                                    _mm256_loadu_ps(C + (size_t)i * N + j)));
            for (; j < N; j++)
                C[(size_t)i * N + j] += a_ik * B[(size_t)k * N + j];
#else
            for (int j = 0; j < N; j++)
                C[(size_t)i * N + j] += a_ik * B[(size_t)k * N + j];
#endif
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

/* RoPE */
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

/* ===========================================================================
 * Section 2: BLAS kernels
 * =========================================================================*/

static void k_add_blas(const float *a, const float *b, float *c, int n) {
    memcpy(c, b, (size_t)n * sizeof(float));
    cblas_saxpy(n, 1.0f, a, 1, c, 1);
}
static void k_mul_blas(const float *a, const float *b, float *c, int n) {
    k_mul_simd(a, b, c, n);
}
static void k_scale_blas(const float *a, float alpha, float *b, int n) {
    memcpy(b, a, (size_t)n * sizeof(float));
    cblas_sscal(n, alpha, b, 1);
}
static float k_dot_blas(const float *a, const float *b, int n) {
    return cblas_sdot(n, a, 1, b, 1);
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

/* ===========================================================================
 * Section 3: Blob allocation helpers
 * =========================================================================*/

static void *alloc_vec_blob(int n, const float *data, int *sz_out) {
    int sz = LLM_VEC_BLOB_SIZE(n);
    void *blob = sqlite3_malloc(sz);
    if (!blob)
        return NULL;
    *(int32_t *)blob = n;
    if (data)
        memcpy((char *)blob + 4, data, (size_t)n * sizeof(float));
    else
        memset((char *)blob + 4, 0, (size_t)n * sizeof(float));
    *sz_out = sz;
    return blob;
}

static void *
alloc_mat_blob(int rows, int cols, const float *data, int *sz_out) {
    int sz = LLM_MAT_BLOB_SIZE(rows, cols);
    void *blob = sqlite3_malloc(sz);
    if (!blob)
        return NULL;
    *(int32_t *)blob = rows;
    *((int32_t *)blob + 1) = cols;
    int nel = rows * cols;
    if (data)
        memcpy((char *)blob + 8, data, (size_t)nel * sizeof(float));
    else
        memset((char *)blob + 8, 0, (size_t)nel * sizeof(float));
    *sz_out = sz;
    return blob;
}

/* ===========================================================================
 * Section 4: SQL wrapper macros
 * =========================================================================*/

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

/* ===========================================================================
 * Section 5: Construct / Inspect
 * =========================================================================*/

/* llm_vec(n, val) */
static void sql_vec(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    int n = sqlite3_value_int(argv[0]);
    float val = (float)sqlite3_value_double(argv[1]);
    if (n <= 0)
        ERR(ctx, "n must be positive");
    int sz = LLM_VEC_BLOB_SIZE(n);
    void *blob = sqlite3_malloc(sz);
    if (!blob) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    *(int32_t *)blob = n;
    float *data = (float *)((char *)blob + 4);
    for (int i = 0; i < n; i++)
        data[i] = val;
    sqlite3_result_blob(ctx, blob, sz, sqlite3_free);
}

/* llm_mat(rows, cols, val) */
static void sql_mat(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    int rows = sqlite3_value_int(argv[0]), cols = sqlite3_value_int(argv[1]);
    float val = (float)sqlite3_value_double(argv[2]);
    if (rows <= 0 || cols <= 0)
        ERR(ctx, "rows and cols must be positive");
    int sz = LLM_MAT_BLOB_SIZE(rows, cols);
    void *blob = sqlite3_malloc(sz);
    if (!blob) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    *(int32_t *)blob = rows;
    *((int32_t *)blob + 1) = cols;
    float *data = (float *)((char *)blob + 8);
    for (int i = 0; i < rows * cols; i++)
        data[i] = val;
    sqlite3_result_blob(ctx, blob, sz, sqlite3_free);
}

/* llm_vget(v, i) */
static void sql_vget(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 v;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &v) < 0)
        ERR(ctx, "Invalid vector blob");
    int idx = sqlite3_value_int(argv[1]);
    if (idx < 0 || idx >= v.n)
        ERR(ctx, "Index out of range");
    sqlite3_result_double(ctx, (double)v.data[idx]);
}

/* llm_mget(m, r, c) */
static void sql_mget(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 m;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &m) < 0)
        ERR(ctx, "Invalid matrix blob");
    int r = sqlite3_value_int(argv[1]), c = sqlite3_value_int(argv[2]);
    if (r < 0 || r >= m.rows || c < 0 || c >= m.cols)
        ERR(ctx, "Index out of range");
    sqlite3_result_double(ctx, (double)m.data[(size_t)r * m.cols + c]);
}

/* llm_len(v) */
static void sql_len(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 v;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &v) < 0)
        ERR(ctx, "Invalid vector blob");
    sqlite3_result_int(ctx, v.n);
}

/* llm_rows(m) */
static void sql_rows(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 m;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &m) < 0)
        ERR(ctx, "Invalid matrix blob");
    sqlite3_result_int(ctx, m.rows);
}

/* llm_cols(m) */
static void sql_cols(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 m;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &m) < 0)
        ERR(ctx, "Invalid matrix blob");
    sqlite3_result_int(ctx, m.cols);
}

/* llm_str(blob) -- auto-detect vector or matrix, render as text */
static void sql_str(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int bsz = sqlite3_value_bytes(argv[0]);
    if (!blob || bsz < 4)
        ERR(ctx, "Invalid blob");

    VecF32 v;
    MatF32 m;
    /* For auto-detect, use exact size: a vector blob is exactly 4+n*4 bytes.
     * This prevents a matrix from being incorrectly classified as a vector.
     * Use size_t arithmetic to avoid int overflow for large n. */
    int is_vec = (llm_parse_vec(blob, bsz, &v) == 0 &&
                  bsz == (int)(4 + (size_t)v.n * 4));
    int is_mat = (!is_vec && llm_parse_mat(blob, bsz, &m) == 0);
    if (!is_vec && !is_mat)
        ERR(ctx, "Cannot parse blob as vector or matrix");

    if (is_vec) {
        int cap = v.n * 18 + 4;
        char *buf = sqlite3_malloc(cap);
        if (!buf) {
            sqlite3_result_error_nomem(ctx);
            return;
        }
        int pos = 0;
        buf[pos++] = '[';
        for (int i = 0; i < v.n; i++) {
            if (i) {
                buf[pos++] = ',';
                buf[pos++] = ' ';
            }
            int rem = cap - pos;
            if (rem <= 0)
                break;
            int w = snprintf(buf + pos, (size_t)rem, "%.6g", (double)v.data[i]);
            if (w > 0 && w < rem)
                pos += w;
        }
        buf[pos++] = ']';
        buf[pos] = '\0';
        sqlite3_result_text(ctx, buf, pos, sqlite3_free);
    } else {
        int nel = m.rows * m.cols;
        int cap = nel * 18 + m.rows * 6 + 8;
        char *buf = sqlite3_malloc(cap);
        if (!buf) {
            sqlite3_result_error_nomem(ctx);
            return;
        }
        int pos = 0;
        buf[pos++] = '[';
        for (int i = 0; i < m.rows; i++) {
            if (i) {
                buf[pos++] = ',';
                buf[pos++] = ' ';
            }
            buf[pos++] = '[';
            for (int j = 0; j < m.cols; j++) {
                if (j) {
                    buf[pos++] = ',';
                    buf[pos++] = ' ';
                }
                int rem = cap - pos;
                if (rem <= 0)
                    break;
                int w = snprintf(buf + pos,
                                 (size_t)rem,
                                 "%.6g",
                                 (double)m.data[(size_t)i * m.cols + j]);
                if (w > 0 && w < rem)
                    pos += w;
            }
            buf[pos++] = ']';
        }
        buf[pos++] = ']';
        buf[pos] = '\0';
        sqlite3_result_text(ctx, buf, pos, sqlite3_free);
    }
}

/* ===========================================================================
 * Section 6: Arithmetic SQL wrappers
 * =========================================================================*/

static void sql_add_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    EMIT_VEC_BINOP(ctx, va, k_add_simd(va.data, vb.data, tmp, va.n));
}
static void sql_add_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    EMIT_VEC_BINOP(ctx, va, k_add_blas(va.data, vb.data, tmp, va.n));
}
static void sql_sub_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    EMIT_VEC_BINOP(ctx, va, k_sub_simd(va.data, vb.data, tmp, va.n));
}
static void sql_mul_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    EMIT_VEC_BINOP(ctx, va, k_mul_simd(va.data, vb.data, tmp, va.n));
}
static void sql_mul_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    EMIT_VEC_BINOP(ctx, va, k_mul_blas(va.data, vb.data, tmp, va.n));
}
static void sql_div_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    EMIT_VEC_BINOP(ctx, va, k_div_simd(va.data, vb.data, tmp, va.n));
}
static void
sql_scale_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    float alpha = (float)sqlite3_value_double(argv[1]);
    EMIT_VEC_BINOP(ctx, va, k_scale_simd(va.data, alpha, tmp, va.n));
}
static void
sql_scale_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    float alpha = (float)sqlite3_value_double(argv[1]);
    EMIT_VEC_BINOP(ctx, va, k_scale_blas(va.data, alpha, tmp, va.n));
}
static void sql_dot_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    sqlite3_result_double(ctx, (double)k_dot_simd(va.data, vb.data, va.n));
}
static void sql_dot_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    PARSE_2VEC(ctx, argv, va, vb);
    sqlite3_result_double(ctx, (double)k_dot_blas(va.data, vb.data, va.n));
}

/* ===========================================================================
 * Section 7: Element-wise math + activations
 * =========================================================================*/

DEFINE_UNARY_VEC(sql_neg, k_neg_simd)
DEFINE_UNARY_VEC(sql_abs, k_abs_simd)
DEFINE_UNARY_VEC(sql_sqrt, k_sqrt_simd)
DEFINE_UNARY_VEC(sql_exp, k_exp_simd)
DEFINE_UNARY_VEC(sql_log, k_log_simd)
DEFINE_UNARY_VEC(sql_relu, k_relu_simd)
DEFINE_UNARY_VEC(sql_gelu, k_gelu_simd)
DEFINE_UNARY_VEC(sql_silu, k_silu_simd)
DEFINE_UNARY_VEC(sql_sigmoid, k_sigmoid_simd)
DEFINE_UNARY_VEC(sql_softmax, k_softmax_simd)

/* llm_clip(x, lo, hi) */
static void sql_clip(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    float lo = (float)sqlite3_value_double(argv[1]);
    float hi = (float)sqlite3_value_double(argv[2]);
    EMIT_VEC_BINOP(ctx, va, k_clip_simd(va.data, lo, hi, tmp, va.n));
}

/* ===========================================================================
 * Section 8: Reductions
 * =========================================================================*/

static void sql_sum(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    if (va.n == 0)
        ERR(ctx, "Empty vector");
    sqlite3_result_double(ctx, (double)k_reduce_sum(va.data, va.n));
}
static void sql_mean(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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
static void sql_max(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    if (va.n == 0)
        ERR(ctx, "Empty vector");
    sqlite3_result_double(ctx, (double)k_reduce_max(va.data, va.n));
}
static void sql_min(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob");
    if (va.n == 0)
        ERR(ctx, "Empty vector");
    sqlite3_result_double(ctx, (double)k_reduce_min(va.data, va.n));
}

/* ===========================================================================
 * Section 9: Normalization
 * =========================================================================*/

static void sql_rmsnorm(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

static void
sql_layernorm(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ===========================================================================
 * Section 10: Matrix operations
 * =========================================================================*/

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

static void
sql_gemm_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

static void
sql_gemm_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

static void sql_matadd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

static void
sql_transpose(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

static void
sql_matvec_simd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

static void
sql_matvec_blas(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ===========================================================================
 * Section 11: Shape / Indexing
 * =========================================================================*/

/* llm_reshape(blob, rows, cols) -- reinterpret existing data under new shape */
static void sql_reshape(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    int new_rows = sqlite3_value_int(argv[1]),
        new_cols = sqlite3_value_int(argv[2]);
    if (new_rows <= 0 || new_cols <= 0)
        ERR(ctx, "rows and cols must be positive");
    const void *blob = sqlite3_value_blob(argv[0]);
    int bsz = sqlite3_value_bytes(argv[0]);
    VecF32 v;
    MatF32 m;
    int nel = 0;
    const float *src = NULL;
    if (llm_parse_vec(blob, bsz, &v) == 0) {
        nel = v.n;
        src = v.data;
    } else if (llm_parse_mat(blob, bsz, &m) == 0) {
        nel = m.rows * m.cols;
        src = m.data;
    } else
        ERR(ctx, "Invalid blob");
    if (nel != new_rows * new_cols)
        ERR(ctx, "Total element count mismatch for reshape");
    int out_sz;
    void *out = alloc_mat_blob(new_rows, new_cols, src, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* llm_flatten(m) -- matrix to 1-D vector */
static void sql_flatten(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 m;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &m) < 0)
        ERR(ctx, "Invalid matrix blob");
    int nel = m.rows * m.cols;
    int out_sz;
    void *out = alloc_vec_blob(nel, m.data, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* llm_slice(v, start, end) -- extract sub-vector; negative end: v.n+end */
static void sql_slice(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 v;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &v) < 0)
        ERR(ctx, "Invalid vector blob");
    int start = sqlite3_value_int(argv[1]), end = sqlite3_value_int(argv[2]);
    if (end < -v.n)
        end = 0; /* guard extreme negative values */
    if (end < 0)
        end = v.n + end;
    if (start < 0)
        start = 0;
    if (end > v.n)
        end = v.n;
    if (start >= end)
        ERR(ctx, "Empty or invalid slice range");
    int len = end - start;
    int out_sz;
    void *out = alloc_vec_blob(len, v.data + start, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* llm_concat(a, b) -- concatenate two vectors */
static void sql_concat(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    VecF32 va, vb;
    if (llm_parse_vec(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &va) < 0)
        ERR(ctx, "Invalid vector blob (arg 0)");
    if (llm_parse_vec(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &vb) < 0)
        ERR(ctx, "Invalid vector blob (arg 1)");
    int n = va.n + vb.n;
    int out_sz = LLM_VEC_BLOB_SIZE(n);
    void *out = sqlite3_malloc(out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    *(int32_t *)out = n;
    float *data = (float *)((char *)out + 4);
    memcpy(data, va.data, (size_t)va.n * sizeof(float));
    memcpy(data + va.n, vb.data, (size_t)vb.n * sizeof(float));
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* llm_mrow(m, r) -- extract row r as a vector */
static void sql_mrow(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 m;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &m) < 0)
        ERR(ctx, "Invalid matrix blob");
    int r = sqlite3_value_int(argv[1]);
    if (r < 0 || r >= m.rows)
        ERR(ctx, "Row index out of range");
    int out_sz;
    void *out = alloc_vec_blob(m.cols, m.data + (size_t)r * m.cols, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/*
 * llm_gather(m, idx_vec) -- gather rows from m by float-encoded row indices.
 * Returns a new matrix of shape (len(idx) x m.cols).
 * Used for token embedding lookup (pass token IDs as a float vector).
 */
static void sql_gather(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    MatF32 m;
    VecF32 idx;
    if (llm_parse_mat(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &m) < 0)
        ERR(ctx, "Invalid matrix blob");
    if (llm_parse_vec(sqlite3_value_blob(argv[1]),
                      sqlite3_value_bytes(argv[1]),
                      &idx) < 0)
        ERR(ctx, "Invalid index vector blob");
    int out_rows = idx.n, out_cols = m.cols;
    float *tmp =
        sqlite3_malloc((int)((size_t)out_rows * out_cols * sizeof(float)));
    if (!tmp) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    for (int i = 0; i < out_rows; i++) {
        int r = (int)roundf(idx.data[i]);
        if (r < 0 || r >= m.rows) {
            sqlite3_free(tmp);
            ERR(ctx, "Gather index out of range");
        }
        memcpy(tmp + (size_t)i * out_cols,
               m.data + (size_t)r * out_cols,
               (size_t)out_cols * sizeof(float));
    }
    int out_sz;
    void *out = alloc_mat_blob(out_rows, out_cols, tmp, &out_sz);
    sqlite3_free(tmp);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ===========================================================================
 * Section 12: Transformer -- RoPE
 * =========================================================================*/

/*
 * llm_rope(x, cos_row, sin_row, head_dim)
 *   x        : vector of head_dim floats (single attention head)
 *   cos_row  : vector of head_dim/2 cosine values for this position
 *   sin_row  : vector of head_dim/2 sine values for this position
 *   head_dim : INTEGER (must equal x.n and 2 * cos_row.n)
 */
static void sql_rope(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── Internal N-D helpers ───────────────────────────────────────────────────
 */

/* Compute row-major strides for a given shape array. */
static void nd_strides(const int32_t *shape, int ndim, int32_t *strides) {
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        strides[i] = strides[i + 1] * shape[i + 1];
}

/* Allocate a result ND blob; helper so SQL callbacks are clean. */
static void *
nd_result_alloc(int ndim, const int32_t *shape, int32_t total, int *sz_out) {
    int hdr = LLM_ND_HDR(ndim);
    int sz = hdr + total * (int)sizeof(float);
    char *b = (char *)sqlite3_malloc(sz);
    if (!b)
        return NULL;
    *(int32_t *)b = LLM_ND_MAGIC;
    *((int32_t *)b + 1) = ndim;
    for (int i = 0; i < ndim; i++)
        ((int32_t *)b)[2 + i] = shape[i];
    *sz_out = sz;
    return b;
}

/* ── llm_ndim ───────────────────────────────────────────────────────────────
 */
static void sql_ndim(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0) {
        ERR(ctx, "Invalid tensor blob");
    }
    sqlite3_result_int(ctx, t.ndim);
}

/* ── llm_shape ──────────────────────────────────────────────────────────────
 */
static void sql_shape(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0) {
        ERR(ctx, "Invalid tensor blob");
    }
    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_shape: dim out of range");
    sqlite3_result_int(ctx, t.shape[dim]);
}

/* ── llm_view_nd (variadic: blob, d0 [, d1, ...]) ────────────────────────── */
static void sql_view_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2)
        ERR(ctx, "llm_view_nd: need blob + at least 1 dim");
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_view_nd: invalid blob");

    int new_ndim = argc - 1;
    if (new_ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_view_nd: too many dims");

    int32_t new_shape[LLM_MAX_NDIM];
    int32_t total = 1;
    int infer_idx = -1;
    for (int i = 0; i < new_ndim; i++) {
        new_shape[i] = sqlite3_value_int(argv[i + 1]);
        if (new_shape[i] == -1) {
            if (infer_idx >= 0)
                ERR(ctx, "llm_view_nd: only one -1 allowed");
            infer_idx = i;
        } else {
            if (new_shape[i] < 0)
                ERR(ctx, "llm_view_nd: negative dim");
            total *= new_shape[i];
        }
    }
    if (infer_idx >= 0) {
        if (total == 0 || t.total % total != 0)
            ERR(ctx, "llm_view_nd: cannot infer dim");
        new_shape[infer_idx] = t.total / total;
        total = t.total;
    }
    if (total != t.total)
        ERR(ctx, "llm_view_nd: element count mismatch");

    int out_sz;
    void *out = nd_result_alloc(new_ndim, new_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    memcpy((char *)out + LLM_ND_HDR(new_ndim),
           t.data,
           (size_t)total * sizeof(float));
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_permute_nd (variadic: blob, p0, p1, ..., p(ndim-1)) ─────────────── */
static void
sql_permute_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2)
        ERR(ctx, "llm_permute_nd: need blob + perm axes");
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_permute_nd: invalid blob");
    int ndim = t.ndim;
    if (argc - 1 != ndim)
        ERR(ctx, "llm_permute_nd: wrong number of perm axes");

    int32_t perm[LLM_MAX_NDIM];
    int seen[LLM_MAX_NDIM] = {0};
    for (int i = 0; i < ndim; i++) {
        perm[i] = sqlite3_value_int(argv[i + 1]);
        if (perm[i] < 0 || perm[i] >= ndim || seen[perm[i]])
            ERR(ctx, "llm_permute_nd: invalid permutation");
        seen[perm[i]] = 1;
    }

    int32_t new_shape[LLM_MAX_NDIM];
    for (int i = 0; i < ndim; i++)
        new_shape[i] = t.shape[perm[i]];

    int out_sz;
    void *out = nd_result_alloc(ndim, new_shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ndim));

    /* Compute input strides in original order */
    int32_t src_strides[LLM_MAX_NDIM];
    nd_strides(t.shape, ndim, src_strides);
    /* Compute output strides in new order */
    int32_t dst_strides[LLM_MAX_NDIM];
    nd_strides(new_shape, ndim, dst_strides);

    /* Iterate over output indices, compute input flat index */
    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < t.total; flat++) {
        /* Compute input flat offset from output (permuted) index */
        int32_t src_off = 0;
        for (int d = 0; d < ndim; d++)
            src_off += idx[d] * src_strides[perm[d]];
        dst[flat] = t.data[src_off];
        /* Increment output multi-index */
        for (int d = ndim - 1; d >= 0; d--) {
            if (++idx[d] < new_shape[d])
                break;
            idx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_transpose_nd(blob, d0, d1) ─────────────────────────────────────── */
static void
sql_transpose_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_transpose_nd: invalid blob");
    int d0 = sqlite3_value_int(argv[1]);
    int d1 = sqlite3_value_int(argv[2]);
    if (d0 < 0)
        d0 += t.ndim;
    if (d1 < 0)
        d1 += t.ndim;
    if (d0 < 0 || d0 >= t.ndim || d1 < 0 || d1 >= t.ndim)
        ERR(ctx, "llm_transpose_nd: dim out of range");

    int32_t perm[LLM_MAX_NDIM];
    for (int i = 0; i < t.ndim; i++)
        perm[i] = i;
    perm[d0] = d1;
    perm[d1] = d0;

    /* Reuse permute logic inline */
    int32_t new_shape[LLM_MAX_NDIM];
    for (int i = 0; i < t.ndim; i++)
        new_shape[i] = t.shape[perm[i]];
    int out_sz;
    void *out = nd_result_alloc(t.ndim, new_shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    int32_t src_strides[LLM_MAX_NDIM];
    nd_strides(t.shape, t.ndim, src_strides);
    int32_t dst_strides[LLM_MAX_NDIM];
    nd_strides(new_shape, t.ndim, dst_strides);

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < t.total; flat++) {
        int32_t src_off = 0;
        for (int d = 0; d < t.ndim; d++)
            src_off += idx[d] * src_strides[perm[d]];
        dst[flat] = t.data[src_off];
        for (int d = t.ndim - 1; d >= 0; d--) {
            if (++idx[d] < new_shape[d])
                break;
            idx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_unsqueeze_nd(blob, dim) ─────────────────────────────────────────── */
static void
sql_unsqueeze_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_unsqueeze_nd: invalid blob");
    if (t.ndim >= LLM_MAX_NDIM)
        ERR(ctx, "llm_unsqueeze_nd: already at max dims");

    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim + 1;
    if (dim < 0 || dim > t.ndim)
        ERR(ctx, "llm_unsqueeze_nd: dim out of range");

    int32_t new_shape[LLM_MAX_NDIM];
    int new_ndim = t.ndim + 1;
    for (int i = 0; i < dim; i++)
        new_shape[i] = t.shape[i];
    new_shape[dim] = 1;
    for (int i = dim; i < t.ndim; i++)
        new_shape[i + 1] = t.shape[i];

    int out_sz;
    void *out = nd_result_alloc(new_ndim, new_shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    memcpy((char *)out + LLM_ND_HDR(new_ndim),
           t.data,
           (size_t)t.total * sizeof(float));
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_squeeze_nd(blob, dim) ───────────────────────────────────────────── */

/* ── llm_view_t12_nd(blob, s0, s1, ..., sN)
 *
 * Fused view + transpose(1, 2).  Reshapes the input to the given
 * shape, then swaps dimensions 1 and 2.  Commonly used for the
 * QKV head-reshape pattern: [B, S, H*D] → view [B, S, H, D] → [B, H, S, D]
 *
 * When one of the transposed dimensions has size 1 (decode step),
 * the data layout is unchanged — only the header is rewritten.
 * ──────────────────────────────────────────────────────────────── */
static void
sql_view_t12_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 3)
        ERR(ctx, "llm_view_t12_nd: need blob + at least 2 shape dims");
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_view_t12_nd: invalid blob");

    /* Parse target shape */
    int new_ndim = argc - 1;
    if (new_ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_view_t12_nd: too many dims");
    int32_t view_shape[LLM_MAX_NDIM];
    int neg_idx = -1;
    int32_t known = 1;
    for (int i = 0; i < new_ndim; i++) {
        view_shape[i] = sqlite3_value_int(argv[i + 1]);
        if (view_shape[i] == -1)
            neg_idx = i;
        else
            known *= view_shape[i];
    }
    if (neg_idx >= 0)
        view_shape[neg_idx] = t.total / known;
    if (new_ndim < 3)
        ERR(ctx, "llm_view_t12_nd: need at least 3 dims for transpose(1,2)");

    /* Transposed shape: swap dims 1 and 2 */
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < new_ndim; i++)
        out_shape[i] = view_shape[i];
    out_shape[1] = view_shape[2];
    out_shape[2] = view_shape[1];

    int out_sz;
    void *out = nd_result_alloc(new_ndim, out_shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(new_ndim));

    /* Fast path: when dim 1 or dim 2 has size 1, data layout is unchanged */
    if (view_shape[1] == 1 || view_shape[2] == 1) {
        memcpy(dst, t.data, (size_t)t.total * sizeof(float));
    } else {
        /* General transpose(1,2) after view */
        int32_t perm[LLM_MAX_NDIM];
        for (int i = 0; i < new_ndim; i++)
            perm[i] = i;
        perm[1] = 2;
        perm[2] = 1;

        int32_t src_strides[LLM_MAX_NDIM], dst_strides[LLM_MAX_NDIM];
        nd_strides(view_shape, new_ndim, src_strides);
        nd_strides(out_shape, new_ndim, dst_strides);

        int32_t idx[LLM_MAX_NDIM];
        memset(idx, 0, sizeof(idx));
        for (int32_t flat = 0; flat < t.total; flat++) {
            int32_t src_off = 0;
            for (int d = 0; d < new_ndim; d++)
                src_off += idx[d] * src_strides[perm[d]];
            dst[flat] = t.data[src_off];
            for (int d = new_ndim - 1; d >= 0; d--) {
                if (++idx[d] < out_shape[d])
                    break;
                idx[d] = 0;
            }
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

static void
sql_squeeze_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_squeeze_nd: invalid blob");

    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_squeeze_nd: dim out of range");
    if (t.shape[dim] != 1)
        ERR(ctx, "llm_squeeze_nd: dim is not size 1");
    if (t.ndim <= 1)
        ERR(ctx, "llm_squeeze_nd: cannot reduce to 0 dims");

    int32_t new_shape[LLM_MAX_NDIM];
    int new_ndim = t.ndim - 1;
    int j = 0;
    for (int i = 0; i < t.ndim; i++)
        if (i != dim)
            new_shape[j++] = t.shape[i];

    int out_sz;
    void *out = nd_result_alloc(new_ndim, new_shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    memcpy((char *)out + LLM_ND_HDR(new_ndim),
           t.data,
           (size_t)t.total * sizeof(float));
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_cat_nd(a, b, dim) ───────────────────────────────────────────────── */
static void sql_cat_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_cat_nd: invalid first blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_cat_nd: invalid second blob");
    if (ta.ndim != tb.ndim)
        ERR(ctx, "llm_cat_nd: ndim mismatch");

    int dim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += ta.ndim;
    if (dim < 0 || dim >= ta.ndim)
        ERR(ctx, "llm_cat_nd: dim out of range");

    /* Check all dims except cat dim match */
    for (int i = 0; i < ta.ndim; i++)
        if (i != dim && ta.shape[i] != tb.shape[i])
            ERR(ctx, "llm_cat_nd: shape mismatch on non-cat dim");

    int32_t new_shape[LLM_MAX_NDIM];
    for (int i = 0; i < ta.ndim; i++)
        new_shape[i] = ta.shape[i];
    new_shape[dim] = ta.shape[dim] + tb.shape[dim];
    int32_t new_total = 1;
    for (int i = 0; i < ta.ndim; i++)
        new_total *= new_shape[i];

    int out_sz;
    void *out = nd_result_alloc(ta.ndim, new_shape, new_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ta.ndim));

    /* Block-copy approach: iterate over outer slices, memcpy inner blocks.
     * Much faster than per-element multi-index arithmetic.
     *
     * outer = product of dims before cat-dim
     * inner = product of dims after cat-dim
     * For each outer slice: copy a-block (a_dim * inner), then b-block
     */
    int32_t outer = 1, inner = 1;
    for (int i = 0; i < dim; i++)
        outer *= ta.shape[i];
    for (int i = dim + 1; i < ta.ndim; i++)
        inner *= ta.shape[i];

    int32_t a_block = ta.shape[dim] * inner;
    int32_t b_block = tb.shape[dim] * inner;
    int32_t o_block = a_block + b_block;

    for (int32_t i = 0; i < outer; i++) {
        memcpy(dst + (size_t)i * o_block,
               ta.data + (size_t)i * a_block,
               (size_t)a_block * sizeof(float));
        memcpy(dst + (size_t)i * o_block + a_block,
               tb.data + (size_t)i * b_block,
               (size_t)b_block * sizeof(float));
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_slice_nd(blob, dim, start, end [, step]) ────────────────────────── */
static void sql_slice_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc != 4 && argc != 5)
        ERR(ctx, "llm_slice_nd: need 4 or 5 arguments");
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_slice_nd: invalid blob");

    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_slice_nd: dim out of range");

    int s = sqlite3_value_int(argv[2]);
    int e = sqlite3_value_int(argv[3]);
    int step = argc >= 5 ? sqlite3_value_int(argv[4]) : 1;
    if (step <= 0)
        ERR(ctx, "llm_slice_nd: step must be > 0");
    int d_sz = t.shape[dim];
    if (s < 0)
        s += d_sz;
    if (e < 0)
        e += d_sz;
    if (s < 0)
        s = 0;
    if (s > d_sz)
        s = d_sz;
    if (e < 0)
        e = 0;
    if (e > d_sz)
        e = d_sz;
    /* Match Torch slicing semantics: start >= end yields an empty slice. */
    if (s > e)
        s = e;

    int32_t new_shape[LLM_MAX_NDIM];
    for (int i = 0; i < t.ndim; i++)
        new_shape[i] = t.shape[i];
    int32_t span = e - s;
    int32_t sliced = span <= 0 ? 0 : (span + step - 1) / step;
    new_shape[dim] = sliced;
    int32_t new_total = 1;
    for (int i = 0; i < t.ndim; i++)
        new_total *= new_shape[i];

    int out_sz;
    void *out = nd_result_alloc(t.ndim, new_shape, new_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    int32_t strides[LLM_MAX_NDIM];
    nd_strides(t.shape, t.ndim, strides);
    int32_t new_strides[LLM_MAX_NDIM];
    nd_strides(new_shape, t.ndim, new_strides);

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < new_total; flat++) {
        int32_t src_off = 0;
        for (int d = 0; d < t.ndim; d++) {
            if (d == dim)
                src_off += (s + idx[d] * step) * strides[d];
            else
                src_off += idx[d] * strides[d];
        }
        dst[flat] = t.data[src_off];
        for (int d = t.ndim - 1; d >= 0; d--) {
            if (++idx[d] < new_shape[d])
                break;
            idx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_select_nd(blob, dim, index) — fused slice+squeeze ───────────────
 * Select a single index along a dimension, removing that dimension.
 * Equivalent to squeeze(slice(x, dim, idx, idx+1), dim).
 * ──────────────────────────────────────────────────────────────────────── */
static void
sql_select_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_select_nd: invalid blob");
    if (t.ndim < 2)
        ERR(ctx, "llm_select_nd: need at least 2-D tensor");

    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_select_nd: dim out of range");

    int idx = sqlite3_value_int(argv[2]);
    if (idx < 0)
        idx += t.shape[dim];
    if (idx < 0 || idx >= t.shape[dim])
        ERR(ctx, "llm_select_nd: index out of range");

    /* Output shape: remove the selected dimension */
    int new_ndim = t.ndim - 1;
    int32_t new_shape[LLM_MAX_NDIM];
    int ni = 0;
    for (int i = 0; i < t.ndim; i++) {
        if (i != dim)
            new_shape[ni++] = t.shape[i];
    }
    int32_t new_total = 1;
    for (int i = 0; i < new_ndim; i++)
        new_total *= new_shape[i];

    int out_sz;
    void *out = nd_result_alloc(new_ndim, new_shape, new_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(new_ndim));

    /* Copy the selected slice */
    int32_t strides[LLM_MAX_NDIM];
    nd_strides(t.shape, t.ndim, strides);

    int32_t oidx[LLM_MAX_NDIM];
    memset(oidx, 0, sizeof(oidx));
    for (int32_t flat = 0; flat < new_total; flat++) {
        int32_t src_off = idx * strides[dim];
        int oi = 0;
        for (int d = 0; d < t.ndim; d++) {
            if (d != dim) {
                src_off += oidx[oi] * strides[d];
                oi++;
            }
        }
        dst[flat] = t.data[src_off];
        for (int d = new_ndim - 1; d >= 0; d--) {
            if (++oidx[d] < new_shape[d])
                break;
            oidx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_expand_nd(blob, d0 [, d1, ...]) ────────────────────────────────── */
static void
sql_expand_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2)
        ERR(ctx, "llm_expand_nd: need blob + shape dims");
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_expand_nd: invalid blob");

    int new_ndim = argc - 1;
    if (new_ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_expand_nd: too many dims");
    /* Target ndim must be >= source ndim */
    if (new_ndim < t.ndim)
        ERR(ctx, "llm_expand_nd: cannot reduce ndim via expand");

    int32_t new_shape[LLM_MAX_NDIM];
    for (int i = 0; i < new_ndim; i++)
        new_shape[i] = sqlite3_value_int(argv[i + 1]);

    int offset = new_ndim - t.ndim;
    for (int i = 0; i < new_ndim; i++) {
        if (new_shape[i] == -1)
            new_shape[i] = (i < offset) ? 1 : t.shape[i - offset];
    }

    /* Validate broadcasting: align shapes from the right */
    for (int i = 0; i < t.ndim; i++) {
        int src_d = t.shape[i];
        int dst_d = new_shape[i + offset];
        if (src_d != 1 && src_d != dst_d)
            ERR(ctx, "llm_expand_nd: incompatible shapes for expand");
    }

    int32_t new_total = 1;
    for (int i = 0; i < new_ndim; i++)
        new_total *= new_shape[i];

    int out_sz;
    void *out = nd_result_alloc(new_ndim, new_shape, new_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(new_ndim));

    /* Compute source strides with leading 1s for broadcast dims */
    int32_t src_shape_padded[LLM_MAX_NDIM];
    for (int i = 0; i < offset; i++)
        src_shape_padded[i] = 1;
    for (int i = 0; i < t.ndim; i++)
        src_shape_padded[i + offset] = t.shape[i];

    int32_t src_strides[LLM_MAX_NDIM];
    nd_strides(src_shape_padded, new_ndim, src_strides);
    /* Set stride=0 for broadcast dims (size==1) */
    for (int i = 0; i < new_ndim; i++)
        if (src_shape_padded[i] == 1)
            src_strides[i] = 0;

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < new_total; flat++) {
        int32_t src_off = 0;
        for (int d = 0; d < new_ndim; d++)
            src_off += idx[d] * src_strides[d];
        dst[flat] = t.data[src_off];
        for (int d = new_ndim - 1; d >= 0; d--) {
            if (++idx[d] < new_shape[d])
                break;
            idx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_bmm(A, B) — batch matrix multiply ──────────────────────────────── */
/*
 * A: (..., M, K)   B: (..., K, N)   result: (..., M, N)
 * Supports 2-D (M×K × K×N), 3-D (B×M×K × B×K×N) and 4-D batch dims.
 */
static void sql_bmm(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_bmm: invalid A blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_bmm: invalid B blob");
    if (ta.ndim < 2 || tb.ndim < 2)
        ERR(ctx, "llm_bmm: need at least 2-D tensors");
    if (ta.ndim != tb.ndim)
        ERR(ctx, "llm_bmm: A and B must have same ndim");

    int nd = ta.ndim;
    int M = ta.shape[nd - 2];
    int K = ta.shape[nd - 1];
    int K2 = tb.shape[nd - 2];
    int N = tb.shape[nd - 1];
    if (K != K2)
        ERR(ctx, "llm_bmm: inner dims mismatch");

    /* Compute batch size (product of leading dims) */
    int32_t batch = 1;
    int32_t new_shape[LLM_MAX_NDIM];
    for (int i = 0; i < nd - 2; i++) {
        if (ta.shape[i] != tb.shape[i])
            ERR(ctx, "llm_bmm: batch dims mismatch");
        new_shape[i] = ta.shape[i];
        batch *= ta.shape[i];
    }
    new_shape[nd - 2] = M;
    new_shape[nd - 1] = N;
    int32_t new_total = batch * M * N;

    int out_sz;
    void *out = nd_result_alloc(nd, new_shape, new_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(nd));

    /* Batch GEMM with M=1 fast path using sgemv */
    if (M == 1) {
        for (int b = 0; b < batch; b++) {
            const float *a_ptr = ta.data + (size_t)b * K;
            const float *b_ptr = tb.data + (size_t)b * K * N;
            float *c_ptr = dst + (size_t)b * N;
            /* y = B^T * a: B is K×N, a is K×1, result is N×1 */
            cblas_sgemv(CblasRowMajor,
                        CblasTrans,
                        K,
                        N,
                        1.0f,
                        b_ptr,
                        N,
                        a_ptr,
                        1,
                        0.0f,
                        c_ptr,
                        1);
        }
    } else {
        for (int b = 0; b < batch; b++) {
            const float *a_ptr = ta.data + (size_t)b * M * K;
            const float *b_ptr = tb.data + (size_t)b * K * N;
            float *c_ptr = dst + (size_t)b * M * N;
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        M,
                        N,
                        K,
                        1.0f,
                        a_ptr,
                        K,
                        b_ptr,
                        N,
                        0.0f,
                        c_ptr,
                        N);
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_linear_nd(x, weight) — (..., K) × (N, K)^T -> (..., N) ─────────── */
static void
sql_linear_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, tw;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_linear_nd: invalid x");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tw) < 0)
        ERR(ctx, "llm_linear_nd: invalid weight");
    if (tx.ndim < 1)
        ERR(ctx, "llm_linear_nd: x must be at least 1-D");
    if (tw.ndim != 2)
        ERR(ctx, "llm_linear_nd: weight must be 2-D");

    int32_t k = tx.shape[tx.ndim - 1];
    if (tw.shape[1] != k)
        ERR(ctx, "llm_linear_nd: in_features mismatch");

    int32_t m = tx.total / k;
    int32_t n = tw.shape[0];
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < tx.ndim; i++)
        out_shape[i] = tx.shape[i];
    out_shape[tx.ndim - 1] = n;
    int32_t out_total = m * n;

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    /* Fast path for M=1 (decode step): use sgemv instead of sgemm.
     * y = W^T * x  where W is (n, k) and x is (k,1)
     * sgemv with Trans: computes y(n) = alpha * W(n,k) * x(k) + beta * y(n)
     * This is ~20-30% faster than sgemm for single-vector case. */
    if (m == 1) {
        cblas_sgemv(CblasRowMajor,
                    CblasNoTrans,
                    n,
                    k,
                    1.0f,
                    tw.data,
                    k,
                    tx.data,
                    1,
                    0.0f,
                    dst,
                    1);
    } else {
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    m,
                    n,
                    k,
                    1.0f,
                    tx.data,
                    k,
                    tw.data,
                    k,
                    0.0f,
                    dst,
                    n);
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── Broadcasting element-wise helpers ───────────────────────────────────── */

/* Compute broadcast output shape for two tensors; return total or -1 on error.
 */
static int32_t broadcast_shape(const TensorNDF32 *ta,
                               const TensorNDF32 *tb,
                               int32_t *out_shape,
                               int *out_ndim) {
    int ndim = ta->ndim > tb->ndim ? ta->ndim : tb->ndim;
    *out_ndim = ndim;
    int off_a = ndim - ta->ndim;
    int off_b = ndim - tb->ndim;
    int32_t total = 1;
    for (int i = 0; i < ndim; i++) {
        int32_t da = (i >= off_a) ? ta->shape[i - off_a] : 1;
        int32_t db = (i >= off_b) ? tb->shape[i - off_b] : 1;
        if (da != db && da != 1 && db != 1)
            return -1;
        out_shape[i] = (da > db) ? da : db;
        total *= out_shape[i];
    }
    return total;
}

/* Flat broadcast iteration: for each output element, fetch from a and b. */
static void broadcast_op(const TensorNDF32 *ta,
                         const TensorNDF32 *tb,
                         float *dst,
                         int32_t total,
                         const int32_t *out_shape,
                         int ndim,
                         int op) {
    /* 0=add, 1=mul, 2=sub, 3=div, 4=gt, 5=lt, 6=ge, 7=le, 8=eq, 9=ne */
    int off_a = ndim - ta->ndim;
    int off_b = ndim - tb->ndim;

    int32_t sa[LLM_MAX_NDIM], sb[LLM_MAX_NDIM];
    /* Padded source shapes */
    int32_t pa[LLM_MAX_NDIM], pb[LLM_MAX_NDIM];
    for (int i = 0; i < off_a; i++)
        pa[i] = 1;
    for (int i = 0; i < ta->ndim; i++)
        pa[i + off_a] = ta->shape[i];
    for (int i = 0; i < off_b; i++)
        pb[i] = 1;
    for (int i = 0; i < tb->ndim; i++)
        pb[i + off_b] = tb->shape[i];

    nd_strides(pa, ndim, sa);
    nd_strides(pb, ndim, sb);
    for (int i = 0; i < ndim; i++) {
        if (pa[i] == 1)
            sa[i] = 0;
        if (pb[i] == 1)
            sb[i] = 0;
    }

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < total; flat++) {
        int32_t oa = 0, ob = 0;
        for (int d = 0; d < ndim; d++) {
            oa += idx[d] * sa[d];
            ob += idx[d] * sb[d];
        }
        float va = ta->data[oa], vb = tb->data[ob];
        if (op == 0)
            dst[flat] = va + vb;
        else if (op == 1)
            dst[flat] = va * vb;
        else if (op == 2)
            dst[flat] = va - vb;
        else if (op == 3)
            dst[flat] = (vb != 0.0f) ? va / vb : 0.0f; /* div-by-zero → 0 */
        else if (op == 4)
            dst[flat] = llm_cmp_mask_to_fp32(va > vb);
        else if (op == 5)
            dst[flat] = llm_cmp_mask_to_fp32(va < vb);
        else if (op == 6)
            dst[flat] = llm_cmp_mask_to_fp32(va >= vb);
        else if (op == 7)
            dst[flat] = llm_cmp_mask_to_fp32(va <= vb);
        else if (op == 8)
            dst[flat] = llm_cmp_mask_to_fp32(va == vb);
        else if (op == 9)
            dst[flat] = llm_cmp_mask_to_fp32(va != vb);
        for (int d = ndim - 1; d >= 0; d--) {
            if (++idx[d] < out_shape[d])
                break;
            idx[d] = 0;
        }
    }
}

static void
sql_cmp_nd_impl(sqlite3_context *ctx, sqlite3_value **argv, int op) {
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_cmp_nd: invalid a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_cmp_nd: invalid b");
    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim;
    int32_t total = broadcast_shape(&ta, &tb, out_shape, &out_ndim);
    if (total < 0)
        ERR(ctx, "llm_cmp_nd: incompatible shapes");
    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    broadcast_op(&ta,
                 &tb,
                 (float *)((char *)out + LLM_ND_HDR(out_ndim)),
                 total,
                 out_shape,
                 out_ndim,
                 op);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

static void sql_add_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_add_nd: invalid a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_add_nd: invalid b");
    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim;
    int32_t total = broadcast_shape(&ta, &tb, out_shape, &out_ndim);
    if (total < 0)
        ERR(ctx, "llm_add_nd: incompatible shapes");
    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    broadcast_op(&ta,
                 &tb,
                 (float *)((char *)out + LLM_ND_HDR(out_ndim)),
                 total,
                 out_shape,
                 out_ndim,
                 0);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

static void sql_mul_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_mul_nd: invalid a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_mul_nd: invalid b");
    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim;
    int32_t total = broadcast_shape(&ta, &tb, out_shape, &out_ndim);
    if (total < 0)
        ERR(ctx, "llm_mul_nd: incompatible shapes");
    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    broadcast_op(&ta,
                 &tb,
                 (float *)((char *)out + LLM_ND_HDR(out_ndim)),
                 total,
                 out_shape,
                 out_ndim,
                 1);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

static void sql_sub_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_sub_nd: invalid a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_sub_nd: invalid b");
    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim;
    int32_t total = broadcast_shape(&ta, &tb, out_shape, &out_ndim);
    if (total < 0)
        ERR(ctx, "llm_sub_nd: incompatible shapes");
    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    broadcast_op(&ta,
                 &tb,
                 (float *)((char *)out + LLM_ND_HDR(out_ndim)),
                 total,
                 out_shape,
                 out_ndim,
                 2);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_softmax_nd(blob, dim) ───────────────────────────────────────────── */
static void
sql_softmax_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_softmax_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_softmax_nd: dim out of range");

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    memcpy(dst, t.data, (size_t)t.total * sizeof(float));

    /* Number of "outer" slices and size of each softmax row */
    int32_t strides[LLM_MAX_NDIM];
    nd_strides(t.shape, t.ndim, strides);
    int32_t dim_sz = t.shape[dim];
    int32_t dim_str = strides[dim];

    /* Iterate over every slice except along dim */
    int32_t outer = t.total / dim_sz;
    /* Build a flat iteration that skips the target dim */
    int32_t other_shape[LLM_MAX_NDIM];
    int32_t
        other_strides[LLM_MAX_NDIM]; /* strides in *output* for non-dim axes */
    int on = 0;
    for (int d = 0; d < t.ndim; d++) {
        if (d != dim) {
            other_shape[on] = t.shape[d];
            other_strides[on] = strides[d];
            on++;
        }
    }

    int32_t oidx[LLM_MAX_NDIM];
    memset(oidx, 0, sizeof(oidx));
    for (int32_t slice = 0; slice < outer; slice++) {
        /* Base offset for this slice */
        int32_t base = 0;
        for (int i = 0; i < on; i++)
            base += oidx[i] * other_strides[i];
        /* Numerically stable softmax over dim */
        float mx = -FLT_MAX;
        for (int32_t k = 0; k < dim_sz; k++) {
            float v = dst[base + k * dim_str];
            if (v > mx)
                mx = v;
        }
        float s = 0.0f;
        for (int32_t k = 0; k < dim_sz; k++) {
            float e = expf(dst[base + k * dim_str] - mx);
            dst[base + k * dim_str] = e;
            s += e;
        }
        for (int32_t k = 0; k < dim_sz; k++)
            dst[base + k * dim_str] /= s;
        /* Increment outer multi-index */
        for (int i = on - 1; i >= 0; i--) {
            if (++oidx[i] < other_shape[i])
                break;
            oidx[i] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_mean_nd(blob, dim, keepdim) ─────────────────────────────────────── */
static void sql_mean_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_mean_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    int keepdim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_mean_nd: dim out of range");

    int32_t new_shape[LLM_MAX_NDIM];
    int new_ndim;
    if (keepdim) {
        new_ndim = t.ndim;
        for (int i = 0; i < t.ndim; i++)
            new_shape[i] = t.shape[i];
        new_shape[dim] = 1;
    } else {
        new_ndim = t.ndim - 1;
        int j = 0;
        for (int i = 0; i < t.ndim; i++)
            if (i != dim)
                new_shape[j++] = t.shape[i];
    }
    int32_t new_total = 1;
    for (int i = 0; i < new_ndim; i++)
        new_total *= new_shape[i];

    int out_sz;
    void *out = nd_result_alloc(new_ndim, new_shape, new_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(new_ndim));
    memset(dst, 0, (size_t)new_total * sizeof(float));

    int32_t strides[LLM_MAX_NDIM];
    nd_strides(t.shape, t.ndim, strides);
    int32_t dim_sz = t.shape[dim];

    /* Output strides in the keepdim or no-keepdim shape */
    int32_t out_strides[LLM_MAX_NDIM];
    nd_strides(new_shape, new_ndim, out_strides);

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < t.total; flat++) {
        /* Map input flat index → output flat index (drop/collapse dim) */
        int32_t out_off = 0;
        if (keepdim) {
            for (int d = 0; d < t.ndim; d++) {
                int32_t coord = (d == dim) ? 0 : idx[d];
                out_off += coord * out_strides[d];
            }
        } else {
            int j = 0;
            for (int d = 0; d < t.ndim; d++) {
                if (d == dim)
                    continue;
                out_off += idx[d] * out_strides[j++];
            }
        }
        dst[out_off] += t.data[flat];
        for (int d = t.ndim - 1; d >= 0; d--) {
            if (++idx[d] < t.shape[d])
                break;
            idx[d] = 0;
        }
    }
    for (int32_t i = 0; i < new_total; i++)
        dst[i] /= (float)dim_sz;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_sum_nd(blob, dim, keepdim) ─────────────────────────────────────── */
static void sql_sum_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_sum_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    int keepdim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_sum_nd: dim out of range");

    int32_t new_shape[LLM_MAX_NDIM];
    int new_ndim;
    if (keepdim) {
        new_ndim = t.ndim;
        for (int i = 0; i < t.ndim; i++)
            new_shape[i] = t.shape[i];
        new_shape[dim] = 1;
    } else {
        new_ndim = t.ndim - 1;
        int j = 0;
        for (int i = 0; i < t.ndim; i++)
            if (i != dim)
                new_shape[j++] = t.shape[i];
    }
    int32_t new_total = 1;
    for (int i = 0; i < new_ndim; i++)
        new_total *= new_shape[i];

    int out_sz;
    void *out = nd_result_alloc(new_ndim, new_shape, new_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(new_ndim));
    memset(dst, 0, (size_t)new_total * sizeof(float));

    int32_t strides[LLM_MAX_NDIM];
    nd_strides(t.shape, t.ndim, strides);
    int32_t out_strides[LLM_MAX_NDIM];
    nd_strides(new_shape, new_ndim, out_strides);

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < t.total; flat++) {
        int32_t out_off = 0;
        if (keepdim) {
            for (int d = 0; d < t.ndim; d++) {
                int32_t coord = (d == dim) ? 0 : idx[d];
                out_off += coord * out_strides[d];
            }
        } else {
            int j = 0;
            for (int d = 0; d < t.ndim; d++) {
                if (d == dim)
                    continue;
                out_off += idx[d] * out_strides[j++];
            }
        }
        dst[out_off] += t.data[flat];
        for (int d = t.ndim - 1; d >= 0; d--) {
            if (++idx[d] < t.shape[d])
                break;
            idx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_str_nd(blob) ───────────────────────────────────────────────────────
 */
static void sql_str_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0) {
        sqlite3_result_text(ctx, "<invalid blob>", -1, SQLITE_STATIC);
        return;
    }
    char buf[512];
    int off = snprintf(buf, sizeof(buf), "tensor(shape=[");
    if (off < 0)
        off = 0;
    if ((size_t)off >= sizeof(buf))
        off = (int)(sizeof(buf) - 1);
    for (int i = 0; i < t.ndim; i++) {
        int n = snprintf(buf + off,
                         sizeof(buf) - (size_t)off,
                         "%d%s",
                         t.shape[i],
                         i < t.ndim - 1 ? "," : "");
        if (n > 0)
            off += n;
        if ((size_t)off >= sizeof(buf)) {
            off = (int)(sizeof(buf) - 1);
            break;
        }
    }
    if ((size_t)off < sizeof(buf) - 1) {
        int n = snprintf(buf + off, sizeof(buf) - (size_t)off, "] data=[");
        if (n > 0)
            off += n;
        if ((size_t)off >= sizeof(buf))
            off = (int)(sizeof(buf) - 1);
    }
    int show = t.total < 6 ? t.total : 6;
    for (int i = 0; i < show; i++) {
        if ((size_t)off >= sizeof(buf) - 1) {
            off = (int)(sizeof(buf) - 1);
            break;
        }
        int n = snprintf(buf + off,
                         sizeof(buf) - (size_t)off,
                         "%.4g%s",
                         t.data[i],
                         i < show - 1 ? "," : "");
        if (n > 0)
            off += n;
        if ((size_t)off >= sizeof(buf)) {
            off = (int)(sizeof(buf) - 1);
            break;
        }
    }
    if (t.total > 6 && (size_t)off < sizeof(buf) - 1) {
        int n = snprintf(buf + off, sizeof(buf) - (size_t)off, ",...");
        if (n > 0)
            off += n;
        if ((size_t)off >= sizeof(buf))
            off = (int)(sizeof(buf) - 1);
    }
    if ((size_t)off < sizeof(buf) - 1)
        snprintf(buf + off, sizeof(buf) - (size_t)off, "])");
    sqlite3_result_text(ctx, buf, -1, SQLITE_TRANSIENT);
}

/* ── llm_div_nd(a, b) ────────────────────────────────────────────────────── */
static void sql_div_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_div_nd: invalid a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_div_nd: invalid b");
    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim;
    int32_t total = broadcast_shape(&ta, &tb, out_shape, &out_ndim);
    if (total < 0)
        ERR(ctx, "llm_div_nd: incompatible shapes");
    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    broadcast_op(&ta,
                 &tb,
                 (float *)((char *)out + LLM_ND_HDR(out_ndim)),
                 total,
                 out_shape,
                 out_ndim,
                 3);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

static void sql_gt_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_cmp_nd_impl(ctx, argv, 4);
}

static void sql_lt_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_cmp_nd_impl(ctx, argv, 5);
}

static void sql_ge_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_cmp_nd_impl(ctx, argv, 6);
}

static void sql_le_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_cmp_nd_impl(ctx, argv, 7);
}

static void sql_eq_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_cmp_nd_impl(ctx, argv, 8);
}

static void sql_ne_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    sql_cmp_nd_impl(ctx, argv, 9);
}

/* ── llm_neg_nd(blob) ───────────────────────────────────────────────────── */
static void sql_neg_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_scale_nd(blob, scalar) ─────────────────────────────────────────── */
static void sql_scale_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_rsqrt_nd(blob) ─────────────────────────────────────────────────── */
static void sql_rsqrt_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_silu_nd(blob) ──────────────────────────────────────────────────── */
static void sql_silu_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_silu_mul_nd(gate, up) ── fused SiLU-gate: silu(gate) * up ──────── */
static void
sql_silu_mul_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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
            "llm_silu_mul_nd: gate and up must have the same number of "
            "elements");
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

/* ── llm_rope_nd(x, cos, sin) ── fused HuggingFace-style RoPE ───────────── *
 *  Computes: out = x * cos + rotate_half(x) * sin                           *
 *  where rotate_half(x)[..., :D/2] = -x[..., D/2:]                          *
 *        rotate_half(x)[..., D/2:] =  x[..., :D/2]                          *
 *                                                                            *
 *  cos and sin may have fewer "rows" than x (broadcast along head dim).      *
 *  Requirement: x.shape[-1] == cos.shape[-1] == sin.shape[-1] (= head_dim). *
 * =========================================================================*/
static void sql_rope_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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
    int32_t ratio = n_rows_x / n_rows_cs; /* broadcast ratio */
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

/* ── llm_exp_nd(blob) ───────────────────────────────────────────────────── */
static void sql_exp_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_cos_nd(blob) ───────────────────────────────────────────────────── */
static void sql_cos_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_sin_nd(blob) ───────────────────────────────────────────────────── */
static void sql_sin_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_tanh_nd(blob) ──────────────────────────────────────────────────── */
static void sql_tanh_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_pow_nd(blob, exponent) ─────────────────────────────────────────── */
static void sql_pow_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_abs_nd(blob) ───────────────────────────────────────────────────── */
static void sql_abs_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_sqrt_nd(blob) ──────────────────────────────────────────────────── */
static void sql_sqrt_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_log_nd(blob) ───────────────────────────────────────────────────── */
static void sql_log_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_sigmoid_nd(blob) ───────────────────────────────────────────────── */
static void
sql_sigmoid_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_relu_nd(blob) ──────────────────────────────────────────────────── */
static void sql_relu_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_gelu_nd(blob) ──────────────────────────────────────────────────── */
static void sql_gelu_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_floor_nd(blob) ─────────────────────────────────────────────────── */
static void sql_floor_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_ceil_nd(blob) ──────────────────────────────────────────────────── */
static void sql_ceil_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_round_nd(blob) ─────────────────────────────────────────────────── */
static void sql_round_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_trunc_nd(blob) ─────────────────────────────────────────────────── */
static void sql_trunc_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_sign_nd(blob) ──────────────────────────────────────────────────── */
static void sql_sign_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_clamp_nd(blob, min_val, max_val) ───────────────────────────────── */
/* Pass NULL for min_val or max_val to skip that bound.                      */
static void sql_clamp_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
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

/* ── llm_tril_nd(blob, diagonal) ────────────────────────────────────────── */
static void sql_tril_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_tril_nd: invalid blob");
    if (t.ndim < 2)
        ERR(ctx, "llm_tril_nd: need at least 2-D tensor");
    int diag = sqlite3_value_int(argv[1]);
    int rows = t.shape[t.ndim - 2];
    int cols = t.shape[t.ndim - 1];
    int mat_size = rows * cols;
    int batch = t.total / mat_size;
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int b = 0; b < batch; b++) {
        const float *src_mat = t.data + b * mat_size;
        float *dst_mat = dst + b * mat_size;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                dst_mat[r * cols + c] =
                    (c <= r + diag) ? src_mat[r * cols + c] : 0.0f;
            }
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_logical_not_nd(blob) ───────────────────────────────────────────── */
static void
sql_logical_not_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_logical_not_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = (t.data[i] == 0.0f) ? 1.0f : 0.0f;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_logical_and_nd(a, b) ───────────────────────────────────────────── */
static void
sql_logical_and_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_logical_and_nd: invalid blob a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_logical_and_nd: invalid blob b");
    if (ta.total != tb.total)
        ERR(ctx, "llm_logical_and_nd: size mismatch");
    int out_sz;
    void *out = nd_result_alloc(ta.ndim, ta.shape, ta.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ta.ndim));
    for (int32_t i = 0; i < ta.total; i++)
        dst[i] = (ta.data[i] != 0.0f && tb.data[i] != 0.0f) ? 1.0f : 0.0f;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_logical_or_nd(a, b) ────────────────────────────────────────────── */
static void
sql_logical_or_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_logical_or_nd: invalid blob a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_logical_or_nd: invalid blob b");
    if (ta.total != tb.total)
        ERR(ctx, "llm_logical_or_nd: size mismatch");
    int out_sz;
    void *out = nd_result_alloc(ta.ndim, ta.shape, ta.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ta.ndim));
    for (int32_t i = 0; i < ta.total; i++)
        dst[i] = (ta.data[i] != 0.0f || tb.data[i] != 0.0f) ? 1.0f : 0.0f;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_bitwise_not_nd(blob) ───────────────────────────────────────────── */
static void
sql_bitwise_not_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_bitwise_not_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = (t.data[i] == 0.0f) ? 1.0f : 0.0f;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_bitwise_and_nd(a, b) ───────────────────────────────────────────── */
static void
sql_bitwise_and_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_bitwise_and_nd: invalid blob a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_bitwise_and_nd: invalid blob b");
    if (ta.total != tb.total)
        ERR(ctx, "llm_bitwise_and_nd: size mismatch");
    int out_sz;
    void *out = nd_result_alloc(ta.ndim, ta.shape, ta.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ta.ndim));
    for (int32_t i = 0; i < ta.total; i++)
        dst[i] = (ta.data[i] != 0.0f && tb.data[i] != 0.0f) ? 1.0f : 0.0f;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_bitwise_or_nd(a, b) ────────────────────────────────────────────── */
static void
sql_bitwise_or_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_bitwise_or_nd: invalid blob a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_bitwise_or_nd: invalid blob b");
    if (ta.total != tb.total)
        ERR(ctx, "llm_bitwise_or_nd: size mismatch");
    int out_sz;
    void *out = nd_result_alloc(ta.ndim, ta.shape, ta.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ta.ndim));
    for (int32_t i = 0; i < ta.total; i++)
        dst[i] = (ta.data[i] != 0.0f || tb.data[i] != 0.0f) ? 1.0f : 0.0f;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_bitwise_and_scalar_nd(blob, scalar) ────────────────────────────── */
static void sql_bitwise_and_scalar_nd(sqlite3_context *ctx,
                                      int argc,
                                      sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_bitwise_and_scalar_nd: invalid blob");
    int64_t s = sqlite3_value_int64(argv[1]);
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = (float)((int64_t)t.data[i] & s);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_pow_scalar_nd(base_scalar, exponent_blob) ──────────────────────── */
/* Computes scalar^tensor element-wise.                                      */
static void
sql_pow_scalar_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    float base = (float)sqlite3_value_double(argv[0]);
    const void *blob = sqlite3_value_blob(argv[1]);
    int sz = sqlite3_value_bytes(argv[1]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_pow_scalar_nd: invalid blob");
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = powf(base, t.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_floor_div_nd(a, b) ─────────────────────────────────────────────── */
static void
sql_floor_div_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_floor_div_nd: invalid blob a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_floor_div_nd: invalid blob b");
    if (ta.total != tb.total)
        ERR(ctx, "llm_floor_div_nd: size mismatch");
    int out_sz;
    void *out = nd_result_alloc(ta.ndim, ta.shape, ta.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ta.ndim));
    for (int32_t i = 0; i < ta.total; i++)
        dst[i] = floorf(ta.data[i] / tb.data[i]);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_remainder_nd(a, b_scalar) ──────────────────────────────────────── */
static void
sql_remainder_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_remainder_nd: invalid blob");
    float b = (float)sqlite3_value_double(argv[1]);
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = fmodf(t.data[i], b) +
                 (fmodf(t.data[i], b) != 0.0f && ((t.data[i] < 0) != (b < 0))
                      ? b
                      : 0.0f);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_remainder_tensor_nd(a, b) ──────────────────────────────────────── */
static void
sql_remainder_tensor_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 ta, tb;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &ta) < 0)
        ERR(ctx, "llm_remainder_tensor_nd: invalid blob a");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tb) < 0)
        ERR(ctx, "llm_remainder_tensor_nd: invalid blob b");
    if (ta.total != tb.total)
        ERR(ctx, "llm_remainder_tensor_nd: size mismatch");
    int out_sz;
    void *out = nd_result_alloc(ta.ndim, ta.shape, ta.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ta.ndim));
    for (int32_t i = 0; i < ta.total; i++) {
        float r = fmodf(ta.data[i], tb.data[i]);
        if (r != 0.0f && ((ta.data[i] < 0) != (tb.data[i] < 0)))
            r += tb.data[i];
        dst[i] = r;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_reduce_sum_nd(blob) — global sum → scalar blob ─────────────────── */
static void
sql_reduce_sum_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_reduce_sum_nd: invalid blob");
    double acc = 0.0;
    for (int32_t i = 0; i < t.total; i++)
        acc += (double)t.data[i];
    int32_t shape1 = 1;
    int out_sz;
    void *out = nd_result_alloc(1, &shape1, 1, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(1));
    dst[0] = (float)acc;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_reduce_mean_nd(blob) — global mean → scalar blob ───────────────── */
static void
sql_reduce_mean_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_reduce_mean_nd: invalid blob");
    double acc = 0.0;
    for (int32_t i = 0; i < t.total; i++)
        acc += (double)t.data[i];
    int32_t shape1 = 1;
    int out_sz;
    void *out = nd_result_alloc(1, &shape1, 1, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(1));
    dst[0] = (float)(acc / (double)t.total);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_reduce_max_nd(blob) — global max → scalar blob ─────────────────── */
static void
sql_reduce_max_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_reduce_max_nd: invalid blob");
    float mx = -FLT_MAX;
    for (int32_t i = 0; i < t.total; i++)
        if (t.data[i] > mx)
            mx = t.data[i];
    int32_t shape1 = 1;
    int out_sz;
    void *out = nd_result_alloc(1, &shape1, 1, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(1));
    dst[0] = mx;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_reduce_min_nd(blob) — global min → scalar blob ─────────────────── */
static void
sql_reduce_min_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_reduce_min_nd: invalid blob");
    float mn = FLT_MAX;
    for (int32_t i = 0; i < t.total; i++)
        if (t.data[i] < mn)
            mn = t.data[i];
    int32_t shape1 = 1;
    int out_sz;
    void *out = nd_result_alloc(1, &shape1, 1, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(1));
    dst[0] = mn;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_item_nd(blob, flat_index) — extract scalar float at flat index ──── */
static void sql_item_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_item_nd: invalid blob");
    int32_t idx = sqlite3_value_int(argv[1]);
    if (idx < 0)
        idx += t.total;
    if (idx < 0 || idx >= t.total)
        ERR(ctx, "llm_item_nd: index out of range");
    sqlite3_result_double(ctx, (double)t.data[idx]);
}

/* ── llm_reduce_argmax_nd(blob) — global argmax → int ───────────────────── */
static void
sql_reduce_argmax_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_reduce_argmax_nd: invalid blob");
    float mx = -FLT_MAX;
    int32_t idx = 0;
    for (int32_t i = 0; i < t.total; i++) {
        if (t.data[i] > mx) {
            mx = t.data[i];
            idx = i;
        }
    }
    sqlite3_result_int(ctx, idx);
}

/* ── llm_max_dim_nd(blob, dim, keepdim) — max along dim → (values, indices) */
/* Returns just the values blob; indices returned via separate call.          */
static void
sql_max_dim_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_max_dim_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    int keepdim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_max_dim_nd: dim out of range");

    /* Compute strides */
    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    /* Output shape */
    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim = 0;
    for (int i = 0; i < t.ndim; i++) {
        if (i == dim) {
            if (keepdim)
                out_shape[out_ndim++] = 1;
        } else {
            out_shape[out_ndim++] = t.shape[i];
        }
    }
    if (out_ndim == 0) {
        out_shape[0] = 1;
        out_ndim = 1;
    }
    int32_t out_total = outer * inner;

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    for (int32_t o = 0; o < outer; o++) {
        for (int32_t n = 0; n < inner; n++) {
            float mx = -FLT_MAX;
            for (int32_t d = 0; d < dimsize; d++) {
                float v = t.data[(o * dimsize + d) * inner + n];
                if (v > mx)
                    mx = v;
            }
            dst[o * inner + n] = mx;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_argmax_dim_nd(blob, dim, keepdim) — argmax along dim ───────────── */
static void
sql_argmax_dim_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_argmax_dim_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    int keepdim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_argmax_dim_nd: dim out of range");

    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim = 0;
    for (int i = 0; i < t.ndim; i++) {
        if (i == dim) {
            if (keepdim)
                out_shape[out_ndim++] = 1;
        } else {
            out_shape[out_ndim++] = t.shape[i];
        }
    }
    if (out_ndim == 0) {
        out_shape[0] = 1;
        out_ndim = 1;
    }
    int32_t out_total = outer * inner;

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    for (int32_t o = 0; o < outer; o++) {
        for (int32_t n = 0; n < inner; n++) {
            float mx = -FLT_MAX;
            int32_t best = 0;
            for (int32_t d = 0; d < dimsize; d++) {
                float v = t.data[(o * dimsize + d) * inner + n];
                if (v > mx) {
                    mx = v;
                    best = d;
                }
            }
            dst[o * inner + n] = (float)best;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_layernorm_nd(x, weight, bias, eps) ─────────────────────────────── */
/* Full layer norm: y = (x - mean) / sqrt(var + eps) * weight + bias         */
/* weight is 1-D, bias can be a blob or NULL. Returns result blob.           */
static void
sql_layernorm_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, tw;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_layernorm_nd: invalid x blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tw) < 0)
        ERR(ctx, "llm_layernorm_nd: invalid weight blob");
    if (tw.ndim != 1)
        ERR(ctx, "llm_layernorm_nd: weight must be 1-D");

    int has_bias = (sqlite3_value_type(argv[2]) != SQLITE_NULL);
    TensorNDF32 tb;
    memset(&tb, 0, sizeof(tb));
    if (has_bias) {
        if (llm_to_nd(sqlite3_value_blob(argv[2]),
                      sqlite3_value_bytes(argv[2]),
                      &tb) < 0)
            ERR(ctx, "llm_layernorm_nd: invalid bias blob");
    }

    float eps = (float)sqlite3_value_double(argv[3]);
    int32_t norm_size = tw.shape[0];
    if (tx.total % norm_size != 0)
        ERR(ctx, "llm_layernorm_nd: size mismatch");
    int32_t outer = tx.total / norm_size;

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, tx.shape, tx.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    for (int32_t r = 0; r < outer; r++) {
        const float *src = tx.data + r * norm_size;
        float *d = dst + r * norm_size;
        /* compute mean */
        double sum = 0.0;
        for (int32_t j = 0; j < norm_size; j++)
            sum += (double)src[j];
        double mean = sum / (double)norm_size;
        /* compute variance */
        double var = 0.0;
        for (int32_t j = 0; j < norm_size; j++) {
            double diff = (double)src[j] - mean;
            var += diff * diff;
        }
        var /= (double)norm_size;
        double rstd = 1.0 / sqrt(var + (double)eps);
        /* normalize */
        for (int32_t j = 0; j < norm_size; j++) {
            double normed = ((double)src[j] - mean) * rstd;
            d[j] = (float)(normed * (double)tw.data[j]);
            if (has_bias)
                d[j] += tb.data[j];
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_embedding_nd(weight, indices) ──────────────────────────────────── */
static void
sql_embedding_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tw, ti;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tw) < 0)
        ERR(ctx, "llm_embedding_nd: invalid weight blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &ti) < 0)
        ERR(ctx, "llm_embedding_nd: invalid indices blob");
    if (tw.ndim != 2)
        ERR(ctx, "llm_embedding_nd: weight must be 2-D");

    int32_t vocab = tw.shape[0];
    int32_t embed = tw.shape[1];

    /* Output shape = [...indices.shape, embed_dim] */
    int out_ndim = ti.ndim + 1;
    if (out_ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_embedding_nd: too many dims");
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < ti.ndim; i++)
        out_shape[i] = ti.shape[i];
    out_shape[ti.ndim] = embed;
    int32_t out_total = ti.total * embed;

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    for (int32_t i = 0; i < ti.total; i++) {
        int idx = (int)roundf(ti.data[i]);
        if (idx < 0 || idx >= vocab)
            ERR(ctx, "llm_embedding_nd: index out of range");
        memcpy(dst + (size_t)i * embed,
               tw.data + (size_t)idx * embed,
               (size_t)embed * sizeof(float));
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_rmsnorm_nd(x, weight, eps) ─────────────────────────────────────── */
static void
sql_rmsnorm_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, tw;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_rmsnorm_nd: invalid x blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tw) < 0)
        ERR(ctx, "llm_rmsnorm_nd: invalid weight blob");
    if (tw.ndim != 1)
        ERR(ctx, "llm_rmsnorm_nd: weight must be 1-D");
    float eps = (float)sqlite3_value_double(argv[2]);

    int32_t last = tx.shape[tx.ndim - 1];
    if (tw.shape[0] != last)
        ERR(ctx, "llm_rmsnorm_nd: weight size mismatch");

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, tx.shape, tx.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    int32_t nrows = tx.total / last;
    for (int32_t r = 0; r < nrows; r++) {
        const float *row = tx.data + (size_t)r * last;
        float *out_row = dst + (size_t)r * last;
        float ss = 0.0f;
        for (int32_t j = 0; j < last; j++)
            ss += row[j] * row[j];
        float rms = 1.0f / sqrtf(ss / (float)last + eps);
        for (int32_t j = 0; j < last; j++)
            out_row[j] = row[j] * rms * tw.data[j];
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_triu_nd(blob, diagonal) ────────────────────────────────────────── */
static void sql_triu_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_triu_nd: invalid blob");
    if (t.ndim != 2)
        ERR(ctx, "llm_triu_nd: only 2-D tensors supported");
    int k = sqlite3_value_int(argv[1]);
    int32_t rows = t.shape[0], cols = t.shape[1];

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t r = 0; r < rows; r++)
        for (int32_t c = 0; c < cols; c++)
            dst[r * cols + c] = (c >= r + k) ? t.data[r * cols + c] : 0.0f;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_where_nd(cond, x, y) ───────────────────────────────────────────── */
static void sql_where_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tc, tx, ty;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tc) < 0)
        ERR(ctx, "llm_where_nd: invalid cond blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tx) < 0)
        ERR(ctx, "llm_where_nd: invalid x blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &ty) < 0)
        ERR(ctx, "llm_where_nd: invalid y blob");

    /* Broadcast x and y first, then broadcast result with cond */
    int32_t xy_shape[LLM_MAX_NDIM];
    int xy_ndim;
    int32_t xy_total = broadcast_shape(&tx, &ty, xy_shape, &xy_ndim);
    if (xy_total < 0)
        ERR(ctx, "llm_where_nd: x/y shapes incompatible");

    /* Build a temporary TensorNDF32 for the xy broadcast shape to broadcast
     * with cond */
    TensorNDF32 txy;
    txy.ndim = xy_ndim;
    for (int i = 0; i < xy_ndim; i++)
        txy.shape[i] = xy_shape[i];
    txy.total = xy_total;

    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim;
    int32_t total = broadcast_shape(&tc, &txy, out_shape, &out_ndim);
    if (total < 0)
        ERR(ctx, "llm_where_nd: cond shape incompatible");

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    /* Compute padded strides for cond, x, y with broadcast */
    int off_c = out_ndim - tc.ndim;
    int off_x = out_ndim - tx.ndim;
    int off_y = out_ndim - ty.ndim;

    int32_t pc[LLM_MAX_NDIM], px[LLM_MAX_NDIM], py[LLM_MAX_NDIM];
    int32_t sc[LLM_MAX_NDIM], sx[LLM_MAX_NDIM], sy[LLM_MAX_NDIM];
    for (int i = 0; i < off_c; i++)
        pc[i] = 1;
    for (int i = 0; i < tc.ndim; i++)
        pc[i + off_c] = tc.shape[i];
    for (int i = 0; i < off_x; i++)
        px[i] = 1;
    for (int i = 0; i < tx.ndim; i++)
        px[i + off_x] = tx.shape[i];
    for (int i = 0; i < off_y; i++)
        py[i] = 1;
    for (int i = 0; i < ty.ndim; i++)
        py[i + off_y] = ty.shape[i];

    nd_strides(pc, out_ndim, sc);
    nd_strides(px, out_ndim, sx);
    nd_strides(py, out_ndim, sy);
    for (int i = 0; i < out_ndim; i++) {
        if (pc[i] == 1)
            sc[i] = 0;
        if (px[i] == 1)
            sx[i] = 0;
        if (py[i] == 1)
            sy[i] = 0;
    }

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < total; flat++) {
        int32_t oc = 0, ox = 0, oy = 0;
        for (int d = 0; d < out_ndim; d++) {
            oc += idx[d] * sc[d];
            ox += idx[d] * sx[d];
            oy += idx[d] * sy[d];
        }
        dst[flat] = (tc.data[oc] > 0.5f)
                        ? tx.data[ox]
                        : ty.data[oy]; /* boolean: >0.5 = true */
        for (int d = out_ndim - 1; d >= 0; d--) {
            if (++idx[d] < out_shape[d])
                break;
            idx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_masked_fill_nd(x, mask, value) ─────────────────────────────────── */
static void
sql_masked_fill_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, tm;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_masked_fill_nd: invalid x blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tm) < 0)
        ERR(ctx, "llm_masked_fill_nd: invalid mask blob");
    float val = (float)sqlite3_value_double(argv[2]);

    /* Broadcast x and mask to get output shape */
    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim;
    int32_t total = broadcast_shape(&tx, &tm, out_shape, &out_ndim);
    if (total < 0)
        ERR(ctx, "llm_masked_fill_nd: incompatible shapes");

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    int off_x = out_ndim - tx.ndim;
    int off_m = out_ndim - tm.ndim;
    int32_t px[LLM_MAX_NDIM], pm[LLM_MAX_NDIM];
    int32_t sx[LLM_MAX_NDIM], sm[LLM_MAX_NDIM];
    for (int i = 0; i < off_x; i++)
        px[i] = 1;
    for (int i = 0; i < tx.ndim; i++)
        px[i + off_x] = tx.shape[i];
    for (int i = 0; i < off_m; i++)
        pm[i] = 1;
    for (int i = 0; i < tm.ndim; i++)
        pm[i + off_m] = tm.shape[i];

    nd_strides(px, out_ndim, sx);
    nd_strides(pm, out_ndim, sm);
    for (int i = 0; i < out_ndim; i++) {
        if (px[i] == 1)
            sx[i] = 0;
        if (pm[i] == 1)
            sm[i] = 0;
    }

    int32_t idx[LLM_MAX_NDIM];
    memset(idx, 0, sizeof(idx));
    for (int32_t flat = 0; flat < total; flat++) {
        int32_t ox = 0, om = 0;
        for (int d = 0; d < out_ndim; d++) {
            ox += idx[d] * sx[d];
            om += idx[d] * sm[d];
        }
        dst[flat] = (tm.data[om] > 0.5f)
                        ? val
                        : tx.data[ox]; /* boolean: >0.5 = masked */
        for (int d = out_ndim - 1; d >= 0; d--) {
            if (++idx[d] < out_shape[d])
                break;
            idx[d] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_arange_nd(n) ───────────────────────────────────────────────────── */
static void
sql_arange_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    int32_t n = sqlite3_value_int(argv[0]);
    if (n <= 0)
        ERR(ctx, "llm_arange_nd: n must be > 0");
    int32_t shape[1] = {n};
    int out_sz;
    void *out = nd_result_alloc(1, shape, n, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(1));
    for (int32_t i = 0; i < n; i++)
        dst[i] = (float)i;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_full_nd(value, d0 [, d1, ...]) ─────────────────────────────────── */
static void sql_full_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2)
        ERR(ctx, "llm_full_nd: need value + at least 1 dim");
    float val = (float)sqlite3_value_double(argv[0]);
    int ndim = argc - 1;
    if (ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_full_nd: too many dims");
    int32_t shape[LLM_MAX_NDIM];
    int32_t total = 1;
    for (int i = 0; i < ndim; i++) {
        shape[i] = sqlite3_value_int(argv[i + 1]);
        if (shape[i] <= 0)
            ERR(ctx, "llm_full_nd: dims must be > 0");
        total *= shape[i];
    }
    int out_sz;
    void *out = nd_result_alloc(ndim, shape, total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ndim));
    for (int32_t i = 0; i < total; i++)
        dst[i] = val;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_argmax_nd(blob, dim) ───────────────────────────────────────────── */
static void
sql_argmax_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_argmax_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_argmax_nd: dim out of range");

    /* Output shape: remove the target dimension */
    int32_t new_shape[LLM_MAX_NDIM];
    int new_ndim = t.ndim - 1;
    if (new_ndim < 1) {
        /* Scalar case: 1-D input → output shape [1] */
        new_ndim = 1;
        new_shape[0] = 1;
    } else {
        int j = 0;
        for (int i = 0; i < t.ndim; i++)
            if (i != dim)
                new_shape[j++] = t.shape[i];
    }
    int32_t new_total = 1;
    for (int i = 0; i < new_ndim; i++)
        new_total *= new_shape[i];

    int out_sz;
    void *out = nd_result_alloc(new_ndim, new_shape, new_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(new_ndim));

    int32_t strides[LLM_MAX_NDIM];
    nd_strides(t.shape, t.ndim, strides);
    int32_t dim_sz = t.shape[dim];
    int32_t dim_str = strides[dim];

    /* Build outer iteration strides (all axes except dim) */
    int32_t other_shape[LLM_MAX_NDIM];
    int32_t other_strides[LLM_MAX_NDIM];
    int on = 0;
    for (int d = 0; d < t.ndim; d++) {
        if (d != dim) {
            other_shape[on] = t.shape[d];
            other_strides[on] = strides[d];
            on++;
        }
    }

    int32_t outer = t.total / dim_sz;
    int32_t oidx[LLM_MAX_NDIM];
    memset(oidx, 0, sizeof(oidx));
    for (int32_t slice = 0; slice < outer; slice++) {
        int32_t base = 0;
        for (int i = 0; i < on; i++)
            base += oidx[i] * other_strides[i];
        float best = t.data[base];
        int32_t best_idx = 0;
        for (int32_t k = 1; k < dim_sz; k++) {
            float v = t.data[base + k * dim_str];
            if (v > best) {
                best = v;
                best_idx = k;
            }
        }
        dst[slice] = (float)best_idx;
        for (int i = on - 1; i >= 0; i--) {
            if (++oidx[i] < other_shape[i])
                break;
            oidx[i] = 0;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ===========================================================================
 * Section 13b: Additional N-D ops (cumsum, gather, index_select, repeat, etc.)
 * =========================================================================*/

/* ── llm_cumsum_nd(blob, dim) ───────────────────────────────────────────── */
static void
sql_cumsum_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_cumsum_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_cumsum_nd: dim out of range");

    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    for (int32_t o = 0; o < outer; o++) {
        for (int32_t n = 0; n < inner; n++) {
            double acc = 0.0;
            for (int32_t d = 0; d < dimsize; d++) {
                int32_t idx = (o * dimsize + d) * inner + n;
                acc += (double)t.data[idx];
                dst[idx] = (float)acc;
            }
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_cumprod_nd(blob, dim) ──────────────────────────────────────────── */
static void
sql_cumprod_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_cumprod_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_cumprod_nd: dim out of range");

    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    for (int32_t o = 0; o < outer; o++) {
        for (int32_t n = 0; n < inner; n++) {
            double acc = 1.0;
            for (int32_t d = 0; d < dimsize; d++) {
                int32_t idx = (o * dimsize + d) * inner + n;
                acc *= (double)t.data[idx];
                dst[idx] = (float)acc;
            }
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_gather_nd(x, dim, index) ───────────────────────────────────────── */
static void
sql_gather_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, ti;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_gather_nd: invalid x blob");
    int dim = sqlite3_value_int(argv[1]);
    if (llm_to_nd(
            sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &ti) < 0)
        ERR(ctx, "llm_gather_nd: invalid index blob");
    if (dim < 0)
        dim += tx.ndim;
    if (dim < 0 || dim >= tx.ndim)
        ERR(ctx, "llm_gather_nd: dim out of range");
    if (tx.ndim != ti.ndim)
        ERR(ctx, "llm_gather_nd: ndim mismatch");

    /* output has same shape as index */
    int out_sz;
    void *out = nd_result_alloc(ti.ndim, ti.shape, ti.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ti.ndim));

    int32_t x_strides[LLM_MAX_NDIM], i_strides[LLM_MAX_NDIM];
    nd_strides(tx.shape, tx.ndim, x_strides);
    nd_strides(ti.shape, ti.ndim, i_strides);

    for (int32_t flat = 0; flat < ti.total; flat++) {
        /* Convert flat index to multi-dim */
        int32_t rem = flat;
        int32_t x_off = 0;
        for (int d = 0; d < ti.ndim; d++) {
            int32_t coord = rem / i_strides[d];
            rem %= i_strides[d];
            if (d == dim) {
                int32_t idx_val = (int32_t)ti.data[flat];
                x_off += idx_val * x_strides[d];
            } else {
                x_off += coord * x_strides[d];
            }
        }
        dst[flat] = tx.data[x_off];
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_index_select_nd(x, dim, index_1d) ──────────────────────────────── */
static void
sql_index_select_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, ti;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_index_select_nd: invalid x blob");
    int dim = sqlite3_value_int(argv[1]);
    if (llm_to_nd(
            sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &ti) < 0)
        ERR(ctx, "llm_index_select_nd: invalid index blob");
    if (dim < 0)
        dim += tx.ndim;
    if (dim < 0 || dim >= tx.ndim)
        ERR(ctx, "llm_index_select_nd: dim out of range");

    /* Output shape: same as tx but shape[dim] = len(index) */
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < tx.ndim; i++)
        out_shape[i] = tx.shape[i];
    out_shape[dim] = ti.total;
    int32_t out_total = 1;
    for (int i = 0; i < tx.ndim; i++)
        out_total *= out_shape[i];

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    int32_t outer = 1, inner = 1;
    for (int i = 0; i < dim; i++)
        outer *= tx.shape[i];
    for (int i = dim + 1; i < tx.ndim; i++)
        inner *= tx.shape[i];
    int32_t dim_sz = tx.shape[dim];

    for (int32_t o = 0; o < outer; o++) {
        for (int32_t k = 0; k < ti.total; k++) {
            int32_t idx = (int32_t)ti.data[k];
            if (idx < 0)
                idx += dim_sz;
            const float *src = tx.data + (o * dim_sz + idx) * inner;
            float *d = dst + (o * ti.total + k) * inner;
            memcpy(d, src, (size_t)inner * sizeof(float));
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_scatter_nd(x, dim, index, value_scalar) ───────────────────────── */
static void
sql_scatter_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, ti;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_scatter_nd: invalid x blob");
    int dim = sqlite3_value_int(argv[1]);
    if (llm_to_nd(
            sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &ti) < 0)
        ERR(ctx, "llm_scatter_nd: invalid index blob");
    float val = (float)sqlite3_value_double(argv[3]);
    if (dim < 0)
        dim += tx.ndim;
    if (dim < 0 || dim >= tx.ndim)
        ERR(ctx, "llm_scatter_nd: dim out of range");

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, tx.shape, tx.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));
    memcpy(dst, tx.data, (size_t)tx.total * sizeof(float));

    int32_t x_strides[LLM_MAX_NDIM], i_strides[LLM_MAX_NDIM];
    nd_strides(tx.shape, tx.ndim, x_strides);
    nd_strides(ti.shape, ti.ndim, i_strides);

    for (int32_t flat = 0; flat < ti.total; flat++) {
        int32_t rem = flat;
        int32_t x_off = 0;
        for (int d = 0; d < ti.ndim; d++) {
            int32_t coord = rem / i_strides[d];
            rem %= i_strides[d];
            if (d == dim) {
                int32_t idx_val = (int32_t)ti.data[flat];
                x_off += idx_val * x_strides[d];
            } else {
                x_off += coord * x_strides[d];
            }
        }
        dst[x_off] = val;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_repeat_nd(blob, ...) — variadic: blob, r0, r1, ... ────────────── */
static void
sql_repeat_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2)
        ERR(ctx, "llm_repeat_nd: need blob + at least 1 repeat");
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_repeat_nd: invalid blob");
    int nrep = argc - 1;
    int32_t reps[LLM_MAX_NDIM];
    for (int i = 0; i < nrep; i++)
        reps[i] = sqlite3_value_int(argv[i + 1]);

    /* Output shape */
    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim = nrep > t.ndim ? nrep : t.ndim;
    if (out_ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_repeat_nd: too many dims");
    int32_t out_total = 1;
    for (int i = 0; i < out_ndim; i++) {
        int si = i - (out_ndim - t.ndim);
        int ri = i - (out_ndim - nrep);
        int32_t s = (si >= 0 && si < t.ndim) ? t.shape[si] : 1;
        int32_t r = (ri >= 0 && ri < nrep) ? reps[ri] : 1;
        out_shape[i] = s * r;
        out_total *= out_shape[i];
    }

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    /* For each output element, compute the source element */
    int32_t out_strides[LLM_MAX_NDIM], src_strides[LLM_MAX_NDIM];
    nd_strides(out_shape, out_ndim, out_strides);
    nd_strides(t.shape, t.ndim, src_strides);

    for (int32_t flat = 0; flat < out_total; flat++) {
        int32_t rem = flat;
        int32_t src_off = 0;
        for (int d = 0; d < out_ndim; d++) {
            int32_t coord = rem / out_strides[d];
            rem %= out_strides[d];
            int si = d - (out_ndim - t.ndim);
            if (si >= 0 && si < t.ndim) {
                src_off += (coord % t.shape[si]) * src_strides[si];
            }
        }
        dst[flat] = t.data[src_off];
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_repeat_kv_nd(blob, num_repeats)
 *
 * Efficient KV-head repetition for grouped-query attention (GQA).
 * Input  : 4-D tensor [batch, kv_heads, seq_len, head_dim]
 * Output : 4-D tensor [batch, kv_heads * num_repeats, seq_len, head_dim]
 *
 * Fuses the pattern: unsqueeze(dim=2) → expand(dim=2, n) → reshape
 * Each KV head's data is copied *num_repeats* times via memcpy.
 * ────────────────────────────────────────────────────────────────────── */
static void
sql_repeat_kv_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_repeat_kv_nd: invalid blob");
    int num_repeats = sqlite3_value_int(argv[1]);
    if (num_repeats <= 0)
        ERR(ctx, "llm_repeat_kv_nd: num_repeats must be > 0");
    if (num_repeats == 1) {
        /* Identity — return input unchanged */
        sqlite3_result_blob(ctx, blob, sz, SQLITE_TRANSIENT);
        return;
    }
    if (t.ndim != 4)
        ERR(ctx, "llm_repeat_kv_nd: expected 4-D tensor");

    int32_t batch = t.shape[0];
    int32_t kv_heads = t.shape[1];
    int32_t seq_len = t.shape[2];
    int32_t head_dim = t.shape[3];

    int32_t out_shape[4] = {batch, kv_heads * num_repeats, seq_len, head_dim};
    int32_t out_total = batch * kv_heads * num_repeats * seq_len * head_dim;

    int out_sz;
    void *out = nd_result_alloc(4, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(4));

    /* Each head's data block: seq_len × head_dim contiguous floats */
    int32_t head_elems = seq_len * head_dim;
    int32_t head_bytes = head_elems * (int32_t)sizeof(float);

    for (int32_t b = 0; b < batch; b++) {
        for (int32_t h = 0; h < kv_heads; h++) {
            const float *src_head = t.data + (b * kv_heads + h) * head_elems;
            for (int r = 0; r < num_repeats; r++) {
                memcpy(dst, src_head, head_bytes);
                dst += head_elems;
            }
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_flip_nd(blob, dim) ─────────────────────────────────────────────── */
static void sql_flip_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_flip_nd: invalid blob");
    int dim = sqlite3_value_int(argv[1]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_flip_nd: dim out of range");

    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    for (int32_t o = 0; o < outer; o++) {
        for (int32_t d = 0; d < dimsize; d++) {
            const float *src =
                t.data + (o * dimsize + (dimsize - 1 - d)) * inner;
            float *dd = dst + (o * dimsize + d) * inner;
            memcpy(dd, src, (size_t)inner * sizeof(float));
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_roll_nd(blob, shift, dim) ──────────────────────────────────────── */
static void sql_roll_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_roll_nd: invalid blob");
    int shift = sqlite3_value_int(argv[1]);
    int dim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_roll_nd: dim out of range");

    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    /* Normalize shift */
    shift = ((shift % dimsize) + dimsize) % dimsize;

    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    for (int32_t o = 0; o < outer; o++) {
        for (int32_t d = 0; d < dimsize; d++) {
            int32_t src_d = (d - shift + dimsize) % dimsize;
            const float *src = t.data + (o * dimsize + src_d) * inner;
            float *dd = dst + (o * dimsize + d) * inner;
            memcpy(dd, src, (size_t)inner * sizeof(float));
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_repeat_interleave_nd(blob, repeats_int, dim) ───────────────────── */
static void
sql_repeat_interleave_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_repeat_interleave_nd: invalid blob");
    int32_t reps = sqlite3_value_int(argv[1]);
    int dim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_repeat_interleave_nd: dim out of range");

    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < t.ndim; i++)
        out_shape[i] = t.shape[i];
    out_shape[dim] = dimsize * reps;
    int32_t out_total = 1;
    for (int i = 0; i < t.ndim; i++)
        out_total *= out_shape[i];

    int out_sz;
    void *out = nd_result_alloc(t.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    for (int32_t o = 0; o < outer; o++) {
        for (int32_t d = 0; d < dimsize; d++) {
            const float *src = t.data + (o * dimsize + d) * inner;
            for (int32_t r = 0; r < reps; r++) {
                float *dd = dst + (o * dimsize * reps + d * reps + r) * inner;
                memcpy(dd, src, (size_t)inner * sizeof(float));
            }
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_index_put_nd(x, values, accumulate, idx0, idx1, ...) ──────────── */
/* Variadic: first 3 args are x, values, accumulate (0/1).                   */
/* Remaining args are per-dimension index tensors (BLOB or NULL).            */
/* Implements torch.index_put semantics: for each broadcast position i,      */
/*   offset = sum_d idx[d][i] * stride[d]  (non-NULL dims)                   */
/*   x[offset : offset+tail] = values[i*tail : (i+1)*tail] (or +=)          */
static void
sql_index_put_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 4)
        ERR(ctx, "llm_index_put_nd: need x, values, accumulate, idx...");
    TensorNDF32 tx, tv;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_index_put_nd: invalid x blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tv) < 0)
        ERR(ctx, "llm_index_put_nd: invalid values blob");
    int accumulate = sqlite3_value_int(argv[2]);

    int n_idx_dims = argc - 3;
    if (n_idx_dims > tx.ndim)
        ERR(ctx, "llm_index_put_nd: more indices than x dims");

    /* Parse per-dimension index tensors */
    TensorNDF32 idx_t[LLM_MAX_NDIM];
    int has_idx[LLM_MAX_NDIM];
    for (int d = 0; d < n_idx_dims; d++) {
        if (sqlite3_value_type(argv[3 + d]) == SQLITE_NULL) {
            has_idx[d] = 0;
        } else {
            if (llm_to_nd(sqlite3_value_blob(argv[3 + d]),
                          sqlite3_value_bytes(argv[3 + d]),
                          &idx_t[d]) < 0)
                ERR(ctx, "llm_index_put_nd: invalid index blob");
            has_idx[d] = 1;
        }
    }

    /* Find first non-NULL index to get n_positions */
    int32_t n_positions = 0;
    for (int d = 0; d < n_idx_dims; d++) {
        if (has_idx[d]) {
            n_positions = idx_t[d].total;
            break;
        }
    }
    if (n_positions == 0) {
        /* No tensor indices: return x unchanged */
        sqlite3_result_blob(ctx,
                            sqlite3_value_blob(argv[0]),
                            sqlite3_value_bytes(argv[0]),
                            SQLITE_TRANSIENT);
        return;
    }

    /* Compute strides of x for the indexed dimensions */
    int32_t x_strides[LLM_MAX_NDIM];
    nd_strides(tx.shape, tx.ndim, x_strides);

    /* Tail size: product of dims after the indexed range */
    int32_t tail_size = 1;
    for (int d = n_idx_dims; d < tx.ndim; d++)
        tail_size *= tx.shape[d];

    /* Allocate output (copy of x) */
    int out_sz;
    void *out = nd_result_alloc(tx.ndim, tx.shape, tx.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));
    memcpy(dst, tx.data, (size_t)tx.total * sizeof(float));

    /* For each position, compute flat offset and copy/accumulate values */
    for (int32_t pos = 0; pos < n_positions; pos++) {
        int32_t flat_offset = 0;
        for (int d = 0; d < n_idx_dims; d++) {
            int32_t dim_idx;
            if (has_idx[d]) {
                dim_idx = (int32_t)idx_t[d].data[pos];
                if (dim_idx < 0)
                    dim_idx += tx.shape[d];
                if (dim_idx < 0)
                    dim_idx = 0;
                if (dim_idx >= tx.shape[d])
                    dim_idx = tx.shape[d] - 1;
            } else {
                dim_idx = 0;
            }
            flat_offset += dim_idx * x_strides[d];
        }

        /* Copy or accumulate tail_size elements */
        int32_t val_offset = pos * tail_size;
        if (val_offset + tail_size > tv.total)
            break;
        if (flat_offset + tail_size > tx.total)
            break;
        if (accumulate) {
            for (int32_t k = 0; k < tail_size; k++)
                dst[flat_offset + k] += tv.data[val_offset + k];
        } else {
            memcpy(dst + flat_offset,
                   tv.data + val_offset,
                   (size_t)tail_size * sizeof(float));
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_repeat_interleave_tensor_nd(x, repeats_blob, dim) ─────────────── */
/* Like repeat_interleave but with a tensor of per-element repeat counts.    */
static void sql_repeat_interleave_tensor_nd(sqlite3_context *ctx,
                                            int argc,
                                            sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 t, tr;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &t) < 0)
        ERR(ctx, "llm_repeat_interleave_tensor_nd: invalid x blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tr) < 0)
        ERR(ctx, "llm_repeat_interleave_tensor_nd: invalid repeats blob");
    int dim = sqlite3_value_int(argv[2]);
    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_repeat_interleave_tensor_nd: dim out of range");

    int32_t dimsize = t.shape[dim];
    if (tr.total != dimsize)
        ERR(ctx,
            "llm_repeat_interleave_tensor_nd: repeats length != shape[dim]");

    /* Compute output dim size = sum of repeats */
    int32_t out_dimsize = 0;
    for (int32_t i = 0; i < dimsize; i++)
        out_dimsize += (int32_t)tr.data[i];

    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < t.ndim; i++)
        out_shape[i] = t.shape[i];
    out_shape[dim] = out_dimsize;
    int32_t out_total = 1;
    for (int i = 0; i < t.ndim; i++)
        out_total *= out_shape[i];

    int out_sz;
    void *out = nd_result_alloc(t.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));

    int32_t outer = 1, inner = 1;
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    for (int32_t o = 0; o < outer; o++) {
        int32_t dst_d = 0;
        for (int32_t d = 0; d < dimsize; d++) {
            int32_t r = (int32_t)tr.data[d];
            const float *src = t.data + (o * dimsize + d) * inner;
            for (int32_t rep = 0; rep < r; rep++) {
                float *dd = dst + (o * out_dimsize + dst_d) * inner;
                memcpy(dd, src, (size_t)inner * sizeof(float));
                dst_d++;
            }
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_squeeze_all_nd(blob) ──────────────────────────────────────────── */
/* Squeeze all dimensions of size 1.                                         */
static void
sql_squeeze_all_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_squeeze_all_nd: invalid blob");

    int32_t new_shape[LLM_MAX_NDIM];
    int new_ndim = 0;
    for (int i = 0; i < t.ndim; i++) {
        if (t.shape[i] != 1)
            new_shape[new_ndim++] = t.shape[i];
    }
    if (new_ndim == 0) {
        new_shape[0] = 1;
        new_ndim = 1;
    }

    int out_sz;
    void *out = nd_result_alloc(new_ndim, new_shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    memcpy((char *)out + LLM_ND_HDR(new_ndim),
           t.data,
           (size_t)t.total * sizeof(float));
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_flatten_nd(blob, start_dim, end_dim) ──────────────────────────── */
/* Flatten dimensions from start_dim to end_dim (inclusive) into one.         */
static void
sql_flatten_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_flatten_nd: invalid blob");
    int start_dim = sqlite3_value_int(argv[1]);
    int end_dim = sqlite3_value_int(argv[2]);
    if (start_dim < 0)
        start_dim += t.ndim;
    if (end_dim < 0)
        end_dim += t.ndim;
    if (start_dim < 0)
        start_dim = 0;
    if (end_dim >= t.ndim)
        end_dim = t.ndim - 1;
    if (start_dim > end_dim)
        ERR(ctx, "llm_flatten_nd: start_dim > end_dim");

    int32_t new_shape[LLM_MAX_NDIM];
    int new_ndim = 0;
    for (int i = 0; i < start_dim; i++)
        new_shape[new_ndim++] = t.shape[i];
    int32_t flat = 1;
    for (int i = start_dim; i <= end_dim; i++)
        flat *= t.shape[i];
    new_shape[new_ndim++] = flat;
    for (int i = end_dim + 1; i < t.ndim; i++)
        new_shape[new_ndim++] = t.shape[i];

    int out_sz;
    void *out = nd_result_alloc(new_ndim, new_shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    memcpy((char *)out + LLM_ND_HDR(new_ndim),
           t.data,
           (size_t)t.total * sizeof(float));
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_cat_multi_nd(dim, t0, t1, ...) ────────────────────────────────── */
/* Variadic: first arg is dim (int), remaining are tensor blobs.             */
/* Concatenates all tensors along the specified dimension in one call.        */
static void
sql_cat_multi_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2)
        ERR(ctx, "llm_cat_multi_nd: need dim + at least 1 tensor");
    int dim = sqlite3_value_int(argv[0]);
    int n_tensors = argc - 1;

    /* Parse all input tensors */
    TensorNDF32 ts[LLM_MAX_CAT_TENSORS];
    if (n_tensors > LLM_MAX_CAT_TENSORS)
        ERR(ctx, "llm_cat_multi_nd: too many tensors");
    for (int i = 0; i < n_tensors; i++) {
        if (llm_to_nd(sqlite3_value_blob(argv[1 + i]),
                      sqlite3_value_bytes(argv[1 + i]),
                      &ts[i]) < 0)
            ERR(ctx, "llm_cat_multi_nd: invalid tensor blob");
    }

    int ndim = ts[0].ndim;
    if (dim < 0)
        dim += ndim;
    if (dim < 0 || dim >= ndim)
        ERR(ctx, "llm_cat_multi_nd: dim out of range");

    /* Compute output shape */
    int32_t out_shape[LLM_MAX_NDIM];
    for (int d = 0; d < ndim; d++)
        out_shape[d] = ts[0].shape[d];
    int32_t cat_dim_total = ts[0].shape[dim];
    for (int i = 1; i < n_tensors; i++) {
        cat_dim_total += ts[i].shape[dim];
    }
    out_shape[dim] = cat_dim_total;

    int32_t out_total = 1;
    for (int d = 0; d < ndim; d++)
        out_total *= out_shape[d];

    int out_sz;
    void *out = nd_result_alloc(ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(ndim));

    /* Compute outer/inner relative to cat dim */
    int32_t outer = 1, inner = 1;
    for (int d = 0; d < dim; d++)
        outer *= out_shape[d];
    for (int d = dim + 1; d < ndim; d++)
        inner *= out_shape[d];

    /* Copy data: for each outer block, copy each tensor's chunk in order */
    for (int32_t o = 0; o < outer; o++) {
        int32_t dst_d = 0;
        for (int i = 0; i < n_tensors; i++) {
            int32_t tds = ts[i].shape[dim];
            const float *src = ts[i].data + o * tds * inner;
            float *dd = dst + (o * cat_dim_total + dst_d) * inner;
            memcpy(dd, src, (size_t)(tds * inner) * sizeof(float));
            dst_d += tds;
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_stack_nd(dim, t0, t1, ...) ────────────────────────────────────── */
/* Variadic: first arg is dim (int), remaining are tensor blobs.             */
/* Stacks all tensors along a new dimension.                                 */
static void sql_stack_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc < 2)
        ERR(ctx, "llm_stack_nd: need dim + at least 1 tensor");
    int dim = sqlite3_value_int(argv[0]);
    int n_tensors = argc - 1;

    TensorNDF32 ts[LLM_MAX_CAT_TENSORS];
    if (n_tensors > LLM_MAX_CAT_TENSORS)
        ERR(ctx, "llm_stack_nd: too many tensors");
    for (int i = 0; i < n_tensors; i++) {
        if (llm_to_nd(sqlite3_value_blob(argv[1 + i]),
                      sqlite3_value_bytes(argv[1 + i]),
                      &ts[i]) < 0)
            ERR(ctx, "llm_stack_nd: invalid tensor blob");
    }

    int src_ndim = ts[0].ndim;
    int out_ndim = src_ndim + 1;
    if (dim < 0)
        dim += out_ndim;
    if (dim < 0 || dim > src_ndim)
        ERR(ctx, "llm_stack_nd: dim out of range");
    if (out_ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_stack_nd: too many dims");

    /* Output shape: insert n_tensors at dim */
    int32_t out_shape[LLM_MAX_NDIM];
    int di = 0;
    for (int d = 0; d < out_ndim; d++) {
        if (d == dim) {
            out_shape[d] = n_tensors;
        } else {
            out_shape[d] = ts[0].shape[di++];
        }
    }
    int32_t out_total = 1;
    for (int d = 0; d < out_ndim; d++)
        out_total *= out_shape[d];

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    /* outer = product of dims before 'dim' (in source), inner = after */
    int32_t outer = 1, inner = 1;
    for (int d = 0; d < dim; d++)
        outer *= ts[0].shape[d];
    for (int d = dim; d < src_ndim; d++)
        inner *= ts[0].shape[d];

    /* Copy: for each outer block, for each tensor, copy inner block */
    for (int32_t o = 0; o < outer; o++) {
        for (int i = 0; i < n_tensors; i++) {
            const float *src = ts[i].data + o * inner;
            float *dd = dst + (o * n_tensors + i) * inner;
            memcpy(dd, src, (size_t)inner * sizeof(float));
        }
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_norm_nd(blob, p_ord, dim, keepdim) ─────────────────────────────── */
/* L-p norm along a single dimension. dim=-1 means global norm.              */
static void sql_norm_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_norm_nd: invalid blob");
    float p = (float)sqlite3_value_double(argv[1]);
    int dim = sqlite3_value_int(argv[2]);
    int keepdim = sqlite3_value_int(argv[3]);

    /* Global norm if dim is out of range sentinel */
    if (dim == -999) {
        double acc = 0.0;
        for (int32_t i = 0; i < t.total; i++)
            acc += pow(fabs((double)t.data[i]), (double)p);
        float result = (float)pow(acc, 1.0 / (double)p);
        int32_t shape1 = 1;
        int out_sz2;
        void *out = nd_result_alloc(1, &shape1, 1, &out_sz2);
        if (!out) {
            sqlite3_result_error_nomem(ctx);
            return;
        }
        float *dst = (float *)((char *)out + LLM_ND_HDR(1));
        dst[0] = result;
        sqlite3_result_blob(ctx, out, out_sz2, sqlite3_free);
        return;
    }

    if (dim < 0)
        dim += t.ndim;
    if (dim < 0 || dim >= t.ndim)
        ERR(ctx, "llm_norm_nd: dim out of range");

    int32_t outer = 1, inner = 1, dimsize = t.shape[dim];
    for (int i = 0; i < dim; i++)
        outer *= t.shape[i];
    for (int i = dim + 1; i < t.ndim; i++)
        inner *= t.shape[i];

    int32_t out_shape[LLM_MAX_NDIM];
    int out_ndim = 0;
    for (int i = 0; i < t.ndim; i++) {
        if (i == dim) {
            if (keepdim)
                out_shape[out_ndim++] = 1;
        } else {
            out_shape[out_ndim++] = t.shape[i];
        }
    }
    if (out_ndim == 0) {
        out_shape[0] = 1;
        out_ndim = 1;
    }
    int32_t out_total = outer * inner;

    int out_sz2;
    void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz2);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    for (int32_t o = 0; o < outer; o++) {
        for (int32_t n = 0; n < inner; n++) {
            double acc = 0.0;
            for (int32_t d = 0; d < dimsize; d++) {
                float v = t.data[(o * dimsize + d) * inner + n];
                acc += pow(fabs((double)v), (double)p);
            }
            dst[o * inner + n] = (float)pow(acc, 1.0 / (double)p);
        }
    }
    sqlite3_result_blob(ctx, out, out_sz2, sqlite3_free);
}

/* ── llm_bool_to_additive_mask_nd(blob, scale) ────────────────────────────
 * If the first data element > 0.5 (boolean mask: 1.0=attend, 0.0=block),
 * convert to additive float mask: out[i] = (x[i] - 1.0) * scale.
 * This maps True(1.0) → 0.0 and False(0.0) → -scale.
 * If the first element <= 0.5 (already an additive mask), return as-is.
 * ──────────────────────────────────────────────────────────────────────── */
static void sql_bool_to_additive_mask_nd(sqlite3_context *ctx,
                                         int argc,
                                         sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_bool_to_additive_mask_nd: invalid blob");
    float scale = (float)sqlite3_value_double(argv[1]);

    /* Detect boolean mask: first element > 0.5 means boolean format */
    if (t.total == 0 || t.data[0] <= 0.5f) {
        /* Already additive mask (or empty) — return as-is */
        sqlite3_result_blob(ctx, blob, sz, SQLITE_TRANSIENT);
        return;
    }

    /* Convert boolean → additive: (x - 1) * scale */
    int out_sz;
    void *out = nd_result_alloc(t.ndim, t.shape, t.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(t.ndim));
    for (int32_t i = 0; i < t.total; i++)
        dst[i] = (t.data[i] - 1.0f) * scale;
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_sdpa_nd(Q, K, V, scale, is_causal [, attn_mask]) ─────────────────
 * Fused Scaled Dot-Product Attention:
 *   output = softmax(Q @ K^T * scale [+ mask]) @ V
 *
 * Q: (..., seq_q, head_dim)
 * K: (..., seq_k, head_dim)
 * V: (..., seq_k, head_dim_v)
 * scale: float scalar
 * is_causal: int (0 or 1)
 * attn_mask: optional boolean mask (..., seq_q, seq_k) where 1=attend, 0=block
 *
 * Performs the entire attention computation in a single call, avoiding
 * intermediate blob allocation and multiple SQL round-trips.
 * ──────────────────────────────────────────────────────────────────────── */
static void sql_sdpa_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    if (argc != 5 && argc != 6)
        ERR(ctx, "llm_sdpa_nd: need 5 or 6 arguments");
    TensorNDF32 tq, tk, tv;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tq) < 0)
        ERR(ctx, "llm_sdpa_nd: invalid Q blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tk) < 0)
        ERR(ctx, "llm_sdpa_nd: invalid K blob");
    if (llm_to_nd(
            sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &tv) < 0)
        ERR(ctx, "llm_sdpa_nd: invalid V blob");
    float scale = (float)sqlite3_value_double(argv[3]);
    int is_causal = sqlite3_value_int(argv[4]);

    if (tq.ndim < 2 || tk.ndim < 2 || tv.ndim < 2)
        ERR(ctx, "llm_sdpa_nd: need at least 2-D tensors");
    if (tq.ndim != tk.ndim || tq.ndim != tv.ndim)
        ERR(ctx, "llm_sdpa_nd: Q, K, V must have same ndim");

    int nd = tq.ndim;
    int seq_q = tq.shape[nd - 2];
    int d_k = tq.shape[nd - 1];
    int seq_k = tk.shape[nd - 2];
    int d_kk = tk.shape[nd - 1];
    int d_v = tv.shape[nd - 1];
    int seq_kv = tv.shape[nd - 2];

    if (d_k != d_kk)
        ERR(ctx, "llm_sdpa_nd: Q head_dim != K head_dim");
    if (seq_k != seq_kv)
        ERR(ctx, "llm_sdpa_nd: K seq_len != V seq_len");

    /* Compute batch size (product of leading dims) */
    int32_t batch = 1;
    for (int i = 0; i < nd - 2; i++) {
        if (tq.shape[i] != tk.shape[i] || tq.shape[i] != tv.shape[i])
            ERR(ctx, "llm_sdpa_nd: batch dims mismatch");
        batch *= tq.shape[i];
    }

    /* Allocate scores buffer: (batch, seq_q, seq_k) */
    int32_t scores_n = batch * seq_q * seq_k;
    float *scores = (float *)sqlite3_malloc64((size_t)scores_n * sizeof(float));
    if (!scores) {
        sqlite3_result_error_nomem(ctx);
        return;
    }

    /* Step 1: scores = Q @ K^T  (batched GEMM) */
    for (int b = 0; b < batch; b++) {
        const float *q_ptr = tq.data + (size_t)b * seq_q * d_k;
        const float *k_ptr = tk.data + (size_t)b * seq_k * d_k;
        float *s_ptr = scores + (size_t)b * seq_q * seq_k;
        /* C = Q * K^T:  M=seq_q, N=seq_k, K=d_k */
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    seq_q,
                    seq_k,
                    d_k,
                    scale,
                    q_ptr,
                    d_k,
                    k_ptr,
                    d_k,
                    0.0f,
                    s_ptr,
                    seq_k);
    }

    /* Step 2: Apply causal mask (upper triangular → -inf) */
    if (is_causal) {
        int offset = seq_k - seq_q; /* position offset for KV cache */
        for (int b = 0; b < batch; b++) {
            float *s_ptr = scores + (size_t)b * seq_q * seq_k;
            for (int i = 0; i < seq_q; i++) {
                for (int j = i + 1 + offset; j < seq_k; j++) {
                    s_ptr[i * seq_k + j] = -1e9f;
                }
            }
        }
    }

    /* Step 2b: Apply explicit boolean attention mask if provided */
    if (argc == 6 && sqlite3_value_type(argv[5]) != SQLITE_NULL) {
        TensorNDF32 tm;
        if (llm_to_nd(sqlite3_value_blob(argv[5]),
                      sqlite3_value_bytes(argv[5]),
                      &tm) < 0) {
            sqlite3_free(scores);
            ERR(ctx, "llm_sdpa_nd: invalid mask blob");
        }
        /* Broadcast mask over batch dimension.
         * Mask can be (..., seq_q, seq_k) with broadcasting on leading dims.
         * For boolean mask: 0.0 → -1e9 (block), 1.0 → 0.0 (attend) */
        int32_t mask_seq_q = tm.shape[tm.ndim - 2];
        int32_t mask_seq_k = tm.shape[tm.ndim - 1];
        int32_t mask_batch = tm.total / (mask_seq_q * mask_seq_k);
        for (int b = 0; b < batch; b++) {
            float *s_ptr = scores + (size_t)b * seq_q * seq_k;
            /* Broadcast: cycle mask batch if mask_batch < batch */
            int mb = b % mask_batch;
            const float *m_ptr = tm.data + (size_t)mb * mask_seq_q * mask_seq_k;
            for (int i = 0; i < seq_q; i++) {
                int mi = i < mask_seq_q ? i : mask_seq_q - 1;
                for (int j = 0; j < seq_k; j++) {
                    int mj = j < mask_seq_k ? j : mask_seq_k - 1;
                    float mv = m_ptr[mi * mask_seq_k + mj];
                    if (mv < 0.5f) { /* Boolean 0 → block */
                        s_ptr[i * seq_k + j] = -1e9f;
                    }
                }
            }
        }
    }

    /* Step 3: Softmax along last dim (seq_k) for each (batch, seq_q) row */
    for (int b = 0; b < batch; b++) {
        float *s_ptr = scores + (size_t)b * seq_q * seq_k;
        for (int i = 0; i < seq_q; i++) {
            float *row = s_ptr + i * seq_k;
            /* Numerically stable softmax */
            float mx = -FLT_MAX;
            for (int j = 0; j < seq_k; j++)
                if (row[j] > mx)
                    mx = row[j];
            float s = 0.0f;
            for (int j = 0; j < seq_k; j++) {
                row[j] = expf(row[j] - mx);
                s += row[j];
            }
            for (int j = 0; j < seq_k; j++)
                row[j] /= s;
        }
    }

    /* Step 4: output = scores @ V  (batched GEMM) */
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < nd - 2; i++)
        out_shape[i] = tq.shape[i];
    out_shape[nd - 2] = seq_q;
    out_shape[nd - 1] = d_v;
    int32_t out_total = batch * seq_q * d_v;

    int out_sz;
    void *out = nd_result_alloc(nd, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_free(scores);
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(nd));

    for (int b = 0; b < batch; b++) {
        const float *s_ptr = scores + (size_t)b * seq_q * seq_k;
        const float *v_ptr = tv.data + (size_t)b * seq_k * d_v;
        float *o_ptr = dst + (size_t)b * seq_q * d_v;
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    seq_q,
                    d_v,
                    seq_k,
                    1.0f,
                    s_ptr,
                    seq_k,
                    v_ptr,
                    d_v,
                    0.0f,
                    o_ptr,
                    d_v);
    }

    sqlite3_free(scores);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_numel_nd(blob) ──────────────────────────────────────────────────
 * Return the total number of elements in the tensor (product of shape).
 * ──────────────────────────────────────────────────────────────────────── */
static void sql_numel_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    TensorNDF32 t;
    if (llm_to_nd(blob, sz, &t) < 0)
        ERR(ctx, "llm_numel_nd: invalid blob");
    sqlite3_result_int(ctx, t.total);
}

/* ===========================================================================
 * Section 14: INT8 quantization operations
 *
 * Per-row symmetric quantization:
 *   scale[r]  = max(|row[r]|) / 127.0
 *   q[r][c]   = clamp(round(data[r][c] * (127.0 / max(|row[r]|))), -127, 127)
 *   deq[r][c] = q[r][c] * scale[r]
 *
 * Threading: parallel over rows using pthreads (LLM_NUM_THREADS).
 * SIMD: GCC -O3 auto-vectorises the inner loops; no explicit intrinsics
 *       needed for int8 since the bottleneck is memory bandwidth.
 * =========================================================================*/

/* ── Thread pool for int8 parallel ops ───────────────────────────────────── *
 *
 * Uses pthread_create/join per call.  For hidden_size < 1024 the
 * threshold LLM_INT8_THREAD_OPS_MIN prevents threading entirely, so the
 * ~100 µs create/join overhead is only paid for large models where the
 * compute time (>>1 ms) dwarfs it.
 * ─────────────────────────────────────────────────────────────────────── */

typedef struct {
    void (*func)(void *arg, int32_t row_start, int32_t row_end);
    void *arg;
    int32_t total_rows;
    int32_t chunk; /* rows per thread */
    int tid;
} ThreadArg;

static void *thread_worker(void *raw) {
    ThreadArg *ta = (ThreadArg *)raw;
    int32_t start = ta->tid * ta->chunk;
    int32_t end = start + ta->chunk;
    if (end > ta->total_rows)
        end = ta->total_rows;
    if (start < end)
        ta->func(ta->arg, start, end);
    return NULL;
}

static void parallel_rows(void (*func)(void *, int32_t, int32_t),
                          void *arg,
                          int32_t nrows) {
    int nthreads = g_llm_num_threads;
    if (nrows < nthreads)
        nthreads = nrows;
    if (nthreads <= 1) {
        func(arg, 0, nrows);
        return;
    }
    int32_t chunk = (nrows + nthreads - 1) / nthreads;
    pthread_t threads[LLM_NUM_THREADS];
    ThreadArg args[LLM_NUM_THREADS];
    /* Main thread does chunk 0; only create (nthreads-1) workers */
    for (int i = 1; i < nthreads; i++) {
        args[i].func = func;
        args[i].arg = arg;
        args[i].total_rows = nrows;
        args[i].chunk = chunk;
        args[i].tid = i;
        pthread_create(&threads[i], NULL, thread_worker, &args[i]);
    }
    /* Main thread work */
    int32_t end0 = chunk < nrows ? chunk : nrows;
    func(arg, 0, end0);
    /* Join workers */
    for (int i = 1; i < nthreads; i++)
        pthread_join(threads[i], NULL);
}

/* ── Helper: int8 result blob allocation ─────────────────────────────────── */
static void *int8_result_alloc(
    int ndim, const int32_t *shape, int32_t total, int32_t nrows, int *sz_out) {
    int hdr = LLM_INT8_HDR(ndim);
    int scales_bytes = nrows * 4;
    int data_bytes = total;
    int sz = hdr + scales_bytes + data_bytes;
    char *b = (char *)sqlite3_malloc(sz);
    if (!b)
        return NULL;
    *(int32_t *)b = LLM_INT8_MAGIC;
    *((int32_t *)b + 1) = ndim;
    for (int i = 0; i < ndim; i++)
        ((int32_t *)b)[2 + i] = shape[i];
    *sz_out = sz;
    return b;
}

/* ── llm_quantize_int8_nd(fp32_tensor) → int8 quantized tensor ──────────── */

typedef struct {
    const float *src;
    float *scales;
    int8_t *dst;
    int32_t row_len;
} QuantizeArg;

static void quantize_rows(void *raw, int32_t r0, int32_t r1) {
    QuantizeArg *a = (QuantizeArg *)raw;
    int32_t K = a->row_len;
    for (int32_t r = r0; r < r1; r++) {
        const float *row = a->src + (size_t)r * K;
        int8_t *qrow = a->dst + (size_t)r * K;
        /* Find absmax */
        float amax = 0.0f;
        int32_t j = 0;
#if LLM_HAVE_AVX2
        {
            __m256 vmax = _mm256_setzero_ps();
            __m256 sign_mask =
                _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            for (; j + 8 <= K; j += 8) {
                __m256 v = _mm256_loadu_ps(row + j);
                vmax = _mm256_max_ps(vmax, _mm256_and_ps(v, sign_mask));
            }
            /* horizontal max of 8 floats */
            __m128 lo = _mm256_castps256_ps128(vmax);
            __m128 hi = _mm256_extractf128_ps(vmax, 1);
            lo = _mm_max_ps(lo, hi);
            lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0x4E));
            lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0xB1));
            amax = _mm_cvtss_f32(lo);
        }
#endif
        for (; j < K; j++) {
            float v = row[j] < 0 ? -row[j] : row[j];
            if (v > amax)
                amax = v;
        }
        float scale = amax / 127.0f;
        a->scales[r] = scale;
        if (scale < 1e-10f) {
            /* Near-zero row → all zeros */
            memset(qrow, 0, (size_t)K);
        } else {
            float inv_scale = 127.0f / amax;
            j = 0;
#if LLM_HAVE_AVX2
            {
                __m256 vinv = _mm256_set1_ps(inv_scale);
                __m256 vmin = _mm256_set1_ps(-127.0f);
                __m256 vmax_q = _mm256_set1_ps(127.0f);
                for (; j + 8 <= K; j += 8) {
                    __m256 v = _mm256_mul_ps(_mm256_loadu_ps(row + j), vinv);
                    v = _mm256_round_ps(
                        v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    v = _mm256_max_ps(v, vmin);
                    v = _mm256_min_ps(v, vmax_q);
                    __m256i vi = _mm256_cvtps_epi32(v);
                    /* Pack 8 int32 → 8 int8 via SIMD:
                     * packs_epi32: int32×8 → int16×8 (saturating)
                     * packs_epi16: int16×8 → int8×8  (saturating)
                     * AVX2 packs works per 128-bit lane, so we
                     * extract, pack in 128-bit, and store 8 bytes. */
                    __m128i lo = _mm256_castsi256_si128(vi);
                    __m128i hi = _mm256_extracti128_si256(vi, 1);
                    __m128i i16 = _mm_packs_epi32(lo, hi);
                    __m128i i8 = _mm_packs_epi16(i16, i16);
                    _mm_storel_epi64((__m128i *)(qrow + j), i8);
                }
            }
#endif
            for (; j < K; j++) {
                float v = row[j] * inv_scale;
                int q = (int)roundf(v);
                if (q < -127)
                    q = -127;
                if (q > 127)
                    q = 127;
                qrow[j] = (int8_t)q;
            }
        }
    }
}

static void
sql_quantize_int8_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_quantize_int8_nd: invalid tensor");

    int32_t K = tx.shape[tx.ndim - 1];
    int32_t nrows = (K > 0) ? tx.total / K : 0;

    int out_sz;
    void *out = int8_result_alloc(tx.ndim, tx.shape, tx.total, nrows, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    int hdr = LLM_INT8_HDR(tx.ndim);
    float *scales = (float *)((char *)out + hdr);
    int8_t *qdata = (int8_t *)((char *)out + hdr + nrows * 4);

    QuantizeArg qa = {tx.data, scales, qdata, K};
    parallel_rows(quantize_rows, &qa, nrows);

    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_dequantize_int8_nd(int8_tensor) → fp32 tensor ──────────────────── */

typedef struct {
    const float *scales;
    const int8_t *src;
    float *dst;
    int32_t row_len;
} DequantizeArg;

static void dequantize_rows(void *raw, int32_t r0, int32_t r1) {
    DequantizeArg *a = (DequantizeArg *)raw;
    int32_t K = a->row_len;
    for (int32_t r = r0; r < r1; r++) {
        const int8_t *qrow = a->src + (size_t)r * K;
        float *out_row = a->dst + (size_t)r * K;
        float scale = a->scales[r];
        int32_t j = 0;
#if LLM_HAVE_AVX2
        __m256 vscale = _mm256_set1_ps(scale);
        for (; j + 8 <= K; j += 8) {
            __m128i vi8 = _mm_loadl_epi64((const __m128i *)(qrow + j));
            __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vi8));
            _mm256_storeu_ps(out_row + j, _mm256_mul_ps(vf, vscale));
        }
#endif
        for (; j < K; j++)
            out_row[j] = (float)qrow[j] * scale;
    }
}

static void
sql_dequantize_int8_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDInt8 tq;
    if (llm_parse_int8(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tq) < 0)
        ERR(ctx, "llm_dequantize_int8_nd: invalid int8 tensor");

    int out_sz;
    void *out = nd_result_alloc(tq.ndim, tq.shape, tq.total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tq.ndim));

    DequantizeArg da = {tq.scales, tq.data, dst, tq.row_len};
    parallel_rows(dequantize_rows, &da, tq.nrows);

    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── AVX2 SIMD dot product: float[K] · int8[K] → float ─────────────────── */
/* Note: unaligned loads (_mm256_loadu_ps) are used because SQLite blob
 * pointers are NOT guaranteed to be 32-byte aligned.  On modern x86
 * (Haswell+) the penalty for unaligned loads is negligible.                */
#if LLM_HAVE_AVX2
LLM_INLINE float dot_i8f32_avx2(const float *x, const int8_t *w, int32_t K) {
    __m256 vacc0 = _mm256_setzero_ps();
    __m256 vacc1 = _mm256_setzero_ps();
    __m256 vacc2 = _mm256_setzero_ps();
    __m256 vacc3 = _mm256_setzero_ps();
    int32_t k = 0;
    /* Unrolled 4×8 = 32 elements per iteration */
    for (; k + 32 <= K; k += 32) {
        __m128i vi8a = _mm_loadl_epi64((const __m128i *)(w + k));
        __m128i vi8b = _mm_loadl_epi64((const __m128i *)(w + k + 8));
        __m128i vi8c = _mm_loadl_epi64((const __m128i *)(w + k + 16));
        __m128i vi8d = _mm_loadl_epi64((const __m128i *)(w + k + 24));
        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k),
                                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vi8a)),
                                vacc0);
        vacc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 8),
                                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vi8b)),
                                vacc1);
        vacc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 16),
                                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vi8c)),
                                vacc2);
        vacc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 24),
                                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vi8d)),
                                vacc3);
    }
    vacc0 = _mm256_add_ps(vacc0, vacc1);
    vacc2 = _mm256_add_ps(vacc2, vacc3);
    vacc0 = _mm256_add_ps(vacc0, vacc2);
    /* Process remaining 8 elements at a time */
    for (; k + 8 <= K; k += 8) {
        __m128i vi8 = _mm_loadl_epi64((const __m128i *)(w + k));
        vacc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k),
                                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vi8)),
                                vacc0);
    }
    float acc = llm_hsum256(vacc0);
    for (; k < K; k++)
        acc += x[k] * (float)w[k];
    return acc;
}
#endif /* LLM_HAVE_AVX2 */

LLM_INLINE float dot_i8f32(const float *x, const int8_t *w, int32_t K) {
#if LLM_HAVE_AVX2
    return dot_i8f32_avx2(x, w, K);
#else
    float acc = 0.0f;
    for (int32_t k = 0; k < K; k++)
        acc += x[k] * (float)w[k];
    return acc;
#endif
}

/* ── llm_linear_int8_nd(x_fp32, w_int8) → fp32 output ───────────────────── *
 *
 * Computes y = x @ W^T where x is fp32 [*, K] and W is int8 [N, K].
 * Output shape: [*, N] in fp32.
 *
 * Per output row m:
 *   y[m][n] = scale[n] * Σ_k( x[m][k] * (float)w_int8[n][k] )
 *
 * Threading: M=1 (decode) parallelises over N; M>1 over M.
 * SIMD: AVX2 FMA for the int8×fp32 dot product.                            */

typedef struct {
    const float *x;      /* [M, K] fp32 input */
    const int8_t *wq;    /* [N, K] int8 weights */
    const float *scales; /* [N] per-row scales */
    float *dst;          /* [M, N] fp32 output */
    int32_t M, N, K;
} LinearInt8Arg;

/* Worker: parallel over M (prefill path, M > 1) */
static void linear_int8_rows_m(void *raw, int32_t m0, int32_t m1) {
    LinearInt8Arg *a = (LinearInt8Arg *)raw;
    int32_t N = a->N, K = a->K;
    for (int32_t m = m0; m < m1; m++) {
        const float *xrow = a->x + (size_t)m * K;
        float *orow = a->dst + (size_t)m * N;
        for (int32_t n = 0; n < N; n++) {
            const int8_t *wrow = a->wq + (size_t)n * K;
            orow[n] = dot_i8f32(xrow, wrow, K) * a->scales[n];
        }
    }
}

/* Worker: parallel over N (decode fast path, M = 1) */
static void linear_int8_rows_n(void *raw, int32_t n0, int32_t n1) {
    LinearInt8Arg *a = (LinearInt8Arg *)raw;
    const float *xrow = a->x;
    int32_t K = a->K;
    int32_t n = n0;
#if LLM_HAVE_AVX2
    /* Process 4 output features simultaneously to maximize ILP —
     * the four FMA chains are independent, so the CPU can issue
     * them in parallel on separate execution ports. */
    for (; n + 4 <= n1; n += 4) {
        const int8_t *w0 = a->wq + (size_t)(n)*K;
        const int8_t *w1 = a->wq + (size_t)(n + 1) * K;
        const int8_t *w2 = a->wq + (size_t)(n + 2) * K;
        const int8_t *w3 = a->wq + (size_t)(n + 3) * K;
        __m256 va0 = _mm256_setzero_ps();
        __m256 va1 = _mm256_setzero_ps();
        __m256 va2 = _mm256_setzero_ps();
        __m256 va3 = _mm256_setzero_ps();
        int32_t k = 0;
        for (; k + 8 <= K; k += 8) {
            __m256 vx = _mm256_loadu_ps(xrow + k);
            va0 = _mm256_fmadd_ps(
                vx,
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                    _mm_loadl_epi64((const __m128i *)(w0 + k)))),
                va0);
            va1 = _mm256_fmadd_ps(
                vx,
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                    _mm_loadl_epi64((const __m128i *)(w1 + k)))),
                va1);
            va2 = _mm256_fmadd_ps(
                vx,
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                    _mm_loadl_epi64((const __m128i *)(w2 + k)))),
                va2);
            va3 = _mm256_fmadd_ps(
                vx,
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                    _mm_loadl_epi64((const __m128i *)(w3 + k)))),
                va3);
        }
        float s0 = llm_hsum256(va0), s1 = llm_hsum256(va1);
        float s2 = llm_hsum256(va2), s3 = llm_hsum256(va3);
        for (; k < K; k++) {
            float xk = xrow[k];
            s0 += xk * (float)w0[k];
            s1 += xk * (float)w1[k];
            s2 += xk * (float)w2[k];
            s3 += xk * (float)w3[k];
        }
        a->dst[n] = s0 * a->scales[n];
        a->dst[n + 1] = s1 * a->scales[n + 1];
        a->dst[n + 2] = s2 * a->scales[n + 2];
        a->dst[n + 3] = s3 * a->scales[n + 3];
    }
#endif
    for (; n < n1; n++) {
        const int8_t *wrow = a->wq + (size_t)n * K;
        a->dst[n] = dot_i8f32(xrow, wrow, K) * a->scales[n];
    }
}

static void
sql_linear_int8_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx;
    TensorNDInt8 tw;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_linear_int8_nd: invalid x");
    if (llm_parse_int8(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tw) < 0)
        ERR(ctx, "llm_linear_int8_nd: invalid int8 weight");
    if (tx.ndim < 1)
        ERR(ctx, "llm_linear_int8_nd: x must be at least 1-D");
    if (tw.ndim != 2)
        ERR(ctx, "llm_linear_int8_nd: weight must be 2-D");

    int32_t K = tx.shape[tx.ndim - 1];
    if (tw.shape[1] != K)
        ERR(ctx, "llm_linear_int8_nd: in_features mismatch");

    int32_t M = tx.total / K;
    int32_t N = tw.shape[0];
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < tx.ndim; i++)
        out_shape[i] = tx.shape[i];
    out_shape[tx.ndim - 1] = N;
    int32_t out_total = M * N;

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    LinearInt8Arg la = {tx.data, tw.data, tw.scales, dst, M, N, K};
    if (M == 1) {
        /* Decode path: parallelise over N when work is large enough to
         * justify thread creation overhead.  See LLM_INT8_THREAD_OPS_MIN. */
        if ((int64_t)N * K >= LLM_INT8_THREAD_OPS_MIN)
            parallel_rows(linear_int8_rows_n, &la, N);
        else
            linear_int8_rows_n(&la, 0, N);
    } else {
        /* Prefill path: parallelise over M batch rows */
        parallel_rows(linear_int8_rows_m, &la, M);
    }

    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_linear_int8_bias_nd(x, w_int8, bias_fp32) → x @ W^T + bias ───── *
 *
 * Fused linear+add: avoids a separate allocation and SQL call for the
 * bias addition.  Same as llm_linear_int8_nd but adds fp32 bias[N] to
 * each output row in-place before returning.                               */

static void
sql_linear_int8_bias_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, tb;
    TensorNDInt8 tw;
    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_linear_int8_bias_nd: invalid x");
    if (llm_parse_int8(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &tw) < 0)
        ERR(ctx, "llm_linear_int8_bias_nd: invalid int8 weight");
    if (llm_to_nd(
            sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &tb) < 0)
        ERR(ctx, "llm_linear_int8_bias_nd: invalid bias");
    if (tx.ndim < 1)
        ERR(ctx, "llm_linear_int8_bias_nd: x must be at least 1-D");
    if (tw.ndim != 2)
        ERR(ctx, "llm_linear_int8_bias_nd: weight must be 2-D");

    int32_t K = tx.shape[tx.ndim - 1];
    if (tw.shape[1] != K)
        ERR(ctx, "llm_linear_int8_bias_nd: in_features mismatch");

    int32_t M = tx.total / K;
    int32_t N = tw.shape[0];
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < tx.ndim; i++)
        out_shape[i] = tx.shape[i];
    out_shape[tx.ndim - 1] = N;
    int32_t out_total = M * N;

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    LinearInt8Arg la = {tx.data, tw.data, tw.scales, dst, M, N, K};
    if (M == 1) {
        if ((int64_t)N * K >= LLM_INT8_THREAD_OPS_MIN)
            parallel_rows(linear_int8_rows_n, &la, N);
        else
            linear_int8_rows_n(&la, 0, N);
    } else {
        parallel_rows(linear_int8_rows_m, &la, M);
    }

    /* Add bias in-place */
    const float *bias = tb.data;
    for (int32_t m = 0; m < M; m++) {
        float *row = dst + (size_t)m * N;
        for (int32_t n = 0; n < N; n++)
            row[n] += bias[n];
    }

    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

typedef struct llm_param_lookup_t llm_param_lookup_t;
static int llm_param_lookup_step_blob(llm_param_lookup_t *lookup,
                                      sqlite3 *db,
                                      sqlite3_context *ctx,
                                      const char *name,
                                      const void **blob,
                                      int *size);
static void llm_param_lookup_release(llm_param_lookup_t *lookup);

/* ── llm_linear_param_nd(x, weight_name_text) ──────────────────────────── */
static void
sql_linear_param_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, tw;
    TensorNDInt8 twq;
    llm_param_lookup_t *lookup = (llm_param_lookup_t *)sqlite3_user_data(ctx);
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    const unsigned char *weight_name = sqlite3_value_text(argv[1]);
    const void *weight_blob = NULL;
    int weight_size = 0;

    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_linear_param_nd: invalid x");
    if (llm_param_lookup_step_blob(lookup,
                                   db,
                                   ctx,
                                   (const char *)weight_name,
                                   &weight_blob,
                                   &weight_size) != SQLITE_OK) {
        return;
    }

    if (llm_parse_int8(weight_blob, weight_size, &twq) == 0) {
        if (tx.ndim < 1) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_linear_param_nd: x must be at least 1-D");
        }
        if (twq.ndim != 2) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_linear_param_nd: weight must be 2-D");
        }
        int32_t K = tx.shape[tx.ndim - 1];
        if (twq.shape[1] != K) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_linear_param_nd: in_features mismatch");
        }

        int32_t M = tx.total / K;
        int32_t N = twq.shape[0];
        int32_t out_shape[LLM_MAX_NDIM];
        for (int i = 0; i < tx.ndim; i++)
            out_shape[i] = tx.shape[i];
        out_shape[tx.ndim - 1] = N;
        int32_t out_total = M * N;

        int out_sz;
        void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
        if (!out) {
            llm_param_lookup_release(lookup);
            sqlite3_result_error_nomem(ctx);
            return;
        }
        float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

        LinearInt8Arg la = {tx.data, twq.data, twq.scales, dst, M, N, K};
        if (M == 1) {
            if ((int64_t)N * K >= LLM_INT8_THREAD_OPS_MIN)
                parallel_rows(linear_int8_rows_n, &la, N);
            else
                linear_int8_rows_n(&la, 0, N);
        } else {
            parallel_rows(linear_int8_rows_m, &la, M);
        }

        llm_param_lookup_release(lookup);
        sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
        return;
    }

    if (llm_to_nd(weight_blob, weight_size, &tw) < 0) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_nd: invalid weight");
    }
    if (tx.ndim < 1) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_nd: x must be at least 1-D");
    }
    if (tw.ndim != 2) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_nd: weight must be 2-D");
    }

    int32_t k = tx.shape[tx.ndim - 1];
    if (tw.shape[1] != k) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_nd: in_features mismatch");
    }

    int32_t m = tx.total / k;
    int32_t n = tw.shape[0];
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < tx.ndim; i++)
        out_shape[i] = tx.shape[i];
    out_shape[tx.ndim - 1] = n;
    int32_t out_total = m * n;

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        llm_param_lookup_release(lookup);
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    if (m == 1) {
        cblas_sgemv(CblasRowMajor,
                    CblasNoTrans,
                    n,
                    k,
                    1.0f,
                    tw.data,
                    k,
                    tx.data,
                    1,
                    0.0f,
                    dst,
                    1);
    } else {
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    m,
                    n,
                    k,
                    1.0f,
                    tx.data,
                    k,
                    tw.data,
                    k,
                    0.0f,
                    dst,
                    n);
    }

    llm_param_lookup_release(lookup);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_linear_param_bias_nd(x, weight_name_text, bias_fp32) ──────────── */
static void
sql_linear_param_bias_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tx, tb, tw;
    TensorNDInt8 twq;
    llm_param_lookup_t *lookup = (llm_param_lookup_t *)sqlite3_user_data(ctx);
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    const unsigned char *weight_name = sqlite3_value_text(argv[1]);
    const void *weight_blob = NULL;
    int weight_size = 0;

    if (llm_to_nd(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tx) < 0)
        ERR(ctx, "llm_linear_param_bias_nd: invalid x");
    if (llm_to_nd(
            sqlite3_value_blob(argv[2]), sqlite3_value_bytes(argv[2]), &tb) < 0)
        ERR(ctx, "llm_linear_param_bias_nd: invalid bias");
    if (llm_param_lookup_step_blob(lookup,
                                   db,
                                   ctx,
                                   (const char *)weight_name,
                                   &weight_blob,
                                   &weight_size) != SQLITE_OK) {
        return;
    }

    if (llm_parse_int8(weight_blob, weight_size, &twq) == 0) {
        if (tx.ndim < 1) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_linear_param_bias_nd: x must be at least 1-D");
        }
        if (twq.ndim != 2) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_linear_param_bias_nd: weight must be 2-D");
        }
        int32_t K = tx.shape[tx.ndim - 1];
        if (twq.shape[1] != K) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_linear_param_bias_nd: in_features mismatch");
        }

        int32_t M = tx.total / K;
        int32_t N = twq.shape[0];
        int32_t out_shape[LLM_MAX_NDIM];
        for (int i = 0; i < tx.ndim; i++)
            out_shape[i] = tx.shape[i];
        out_shape[tx.ndim - 1] = N;
        int32_t out_total = M * N;

        int out_sz;
        void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
        if (!out) {
            llm_param_lookup_release(lookup);
            sqlite3_result_error_nomem(ctx);
            return;
        }
        float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

        LinearInt8Arg la = {tx.data, twq.data, twq.scales, dst, M, N, K};
        if (M == 1) {
            if ((int64_t)N * K >= LLM_INT8_THREAD_OPS_MIN)
                parallel_rows(linear_int8_rows_n, &la, N);
            else
                linear_int8_rows_n(&la, 0, N);
        } else {
            parallel_rows(linear_int8_rows_m, &la, M);
        }

        const float *bias = tb.data;
        for (int32_t m = 0; m < M; m++) {
            float *row = dst + (size_t)m * N;
            for (int32_t n = 0; n < N; n++)
                row[n] += bias[n];
        }

        llm_param_lookup_release(lookup);
        sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
        return;
    }

    if (llm_to_nd(weight_blob, weight_size, &tw) < 0) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_bias_nd: invalid weight");
    }
    if (tx.ndim < 1) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_bias_nd: x must be at least 1-D");
    }
    if (tw.ndim != 2) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_bias_nd: weight must be 2-D");
    }

    int32_t k = tx.shape[tx.ndim - 1];
    if (tw.shape[1] != k) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_linear_param_bias_nd: in_features mismatch");
    }

    int32_t m = tx.total / k;
    int32_t n = tw.shape[0];
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < tx.ndim; i++)
        out_shape[i] = tx.shape[i];
    out_shape[tx.ndim - 1] = n;
    int32_t out_total = m * n;

    int out_sz;
    void *out = nd_result_alloc(tx.ndim, out_shape, out_total, &out_sz);
    if (!out) {
        llm_param_lookup_release(lookup);
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(tx.ndim));

    if (m == 1) {
        cblas_sgemv(CblasRowMajor,
                    CblasNoTrans,
                    n,
                    k,
                    1.0f,
                    tw.data,
                    k,
                    tx.data,
                    1,
                    0.0f,
                    dst,
                    1);
    } else {
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    m,
                    n,
                    k,
                    1.0f,
                    tx.data,
                    k,
                    tw.data,
                    k,
                    0.0f,
                    dst,
                    n);
    }

    const float *bias = tb.data;
    for (int32_t row_idx = 0; row_idx < m; row_idx++) {
        float *row = dst + (size_t)row_idx * n;
        for (int32_t col_idx = 0; col_idx < n; col_idx++)
            row[col_idx] += bias[col_idx];
    }

    llm_param_lookup_release(lookup);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_embedding_int8_nd(w_int8, indices_fp32) → fp32 output ──────────── *
 *
 * Looks up rows from an int8 quantized embedding table, dequantising
 * each selected row to fp32 on the fly.
 *
 * w_int8:  [vocab, embed_dim]  (int8)
 * indices: N-D fp32 tensor (values cast to int for lookup)
 * output:  [...indices.shape, embed_dim]  (fp32)                            */

static void
sql_embedding_int8_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDInt8 tw;
    TensorNDF32 ti;
    if (llm_parse_int8(
            sqlite3_value_blob(argv[0]), sqlite3_value_bytes(argv[0]), &tw) < 0)
        ERR(ctx, "llm_embedding_int8_nd: invalid int8 weight");
    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &ti) < 0)
        ERR(ctx, "llm_embedding_int8_nd: invalid indices");
    if (tw.ndim != 2)
        ERR(ctx, "llm_embedding_int8_nd: weight must be 2-D");

    int32_t vocab = tw.shape[0];
    int32_t embed = tw.shape[1];

    int out_ndim = ti.ndim + 1;
    if (out_ndim > LLM_MAX_NDIM)
        ERR(ctx, "llm_embedding_int8_nd: too many dims");
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < ti.ndim; i++)
        out_shape[i] = ti.shape[i];
    out_shape[ti.ndim] = embed;
    int32_t out_total = ti.total * embed;

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
    if (!out) {
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    for (int32_t i = 0; i < ti.total; i++) {
        int idx = (int)roundf(ti.data[i]);
        if (idx < 0 || idx >= vocab) {
            sqlite3_free(out);
            ERR(ctx, "llm_embedding_int8_nd: index out of range");
        }
        float scale = tw.scales[idx];
        const int8_t *qrow = tw.data + (size_t)idx * embed;
        float *drow = dst + (size_t)i * embed;
        int32_t j = 0;
#if LLM_HAVE_AVX2
        __m256 vscale = _mm256_set1_ps(scale);
        for (; j + 8 <= embed; j += 8) {
            __m128i vi8 = _mm_loadl_epi64((const __m128i *)(qrow + j));
            __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vi8));
            _mm256_storeu_ps(drow + j, _mm256_mul_ps(vf, vscale));
        }
#endif
        for (; j < embed; j++)
            drow[j] = (float)qrow[j] * scale;
    }
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_embedding_param_nd(weight_name_text, indices) ─────────────────── */
static void
sql_embedding_param_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    TensorNDF32 tw, ti;
    TensorNDInt8 twq;
    llm_param_lookup_t *lookup = (llm_param_lookup_t *)sqlite3_user_data(ctx);
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    const unsigned char *weight_name = sqlite3_value_text(argv[0]);
    const void *weight_blob = NULL;
    int weight_size = 0;

    if (llm_to_nd(
            sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]), &ti) < 0)
        ERR(ctx, "llm_embedding_param_nd: invalid indices");
    if (llm_param_lookup_step_blob(lookup,
                                   db,
                                   ctx,
                                   (const char *)weight_name,
                                   &weight_blob,
                                   &weight_size) != SQLITE_OK) {
        return;
    }

    if (llm_parse_int8(weight_blob, weight_size, &twq) == 0) {
        if (twq.ndim != 2) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_embedding_param_nd: weight must be 2-D");
        }
        int32_t vocab = twq.shape[0];
        int32_t embed = twq.shape[1];
        int out_ndim = ti.ndim + 1;
        if (out_ndim > LLM_MAX_NDIM) {
            llm_param_lookup_release(lookup);
            ERR(ctx, "llm_embedding_param_nd: too many dims");
        }
        int32_t out_shape[LLM_MAX_NDIM];
        for (int i = 0; i < ti.ndim; i++)
            out_shape[i] = ti.shape[i];
        out_shape[ti.ndim] = embed;
        int32_t out_total = ti.total * embed;

        int out_sz;
        void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
        if (!out) {
            llm_param_lookup_release(lookup);
            sqlite3_result_error_nomem(ctx);
            return;
        }
        float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

        for (int32_t i = 0; i < ti.total; i++) {
            int idx = (int)roundf(ti.data[i]);
            if (idx < 0 || idx >= vocab) {
                llm_param_lookup_release(lookup);
                sqlite3_free(out);
                ERR(ctx, "llm_embedding_param_nd: index out of range");
            }
            float scale = twq.scales[idx];
            const int8_t *qrow = twq.data + (size_t)idx * embed;
            float *drow = dst + (size_t)i * embed;
            int32_t j = 0;
#if LLM_HAVE_AVX2
            __m256 vscale = _mm256_set1_ps(scale);
            for (; j + 8 <= embed; j += 8) {
                __m128i vi8 = _mm_loadl_epi64((const __m128i *)(qrow + j));
                __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vi8));
                _mm256_storeu_ps(drow + j, _mm256_mul_ps(vf, vscale));
            }
#endif
            for (; j < embed; j++)
                drow[j] = (float)qrow[j] * scale;
        }

        llm_param_lookup_release(lookup);
        sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
        return;
    }

    if (llm_to_nd(weight_blob, weight_size, &tw) < 0) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_embedding_param_nd: invalid weight blob");
    }
    if (tw.ndim != 2) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_embedding_param_nd: weight must be 2-D");
    }

    int32_t vocab = tw.shape[0];
    int32_t embed = tw.shape[1];
    int out_ndim = ti.ndim + 1;
    if (out_ndim > LLM_MAX_NDIM) {
        llm_param_lookup_release(lookup);
        ERR(ctx, "llm_embedding_param_nd: too many dims");
    }
    int32_t out_shape[LLM_MAX_NDIM];
    for (int i = 0; i < ti.ndim; i++)
        out_shape[i] = ti.shape[i];
    out_shape[ti.ndim] = embed;
    int32_t out_total = ti.total * embed;

    int out_sz;
    void *out = nd_result_alloc(out_ndim, out_shape, out_total, &out_sz);
    if (!out) {
        llm_param_lookup_release(lookup);
        sqlite3_result_error_nomem(ctx);
        return;
    }
    float *dst = (float *)((char *)out + LLM_ND_HDR(out_ndim));

    for (int32_t i = 0; i < ti.total; i++) {
        int idx = (int)roundf(ti.data[i]);
        if (idx < 0 || idx >= vocab) {
            llm_param_lookup_release(lookup);
            sqlite3_free(out);
            ERR(ctx, "llm_embedding_param_nd: index out of range");
        }
        memcpy(dst + (size_t)i * embed,
               tw.data + (size_t)idx * embed,
               (size_t)embed * sizeof(float));
    }

    llm_param_lookup_release(lookup);
    sqlite3_result_blob(ctx, out, out_sz, sqlite3_free);
}

/* ── llm_is_int8_nd(blob) → 1 if INT8, 0 otherwise ──────────────────────── */
static void
sql_is_int8_nd(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    const void *blob = sqlite3_value_blob(argv[0]);
    int sz = sqlite3_value_bytes(argv[0]);
    if (blob && sz >= 4) {
        int32_t magic = *(const int32_t *)blob;
        sqlite3_result_int(ctx, magic == LLM_INT8_MAGIC ? 1 : 0);
    } else {
        sqlite3_result_int(ctx, 0);
    }
}

typedef struct llm_param_lookup_t {
    sqlite3 *lookup_db;
    sqlite3_stmt *lookup_stmt;
} llm_param_lookup_t;

static void llm_param_lookup_destroy(void *raw) {
    llm_param_lookup_t *lookup = (llm_param_lookup_t *)raw;
    if (!lookup)
        return;
    if (lookup->lookup_stmt)
        sqlite3_finalize(lookup->lookup_stmt);
    if (lookup->lookup_db)
        sqlite3_close(lookup->lookup_db);
    sqlite3_free(lookup);
}

static int llm_register_lookup_function(sqlite3 *db,
                                        const char *name,
                                        int narg,
                                        void (*fn)(sqlite3_context *,
                                                   int,
                                                   sqlite3_value **)) {
    llm_param_lookup_t *lookup =
        (llm_param_lookup_t *)sqlite3_malloc(sizeof(llm_param_lookup_t));
    if (!lookup)
        return SQLITE_NOMEM;
    lookup->lookup_db = NULL;
    lookup->lookup_stmt = NULL;

    int rc = sqlite3_create_function_v2(db,
                                        name,
                                        narg,
                                        SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                        lookup,
                                        fn,
                                        NULL,
                                        NULL,
                                        llm_param_lookup_destroy);
    if (rc != SQLITE_OK) {
        llm_param_lookup_destroy(lookup);
    }
    return rc;
}

static int llm_param_lookup_ensure(llm_param_lookup_t *lookup,
                                   sqlite3 *db,
                                   sqlite3_context *ctx) {
    if (lookup->lookup_stmt)
        return SQLITE_OK;

    const char *db_path = sqlite3_db_filename(db, "main");
    if (!db_path || db_path[0] == '\0' || strcmp(db_path, ":memory:") == 0) {
        sqlite3_result_error(
            ctx, "llm_get_param requires a file-backed SQLite database", -1);
        return SQLITE_ERROR;
    }

    int rc = sqlite3_open_v2(db_path,
                             &lookup->lookup_db,
                             SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX,
                             NULL);
    if (rc != SQLITE_OK) {
        sqlite3_result_error(
            ctx,
            sqlite3_errmsg(lookup->lookup_db ? lookup->lookup_db : db),
            -1);
        if (lookup->lookup_db) {
            sqlite3_close(lookup->lookup_db);
            lookup->lookup_db = NULL;
        }
        return rc;
    }

    sqlite3_exec(lookup->lookup_db, "PRAGMA mmap_size=0", NULL, NULL, NULL);
    rc = sqlite3_prepare_v2(lookup->lookup_db,
                            "SELECT data FROM model_params WHERE name = ?1",
                            -1,
                            &lookup->lookup_stmt,
                            NULL);
    if (rc != SQLITE_OK) {
        sqlite3_result_error(ctx, sqlite3_errmsg(lookup->lookup_db), -1);
        sqlite3_close(lookup->lookup_db);
        lookup->lookup_db = NULL;
        lookup->lookup_stmt = NULL;
    }
    return rc;
}

static int llm_param_lookup_step_blob(llm_param_lookup_t *lookup,
                                      sqlite3 *db,
                                      sqlite3_context *ctx,
                                      const char *name,
                                      const void **out_blob,
                                      int *out_size) {
    if (!name || !out_blob || !out_size) {
        sqlite3_result_null(ctx);
        return SQLITE_MISUSE;
    }
    if (llm_param_lookup_ensure(lookup, db, ctx) != SQLITE_OK) {
        return SQLITE_ERROR;
    }

    sqlite3_reset(lookup->lookup_stmt);
    sqlite3_clear_bindings(lookup->lookup_stmt);
    sqlite3_bind_text(lookup->lookup_stmt, 1, name, -1, SQLITE_TRANSIENT);

    int rc = sqlite3_step(lookup->lookup_stmt);
    if (rc == SQLITE_ROW) {
        *out_blob = sqlite3_column_blob(lookup->lookup_stmt, 0);
        *out_size = sqlite3_column_bytes(lookup->lookup_stmt, 0);
        return SQLITE_OK;
    }

    if (rc == SQLITE_DONE) {
        sqlite3_result_error(ctx, "llm parameter not found", -1);
    } else {
        sqlite3_result_error(ctx, sqlite3_errmsg(lookup->lookup_db), -1);
    }
    sqlite3_reset(lookup->lookup_stmt);
    sqlite3_clear_bindings(lookup->lookup_stmt);
    return rc;
}

static void llm_param_lookup_release(llm_param_lookup_t *lookup) {
    if (!lookup || !lookup->lookup_stmt)
        return;
    sqlite3_reset(lookup->lookup_stmt);
    sqlite3_clear_bindings(lookup->lookup_stmt);
}

static void
sql_get_param(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    llm_param_lookup_t *lookup = (llm_param_lookup_t *)sqlite3_user_data(ctx);
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    const unsigned char *name = sqlite3_value_text(argv[0]);

    if (!lookup || !name) {
        sqlite3_result_null(ctx);
        return;
    }

    const void *blob = NULL;
    int size = 0;
    if (llm_param_lookup_step_blob(
            lookup, db, ctx, (const char *)name, &blob, &size) != SQLITE_OK) {
        return;
    }
    sqlite3_result_blob(ctx, blob, size, SQLITE_TRANSIENT);
    llm_param_lookup_release(lookup);
}

/* llm_set_threads(n) — set the runtime thread count for parallel ops.
 * Returns the new value.  n is clamped to [1, 256]. */
static void
sql_set_threads(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    int n = sqlite3_value_int(argv[0]);
    if (n < 1)
        n = 1;
    if (n > 256)
        n = 256;
    g_llm_num_threads = n;
    sqlite3_result_int(ctx, g_llm_num_threads);
}

/* llm_get_threads() — return the current runtime thread count. */
static void
sql_get_threads(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    (void)argv;
    sqlite3_result_int(ctx, g_llm_num_threads);
}

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
