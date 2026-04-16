/*
 * @Author: xuxianghong12
 * @Date: 2026-04-13 18:39:02
 * @LastEditors: xuxianghong12
 * @LastEditTime: 2026-04-14 21:18:54
 * @SPDX-License-Identifier: Apache-2.0
 */
/*
 * Minimal CBLAS compatibility layer for llm.sql.
 *
 * This header exists so llm_ops_x86.c can keep a stable internal call
 * surface without depending on an external BLAS implementation.
 * Only the small subset of APIs used by the extension is implemented.
 */

#ifndef LLM_SQL_CBLAS_H
#define LLM_SQL_CBLAS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CblasRowMajor = 101,
} CBLAS_LAYOUT;

typedef enum {
    CblasNoTrans = 111,
    CblasTrans = 112,
} CBLAS_TRANSPOSE;

static inline void cblas_saxpy(const int n,
                               const float alpha,
                               const float *x,
                               const int incx,
                               float *y,
                               const int incy) {
    for (int i = 0; i < n; ++i) {
        y[i * incy] += alpha * x[i * incx];
    }
}

static inline void
cblas_sscal(const int n, const float alpha, float *x, const int incx) {
    for (int i = 0; i < n; ++i) {
        x[i * incx] *= alpha;
    }
}

static inline float cblas_sdot(const int n,
                               const float *x,
                               const int incx,
                               const float *y,
                               const int incy) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i * incx] * y[i * incy];
    }
    return sum;
}

static inline void cblas_sgemv(const CBLAS_LAYOUT layout,
                               const CBLAS_TRANSPOSE trans,
                               const int m,
                               const int n,
                               const float alpha,
                               const float *a,
                               const int lda,
                               const float *x,
                               const int incx,
                               const float beta,
                               float *y,
                               const int incy) {
    if (layout != CblasRowMajor) {
        return;
    }

    const int beta_is_zero = (beta == 0.0f);

    if (trans == CblasNoTrans) {
        for (int row = 0; row < m; ++row) {
            float sum = 0.0f;
            const float *row_ptr = a + row * lda;
            for (int col = 0; col < n; ++col) {
                sum += row_ptr[col] * x[col * incx];
            }
            if (beta_is_zero) {
                y[row * incy] = alpha * sum;
            } else {
                y[row * incy] = alpha * sum + beta * y[row * incy];
            }
        }
        return;
    }

    for (int col = 0; col < n; ++col) {
        float sum = 0.0f;
        for (int row = 0; row < m; ++row) {
            sum += a[row * lda + col] * x[row * incx];
        }
        if (beta_is_zero) {
            y[col * incy] = alpha * sum;
        } else {
            y[col * incy] = alpha * sum + beta * y[col * incy];
        }
    }
}

static inline void cblas_sgemm(const CBLAS_LAYOUT layout,
                               const CBLAS_TRANSPOSE transa,
                               const CBLAS_TRANSPOSE transb,
                               const int m,
                               const int n,
                               const int k,
                               const float alpha,
                               const float *a,
                               const int lda,
                               const float *b,
                               const int ldb,
                               const float beta,
                               float *c,
                               const int ldc) {
    if (layout != CblasRowMajor) {
        return;
    }

    const int beta_is_zero = (beta == 0.0f);

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                const float a_val = (transa == CblasNoTrans)
                                        ? a[row * lda + inner]
                                        : a[inner * lda + row];
                const float b_val = (transb == CblasNoTrans)
                                        ? b[inner * ldb + col]
                                        : b[col * ldb + inner];
                sum += a_val * b_val;
            }
            if (beta_is_zero) {
                c[row * ldc + col] = alpha * sum;
            } else {
                c[row * ldc + col] = alpha * sum + beta * c[row * ldc + col];
            }
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif