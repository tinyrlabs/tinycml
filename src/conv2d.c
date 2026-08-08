/**
 * conv2d.c - 2D Convolution Layer implementation
 *
 * im2col + GEMM approach for embedded efficiency.
 * Uses tinycml's matrix_matmul for the heavy lifting.
 */

#include "conv2d.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Random seed state for Xavier init */
static unsigned int conv_seed = 42;

/* ============================================
 * Helpers
 * ============================================ */

static double xavier_uniform(int fan_in, int fan_out) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    conv_seed = conv_seed * 1103515245 + 12345;
    double r = (double)(conv_seed & 0x7FFFFFFF) / 0x7FFFFFFF;
    return (2.0 * r - 1.0) * limit;
}

int conv2d_out_size(int input_size, int kernel, int stride, int pad) {
    return (input_size + 2 * pad - kernel) / stride + 1;
}

/* ============================================
 * im2col
 * ============================================ */

Matrix* im2col(const Matrix *input, int n, int c, int h, int w,
               int kh, int kw, int sh, int sw, int ph, int pw) {
    int out_h = conv2d_out_size(h, kh, sh, ph);
    int out_w = conv2d_out_size(w, kw, sw, pw);
    int col_rows = c * kh * kw;       /* each patch: C*kh*kw elements */
    int col_cols = n * out_h * out_w;  /* total patches across batch */

    Matrix *col = matrix_alloc(col_rows, col_cols);
    if (!col) return NULL;

    /* Fill im2col matrix */
    for (int ni = 0; ni < n; ni++) {
        for (int ci = 0; ci < c; ci++) {
            for (int ky = 0; ky < kh; ky++) {
                for (int kx = 0; kx < kw; kx++) {
                    int row_idx = (ci * kh + ky) * kw + kx;  /* row in col matrix */

                    for (int oy = 0; oy < out_h; oy++) {
                        for (int ox = 0; ox < out_w; ox++) {
                            int iy = oy * sh + ky - ph;
                            int ix = ox * sw + kx - pw;

                            double val = 0.0;
                            if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                                /* input[ni, ci, iy, ix] = input->data[ni * c*h*w + ci * h*w + iy * w + ix] */
                                val = input->data[ni * c * h * w + ci * h * w + iy * w + ix];
                            }

                            int col_col = ni * out_h * out_w + oy * out_w + ox;
                            col->data[row_idx * col_cols + col_col] = val;
                        }
                    }
                }
            }
        }
    }

    return col;
}

/* ============================================
 * Conv2D Lifecycle
 * ============================================ */

Conv2D* conv2d_create(int in_c, int out_c, int kh, int kw,
                      int sh, int sw, int ph, int pw) {
    Conv2D *layer = calloc(1, sizeof(Conv2D));
    if (!layer) return NULL;

    layer->in_channels = in_c;
    layer->out_channels = out_c;
    layer->kernel_h = kh;
    layer->kernel_w = kw;
    layer->stride_h = sh;
    layer->stride_w = sw;
    layer->pad_h = ph;
    layer->pad_w = pw;

    /* Weight: (out_c × (in_c * kh * kw)) */
    int weight_rows = out_c;
    int weight_cols = in_c * kh * kw;
    layer->weight = matrix_alloc(weight_rows, weight_cols);
    if (!layer->weight) { free(layer); return NULL; }

    /* Xavier uniform init */
    int fan_in = in_c * kh * kw;
    int fan_out = out_c * kh * kw;
    for (size_t i = 0; i < (size_t)weight_rows * weight_cols; i++) {
        layer->weight->data[i] = xavier_uniform(fan_in, fan_out);
    }

    /* Bias: (1 × out_c) */
    layer->bias = matrix_alloc(1, out_c);
    if (!layer->bias) { conv2d_free(layer); return NULL; }
    /* Bias initialized to zero (matrix_alloc zeros by default) */

    return layer;
}

Conv2D* conv2d_create_with_weights(int in_c, int out_c, int kh, int kw,
                                    int sh, int sw, int ph, int pw,
                                    const double *weight_data,
                                    const double *bias_data) {
    Conv2D *layer = calloc(1, sizeof(Conv2D));
    if (!layer) return NULL;

    layer->in_channels = in_c;
    layer->out_channels = out_c;
    layer->kernel_h = kh;
    layer->kernel_w = kw;
    layer->stride_h = sh;
    layer->stride_w = sw;
    layer->pad_h = ph;
    layer->pad_w = pw;

    int weight_rows = out_c;
    int weight_cols = in_c * kh * kw;
    layer->weight = matrix_alloc(weight_rows, weight_cols);
    if (!layer->weight) { free(layer); return NULL; }
    if (weight_data) {
        memcpy(layer->weight->data, weight_data,
               (size_t)weight_rows * weight_cols * sizeof(double));
    }

    layer->bias = matrix_alloc(1, out_c);
    if (!layer->bias) { conv2d_free(layer); return NULL; }
    if (bias_data) {
        memcpy(layer->bias->data, bias_data,
               (size_t)out_c * sizeof(double));
    }

    return layer;
}

void conv2d_free(Conv2D *layer) {
    if (!layer) return;
    if (layer->weight) matrix_free(layer->weight);
    if (layer->bias) matrix_free(layer->bias);
    if (layer->d_weight) matrix_free(layer->d_weight);
    if (layer->d_bias) matrix_free(layer->d_bias);
    if (layer->col_cache) matrix_free(layer->col_cache);
    free(layer);
}

/* ============================================
 * Forward Pass
 * ============================================ */

Matrix* conv2d_forward(Conv2D *layer, const Matrix *input,
                       int n, int h, int w) {
    int c = layer->in_channels;
    int k = layer->out_channels;
    int kh = layer->kernel_h;
    int kw = layer->kernel_w;
    int sh = layer->stride_h;
    int sw = layer->stride_w;
    int ph = layer->pad_h;
    int pw = layer->pad_w;

    int out_h = conv2d_out_size(h, kh, sh, ph);
    int out_w = conv2d_out_size(w, kw, sw, pw);

    /* Step 1: im2col — convert input to column matrix */
    Matrix *col = im2col(input, n, c, h, w, kh, kw, sh, sw, ph, pw);
    if (!col) return NULL;

    /* Cache for potential backward pass */
    if (layer->col_cache) matrix_free(layer->col_cache);
    layer->col_cache = matrix_copy(col);
    layer->input_h = h;
    layer->input_w = w;
    layer->input_n = n;

    /* Step 2: Weight × Col → (out_c × N*out_h*out_w) */
    Matrix *gemm_result = matrix_matmul(layer->weight, col);
    matrix_free(col);
    if (!gemm_result) return NULL;

    /* Step 3: Add bias to each output channel */
    for (int i = 0; i < k; i++) {
        double b = layer->bias->data[i];
        for (int j = 0; j < n * out_h * out_w; j++) {
            gemm_result->data[i * n * out_h * out_w + j] += b;
        }
    }

    /* Step 4: Transpose+reshape to (N × K*out_h*out_w) — NCHW order */
    Matrix *output = matrix_alloc(n, (size_t)k * out_h * out_w);
    if (!output) { matrix_free(gemm_result); return NULL; }

    for (int ni = 0; ni < n; ni++) {
        for (int ki = 0; ki < k; ki++) {
            for (int oy = 0; oy < out_h; oy++) {
                for (int ox = 0; ox < out_w; ox++) {
                    /* gemm_result[k, n*out_h*out_w + oy*out_w + ox] -> output[n, k*out_h*out_w + oy*out_w + ox] */
                    int src_idx = oy * out_w + ox + ni * out_h * out_w;
                    int dst_idx = ni * k * out_h * out_w + ki * out_h * out_w + oy * out_w + ox;
                    output->data[dst_idx] = gemm_result->data[ki * n * out_h * out_w + src_idx];
                }
            }
        }
    }

    matrix_free(gemm_result);
    return output;
}
