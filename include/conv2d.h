/**
 * conv2d.h - 2D Convolution Layer for tinycml
 *
 * Implements Conv2D forward pass using im2col + GEMM (matrix multiply).
 * Designed for CNN inference on resource-constrained embedded systems.
 *
 * Memory layout: NCHW (batch, channels, height, width)
 * - Input:  Matrix(N, C*H*W)  — flattened row-major
 * - Weight: Matrix(K, C*R*S)  — K filters, each C×R×S
 * - Bias:   Matrix(1, K)       — one bias per output channel
 * - Output: Matrix(N, K*out_h*out_w)
 *
 * Uses cml_pool allocator when CML_USE_POOL is defined.
 * Feature flag: CML_ENABLE_CONV2D
 */

#ifndef CONV2D_H
#define CONV2D_H

#include "matrix.h"
#include "estimator.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================
 * Conv2D Layer
 * ============================================ */

typedef struct {
    int in_channels;      /* C  — input channels (e.g., 3 for RGB) */
    int out_channels;     /* K  — number of filters */
    int kernel_h;         /* R  — kernel height */
    int kernel_w;         /* S  — kernel width */
    int stride_h;         /* vertical stride */
    int stride_w;         /* horizontal stride */
    int pad_h;            /* vertical zero-padding */
    int pad_w;            /* horizontal zero-padding */

    /* Parameters (trained) */
    Matrix *weight;       /* (out_channels × (in_channels * kernel_h * kernel_w)) */
    Matrix *bias;         /* (1 × out_channels) */

    /* Gradients (for training — optional) */
    Matrix *d_weight;
    Matrix *d_bias;

    /* Forward cache (for im2col output reusable in backward pass) */
    Matrix *col_cache;    /* im2col result: (in_c*kh*kw × N*out_h*out_w) */
    int input_h;          /* cached input height */
    int input_w;          /* cached input width */
    int input_n;          /* cached batch size */
} Conv2D;

/* ============================================
 * Lifecycle
 * ============================================ */

/**
 * Create a Conv2D layer with random Xavier-initialized weights.
 *
 * @param in_c    Input channels
 * @param out_c   Number of filters
 * @param kh      Kernel height
 * @param kw      Kernel width
 * @param sh      Stride height (default 1)
 * @param sw      Stride width (default 1)
 * @param ph      Padding height (default 0)
 * @param pw      Padding width (default 0)
 * @return Conv2D pointer, or NULL on allocation failure
 */
Conv2D* conv2d_create(int in_c, int out_c, int kh, int kw,
                      int sh, int sw, int ph, int pw);

/**
 * Create a Conv2D with pre-loaded weights (from exported C arrays).
 * bias can be NULL to skip (zero bias).
 */
Conv2D* conv2d_create_with_weights(int in_c, int out_c, int kh, int kw,
                                   int sh, int sw, int ph, int pw,
                                   const double *weight_data,
                                   const double *bias_data);

/** Free Conv2D and all internal allocations. */
void conv2d_free(Conv2D *layer);

/**
 * Compute gradients (backward pass). Must call forward() first.
 * @return Gradient w.r.t. input, caller frees.
 */
Matrix* conv2d_backward(Conv2D *layer, const Matrix *dout);

/**
 * Apply accumulated gradients with SGD step.
 */
void conv2d_update_weights(Conv2D *layer, double learning_rate);

/* ============================================
 * Forward Pass
 * ============================================ */

/**
 * Run Conv2D forward pass.
 *
 * @param layer  Conv2D layer
 * @param input  Input matrix (N × C*H*W), row-major flattened
 * @param n      Batch size
 * @param h      Input height
 * @param w      Input width
 * @return Output matrix (N × K*out_h*out_w), caller frees.
 */
Matrix* conv2d_forward(Conv2D *layer, const Matrix *input,
                       int n, int h, int w);

/**
 * Compute output spatial dimensions.
 *
 * out_h = floor((h + 2*pad_h - kernel_h) / stride_h) + 1
 * out_w = floor((w + 2*pad_w - kernel_w) / stride_w) + 1
 */
int conv2d_out_size(int input_size, int kernel, int stride, int pad);

/* ============================================
 * im2col Utility
 * ============================================ */

/**
 * Convert image regions to matrix columns (im2col).
 *
 * Given a 4D tensor (N, C, H, W) stored as a flat Matrix(N, C*H*W),
 * extracts each convolution patch and stores it as a column.
 *
 * @param input    Input matrix (N × C*H*W)
 * @param n        Batch size
 * @param c        Channels
 * @param h        Height
 * @param w        Width
 * @param kh       Kernel height
 * @param kw       Kernel width
 * @param sh       Stride height
 * @param sw       Stride width
 * @param ph       Padding height
 * @param pw       Padding width
 * @return Column matrix ((C*kh*kw) × (N*out_h*out_w)), caller frees.
 */
Matrix* im2col(const Matrix *input, int n, int c, int h, int w,
               int kh, int kw, int sh, int sw, int ph, int pw);

/**
 * col2im — reverse of im2col.
 * Accumulates gradient columns back into image gradient.
 *
 * @param dcol    Gradient of loss w.r.t. col matrix ((C*kh*kw) × (N*out_h*out_w))
 * @param n,c,h,w Original input dimensions
 * @param kh,kw   Kernel size
 * @param sh,sw   Stride
 * @param ph,pw   Padding
 * @return Image gradient (N × C*H*W), caller frees.
 */
Matrix* col2im(const Matrix *dcol, int n, int c, int h, int w,
               int kh, int kw, int sh, int sw, int ph, int pw);

#ifdef __cplusplus
}
#endif

#endif /* CONV2D_H */
