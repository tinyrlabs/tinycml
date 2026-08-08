/**
 * pool2d.h - 2D Max Pooling Layer for tinycml
 *
 * Downsamples spatial dimensions by taking the maximum over
 * non-overlapping rectangular regions. No learnable parameters.
 *
 * Feature flag: CML_ENABLE_POOL2D
 */

#ifndef POOL2D_H
#define POOL2D_H

#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int pool_h;          /* pooling window height (typically 2) */
    int pool_w;          /* pooling window width (typically 2) */
    int stride_h;        /* vertical stride (typically 2) */
    int stride_w;        /* horizontal stride (typically 2) */

    /* Cache for backward pass: argmax indices */
    int *argmax_cache;   /* size: N*C*out_h*out_w */
    int input_h;         /* cached input height */
    int input_w;         /* cached input width */
    int input_c;         /* cached input channels */
    int input_n;         /* cached batch size */
} MaxPool2D;

/**
 * Create a MaxPool2D layer.
 *
 * @param ph  Pool height (default 2)
 * @param pw  Pool width (default 2)
 * @param sh  Stride height (default 2)
 * @param sw  Stride width (default 2)
 */
MaxPool2D* maxpool2d_create(int ph, int pw, int sh, int sw);

void maxpool2d_free(MaxPool2D *layer);

/**
 * Run max pooling forward pass.
 *
 * @param layer  MaxPool2D layer
 * @param input  Input matrix (N × C*H*W), NCHW flattened, row-major
 * @param n      Batch size
 * @param c      Number of channels
 * @param h      Input height
 * @param w      Input width
 * @return Output matrix (N × C*out_h*out_w), caller frees.
 */
Matrix* maxpool2d_forward(MaxPool2D *layer, const Matrix *input,
                          int n, int c, int h, int w);

/**
 * Output size after pooling.
 * out = floor((input_size - pool_size) / stride) + 1
 */
int pool2d_out_size(int input_size, int pool_size, int stride);

#ifdef __cplusplus
}
#endif

#endif /* POOL2D_H */
