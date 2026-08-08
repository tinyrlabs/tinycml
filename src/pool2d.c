/**
 * pool2d.c - 2D Max Pooling Layer implementation
 */

#include "pool2d.h"
#include <stdlib.h>
#include <string.h>

int pool2d_out_size(int input_size, int pool_size, int stride) {
    return (input_size - pool_size) / stride + 1;
}

MaxPool2D* maxpool2d_create(int ph, int pw, int sh, int sw) {
    MaxPool2D *layer = calloc(1, sizeof(MaxPool2D));
    if (!layer) return NULL;

    layer->pool_h = ph;
    layer->pool_w = pw;
    layer->stride_h = sh;
    layer->stride_w = sw;
    return layer;
}

void maxpool2d_free(MaxPool2D *layer) {
    if (!layer) return;
    free(layer->argmax_cache);
    free(layer);
}

Matrix* maxpool2d_forward(MaxPool2D *layer, const Matrix *input,
                          int n, int c, int h, int w) {
    int ph = layer->pool_h;
    int pw = layer->pool_w;
    int sh = layer->stride_h;
    int sw = layer->stride_w;

    int out_h = pool2d_out_size(h, ph, sh);
    int out_w = pool2d_out_size(w, pw, sw);

    /* Output: (N × C*out_h*out_w) */
    Matrix *output = matrix_alloc((size_t)n, (size_t)c * out_h * out_w);
    if (!output) return NULL;

    /* Allocate argmax cache for potential backward pass */
    int cache_size = n * c * out_h * out_w;
    free(layer->argmax_cache);
    layer->argmax_cache = malloc((size_t)cache_size * sizeof(int));
    layer->input_n = n;
    layer->input_c = c;
    layer->input_h = h;
    layer->input_w = w;

    /* For each element in the output */
    for (int ni = 0; ni < n; ni++) {
        for (int ci = 0; ci < c; ci++) {
            /* Channel offset in flattened input/output */
            size_t input_ch_offset = (size_t)ni * c * h * w + (size_t)ci * h * w;
            size_t output_ch_offset = (size_t)ni * c * out_h * out_w + (size_t)ci * out_h * out_w;

            for (int oy = 0; oy < out_h; oy++) {
                for (int ox = 0; ox < out_w; ox++) {
                    double max_val = -1e308;  /* -inf */
                    int max_idx = -1;

                    for (int ky = 0; ky < ph; ky++) {
                        for (int kx = 0; kx < pw; kx++) {
                            int iy = oy * sh + ky;
                            int ix = ox * sw + kx;

                            if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
                                size_t idx = input_ch_offset + (size_t)iy * w + ix;
                                if (input->data[idx] > max_val) {
                                    max_val = input->data[idx];
                                    max_idx = (int)idx;
                                }
                            }
                        }
                    }

                    int out_idx = (int)(output_ch_offset + (size_t)oy * out_w + ox);
                    output->data[out_idx] = max_val;

                    /* Save argmax */
                    int cache_idx = ni * c * out_h * out_w + ci * out_h * out_w + oy * out_w + ox;
                    if (layer->argmax_cache) {
                        layer->argmax_cache[cache_idx] = max_idx;
                    }
                }
            }
        }
    }

    return output;
}

Matrix* maxpool2d_backward(MaxPool2D *layer, const Matrix *dout) {
    if (!layer->argmax_cache) return NULL;

    int n = layer->input_n;
    int c = layer->input_c;
    int h = layer->input_h;
    int w = layer->input_w;
    int ph = layer->pool_h;
    int pw = layer->pool_w;
    int sh = layer->stride_h;
    int sw = layer->stride_w;

    int out_h = pool2d_out_size(h, ph, sh);
    int out_w = pool2d_out_size(w, pw, sw);

    Matrix *dimg = matrix_alloc((size_t)n, (size_t)c * h * w);
    if (!dimg) return NULL;

    for (int ni = 0; ni < n; ni++) {
        for (int ci = 0; ci < c; ci++) {
            size_t out_offset = (size_t)ni * c * out_h * out_w + (size_t)ci * out_h * out_w;

            for (int oy = 0; oy < out_h; oy++) {
                for (int ox = 0; ox < out_w; ox++) {
                    int cache_idx = ni * c * out_h * out_w + ci * out_h * out_w + oy * out_w + ox;
                    int max_pos = layer->argmax_cache[cache_idx];
                    double grad = dout->data[out_offset + (size_t)oy * out_w + ox];
                    dimg->data[max_pos] += grad;
                }
            }
        }
    }

    return dimg;
}
