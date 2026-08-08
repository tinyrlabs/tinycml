/**
 * test_conv2d.c - Conv2D layer test suite
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "conv2d.h"
#include "test_harness.h"

/* ============================================
 * Test 1: Create and free
 * ============================================ */
TEST(test_conv2d_create_free) {
    Conv2D *layer = conv2d_create(3, 16, 3, 3, 1, 1, 1, 1);
    ASSERT_NOT_NULL(layer);
    ASSERT_EQ(layer->in_channels, 3);
    ASSERT_EQ(layer->out_channels, 16);
    ASSERT_EQ(layer->kernel_h, 3);
    ASSERT_EQ(layer->kernel_w, 3);
    ASSERT_NOT_NULL(layer->weight);
    ASSERT_NOT_NULL(layer->bias);
    ASSERT_EQ(layer->weight->rows, (size_t)16);
    ASSERT_EQ(layer->weight->cols, (size_t)(3 * 3 * 3));  /* 27 */
    conv2d_free(layer);
}

/* ============================================
 * Test 2: Output shape calculation
 * ============================================ */
TEST(test_conv2d_out_size) {
    /* 32x32 input, 3x3 kernel, stride 1, pad 1 */
    ASSERT_EQ(conv2d_out_size(32, 3, 1, 1), 32);
    /* 32x32 input, 3x3 kernel, stride 2, pad 0 */
    ASSERT_EQ(conv2d_out_size(32, 3, 2, 0), 15);
    /* 32x32 input, 5x5 kernel, stride 1, pad 2 */
    ASSERT_EQ(conv2d_out_size(32, 5, 1, 2), 32);
    /* 8x8 input, 3x3 kernel, stride 2, pad 0 (MaxPool-like) */
    ASSERT_EQ(conv2d_out_size(8, 2, 2, 0), 4);
}

/* ============================================
 * Test 3: im2col correctness (small tensor)
 * ============================================ */
TEST(test_im2col_small) {
    /* 1x1x2x2 input: batch=1, channel=1, 2x2 image
     * Values: [1, 2, 3, 4] (row-major)
     * Kernel: 2x2, stride=1, pad=0
     * Expected output patches: [[1,2,3,4]^T] (single column)
     */
    Matrix *input = matrix_alloc(1, 4);  /* 1 sample, C*H*W = 1*2*2 = 4 */
    input->data[0] = 1.0; input->data[1] = 2.0;
    input->data[2] = 3.0; input->data[3] = 4.0;

    Matrix *col = im2col(input, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0);
    ASSERT_NOT_NULL(col);

    /* col should be (4 × 1): one column with [1,2,3,4] */
    ASSERT_EQ(col->rows, (size_t)4);
    ASSERT_EQ(col->cols, (size_t)1);
    ASSERT_NEAR(col->data[0], 1.0, 1e-6);
    ASSERT_NEAR(col->data[1], 2.0, 1e-6);
    ASSERT_NEAR(col->data[2], 3.0, 1e-6);
    ASSERT_NEAR(col->data[3], 4.0, 1e-6);

    matrix_free(input);
    matrix_free(col);
}

/* ============================================
 * Test 4: im2col with stride and padding
 * ============================================ */
TEST(test_im2col_stride) {
    /* 1x1x3x3 input, 2x2 kernel, stride=2, pad=0
     * Expected: out_h=1, out_w=1, single patch from top-left 2x2
     */
    Matrix *input = matrix_alloc(1, 9);
    for (int i = 0; i < 9; i++) input->data[i] = (double)(i + 1);

    Matrix *col = im2col(input, 1, 1, 3, 3, 2, 2, 2, 2, 0, 0);
    ASSERT_NOT_NULL(col);
    ASSERT_EQ(col->rows, (size_t)4);  /* C*kh*kw = 1*2*2 = 4 */
    ASSERT_EQ(col->cols, (size_t)1);  /* N*out_h*out_w = 1*1*1 = 1 */

    /* Top-left 2x2 of the 3x3 grid (row-major): [1,2,4,5] */
    ASSERT_NEAR(col->data[0], 1.0, 1e-6);
    ASSERT_NEAR(col->data[1], 2.0, 1e-6);
    ASSERT_NEAR(col->data[2], 4.0, 1e-6);
    ASSERT_NEAR(col->data[3], 5.0, 1e-6);

    matrix_free(input);
    matrix_free(col);
}

/* ============================================
 * Test 5: Conv2D forward pass (known weights)
 * ============================================ */
TEST(test_conv2d_forward_known) {
    /* 1x1x2x2 input, 1 filter, 2x2 kernel, stride=1, pad=0
     * Weight: [[1,1,1,1]] (all ones)
     * Bias: [0]
     * Expected output: sum of input patch = [1+2+3+4] = [10]
     */
    double w_data[4] = {1.0, 1.0, 1.0, 1.0};
    double b_data[1] = {0.0};
    Conv2D *layer = conv2d_create_with_weights(1, 1, 2, 2, 1, 1, 0, 0,
                                                w_data, b_data);
    ASSERT_NOT_NULL(layer);

    Matrix *input = matrix_alloc(1, 4);
    input->data[0] = 1.0; input->data[1] = 2.0;
    input->data[2] = 3.0; input->data[3] = 4.0;

    Matrix *out = conv2d_forward(layer, input, 1, 2, 2);
    ASSERT_NOT_NULL(out);
    ASSERT_EQ(out->rows, (size_t)1);
    ASSERT_EQ(out->cols, (size_t)1);  /* 1*1*1 */
    ASSERT_NEAR(out->data[0], 10.0, 1e-6);  /* 1+2+3+4 */

    matrix_free(input);
    matrix_free(out);
    conv2d_free(layer);
}

/* ============================================
 * Test 6: Conv2D with bias
 * ============================================ */
TEST(test_conv2d_forward_bias) {
    double w_data[4] = {1.0, 0.0, 0.0, 1.0};
    double b_data[1] = {5.0};
    Conv2D *layer = conv2d_create_with_weights(1, 1, 2, 2, 1, 1, 0, 0,
                                                w_data, b_data);

    Matrix *input = matrix_alloc(1, 4);
    input->data[0] = 1.0; input->data[1] = 2.0;
    input->data[2] = 3.0; input->data[3] = 4.0;

    Matrix *out = conv2d_forward(layer, input, 1, 2, 2);
    /* 1*1 + 0*2 + 0*3 + 1*4 + 5 = 10 */
    ASSERT_NEAR(out->data[0], 10.0, 1e-6);

    matrix_free(input);
    matrix_free(out);
    conv2d_free(layer);
}

int main(void) {
    printf("Conv2D Tests\n================\n\n");
    RUN_TEST(test_conv2d_create_free);
    RUN_TEST(test_conv2d_out_size);
    RUN_TEST(test_im2col_small);
    RUN_TEST(test_im2col_stride);
    RUN_TEST(test_conv2d_forward_known);
    RUN_TEST(test_conv2d_forward_bias);
    TEST_SUMMARY();
}
