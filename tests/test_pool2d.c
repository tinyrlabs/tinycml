/**
 * test_pool2d.c - MaxPool2D test suite
 */
#include <stdio.h>
#include "pool2d.h"
#include "test_harness.h"

TEST(test_pool2d_create_free) {
    MaxPool2D *layer = maxpool2d_create(2, 2, 2, 2);
    ASSERT_NOT_NULL(layer);
    ASSERT_EQ(layer->pool_h, 2);
    ASSERT_EQ(layer->pool_w, 2);
    maxpool2d_free(layer);
}

TEST(test_pool2d_out_size) {
    ASSERT_EQ(pool2d_out_size(32, 2, 2), 16);
    ASSERT_EQ(pool2d_out_size(16, 2, 2), 8);
    ASSERT_EQ(pool2d_out_size(8, 2, 2), 4);
    ASSERT_EQ(pool2d_out_size(4, 2, 2), 2);
    ASSERT_EQ(pool2d_out_size(7, 2, 2), 3);  /* (7-2)/2+1 = 3 */
}

TEST(test_pool2d_forward_2x2) {
    /* 1x1x4x4 input, maxpool 2x2, stride 2
     * Input:
     *   1  2  3  4
     *   5  6  7  8
     *   9 10 11 12
     *  13 14 15 16
     * Output (2x2): max of each 2x2 block = [6, 8, 14, 16]
     */
    Matrix *input = matrix_alloc(1, 16);  /* 1*1*4*4 */
    for (int i = 0; i < 16; i++) input->data[i] = (double)(i + 1);

    MaxPool2D *layer = maxpool2d_create(2, 2, 2, 2);
    Matrix *out = maxpool2d_forward(layer, input, 1, 1, 4, 4);

    ASSERT_NOT_NULL(out);
    ASSERT_EQ(out->rows, (size_t)1);
    ASSERT_EQ(out->cols, (size_t)4);  /* 1*2*2 */
    ASSERT_NEAR(out->data[0], 6.0, 1e-6);   /* max(1,2,5,6) */
    ASSERT_NEAR(out->data[1], 8.0, 1e-6);   /* max(3,4,7,8) */
    ASSERT_NEAR(out->data[2], 14.0, 1e-6);  /* max(9,10,13,14) */
    ASSERT_NEAR(out->data[3], 16.0, 1e-6);  /* max(11,12,15,16) */

    matrix_free(input);
    matrix_free(out);
    maxpool2d_free(layer);
}

TEST(test_pool2d_forward_multichannel) {
    /* 1x2x2x2 input, 2 channels, 2x2 each, maxpool 2x2
     * Channel 0: [1,2 ; 3,4] -> max = 4
     * Channel 1: [5,6 ; 7,8] -> max = 8
     * Output: [4, 8]
     */
    Matrix *input = matrix_alloc(1, 8);
    /* C0 H0W0, C0H0W1, C0H1W0, C0H1W1 = 1,2,3,4 */
    /* C1 H0W0, C1H0W1, C1H1W0, C1H1W1 = 5,6,7,8 */
    for (int i = 0; i < 8; i++) input->data[i] = (double)(i + 1);

    MaxPool2D *layer = maxpool2d_create(2, 2, 2, 2);
    Matrix *out = maxpool2d_forward(layer, input, 1, 2, 2, 2);

    ASSERT_EQ(out->cols, (size_t)2);  /* C*1*1 = 2 */
    ASSERT_NEAR(out->data[0], 4.0, 1e-6);
    ASSERT_NEAR(out->data[1], 8.0, 1e-6);

    matrix_free(input);
    matrix_free(out);
    maxpool2d_free(layer);
}

TEST(test_pool2d_backward) {
    /* 1x1x4x4, pool 2x2 stride 2
     * Forward: [6, 8, 14, 16]
     * Backward: dout = [1, 2, 3, 4]
     * Positions: 6=idx5, 8=idx7, 14=idx13, 16=idx15
     * Expected: dimg[idx5]=1, dimg[idx7]=2, dimg[idx13]=3, dimg[idx15]=4
     */
    Matrix *input = matrix_alloc(1, 16);
    for (int i = 0; i < 16; i++) input->data[i] = (double)(i + 1);

    MaxPool2D *layer = maxpool2d_create(2, 2, 2, 2);
    Matrix *out = maxpool2d_forward(layer, input, 1, 1, 4, 4);

    Matrix *dout = matrix_alloc(1, 4);
    dout->data[0] = 1.0; dout->data[1] = 2.0;
    dout->data[2] = 3.0; dout->data[3] = 4.0;

    Matrix *dimg = maxpool2d_backward(layer, dout);
    ASSERT_NOT_NULL(dimg);

    /* Only argmax positions should have non-zero gradient */
    ASSERT_NEAR(dimg->data[5], 1.0, 1e-6);   /* max of top-left */
    ASSERT_NEAR(dimg->data[7], 2.0, 1e-6);   /* max of top-right */
    ASSERT_NEAR(dimg->data[13], 3.0, 1e-6);  /* max of bottom-left */
    ASSERT_NEAR(dimg->data[15], 4.0, 1e-6);  /* max of bottom-right */
    /* All other positions should be 0 */
    ASSERT_NEAR(dimg->data[0], 0.0, 1e-6);

    matrix_free(input);
    matrix_free(out);
    matrix_free(dout);
    matrix_free(dimg);
    maxpool2d_free(layer);
}

int main(void) {
    printf("MaxPool2D Tests\n================\n\n");
    RUN_TEST(test_pool2d_create_free);
    RUN_TEST(test_pool2d_out_size);
    RUN_TEST(test_pool2d_forward_2x2);
    RUN_TEST(test_pool2d_forward_multichannel);
    RUN_TEST(test_pool2d_backward);
    TEST_SUMMARY();
}
