/**
 * test_cnn.c - CNN model test suite
 */
#include <stdio.h>
#include "cnn.h"
#include "test_harness.h"

TEST(test_cnn_create_free) {
    CNNModel *net = cnn_create(32, 32, 3, 5);
    ASSERT_NOT_NULL(net);
    ASSERT_EQ(net->input_h, 32);
    ASSERT_EQ(net->input_w, 32);
    ASSERT_EQ(net->input_c, 3);
    ASSERT_EQ(net->n_classes, 5);
    cnn_free(net);
}

TEST(test_cnn_build_architecture) {
    /* Build: Conv(16,3x3) → ReLU → MaxPool(2x2) → Conv(32,3x3) → ReLU → MaxPool → Flatten → Dense(128) → Dense(5) → Softmax */
    CNNModel *net = cnn_create(32, 32, 3, 5);
    cnn_add_conv2d(net, 16, 3, 3, 1, 1);
    cnn_add_relu(net);
    cnn_add_maxpool(net, 2, 2, 2);  /* 32→16 */
    cnn_add_conv2d(net, 32, 3, 3, 1, 1);
    cnn_add_relu(net);
    cnn_add_maxpool(net, 2, 2, 2);  /* 16→8 */
    cnn_add_flatten(net);            /* 32*8*8 = 2048? Wait: conv2 keeps 16x16 then 8x8 = 32 channels * 8 * 8 = 2048 */
    cnn_add_dense(net, 128, ACTIVATION_RELU);
    cnn_add_dense(net, 5, ACTIVATION_SIGMOID);
    cnn_add_softmax(net);

    ASSERT_EQ(net->n_blocks, 10);
    ASSERT_EQ(net->n_classes, 5);
    cnn_free(net);
}

TEST(test_cnn_forward_shape) {
    /* Tiny CNN: Conv(4,3x3) → ReLU → MaxPool(2x2) → Flatten → Dense(5) → Softmax
     * Input: 1x3x8x8 = 192 values
     * After Conv(4,3x3,pad=1): 8x8x4 = 256
     * After MaxPool(2x2,stride=2): 4x4x4 = 64
     * Flatten: 64
     * Dense(5): output = 1x5
     */
    CNNModel *net = cnn_create(8, 8, 3, 5);
    cnn_add_conv2d(net, 4, 3, 3, 1, 1);  /* 8→8, 3ch→4ch */
    cnn_add_relu(net);
    cnn_add_maxpool(net, 2, 2, 2);        /* 8→4 */
    cnn_add_flatten(net);                  /* 4*4*4 = 64 */
    cnn_add_dense(net, 5, ACTIVATION_SIGMOID);
    cnn_add_softmax(net);

    /* Create random input */
    Matrix *input = matrix_alloc(1, 3 * 8 * 8);
    for (size_t i = 0; i < 192; i++) {
        input->data[i] = ((double)(i % 100) - 50.0) / 50.0;
    }

    Matrix *out = cnn_forward(net, input);
    ASSERT_NOT_NULL(out);
    ASSERT_EQ(out->rows, (size_t)1);
    ASSERT_EQ(out->cols, (size_t)5);

    /* Check softmax: sum per row ≈ 1.0 */
    double sum = 0.0;
    for (size_t i = 0; i < 5; i++) sum += out->data[i];
    ASSERT_NEAR(sum, 1.0, 0.01);

    matrix_free(input);
    matrix_free(out);
    cnn_free(net);
}

TEST(test_cnn_estimator_vtable) {
    CNNModel *net = cnn_create(8, 8, 3, 3);
    cnn_add_conv2d(net, 2, 3, 3, 1, 1);
    cnn_add_relu(net);
    cnn_add_flatten(net);
    cnn_add_dense(net, 3, ACTIVATION_SIGMOID);
    cnn_add_softmax(net);

    Estimator *est = &net->base;
    ASSERT_NOT_NULL(est->fit);
    ASSERT_NOT_NULL(est->predict);
    ASSERT_NOT_NULL(est->predict_proba);
    ASSERT_NOT_NULL(est->score);
    ASSERT_NOT_NULL(est->free);

    /* fit stub */
    est->fit(est, NULL, NULL);
    ASSERT_EQ(est->is_fitted, 1);

    /* predict with 2 samples */
    Matrix *X = matrix_alloc(2, 3 * 8 * 8);
    Matrix *y = matrix_alloc(2, 1);
    for (size_t i = 0; i < 384; i++) X->data[i] = 0.0;
    y->data[0] = 1.0; y->data[1] = 2.0;

    Matrix *probs = est->predict_proba(est, X);
    ASSERT_NOT_NULL(probs);
    ASSERT_EQ(probs->rows, (size_t)2);
    ASSERT_EQ(probs->cols, (size_t)3);

    Matrix *preds = est->predict(est, X);
    ASSERT_NOT_NULL(preds);

    est->score(est, X, y);
    /* With zero input, all classes get equal probability → predict class 0 */
    /* Score is meaningless here but shouldn't crash */

    matrix_free(X);
    matrix_free(y);
    matrix_free(probs);
    matrix_free(preds);
    cnn_free(net);
}

int main(void) {
    printf("CNN Model Tests\n================\n\n");
    RUN_TEST(test_cnn_create_free);
    RUN_TEST(test_cnn_build_architecture);
    RUN_TEST(test_cnn_forward_shape);
    RUN_TEST(test_cnn_estimator_vtable);
    TEST_SUMMARY();
}
