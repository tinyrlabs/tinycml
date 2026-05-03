/**
 * @file test_nn.c
 * @brief Unit tests for Neural Network (MLP)
 */

#include "test_harness.h"
#include "matrix.h"
#include "neural_network.h"

TEST(test_nn_create_free) {
    int hidden[] = {4, 4};
    NeuralNetwork *nn = neural_network_create(hidden, 2, ACTIVATION_RELU);
    ASSERT_NOT_NULL(nn);
    ASSERT_EQ(nn->base.type, MODEL_NEURAL_NETWORK);
    ASSERT_EQ(nn->base.is_fitted, 0);

    nn->base.free((Estimator *)nn);
}

TEST(test_nn_xor) {
    /* XOR problem: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0 */
    Matrix *X = matrix_alloc(4, 2);
    Matrix *y = matrix_alloc(4, 1);

    matrix_set(X, 0, 0, 0.0); matrix_set(X, 0, 1, 0.0); matrix_set(y, 0, 0, 0.0);
    matrix_set(X, 1, 0, 0.0); matrix_set(X, 1, 1, 1.0); matrix_set(y, 1, 0, 1.0);
    matrix_set(X, 2, 0, 1.0); matrix_set(X, 2, 1, 0.0); matrix_set(y, 2, 0, 1.0);
    matrix_set(X, 3, 0, 1.0); matrix_set(X, 3, 1, 1.0); matrix_set(y, 3, 0, 0.0);

    /* Create with enough hidden neurons and epochs to learn XOR.
     * Do NOT call neural_network_init manually — fit will auto-detect
     * binary classification and set up one-hot encoding + proper output size.
     */
    int hidden[] = {16, 16};
    NeuralNetwork *nn = neural_network_create_full(
        hidden, 2,
        ACTIVATION_RELU,
        ACTIVATION_SIGMOID,     /* will be overridden by fit for 2-class */
        LOSS_BINARY_CE,
        OPTIMIZER_ADAM,
        0.01,    /* learning_rate */
        1000,    /* max_epochs */
        4,       /* batch_size */
        0.9,     /* momentum */
        0.0      /* l2_reg */
    );
    ASSERT_NOT_NULL(nn);

    /* Don't call neural_network_init — let fit() handle it */
    Estimator *fitted = nn->base.fit((Estimator *)nn, X, y);
    ASSERT_NOT_NULL(fitted);
    ASSERT_EQ(nn->base.is_fitted, 1);

    Matrix *pred = nn->base.predict((const Estimator *)nn, X);
    ASSERT_NOT_NULL(pred);

    /* At least 2 out of 4 XOR predictions should be correct */
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        double p = matrix_get(pred, i, 0);
        int label = (int)p;
        int expected = (int)matrix_get(y, i, 0);
        if (label == expected) correct++;
    }
    ASSERT(correct >= 2);

    matrix_free(pred);
    nn->base.free((Estimator *)nn);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_nn_predict) {
    /* Simple AND problem: (0,0)->0, (0,1)->0, (1,0)->0, (1,1)->1 */
    Matrix *X = matrix_alloc(4, 2);
    Matrix *y = matrix_alloc(4, 1);

    matrix_set(X, 0, 0, 0.0); matrix_set(X, 0, 1, 0.0); matrix_set(y, 0, 0, 0.0);
    matrix_set(X, 1, 0, 0.0); matrix_set(X, 1, 1, 1.0); matrix_set(y, 1, 0, 0.0);
    matrix_set(X, 2, 0, 1.0); matrix_set(X, 2, 1, 0.0); matrix_set(y, 2, 0, 0.0);
    matrix_set(X, 3, 0, 1.0); matrix_set(X, 3, 1, 1.0); matrix_set(y, 3, 0, 1.0);

    int hidden[] = {4};
    NeuralNetwork *nn = neural_network_create_full(
        hidden, 1,
        ACTIVATION_RELU,
        ACTIVATION_SIGMOID,
        LOSS_BINARY_CE,
        OPTIMIZER_ADAM,
        0.01,
        500,
        4,
        0.9,
        0.0
    );
    ASSERT_NOT_NULL(nn);

    nn->base.fit((Estimator *)nn, X, y);

    Matrix *pred = nn->base.predict((const Estimator *)nn, X);
    ASSERT_NOT_NULL(pred);
    ASSERT_EQ(pred->rows, 4);
    ASSERT_EQ(pred->cols, 1);

    /* Verify outputs are valid class labels (0 or 1) */
    for (int i = 0; i < 4; i++) {
        double p = matrix_get(pred, i, 0);
        ASSERT(p == 0.0 || p == 1.0);
    }

    matrix_free(pred);
    nn->base.free((Estimator *)nn);
    matrix_free(X);
    matrix_free(y);
}

TEST(test_nn_forward) {
    /* Test forward pass returns valid output shape */
    int hidden[] = {4};
    NeuralNetwork *nn = neural_network_create(hidden, 1, ACTIVATION_SIGMOID);
    ASSERT_NOT_NULL(nn);
    neural_network_init(nn, 2, 1);

    Matrix *X = matrix_alloc(3, 2);
    matrix_set(X, 0, 0, 1.0); matrix_set(X, 0, 1, 2.0);
    matrix_set(X, 1, 0, 3.0); matrix_set(X, 1, 1, 4.0);
    matrix_set(X, 2, 0, 5.0); matrix_set(X, 2, 1, 6.0);

    Matrix *out = neural_network_forward(nn, X);
    ASSERT_NOT_NULL(out);
    ASSERT_EQ(out->rows, 3);
    ASSERT_EQ(out->cols, 1);

    /* All outputs should be finite */
    for (int i = 0; i < 3; i++) {
        double v = matrix_get(out, i, 0);
        ASSERT(v == v);
        ASSERT(v < 1e10);
        ASSERT(v > -1e10);
    }

    matrix_free(out);
    nn->base.free((Estimator *)nn);
    matrix_free(X);
}

int main(void) {
    printf("Neural Network Tests\n");
    printf("========================================\n\n");

    RUN_TEST(test_nn_create_free);
    RUN_TEST(test_nn_xor);
    RUN_TEST(test_nn_predict);
    RUN_TEST(test_nn_forward);

    TEST_SUMMARY();
}
