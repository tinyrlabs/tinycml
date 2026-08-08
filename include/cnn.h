/**
 * cnn.h - Convolutional Neural Network API for tinycml
 *
 * Builds a CNN as a sequence of blocks:
 *   Conv2D → ReLU → MaxPool2D → ... → Flatten → Dense → Softmax
 *
 * Uses the Estimator vtable for fit/predict/predict_proba/score.
 * Feature flag: CML_ENABLE_CNN (implies CML_ENABLE_CONV2D + CML_ENABLE_POOL2D)
 */

#ifndef CNN_H
#define CNN_H

#include "matrix.h"
#include "estimator.h"
#include "conv2d.h"
#include "pool2d.h"
#include "neural_network.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================
 * CNN Layer Types
 * ============================================ */

typedef enum {
    CNN_LAYER_CONV2D,
    CNN_LAYER_RELU,
    CNN_LAYER_MAXPOOL,
    CNN_LAYER_FLATTEN,
    CNN_LAYER_DENSE,
    CNN_LAYER_SOFTMAX
} CNNLayerType;

/* ============================================
 * CNN Block (one layer in the network)
 * ============================================ */

typedef struct CNNBlock {
    CNNLayerType type;
    void *layer;               /* Conv2D*, MaxPool2D*, Layer* (dense) */
    ActivationType activation; /* For DENSE layers */
    int output_size;           /* For DENSE: number of neurons */
} CNNBlock;

/* ============================================
 * CNN Model
 * ============================================ */

typedef struct {
    Estimator base;

    CNNBlock **blocks;
    int n_blocks;
    int block_capacity;

    /* Input shape */
    int input_h;
    int input_w;
    int input_c;

    /* Original input shape (before layer modifications) */
    int orig_input_h;
    int orig_input_w;
    int orig_input_c;

    /* Number of output classes (for classification) */
    int n_classes;

    /* Internal: dense layers for the classifier head */
    NeuralNetwork *classifier;  /* MLP head after flatten */
    int flatten_size;           /* Size after flatten */
} CNNModel;

/* ============================================
 * API
 * ============================================ */

/**
 * Create an empty CNN model.
 *
 * @param h         Input height (e.g., 32)
 * @param w         Input width (e.g., 32)
 * @param c         Input channels (e.g., 3 for RGB)
 * @param n_classes Number of output classes
 */
CNNModel* cnn_create(int h, int w, int c, int n_classes);

/** Free CNN and all layers */
void cnn_free(CNNModel *net);

/** Add a Conv2D layer */
CNNModel* cnn_add_conv2d(CNNModel *net, int out_c, int kh, int kw,
                         int stride, int pad);

/** Add a ReLU activation (in-place, no parameters) */
CNNModel* cnn_add_relu(CNNModel *net);

/** Add a MaxPool2D layer */
CNNModel* cnn_add_maxpool(CNNModel *net, int ph, int pw, int stride);

/** Add a Flatten layer (4D → 2D) */
CNNModel* cnn_add_flatten(CNNModel *net);

/** Add a Dense (fully connected) layer */
CNNModel* cnn_add_dense(CNNModel *net, int n_neurons, ActivationType act);

/** Add a Softmax output layer */
CNNModel* cnn_add_softmax(CNNModel *net);

/** Load weights from exported C arrays (call after building architecture) */
void cnn_load_weights(CNNModel *net,
                      const double **conv_weights,
                      const double **conv_biases,
                      int n_conv_layers,
                      const double **dense_weights,
                      const double **dense_biases,
                      const int *dense_sizes,
                      int n_dense_layers);

/* ============================================
 * Estimator vtable functions
 * ============================================ */

Estimator* cnn_fit(Estimator *self, const Matrix *X, const Matrix *y);
Matrix* cnn_predict(const Estimator *self, const Matrix *X);
Matrix* cnn_predict_proba(const Estimator *self, const Matrix *X);
double cnn_score(const Estimator *self, const Matrix *X, const Matrix *y);
Estimator* cnn_clone(const Estimator *self);
void cnn_free_estimator(Estimator *self);

/**
 * Run a single forward pass through the CNN.
 * Useful for inference without the Estimator overhead.
 *
 * @param net   CNN model
 * @param input Input matrix (N × C*H*W)
 * @return Output probabilities (N × n_classes), caller frees.
 */
Matrix* cnn_forward(CNNModel *net, const Matrix *input);

#ifdef __cplusplus
}
#endif

#endif /* CNN_H */
