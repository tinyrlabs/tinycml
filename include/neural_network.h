/**
 * neural_network.h - Feedforward Neural Network (Multi-Layer Perceptron)
 *
 * Features:
 * - Configurable hidden layers and neurons
 * - Multiple activation functions (ReLU, Sigmoid, Tanh, Softmax)
 * - Backpropagation with gradient descent
 * - Mini-batch training
 * - Dropout regularization (optional)
 */

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "matrix.h"
#include "estimator.h"

/**
 * Activation functions
 */
typedef enum {
    ACTIVATION_IDENTITY,    // f(x) = x
    ACTIVATION_SIGMOID,     // f(x) = 1 / (1 + exp(-x))
    ACTIVATION_TANH,        // f(x) = tanh(x)
    ACTIVATION_RELU,        // f(x) = max(0, x)
    ACTIVATION_LEAKY_RELU,  // f(x) = x if x > 0 else 0.01x
    ACTIVATION_SOFTMAX      // f(x_i) = exp(x_i) / sum(exp(x_j))
} ActivationType;

/**
 * Loss functions
 */
typedef enum {
    LOSS_MSE,              // Mean Squared Error (regression)
    LOSS_CROSS_ENTROPY,    // Cross-entropy (classification)
    LOSS_BINARY_CE         // Binary cross-entropy
} LossType;

/**
 * Optimizer types
 */
typedef enum {
    OPTIMIZER_SGD,         // Stochastic Gradient Descent
    OPTIMIZER_MOMENTUM,    // SGD with momentum
    OPTIMIZER_ADAM         // Adaptive Moment Estimation
} OptimizerType;

/**
 * Layer structure
 */
typedef struct {
    Matrix *weights;        // (n_input x n_output)
    Matrix *biases;         // (1 x n_output)
    ActivationType activation;

    // Gradients
    Matrix *d_weights;
    Matrix *d_biases;

    // For momentum/Adam
    Matrix *v_weights;      // Velocity for weights
    Matrix *v_biases;       // Velocity for biases
    Matrix *m_weights;      // First moment (Adam)
    Matrix *m_biases;       // First moment (Adam)

    // Cache for backprop
    Matrix *input_cache;    // Input to this layer
    Matrix *output_cache;   // Output after activation
    Matrix *z_cache;        // Pre-activation values
} Layer;

/**
 * Neural Network structure
 */
typedef struct {
    Estimator base;

    Layer **layers;
    int n_layers;

    // Architecture
    int *layer_sizes;       // Array of layer sizes
    int n_layer_sizes;

    // Hyperparameters
    double learning_rate;
    int max_epochs;
    int batch_size;
    double momentum;
    double beta1;           // Adam first moment decay
    double beta2;           // Adam second moment decay
    double epsilon;         // Adam epsilon
    double l2_reg;          // L2 regularization
    double dropout_rate;    // Dropout probability

    LossType loss_type;
    OptimizerType optimizer;
    ActivationType hidden_activation;
    ActivationType output_activation;

    // Training state
    int epoch;
    int t;                  // Time step for Adam

    // Classification info
    int n_classes;
    int n_features;
} NeuralNetwork;

/**
 * MLP Classifier (wrapper)
 */
typedef NeuralNetwork MLPClassifier;

/**
 * MLP Regressor (wrapper)
 */
typedef NeuralNetwork MLPRegressor;

/* ============================================
 * Neural Network API
 * ============================================ */

/**
 * Create a simple neural network
 *
 * @param hidden_layer_sizes Array of hidden layer sizes (e.g., [64, 32])
 * @param n_hidden Number of hidden layers
 * @param activation Activation for hidden layers
 * @return Neural network
 */
NeuralNetwork* neural_network_create(
    const int *hidden_layer_sizes,
    int n_hidden,
    ActivationType activation
);

/**
 * Create with full configuration
 */
NeuralNetwork* neural_network_create_full(
    const int *hidden_layer_sizes,
    int n_hidden,
    ActivationType hidden_activation,
    ActivationType output_activation,
    LossType loss,
    OptimizerType optimizer,
    double learning_rate,
    int max_epochs,
    int batch_size,
    double momentum,
    double l2_reg
);

/**
 * Initialize network with input/output dimensions
 */
void neural_network_init(NeuralNetwork *nn, int n_features, int n_outputs);

/**
 * Fit the network
 */
Estimator* neural_network_fit(Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Forward pass
 */
Matrix* neural_network_forward(NeuralNetwork *nn, const Matrix *X);

/**
 * Backward pass (compute gradients)
 */
void neural_network_backward(NeuralNetwork *nn, const Matrix *y);

/**
 * Update weights using gradients
 */
void neural_network_update_weights(NeuralNetwork *nn);

/**
 * Predict
 */
Matrix* neural_network_predict(const Estimator *self, const Matrix *X);

/**
 * Predict probabilities (for classification)
 */
Matrix* neural_network_predict_proba(const Estimator *self, const Matrix *X);

/**
 * Score
 */
double neural_network_score(const Estimator *self, const Matrix *X, const Matrix *y);

/**
 * Clone
 */
Estimator* neural_network_clone(const Estimator *self);

/**
 * Free
 */
void neural_network_free(Estimator *self);

/**
 * Print network summary
 */
void neural_network_print_summary(const Estimator *self);

/* ============================================
 * MLP Classifier/Regressor API
 * ============================================ */

/**
 * Create MLP Classifier
 */
MLPClassifier* mlp_classifier_create(const int *hidden_layer_sizes, int n_hidden);

/**
 * Create MLP Regressor
 */
MLPRegressor* mlp_regressor_create(const int *hidden_layer_sizes, int n_hidden);

/* ============================================
 * Activation function utilities
 * ============================================ */

/**
 * Apply activation function element-wise
 */
Matrix* activation_forward(const Matrix *z, ActivationType type);

/**
 * Compute activation derivative
 */
Matrix* activation_derivative(const Matrix *z, const Matrix *a, ActivationType type);

/* ============================================
 * Layer utilities
 * ============================================ */

/**
 * Create a layer
 */
Layer* layer_create(int n_input, int n_output, ActivationType activation);

/**
 * Free a layer
 */
void layer_free(Layer *layer);

#endif /* NEURAL_NETWORK_H */
