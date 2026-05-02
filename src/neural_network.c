/**
 * neural_network.c - Neural Network implementation
 *
 * Implements feedforward neural network with backpropagation
 */

#include "neural_network.h"
#include "metrics.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================
 * Activation Functions
 * ============================================ */

static double sigmoid(double x) {
    if (x > 500) return 1.0;
    if (x < -500) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

Matrix* activation_forward(const Matrix *z, ActivationType type) {
    Matrix *a = matrix_alloc(z->rows, z->cols);
    if (!a) return NULL;

    switch (type) {
        case ACTIVATION_IDENTITY:
            memcpy(a->data, z->data, z->rows * z->cols * sizeof(double));
            break;

        case ACTIVATION_SIGMOID:
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                a->data[i] = sigmoid(z->data[i]);
            }
            break;

        case ACTIVATION_TANH:
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                a->data[i] = tanh(z->data[i]);
            }
            break;

        case ACTIVATION_RELU:
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                a->data[i] = z->data[i] > 0 ? z->data[i] : 0;
            }
            break;

        case ACTIVATION_LEAKY_RELU:
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                a->data[i] = z->data[i] > 0 ? z->data[i] : 0.01 * z->data[i];
            }
            break;

        case ACTIVATION_SOFTMAX:
            // Softmax per row
            for (size_t i = 0; i < z->rows; i++) {
                // Find max for numerical stability
                double max_val = z->data[i * z->cols];
                for (size_t j = 1; j < z->cols; j++) {
                    if (z->data[i * z->cols + j] > max_val) {
                        max_val = z->data[i * z->cols + j];
                    }
                }

                // Compute exp and sum
                double sum = 0.0;
                for (size_t j = 0; j < z->cols; j++) {
                    a->data[i * z->cols + j] = exp(z->data[i * z->cols + j] - max_val);
                    sum += a->data[i * z->cols + j];
                }

                // Normalize
                for (size_t j = 0; j < z->cols; j++) {
                    a->data[i * z->cols + j] /= sum;
                }
            }
            break;
    }

    return a;
}

Matrix* activation_derivative(const Matrix *z, const Matrix *a, ActivationType type) {
    Matrix *d = matrix_alloc(z->rows, z->cols);
    if (!d) return NULL;

    switch (type) {
        case ACTIVATION_IDENTITY:
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                d->data[i] = 1.0;
            }
            break;

        case ACTIVATION_SIGMOID:
            // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                d->data[i] = a->data[i] * (1.0 - a->data[i]);
            }
            break;

        case ACTIVATION_TANH:
            // tanh'(x) = 1 - tanh²(x)
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                d->data[i] = 1.0 - a->data[i] * a->data[i];
            }
            break;

        case ACTIVATION_RELU:
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                d->data[i] = z->data[i] > 0 ? 1.0 : 0.0;
            }
            break;

        case ACTIVATION_LEAKY_RELU:
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                d->data[i] = z->data[i] > 0 ? 1.0 : 0.01;
            }
            break;

        case ACTIVATION_SOFTMAX:
            // For softmax with cross-entropy, derivative is handled specially
            for (size_t i = 0; i < z->rows * z->cols; i++) {
                d->data[i] = 1.0;  // Will be handled in loss
            }
            break;
    }

    return d;
}

/* ============================================
 * Layer Management
 * ============================================ */

Layer* layer_create(int n_input, int n_output, ActivationType activation) {
    Layer *layer = calloc(1, sizeof(Layer));
    if (!layer) return NULL;

    layer->weights = matrix_alloc(n_input, n_output);
    layer->biases = matrix_alloc(1, n_output);
    layer->d_weights = matrix_alloc(n_input, n_output);
    layer->d_biases = matrix_alloc(1, n_output);
    layer->activation = activation;

    if (!layer->weights || !layer->biases || !layer->d_weights || !layer->d_biases) {
        layer_free(layer);
        return NULL;
    }

    // Xavier initialization
    double std = sqrt(2.0 / (n_input + n_output));
    for (size_t i = 0; i < layer->weights->rows * layer->weights->cols; i++) {
        layer->weights->data[i] = rand_normal() * std;
    }

    // Initialize biases to small values
    for (size_t i = 0; i < layer->biases->cols; i++) {
        layer->biases->data[i] = 0.01;
    }

    return layer;
}

void layer_free(Layer *layer) {
    if (!layer) return;
    matrix_free(layer->weights);
    matrix_free(layer->biases);
    matrix_free(layer->d_weights);
    matrix_free(layer->d_biases);
    matrix_free(layer->v_weights);
    matrix_free(layer->v_biases);
    matrix_free(layer->m_weights);
    matrix_free(layer->m_biases);
    matrix_free(layer->input_cache);
    matrix_free(layer->output_cache);
    matrix_free(layer->z_cache);
    free(layer);
}

/* ============================================
 * Neural Network Implementation
 * ============================================ */

NeuralNetwork* neural_network_create(
    const int *hidden_layer_sizes,
    int n_hidden,
    ActivationType activation
) {
    return neural_network_create_full(
        hidden_layer_sizes, n_hidden,
        activation, ACTIVATION_IDENTITY,  // Output will be set in init
        LOSS_MSE, OPTIMIZER_SGD,
        0.001, 200, 32, 0.9, 0.0001
    );
}

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
) {
    NeuralNetwork *nn = calloc(1, sizeof(NeuralNetwork));
    if (!nn) return NULL;

    // Store hidden layer sizes
    nn->layer_sizes = malloc((n_hidden + 2) * sizeof(int));  // +2 for input/output
    if (!nn->layer_sizes) {
        free(nn);
        return NULL;
    }
    for (int i = 0; i < n_hidden; i++) {
        nn->layer_sizes[i + 1] = hidden_layer_sizes[i];
    }
    nn->n_layer_sizes = n_hidden + 2;

    // Setup Estimator base
    nn->base.type = MODEL_NEURAL_NETWORK;
    nn->base.task = TASK_CLASSIFICATION;  // Will be set based on output
    nn->base.is_fitted = 0;
    nn->base.verbose = VERBOSE_SILENT;

    nn->base.fit = neural_network_fit;
    nn->base.predict = neural_network_predict;
    nn->base.predict_proba = neural_network_predict_proba;
    nn->base.transform = NULL;
    nn->base.score = neural_network_score;
    nn->base.clone = neural_network_clone;
    nn->base.free = neural_network_free;
    nn->base.save = NULL;
    nn->base.load = NULL;
    nn->base.print_summary = neural_network_print_summary;

    // Hyperparameters
    nn->learning_rate = learning_rate;
    nn->max_epochs = max_epochs;
    nn->batch_size = batch_size;
    nn->momentum = momentum;
    nn->beta1 = 0.9;
    nn->beta2 = 0.999;
    nn->epsilon = 1e-8;
    nn->l2_reg = l2_reg;
    nn->dropout_rate = 0.0;

    nn->loss_type = loss;
    nn->optimizer = optimizer;
    nn->hidden_activation = hidden_activation;
    nn->output_activation = output_activation;

    nn->layers = NULL;
    nn->n_layers = 0;
    nn->epoch = 0;
    nn->t = 0;

    return nn;
}

void neural_network_init(NeuralNetwork *nn, int n_features, int n_outputs) {
    if (!nn) return;

    nn->n_features = n_features;
    nn->n_classes = n_outputs;

    // Set layer sizes
    nn->layer_sizes[0] = n_features;
    nn->layer_sizes[nn->n_layer_sizes - 1] = n_outputs;

    // Create layers
    nn->n_layers = nn->n_layer_sizes - 1;
    nn->layers = malloc(nn->n_layers * sizeof(Layer*));
    if (!nn->layers) return;

    /* RNG seeded by caller or auto-seeded on first use */

    for (int i = 0; i < nn->n_layers; i++) {
        ActivationType act;
        if (i == nn->n_layers - 1) {
            // Output layer
            act = nn->output_activation;
        } else {
            // Hidden layers
            act = nn->hidden_activation;
        }

        nn->layers[i] = layer_create(nn->layer_sizes[i], nn->layer_sizes[i + 1], act);
        if (!nn->layers[i]) return;

        // Initialize momentum/Adam if needed
        if (nn->optimizer == OPTIMIZER_MOMENTUM || nn->optimizer == OPTIMIZER_ADAM) {
            nn->layers[i]->v_weights = matrix_alloc(nn->layer_sizes[i], nn->layer_sizes[i + 1]);
            nn->layers[i]->v_biases = matrix_alloc(1, nn->layer_sizes[i + 1]);
            if (!nn->layers[i]->v_weights || !nn->layers[i]->v_biases) { return; }
            matrix_fill(nn->layers[i]->v_weights, 0);
            matrix_fill(nn->layers[i]->v_biases, 0);
        }
        if (nn->optimizer == OPTIMIZER_ADAM) {
            nn->layers[i]->m_weights = matrix_alloc(nn->layer_sizes[i], nn->layer_sizes[i + 1]);
            nn->layers[i]->m_biases = matrix_alloc(1, nn->layer_sizes[i + 1]);
            if (!nn->layers[i]->m_weights || !nn->layers[i]->m_biases) { return; }
            matrix_fill(nn->layers[i]->m_weights, 0);
            matrix_fill(nn->layers[i]->m_biases, 0);
        }
    }
}

Matrix* neural_network_forward(NeuralNetwork *nn, const Matrix *X) {
    Matrix *current = matrix_copy(X);
    if (!current) return NULL;

    for (int i = 0; i < nn->n_layers; i++) {
        Layer *layer = nn->layers[i];

        // Store input for backprop
        matrix_free(layer->input_cache);
        layer->input_cache = matrix_copy(current);

        // z = X @ W + b
        Matrix *z = matrix_matmul(current, layer->weights);
        if (!z) {
            matrix_free(current);
            return NULL;
        }

        // Add bias (broadcast)
        for (size_t r = 0; r < z->rows; r++) {
            for (size_t c = 0; c < z->cols; c++) {
                z->data[r * z->cols + c] += layer->biases->data[c];
            }
        }

        // Store z for backprop
        matrix_free(layer->z_cache);
        layer->z_cache = matrix_copy(z);

        // Apply activation
        Matrix *a = activation_forward(z, layer->activation);
        matrix_free(z);
        matrix_free(current);

        if (!a) return NULL;

        // Store output
        matrix_free(layer->output_cache);
        layer->output_cache = matrix_copy(a);

        current = a;
    }

    return current;
}

void neural_network_backward(NeuralNetwork *nn, const Matrix *y) {
    // Get output layer
    Layer *output_layer = nn->layers[nn->n_layers - 1];
    Matrix *output = output_layer->output_cache;

    // Compute output error (delta)
    Matrix *delta = matrix_alloc(output->rows, output->cols);

    // For softmax + cross-entropy or sigmoid + BCE: delta = output - y
    for (size_t i = 0; i < output->rows * output->cols; i++) {
        delta->data[i] = output->data[i] - y->data[i];
    }

    // Backpropagate through layers
    for (int i = nn->n_layers - 1; i >= 0; i--) {
        Layer *layer = nn->layers[i];
        size_t batch_size = layer->input_cache->rows;

        // Compute gradients
        // d_weights = input.T @ delta / batch_size
        Matrix *input_t = matrix_transpose(layer->input_cache);
        Matrix *d_w = matrix_matmul(input_t, delta);
        if (!d_w) { matrix_free(input_t); matrix_free(delta); return; }
        matrix_free(input_t);

        for (size_t j = 0; j < d_w->rows * d_w->cols; j++) {
            layer->d_weights->data[j] = d_w->data[j] / batch_size;
            // Add L2 regularization
            layer->d_weights->data[j] += nn->l2_reg * layer->weights->data[j];
        }
        matrix_free(d_w);

        // d_biases = mean(delta, axis=0)
        for (size_t j = 0; j < layer->biases->cols; j++) {
            double sum = 0;
            for (size_t r = 0; r < batch_size; r++) {
                sum += delta->data[r * delta->cols + j];
            }
            layer->d_biases->data[j] = sum / batch_size;
        }

        // Propagate delta to previous layer (if not input layer)
        if (i > 0) {
            Layer *prev_layer = nn->layers[i - 1];

            // delta_prev = delta @ W.T * activation'(z)
            Matrix *w_t = matrix_transpose(layer->weights);
            Matrix *delta_back = matrix_matmul(delta, w_t);
            if (!delta_back) { matrix_free(w_t); matrix_free(delta); return; }
            matrix_free(w_t);
            matrix_free(delta);

            // Multiply by activation derivative
            Matrix *act_deriv = activation_derivative(prev_layer->z_cache, prev_layer->output_cache, prev_layer->activation);
            for (size_t j = 0; j < delta_back->rows * delta_back->cols; j++) {
                delta_back->data[j] *= act_deriv->data[j];
            }
            matrix_free(act_deriv);

            delta = delta_back;
        }
    }

    matrix_free(delta);
}

void neural_network_update_weights(NeuralNetwork *nn) {
    nn->t++;  // Increment time step

    for (int i = 0; i < nn->n_layers; i++) {
        Layer *layer = nn->layers[i];
        size_t n_weights = layer->weights->rows * layer->weights->cols;
        size_t n_biases = layer->biases->cols;

        if (nn->optimizer == OPTIMIZER_SGD) {
            // Standard SGD
            for (size_t j = 0; j < n_weights; j++) {
                layer->weights->data[j] -= nn->learning_rate * layer->d_weights->data[j];
            }
            for (size_t j = 0; j < n_biases; j++) {
                layer->biases->data[j] -= nn->learning_rate * layer->d_biases->data[j];
            }
        } else if (nn->optimizer == OPTIMIZER_MOMENTUM) {
            // SGD with momentum
            for (size_t j = 0; j < n_weights; j++) {
                layer->v_weights->data[j] = nn->momentum * layer->v_weights->data[j] - nn->learning_rate * layer->d_weights->data[j];
                layer->weights->data[j] += layer->v_weights->data[j];
            }
            for (size_t j = 0; j < n_biases; j++) {
                layer->v_biases->data[j] = nn->momentum * layer->v_biases->data[j] - nn->learning_rate * layer->d_biases->data[j];
                layer->biases->data[j] += layer->v_biases->data[j];
            }
        } else if (nn->optimizer == OPTIMIZER_ADAM) {
            // Adam optimizer
            double bias_correction1 = 1.0 - pow(nn->beta1, nn->t);
            double bias_correction2 = 1.0 - pow(nn->beta2, nn->t);

            for (size_t j = 0; j < n_weights; j++) {
                layer->m_weights->data[j] = nn->beta1 * layer->m_weights->data[j] + (1 - nn->beta1) * layer->d_weights->data[j];
                layer->v_weights->data[j] = nn->beta2 * layer->v_weights->data[j] + (1 - nn->beta2) * layer->d_weights->data[j] * layer->d_weights->data[j];

                double m_hat = layer->m_weights->data[j] / bias_correction1;
                double v_hat = layer->v_weights->data[j] / bias_correction2;

                layer->weights->data[j] -= nn->learning_rate * m_hat / (sqrt(v_hat) + nn->epsilon);
            }
            for (size_t j = 0; j < n_biases; j++) {
                layer->m_biases->data[j] = nn->beta1 * layer->m_biases->data[j] + (1 - nn->beta1) * layer->d_biases->data[j];
                layer->v_biases->data[j] = nn->beta2 * layer->v_biases->data[j] + (1 - nn->beta2) * layer->d_biases->data[j] * layer->d_biases->data[j];

                double m_hat = layer->m_biases->data[j] / bias_correction1;
                double v_hat = layer->v_biases->data[j] / bias_correction2;

                layer->biases->data[j] -= nn->learning_rate * m_hat / (sqrt(v_hat) + nn->epsilon);
            }
        }
    }
}

Estimator* neural_network_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    NeuralNetwork *nn = (NeuralNetwork*)self;

    // Determine output dimension and type
    int n_outputs = y->cols;
    if (n_outputs == 1) {
        // Check if classification or regression
        int is_classification = 1;
        for (size_t i = 0; i < y->rows; i++) {
            double val = y->data[i];
            if (val != floor(val) || val < 0) {
                is_classification = 0;
                break;
            }
        }

        if (is_classification) {
            // Find max class and one-hot encode
            int max_class = 0;
            for (size_t i = 0; i < y->rows; i++) {
                int label = (int)y->data[i];
                if (label > max_class) max_class = label;
            }
            n_outputs = max_class + 1;
            nn->base.task = TASK_CLASSIFICATION;
            nn->output_activation = (n_outputs == 2) ? ACTIVATION_SIGMOID : ACTIVATION_SOFTMAX;
            nn->loss_type = (n_outputs == 2) ? LOSS_BINARY_CE : LOSS_CROSS_ENTROPY;
        } else {
            nn->base.task = TASK_REGRESSION;
            nn->output_activation = ACTIVATION_IDENTITY;
            nn->loss_type = LOSS_MSE;
        }
    }

    // Initialize network if not done
    if (!nn->layers) {
        neural_network_init(nn, X->cols, n_outputs);
    }

    // Prepare target (one-hot encoding for classification)
    Matrix *y_train = NULL;
    if (nn->base.task == TASK_CLASSIFICATION && y->cols == 1) {
        y_train = matrix_alloc(y->rows, n_outputs);
        for (size_t i = 0; i < y->rows; i++) {
            int label = (int)y->data[i];
            for (int c = 0; c < n_outputs; c++) {
                y_train->data[i * n_outputs + c] = (c == label) ? 1.0 : 0.0;
            }
        }
    } else {
        y_train = matrix_copy(y);
    }

    // Training history
    if (nn->base.verbose != VERBOSE_SILENT) {
        if (nn->base.history) training_history_free(nn->base.history);
        nn->base.history = training_history_alloc(nn->max_epochs);
    }

    // Training loop
    clock_t start = clock();
    size_t n_samples = X->rows;
    size_t n_batches = (n_samples + nn->batch_size - 1) / nn->batch_size;

    for (nn->epoch = 0; nn->epoch < nn->max_epochs; nn->epoch++) {
        double epoch_loss = 0.0;

        // Shuffle indices
        size_t *indices = malloc(n_samples * sizeof(size_t));
        for (size_t i = 0; i < n_samples; i++) indices[i] = i;
        shuffle_indices(indices, n_samples);

        // Mini-batch training
        for (size_t batch = 0; batch < n_batches; batch++) {
            size_t batch_start = batch * nn->batch_size;
            size_t batch_end = batch_start + nn->batch_size;
            if (batch_end > n_samples) batch_end = n_samples;
            size_t actual_batch_size = batch_end - batch_start;

            // Get batch
            Matrix *X_batch = matrix_alloc(actual_batch_size, X->cols);
            Matrix *y_batch = matrix_alloc(actual_batch_size, y_train->cols);

            for (size_t i = 0; i < actual_batch_size; i++) {
                size_t idx = indices[batch_start + i];
                for (size_t j = 0; j < X->cols; j++) {
                    X_batch->data[i * X->cols + j] = X->data[idx * X->cols + j];
                }
                for (size_t j = 0; j < y_train->cols; j++) {
                    y_batch->data[i * y_train->cols + j] = y_train->data[idx * y_train->cols + j];
                }
            }

            // Forward
            Matrix *output = neural_network_forward(nn, X_batch);

            // Compute loss
            double batch_loss = 0.0;
            if (nn->loss_type == LOSS_MSE) {
                for (size_t i = 0; i < output->rows * output->cols; i++) {
                    double diff = output->data[i] - y_batch->data[i];
                    batch_loss += diff * diff;
                }
                batch_loss /= (output->rows * output->cols);
            } else {
                // Cross-entropy
                for (size_t i = 0; i < output->rows * output->cols; i++) {
                    double p = output->data[i];
                    double t = y_batch->data[i];
                    if (p > 1e-10 && p < 1 - 1e-10) {
                        batch_loss -= t * log(p) + (1 - t) * log(1 - p);
                    }
                }
                batch_loss /= output->rows;
            }
            epoch_loss += batch_loss;

            // Backward
            neural_network_backward(nn, y_batch);

            // Update weights
            neural_network_update_weights(nn);

            matrix_free(X_batch);
            matrix_free(y_batch);
            matrix_free(output);
        }

        free(indices);
        epoch_loss /= n_batches;

        // Compute accuracy/R² for history
        double metric = 0.0;
        if (nn->base.task == TASK_CLASSIFICATION) {
            Matrix *pred = neural_network_predict(self, X);
            metric = accuracy(y, pred);
            matrix_free(pred);
        } else {
            metric = 1.0 - epoch_loss;  // Approximate R²
        }

        // Record history
        if (nn->base.history) {
            training_history_append(nn->base.history, epoch_loss, metric);
        }

        // Verbose output
        const char *metric_name = nn->base.task == TASK_CLASSIFICATION ? "accuracy" : "R²";
        verbose_print_epoch(nn->base.verbose, nn->epoch + 1, nn->max_epochs, epoch_loss, metric, metric_name);

        // Callback
        if (nn->base.callback) {
            nn->base.callback(nn->epoch + 1, epoch_loss, metric, nn->base.callback_data);
        }
    }

    matrix_free(y_train);
    nn->base.is_fitted = 1;

    // Final verbose output
    if (nn->base.verbose >= VERBOSE_MINIMAL) {
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        double final_score = neural_network_score(self, X, y);
        const char *metric_name = nn->base.task == TASK_CLASSIFICATION ? "accuracy" : "R²";
        verbose_print_final(nn->base.verbose, "NeuralNetwork", final_score, metric_name, elapsed);
    }

    return self;
}

Matrix* neural_network_predict(const Estimator *self, const Matrix *X) {
    NeuralNetwork *nn = (NeuralNetwork*)self;
    if (!nn->base.is_fitted) return NULL;

    Matrix *output = neural_network_forward(nn, X);
    if (!output) return NULL;

    if (nn->base.task == TASK_CLASSIFICATION) {
        // Convert to class labels
        Matrix *predictions = matrix_alloc(output->rows, 1);
        for (size_t i = 0; i < output->rows; i++) {
            if (output->cols == 1) {
                // Binary classification
                predictions->data[i] = output->data[i] > 0.5 ? 1.0 : 0.0;
            } else {
                // Multi-class: argmax
                int max_class = 0;
                double max_val = output->data[i * output->cols];
                for (size_t c = 1; c < output->cols; c++) {
                    if (output->data[i * output->cols + c] > max_val) {
                        max_val = output->data[i * output->cols + c];
                        max_class = c;
                    }
                }
                predictions->data[i] = max_class;
            }
        }
        matrix_free(output);
        return predictions;
    }

    return output;
}

Matrix* neural_network_predict_proba(const Estimator *self, const Matrix *X) {
    NeuralNetwork *nn = (NeuralNetwork*)self;
    if (!nn->base.is_fitted) return NULL;
    return neural_network_forward(nn, X);
}

double neural_network_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    const NeuralNetwork *nn = (const NeuralNetwork*)self;
    if (nn->base.task == TASK_CLASSIFICATION) {
        return classification_score_accuracy(self, X, y);
    }
    return regression_score_r2(self, X, y);
}

Estimator* neural_network_clone(const Estimator *self) {
    const NeuralNetwork *nn = (const NeuralNetwork*)self;

    int *hidden_sizes = malloc((nn->n_layer_sizes - 2) * sizeof(int));
    for (int i = 1; i < nn->n_layer_sizes - 1; i++) {
        hidden_sizes[i - 1] = nn->layer_sizes[i];
    }

    NeuralNetwork *clone = neural_network_create_full(
        hidden_sizes, nn->n_layer_sizes - 2,
        nn->hidden_activation, nn->output_activation,
        nn->loss_type, nn->optimizer,
        nn->learning_rate, nn->max_epochs, nn->batch_size,
        nn->momentum, nn->l2_reg
    );

    free(hidden_sizes);
    return (Estimator*)clone;
}

void neural_network_free(Estimator *self) {
    NeuralNetwork *nn = (NeuralNetwork*)self;
    if (!nn) return;

    for (int i = 0; i < nn->n_layers; i++) {
        layer_free(nn->layers[i]);
    }
    free(nn->layers);
    free(nn->layer_sizes);
    training_history_free(nn->base.history);
    free(nn);
}

void neural_network_print_summary(const Estimator *self) {
    const NeuralNetwork *nn = (const NeuralNetwork*)self;

    printf("\n=== Neural Network Summary ===\n");
    printf("Fitted: %s\n", nn->base.is_fitted ? "Yes" : "No");
    printf("Task: %s\n", nn->base.task == TASK_CLASSIFICATION ? "Classification" : "Regression");

    printf("\nArchitecture:\n");
    for (int i = 0; i < nn->n_layer_sizes; i++) {
        if (i == 0) printf("  Input:  %d neurons\n", nn->layer_sizes[i]);
        else if (i == nn->n_layer_sizes - 1) printf("  Output: %d neurons\n", nn->layer_sizes[i]);
        else printf("  Hidden: %d neurons\n", nn->layer_sizes[i]);
    }

    printf("\nHyperparameters:\n");
    printf("  Learning rate: %.6f\n", nn->learning_rate);
    printf("  Epochs: %d\n", nn->max_epochs);
    printf("  Batch size: %d\n", nn->batch_size);

    const char *optimizer_names[] = {"SGD", "Momentum", "Adam"};
    printf("  Optimizer: %s\n", optimizer_names[nn->optimizer]);

    if (nn->base.is_fitted) {
        int total_params = 0;
        for (int i = 0; i < nn->n_layers; i++) {
            total_params += nn->layers[i]->weights->rows * nn->layers[i]->weights->cols;
            total_params += nn->layers[i]->biases->cols;
        }
        printf("\nTotal parameters: %d\n", total_params);
    }
    printf("==============================\n\n");
}

/* ============================================
 * MLP Classifier/Regressor shortcuts
 * ============================================ */

MLPClassifier* mlp_classifier_create(const int *hidden_layer_sizes, int n_hidden) {
    NeuralNetwork *nn = neural_network_create_full(
        hidden_layer_sizes, n_hidden,
        ACTIVATION_RELU, ACTIVATION_SOFTMAX,
        LOSS_CROSS_ENTROPY, OPTIMIZER_ADAM,
        0.001, 200, 32, 0.9, 0.0001
    );
    nn->base.task = TASK_CLASSIFICATION;
    return nn;
}

MLPRegressor* mlp_regressor_create(const int *hidden_layer_sizes, int n_hidden) {
    NeuralNetwork *nn = neural_network_create_full(
        hidden_layer_sizes, n_hidden,
        ACTIVATION_RELU, ACTIVATION_IDENTITY,
        LOSS_MSE, OPTIMIZER_ADAM,
        0.001, 200, 32, 0.9, 0.0001
    );
    nn->base.task = TASK_REGRESSION;
    return nn;
}
