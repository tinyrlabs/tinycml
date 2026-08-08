/**
 * cnn.c - CNN model implementation for tinycml
 *
 * Forward pass through a sequence of Conv2D → ReLU → MaxPool2D → ... → Dense layers.
 */

#include "cnn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================
 * Internal helpers
 * ============================================ */

static void block_free(CNNBlock *b) {
    if (!b) return;
    switch (b->type) {
        case CNN_LAYER_CONV2D:
            conv2d_free((Conv2D*)b->layer);
            break;
        case CNN_LAYER_MAXPOOL:
            maxpool2d_free((MaxPool2D*)b->layer);
            break;
        case CNN_LAYER_DENSE:
            if (b->layer) {
                /* Dense layer stored as a simple {Matrix *w, Matrix *b} pair */
                void **ptr = (void**)b->layer;
                if (ptr[0]) matrix_free((Matrix*)ptr[0]);
                if (ptr[1]) matrix_free((Matrix*)ptr[1]);
                free(ptr);
            }
            break;
        default:
            break;
    }
    free(b);
}

static void relu_inplace(Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        if (m->data[i] < 0.0) m->data[i] = 0.0;
    }
}

static void softmax_inplace(Matrix *m) {
    /* Softmax per row */
    for (size_t r = 0; r < m->rows; r++) {
        double max_val = -1e308;
        for (size_t c = 0; c < m->cols; c++) {
            double v = m->data[r * m->cols + c];
            if (v > max_val) max_val = v;
        }
        double sum = 0.0;
        for (size_t c = 0; c < m->cols; c++) {
            m->data[r * m->cols + c] = exp(m->data[r * m->cols + c] - max_val);
            sum += m->data[r * m->cols + c];
        }
        for (size_t c = 0; c < m->cols; c++) {
            m->data[r * m->cols + c] /= sum;
        }
    }
}

/* ============================================
 * CNN Lifecycle
 * ============================================ */

CNNModel* cnn_create(int h, int w, int c, int n_classes) {
    CNNModel *net = calloc(1, sizeof(CNNModel));
    if (!net) return NULL;

    net->input_h = h;
    net->input_w = w;
    net->input_c = c;
    net->orig_input_h = h;
    net->orig_input_w = w;
    net->orig_input_c = c;
    net->n_classes = n_classes;
    net->block_capacity = 8;
    net->blocks = calloc((size_t)net->block_capacity, sizeof(CNNBlock*));
    if (!net->blocks) { free(net); return NULL; }

    /* Setup Estimator vtable */
    net->base.type = MODEL_CNN;
    net->base.task = TASK_CLASSIFICATION;
    net->base.fit = cnn_fit;
    net->base.predict = cnn_predict;
    net->base.predict_proba = cnn_predict_proba;
    net->base.score = cnn_score;
    net->base.clone = cnn_clone;
    net->base.free = cnn_free_estimator;
    net->base.save = NULL;
    net->base.load = NULL;
    net->base.print_summary = NULL;

    return net;
}

void cnn_free(CNNModel *net) {
    if (!net) return;
    for (int i = 0; i < net->n_blocks; i++) {
        block_free(net->blocks[i]);
    }
    free(net->blocks);
    if (net->classifier) {
        net->classifier->base.free((Estimator*)net->classifier);
    }
    free(net);
}

static void cnn_add_block(CNNModel *net, CNNBlock *block) {
    if (net->n_blocks >= net->block_capacity) {
        net->block_capacity *= 2;
        net->blocks = realloc(net->blocks, (size_t)net->block_capacity * sizeof(CNNBlock*));
    }
    net->blocks[net->n_blocks++] = block;
}

CNNModel* cnn_add_conv2d(CNNModel *net, int out_c, int kh, int kw,
                         int stride, int pad) {
    Conv2D *conv = conv2d_create(net->input_c, out_c, kh, kw, stride, stride, pad, pad);
    if (!conv) return net;

    /* Update input shape for next layer */
    net->input_h = conv2d_out_size(net->input_h, kh, stride, pad);
    net->input_w = conv2d_out_size(net->input_w, kw, stride, pad);
    net->input_c = out_c;

    CNNBlock *block = calloc(1, sizeof(CNNBlock));
    block->type = CNN_LAYER_CONV2D;
    block->layer = conv;
    cnn_add_block(net, block);
    return net;
}

CNNModel* cnn_add_relu(CNNModel *net) {
    CNNBlock *block = calloc(1, sizeof(CNNBlock));
    block->type = CNN_LAYER_RELU;
    cnn_add_block(net, block);
    return net;
}

CNNModel* cnn_add_maxpool(CNNModel *net, int ph, int pw, int stride) {
    MaxPool2D *pool = maxpool2d_create(ph, pw, stride, stride);
    if (!pool) return net;

    net->input_h = pool2d_out_size(net->input_h, ph, stride);
    net->input_w = pool2d_out_size(net->input_w, pw, stride);

    CNNBlock *block = calloc(1, sizeof(CNNBlock));
    block->type = CNN_LAYER_MAXPOOL;
    block->layer = pool;
    cnn_add_block(net, block);
    return net;
}

CNNModel* cnn_add_flatten(CNNModel *net) {
    net->flatten_size = net->input_c * net->input_h * net->input_w;

    CNNBlock *block = calloc(1, sizeof(CNNBlock));
    block->type = CNN_LAYER_FLATTEN;
    cnn_add_block(net, block);
    return net;
}

CNNModel* cnn_add_dense(CNNModel *net, int n_neurons, ActivationType act) {
    int input_size = net->flatten_size > 0 ? net->flatten_size : net->input_c;
    int output_size = n_neurons;

    /* Store weight+bias as pair of Matrix pointers */
    void **ptr = calloc(2, sizeof(void*));
    Matrix *w = matrix_alloc((size_t)input_size, (size_t)output_size);
    Matrix *b = matrix_alloc(1, (size_t)output_size);
    if (!w || !b) { if (w) matrix_free(w); if (b) matrix_free(b); free(ptr); return net; }

    /* Xavier init */
    double limit = sqrt(6.0 / (input_size + output_size));
    for (size_t i = 0; i < (size_t)input_size * output_size; i++) {
        w->data[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit;
    }

    ptr[0] = w;
    ptr[1] = b;

    CNNBlock *block = calloc(1, sizeof(CNNBlock));
    block->type = CNN_LAYER_DENSE;
    block->layer = ptr;
    block->activation = act;
    block->output_size = n_neurons;
    cnn_add_block(net, block);

    net->flatten_size = n_neurons;  /* Next dense layer's input */
    return net;
}

CNNModel* cnn_add_softmax(CNNModel *net) {
    CNNBlock *block = calloc(1, sizeof(CNNBlock));
    block->type = CNN_LAYER_SOFTMAX;
    cnn_add_block(net, block);
    return net;
}

/* ============================================
 * Forward Pass
 * ============================================ */

Matrix* cnn_forward(CNNModel *net, const Matrix *input) {
    int n = (int)input->rows;
    /* Track shapes dynamically, starting from the original input dimensions */
    int current_h = net->orig_input_h;
    int current_w = net->orig_input_w;
    int current_c = net->orig_input_c;

    Matrix *current = matrix_copy(input);
    if (!current) return NULL;

    for (int i = 0; i < net->n_blocks; i++) {
        CNNBlock *block = net->blocks[i];
        Matrix *next = NULL;

        switch (block->type) {
            case CNN_LAYER_CONV2D: {
                Conv2D *conv = (Conv2D*)block->layer;
                next = conv2d_forward(conv, current, n, current_h, current_w);
                current_h = conv2d_out_size(current_h, conv->kernel_h, conv->stride_h, conv->pad_h);
                current_w = conv2d_out_size(current_w, conv->kernel_w, conv->stride_w, conv->pad_w);
                current_c = conv->out_channels;
                break;
            }
            case CNN_LAYER_RELU:
                relu_inplace(current);
                next = current;
                current = NULL;  /* Don't free */
                break;
            case CNN_LAYER_MAXPOOL: {
                MaxPool2D *pool = (MaxPool2D*)block->layer;
                next = maxpool2d_forward(pool, current, n, current_c, current_h, current_w);
                current_h = pool2d_out_size(current_h, pool->pool_h, pool->stride_h);
                current_w = pool2d_out_size(current_w, pool->pool_w, pool->stride_w);
                break;
            }
            case CNN_LAYER_FLATTEN:
                /* Already flat in NCHW — no structural change needed */
                next = current;
                current = NULL;
                break;
            case CNN_LAYER_DENSE: {
                void **ptr = (void**)block->layer;
                Matrix *w = (Matrix*)ptr[0];
                Matrix *b = (Matrix*)ptr[1];
                /* current: (N × input_size), w: (input_size × output_size) */
                next = matrix_matmul(current, w);
                /* Add bias per row */
                for (int r = 0; r < n; r++) {
                    for (int oc = 0; oc < block->output_size; oc++) {
                        next->data[(size_t)r * block->output_size + oc] += b->data[oc];
                    }
                }
                if (block->activation == ACTIVATION_RELU) {
                    relu_inplace(next);
                } else if (block->activation == ACTIVATION_SIGMOID) {
                    for (size_t j = 0; j < next->rows * next->cols; j++) {
                        next->data[j] = 1.0 / (1.0 + exp(-next->data[j]));
                    }
                }
                break;
            }
            case CNN_LAYER_SOFTMAX:
                softmax_inplace(current);
                next = current;
                current = NULL;
                break;
        }

        if (current && current != next) matrix_free(current);
        current = next;
        if (!current) return NULL;
    }

    return current;
}

/* ============================================
 * Estimator vtable
 * ============================================ */

Estimator* cnn_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    (void)X; (void)y;
    /* CNN training is complex — for now, weights are loaded externally.
     * This stub allows the model to be used in inference mode.
     * In Phase 2, weights are loaded via cnn_load_weights() after Python training. */
    self->is_fitted = 1;
    return self;
}

Matrix* cnn_predict(const Estimator *self, const Matrix *X) {
    CNNModel *net = (CNNModel*)self;
    Matrix *probs = cnn_forward(net, X);
    if (!probs) return NULL;

    /* Argmax per row */
    Matrix *pred = matrix_alloc(X->rows, 1);
    if (!pred) { matrix_free(probs); return NULL; }

    for (size_t r = 0; r < X->rows; r++) {
        int best = 0;
        double best_val = probs->data[r * probs->cols];
        for (size_t c = 1; c < probs->cols; c++) {
            if (probs->data[r * probs->cols + c] > best_val) {
                best_val = probs->data[r * probs->cols + c];
                best = (int)c;
            }
        }
        pred->data[r] = (double)best;
    }

    matrix_free(probs);
    return pred;
}

Matrix* cnn_predict_proba(const Estimator *self, const Matrix *X) {
    CNNModel *net = (CNNModel*)self;
    return cnn_forward(net, X);
}

double cnn_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    Matrix *pred = cnn_predict(self, X);
    if (!pred) return 0.0;

    int correct = 0;
    for (size_t i = 0; i < y->rows; i++) {
        if (fabs(pred->data[i] - y->data[i]) < 0.5) correct++;
    }
    matrix_free(pred);
    return (double)correct / (double)y->rows;
}

Estimator* cnn_clone(const Estimator *self) {
    /* Shallow clone — weights not copied (would need deep copy of all layers) */
    CNNModel *src = (CNNModel*)self;
    CNNModel *dst = cnn_create(src->input_h, src->input_w, src->input_c, src->n_classes);
    /* Copy architecture? Not yet — this is a stub for interface compatibility */
    return &dst->base;
}

void cnn_free_estimator(Estimator *self) {
    cnn_free((CNNModel*)self);
}
