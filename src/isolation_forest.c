/**
 * isolation_forest.c - Isolation Forest for anomaly detection
 *
 * Based on Liu, Ting & Zhou (2008) "Isolation Forest".
 * Anomalies are isolated closer to the root → shorter path lengths.
 */

#include "isolation_forest.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Isolation Tree node ─────────────────────────────────────────── */

typedef struct IsoNode {
    int is_leaf;
    int split_feature;         /* Which feature to split on */
    double split_value;        /* Threshold */
    int depth;                 /* Depth of this node */
    int size;                  /* Number of samples at this leaf */
    struct IsoNode *left;
    struct IsoNode *right;
} IsoNode;

typedef struct {
    IsoNode *root;
} IsoTree;

/* ── helpers ─────────────────────────────────────────────────────── */

static unsigned int iso_rand(unsigned int *s) {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}

/* Average path length of unsuccessful search in BST */
static double avg_path_length(int n) {
    if (n <= 1) return 0.0;
    if (n == 2) return 1.0;
    double euler = 0.5772156649015329;  /* Euler-Mascheroni */
    return 2.0 * (log((double)n - 1.0) + euler) - 2.0 * (double)(n - 1) / (double)n;
}

/* ── tree building ───────────────────────────────────────────────── */

static IsoNode* build_iso_tree(const Matrix *X, const size_t *indices, size_t n,
                                int depth, int max_depth, unsigned int *seed) {
    IsoNode *node = calloc(1, sizeof(IsoNode));
    if (!node) return NULL;
    node->depth = depth;

    /* Stop conditions */
    if (depth >= max_depth || n <= 1) {
        node->is_leaf = 1;
        node->size = (int)n;
        return node;
    }

    /* Check if all samples identical */
    int all_same = 1;
    for (size_t j = 0; j < X->cols && all_same; j++) {
        double val = matrix_get(X, indices[0], j);
        for (size_t i = 1; i < n; i++) {
            if (matrix_get(X, indices[i], j) != val) {
                all_same = 0;
                break;
            }
        }
    }
    if (all_same) {
        node->is_leaf = 1;
        node->size = (int)n;
        return node;
    }

    /* Pick random feature and split value */
    int feat = (int)(iso_rand(seed) % X->cols);
    double min_val = matrix_get(X, indices[0], feat);
    double max_val = min_val;
    for (size_t i = 1; i < n; i++) {
        double v = matrix_get(X, indices[i], feat);
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    if (min_val == max_val) {
        node->is_leaf = 1;
        node->size = (int)n;
        return node;
    }

    /* Random split between min and max */
    double frac = (double)(iso_rand(seed) % 10000) / 10000.0;
    double split_val = min_val + frac * (max_val - min_val);

    node->is_leaf = 0;
    node->split_feature = feat;
    node->split_value = split_val;

    /* Partition */
    size_t *left_idx = malloc(n * sizeof(size_t));
    size_t *right_idx = malloc(n * sizeof(size_t));
    size_t nl = 0, nr = 0;

    for (size_t i = 0; i < n; i++) {
        if (matrix_get(X, indices[i], feat) < split_val) {
            left_idx[nl++] = indices[i];
        } else {
            right_idx[nr++] = indices[i];
        }
    }

    node->left = build_iso_tree(X, left_idx, nl, depth + 1, max_depth, seed);
    node->right = build_iso_tree(X, right_idx, nr, depth + 1, max_depth, seed);

    free(left_idx);
    free(right_idx);
    return node;
}

static void free_iso_tree(IsoNode *node) {
    if (!node) return;
    free_iso_tree(node->left);
    free_iso_tree(node->right);
    free(node);
}

/* ── path length computation ─────────────────────────────────────── */

static double path_length(const IsoNode *node, const Matrix *X, size_t row, int max_depth) {
    if (!node) return (double)max_depth;

    if (node->is_leaf) {
        return (double)node->depth + avg_path_length(node->size);
    }

    double val = matrix_get(X, row, node->split_feature);
    if (val < node->split_value) {
        return path_length(node->left, X, row, max_depth);
    } else {
        return path_length(node->right, X, row, max_depth);
    }
}

/* ── create / free ───────────────────────────────────────────────── */

IsolationForest* isolation_forest_create(int n_trees, int max_samples) {
    IsolationForest *ifor = calloc(1, sizeof(IsolationForest));
    if (!ifor) return NULL;

    ifor->n_trees     = n_trees > 0 ? n_trees : 100;
    ifor->max_samples = max_samples;
    ifor->max_depth   = 0;  /* auto */
    ifor->seed        = (unsigned int)time(NULL) ^ 0x1F0FE57;

    ifor->trees       = NULL;
    ifor->n_features  = 0;
    ifor->n_train     = 0;
    ifor->threshold   = NULL;

    ifor->base.fit     = isolation_forest_fit;
    ifor->base.predict = isolation_forest_predict;
    ifor->base.score   = isolation_forest_score;
    ifor->base.clone   = isolation_forest_clone;
    ifor->base.free    = isolation_forest_free;
    ifor->base.type    = MODEL_KMEANS;  /* closest: unsupervised */

    return ifor;
}

void isolation_forest_free(Estimator *self) {
    if (!self) return;
    IsolationForest *ifor = (IsolationForest*)self;
    if (ifor->trees) {
        for (int t = 0; t < ifor->n_trees; t++) {
            IsoTree *tree = (IsoTree*)ifor->trees[t];
            if (tree) {
                free_iso_tree(tree->root);
                free(tree);
            }
        }
        free(ifor->trees);
    }
    free(ifor->threshold);
    free(ifor);
}

/* ── fit ─────────────────────────────────────────────────────────── */

Estimator* isolation_forest_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    (void)y;  /* Unsupervised */
    if (!self || !X) return NULL;

    IsolationForest *ifor = (IsolationForest*)self;
    size_t n = X->rows;
    ifor->n_features = (int)X->cols;
    ifor->n_train = n;

    int sample_size = ifor->max_samples > 0 ? ifor->max_samples : (n < 256 ? (int)n : 256);
    if (sample_size > (int)n) sample_size = (int)n;

    int depth = ifor->max_depth > 0 ? ifor->max_depth : (int)ceil(log2(sample_size + 1));

    ifor->trees = calloc(ifor->n_trees, sizeof(IsoTree*));
    if (!ifor->trees) return NULL;

    size_t *indices = malloc(sample_size * sizeof(size_t));

    for (int t = 0; t < ifor->n_trees; t++) {
        /* Subsample */
        if ((size_t)sample_size >= n) {
            for (int i = 0; i < sample_size; i++) indices[i] = (size_t)i;
        } else {
            for (int i = 0; i < sample_size; i++) {
                indices[i] = iso_rand(&ifor->seed) % n;
            }
        }

        IsoTree *tree = calloc(1, sizeof(IsoTree));
        if (!tree) continue;

        tree->root = build_iso_tree(X, indices, sample_size, 0, depth, &ifor->seed);
        ifor->trees[t] = tree;
    }

    free(indices);

    /* Compute threshold from training data (5th percentile score → threshold) */
    Matrix *scores = isolation_forest_score_samples(self, X);
    if (scores && scores->rows > 0) {
        /* Sort scores to find 5th percentile */
        double *sarr = malloc(scores->rows * sizeof(double));
        for (size_t i = 0; i < scores->rows; i++) sarr[i] = scores->data[i];
        /* Simple insertion sort (fine for threshold computation) */
        for (size_t i = 1; i < scores->rows; i++) {
            double key = sarr[i];
            size_t j = i;
            while (j > 0 && sarr[j - 1] > key) {
                sarr[j] = sarr[j - 1];
                j--;
            }
            sarr[j] = key;
        }
        size_t idx = (size_t)(0.05 * scores->rows);
        if (idx >= scores->rows) idx = scores->rows - 1;

        ifor->threshold = malloc(sizeof(double));
        if (ifor->threshold) *ifor->threshold = sarr[idx];
        free(sarr);
        matrix_free(scores);
    }

    return self;
}

/* ── score_samples ───────────────────────────────────────────────── */

Matrix* isolation_forest_score_samples(const Estimator *self, const Matrix *X) {
    if (!self || !X) return NULL;
    IsolationForest *ifor = (IsolationForest*)self;

    size_t n = X->rows;
    Matrix *out = matrix_alloc(n, 1);
    if (!out) return NULL;

    int sample_size = ifor->max_samples > 0 ? ifor->max_samples : (ifor->n_train < 256 ? (int)ifor->n_train : 256);
    double c_n = avg_path_length(sample_size);

    for (size_t i = 0; i < n; i++) {
        double avg_path = 0.0;
        int valid_trees = 0;

        for (int t = 0; t < ifor->n_trees; t++) {
            IsoTree *tree = (IsoTree*)ifor->trees[t];
            if (tree && tree->root) {
                avg_path += path_length(tree->root, X, i, ifor->max_depth > 0 ? ifor->max_depth : 50);
                valid_trees++;
            }
        }

        if (valid_trees > 0) avg_path /= (double)valid_trees;

        /* Anomaly score: 2^(-avg_path / c(n)) */
        double score;
        if (c_n > 0) {
            score = pow(2.0, -avg_path / c_n);
        } else {
            score = 0.5;
        }

        out->data[i] = score;
    }

    return out;
}

/* ── predict ─────────────────────────────────────────────────────── */

Matrix* isolation_forest_predict(const Estimator *self, const Matrix *X) {
    if (!self || !X) return NULL;
    IsolationForest *ifor = (IsolationForest*)self;

    Matrix *scores = isolation_forest_score_samples(self, X);
    if (!scores) return NULL;

    size_t n = X->rows;
    Matrix *out = matrix_alloc(n, 1);
    if (!out) { matrix_free(scores); return NULL; }

    double thresh = ifor->threshold ? *ifor->threshold : 0.5;

    for (size_t i = 0; i < n; i++) {
        /* score > threshold → anomaly */
        out->data[i] = (scores->data[i] > thresh) ? -1.0 : 1.0;
    }

    matrix_free(scores);
    return out;
}

/* ── score ───────────────────────────────────────────────────────── */

double isolation_forest_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    if (!self || !X || !y) return -1.0;

    Matrix *pred = isolation_forest_predict(self, X);
    if (!pred) return -1.0;

    size_t correct = 0;
    for (size_t i = 0; i < y->rows; i++) {
        int p = (int)pred->data[i];
        int t = (int)y->data[i];
        if (t == -1 || t == 1) {
            if (p == t) correct++;
        }
    }

    matrix_free(pred);

    size_t labeled = 0;
    for (size_t i = 0; i < y->rows; i++) {
        int t = (int)y->data[i];
        if (t == -1 || t == 1) labeled++;
    }

    return labeled > 0 ? (double)correct / (double)labeled : 0.0;
}

/* ── clone ───────────────────────────────────────────────────────── */

Estimator* isolation_forest_clone(const Estimator *self) {
    (void)self;
    /* Simplified: return NULL — clone not critical for IF */
    return NULL;
}
