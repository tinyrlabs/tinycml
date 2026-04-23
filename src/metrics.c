/**
 * @file metrics.c
 * @brief Implementation of evaluation metrics
 */

#include "metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double mse(const Matrix *y_true, const Matrix *y_pred) {
    if (!y_true || !y_pred) {
        return 0.0;
    }

    size_t n = y_true->rows * y_true->cols;
    if (n != y_pred->rows * y_pred->cols || n == 0) {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = y_true->data[i] - y_pred->data[i];
        sum += diff * diff;
    }

    return sum / (double)n;
}

double rmse(const Matrix *y_true, const Matrix *y_pred) {
    return sqrt(mse(y_true, y_pred));
}

double mae(const Matrix *y_true, const Matrix *y_pred) {
    if (!y_true || !y_pred) {
        return 0.0;
    }

    size_t n = y_true->rows * y_true->cols;
    if (n != y_pred->rows * y_pred->cols || n == 0) {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += fabs(y_true->data[i] - y_pred->data[i]);
    }

    return sum / (double)n;
}

double accuracy(const Matrix *y_true, const Matrix *y_pred) {
    if (!y_true || !y_pred) {
        return 0.0;
    }

    size_t n = y_true->rows * y_true->cols;
    if (n != y_pred->rows * y_pred->cols || n == 0) {
        return 0.0;
    }

    int correct = 0;
    for (size_t i = 0; i < n; i++) {
        /* Compare as integers for classification */
        if ((int)y_true->data[i] == (int)y_pred->data[i]) {
            correct++;
        }
    }

    return (double)correct / (double)n;
}

ConfusionMatrix confusion_matrix(const Matrix *y_true, const Matrix *y_pred) {
    ConfusionMatrix cm = {0, 0, 0, 0};

    if (!y_true || !y_pred) {
        return cm;
    }

    size_t n = y_true->rows * y_true->cols;
    if (n != y_pred->rows * y_pred->cols) {
        return cm;
    }

    for (size_t i = 0; i < n; i++) {
        int actual = (int)y_true->data[i];
        int predicted = (int)y_pred->data[i];

        if (actual == 1 && predicted == 1) {
            cm.tp++;
        } else if (actual == 0 && predicted == 0) {
            cm.tn++;
        } else if (actual == 0 && predicted == 1) {
            cm.fp++;
        } else if (actual == 1 && predicted == 0) {
            cm.fn++;
        }
    }

    return cm;
}

double precision(const Matrix *y_true, const Matrix *y_pred) {
    ConfusionMatrix cm = confusion_matrix(y_true, y_pred);
    int denominator = cm.tp + cm.fp;
    if (denominator == 0) {
        return 0.0;
    }
    return (double)cm.tp / (double)denominator;
}

double recall(const Matrix *y_true, const Matrix *y_pred) {
    ConfusionMatrix cm = confusion_matrix(y_true, y_pred);
    int denominator = cm.tp + cm.fn;
    if (denominator == 0) {
        return 0.0;
    }
    return (double)cm.tp / (double)denominator;
}

double f1_score(const Matrix *y_true, const Matrix *y_pred) {
    double p = precision(y_true, y_pred);
    double r = recall(y_true, y_pred);

    if (p + r == 0.0) {
        return 0.0;
    }

    return 2.0 * (p * r) / (p + r);
}

void confusion_matrix_print(const ConfusionMatrix *cm) {
    if (!cm) {
        return;
    }

    printf("Confusion Matrix:\n");
    printf("                 Predicted\n");
    printf("              Neg      Pos\n");
    printf("Actual Neg  %5d    %5d\n", cm->tn, cm->fp);
    printf("       Pos  %5d    %5d\n", cm->fn, cm->tp);
    printf("\n");
    printf("TP: %d, TN: %d, FP: %d, FN: %d\n", cm->tp, cm->tn, cm->fp, cm->fn);
}

/* ============================================
 * Multi-class metrics
 * ============================================ */

Matrix* confusion_matrix_multi(const Matrix *y_true, const Matrix *y_pred) {
    if (!y_true || !y_pred) return NULL;

    size_t n = y_true->rows * y_true->cols;
    if (n != y_pred->rows * y_pred->cols || n == 0) return NULL;

    /* Find max label to determine n_classes */
    int max_label = 0;
    for (size_t i = 0; i < n; i++) {
        int lt = (int)y_true->data[i];
        int lp = (int)y_pred->data[i];
        if (lt > max_label) max_label = lt;
        if (lp > max_label) max_label = lp;
    }
    int nc = max_label + 1;

    Matrix *cm = matrix_alloc((size_t)nc, (size_t)nc);
    if (!cm) return NULL;

    /* Count occurrences: cm[true_class][pred_class] */
    for (size_t i = 0; i < n; i++) {
        int t = (int)y_true->data[i];
        int p = (int)y_pred->data[i];
        if (t >= 0 && t < nc && p >= 0 && p < nc) {
            cm->data[t * (size_t)nc + p] += 1.0;
        }
    }

    return cm;
}

/**
 * Helper: compute per-class precision for class c from multi-class confusion matrix.
 * Precision_c = cm[c][c] / sum over all rows of cm[?][c]  (column sum for c)
 */
static double per_class_precision(const Matrix *cm, int c) {
    int nc = (int)cm->rows;
    double tp = cm->data[c * (size_t)nc + c];
    double col_sum = 0.0;
    for (int i = 0; i < nc; i++) {
        col_sum += cm->data[i * (size_t)nc + c];
    }
    if (col_sum == 0.0) return 0.0;
    return tp / col_sum;
}

/**
 * Helper: compute per-class recall for class c from multi-class confusion matrix.
 * Recall_c = cm[c][c] / sum over all cols of cm[c][?]  (row sum for c)
 */
static double per_class_recall(const Matrix *cm, int c) {
    int nc = (int)cm->cols;
    double tp = cm->data[c * (size_t)nc + c];
    double row_sum = 0.0;
    for (int j = 0; j < nc; j++) {
        row_sum += cm->data[c * (size_t)nc + j];
    }
    if (row_sum == 0.0) return 0.0;
    return tp / row_sum;
}

double precision_macro(const Matrix *y_true, const Matrix *y_pred, int n_classes) {
    if (!y_true || !y_pred || n_classes <= 0) return 0.0;

    Matrix *cm = confusion_matrix_multi(y_true, y_pred);
    if (!cm) return 0.0;

    double sum = 0.0;
    int counted = 0;
    for (int c = 0; c < n_classes; c++) {
        /* Only include classes that appear in predictions (column sum > 0) */
        double p = per_class_precision(cm, c);
        sum += p;
        counted++;
    }

    matrix_free(cm);
    if (counted == 0) return 0.0;
    return sum / (double)counted;
}

double recall_macro(const Matrix *y_true, const Matrix *y_pred, int n_classes) {
    if (!y_true || !y_pred || n_classes <= 0) return 0.0;

    Matrix *cm = confusion_matrix_multi(y_true, y_pred);
    if (!cm) return 0.0;

    double sum = 0.0;
    int counted = 0;
    for (int c = 0; c < n_classes; c++) {
        double r = per_class_recall(cm, c);
        sum += r;
        counted++;
    }

    matrix_free(cm);
    if (counted == 0) return 0.0;
    return sum / (double)counted;
}

double f1_score_macro(const Matrix *y_true, const Matrix *y_pred, int n_classes) {
    if (!y_true || !y_pred || n_classes <= 0) return 0.0;

    Matrix *cm = confusion_matrix_multi(y_true, y_pred);
    if (!cm) return 0.0;

    double sum = 0.0;
    int counted = 0;
    for (int c = 0; c < n_classes; c++) {
        double p = per_class_precision(cm, c);
        double r = per_class_recall(cm, c);
        if (p + r > 0.0) {
            sum += 2.0 * (p * r) / (p + r);
        }
        counted++;
    }

    matrix_free(cm);
    if (counted == 0) return 0.0;
    return sum / (double)counted;
}

double f1_score_weighted(const Matrix *y_true, const Matrix *y_pred, int n_classes) {
    if (!y_true || !y_pred || n_classes <= 0) return 0.0;

    size_t n = y_true->rows * y_true->cols;

    Matrix *cm = confusion_matrix_multi(y_true, y_pred);
    if (!cm) return 0.0;

    /* Compute support (count of each class in y_true) */
    double *support = calloc((size_t)n_classes, sizeof(double));
    if (!support) {
        matrix_free(cm);
        return 0.0;
    }
    for (size_t i = 0; i < n; i++) {
        int label = (int)y_true->data[i];
        if (label >= 0 && label < n_classes) {
            support[label] += 1.0;
        }
    }

    double weighted_sum = 0.0;
    for (int c = 0; c < n_classes; c++) {
        double p = per_class_precision(cm, c);
        double r = per_class_recall(cm, c);
        double f1 = 0.0;
        if (p + r > 0.0) {
            f1 = 2.0 * (p * r) / (p + r);
        }
        weighted_sum += f1 * support[c];
    }

    matrix_free(cm);
    free(support);

    if ((double)n == 0.0) return 0.0;
    return weighted_sum / (double)n;
}

/* ============================================
 * Clustering Metrics
 * ============================================ */

double silhouette_score(const Matrix *X, const Matrix *labels, int n_clusters) {
    if (!X || !labels || n_clusters <= 1) return 0.0;

    size_t n = X->rows;
    size_t p = X->cols;
    size_t nl = labels->rows * labels->cols;
    if (n != nl || n == 0) return 0.0;

    /* Count cluster sizes */
    size_t *cluster_size = calloc((size_t)n_clusters, sizeof(size_t));
    if (!cluster_size) return 0.0;

    for (size_t i = 0; i < n; i++) {
        int c = (int)labels->data[i];
        if (c < 0 || c >= n_clusters) {
            free(cluster_size);
            return 0.0;
        }
        cluster_size[c]++;
    }

    /* Silhouette is undefined if any cluster has < 2 members (a(i)=0 division) */
    for (int c = 0; c < n_clusters; c++) {
        if (cluster_size[c] < 2) {
            free(cluster_size);
            return 0.0;
        }
    }

    double total_sil = 0.0;

    for (size_t i = 0; i < n; i++) {
        int ci = (int)labels->data[i];

        /* Compute pairwise distances from i to all other points */
        double *dist_to_cluster = calloc((size_t)n_clusters, sizeof(double));
        if (!dist_to_cluster) {
            free(cluster_size);
            return 0.0;
        }

        for (size_t j = 0; j < n; j++) {
            if (j == i) continue;
            int cj = (int)labels->data[j];

            double d = 0.0;
            for (size_t f = 0; f < p; f++) {
                double diff = X->data[i * p + f] - X->data[j * p + f];
                d += diff * diff;
            }
            dist_to_cluster[cj] += sqrt(d);
        }

        /* a(i): mean distance to same cluster (exclude self) */
        double a_i = 0.0;
        if (cluster_size[ci] > 1) {
            a_i = dist_to_cluster[ci] / (double)(cluster_size[ci] - 1);
        }

        /* b(i): min mean distance to other clusters */
        double b_i = 1e300;
        for (int c = 0; c < n_clusters; c++) {
            if (c == ci) continue;
            if (cluster_size[c] == 0) continue;
            double mean_d = dist_to_cluster[c] / (double)cluster_size[c];
            if (mean_d < b_i) b_i = mean_d;
        }

        /* s(i) */
        double s_i = 0.0;
        if (a_i > 0.0 || b_i > 0.0) {
            double denom = a_i > b_i ? a_i : b_i;
            s_i = (b_i - a_i) / denom;
        }

        total_sil += s_i;
        free(dist_to_cluster);
    }

    free(cluster_size);
    return total_sil / (double)n;
}
