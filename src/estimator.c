/**
 * estimator.c - Unified Estimator API implementation
 */

#include "estimator.h"
#include "metrics.h"
#include "cml_error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * Training history management
 */

TrainingHistory* training_history_alloc(size_t initial_capacity) {
    TrainingHistory *history = malloc(sizeof(TrainingHistory));
    if (!history) return NULL;

    history->loss_history = malloc(initial_capacity * sizeof(double));
    history->metric_history = malloc(initial_capacity * sizeof(double));

    if (!history->loss_history || !history->metric_history) {
        free(history->loss_history);
        free(history->metric_history);
        free(history);
        return NULL;
    }

    history->n_epochs = 0;
    history->capacity = initial_capacity;
    history->converged = 0;
    history->best_epoch = 0;

    return history;
}

void training_history_append(TrainingHistory *history, double loss, double metric) {
    if (!history) return;

    // Expand if needed
    if (history->n_epochs >= history->capacity) {
        size_t new_capacity = history->capacity * 2;
        double *new_loss = realloc(history->loss_history, new_capacity * sizeof(double));
        double *new_metric = realloc(history->metric_history, new_capacity * sizeof(double));

        if (!new_loss || !new_metric) {
            // Keep old arrays if realloc fails
            return;
        }

        history->loss_history = new_loss;
        history->metric_history = new_metric;
        history->capacity = new_capacity;
    }

    history->loss_history[history->n_epochs] = loss;
    history->metric_history[history->n_epochs] = metric;
    history->n_epochs++;
}

void training_history_free(TrainingHistory *history) {
    if (!history) return;
    free(history->loss_history);
    free(history->metric_history);
    free(history);
}

void training_history_print(const TrainingHistory *history) {
    if (!history) {
        printf("No training history available.\n");
        return;
    }

    printf("Training History (%zu epochs):\n", history->n_epochs);
    printf("%-10s %-15s %-15s\n", "Epoch", "Loss", "Metric");
    printf("----------------------------------------\n");

    // Print first 5, last 5 if too many
    size_t print_limit = 10;
    if (history->n_epochs <= print_limit) {
        for (size_t i = 0; i < history->n_epochs; i++) {
            printf("%-10zu %-15.6f %-15.6f\n",
                   i + 1, history->loss_history[i], history->metric_history[i]);
        }
    } else {
        for (size_t i = 0; i < 5; i++) {
            printf("%-10zu %-15.6f %-15.6f\n",
                   i + 1, history->loss_history[i], history->metric_history[i]);
        }
        printf("...        ...            ...\n");
        for (size_t i = history->n_epochs - 5; i < history->n_epochs; i++) {
            printf("%-10zu %-15.6f %-15.6f\n",
                   i + 1, history->loss_history[i], history->metric_history[i]);
        }
    }

    printf("----------------------------------------\n");
    if (history->converged) {
        printf("Converged at epoch %zu\n", history->best_epoch + 1);
    }
    printf("Final loss: %.6f, Final metric: %.6f\n",
           history->loss_history[history->n_epochs - 1],
           history->metric_history[history->n_epochs - 1]);
}

int training_history_save_csv(const TrainingHistory *history, const char *filename) {
    if (!history || !filename) return -1;

    FILE *f = fopen(filename, "w");
    if (!f) return -1;

    fprintf(f, "epoch,loss,metric\n");
    for (size_t i = 0; i < history->n_epochs; i++) {
        fprintf(f, "%zu,%.10f,%.10f\n",
                i + 1, history->loss_history[i], history->metric_history[i]);
    }

    fclose(f);
    return 0;
}

/**
 * Estimator utilities
 */

int estimator_check_fitted(const Estimator *est, const char *method_name) {
    if (!est) {
        cml_set_error(CML_ERROR_NULL_PTR, "NULL estimator passed to %s", method_name);
        return 0;
    }
    if (!est->is_fitted) {
        cml_set_error(CML_ERROR_NOT_FITTED, "Estimator not fitted. Call fit() before %s", method_name);
        return 0;
    }
    return 1;
}

void estimator_set_verbose(Estimator *est, VerboseLevel level) {
    if (est) est->verbose = level;
}

void estimator_set_callback(Estimator *est, TrainingCallback cb, void *user_data) {
    if (est) {
        est->callback = cb;
        est->callback_data = user_data;
    }
}

const TrainingHistory* estimator_get_history(const Estimator *est) {
    return est ? est->history : NULL;
}

Estimator* estimator_clone(const Estimator *est) {
    if (!est || !est->clone) return NULL;
    return est->clone(est);
}

/**
 * Default score implementations
 */

double regression_score_r2(const Estimator *est, const Matrix *X, const Matrix *y) {
    if (!estimator_check_fitted(est, "score")) return -1.0;
    if (!est->predict) return -1.0;

    Matrix *y_pred = est->predict(est, X);
    if (!y_pred) return -1.0;

    // R² = 1 - SS_res / SS_tot
    double ss_res = 0.0;
    double ss_tot = 0.0;
    double y_mean = 0.0;

    // Calculate mean
    for (size_t i = 0; i < y->rows; i++) {
        y_mean += matrix_get(y, i, 0);
    }
    y_mean /= y->rows;

    // Calculate R²
    for (size_t i = 0; i < y->rows; i++) {
        double y_true = matrix_get(y, i, 0);
        double y_hat = matrix_get(y_pred, i, 0);
        ss_res += (y_true - y_hat) * (y_true - y_hat);
        ss_tot += (y_true - y_mean) * (y_true - y_mean);
    }

    matrix_free(y_pred);

    if (ss_tot == 0.0) return 0.0;  // Avoid division by zero
    return 1.0 - (ss_res / ss_tot);
}

double classification_score_accuracy(const Estimator *est, const Matrix *X, const Matrix *y) {
    if (!estimator_check_fitted(est, "score")) return -1.0;
    if (!est->predict) return -1.0;

    Matrix *y_pred = est->predict(est, X);
    if (!y_pred) return -1.0;

    double acc = accuracy(y, y_pred);
    matrix_free(y_pred);

    return acc;
}

/**
 * Verbose output helpers
 */

void verbose_print_epoch(VerboseLevel level, int epoch, int total_epochs,
                         double loss, double metric, const char *metric_name) {
    if (level == VERBOSE_SILENT) return;

    if (level == VERBOSE_DETAILED) {
        printf("Epoch %d/%d - loss: %.6f - %s: %.6f\n",
               epoch, total_epochs, loss, metric_name, metric);
    } else if (level == VERBOSE_PROGRESS) {
        // Print every 10% or every 100 epochs, whichever is smaller
        int interval = total_epochs / 10;
        if (interval < 1) interval = 1;
        if (interval > 100) interval = 100;

        if (epoch % interval == 0 || epoch == total_epochs) {
            printf("Epoch %d/%d - loss: %.6f - %s: %.6f\n",
                   epoch, total_epochs, loss, metric_name, metric);
        }
    }
}

void verbose_print_final(VerboseLevel level, const char *model_name,
                         double final_metric, const char *metric_name, double elapsed_time) {
    if (level == VERBOSE_SILENT) return;

    printf("\n%s trained in %.3f seconds\n", model_name, elapsed_time);
    printf("Final %s: %.6f\n", metric_name, final_metric);
}

void verbose_print_progress_bar(int current, int total, int bar_width) {
    float progress = (float)current / total;
    int filled = (int)(bar_width * progress);

    printf("\r[");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %d%%", (int)(progress * 100));
    fflush(stdout);

    if (current == total) printf("\n");
}
