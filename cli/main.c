/**
 * main.c - tinycml CLI: train, predict, info, benchmark
 *
 * Usage:
 *   tinycml train --model <type> --data <file.csv> [--target <col>] --output <model.bin>
 *   tinycml predict --model <model.bin> --input <data.csv> --output <preds.csv>
 *   tinycml info --model <model.bin>
 *   tinycml benchmark [--dataset iris|wine|boston]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include <time.h>

#include "matrix.h"
#include "csv.h"
#include "linear_regression.h"
#include "logistic_regression.h"
#include "knn.h"
#include "svm.h"
#include "naive_bayes.h"
#include "decision_tree.h"
#include "ensemble.h"
#include "neural_network.h"
#include "ridge.h"
#include "lasso.h"
#include "gradient_boosting.h"
#include "preprocessing.h"
#include "estimator.h"

/* ── helpers ─────────────────────────────────────────────────────── */

static void usage(void) {
    printf(
        "tinycml v0.1.0 - CLI for lightweight ML in C\n"
        "\n"
        "Commands:\n"
        "  train       Train a model on CSV data\n"
        "  predict     Make predictions using a saved model\n"
        "  info        Print model information\n"
        "  benchmark   Run built-in benchmark datasets\n"
        "\n"
        "Models: linreg, logreg, knn, svm, nb, dtree, rf, gb, nn, ridge, lasso, softmax\n"
        "\n"
        "Examples:\n"
        "  tinycml train --model logreg --data train.csv --output model.bin\n"
        "  tinycml train --model knn --data train.csv -p k=5 --output model.bin\n"
        "  tinycml predict --model model.bin --input test.csv --output preds.csv\n"
        "  tinycml info --model model.bin\n"
        "  tinycml benchmark --dataset iris\n"
    );
}

static char *model_type_name(ModelType t) {
    switch (t) {
        case MODEL_LINEAR_REGRESSION:   return "LinearRegression";
        case MODEL_LOGISTIC_REGRESSION: return "LogisticRegression";
        case MODEL_KNN:                 return "KNN";
        case MODEL_KMEANS:              return "KMeans";
        case MODEL_NAIVE_BAYES:         return "NaiveBayes";
        case MODEL_DECISION_TREE:       return "DecisionTree";
        case MODEL_RANDOM_FOREST:       return "RandomForest";
        case MODEL_NEURAL_NETWORK:      return "NeuralNetwork";
        case MODEL_SVM:                 return "SVM";
        case MODEL_PCA:                 return "PCA";
        case MODEL_FEATURE_SELECTOR:    return "FeatureSelector";
        case MODEL_GRADIENT_BOOSTING:   return "GradientBoosting";
        default:                        return "Unknown";
    }
}

static Matrix *csv_load_wrapper(const char *path) {
    return csv_load(path, 1);  /* has_header=1 */
}

static int write_predictions_csv(const char *path, const Matrix *preds) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Error: cannot write to %s\n", path);
        return -1;
    }
    fprintf(f, "prediction\n");
    for (size_t i = 0; i < preds->rows; i++) {
        double val = preds->data[i];
        if (val == (int)val && fabs(val) < 1e9) {
            fprintf(f, "%d\n", (int)val);
        } else {
            fprintf(f, "%.6f\n", val);
        }
    }
    fclose(f);
    return 0;
}

/* ── synthetic datasets ──────────────────────────────────────────── */

static Matrix *generate_iris_X(void) {
    /* 150 samples, 4 features — approximate iris data */
    Matrix *X = matrix_alloc(150, 4);
    unsigned int seed = 42;
    double setosa[] = {5.0, 3.4, 1.5, 0.2};
    double versicolor[] = {5.9, 2.8, 4.3, 1.3};
    double virginica[] = {6.6, 3.0, 5.6, 2.0};

    for (int i = 0; i < 50; i++) {
        for (int j = 0; j < 4; j++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            double noise = ((double)(seed % 1000) - 500.0) / 1000.0 * 0.3;
            matrix_set(X, i, j, setosa[j] + noise);
            matrix_set(X, i + 50, j, versicolor[j] + noise);
            matrix_set(X, i + 100, j, virginica[j] + noise);
        }
    }
    return X;
}

static Matrix *generate_iris_y(void) {
    Matrix *y = matrix_alloc(150, 1);
    for (int i = 0; i < 50; i++) {
        matrix_set(y, i, 0, 0.0);
        matrix_set(y, i + 50, 0, 1.0);
        matrix_set(y, i + 100, 0, 2.0);
    }
    return y;
}

static Matrix *generate_boston_X(void) {
    /* 100 samples, 3 features — synthetic regression */
    Matrix *X = matrix_alloc(100, 3);
    unsigned int seed = 42;
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 3; j++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            matrix_set(X, i, j, (double)(seed % 10000) / 1000.0);
        }
    }
    return X;
}

static Matrix *generate_boston_y(const Matrix *X) {
    /* y = 3*x0 + 2*x1 - x2 + 5 */
    Matrix *y = matrix_alloc(X->rows, 1);
    for (size_t i = 0; i < X->rows; i++) {
        double val = 3.0 * matrix_get(X, i, 0)
                   + 2.0 * matrix_get(X, i, 1)
                   - matrix_get(X, i, 2) + 5.0;
        matrix_set(y, i, 0, val);
    }
    return y;
}

/* ── model creation ──────────────────────────────────────────────── */

typedef struct {
    const char *name;
    int is_classifier;
} ModelInfo;

static const ModelInfo MODEL_TABLE[] = {
    {"linreg",   0},
    {"logreg",   1},
    {"knn",      1},
    {"svm",      1},
    {"nb",       1},
    {"dtree",    1},
    {"rf",       1},
    {"gb",       1},
    {"nn",       1},
    {"ridge",    0},
    {"lasso",    0},
    {"softmax",  1},
    {NULL, 0}
};

static int find_model(const char *name) {
    for (int i = 0; MODEL_TABLE[i].name; i++) {
        if (strcmp(MODEL_TABLE[i].name, name) == 0) return i;
    }
    return -1;
}

static Estimator *create_model(const char *name, int k_param) {
    if (strcmp(name, "linreg") == 0)
        return (Estimator*)linear_regression_create(LINREG_SOLVER_AUTO);
    if (strcmp(name, "logreg") == 0)
        return (Estimator*)logreg_model_create();
    if (strcmp(name, "knn") == 0) {
        /* KNN uses knn_fit, not Estimator API — wrap as placeholder */
        return NULL;
    }
    if (strcmp(name, "svm") == 0)
        return (Estimator*)svm_classifier_create(CML_KERNEL_LINEAR);
    if (strcmp(name, "nb") == 0)
        return (Estimator*)gaussian_nb_create();
    if (strcmp(name, "dtree") == 0)
        return (Estimator*)decision_tree_classifier_create();
    if (strcmp(name, "rf") == 0)
        return (Estimator*)random_forest_classifier_create(10);
    if (strcmp(name, "gb") == 0)
        return (Estimator*)gradient_boosting_create(50, 0.1, 3, 2, 1.0);
    if (strcmp(name, "nn") == 0) {
        int layers[] = {64, 32};
        return (Estimator*)mlp_classifier_create(layers, 2);
    }
    if (strcmp(name, "ridge") == 0)
        return (Estimator*)ridge_model_create();
    if (strcmp(name, "lasso") == 0)
        return (Estimator*)lasso_model_create();
    if (strcmp(name, "softmax") == 0)
        return (Estimator*)softmax_model_create();
    return NULL;
}

/* ── train command ───────────────────────────────────────────────── */

static int cmd_train(int argc, char **argv) {
    static struct option long_opts[] = {
        {"model",  required_argument, 0, 'm'},
        {"data",   required_argument, 0, 'd'},
        {"output", required_argument, 0, 'o'},
        {"target", required_argument, 0, 't'},
        {"params", required_argument, 0, 'p'},
        {"help",   no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    const char *model_name = NULL;
    const char *data_path = NULL;
    const char *output_path = NULL;
    int target_col = -1;  /* -1 = last column */
    int k_param = 5;

    optind = 2;  /* skip "train" subcommand */
    int c;
    while ((c = getopt_long(argc, argv, "m:d:o:t:p:h", long_opts, NULL)) != -1) {
        switch (c) {
            case 'm': model_name = optarg; break;
            case 'd': data_path = optarg; break;
            case 'o': output_path = optarg; break;
            case 't': target_col = atoi(optarg); break;
            case 'p': {
                /* Parse key=value params */
                char *eq = strchr(optarg, '=');
                if (eq) {
                    *eq = '\0';
                    if (strcmp(optarg, "k") == 0) k_param = atoi(eq + 1);
                    *eq = '=';
                }
                break;
            }
            case 'h': default:
                printf("Usage: tinycml train --model <type> --data <file.csv> --output <model.bin>\n");
                return 0;
        }
    }

    if (!model_name || !data_path || !output_path) {
        fprintf(stderr, "Error: --model, --data, and --output are required\n");
        return 1;
    }

    int midx = find_model(model_name);
    if (midx < 0) {
        fprintf(stderr, "Error: unknown model '%s'\n", model_name);
        return 1;
    }

    /* Load CSV */
    Matrix *data = csv_load_wrapper(data_path);
    if (!data) {
        fprintf(stderr, "Error: cannot load %s\n", data_path);
        return 1;
    }

    /* Split features and target */
    int tcol = (target_col >= 0) ? target_col : (int)data->cols - 1;
    size_t n_features = data->cols - 1;

    Matrix *X = matrix_alloc(data->rows, n_features);
    Matrix *y = matrix_alloc(data->rows, 1);
    for (size_t i = 0; i < data->rows; i++) {
        int fc = 0;
        for (size_t j = 0; j < data->cols; j++) {
            if ((int)j == tcol) {
                matrix_set(y, i, 0, matrix_get(data, i, j));
            } else {
                matrix_set(X, i, fc, matrix_get(data, i, j));
                fc++;
            }
        }
    }

    /* Create and train model */
    Estimator *model = create_model(model_name, k_param);
    if (!model) {
        fprintf(stderr, "Error: failed to create model\n");
        matrix_free(data); matrix_free(X); matrix_free(y);
        return 1;
    }

    printf("Training %s on %zu samples, %zu features...\n",
           model_name, X->rows, X->cols);

    clock_t start = clock();
    Estimator *fitted = model->fit(model, X, y);
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    if (!fitted) {
        fprintf(stderr, "Error: training failed\n");
        matrix_free(data); matrix_free(X); matrix_free(y);
        model->free(model);
        return 1;
    }

    /* Evaluate */
    double score = model->score(model, X, y);
    const char *metric = MODEL_TABLE[midx].is_classifier ? "Accuracy" : "R²";

    printf("%s: %.4f\n", metric, score);
    printf("Training time: %.3fs\n", elapsed);

    /* Save */
    if (model->save) {
        int rc = model->save(model, output_path);
        if (rc != 0) {
            fprintf(stderr, "Warning: could not save model (not all models support serialization)\n");
        } else {
            printf("Model saved to %s\n", output_path);
        }
    } else {
        fprintf(stderr, "Warning: this model does not support serialization\n");
    }

    matrix_free(data);
    matrix_free(X);
    matrix_free(y);
    model->free(model);
    return 0;
}

/* ── predict command ─────────────────────────────────────────────── */

static int cmd_predict(int argc, char **argv) {
    static struct option long_opts[] = {
        {"model",  required_argument, 0, 'm'},
        {"input",  required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"help",   no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    const char *model_path = NULL;
    const char *input_path = NULL;
    const char *output_path = NULL;

    optind = 2;
    int c;
    while ((c = getopt_long(argc, argv, "m:i:o:h", long_opts, NULL)) != -1) {
        switch (c) {
            case 'm': model_path = optarg; break;
            case 'i': input_path = optarg; break;
            case 'o': output_path = optarg; break;
            case 'h': default:
                printf("Usage: tinycml predict --model <model.bin> --input <data.csv> --output <preds.csv>\n");
                return 0;
        }
    }

    if (!model_path || !input_path || !output_path) {
        fprintf(stderr, "Error: --model, --input, and --output are required\n");
        return 1;
    }

    /* Try loading as each supported model type */
    /* For simplicity, try gradient_boosting_load first, then logreg, etc. */
    Estimator *model = gradient_boosting_load(model_path);
    if (!model) {
        /* Try other load functions — this is a simplified approach */
        fprintf(stderr, "Error: could not load model from %s\n", model_path);
        fprintf(stderr, "Note: prediction currently supports GBDT models\n");
        return 1;
    }

    Matrix *X = csv_load_wrapper(input_path);
    if (!X) {
        fprintf(stderr, "Error: cannot load %s\n", input_path);
        model->free(model);
        return 1;
    }

    printf("Predicting %zu samples...\n", X->rows);
    Matrix *preds = model->predict(model, X);
    if (!preds) {
        fprintf(stderr, "Error: prediction failed\n");
        matrix_free(X);
        model->free(model);
        return 1;
    }

    int rc = write_predictions_csv(output_path, preds);
    if (rc == 0) {
        printf("Predictions written to %s (%zu rows)\n", output_path, preds->rows);
    }

    matrix_free(preds);
    matrix_free(X);
    model->free(model);
    return rc;
}

/* ── info command ────────────────────────────────────────────────── */

static int cmd_info(int argc, char **argv) {
    static struct option long_opts[] = {
        {"model", required_argument, 0, 'm'},
        {"help",  no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    const char *model_path = NULL;
    optind = 2;
    int c;
    while ((c = getopt_long(argc, argv, "m:h", long_opts, NULL)) != -1) {
        switch (c) {
            case 'm': model_path = optarg; break;
            case 'h': default:
                printf("Usage: tinycml info --model <model.bin>\n");
                return 0;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: --model is required\n");
        return 1;
    }

    /* Load and display info */
    Estimator *model = gradient_boosting_load(model_path);
    if (model) {
        GradientBoosting *gb = (GradientBoosting*)model;
        printf("Model type:      %s\n", model_type_name(model->type));
        printf("Estimators:      %d\n", gb->n_estimators);
        printf("Learning rate:   %.3f\n", gb->learning_rate);
        printf("Max depth:       %d\n", gb->max_depth);
        printf("Classes:         %d\n", gb->n_classes);
        printf("Features:        %d\n", gb->n_features);
        model->free(model);
        return 0;
    }

    fprintf(stderr, "Error: could not read model info from %s\n", model_path);
    return 1;
}

/* ── benchmark command ───────────────────────────────────────────── */

typedef struct {
    const char *name;
    double score;
    double time_ms;
} BenchResult;

static int cmd_benchmark(int argc, char **argv) {
    static struct option long_opts[] = {
        {"dataset", required_argument, 0, 'd'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    const char *dataset = "iris";
    optind = 2;
    int c;
    while ((c = getopt_long(argc, argv, "d:h", long_opts, NULL)) != -1) {
        switch (c) {
            case 'd': dataset = optarg; break;
            case 'h': default:
                printf("Usage: tinycml benchmark [--dataset iris|boston]\n");
                return 0;
        }
    }

    Matrix *X, *y;
    int is_cls = 1;

    if (strcmp(dataset, "boston") == 0) {
        X = generate_boston_X();
        y = generate_boston_y(X);
        is_cls = 0;
        printf("Dataset: boston (%zu samples, %zu features, regression)\n\n", X->rows, X->cols);
    } else {
        X = generate_iris_X();
        y = generate_iris_y();
        printf("Dataset: iris (%zu samples, %zu features, classification)\n\n", X->rows, X->cols);
    }

    const char *cls_models[] = {"knn", "nb", "dtree", "rf", "gb", "logreg", "svm", NULL};
    const char *reg_models[] = {"linreg", "ridge", "lasso", "dtree", "rf", "gb", NULL};

    BenchResult results[16];
    int n_results = 0;

    const char **models = is_cls ? cls_models : reg_models;

    for (int m = 0; models[m]; m++) {
        Estimator *model = create_model(models[m], 5);
        if (!model) continue;

        clock_t start = clock();
        Estimator *fitted = model->fit(model, X, y);
        clock_t end = clock();
        double ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

        if (fitted) {
            double score = model->score(model, X, y);
            results[n_results].name = models[m];
            results[n_results].score = score;
            results[n_results].time_ms = ms;
            n_results++;
        }

        model->free(model);
    }

    /* Print results */
    printf("%-20s %-12s %s\n", "Model", is_cls ? "Accuracy" : "R²", "Time (ms)");
    printf("-----------------------------------------\n");
    for (int i = 0; i < n_results; i++) {
        printf("%-20s %-12.4f %.2f\n",
               results[i].name, results[i].score, results[i].time_ms);
    }

    matrix_free(X);
    matrix_free(y);
    return 0;
}

/* ── main ────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 2) {
        usage();
        return 0;
    }

    const char *cmd = argv[1];

    if (strcmp(cmd, "train") == 0)     return cmd_train(argc, argv);
    if (strcmp(cmd, "predict") == 0)   return cmd_predict(argc, argv);
    if (strcmp(cmd, "info") == 0)      return cmd_info(argc, argv);
    if (strcmp(cmd, "benchmark") == 0) return cmd_benchmark(argc, argv);
    if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        usage();
        return 0;
    }

    fprintf(stderr, "Unknown command: %s\n", cmd);
    usage();
    return 1;
}
