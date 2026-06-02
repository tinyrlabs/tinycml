/**
 * @file esp32/main.c
 * @brief tinycml GaussianNB classifier on ESP32 — sensor anomaly detection
 *
 * Classifies sensor readings (temperature, humidity, pressure) into
 * two categories: normal (0) and anomaly (1) using Gaussian Naive Bayes.
 *
 * Target: ESP32 (Xtensa LX6, 520 KB SRAM, 4 MB Flash)
 * Framework: FreeRTOS-compatible
 *
 * Memory budget (estimated):
 *   Training data (16 samples × 3 features + labels): ~544 bytes
 *   GaussianNB parameters (2 classes × 3 features):   ~144 bytes
 *   Prediction matrix (1 × 3):                        ~48 bytes
 *   tinycml code (NB + matrix subset):                ~10 KB Flash
 *   Total RAM: < 2 KB peak
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

/* ESP-IDF / FreeRTOS headers — uncomment for real ESP32 build */
/* #include "freertos/FreeRTOS.h" */
/* #include "freertos/task.h" */
/* #include "driver/uart.h" */
/* #include "esp_log.h" */

/* ============================================================
 * tinycml-embed configuration
 * Sensor anomaly detection on ESP32 with GaussianNB
 * ============================================================ */
#define CML_ENABLE_NAIVE_BAYES
#define CML_USE_POOL
#define CML_POOL_SIZE 2048

#include "cml_config.h"
#include "cml_pool.h"

/* ============================================================
 * tinycml — single-header mode
 * ============================================================ */
#define CML_IMPLEMENTATION
#include "tinycml.h"

/* ============================================================
 * Application: Sensor anomaly detection
 * ============================================================
 * Training data: 16 labeled sensor readings from an environmental
 * monitoring system.
 *
 * Features: temperature (°C), humidity (%), pressure (hPa)
 * Label:    0 = normal operation, 1 = anomaly
 *
 * Normal range:
 *   Temperature: 20–26°C
 *   Humidity:    30–60%
 *   Pressure:    1000–1020 hPa
 *
 * Anomalies include sensor faults, extreme conditions, etc.
 */

#define TAG "tinycml-demo"

static const double train_data[][4] = {
    /* temp   hum    press   label */
    /* Normal readings */
    { 22.5,  45.0,  1013.0,  0.0 },
    { 23.0,  48.0,  1015.0,  0.0 },
    { 21.5,  42.0,  1010.0,  0.0 },
    { 24.0,  50.0,  1012.0,  0.0 },
    { 22.0,  46.0,  1014.0,  0.0 },
    { 23.5,  44.0,  1011.0,  0.0 },
    { 21.0,  47.0,  1016.0,  0.0 },
    { 25.0,  52.0,  1009.0,  0.0 },

    /* Anomalous readings */
    { 35.0,  85.0,   980.0,  1.0 },   /* Overheat + high humidity + low pressure */
    {  5.0,  15.0,  1050.0,  1.0 },   /* Freezing + dry + high pressure */
    { 38.0,  90.0,   975.0,  1.0 },   /* Extreme heat */
    { -2.0,  10.0,  1045.0,  1.0 },   /* Sub-zero */
    { 33.0,  80.0,   985.0,  1.0 },   /* Hot & humid */
    {  8.0,  20.0,  1040.0,  1.0 },   /* Very cold */
    { 36.0,  88.0,   978.0,  1.0 },   /* Extreme anomaly */
    { 10.0,  25.0,  1035.0,  1.0 },   /* Cold anomaly */
};

#define N_TRAIN    16
#define N_FEATURES  3
#define N_CLASSES   2

/* Test samples — mix of normal and edge cases */
static const double test_samples[][3] = {
    { 22.0,  45.0,  1012.0 },   /* Clearly normal */
    { 34.0,  82.0,   982.0 },   /* Clearly anomaly */
    { 26.5,  58.0,  1008.0 },   /* Borderline normal */
    { 12.0,  22.0,  1038.0 },   /* Likely anomaly */
    { 23.0,  47.0,  1013.0 },   /* Normal */
    { 37.0,  87.0,   977.0 },   /* Extreme anomaly */
};
#define N_TEST 6

static const char *class_labels[] = { "NORMAL", "ANOMALY" };

/* ============================================================
 * app_main — ESP32 entry point (renamed for ESP-IDF convention)
 * ============================================================ */

/* Use app_main for ESP-IDF, or main for standalone testing */
#ifdef ESP_PLATFORM
#define ENTRY_POINT app_main
#else
#define ENTRY_POINT main
#endif

void ENTRY_POINT(void) {
    /* --- Initialize memory pool --- */
    cml_pool_init();

    printf("\n========================================\n");
    printf("  tinycml on ESP32\n");
    printf("  GaussianNB Sensor Anomaly Detector\n");
    printf("========================================\n\n");

    printf("FreeRTOS-compatible demo\n");
    printf("Training samples: %d, Features: %d, Classes: %d\n\n",
           N_TRAIN, N_FEATURES, N_CLASSES);

    /* --- Build training matrices --- */
    Matrix *X_train = matrix_alloc(N_TRAIN, N_FEATURES);
    Matrix *y_train = matrix_alloc(N_TRAIN, 1);

    if (!X_train || !y_train) {
        printf("ERROR: matrix allocation failed\n");
        return;
    }

    for (int i = 0; i < N_TRAIN; i++) {
        matrix_set(X_train, i, 0, train_data[i][0]); /* temperature */
        matrix_set(X_train, i, 1, train_data[i][1]); /* humidity */
        matrix_set(X_train, i, 2, train_data[i][2]); /* pressure */
        matrix_set(y_train, i, 0, train_data[i][3]); /* label */
    }

    /* Print training data summary */
    int n_normal = 0, n_anomaly = 0;
    for (int i = 0; i < N_TRAIN; i++) {
        if (train_data[i][3] == 0.0) n_normal++;
        else n_anomaly++;
    }
    printf("Dataset: %d normal, %d anomaly\n\n", n_normal, n_anomaly);

    /* --- Train Gaussian Naive Bayes --- */
    printf("Training GaussianNB...\n");

    GaussianNaiveBayes *nb = gaussian_nb_create();
    if (!nb) {
        printf("ERROR: NB creation failed\n");
        matrix_free(X_train);
        matrix_free(y_train);
        return;
    }

    Estimator *fitted = nb->base.fit((Estimator *)nb, X_train, y_train);
    if (!fitted) {
        printf("ERROR: NB fit failed\n");
        nb->base.free((Estimator *)nb);
        matrix_free(X_train);
        matrix_free(y_train);
        return;
    }

    printf("  Classes:       %d\n", nb->n_classes);
    printf("  Features:      %d\n", nb->n_features);
    printf("  Var smoothing: %e\n\n", nb->var_smoothing);

    /* --- Print learned parameters --- */
    printf("Learned parameters:\n");
    for (int c = 0; c < nb->n_classes; c++) {
        printf("  Class %d (%s): prior=%.3f\n",
               c, class_labels[c], nb->class_priors[c]);
        printf("    Means:  ");
        for (int f = 0; f < nb->n_features; f++) {
            printf("%.1f ", nb->theta_[c * nb->n_features + f]);
        }
        printf("\n    Vars:   ");
        for (int f = 0; f < nb->n_features; f++) {
            printf("%.1f ", nb->var_[c * nb->n_features + f]);
        }
        printf("\n");
    }
    printf("\n");

    /* --- Run predictions on test samples --- */
    Matrix *X_test = matrix_alloc(N_TEST, N_FEATURES);
    if (!X_test) {
        printf("ERROR: test allocation failed\n");
        nb->base.free((Estimator *)nb);
        matrix_free(X_train);
        matrix_free(y_train);
        return;
    }

    for (int i = 0; i < N_TEST; i++) {
        matrix_set(X_test, i, 0, test_samples[i][0]);
        matrix_set(X_test, i, 1, test_samples[i][1]);
        matrix_set(X_test, i, 2, test_samples[i][2]);
    }

    Matrix *predictions = nb->base.predict((const Estimator *)nb, X_test);
    if (!predictions) {
        printf("ERROR: prediction failed\n");
        matrix_free(X_test);
        nb->base.free((Estimator *)nb);
        matrix_free(X_train);
        matrix_free(y_train);
        return;
    }

    /* --- Get probabilities for confidence --- */
    Matrix *proba = gaussian_nb_predict_proba((const Estimator *)nb, X_test);

    /* --- Display results --- */
    printf("Predictions:\n");
    printf("-------------------------------------------------------\n");
    printf("  Temp    Hum    Press     => Class (confidence)\n");
    printf("-------------------------------------------------------\n");

    for (int i = 0; i < N_TEST; i++) {
        int label = (int)predictions->data[i];
        double conf = 0.0;
        if (proba) {
            conf = matrix_get(proba, i, label) * 100.0;
        }
        printf("  %5.1f C  %4.0f%%  %6.0f hPa => %s (%.0f%%)\n",
               test_samples[i][0],
               test_samples[i][1],
               test_samples[i][2],
               class_labels[label],
               conf);
    }
    printf("-------------------------------------------------------\n\n");

    /* --- Evaluate on training data (sanity check) --- */
    double acc = nb->base.score((const Estimator *)nb, X_train, y_train);
    printf("Training accuracy: %.1f%%\n\n", acc * 100.0);

    /* --- FreeRTOS task example --- */
    printf("FreeRTOS integration example:\n");
    printf("  In production, run inference in a FreeRTOS task:\n");
    printf("\n");
    printf("  void sensor_task(void *pvParameters) {\n");
    printf("      GaussianNaiveBayes *model = (GaussianNaiveBayes*)pvParameters;\n");
    printf("      while (1) {\n");
    printf("          // Read sensors\n");
    printf("          double temp = read_bme280_temperature();\n");
    printf("          double hum  = read_bme280_humidity();\n");
    printf("          double pres = read_bme280_pressure();\n");
    printf("\n");
    printf("          // Predict\n");
    printf("          Matrix *X = matrix_alloc(1, 3);\n");
    printf("          matrix_set(X, 0, 0, temp);\n");
    printf("          matrix_set(X, 0, 1, hum);\n");
    printf("          matrix_set(X, 0, 2, pres);\n");
    printf("          Matrix *pred = model->base.predict(model, X);\n");
    printf("\n");
    printf("          if ((int)pred->data[0] == 1)\n");
    printf("              trigger_alert();\n");
    printf("\n");
    printf("          matrix_free(pred);\n");
    printf("          matrix_free(X);\n");
    printf("          vTaskDelay(pdMS_TO_TICKS(5000));\n");
    printf("      }\n");
    printf("  }\n\n");

    /* --- Memory usage summary --- */
    printf("Memory Usage (estimated):\n");
    printf("  NB parameters (theta+var+prior): %d bytes\n",
           (int)(N_CLASSES * N_FEATURES * sizeof(double) * 2 +   /* theta + var */
                N_CLASSES * sizeof(double) +                       /* priors */
                N_CLASSES * sizeof(int) +                          /* class_count */
                N_CLASSES * sizeof(double)));                      /* log_prior */
    printf("  Training data matrices:         %d bytes\n",
           (int)((N_TRAIN * N_FEATURES + N_TRAIN) * sizeof(double)));
    printf("  Prediction buffers:             %d bytes\n",
           (int)((N_TEST * N_FEATURES + N_TEST) * sizeof(double)));
    printf("  Probability matrix:             %d bytes\n",
           (int)(N_TEST * N_CLASSES * sizeof(double)));
    printf("  Total peak heap:                ~%d bytes\n",
           (int)(N_TRAIN * N_FEATURES * sizeof(double) * 2 +
                N_TRAIN * sizeof(double) * 2 +
                N_TEST * N_FEATURES * sizeof(double) +
                N_TEST * sizeof(double) +
                N_TEST * N_CLASSES * sizeof(double) +
                sizeof(GaussianNaiveBayes)));
    printf("  Target (ESP32): 520 KB SRAM available\n");
    printf("  Usage: < 1%% of available SRAM\n\n");

    /* --- Cleanup --- */
    if (proba)  matrix_free(proba);
    matrix_free(predictions);
    matrix_free(X_test);
    nb->base.free((Estimator *)nb);
    matrix_free(X_train);
    matrix_free(y_train);

    printf("Demo complete.\n\n");

    /* --- On real ESP32, spin up FreeRTOS task --- */
    /* xTaskCreate(sensor_task, "sensor_ml", 4096, nb, 5, NULL); */
    /* (model would be allocated before this and passed as parameter) */

#ifdef ESP_PLATFORM
    /* ESP-IDF expects app_main to return (FreeRTOS scheduler takes over) */
#else
    /* Standalone: loop forever like an embedded system */
    while (1) {
        /* In production: vTaskDelay(pdMS_TO_TICKS(1000)); */
    }
#endif
}
