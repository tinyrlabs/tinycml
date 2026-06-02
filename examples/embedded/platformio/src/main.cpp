/**
 * @file main.cpp
 * @brief tinycml KNN comfort predictor on ESP32 via PlatformIO
 *
 * Predicts comfort level from temperature and humidity using KNN.
 * Retrains and predicts on a loop every 10 seconds.
 *
 * Build:  pio run
 * Upload: pio run --target upload
 */
#include <Arduino.h>
#include "cml_config.h"
#include "cml_pool.h"
#define CML_IMPLEMENTATION
#include "tinycml.h"

static const float train_data[][3] = {
    {18.0, 25.0, 0}, {19.5, 30.0, 0}, {17.0, 70.0, 0},
    {15.5, 80.0, 0}, {30.0, 85.0, 0}, {28.5, 75.0, 0},
    {22.0, 45.0, 1}, {23.5, 50.0, 1}, {21.0, 40.0, 1},
    {24.0, 55.0, 1}, {22.5, 48.0, 1}, {20.5, 35.0, 1},
};
#define N_TRAIN 12
#define K_NEIGH 3

static const float test_samples[][2] = {
    {23.0, 50.0}, {16.0, 75.0}, {25.0, 42.0}, {29.0, 80.0},
};
#define N_TEST 4
static const char *labels[] = {"UNCOMFORTABLE", "COMFORTABLE"};

void train_and_predict() {
    cml_pool_init();

    Matrix *X = matrix_alloc(N_TRAIN, 2);
    Matrix *y = matrix_alloc(N_TRAIN, 1);
    if (!X || !y) { Serial.println("alloc FAIL"); return; }

    for (int i = 0; i < N_TRAIN; i++) {
        matrix_set(X, i, 0, train_data[i][0]);
        matrix_set(X, i, 1, train_data[i][1]);
        matrix_set(y, i, 0, train_data[i][2]);
    }

    KNNModel *knn = knn_fit(X, y, K_NEIGH);
    if (!knn) { Serial.println("fit FAIL"); matrix_free(X); matrix_free(y); return; }

    Matrix *Xt = matrix_alloc(N_TEST, 2);
    for (int i = 0; i < N_TEST; i++) {
        matrix_set(Xt, i, 0, test_samples[i][0]);
        matrix_set(Xt, i, 1, test_samples[i][1]);
    }
    Matrix *pred = knn_predict(knn, Xt);

    Serial.println("Temp  Humidity  => Comfort");
    for (int i = 0; i < N_TEST; i++) {
        Serial.printf("%.1f  %.0f%%    => %s\n",
            test_samples[i][0], test_samples[i][1],
            pred ? labels[(int)pred->data[i]] : "ERR");
    }

    matrix_free(Xt);
    if (pred) matrix_free(pred);
    knn_free(knn);
    matrix_free(X);
    matrix_free(y);
    cml_pool_reset();
}

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    Serial.println("\n=== tinycml on ESP32 (PlatformIO) ===");
    train_and_predict();
}

void loop() {
    delay(10000);
    Serial.println("\n--- Inference cycle ---");
    train_and_predict();
}
