/**
 * @file stm32/main.c
 * @brief tinycml KNN classifier on STM32F4 — comfort-level prediction
 *
 * Predicts comfort level (0=uncomfortable, 1=comfortable) from
 * temperature and humidity sensor readings using k-Nearest Neighbors.
 *
 * Target: STM32F4 (Cortex-M4, 192 KB RAM, 1 MB Flash)
 * Toolchain: arm-none-eabi-gcc
 *
 * Memory budget (estimated):
 *   Training data (12 samples × 2 features + labels): ~312 bytes
 *   KNN model struct + distance buffer: ~280 bytes
 *   Prediction matrix (1 × 2): ~48 bytes
 *   tinycml code (KNN + matrix subset): ~8 KB Flash
 *   Total RAM: < 1 KB peak
 */

/* ============================================================
 * Bare-metal compatibility shims
 * These replace stdlib functions used by tinycml internally.
 * In a real project, link against your HAL's minimal libc or
 * newlib-nano.
 * ============================================================ */
#include <stddef.h>
#include <stdint.h>
#include <math.h>

/* Minimal printf via UART — replace with your HAL's implementation.
 * STM32 HAL example: HAL_UART_Transmit(&huart2, (uint8_t*)buf, len, timeout)
 * For this demo we provide a stub that can be wired to any UART backend. */

/* User-provided UART transmit function. Override this in your board support. */
__attribute__((weak)) void uart_send(const char *buf, int len) {
    (void)buf;
    (void)len;
    /* Default: no-op. Replace with HAL_UART_Transmit or equivalent. */
}

/* Simplified printf for embedded — supports %d, %f, %s, %%. */
static int embedded_printf(const char *fmt, ...) {
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);

    char buf[128];
    int pos = 0;

    #define PUTC(c) do { if (pos < (int)sizeof(buf)-1) buf[pos++] = (c); } while(0)

    while (*fmt) {
        if (*fmt != '%') { PUTC(*fmt++); continue; }
        fmt++; /* skip '%' */
        switch (*fmt) {
            case 'd': {
                int val = __builtin_va_arg(ap, int);
                if (val < 0) { PUTC('-'); val = -val; }
                char tmp[12]; int t = 0;
                if (val == 0) tmp[t++] = '0';
                while (val > 0) { tmp[t++] = '0' + (val % 10); val /= 10; }
                for (int i = t-1; i >= 0; i--) PUTC(tmp[i]);
                break;
            }
            case 'f': {
                double val = __builtin_va_arg(ap, double);
                if (val < 0) { PUTC('-'); val = -val; }
                int intpart = (int)val;
                int frac = (int)((val - intpart) * 100.0);
                /* Print integer part */
                char tmp[12]; int t = 0;
                if (intpart == 0) tmp[t++] = '0';
                while (intpart > 0) { tmp[t++] = '0' + (intpart % 10); intpart /= 10; }
                for (int i = t-1; i >= 0; i--) PUTC(tmp[i]);
                PUTC('.');
                /* Fractional part (2 digits) */
                PUTC('0' + (frac / 10));
                PUTC('0' + (frac % 10));
                break;
            }
            case 's': {
                const char *s = __builtin_va_arg(ap, const char*);
                while (*s) PUTC(*s++);
                break;
            }
            case '%': PUTC('%'); break;
            default: PUTC('%'); PUTC(*fmt); break;
        }
        fmt++;
    }

    #undef PUTC
    buf[pos] = '\0';
    uart_send(buf, pos);
    __builtin_va_end(ap);
    return pos;
}

/* Redefine printf for tinycml internal use */
#define printf embedded_printf

/* ============================================================
 * tinycml-embed configuration
 * Minimal ML on STM32F4: KNN for comfort prediction
 * ============================================================ */
#define CML_ENABLE_KNN
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
 * Application: Comfort-level prediction
 * ============================================================
 * Training data: 12 sensor readings collected from a room environment.
 *   Features: temperature (°C), humidity (%)
 *   Label:    0 = uncomfortable, 1 = comfortable
 *
 * Comfortable range (approximate):
 *   Temperature: 20–26°C
 *   Humidity:    30–60%
 */

static const double train_data[][3] = {
    /* temp   hum   label */
    { 18.0,  25.0,  0.0 },   /* Too cold & dry */
    { 19.5,  30.0,  0.0 },   /* Slightly cool */
    { 17.0,  70.0,  0.0 },   /* Cold & humid */
    { 15.5,  80.0,  0.0 },   /* Very cold & humid */
    { 30.0,  85.0,  0.0 },   /* Hot & very humid */
    { 28.5,  75.0,  0.0 },   /* Hot & humid */
    { 22.0,  45.0,  1.0 },   /* Comfortable */
    { 23.5,  50.0,  1.0 },   /* Comfortable */
    { 21.0,  40.0,  1.0 },   /* Comfortable */
    { 24.0,  55.0,  1.0 },   /* Comfortable */
    { 22.5,  48.0,  1.0 },   /* Comfortable */
    { 20.5,  35.0,  1.0 },   /* Comfortable */
};

#define N_TRAIN 12
#define N_FEATURES 2
#define K_NEIGHBORS 3

/* Test samples to predict */
static const double test_samples[][2] = {
    { 23.0,  50.0 },   /* Should be comfortable (1) */
    { 16.0,  75.0 },   /* Should be uncomfortable (0) */
    { 25.0,  42.0 },   /* Edge case — likely comfortable (1) */
    { 29.0,  80.0 },   /* Should be uncomfortable (0) */
};
#define N_TEST 4

static const char *comfort_labels[] = { "UNCOMFORTABLE", "COMFORTABLE" };

/* ============================================================
 * main — entry point for bare-metal
 * ============================================================ */
int main(void) {
    /* --- Initialize memory pool --- */
    cml_pool_init();

    /* --- Board init (replace with your HAL init) --- */
    /* SystemClock_Config(); */
    /* MX_GPIO_Init(); */
    /* MX_USART2_UART_Init(); */

    printf("\r\n=== tinycml on STM32F4 ===\r\n");
    printf("KNN Comfort Predictor (k=%d)\r\n", K_NEIGHBORS);
    printf("Training samples: %d\r\n", N_TRAIN);

    /* --- Build training matrices --- */
    Matrix *X_train = matrix_alloc(N_TRAIN, N_FEATURES);
    Matrix *y_train = matrix_alloc(N_TRAIN, 1);

    if (!X_train || !y_train) {
        printf("ERROR: matrix allocation failed\r\n");
        while (1); /* Halt */
    }

    for (int i = 0; i < N_TRAIN; i++) {
        matrix_set(X_train, i, 0, train_data[i][0]); /* temperature */
        matrix_set(X_train, i, 1, train_data[i][1]); /* humidity */
        matrix_set(y_train, i, 0, train_data[i][2]); /* label */
    }

    printf("Training data loaded\r\n");

    /* --- Train KNN model --- */
    KNNModel *knn = knn_fit(X_train, y_train, K_NEIGHBORS);

    if (!knn) {
        printf("ERROR: KNN fit failed\r\n");
        while (1); /* Halt */
    }

    printf("KNN model fitted\r\n\r\n");

    /* --- Run predictions on test samples --- */
    Matrix *X_test = matrix_alloc(N_TEST, N_FEATURES);

    if (!X_test) {
        printf("ERROR: test allocation failed\r\n");
        while (1);
    }

    for (int i = 0; i < N_TEST; i++) {
        matrix_set(X_test, i, 0, test_samples[i][0]);
        matrix_set(X_test, i, 1, test_samples[i][1]);
    }

    Matrix *predictions = knn_predict(knn, X_test);

    if (!predictions) {
        printf("ERROR: prediction failed\r\n");
        while (1);
    }

    /* --- Display results --- */
    printf("Predictions:\r\n");
    printf("--------------------------------\r\n");
    printf("  Temp    Humidity  =>  Result\r\n");
    printf("--------------------------------\r\n");

    for (int i = 0; i < N_TEST; i++) {
        int label = (int)predictions->data[i];
        printf("  %f C   %f %%  =>  %s\r\n",
               test_samples[i][0],
               test_samples[i][1],
               comfort_labels[label]);
    }
    printf("--------------------------------\r\n\r\n");

    /* --- Evaluate on training data (sanity check) --- */
    Matrix *train_pred = knn_predict(knn, X_train);
    if (train_pred) {
        double acc = accuracy(y_train, train_pred);
        printf("Training accuracy: %f%%\r\n", acc * 100.0);
        matrix_free(train_pred);
    }

    /* --- Memory usage summary --- */
    printf("\r\nMemory Usage (estimated):\r\n");
    printf("  Training data:  %d bytes\r\n",
           (int)(N_TRAIN * N_FEATURES * sizeof(double) + N_TRAIN * sizeof(double)));
    printf("  KNN model:      ~%d bytes (stores refs to X/y)\r\n",
           (int)sizeof(KNNModel));
    printf("  Test data:      %d bytes\r\n",
           (int)(N_TEST * N_FEATURES * sizeof(double)));
    printf("  Prediction buf: %d bytes\r\n",
           (int)(N_TEST * sizeof(double)));
    printf("  Total heap:     ~%d bytes\r\n",
           (int)(N_TRAIN * N_FEATURES * sizeof(double) * 2 + /* X + internal copy */
                N_TRAIN * sizeof(double) * 2 +               /* y + internal copy */
                N_TEST * N_FEATURES * sizeof(double) +
                N_TEST * sizeof(double) +
                sizeof(KNNModel)));
    printf("  Target (STM32F4): 192 KB RAM / 1 MB Flash\r\n");
    printf("  Usage: < 1%% of available RAM\r\n\r\n");

    /* --- Cleanup --- */
    matrix_free(X_test);
    matrix_free(predictions);
    knn_free(knn);
    matrix_free(X_train);
    matrix_free(y_train);

    printf("Done. Entering main loop.\r\n\r\n");

    /* --- Continuous inference loop --- */
    /* In a real application, read sensors here and predict periodically.
     *
     * while (1) {
     *     double temp = read_temperature_sensor();
     *     double hum  = read_humidity_sensor();
     *
     *     Matrix *X_live = matrix_alloc(1, 2);
     *     matrix_set(X_live, 0, 0, temp);
     *     matrix_set(X_live, 0, 1, hum);
     *
     *     Matrix *pred = knn_predict(knn, X_live);
     *     int comfort = (int)pred->data[0];
     *     printf("Temp=%.1f Hum=%.1f => %s\r\n",
     *            temp, hum, comfort_labels[comfort]);
     *
     *     // Control HVAC based on prediction
     *     if (comfort == 0) adjust_hvac();
     *
     *     matrix_free(pred);
     *     matrix_free(X_live);
     *
     *     HAL_Delay(5000);  // Predict every 5 seconds
     * }
     */

    while (1) {
        /* __WFI(); */  /* Sleep until interrupt */
    }

    return 0;
}
