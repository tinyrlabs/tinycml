/**
 * @file riscv/main.c
 * @brief tinycml Linear Regression on RISC-V — temperature prediction
 *
 * Predicts temperature from ADC voltage using Linear Regression.
 *
 * Target: RISC-V RV32IMC (e.g., CH32V307, GD32VF103)
 * Toolchain: riscv-none-elf-gcc
 *
 * Memory budget (estimated):
 *   Training data (8 samples × 1 feature):       ~128 bytes
 *   LinearRegression model (coef + intercept):    ~16 bytes
 *   Prediction matrix (1 × 1):                   ~48 bytes
 *   tinycml code (LR + matrix subset):           ~4 KB Flash
 *   Total RAM: < 512 bytes peak
 */

#include <stddef.h>
#include <stdint.h>

/* Minimal printf via UART — replace with your HAL's implementation */
__attribute__((weak)) void uart_send(const char *buf, int len) {
    (void)buf;
    (void)len;
}

/* Simplified printf for embedded */
static int embed_printf(const char *fmt, ...) {
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);
    char buf[128];
    int pos = 0;
#define PUTC(c) do { if (pos < (int)sizeof(buf)-1) buf[pos++] = (c); } while(0)
    while (*fmt) {
        if (*fmt != '%') { PUTC(*fmt++); continue; }
        fmt++;
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
                char tmp[12]; int t = 0;
                if (intpart == 0) tmp[t++] = '0';
                while (intpart > 0) { tmp[t++] = '0' + (intpart % 10); intpart /= 10; }
                for (int i = t-1; i >= 0; i--) PUTC(tmp[i]);
                PUTC('.');
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

#ifdef printf
#undef printf
#endif
#define printf embed_printf

/* ============================================================
 * tinycml-embed configuration: Linear Regression only
 * ============================================================ */
#define CML_ENABLE_LINEAR_REGRESSION
#define CML_USE_POOL
#define CML_POOL_SIZE 1024

#include "cml_config.h"
#include "cml_pool.h"

#define CML_IMPLEMENTATION
#include "tinycml.h"

/* ============================================================
 * Application: Temperature Sensor Calibration
 * ============================================================
 * Maps ADC voltage (0-3.3V scaled to 0-3300) to temperature (°C).
 * Training data from a thermistor calibration table.
 *
 * ADC(mV)  Temperature(°C)
 *  200       -10
 *  500         0
 *  800        10
 * 1100        20
 * 1400        30
 * 1700        40
 * 2000        50
 * 2300        60
 */

static const double calib_data[][2] = {
    { 200.0, -10.0 },
    { 500.0,   0.0 },
    { 800.0,  10.0 },
    { 1100.0, 20.0 },
    { 1400.0, 30.0 },
    { 1700.0, 40.0 },
    { 2000.0, 50.0 },
    { 2300.0, 60.0 },
};

#define N_SAMPLES 8

/* ============================================================
 * main
 * ============================================================ */
int main(void) {
    cml_pool_init();

    /* Board init placeholder */
    /* SystemClock_Config(); */

    printf("\r\n=== tinycml on RISC-V ===\r\n");
    printf("Linear Regression Temperature Predictor\r\n");
    printf("Calibration samples: %d\r\n\r\n", N_SAMPLES);

    /* Build training matrices */
    Matrix *X = matrix_alloc(N_SAMPLES, 1);
    Matrix *y = matrix_alloc(N_SAMPLES, 1);

    if (!X || !y) {
        printf("ERROR: matrix alloc failed\r\n");
        while (1);
    }

    for (int i = 0; i < N_SAMPLES; i++) {
        matrix_set(X, i, 0, calib_data[i][0]); /* ADC mV */
        matrix_set(y, i, 0, calib_data[i][1]); /* °C */
    }

    /* Train */
    LinearRegression *model = linear_regression_create(LINREG_SOLVER_CLOSED);
    if (!model) {
        printf("ERROR: model create failed\r\n");
        while (1);
    }

    Estimator *fitted = model->base.fit((Estimator*)model, X, y);
    if (!fitted) {
        printf("ERROR: fit failed\r\n");
        while (1);
    }

    printf("Model: y = %.2f * x + %.2f\r\n\r\n",
           model->coef[0], model->intercept);

    /* Predict for test ADC values */
    double test_adc[] = { 350.0, 950.0, 1850.0, 2150.0 };
    int n_test = 4;

    printf("Predictions:\r\n");
    printf("-----------------------------\r\n");
    printf("  ADC(mV)  =>  Temp(°C)\r\n");
    printf("-----------------------------\r\n");

    for (int i = 0; i < n_test; i++) {
        Matrix *X_test = matrix_alloc(1, 1);
        matrix_set(X_test, 0, 0, test_adc[i]);
        Matrix *pred = model->base.predict((const Estimator*)model, X_test);
        printf("  %.0f     =>   %.1f\r\n",
               test_adc[i], pred ? pred->data[0] : -999.0);
        matrix_free(X_test);
        if (pred) matrix_free(pred);
    }
    printf("-----------------------------\r\n\r\n");

    double score = model->base.score((const Estimator*)model, X, y);
    printf("R2 score: %.4f\r\n\r\n", score);

    printf("Memory:\r\n");
    printf("  Calibration data: %d bytes\r\n",
           (int)(N_SAMPLES * 2 * sizeof(double)));
    printf("  LR model:         ~%d bytes\r\n",
           (int)sizeof(LinearRegression));
    printf("  Total peak:       <%d bytes\r\n", CML_POOL_SIZE);
    printf("  Target (RISC-V):  typically 32 KB SRAM\r\n\r\n");

    model->base.free((Estimator*)model);
    matrix_free(X);
    matrix_free(y);

    printf("Done.\r\n");

    while (1) {
        /* __WFI(); */
    }

    return 0;
}
