# tinycml-embed

tinycml'nin embedded sistemler için portu. ARM Cortex-M, ESP32 ve RISC-V'de ML.
*Embedded port of tinycml. ML on ARM Cortex-M, ESP32, and RISC-V.*

## Kullanım / Usage

KNN ile STM32F4'te ML örneği / *KNN comfort predictor on STM32F4:*

```c
#define CML_ENABLE_KNN
#define CML_USE_POOL
#define CML_POOL_SIZE 2048

#include "embed/cml_config.h"
#include "embed/cml_pool.h"
#define CML_IMPLEMENTATION
#include "tinycml.h"

int main(void) {
    cml_pool_init();  // init memory pool

    // train, predict...
    cml_pool_reset(); // reset between inference cycles
}
```

## Compile-Time Seçenekler / *Options*

| Flag | Açıklama / *Description* |
|------|--------------------------|
| `CML_USE_POOL` | Static pool allocator (no malloc) |
| `CML_POOL_SIZE=N` | Pool size (default: 4096) |
| `CML_NO_MATH` | Use math lookup tables (no -lm) |
| `CML_ENABLE_KNN` | Include only KNN |
| `CML_ENABLE_NAIVE_BAYES` | Include only Naive Bayes |
| `CML_ENABLE_ALL` | All algorithms (default) |

## Build / Derleme

```bash
# ARM Cortex-M
make -f embed/Makefile.embed arm CML_ALGOS="KNN"

# ESP32 (ESP-IDF)
make -f embed/Makefile.embed esp32 CML_ALGOS="NAIVE_BAYES"

# RISC-V
make -f embed/Makefile.embed riscv CML_ALGOS="LINEAR_REGRESSION"

# Without math.h (lookup tables)
make arm CML_ALGOS="KNN" CML_NO_MATH=1

# Native test on host
make -f embed/Makefile.embed test_native
```

## Bellek Kullanımı / *Memory Usage*

| Algorithm / Algoritma | RAM (peak) | Flash (est.) |
|-----------------------|-----------|--------------|
| KNN (12 samples, 2 features) | ~600 B | ~4 KB |
| GaussianNB (16 samples, 3 features) | ~800 B | ~5 KB |
| Linear Regression (8 samples) | ~400 B | ~3 KB |
| KNN + NB + LR | ~1.5 KB | ~12 KB |
| KNN only | ~600 B | ~4 KB |

## Dizin Yapısı / *Directory Structure*

```
embed/
  cml_pool.h            # Pool allocator / Havuz ayırıcı
  cml_math_lut.h        # Math LUT (CML_NO_MATH)
  cml_config.h          # Compile-time config / Yapılandırma
  Makefile.embed        # Cross-compilation / Çapraz derleme
examples/embedded/
  stm32/main.c          # STM32F4 + KNN comfort predictor
  esp32/main.c          # ESP32 + GaussianNB anomaly detector
  riscv/main.c          # RISC-V + Linear Regression temp predictor
  platformio/           # ESP32 + Arduino via PlatformIO
  test_embed_native.c   # Native test suite (55 test)
```
