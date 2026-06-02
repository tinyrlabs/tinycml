# tinycml-embed

tinycml'nin embedded sistemler için portu. ARM Cortex-M, ESP32 ve RISC-V'de ML.

## Kullanım

KNN ile STM32F4'te ML örneği:

```c
#define CML_ENABLE_KNN
#define CML_USE_POOL
#define CML_POOL_SIZE 2048

#include "embed/cml_config.h"
#include "embed/cml_pool.h"
#define CML_IMPLEMENTATION
#include "tinycml.h"

int main(void) {
    cml_pool_init();  // Bellek havuzunu başlat

    // KNN train/predict...
    cml_pool_reset();  // İnference döngüleri arasında belleği sıfırla
}
```

## Compile-Time Seçenekler

| Flag | Ne işe yarar |
|------|-------------|
| `CML_USE_POOL` | Statik bellek havuzu kullan (malloc yok) |
| `CML_POOL_SIZE=N` | Havuz boyutu (default: 4096) |
| `CML_ENABLE_KNN` | Sadece KNN dahil et |
| `CML_ENABLE_NAIVE_BAYES` | Sadece Naive Bayes dahil et |
| `CML_ENABLE_ALL` | Tüm algoritmalar (default) |

## Build

```bash
# ARM Cortex-M
make -f embed/Makefile.embed arm CML_ALGOS="KNN NAIVE_BAYES"

# ESP32
make -f embed/Makefile.embed esp32 CML_ALGOS="KNN"

# RISC-V
make -f embed/Makefile.embed riscv CML_ALGOS="LINEAR_REGRESSION"

# Native test
make -f embed/Makefile.embed test_native
```

## Bellek Kullanımı

| Algoritma | RAM (peak) | Flash (tahmini) |
|-----------|-----------|----------------|
| KNN (12 örnek, 2 feature) | ~600 bytes | ~4 KB |
| GaussianNB (16 örnek, 3 feature) | ~800 bytes | ~5 KB |
| Linear Regression (8 örnek) | ~400 bytes | ~3 KB |
| Linear Regression + KNN + NB | ~1.5 KB | ~12 KB |

## Dizin Yapısı

```
embed/
  cml_pool.h       # Statik bellek havuzu
  cml_config.h     # Compile-time yapılandırma
  Makefile.embed   # Cross-compilation
examples/embedded/
  stm32/main.c     # STM32F4 + KNN comfort predictor
  esp32/main.c     # ESP32 + GaussianNB anomaly detector
  riscv/main.c     # RISC-V + Linear Regression temp predictor
  test_embed_native.c  # Native test suite (55 test)
```
