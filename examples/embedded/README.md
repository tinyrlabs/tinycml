# tinycml on Embedded Systems

Run machine learning inference directly on microcontrollers — no OS, no dynamic allocation in core ops, no external dependencies.

## Why tinycml on Embedded?

- **Single-header mode**: Drop `tinycml.h` into any project with `#define CML_IMPLEMENTATION`
- **C11, no dependencies**: Works with bare-metal toolchains (ARM GCC, Xtensa, RISC-V)
- **No dynamic allocation in core matrix ops**: Predictable memory usage
- **Deterministic inference**: No hidden allocations during prediction

## Recommended Models for Constrained Devices

| Model | RAM (peak) | Flash (code) | Train Time | Best For |
|-------|-----------|-------------|-----------|----------|
| **KNN** | `O(n_samples × n_features)` | ~2 KB | Instant (lazy) | Small datasets, prototyping |
| **GaussianNB** | `O(n_classes × n_features)` | ~3 KB | Fast | Sensor classification, anomaly detection |
| **DecisionTree** | `O(n_nodes × sizeof(TreeNode))` | ~4 KB | Fast | Rule-based logic, interpretable |
| LinearRegression | `O(n_features²)` | ~2 KB | Fast | Trend prediction, calibration |
| LogisticRegression | `O(n_features²)` | ~3 KB | Fast | Binary classification |

**For devices with < 32 KB RAM**: Use GaussianNB or DecisionTree with constrained depth.
**For devices with 64–192 KB RAM**: All models above work comfortably.

### Memory Estimation Formulas

**KNN** (n=20 samples, 2 features):
- Training data: `(20 × 2 + 20 × 1) × 8 = 480 bytes`
- Prediction temp: `~200 bytes`
- **Total: ~700 bytes**

**GaussianNB** (3 classes, 3 features):
- Parameters: `(3 × 3 + 3 × 3 + 3) × 8 = 168 bytes`
- Prediction temp: `~100 bytes`
- **Total: ~300 bytes**

**DecisionTree** (max_depth=4, ~15 nodes):
- Nodes: `15 × ~48 bytes = ~720 bytes`
- Prediction stack: `~100 bytes`
- **Total: ~850 bytes**

## Supported Targets

| MCU | Core | RAM | Flash | Status |
|-----|------|-----|-------|--------|
| STM32F401 | Cortex-M4 | 96 KB | 256 KB | ✅ Tested |
| STM32F411 | Cortex-M4 | 128 KB | 512 KB | ✅ Compatible |
| STM32F407 | Cortex-M4 | 192 KB | 1 MB | ✅ Compatible |
| STM32F103 | Cortex-M3 | 20 KB | 64–128 KB | ⚠️ GaussianNB/DT only |
| ESP32 | Xtensa LX6 | 520 KB | 4 MB | ✅ Tested |
| ESP32-S3 | Xtensa LX7 | 512 KB | 8–16 MB | ✅ Compatible |
| RP2040 | Cortex-M0+ | 264 KB | 16 MB | ✅ Compatible |

## Build Instructions

### Prerequisites

Generate the single-header amalgamation first:

```bash
cd /path/to/tinycml
make single_header
# produces tinycml.h in project root
```

### STM32 (ARM Cortex-M)

```bash
# Install toolchain
sudo apt install gcc-arm-none-eabi

# Build
cd examples/embedded
make stm32

# Output: build/stm32/main.o  (link into your firmware project)
```

### ESP32 (Xtensa / ESP-IDF)

```bash
# Using ESP-IDF component
cp tinycml.h components/tinycml/include/
cp examples/embedded/esp32/main.c main/

# Or compile directly with Xtensa toolchain
cd examples/embedded
make esp32
```

### Custom Bare-Metal Target

```bash
# Generic flags for any C11 bare-metal compiler
<cross-gcc> -std=c11 -Os -ffreestanding -nostdlib \
    -DCML_NO_FILE_IO \
    -DCML_NO_SERIALIZATION \
    -I. main.c -o firmware.elf
```

## Reducing Code Size

Strip unused features at compile time:

```c
/* Define BEFORE including tinycml.h to disable features */
#define CML_NO_FILE_IO          /* Remove CSV loading */
#define CML_NO_SERIALIZATION    /* Remove model save/load */
#define CML_NO_PIPELINES        /* Remove pipeline framework */
```

**Typical size reductions:**
- Full library: ~80 KB Flash
- Without CSV/serialization: ~45 KB Flash
- KNN + NB + DecisionTree only: ~25 KB Flash

## Examples

- **stm32/** — KNN classifier for comfort-level prediction (temperature + humidity)
- **esp32/** — GaussianNB for sensor anomaly classification (3 features, 2 classes)
