# tinycml Embedded Cross-Compilation Makefile
#
# Targets:
#   stm32  - ARM Cortex-M4 (arm-none-eabi-gcc)
#   esp32  - Xtensa LX6 (xtensa-esp32-elf-gcc)
#   clean  - Remove build artifacts
#   size   - Show code size for stm32 build
#
# Prerequisites:
#   STM32: sudo apt install gcc-arm-none-eabi
#   ESP32: Install ESP-IDF or Xtensa toolchain
#   Both:  Run `make single_header` in tinycml root first
#
# Usage:
#   make stm32          # Compile STM32 example
#   make esp32          # Compile ESP32 example
#   make all            # Compile both
#   make size           # Show size analysis for STM32

# -----------------------------------------------------------
# Toolchain configuration
# -----------------------------------------------------------

# STM32 ARM Cortex-M4
ARM_CC      = arm-none-eabi-gcc
ARM_OBJDUMP = arm-none-eabi-objdump
ARM_SIZE    = arm-none-eabi-size
ARM_FLAGS   = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16

# ESP32 Xtensa LX6
ESP_CC      = xtensa-esp32-elf-gcc
ESP_SIZE    = xtensa-esp32-elf-size
ESP_FLAGS   = -mlongcalls

# -----------------------------------------------------------
# Common flags — optimized for embedded
# -----------------------------------------------------------

# tinycml root (relative to this Makefile)
TINYCML_ROOT = ../..

# Path to single-header amalgamation
TINYCML_HEADER = $(TINYCML_ROOT)/tinycml.h

INCLUDES = -I$(TINYCML_ROOT)

# Compiler flags for both targets
COMMON_FLAGS = \
    -std=c11 \
    -Os \
    -Wall \
    -Wextra \
    -ffreestanding \
    -nostdlib \
    -fno-builtin \
    -fno-exceptions \
    -fno-unwind-tables \
    -ffunction-sections \
    -fdata-sections \
    -Wl,--gc-sections \
    -Wl,--no-warn-mismatch

# Optional: strip unused tinycml features to reduce code size
# Uncomment as needed:
# CFLAGS_STRIP = \
#     -DCML_NO_FILE_IO \
#     -DCML_NO_SERIALIZATION

CFLAGS_STRIP =

# -----------------------------------------------------------
# Build directories
# -----------------------------------------------------------

BUILD_DIR = build
STM32_DIR = $(BUILD_DIR)/stm32
ESP32_DIR = $(BUILD_DIR)/esp32

# -----------------------------------------------------------
# Source files
# -----------------------------------------------------------

STM32_SRC = stm32/main.c
ESP32_SRC = esp32/main.c

STM32_OBJ = $(STM32_DIR)/main.o
ESP32_OBJ = $(ESP32_DIR)/main.o

# -----------------------------------------------------------
# Targets
# -----------------------------------------------------------

.PHONY: all stm32 esp32 clean size header_check help

all: stm32 esp32

# --- STM32 (ARM Cortex-M4) ---

stm32: $(STM32_OBJ)
	@echo ""
	@echo "=== STM32 Build Complete ==="
	@echo "Object: $(STM32_OBJ)"
	@$(ARM_SIZE) $(STM32_OBJ) 2>/dev/null || echo "(arm-none-eabi-size not available)"
	@echo ""
	@echo "To link into your firmware, add $(STM32_OBJ) to your linker command."
	@echo "Example: $(ARM_CC) $(ARM_FLAGS) $(COMMON_FLAGS) startup.o system.o $(STM32_OBJ) -T STM32F407.ld -o firmware.elf"

$(STM32_OBJ): $(STM32_SRC) $(TINYCML_HEADER) | $(STM32_DIR)
	$(ARM_CC) $(ARM_FLAGS) $(COMMON_FLAGS) $(CFLAGS_STRIP) $(INCLUDES) \
	    -c $< -o $@

# --- ESP32 (Xtensa LX6) ---

esp32: $(ESP32_OBJ)
	@echo ""
	@echo "=== ESP32 Build Complete ==="
	@echo "Object: $(ESP32_OBJ)"
	@$(ESP_SIZE) $(ESP32_OBJ) 2>/dev/null || echo "(xtensa-esp32-elf-size not available)"
	@echo ""
	@echo "For ESP-IDF projects, copy tinycml.h into components/tinycml/include/"
	@echo "and add esp32/main.c to your main component."

$(ESP32_OBJ): $(ESP32_SRC) $(TINYCML_HEADER) | $(ESP32_DIR)
	$(ESP_CC) $(ESP_FLAGS) $(COMMON_FLAGS) $(CFLAGS_STRIP) $(INCLUDES) \
	    -c $< -o $@

# --- Size analysis (STM32) ---

size: $(STM32_OBJ)
	@echo "=== Code Size Analysis (STM32 / Cortex-M4) ==="
	@echo ""
	@$(ARM_SIZE) -A $(STM32_OBJ) 2>/dev/null || \
	    echo "arm-none-eabi-size not found. Install: sudo apt install gcc-arm-none-eabi"
	@echo ""
	@echo "=== Section Breakdown ==="
	@$(ARM_OBJDUMP) -h $(STM32_OBJ) 2>/dev/null || true
	@echo ""
	@echo "STM32F407 Flash: 1 MB = 1,048,576 bytes"
	@echo "STM32F407 RAM:   192 KB = 196,608 bytes"

# --- Verify single-header exists ---

header_check:
	@test -f $(TINYCML_HEADER) && \
	    echo "tinycml.h found ($(shell wc -c < $(TINYCML_HEADER)) bytes)" || \
	    (echo "ERROR: $(TINYCML_HEADER) not found. Run 'make single_header' in $(TINYCML_ROOT)" && exit 1)

# --- Clean ---

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory."

# --- Help ---

help:
	@echo "tinycml Embedded Build System"
	@echo ""
	@echo "Targets:"
	@echo "  stm32    Compile STM32 example (ARM Cortex-M4)"
	@echo "  esp32    Compile ESP32 example (Xtensa LX6)"
	@echo "  all      Compile both targets"
	@echo "  size     Show code size analysis for STM32"
	@echo "  clean    Remove build artifacts"
	@echo "  help     Show this help"
	@echo ""
	@echo "Prerequisites:"
	@echo "  1. Run 'make single_header' in $(TINYCML_ROOT)"
	@echo "  2. Install cross-compilers:"
	@echo "     STM32: sudo apt install gcc-arm-none-eabi"
	@echo "     ESP32: Install ESP-IDF toolchain"
	@echo ""
	@echo "Size reduction flags (uncomment in Makefile):"
	@echo "  -DCML_NO_FILE_IO          Remove CSV loading"
	@echo "  -DCML_NO_SERIALIZATION    Remove model save/load"
	@echo ""
	@echo "Memory requirements:"
	@echo "  KNN (20 samples, 2 features):  ~700 bytes RAM"
	@echo "  GaussianNB (3 classes, 3 feat): ~300 bytes RAM"
	@echo "  DecisionTree (depth=4):        ~850 bytes RAM"
	@echo "  Code: ~8-25 KB Flash (model-dependent)"

# --- Directory creation ---

$(STM32_DIR):
	mkdir -p $@

$(ESP32_DIR):
	mkdir -p $@
