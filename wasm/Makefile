# ────────────────────────────────────────────────────────────
#  tinycml WebAssembly Build
#  Compiles the C ML library to .js + .wasm via Emscripten
# ────────────────────────────────────────────────────────────

EMCC ?= emcc

# Project roots
SRCDIR   := ../src
INCDIR   := ../include

# Collect every src/*.c
LIB_SRCS := $(wildcard $(SRCDIR)/*.c)

# Output
TARGET_JS   := tinycml.js
TARGET_WASM := tinycml.wasm

# Exported C functions (must match actual symbols in the library)
EXPORTED_FUNCTIONS := \
	['_matrix_alloc','_matrix_free','_matrix_set','_matrix_get', \
	 '_linear_regression_create','_linear_regression_create_full', \
	 '_linreg_fit_closed','_linreg_fit_gd','_linreg_predict', \
	 '_logreg_fit','_logreg_predict','_logreg_predict_proba', \
	 '_logistic_regression_create','_logistic_regression_create_full', \
	 '_knn_fit','_knn_predict','_knn_free', \
	 '_gaussian_nb_create','_gaussian_nb_predict_proba']

# Common Emscripten flags
EMFLAGS := \
	-O2 \
	-s MODULARIZE=1 \
	-s EXPORT_NAME='TinyCML' \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s EXPORTED_FUNCTIONS='$(EXPORTED_FUNCTIONS)' \
	-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","_malloc","_free"]' \
	-std=c11 \
	-Wall -Wextra

# ── Targets ──────────────────────────────────────────────────

.PHONY: all single_header clean help

all: $(TARGET_JS)

# Build from src/*.c (multi-file mode)
$(TARGET_JS): $(LIB_SRCS)
	$(EMCC) $(EMFLAGS) -I$(INCDIR) $^ -o $@

# Build from the single-header amalgamation
single_header: ../tinycml.h
	$(EMCC) $(EMFLAGS) -x c $< -o $(TARGET_JS)

clean:
	rm -f $(TARGET_JS) $(TARGET_WASM) tinycml.wasm.js tinycml.worker.js

help:
	@echo "tinycml WASM build targets:"
	@echo ""
	@echo "  all            Build tinycml.js + tinycml.wasm from src/*.c"
	@echo "  single_header  Build from the single-header tinycml.h"
	@echo "  clean          Remove generated files"
	@echo "  help           Show this message"
	@echo ""
	@echo "Variables:"
	@echo "  EMCC  Emscripten compiler (default: emcc)"
	@echo ""
	@echo "Requirements:"
	@echo "  Emscripten SDK must be activated in the current shell."
	@echo "  See README.md for setup instructions."
