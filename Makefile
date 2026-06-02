# tinycml Makefile
# Tiny C Machine Learning Library - Direct compilation without CMake dependency

CC = cc
CFLAGS = -std=c11 -Wall -Wextra -pedantic -O2
ifeq ($(OPENMP),1)
CFLAGS += -fopenmp
endif
CFLAGS_PIC = $(CFLAGS) -fPIC
LDFLAGS = -lm
PREFIX = /usr/local
MAJOR_VERSION = 0
MINOR_VERSION = 1
PATCH_VERSION = 0
VERSION = $(MAJOR_VERSION).$(MINOR_VERSION).$(PATCH_VERSION)

BUILD_DIR = build
LIB_DIR = $(BUILD_DIR)/lib
OBJ_DIR = $(BUILD_DIR)/obj
OBJ_PIC_DIR = $(BUILD_DIR)/obj-pic
BIN_DIR = $(BUILD_DIR)/bin
TEST_DIR = $(BUILD_DIR)/tests
EXAMPLES_DIR = $(BUILD_DIR)/examples

INCLUDE = -Iinclude

# Source files
LIB_SRCS = $(wildcard src/*.c)
LIB_OBJS = $(patsubst src/%.c,$(OBJ_DIR)/%.o,$(LIB_SRCS))
LIB_PIC_OBJS = $(patsubst src/%.c,$(OBJ_PIC_DIR)/%.o,$(LIB_SRCS))

# Libraries
LIBRARY_STATIC = $(LIB_DIR)/libtinycml.a
LIBRARY_SHARED = $(LIB_DIR)/libtinycml.so

# Headers
HEADERS = $(wildcard include/*.h)

# pkg-config
PC_FILE = $(BUILD_DIR)/tinycml.pc
PC_IN = tinycml.pc.in

# Examples
EXAMPLES = linear_regression_example logistic_regression_example knn_example kmeans_example estimator_api_example cross_validation_example pipeline_example random_forest_example pca_example feature_selection_example
EXAMPLE_BINS = $(patsubst %,$(EXAMPLES_DIR)/%,$(EXAMPLES))

# Tests
TESTS = test_matrix test_linreg test_logreg test_logreg_model test_knn test_kmeans test_metrics test_decision_tree test_naive_bayes test_nn test_pca test_preprocessing test_svm test_onehot test_metrics_multiclass test_svm_proba test_knn_proba test_feature_selection test_pipeline test_model_selection test_ridge test_lasso test_softmax test_multinomial_nb test_svm_rbf test_dbscan test_cml_error test_csv_robust test_serialization test_silhouette
TEST_BINS = $(patsubst %,$(TEST_DIR)/%,$(TESTS))

.PHONY: all build library static shared examples tests test clean install uninstall docs help single_header

# Single-header amalgamation
single_header:
	bash scripts/amalgamate.sh

all: build

build: library shared examples tests

# Create directories
$(BUILD_DIR) $(LIB_DIR) $(OBJ_DIR) $(OBJ_PIC_DIR) $(BIN_DIR) $(TEST_DIR) $(EXAMPLES_DIR):
	mkdir -p $@

# Compile static library objects
$(OBJ_DIR)/%.o: src/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

# Compile PIC objects for shared library
$(OBJ_PIC_DIR)/%.o: src/%.c | $(OBJ_PIC_DIR)
	$(CC) $(CFLAGS_PIC) $(INCLUDE) -c $< -o $@

# Create static library
static: $(LIBRARY_STATIC)

library: static   # backward compat alias

$(LIBRARY_STATIC): $(LIB_OBJS) | $(LIB_DIR)
	ar rcs $@ $^

# Create shared library with soname versioning
shared: $(LIBRARY_SHARED)

LIB_REAL = $(LIB_DIR)/libtinycml.so.$(VERSION)
LIB_SONAME = $(LIB_DIR)/libtinycml.so.$(MAJOR_VERSION)

$(LIB_REAL): $(LIB_PIC_OBJS) | $(LIB_DIR)
	$(CC) -shared -Wl,-soname,libtinycml.so.$(MAJOR_VERSION) $^ -o $@ $(LDFLAGS)

$(LIB_SONAME): $(LIB_REAL)
	ln -sf libtinycml.so.$(VERSION) $@

$(LIBRARY_SHARED): $(LIB_REAL) $(LIB_SONAME)
	ln -sf libtinycml.so.$(VERSION) $@

# Generate pkg-config file
$(PC_FILE): $(PC_IN) | $(BUILD_DIR)
	sed \
		-e 's|@prefix@|$(PREFIX)|g' \
		-e 's|@version@|$(VERSION)|g' \
		$< > $@

# Build examples
examples: $(EXAMPLE_BINS)

$(EXAMPLES_DIR)/%: examples/%.c $(LIBRARY_STATIC) | $(EXAMPLES_DIR)
	$(CC) $(CFLAGS) $(INCLUDE) $< $(LIBRARY_STATIC) $(LDFLAGS) -o $@

# Build tests
tests: $(TEST_BINS)

$(TEST_DIR)/%: tests/%.c $(LIBRARY_STATIC) | $(TEST_DIR)
	$(CC) $(CFLAGS) $(INCLUDE) -Itests $< $(LIBRARY_STATIC) $(LDFLAGS) -o $@

# Run tests
test: tests
	@echo "Running tests..."
	@for test in $(TEST_BINS); do \
		echo ""; \
		echo "=== Running $$test ==="; \
		$$test || exit 1; \
	done
	@echo ""
	@echo "All tests passed!"

# Install
install: build $(PC_FILE)
	install -d $(DESTDIR)$(PREFIX)/include/tinycml
	install -d $(DESTDIR)$(PREFIX)/lib
	install -m 644 $(HEADERS) $(DESTDIR)$(PREFIX)/include/tinycml/
	install -m 644 $(LIBRARY_STATIC) $(DESTDIR)$(PREFIX)/lib/
	install -m 755 $(LIB_REAL) $(DESTDIR)$(PREFIX)/lib/
	ln -sf libtinycml.so.$(VERSION) $(DESTDIR)$(PREFIX)/lib/libtinycml.so.$(MAJOR_VERSION)
	ln -sf libtinycml.so.$(VERSION) $(DESTDIR)$(PREFIX)/lib/libtinycml.so
	install -d $(DESTDIR)$(PREFIX)/lib/pkgconfig
	install -m 644 $(PC_FILE) $(DESTDIR)$(PREFIX)/lib/pkgconfig/
	-ldconfig

# Uninstall
uninstall:
	rm -f $(addprefix $(DESTDIR)$(PREFIX)/include/tinycml/,$(notdir $(HEADERS)))
	rm -f $(DESTDIR)$(PREFIX)/lib/libtinycml.a
	rm -f $(DESTDIR)$(PREFIX)/lib/libtinycml.so
	rm -f $(DESTDIR)$(PREFIX)/lib/pkgconfig/tinycml.pc
	rm -rf $(DESTDIR)$(PREFIX)/include/tinycml
	-ldconfig

# Documentation
docs:
	doxygen Doxyfile

clean:
	rm -rf $(BUILD_DIR)

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Build everything (static lib, shared lib, examples, tests)"
	@echo "  build    - Same as all"
	@echo "  static   - Build static library only (libtinycml.a)"
	@echo "  library  - Alias for static (backward compat)"
	@echo "  shared   - Build shared library only (libtinycml.so)"
	@echo "  examples - Build example programs"
	@echo "  tests    - Build test programs"
	@echo "  test     - Build and run tests"
	@echo "  docs     - Generate documentation with Doxygen"
	@echo "  install  - Install headers, libraries, and pkg-config file"
	@echo "  uninstall- Remove installed files"
	@echo "  single_header - Generate single-header tinycml.h amalgamation"
	@echo "  clean    - Remove build directory"
	@echo ""
	@echo "Variables:"
	@echo "  CC       - C compiler (default: cc)"
	@echo "  CFLAGS   - Compiler flags (default: -std=c11 -Wall -Wextra -pedantic -O2)"
	@echo "  PREFIX   - Install prefix (default: /usr/local)"
	@echo "  VERSION  - Library version (default: 0.1.0)"
