# Build System Specification

## CMake Configuration

### Minimum Requirements

- CMake 3.10+
- C11 compiler (GCC, Clang)
- Math library (-lm)

### Targets

| Target | Type | Description |
|--------|------|-------------|
| `cml` | STATIC | Main library (libcml.a) |
| `test_matrix` | EXECUTABLE | Matrix tests |
| `test_linreg` | EXECUTABLE | Linear regression tests |
| `linear_regression_example` | EXECUTABLE | LinReg demo |
| `logistic_regression_example` | EXECUTABLE | LogReg demo |
| `knn_example` | EXECUTABLE | k-NN demo |
| `kmeans_example` | EXECUTABLE | k-Means demo |

### CMakeLists.txt Structure

```cmake
cmake_minimum_required(VERSION 3.10)
project(c-ml-from-scratch C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler warnings
add_compile_options(-Wall -Wextra -pedantic)

# Library sources
file(GLOB LIB_SOURCES "src/*.c")
add_library(cml STATIC ${LIB_SOURCES})
target_include_directories(cml PUBLIC include)
target_link_libraries(cml m)

# Examples
add_subdirectory(examples)

# Tests
enable_testing()
add_subdirectory(tests)
```

---

## Makefile Wrapper

Convenience wrapper around CMake:

```makefile
BUILD_DIR = build

all: build

build:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && cmake .. && make

test: build
	cd $(BUILD_DIR) && ctest --output-on-failure

examples: build
	@echo "Examples built in $(BUILD_DIR)/examples/"

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all build test examples clean
```

---

## Directory Structure for Build

```
c-ml-from-scratch/
├── CMakeLists.txt          # Root CMake
├── Makefile                 # Convenience wrapper
├── include/
│   └── *.h
├── src/
│   └── *.c
├── examples/
│   ├── CMakeLists.txt      # Examples CMake
│   └── *.c
├── tests/
│   ├── CMakeLists.txt      # Tests CMake
│   ├── test_harness.h      # Test framework
│   └── test_*.c
└── build/                   # Generated (gitignored)
    ├── libcml.a
    ├── examples/
    └── tests/
```

---

## Test Harness

Simple assert-based testing (no external framework):

```c
// tests/test_harness.h
#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) void name(void)

#define RUN_TEST(name) do { \
    printf("Running %s... ", #name); \
    tests_run++; \
    name(); \
    tests_passed++; \
    printf("PASSED\n"); \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("FAILED\n  Assertion failed: %s\n  at %s:%d\n", \
               #cond, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))

#define ASSERT_NEAR(a, b, eps) ASSERT(fabs((a) - (b)) < (eps))

#define TEST_SUMMARY() do { \
    printf("\n%d/%d tests passed\n", tests_passed, tests_run); \
    return tests_passed == tests_run ? 0 : 1; \
} while(0)

#endif
```

---

## GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: sudo apt-get update && sudo apt-get install -y cmake build-essential

      - name: Build
        run: make build

      - name: Run tests
        run: make test
```
