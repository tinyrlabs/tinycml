# Matrix Operations Specification

## Overview

Core matrix data structure and operations for the ML library.

## Data Structure

```c
typedef struct {
    size_t rows;
    size_t cols;
    double *data;  // Row-major order
} Matrix;
```

## Required Functions

### Memory Management

| Function | Signature | Description |
|----------|-----------|-------------|
| `matrix_alloc` | `Matrix* matrix_alloc(size_t rows, size_t cols)` | Allocate matrix, zero-initialized |
| `matrix_free` | `void matrix_free(Matrix *m)` | Free matrix memory |
| `matrix_copy` | `Matrix* matrix_copy(const Matrix *m)` | Deep copy |

### Element Access

| Function | Signature | Description |
|----------|-----------|-------------|
| `matrix_get` | `double matrix_get(const Matrix *m, size_t i, size_t j)` | Get element at (i,j) |
| `matrix_set` | `void matrix_set(Matrix *m, size_t i, size_t j, double val)` | Set element at (i,j) |

### Arithmetic Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `matrix_add` | `Matrix* matrix_add(const Matrix *a, const Matrix *b)` | Element-wise addition |
| `matrix_sub` | `Matrix* matrix_sub(const Matrix *a, const Matrix *b)` | Element-wise subtraction |
| `matrix_mul` | `Matrix* matrix_mul(const Matrix *a, const Matrix *b)` | Element-wise multiplication |
| `matrix_scale` | `Matrix* matrix_scale(const Matrix *m, double scalar)` | Scalar multiplication |
| `matrix_matmul` | `Matrix* matrix_matmul(const Matrix *a, const Matrix *b)` | Matrix multiplication |

### Transformations

| Function | Signature | Description |
|----------|-----------|-------------|
| `matrix_transpose` | `Matrix* matrix_transpose(const Matrix *m)` | Transpose matrix |

### Utilities

| Function | Signature | Description |
|----------|-----------|-------------|
| `matrix_print` | `void matrix_print(const Matrix *m)` | Print to stdout |
| `matrix_fill` | `void matrix_fill(Matrix *m, double val)` | Fill with value |
| `matrix_identity` | `Matrix* matrix_identity(size_t n)` | Create n×n identity |

## Error Handling

- Return `NULL` on allocation failure
- Return `NULL` on dimension mismatch (with stderr message)
- Check bounds in get/set (assert in debug mode)

## Memory Layout

Row-major order: `data[i * cols + j]` accesses element (i, j)

## Test Cases

1. Alloc/free cycle - no memory leaks
2. Identity matrix properties: I × A = A
3. Transpose of transpose: (A^T)^T = A
4. Matrix multiplication associativity
5. Element-wise operations with known values
