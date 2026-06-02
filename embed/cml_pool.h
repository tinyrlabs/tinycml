/**
 * @file cml_pool.h
 * @brief Static memory pool allocator for embedded tinycml
 *
 * Replaces malloc/free with a fixed-size pool allocator.
 * Use when your MCU has no heap or you need deterministic allocation.
 *
 * Usage:
 * @code
 *   #define CML_POOL_SIZE 4096   // 4 KB pool
 *   #define CML_IMPLEMENTATION
 *   #include "cml_pool.h"
 *   #include "tinycml.h"         // uses pool internally
 * @endcode
 *
 * Memory budget guide:
 *   KNN (12 samples, 2 features):  ~600 bytes
 *   GaussianNB (16 samples, 3f):   ~800 bytes
 *   Linear Regression (10 samples): ~400 bytes
 *   Full tinycml + pool overhead:  ~4 KB recommended minimum
 */
#ifndef CML_POOL_H
#define CML_POOL_H

#include <stddef.h>

/* Default pool size if not configured */
#ifndef CML_POOL_SIZE
#define CML_POOL_SIZE (4 * 1024)  /* 4 KB default */
#endif

/* ============================================================
 * Pool API
 * ============================================================ */

/**
 * @brief Initialize the memory pool (call once at startup)
 * @return 0 on success, -1 if pool too small for internal structures
 */
int cml_pool_init(void);

/**
 * @brief Allocate from pool (replaces malloc)
 * @param size Number of bytes
 * @return Pointer or NULL if pool exhausted
 */
void* cml_pool_alloc(size_t size);

/**
 * @brief Free pool memory (resets to last mark — no per-block free)
 * @param ptr Pointer from cml_pool_alloc (ignored, pool uses linear alloc)
 *
 * In embedded mode, we use a simple bump allocator.
 * free() is a no-op; the entire pool resets on cml_pool_reset().
 */
void cml_pool_free(void *ptr);

/**
 * @brief Reset pool (frees all allocations at once)
 *
 * Call between inference cycles to reclaim memory.
 */
void cml_pool_reset(void);

/**
 * @brief Get pool stats
 */
typedef struct {
    size_t total_size;
    size_t used;
    size_t peak_used;
    size_t remaining;
    int    init_called;
} CMLPoolStats;

CMLPoolStats cml_pool_stats(void);

/* ============================================================
 * malloc/free override macros
 * ============================================================
 * Define CML_USE_POOL before including this header to redirect
 * standard allocation through the pool.
 * This requires that tinycml uses cml_malloc/cml_free internally.
 */

#ifdef CML_USE_POOL
#define cml_malloc(s)  cml_pool_alloc(s)
#define cml_free(p)    cml_pool_free(p)
#define cml_calloc(n,s) cml_pool_calloc(n,s)
#else
/* Fallback to stdlib */
#include <stdlib.h>
#define cml_malloc(s)  malloc(s)
#define cml_free(p)    free(p)
#define cml_calloc(n,s) calloc(n,s)
#endif

/* ============================================================
 * Implementation
 * ============================================================ */
#ifdef CML_IMPLEMENTATION

#include <string.h>

/* Pool memory (statically allocated) */
static char cml_pool_buffer[CML_POOL_SIZE];
static size_t cml_pool_offset = 0;
static size_t cml_pool_peak = 0;
static int cml_pool_initialized = 0;

int cml_pool_init(void) {
    cml_pool_offset = 0;
    cml_pool_peak = 0;
    cml_pool_initialized = 1;
    return 0;
}

void* cml_pool_alloc(size_t size) {
    if (!cml_pool_initialized) {
        if (cml_pool_init() != 0) return NULL;
    }

    /* Align to 8 bytes */
    size_t aligned = (size + 7) & ~7;

    if (cml_pool_offset + aligned > CML_POOL_SIZE) {
        return NULL;  /* Pool exhausted */
    }

    void *ptr = &cml_pool_buffer[cml_pool_offset];
    cml_pool_offset += aligned;

    if (cml_pool_offset > cml_pool_peak) {
        cml_pool_peak = cml_pool_offset;
    }

    return ptr;
}

void* cml_pool_calloc(size_t n, size_t size) {
    size_t total = n * size;
    void *ptr = cml_pool_alloc(total);
    if (ptr) {
        memset(ptr, 0, total);
    }
    return ptr;
}

void cml_pool_free(void *ptr) {
    (void)ptr;
    /* Bump allocator: free is a no-op.
     * Call cml_pool_reset() between inference cycles. */
}

void cml_pool_reset(void) {
    cml_pool_offset = 0;
}

CMLPoolStats cml_pool_stats(void) {
    CMLPoolStats s;
    s.total_size = CML_POOL_SIZE;
    s.used = cml_pool_offset;
    s.peak_used = cml_pool_peak;
    s.remaining = CML_POOL_SIZE - cml_pool_offset;
    s.init_called = cml_pool_initialized;
    return s;
}

#endif /* CML_IMPLEMENTATION */
#endif /* CML_POOL_H */
