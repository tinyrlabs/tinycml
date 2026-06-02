/**
 * @file cml_config.h
 * @brief tinycml compile-time configuration for embedded targets
 *
 * Define these BEFORE including tinycml.h to select which algorithms
 * to include. This reduces code size significantly.
 *
 * Example (KNN + NB only, ~8 KB Flash):
 * @code
 *   #define CML_ENABLE_KNN
 *   #define CML_ENABLE_NAIVE_BAYES
 *   #define CML_USE_POOL
 *   #define CML_POOL_SIZE 2048
 *   #include "embed/cml_config.h"
 *   #include "embed/cml_pool.h"
 *   #include "tinycml.h"
 * @endcode
 */
#ifndef CML_CONFIG_H
#define CML_CONFIG_H

/* ============================================================
 * Algorithm Selection
 * ============================================================
 * Define CML_ENABLE_ALL to include everything (default).
 * For embedded, define only what you need.
 * ============================================================ */

/* If nothing defined, include everything */
#if !defined(CML_ENABLE_KNN) && \
    !defined(CML_ENABLE_NAIVE_BAYES) && \
    !defined(CML_ENABLE_LINEAR_REGRESSION) && \
    !defined(CML_ENABLE_LOGISTIC_REGRESSION) && \
    !defined(CML_ENABLE_SVM) && \
    !defined(CML_ENABLE_DECISION_TREE) && \
    !defined(CML_ENABLE_RANDOM_FOREST) && \
    !defined(CML_ENABLE_KMEANS) && \
    !defined(CML_ENABLE_DBSCAN) && \
    !defined(CML_ENABLE_PCA) && \
    !defined(CML_ENABLE_NEURAL_NETWORK) && \
    !defined(CML_ENABLE_RIDGE) && \
    !defined(CML_ENABLE_LASSO) && \
    !defined(CML_ENABLE_SGD) && \
    !defined(CML_ENABLE_GRADIENT_BOOSTING) && \
    !defined(CML_ENABLE_ISOLATION_FOREST) && \
    !defined(CML_ENABLE_AGGLOMERATIVE) && \
    !defined(CML_ENABLE_FEATURE_SELECTION) && \
    !defined(CML_ENABLE_MODEL_SELECTION) && \
    !defined(CML_ENABLE_PIPELINE) && \
    !defined(CML_ENABLE_SERIALIZATION) && \
    !defined(CML_ENABLE_PREPROCESSING)
#define CML_ENABLE_ALL
#endif

/* ============================================================
 * Memory Configuration
 * ============================================================ */
#ifndef CML_POOL_SIZE
#define CML_POOL_SIZE (4 * 1024)  /* 4 KB default pool */
#endif

/* ============================================================
 * Math Configuration
 * ============================================================
 * Some MCUs lack full math.h or you want to avoid linking -lm.
 * Define CML_NO_MATH to use pre-computed lookup tables with
 * linear interpolation for sqrt, exp, log, and log10 instead
 * of math.h functions.
 *
 * The lookup tables provide ~0.1% accuracy and cover:
 *   sqrt(x):  x in [0, 100], step 0.1,   1001 entries (~8 KB)
 *   exp(x):   x in [-10, 10], step 0.01, 2001 entries (~16 KB)
 *   log(x):   x in [0.1, 100], step 0.1, 1000 entries (~8 KB)
 *   log10(x): derived from log(x) / ln(10)
 * Total table cost: ~32 KB of Flash when CML_NO_MATH is active.
 *
 * Usage with Makefile.embed:
 *   make arm CML_NO_MATH=1     # ARM + lookup tables
 *   make native CML_NO_MATH=1  # Native + lookup tables
 *
 * Usage in source:
 *   #define CML_NO_MATH
 *   #include "embed/cml_math_lut.h"    // before tinycml.h
 *   #include "tinycml.h"
 *
 * When CML_NO_MATH is NOT defined, cml_sqrt/cml_exp/cml_log
 * are thin inline wrappers around math.h (default behavior).
 * ============================================================ */
/* #define CML_NO_MATH */

/* ============================================================
 * Assertions
 * ============================================================ */
#ifndef CML_ASSERT
#ifdef NDEBUG
#define CML_ASSERT(x) ((void)0)
#else
#include <assert.h>
#define CML_ASSERT(x) assert(x)
#endif
#endif

#endif /* CML_CONFIG_H */
