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
 * Some MCUs lack full math.h. Define CML_NO_MATH to use
 * lookup tables for sqrt/exp/log instead.
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
