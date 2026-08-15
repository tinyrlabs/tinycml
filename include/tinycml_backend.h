/**
 * @file tinycml_backend.h
 * @brief Hardware backend contract for tinycml (K230 / KPU class accelerators)
 *
 * This header defines the BACKEND CONTRACT ONLY — no implementation.
 * Implementations live outside the core library:
 *   - Tiny-Aero: reference/mock backend (host tests, no hardware)
 *   - Future: K230 KPU backend using the Canaan SDK (nncase kmodel)
 *
 * Contract rules:
 *   - Pure C11, no platform headers, no heap inside the contract
 *   - Model memory is allocated ONCE at load time; per-frame inference
 *     performs NO allocation and NO model reload
 *   - Vendor error codes are translated to CMLStatus by the implementation
 *   - All timestamps are monotonic microseconds (see tinycml_time.h usage
 *     by the caller — the contract itself is timestamp-free)
 *
 * Naming follows the tinycml convention (CML prefix, cml_* functions).
 */

#ifndef TINYCML_BACKEND_H
#define TINYCML_BACKEND_H

#include <stdint.h>
#include <stddef.h>
#include "cml_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Backend types                                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Backend kind — one contract, multiple implementations.
 */
typedef enum {
    TINYCML_BACKEND_MOCK = 0,  /* reference implementation (host tests) */
    TINYCML_BACKEND_K230,      /* Canaan K230 KPU via nncase kmodel      */
    TINYCML_BACKEND_COUNT
} tinycml_backend_kind_t;

/**
 * @brief Tensor data type.
 */
typedef enum {
    TINYCML_TENSOR_U8 = 0,   /* uint8_t, [0, 255] (raw camera)        */
    TINYCML_TENSOR_I8,       /* int8_t  (quantized activations)        */
    TINYCML_TENSOR_F32,      /* float   (float reference path)         */
    TINYCML_TENSOR_COUNT
} tinycml_tensor_dtype_t;

/**
 * @brief Tensor descriptor + data view.
 *
 * `data` points to memory owned by the caller (input) or by the model
 * (output scratch, allocated at load time). The contract never allocates.
 */
typedef struct {
    tinycml_tensor_dtype_t dtype;
    uint32_t ndim;                  /* 1..4 */
    uint32_t dims[4];               /* e.g. {1, 640, 384, 3} NCHW/NHWC   */
    uint8_t  layout_nchw;           /* 1 = NCHW, 0 = NHWC                */
    float    scale;                 /* INT8 quantization scale           */
    int8_t   zero_point;            /* INT8 quantization zero point      */
    void    *data;                  /* caller/model-owned, never alloc'd  */
    size_t   data_bytes;
} tinycml_tensor_t;

/**
 * @brief Backend configuration (passed once at init).
 */
typedef struct {
    tinycml_backend_kind_t kind;
    const char *model_path;         /* kmodel file (K230) or NULL (mock) */
    uint32_t max_scratch_bytes;     /* cap for model/scratch memory       */
    uint32_t log_level;             /* 0=off, 1=err, 2=warn, 3=info, 4=dbg */
} tinycml_backend_config_t;

/**
 * @brief Opaque backend handle (implementation-defined size).
 */
typedef struct tinycml_backend tinycml_backend_t;

/**
 * @brief Loaded model handle — input/output metadata + scratch.
 * The implementation owns the struct and its scratch; the caller only
 * holds the pointer between load() and release().
 */
typedef struct tinycml_model tinycml_model_t;

/* ------------------------------------------------------------------ */
/* Contract API                                                        */
/* ------------------------------------------------------------------ */

/**
 * @brief Create + initialize a backend handle (implementation allocates).
 * @param config  backend configuration (copied by the implementation)
 * @param backend out: opaque handle (NULL on failure)
 * @return CML_OK or CML_ERROR_* (vendor codes translated)
 */
CMLStatus tinycml_backend_create(const tinycml_backend_config_t *config,
                                 tinycml_backend_t **backend);

/**
 * @brief Destroy a backend handle and release implementation resources.
 * Safe to call with NULL.
 */
void tinycml_backend_destroy(tinycml_backend_t *backend);

/**
 * @brief Load a model, parse input/output metadata, allocate scratch ONCE.
 * @param backend    initialized backend
 * @param model_path model file path (K230: kmodel; mock: may be NULL)
 * @param model      caller-provided storage for the model handle
 * @return CML_OK or CML_ERROR_* (CML_ERROR_FILE_IO, CML_ERROR_MEMORY, ...)
 */
CMLStatus tinycml_model_load(tinycml_backend_t *backend,
                             const char *model_path,
                             tinycml_model_t *model);

/**
 * @brief Run inference. No allocation, no reload. Thread-safe only if the
 *        implementation documents it; default contract: single inference
 *        in flight per model.
 * @param model  loaded model
 * @param input  input tensor view (caller-owned buffer)
 * @param output output tensor view (model-owned scratch, filled by call)
 * @return CML_OK or CML_ERROR_* (CML_ERROR_INVALID_ARG on shape mismatch)
 */
CMLStatus tinycml_infer(tinycml_model_t *model,
                        const tinycml_tensor_t *input,
                        tinycml_tensor_t *output);

/**
 * @brief Query model metadata (input/output shapes, scales) after load.
 * @param model  loaded model
 * @param input  filled with input descriptor (data == NULL)
 * @param output filled with output descriptor (data == NULL)
 * @return CML_OK or CML_ERROR_NULL_PTR
 */
CMLStatus tinycml_model_info(const tinycml_model_t *model,
                             tinycml_tensor_t *input,
                             tinycml_tensor_t *output);

/**
 * @brief Report last inference duration in microseconds.
 * @param model     loaded model
 * @param duration_us out: monotonic microseconds of the last tinycml_infer
 * @return CML_OK or CML_ERROR_NULL_PTR
 */
CMLStatus tinycml_model_last_infer_us(const tinycml_model_t *model,
                                      uint64_t *duration_us);

/**
 * @brief Release a model and all its scratch memory.
 */
void tinycml_model_release(tinycml_model_t *model);

/**
 * @brief Shut a backend down and release implementation resources.
 * @deprecated Use tinycml_backend_destroy (handle-based) in new code.
 */
void tinycml_backend_shutdown(tinycml_backend_t *backend);

#ifdef __cplusplus
}
#endif

#endif /* TINYCML_BACKEND_H */
