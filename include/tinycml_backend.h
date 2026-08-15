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
 * @brief Backend contract API version.
 * Consumers must set config.api_version to this value; mismatches are
 * rejected with CML_ERROR_UNSUPPORTED (ABI hardening).
 */
#define TINYCML_BACKEND_API_VERSION 2u

/**
 * @brief Capability flags (bitmask) — reported by query_info, and
 * accepted (as a required minimum) in the config.
 */
#define TINYCML_CAP_DTYPE_U8   (1u << 0u)  /* uint8 tensors             */
#define TINYCML_CAP_DTYPE_I8   (1u << 1u)  /* int8 tensors              */
#define TINYCML_CAP_DTYPE_F32  (1u << 2u)  /* float32 tensors           */
#define TINYCML_CAP_LAYOUT_NCHW (1u << 8u) /* NCHW layout               */
#define TINYCML_CAP_LAYOUT_NHWC (1u << 9u) /* NHWC layout               */

/**
 * @brief Backend kind — one contract, multiple implementations.
 * The mock backend MUST NOT impersonate K230: requesting
 * TINYCML_BACKEND_K230 without a real runtime fails with
 * CML_ERROR_UNSUPPORTED.
 */
typedef enum {
    TINYCML_BACKEND_MOCK = 0,  /* reference implementation (host tests) */
    TINYCML_BACKEND_K230,      /* Canaan K230 KPU via nncase kmodel      */
    TINYCML_BACKEND_COUNT
} tinycml_backend_kind_t;

/**
 * @brief Explicit backend lifecycle state.
 *
 * Valid transitions (all others are rejected with CML_ERROR_STATE):
 *
 *   (none) --create--> INITIALIZED --model_load--> MODEL_LOADED
 *      MODEL_LOADED --first infer--> READY
 *      READY --infer--> READY
 *      MODEL_LOADED|READY --model_release--> INITIALIZED (model released)
 *      INITIALIZED|MODEL_LOADED|READY --destroy--> RELEASED
 *      RELEASED --(any op)--> CML_ERROR_STATE
 */
typedef enum {
    TINYCML_BACKEND_STATE_NONE = 0,
    TINYCML_BACKEND_STATE_CREATED,     /* storage allocated            */
    TINYCML_BACKEND_STATE_INITIALIZED, /* create() completed           */
    TINYCML_BACKEND_STATE_MODEL_LOADED,/* model loaded, infer not yet  */
    TINYCML_BACKEND_STATE_READY,       /* infer allowed                */
    TINYCML_BACKEND_STATE_RELEASED     /* destroyed                    */
} tinycml_backend_state_t;

/**
 * @brief Backend information (static capabilities) — query_info().
 */
typedef struct {
    uint32_t api_version;             /* contract version implemented  */
    uint32_t struct_size;             /* sizeof(this struct)          */
    char     name[32];                /* e.g. "tinycml-mock"          */
    char     version[16];             /* implementation version        */
    uint32_t capabilities;            /* TINYCML_CAP_* bitmask        */
    uint32_t supported_dtypes;        /* TINYCML_CAP_DTYPE_* mask     */
    uint32_t supported_layouts;       /* TINYCML_CAP_LAYOUT_* mask    */
} tinycml_backend_info_t;

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
 * Shape values are MODEL metadata — the contract itself does NOT fix
 * any dimensions (no YOLO/class-count assumption).
 */
typedef struct {
    tinycml_tensor_dtype_t dtype;
    uint32_t ndim;                  /* 1..4 */
    uint32_t dims[4];               /* shape from model metadata        */
    uint8_t  layout_nchw;           /* 1 = NCHW, 0 = NHWC                */
    float    scale;                 /* INT8 quantization scale           */
    int8_t   zero_point;            /* INT8 quantization zero point      */
    void    *data;                  /* caller/model-owned, never alloc'd  */
    size_t   data_bytes;
} tinycml_tensor_t;

/**
 * @brief Backend configuration (passed once at create).
 *
 * Consumers MUST set api_version and struct_size (ABI hardening).
 * vendor_config is implementation-specific and opaque to the contract
 * (mock: shape fixture; K230: nncase options).
 */
typedef struct {
    uint32_t api_version;           /* must be TINYCML_BACKEND_API_VERSION */
    uint32_t struct_size;           /* must be sizeof(tinycml_backend_config_t) */
    tinycml_backend_kind_t kind;
    const char *model_path;         /* kmodel file (K230) or NULL (mock) */
    uint32_t max_scratch_bytes;     /* cap for model/scratch memory       */
    uint32_t log_level;             /* 0=off, 1=err, 2=warn, 3=info, 4=dbg */
    uint32_t required_capabilities; /* minimum TINYCML_CAP_* mask; 0=any  */
    const void *vendor_config;      /* opaque impl-specific config        */
    uint32_t vendor_config_size;    /* bytes at vendor_config             */
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
 * @brief Query backend static info (name, version, capabilities).
 * @param backend created backend
 * @param info     out: filled info struct
 * @return CML_OK or CML_ERROR_NULL_PTR / CML_ERROR_STATE
 */
CMLStatus tinycml_backend_query_info(const tinycml_backend_t *backend,
                                     tinycml_backend_info_t *info);

/**
 * @brief Create + initialize a backend handle (implementation allocates).
 * @param config  backend configuration (copied by the implementation);
 *                api_version/struct_size MUST be set (ABI check)
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
 * @param backend    created backend
 * @param model_path model file path (K230: kmodel; mock: may be NULL)
 * @param model      out: opaque model handle (NULL on failure)
 * @return CML_OK or CML_ERROR_* (CML_ERROR_FILE_IO, CML_ERROR_MEMORY, ...)
 */
CMLStatus tinycml_model_load(tinycml_backend_t *backend,
                             const char *model_path,
                             tinycml_model_t **model);

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
