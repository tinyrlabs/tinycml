/**
 * @file cml_error.h
 * @brief Unified error handling for tinycml library
 *
 * Provides a thread-local error state with status codes, messages,
 * and convenience macros for common error patterns.
 */

#ifndef CML_ERROR_H
#define CML_ERROR_H

#include <stdio.h>
#include <string.h>

/**
 * @brief Status codes for library operations
 */
typedef enum {
    CML_OK = 0,
    CML_ERROR_NULL_PTR,
    CML_ERROR_OUT_OF_BOUNDS,
    CML_ERROR_INVALID_ARG,
    CML_ERROR_MEMORY,
    CML_ERROR_DIMENSION_MISMATCH,
    CML_ERROR_NOT_FITTED,
    CML_ERROR_FILE_IO,
    CML_ERROR_PARSE,
    CML_ERROR_CONVERGENCE,
    CML_ERROR_UNSUPPORTED
} CMLStatus;

/**
 * @brief Stores the last error code + message
 */
typedef struct {
    CMLStatus code;
    char message[256];
} CMLError;

/**
 * @brief Get the last error
 * @return Copy of the last error struct
 */
CMLError cml_get_last_error(void);

/**
 * @brief Set the last error (variadic printf-style message)
 * @param code Status code
 * @param fmt printf-style format string
 */
void cml_set_error(CMLStatus code, const char *fmt, ...);

/**
 * @brief Get a human-readable string for a status code
 * @param code Status code
 * @return Static string describing the code
 */
const char* cml_status_string(CMLStatus code);

/* Convenience macros */

#define CML_CHECK_NULL(ptr, msg) do { \
    if (!(ptr)) { cml_set_error(CML_ERROR_NULL_PTR, "%s: NULL pointer", (msg)); return NULL; } \
} while(0)

#define CML_CHECK_FITTED(model, msg) do { \
    if (!(model)->fitted) { cml_set_error(CML_ERROR_NOT_FITTED, "%s: model not fitted", (msg)); return NULL; } \
} while(0)

#define CML_RETURN_ERROR(code, msg) do { \
    cml_set_error((code), "%s", (msg)); \
    return NULL; \
} while(0)

#endif /* CML_ERROR_H */
