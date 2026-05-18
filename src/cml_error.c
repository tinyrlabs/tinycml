/**
 * @file cml_error.c
 * @brief Implementation of unified error handling
 */

#include "cml_error.h"
#include <stdarg.h>

/* Thread-local last error (C11 _Thread_local) */
static _Thread_local CMLError cml_last_error = { CML_OK, "" };

void cml_set_error(CMLStatus code, const char *fmt, ...) {
    cml_last_error.code = code;
    va_list args;
    va_start(args, fmt);
    vsnprintf(cml_last_error.message, sizeof(cml_last_error.message), fmt, args);
    va_end(args);
}

CMLError cml_get_last_error(void) {
    return cml_last_error;
}

const char* cml_status_string(CMLStatus code) {
    switch (code) {
        case CML_OK:                    return "OK";
        case CML_ERROR_NULL_PTR:        return "NULL pointer";
        case CML_ERROR_OUT_OF_BOUNDS:   return "Out of bounds";
        case CML_ERROR_INVALID_ARG:     return "Invalid argument";
        case CML_ERROR_MEMORY:          return "Memory allocation failed";
        case CML_ERROR_DIMENSION_MISMATCH: return "Dimension mismatch";
        case CML_ERROR_NOT_FITTED:      return "Model not fitted";
        case CML_ERROR_FILE_IO:         return "File I/O error";
        case CML_ERROR_PARSE:           return "Parse error";
        case CML_ERROR_CONVERGENCE:     return "Convergence failure";
        case CML_ERROR_UNSUPPORTED:     return "Unsupported operation";
        default:                        return "Unknown error";
    }
}
