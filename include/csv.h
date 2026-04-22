/**
 * @file csv.h
 * @brief CSV file loading and saving for ML library
 */

#ifndef CSV_H
#define CSV_H

#include "matrix.h"

/**
 * @brief Load a CSV file into a Matrix
 * @param filename Path to CSV file
 * @param has_header 1 if file has header row to skip, 0 otherwise
 * @return Matrix with loaded data, or NULL on error
 */
Matrix* csv_load(const char *filename, int has_header);

/**
 * @brief Save a Matrix to a CSV file
 * @param m Matrix to save
 * @param filename Output file path
 * @return 0 on success, -1 on error
 */
int csv_save(const Matrix *m, const char *filename);

/**
 * @brief Options for robust CSV loading
 */
typedef struct {
    int has_header;       /**< first row = header names */
    char delimiter;       /**< ',', '\t', ';', etc. 0 = auto-detect */
    char quote;           /**< '"' or '\0' for no quoting */
    double missing_val;   /**< value for empty/invalid fields (NAN) */
    int skip_empty_lines; /**< skip blank lines */
    int max_rows;         /**< 0 = unlimited */
    int max_cols;         /**< 0 = unlimited */
} CSVOptions;

/**
 * @brief Load CSV with options. Returns Matrix (rows x cols).
 * @param path Path to CSV file
 * @param opts Pointer to CSVOptions (NULL for defaults)
 * @return Matrix with loaded data, or NULL on error
 */
Matrix* csv_load_ext(const char *path, const CSVOptions *opts);

/**
 * @brief Get header names (valid after csv_load_ext with has_header=1)
 * @return Array of header name strings
 */
const char** csv_get_headers(void);

/**
 * @brief Get number of headers
 * @return Number of header columns
 */
int csv_get_header_count(void);

/**
 * @brief Auto-detect delimiter by examining first line
 * @param path Path to CSV file
 * @return Detected delimiter character
 */
char csv_detect_delimiter(const char *path);

#endif /* CSV_H */
