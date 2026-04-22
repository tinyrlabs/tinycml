/**
 * @file csv.c
 * @brief Implementation of robust CSV loading and saving
 */

#define _POSIX_C_SOURCE 200809L

#include "csv.h"
#include "cml_error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define MAX_LINE_LENGTH 65536
#define INITIAL_CAPACITY 256
#define MAX_FIELD_LEN   4096
#define MAX_HEADERS      1024

/* ── Module-level header storage ────────────────────────────────────── */

static char    *g_header_storage[MAX_HEADERS];
static const char *g_header_ptrs[MAX_HEADERS];
static int      g_header_count = 0;

static void free_headers(void) {
    for (int i = 0; i < g_header_count; i++) {
        free(g_header_storage[i]);
        g_header_storage[i] = NULL;
        g_header_ptrs[i] = NULL;
    }
    g_header_count = 0;
}

const char** csv_get_headers(void) {
    return (const char**)g_header_ptrs;
}

int csv_get_header_count(void) {
    return g_header_count;
}

/* ── Utility helpers ────────────────────────────────────────────────── */

static int is_empty_line(const char *line) {
    while (*line) {
        if (!isspace((unsigned char)*line)) return 0;
        line++;
    }
    return 1;
}

static int is_comment_line(const char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    if (*line == '#') return 1;
    if (line[0] == '/' && line[1] == '/') return 1;
    return 0;
}

/* Skip UTF-8 BOM if present, return pointer past it */
static const char* skip_bom(const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    if (p[0] == 0xEF && p[1] == 0xBB && p[2] == 0xBF) {
        return s + 3;
    }
    return s;
}

/**
 * Read the entire file content into a malloc'd buffer.
 * Handles BOM skipping at the start.
 */
static char* read_file_contents(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        cml_set_error(CML_ERROR_FILE_IO, "csv: cannot open file '%s'", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz < 0) { fclose(fp); return NULL; }

    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    size_t nread = fread(buf, 1, (size_t)sz, fp);
    buf[nread] = '\0';
    fclose(fp);

    /* Skip BOM in the buffer by shifting content */
    const char *after_bom = skip_bom(buf);
    if (after_bom != buf) {
        size_t offset = (size_t)(after_bom - buf);
        memmove(buf, after_bom, nread - offset + 1);
    }
    return buf;
}

/* ── Delimiter auto-detection ───────────────────────────────────────── */

char csv_detect_delimiter(const char *path) {
    char *contents = read_file_contents(path);
    if (!contents) return ',';

    /* Find first non-empty, non-comment line */
    char *line = contents;
    while (*line) {
        /* skip leading whitespace */
        while (*line && isspace((unsigned char)*line)) line++;
        if (*line == '\0') break;
        if (*line == '#' || (line[0] == '/' && line[1] == '/')) {
            /* skip to next line */
            while (*line && *line != '\n') line++;
            if (*line == '\n') line++;
            continue;
        }
        break;
    }

    /* Count candidate delimiters up to newline */
    int commas = 0, tabs = 0, semis = 0;
    const char *p = line;
    while (*p && *p != '\n' && *p != '\r') {
        switch (*p) {
            case ',': commas++; break;
            case '\t': tabs++; break;
            case ';': semis++; break;
        }
        p++;
    }

    free(contents);

    if (tabs > commas && tabs > semis) return '\t';
    if (semis > commas && semis > tabs) return ';';
    return ',';
}

/* ── Robust field parser ────────────────────────────────────────────── */

/**
 * Parse one field starting at *pos in buf.
 * Writes the raw field string into `field` (up to field_cap-1 chars).
 * Advances *pos past the field and delimiter (if any).
 * Returns 1 if a delimiter followed (more fields), 0 if end-of-record.
 */
static int parse_field(const char *buf, size_t *pos, char *field, size_t field_cap,
                       char delim, char quote) {
    size_t fi = 0;
    int hit_delim = 0;

    /* skip leading whitespace (not newlines) */
    while (buf[*pos] && buf[*pos] != '\n' && buf[*pos] != '\r' &&
           isspace((unsigned char)buf[*pos])) {
        (*pos)++;
    }

    if (buf[*pos] == quote && quote != '\0') {
        /* Quoted field */
        (*pos)++; /* skip opening quote */
        while (buf[*pos]) {
            if (buf[*pos] == quote) {
                /* Peek: double-quote escape? */
                if (buf[*pos + 1] == quote) {
                    if (fi < field_cap - 1) field[fi++] = quote;
                    *pos += 2;
                } else {
                    /* Closing quote */
                    (*pos)++;
                    break;
                }
            } else {
                if (fi < field_cap - 1) field[fi++] = buf[*pos];
                (*pos)++;
            }
        }
        /* skip trailing whitespace before delimiter */
        while (buf[*pos] && buf[*pos] != '\n' && buf[*pos] != '\r' &&
               isspace((unsigned char)buf[*pos])) {
            (*pos)++;
        }
        if (buf[*pos] == delim) {
            (*pos)++;
            hit_delim = 1;
        }
    } else {
        /* Unquoted field */
        while (buf[*pos] && buf[*pos] != delim &&
               buf[*pos] != '\n' && buf[*pos] != '\r') {
            if (fi < field_cap - 1) field[fi++] = buf[*pos];
            (*pos)++;
        }
        if (buf[*pos] == delim) {
            (*pos)++;
            hit_delim = 1;
        }
    }

    field[fi] = '\0';

    /* Trim trailing whitespace from unquoted field */
    if (fi > 0) {
        /* Only trim if it wasn't a quoted field — but we already handled
           quoted fields above with trailing whitespace skip. For unquoted,
           trim right. */
        size_t end = fi;
        while (end > 0 && isspace((unsigned char)field[end - 1])) {
            field[--end] = '\0';
        }
    }

    return hit_delim;
}

/**
 * Parse one record (row) starting at *pos.
 * Fields are written into the values array starting at `values[row * ncols + col_offset]`.
 * Returns number of columns parsed, or -1 on skip (empty/comment line).
 * Advances *pos past the record.
 */
static int parse_record(const char *buf, size_t *pos, char delim, char quote,
                        double missing, int max_cols, int is_header_row,
                        char **headers_out) {
    size_t start = *pos;

    /* skip leading whitespace */
    while (buf[*pos] && buf[*pos] != '\n' && buf[*pos] != '\r' &&
           isspace((unsigned char)buf[*pos])) {
        (*pos)++;
    }

    /* Check comment */
    if (buf[*pos] == '#' || (buf[*pos] == '/' && buf[*pos + 1] == '/')) {
        /* skip to end of line */
        while (buf[*pos] && buf[*pos] != '\n') (*pos)++;
        if (buf[*pos] == '\n') (*pos)++;
        return -1;
    }

    /* Check empty line */
    if (buf[*pos] == '\n' || buf[*pos] == '\r' || buf[*pos] == '\0') {
        if (buf[*pos] == '\r') (*pos)++;
        if (buf[*pos] == '\n') (*pos)++;
        return -1;
    }

    /* Rewind to start of content to parse fields */
    *pos = start;

    char field[MAX_FIELD_LEN];
    int col = 0;

    while (1) {
        /* End of record? */
        if (buf[*pos] == '\0' || buf[*pos] == '\n' || buf[*pos] == '\r') {
            /* Previous parse_field consumed the last field and stopped at
               newline — but if we hit newline here without parse_field,
               it means a trailing empty field wasn't consumed.
               This happens when the line is just "\n" but we already checked that.
               Actually, parse_field will consume up to delim or newline.
               If the record ended right at newline without a trailing delim,
               the loop ends naturally. */
            break;
        }

        int hit_delim = parse_field(buf, pos, field, sizeof(field), delim, quote);

        if (is_header_row) {
            /* Store header name */
            if (headers_out && col < MAX_HEADERS) {
                headers_out[col] = strdup(field);
            }
        } else {
            /* We'll parse the double later — just count columns for now.
               The caller handles storing values. */
        }

        col++;

        if (max_cols > 0 && col >= max_cols) {
            /* skip rest of line */
            while (buf[*pos] && buf[*pos] != '\n' && buf[*pos] != '\r') (*pos)++;
            break;
        }

        if (!hit_delim) {
            /* End of record */
            break;
        }
    }

    /* Consume newline */
    if (buf[*pos] == '\r') (*pos)++;
    if (buf[*pos] == '\n') (*pos)++;

    return col;
}

/* ── csv_load_ext: robust loader ────────────────────────────────────── */

Matrix* csv_load_ext(const char *path, const CSVOptions *opts) {
    if (!path) {
        cml_set_error(CML_ERROR_NULL_PTR, "csv_load_ext: NULL path");
        return NULL;
    }

    /* Defaults */
    int    has_header       = 0;
    char   delimiter        = ',';
    char   quote            = '"';
    double missing_val      = NAN;
    int    skip_empty_lines = 1;
    int    max_rows         = 0;
    int    max_cols         = 0;

    if (opts) {
        has_header       = opts->has_header;
        delimiter        = opts->delimiter  ? opts->delimiter  : ',';
        quote            = opts->quote;
        missing_val      = opts->missing_val;
        skip_empty_lines = opts->skip_empty_lines;
        max_rows         = opts->max_rows;
        max_cols         = opts->max_cols;
    }

    if (delimiter == 0) {
        delimiter = csv_detect_delimiter(path);
    }

    free_headers();

    char *buf = read_file_contents(path);
    if (!buf) return NULL;

    /* Two-pass: first pass to count rows/cols, second pass to fill data.
       Simpler: collect all values in a dynamic array. */
    size_t capacity = INITIAL_CAPACITY;
    double *values = malloc(capacity * sizeof(double));
    if (!values) {
        free(buf);
        cml_set_error(CML_ERROR_MEMORY, "csv_load_ext: allocation failed");
        return NULL;
    }

    size_t total_values = 0;
    size_t rows = 0;
    size_t cols = 0;  /* determined from first data row */
    size_t pos = 0;

    while (buf[pos]) {
        /* Skip whitespace before record */
        while (buf[pos] && (buf[pos] == ' ' || buf[pos] == '\t')) pos++;
        if (buf[pos] == '\0') break;

        /* Check for comment or empty line */
        if (buf[pos] == '#' || (buf[pos] == '/' && buf[pos + 1] == '/')) {
            while (buf[pos] && buf[pos] != '\n') pos++;
            if (buf[pos] == '\n') pos++;
            continue;
        }
        if (buf[pos] == '\n' || buf[pos] == '\r') {
            if (buf[pos] == '\r') pos++;
            if (buf[pos] == '\n') pos++;
            continue;
        }

        /* Parse fields of this record */
        char field[MAX_FIELD_LEN];
        size_t row_cols = 0;

        while (1) {
            if (buf[pos] == '\0' || buf[pos] == '\n' || buf[pos] == '\r') {
                /* If this is the very first char after a delim, it's a trailing
                   empty field. But if row_cols == 0, it's an empty line. */
                break;
            }

            int hit_delim = parse_field(buf, &pos, field, sizeof(field), delimiter, quote);

            /* Header row? */
            if (has_header && rows == 0) {
                if ((int)row_cols < MAX_HEADERS) {
                    g_header_storage[row_cols] = strdup(field);
                    g_header_ptrs[row_cols] = g_header_storage[row_cols];
                }
            } else {
                /* Parse value */
                double val = missing_val;
                if (field[0] != '\0') {
                    char *end;
                    val = strtod(field, &end);
                    if (end == field) {
                        val = missing_val; /* not a number */
                    }
                }

                if (total_values >= capacity) {
                    capacity *= 2;
                    double *nv = realloc(values, capacity * sizeof(double));
                    if (!nv) {
                        free(values);
                        free(buf);
                        free_headers();
                        cml_set_error(CML_ERROR_MEMORY, "csv_load_ext: allocation failed");
                        return NULL;
                    }
                    values = nv;
                }
                values[total_values++] = val;
            }

            row_cols++;

            if (max_cols > 0 && (int)row_cols >= max_cols) {
                /* skip rest of line */
                while (buf[pos] && buf[pos] != '\n' && buf[pos] != '\r') pos++;
                break;
            }

            if (!hit_delim) break;
        }

        /* Consume newline */
        if (buf[pos] == '\r') pos++;
        if (buf[pos] == '\n') pos++;

        if (row_cols == 0) continue; /* empty record */

        if (has_header && rows == 0) {
            /* This was the header row */
            g_header_count = (int)row_cols;
            cols = row_cols;
            rows = 0; /* header doesn't count as data row */
            continue;
        }

        /* Set column count from first data row */
        if (cols == 0) {
            cols = row_cols;
        }

        /* Pad or truncate to match cols */
        if (row_cols < cols) {
            /* Pad with missing_val */
            for (size_t pc = row_cols; pc < cols; pc++) {
                if (total_values >= capacity) {
                    capacity *= 2;
                    double *nv = realloc(values, capacity * sizeof(double));
                    if (!nv) {
                        free(values);
                        free(buf);
                        free_headers();
                        cml_set_error(CML_ERROR_MEMORY, "csv_load_ext: allocation failed");
                        return NULL;
                    }
                    values = nv;
                }
                values[total_values++] = missing_val;
            }
        } else if (row_cols > cols) {
            /* Truncate: remove excess values from the end */
            total_values -= (row_cols - cols);
        }

        rows++;

        if (max_rows > 0 && (int)rows >= max_rows) break;
    }

    free(buf);

    if (rows == 0 || cols == 0) {
        free(values);
        cml_set_error(CML_ERROR_PARSE, "csv_load_ext: no data found in '%s'", path);
        return NULL;
    }

    Matrix *m = matrix_alloc(rows, cols);
    if (!m) {
        free(values);
        return NULL;
    }
    memcpy(m->data, values, rows * cols * sizeof(double));
    free(values);

    return m;
}

/* ── Original csv_load (backward compatible) ────────────────────────── */

static int count_columns_simple(const char *line) {
    int count = 1;
    while (*line) {
        if (*line == ',') count++;
        line++;
    }
    return count;
}

Matrix* csv_load(const char *filename, int has_header) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        cml_set_error(CML_ERROR_FILE_IO, "csv_load: cannot open file '%s'", filename);
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    size_t capacity = INITIAL_CAPACITY;
    size_t rows = 0;
    size_t cols = 0;

    double *values = malloc(capacity * sizeof(double));
    if (!values) {
        fclose(fp);
        cml_set_error(CML_ERROR_MEMORY, "csv_load: memory allocation failed");
        return NULL;
    }

    /* Skip header if present */
    if (has_header) {
        if (!fgets(line, sizeof(line), fp)) {
            free(values);
            fclose(fp);
            cml_set_error(CML_ERROR_PARSE, "csv_load: file is empty");
            return NULL;
        }
    }

    /* Read data lines */
    while (fgets(line, sizeof(line), fp)) {
        if (is_empty_line(line)) continue;

        if (cols == 0) {
            cols = (size_t)count_columns_simple(line);
        }

        char *ptr = line;
        char *end;
        size_t col = 0;

        while (*ptr && col < cols) {
            while (*ptr && isspace((unsigned char)*ptr)) ptr++;

            double val = strtod(ptr, &end);

            if (ptr == end) {
                cml_set_error(CML_ERROR_PARSE, "csv_load: parse error at row %zu, col %zu",
                              rows + 1, col + 1);
                free(values);
                fclose(fp);
                return NULL;
            }

            if (rows * cols + col >= capacity) {
                capacity *= 2;
                double *nv = realloc(values, capacity * sizeof(double));
                if (!nv) {
                    free(values);
                    fclose(fp);
                    cml_set_error(CML_ERROR_MEMORY, "csv_load: memory allocation failed");
                    return NULL;
                }
                values = nv;
            }

            values[rows * cols + col] = val;
            col++;

            ptr = end;
            while (*ptr && (isspace((unsigned char)*ptr) || *ptr == ',')) {
                if (*ptr == ',') { ptr++; break; }
                ptr++;
            }
        }

        rows++;
    }

    fclose(fp);

    if (rows == 0 || cols == 0) {
        free(values);
        cml_set_error(CML_ERROR_PARSE, "csv_load: no data found");
        return NULL;
    }

    Matrix *m = matrix_alloc(rows, cols);
    if (!m) {
        free(values);
        return NULL;
    }

    memcpy(m->data, values, rows * cols * sizeof(double));
    free(values);
    return m;
}

/* ── csv_save ───────────────────────────────────────────────────────── */

int csv_save(const Matrix *m, const char *filename) {
    if (!m || !filename) return -1;

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        cml_set_error(CML_ERROR_FILE_IO, "csv_save: cannot open file '%s' for writing", filename);
        return -1;
    }

    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            fprintf(fp, "%.6f", m->data[i * m->cols + j]);
            if (j < m->cols - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 0;
}
