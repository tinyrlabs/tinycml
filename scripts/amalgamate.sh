#!/usr/bin/env bash
# amalgamate.sh — Build single-header tinycml.h from include/*.h + src/*.c
#
# Usage:  bash scripts/amalgamate.sh
#         Outputs tinycml.h in the project root.
#
# Handles duplicate static function names by prefixing with module name.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT="$PROJECT_DIR/tinycml.h"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# ── Header dependency order (low → high) ──────────────────────────────
HEADERS=(
    matrix.h utils.h cml_error.h
    vector.h csv.h metrics.h estimator.h
    onehot_encoder.h cml_serialization.h
    kmeans.h dbscan.h preprocessing.h knn.h
    linear_regression.h logistic_regression.h ridge.h lasso.h svm.h
    naive_bayes.h decision_tree.h decomposition.h
    neural_network.h sgd.h validation.h feature_selection.h
    ensemble.h gradient_boosting.h isolation_forest.h agglomerative.h
    model_selection.h pipeline.h
    cml_simd.h
)

# ── Source file order (must match HEADERS order) ──────────────────────
SOURCES=(
    matrix.c utils.c cml_error.c
    vector.c csv.c metrics.c estimator.c
    onehot_encoder.c cml_serialization.c
    kmeans.c dbscan.c preprocessing.c knn.c
    linear_regression.c logistic_regression.c ridge.c lasso.c svm.c
    naive_bayes.c decision_tree.c decomposition.c
    neural_network.c sgd.c validation.c feature_selection.c
    ensemble.c gradient_boosting.c isolation_forest.c agglomerative.c
    model_selection.c pipeline.c
    cml_simd.c
)

# ── Duplicate static functions to rename ──────────────────────────────
# Format: "funcname:module" — the function in module.c gets prefixed
# euclidean_distance: in kmeans.c and knn.c (both static)
# sigmoid: in neural_network.c (static), logistic_regression.c (public — rename the static one)
# shuffle_indices: in sgd.c (static with seed param), utils.c (public without seed)
DUPLICATES=(
    "euclidean_distance:knn"
    "sigmoid:neural_network"
    "shuffle_indices:sgd"
)

# ── Helper: strip include-guard lines from a header ───────────────────
strip_header_guards() {
    local file="$1"
    sed -e '/^#ifndef [A-Z_]*_H/,+1{/^#ifndef /d;/^#define /d}' "$file" \
      | sed -e '/^#endif \/\* [A-Z_]*_H/,${/^#endif/d}' \
      | sed -e '/^#endif$/d' \
      | grep -v '^$' | head -n -0
    python3 -c "
import sys
lines = open('$file').readlines()
skip_guard = 0
result = []
for i, line in enumerate(lines):
    if skip_guard == 0 and line.strip().startswith('#ifndef') and '_H' in line:
        skip_guard = 1
        continue
    if skip_guard == 1 and line.strip().startswith('#define') and '_H' in line:
        skip_guard = 2
        continue
    result.append(line)
while result and result[-1].strip() in ('', '#endif', '#endif /*'):
    result.pop()
sys.stdout.write(''.join(result))
"
}

# Simpler: use python for header processing
process_header() {
    python3 -c "
import sys, re
lines = open('$1').readlines()
out = []
guard_removed = False
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    if not guard_removed and stripped.startswith('#ifndef') and re.search(r'_H\b', stripped):
        guard_removed = True
        i += 1
        continue
    if guard_removed and stripped.startswith('#define') and re.search(r'_H\b', stripped):
        i += 1
        continue
    if re.match(r'#include\s+\".*\.h\"', stripped):
        comment_start = line.find('//')
        if comment_start >= 0:
            out.append(line[comment_start:])
        i += 1
        continue
    out.append(line)
    i += 1
while out and out[-1].strip() == '':
    out.pop()
if out and out[-1].strip().startswith('#endif'):
    out.pop()
sys.stdout.write(''.join(out))
"
}

# ── Helper: process .c file, rename duplicates, strip local includes ──
process_source() {
    local srcfile="$1"
    local basename="$(basename "$srcfile" .c)"
    
    local sed_expr=""
    for dup in "${DUPLICATES[@]}"; do
        local func="${dup%%:*}"
        local mod="${dup##*:}"
        if [ "$mod" = "$basename" ]; then
            sed_expr="${sed_expr}s/\\b${func}\\b/${mod}_${func}/g;"
        fi
    done
    
    if [ -n "$sed_expr" ]; then
        grep -v '#include ".*\.h"' "$srcfile" | sed -e "$sed_expr"
    else
        grep -v '#include ".*\.h"' "$srcfile"
    fi
}

# ── Generate tinycml.h ────────────────────────────────────────────────
{
    echo "/* tinycml v0.1.0 - single-header ML library */"
    echo "/* https://github.com/tinyrlabs/tinycml */"
    echo "/*"
    echo " * SINGLE-HEADER AMALGAMATION"
    echo " *"
    echo " * Usage:"
    echo " *   #define CML_IMPLEMENTATION"
    echo " *   #include \"tinycml.h\""
    echo " *"
    echo " * In exactly ONE translation unit, define CML_IMPLEMENTATION"
    echo " * before including this header to get the implementation."
    echo " *"
    echo " * Without CML_IMPLEMENTATION, you only get the declarations."
    echo " */"
    echo ""
    echo "#ifndef TINYCML_H"
    echo "#define TINYCML_H"
    echo ""
    echo "/* ============================================================"
    echo " * Embedded Configuration"
    echo " * ============================================================"
    echo " * Define these BEFORE including tinycml.h to use static pool:"
    echo " *"
    echo " *   #define CML_USE_POOL"
    echo " *   #define CML_POOL_SIZE 4096"
    echo " *   #include \"embed/cml_pool.h\""
    echo " *   #include \"tinycml.h\""
    echo " *"
    echo " * This redirects all internal allocations through a linear"
    echo " * bump allocator -- no malloc/free on your MCU heap."
    echo " * Call cml_pool_reset() between inference cycles."
    echo " * ============================================================ */"
    echo ""
    echo "#ifdef CML_USE_POOL"
    echo "    #define cml_malloc(s)  cml_pool_alloc(s)"
    echo "    #define cml_free(p)    cml_pool_free(p)"
    echo "    #define cml_calloc(n,s) cml_pool_calloc(n,s)"
    echo "    #define cml_realloc(p,s) cml_pool_realloc(p,s)"
    echo "#else"
    echo "    #include <stdlib.h>"
    echo "    #define cml_malloc(s)  malloc(s)"
    echo "    #define cml_free(p)    free(p)"
    echo "    #define cml_calloc(n,s) calloc(n,s)"
    echo "    #define cml_realloc(p,s) realloc(p,s)"
    echo "#endif"
    echo ""
    echo "/* === HEADERS === */"
    echo ""

    for h in "${HEADERS[@]}"; do
        echo "/* --- $h --- */"
        process_header "$PROJECT_DIR/include/$h"
        echo ""
        echo ""
    done

    echo "#ifdef CML_IMPLEMENTATION"
    echo ""
    echo "/* === IMPLEMENTATION === */"
    echo ""

    for s in "${SOURCES[@]}"; do
        echo "/* --- $s --- */"
        process_source "$PROJECT_DIR/src/$s"
        echo ""
        echo ""
    done

    echo "#endif /* CML_IMPLEMENTATION */"
    echo "#endif /* TINYCML_H */"

} > "$OUTPUT"

# ── Post-processing: redirect malloc/free to cml_malloc/cml_free ──────
python3 "$SCRIPT_DIR/amalgamate_post.py" "$OUTPUT"

# ── Report ────────────────────────────────────────────────────────────
LINES=$(wc -l < "$OUTPUT")
SIZE=$(wc -c < "$OUTPUT")
echo "Generated $OUTPUT  (${LINES} lines, ${SIZE} bytes)"
