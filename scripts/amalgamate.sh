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
    vector.h csv.h metrics.h kmeans.h dbscan.h preprocessing.h knn.h onehot_encoder.h estimator.h
    cml_serialization.h linear_regression.h logistic_regression.h ridge.h lasso.h svm.h naive_bayes.h decision_tree.h decomposition.h neural_network.h validation.h feature_selection.h
    ensemble.h model_selection.h pipeline.h
)

# ── Source file order ─────────────────────────────────────────────────
SOURCES=(
    matrix.c utils.c cml_error.c
    vector.c csv.c metrics.c kmeans.c dbscan.c preprocessing.c knn.c onehot_encoder.c estimator.c
    cml_serialization.c linear_regression.c logistic_regression.c ridge.c lasso.c svm.c naive_bayes.c decision_tree.c decomposition.c neural_network.c validation.c feature_selection.c
    ensemble.c model_selection.c pipeline.c
)

# ── Duplicate static functions to rename ──────────────────────────────
# Format: "funcname:module" — the function in module.c gets prefixed
# euclidean_distance: in kmeans.c and knn.c (both static)
# sigmoid: in neural_network.c (static), logistic_regression.c (public — rename the static one)
# compute_mean/compute_variance: only in feature_selection.c (static, but unused warning — keep as-is)
DUPLICATES=(
    "euclidean_distance:knn"
    "sigmoid:neural_network"
)

# ── Helper: strip include-guard lines from a header ───────────────────
strip_header_guards() {
    local file="$1"
    sed -e '/^#ifndef [A-Z_]*_H/,+1{/^#ifndef /d;/^#define /d}' "$file" \
      | sed -e '/^#endif \/\* [A-Z_]*_H/,${/^#endif/d}' \
      | sed -e '/^#endif$/d' \
      | grep -v '^$' | head -n -0
    # Simpler approach: just remove the first #ifndef/#define pair and last #endif
    python3 -c "
import sys
lines = open('$file').readlines()
# Find and remove first #ifndef GUARD and #define GUARD
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
# Remove last #endif
while result and result[-1].strip() in ('', '#endif', '#endif /*'):
    result.pop()
# But keep comments like #endif /* FOO_H */
# Re-add non-guard #endif
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
    # Skip #ifndef GUARD_H
    if not guard_removed and stripped.startswith('#ifndef') and re.search(r'_H\b', stripped):
        guard_removed = True
        i += 1
        continue
    # Skip #define GUARD_H  
    if guard_removed and stripped.startswith('#define') and re.search(r'_H\b', stripped):
        i += 1
        continue
    out.append(line)
    i += 1
# Remove trailing blank lines and last #endif
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
    
    # Build sed expressions for duplicate renaming
    local sed_expr=""
    for dup in "${DUPLICATES[@]}"; do
        local func="${dup%%:*}"
        local mod="${dup##*:}"
        if [ "$mod" = "$basename" ]; then
            sed_expr="${sed_expr}s/\\b${func}\\b/${mod}_${func}/g;"
        fi
    done
    
    # Process: strip local includes, rename duplicates
    if [ -n "$sed_expr" ]; then
        grep -v '#include ".*\.h"' "$srcfile" | sed -e "$sed_expr"
    else
        grep -v '#include ".*\.h"' "$srcfile"
    fi
}

# ── Also need to handle: public sigmoid in logistic_regression.c ──────
# It's not static, so in single-header mode it will conflict with the
# (now renamed) neural_network_sigmoid. That's fine — only one sigmoid remains public.

# ── Generate tinycml.h ────────────────────────────────────────────────
{
    echo "/* tinycml v0.1.0 - single-header ML library */"
    echo "/* https://github.com/sametyilmaztemel/tinycml */"
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

# ── Report ────────────────────────────────────────────────────────────
LINES=$(wc -l < "$OUTPUT")
SIZE=$(wc -c < "$OUTPUT")
echo "Generated $OUTPUT  (${LINES} lines, ${SIZE} bytes)"
