#!/usr/bin/env python3
"""Post-process tinycml.h amalgamation: redirect malloc/free for embedded pool support."""
import re
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <tinycml.h path>")
    sys.exit(1)

output = sys.argv[1]

with open(output, 'r') as f:
    content = f.read()

impl_start = content.find('#ifdef CML_IMPLEMENTATION')
impl_end = content.rfind('#endif /* CML_IMPLEMENTATION */')
if impl_start < 0 or impl_end < 0:
    print("ERROR: could not find implementation section")
    sys.exit(1)

before = content[:impl_start]
impl = content[impl_start:impl_end + len('#endif /* CML_IMPLEMENTATION */')]
after = content[impl_end + len('#endif /* CML_IMPLEMENTATION */'):]

# Insert redirect comment block after #ifdef CML_IMPLEMENTATION
redirect_defs = """
/* === Allocation redirect (CML_USE_POOL support) === */
#ifdef CML_USE_POOL
/* cml_malloc etc. already defined via embed/cml_pool.h */
#else
/* Without pool: cml_malloc -> malloc, already defined above */
#endif

"""
impl = impl.replace('#ifdef CML_IMPLEMENTATION\n\n', '#ifdef CML_IMPLEMENTATION\n' + redirect_defs, 1)

# Replace standalone malloc/free/calloc/realloc calls with cml_ versions
# NOT struct member access (.free, ->free, _free)
replacements = [
    (r'(?<![.\->_a-zA-Z0-9])malloc\(', 'cml_malloc('),
    (r'(?<![.\->_a-zA-Z0-9])calloc\(', 'cml_calloc('),
    (r'(?<![.\->_a-zA-Z0-9])realloc\(', 'cml_realloc('),
    (r'(?<![.\->_a-zA-Z0-9])free\(', 'cml_free('),
]
for pattern, replacement in replacements:
    count = len(re.findall(pattern, impl))
    impl = re.sub(pattern, replacement, impl)
    if count:
        print(f"  {pattern}: {count} replacements")

content = before + impl + after

with open(output, 'w') as f:
    f.write(content)

print("Done: malloc/calloc/realloc/free -> cml_malloc/cml_calloc/cml_realloc/cml_free")
