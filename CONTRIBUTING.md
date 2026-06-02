# Contributing to TinyCML

Thanks for your interest in contributing to TinyCML. This document covers everything you need to get started.

## Code Style

- **Language:** C11 (ISO/IEC 9899:2011). No compiler extensions.
- **Indentation:** 4 spaces. No tabs.
- **Naming:** `snake_case` for all identifiers. Prefix public names with `tinycml_` or the module name (e.g., `matrix_create`, `vector_dot`).
- **Headers:** Use `#ifndef` include guards named `TINYCML_<MODULE>_H`.
- **Types:** Use `size_t` for sizes and indices. Use `<stdint.h>` types where fixed widths matter.
- **Error handling:** Return error codes from enums. Never crash on invalid input.

Run `make lint` if available, or ensure your code passes `-Wall -Wextra -Werror -std=c11` cleanly.

## Adding a New Algorithm

Follow the header → implementation → test pattern:

1. **Header** — Add declarations to the appropriate header in `include/tinycml/`, or create a new one if the module is new. Public functions get Doxygen comments (see Documentation below).
2. **Implementation** — Add the `.c` file in `src/`. Include the project header and keep implementation details `static` where possible.
3. **Test** — Add a corresponding test file in `tests/`. Every public function needs at least one test case covering normal operation and edge cases (null pointers, zero-length inputs, overflow conditions).

Example structure:

```
include/tinycml/cluster.h   ← public API with Doxygen
src/cluster.c               ← implementation
tests/test_cluster.c        ← test suite
```

## Building

```bash
make build          # compile the library
make test           # build and run all tests
make clean          # remove build artifacts
```

The project has zero external dependencies. It should compile with any C11-conforming compiler. CI tests against both GCC and Clang with AddressSanitizer enabled.

## PR Process

1. **Fork** the repository on GitHub.
2. **Create a branch** from `main` with a descriptive name (e.g., `feature/kmeans-clustering`, `fix/matrix-transpose-bug`).
3. **Commit** your changes with clear, concise commit messages.
4. **Open a pull request** against the `main` branch of `sametyilmaztemel/tinycml`.

Keep PRs focused on a single concern. Large refactorings and feature additions should be separate PRs.

## Commit Messages

- Use **imperative mood**: "Add k-means clustering" not "Added k-means clustering".
- Keep the subject line under 72 characters.
- Separate subject from body with a blank line if further explanation is needed.
- Reference issue numbers when applicable: "Fix memory leak in matrix multiplication (#42)".

## Testing Requirements

- **Every new public function** must have at least one test.
- Tests must pass under AddressSanitizer (`make test` runs with ASAN in CI).
- Cover both the happy path and failure/edge-case paths.
- If you fix a bug, add a regression test that would have caught it.
- Run `make test` before opening your PR. All 127+ existing tests must continue to pass.

## Documentation

All public APIs must have Doxygen-style comments in the header file:

```c
/**
 * @brief Compute the dot product of two vectors.
 *
 * @param a First vector (array of doubles).
 * @param b Second vector (array of doubles).
 * @param n Length of both vectors.
 * @return Dot product, or NAN on error.
 */
double vector_dot(const double *a, const double *b, size_t n);
```

Document parameters with `@param`, return values with `@return`, and provide a brief description with `@brief`. Keep comments factual and concise.
