# Package Manager Configuration for tinycml

This directory contains packaging templates for installing **tinycml** through popular C/C++ package managers. These are reference templates — you may need to adjust URLs, checksums, and minor details when submitting to the actual registries.

---

## Homebrew (macOS / Linux)

**File:** `homebrew/tinycml.rb`

### Quick install (local formula)

```bash
brew install --formula ./packaging/homebrew/tinycml.rb
```

### Submitting to a tap

1. Create a Homebrew tap repository (e.g. `homebrew-tinycml`)
2. Place `tinycml.rb` in the `Formula/` directory
3. Users install via:
   ```bash
   brew tap username/tinycml
   brew install tinycml
   ```

### Notes
- Update the `url` and `sha256` to point to the actual GitHub release tarball
- Run `shasum -a 256 <tarball>` to compute the correct checksum

---

## vcpkg

**Files:** `vcpkg/vcpkg.json` and `vcpkg/portfile.cmake`

### Quick install (local port overlay)

1. Copy or symlink this directory into your vcpkg ports tree:
   ```bash
   cp -r packaging/vcpkg ~/.vcpkg/ports/tinycml
   ```
2. Install:
   ```bash
   vcpkg install tinycml
   ```

### Submitting to vcpkg

1. Fork [microsoft/vcpkg](https://github.com/microsoft/vcpkg)
2. Copy the `vcpkg.json` and `portfile.cmake` into `ports/tinycml/`
3. Run `vcpkg x-add-version tinycml` to register the version
4. Open a pull request against the vcpkg repository

### Notes
- Update the `SHA512` placeholder in `portfile.cmake` with the actual hash
- The portfile currently installs both static and shared variants

---

## Conan

**File:** `conan/conanfile.py`

### Quick install (local recipe)

```bash
# Create the conan package
conan create packaging/conan/ --name=tinycml --version=0.1.0

# Use in your project
conan install --requires=tinycml/0.1.0
```

### Using with CMake

```cmake
find_package(tinycml REQUIRED)
target_link_libraries(myapp PRIVATE tinycml::tinycml)
```

### Submitting to Conan Center

1. Fork [conan-io/conan-center-index](https://github.com/conan-io/conan-center-index)
2. Add `conanfile.py` under `recipes/tinycml/all/`
3. Add a `conandata.yml` with the source URL and checksum
4. Open a pull request

### Notes
- Supports both static and shared builds via the `shared` option
- Automatically links `-lm` on Linux/FreeBSD
- Exposes the `tinycml::tinycml` CMake target name

---

## General Notes

- All templates use placeholder GitHub URLs (`https://github.com/username/tinycml`) — replace with the actual repository location
- Checksums are placeholders (`000...`) and must be replaced with real values before publishing
- The library version is set to **0.1.0** across all templates
- tinycml has **no runtime dependencies** beyond the C standard library and libm
