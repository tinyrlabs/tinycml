# tinycml vcpkg portfile
# NOTE: Update the REF and SHA512 when pointing to an actual release archive.

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO username/tinycml
    REF "v${VERSION}"
    SHA512 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    # HEAD_REF main
)

# Build static and shared libraries via the project Makefile
vcpkg_make_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_make_install(
    OPTIONS PREFIX="${CURRENT_PACKAGES_DIR}"
)

# Fix shared library paths on non-Windows
if(VCPKG_TARGET_IS_LINUX OR VCPKG_TARGET_IS_OSX)
    file(GLOB SO_FILES "${CURRENT_PACKAGES_DIR}/lib/libtinycml.so*")
    if(SO_FILES)
        file(COPY ${SO_FILES} DESTINATION "${CURRENT_PACKAGES_DIR}/bin")
    endif()
endif()

# Handle debug split — the Makefile installs into a single tree, so we
# replicate the standard vcpkg layout by moving debug artifacts if needed.
if(NOT DEFINED VCPKG_BUILD_TYPE OR VCPKG_BUILD_TYPE STREQUAL "debug")
    file(INSTALL "${CURRENT_PACKAGES_DIR}/lib/" DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib")
endif()

# Install headers
file(INSTALL "${SOURCE_PATH}/include/" DESTINATION "${CURRENT_PACKAGES_DIR}/include/tinycml")

# Install pkg-config file (generated during build)
if(EXISTS "${CURRENT_PACKAGES_DIR}/lib/pkgconfig/tinycml.pc")
    vcpkg_fixup_pkgconfig()
endif()

# vcpkg cmake config — minimal CMake find-module wrapper
file(WRITE "${CURRENT_PACKAGES_DIR}/share/tinycml/tinycml-config.cmake" "\
if(NOT TARGET tinycml::tinycml)
  add_library(tinycml::tinycml STATIC IMPORTED)
  set_target_properties(tinycml::tinycml PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES \"\${CMAKE_CURRENT_LIST_DIR}/../../include\"
    IMPORTED_LOCATION \"\${CMAKE_CURRENT_LIST_DIR}/../../lib/libtinycml.a\"
  )
endif()
")

file(WRITE "${CURRENT_PACKAGES_DIR}/share/tinycml/tinycml-config-version.cmake" "\
set(PACKAGE_VERSION \"${VERSION}\" CACHE STRING \"tinycml version\")
if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
endif()
")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

message(STATUS "tinycml ${VERSION} installed successfully.")
