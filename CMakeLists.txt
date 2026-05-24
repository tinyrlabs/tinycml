cmake_minimum_required(VERSION 3.10)
project(tinycml C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler warnings
add_compile_options(-Wall -Wextra -pedantic)

# Library sources
file(GLOB LIB_SOURCES "src/*.c")
add_library(tinycml STATIC ${LIB_SOURCES})
target_include_directories(tinycml PUBLIC include)
target_link_libraries(tinycml m)

# Examples
add_subdirectory(examples)

# Tests
enable_testing()
add_subdirectory(tests)
