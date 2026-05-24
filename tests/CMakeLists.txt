# Tests CMakeLists.txt

# Test harness is header-only

# Matrix tests
add_executable(test_matrix test_matrix.c)
target_include_directories(test_matrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_matrix tinycml)
add_test(NAME test_matrix COMMAND test_matrix)

# Linear regression tests
add_executable(test_linreg test_linreg.c)
target_include_directories(test_linreg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_linreg tinycml)
add_test(NAME test_linreg COMMAND test_linreg)
