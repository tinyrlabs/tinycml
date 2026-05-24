# Examples CMakeLists.txt

# Linear regression example
add_executable(linear_regression_example linear_regression_example.c)
target_link_libraries(linear_regression_example tinycml)

# Logistic regression example
add_executable(logistic_regression_example logistic_regression_example.c)
target_link_libraries(logistic_regression_example tinycml)

# k-NN example
add_executable(knn_example knn_example.c)
target_link_libraries(knn_example tinycml)

# k-Means example
add_executable(kmeans_example kmeans_example.c)
target_link_libraries(kmeans_example tinycml)
