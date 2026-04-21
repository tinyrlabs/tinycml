/**
 * @file utils.h
 * @brief Utility functions for ML library
 *
 * Provides random number generation and statistical functions.
 */

#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

/**
 * @brief Seed the random number generator
 * @param seed Random seed
 */
void rand_seed(unsigned int seed);

/**
 * @brief Generate uniform random number in [0, 1)
 * @return Random value
 */
double rand_uniform(void);

/**
 * @brief Generate uniform random number in [min, max)
 * @param min Minimum value
 * @param max Maximum value
 * @return Random value
 */
double rand_uniform_range(double min, double max);

/**
 * @brief Generate random number from standard normal distribution
 * @return Random value from N(0, 1)
 */
double rand_normal(void);

/**
 * @brief Generate random number from normal distribution
 * @param mean Mean of distribution
 * @param std Standard deviation
 * @return Random value from N(mean, std^2)
 */
double rand_normal_params(double mean, double std);

/**
 * @brief Compute mean of an array
 * @param data Array of values
 * @param n Number of elements
 * @return Mean value
 */
double mean(const double *data, size_t n);

/**
 * @brief Compute standard deviation of an array
 * @param data Array of values
 * @param n Number of elements
 * @return Standard deviation
 */
double std_dev(const double *data, size_t n);

/**
 * @brief Compute variance of an array
 * @param data Array of values
 * @param n Number of elements
 * @return Variance
 */
double variance(const double *data, size_t n);

/**
 * @brief Fisher-Yates shuffle for an array of indices
 * @param indices Array to shuffle
 * @param n Number of elements
 */
void shuffle_indices(size_t *indices, size_t n);

/**
 * @brief Portable strdup replacement for C11 compatibility
 * @param s String to duplicate
 * @return Newly allocated copy of s, or NULL on failure
 */
char* cml_strdup(const char *s);

#endif /* UTILS_H */
