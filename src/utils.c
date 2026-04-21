/**
 * @file utils.c
 * @brief Implementation of utility functions
 */

#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* xoshiro256** state */
static uint64_t rng_state[4];
static int rng_seeded = 0;

static uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro256ss(void) {
    const uint64_t result = rotl(rng_state[1] * 5, 7) * 9;
    const uint64_t t = rng_state[1] << 17;
    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = rotl(rng_state[3], 45);
    return result;
}

/* SplitMix64 for seeding from a single value */
static void rng_seed_from_uint(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_state[i] = z ^ (z >> 31);
    }
    rng_seeded = 1;
}

void rand_seed(unsigned int seed) {
    rng_seed_from_uint((uint64_t)seed);
}

double rand_uniform(void) {
    if (!rng_seeded) {
        rng_seed_from_uint((uint64_t)time(NULL));
    }
    return (double)(xoshiro256ss() >> 11) / (double)(1ULL << 53);
}

double rand_uniform_range(double min, double max) {
    return min + rand_uniform() * (max - min);
}

double rand_normal(void) {
    /* Box-Muller transform */
    static int have_spare = 0;
    static double spare;

    if (have_spare) {
        have_spare = 0;
        return spare;
    }

    double u, v, s;
    do {
        u = rand_uniform() * 2.0 - 1.0;
        v = rand_uniform() * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    have_spare = 1;

    return u * s;
}

double rand_normal_params(double mean, double std) {
    return mean + std * rand_normal();
}

double mean(const double *data, size_t n) {
    if (n == 0 || !data) {
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }

    return sum / (double)n;
}

double variance(const double *data, size_t n) {
    if (n <= 1 || !data) {
        return 0.0;
    }

    double m = mean(data, n);
    double sum = 0.0;

    for (size_t i = 0; i < n; i++) {
        double diff = data[i] - m;
        sum += diff * diff;
    }

    return sum / (double)(n - 1);  /* Bessel's correction */
}

double std_dev(const double *data, size_t n) {
    return sqrt(variance(data, n));
}

void shuffle_indices(size_t *indices, size_t n) {
    if (n <= 1 || !indices) {
        return;
    }

    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(rand_uniform() * (i + 1));
        size_t temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

char* cml_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char *dup = malloc(len + 1);
    if (!dup) return NULL;
    memcpy(dup, s, len + 1);
    return dup;
}
