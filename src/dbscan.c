/**
 * @file dbscan.c
 * @brief DBSCAN density-based clustering implementation
 *
 * Brute-force O(n^2) approach. Uses BFS for cluster expansion.
 */

#include "dbscan.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* --- Euclidean distance squared between two rows --- */
static double dist_sq(const Matrix *X, size_t i, size_t j) {
    size_t F = X->cols;
    double sum = 0.0;
    for (size_t d = 0; d < F; d++) {
        double diff = X->data[i * F + d] - X->data[j * F + d];
        sum += diff * diff;
    }
    return sum;
}

/* --- Find neighbors within epsilon, return count and fill indices --- */
static int* find_neighbors(const Matrix *X, size_t point, double eps_sq, int *out_count) {
    size_t N = X->rows;
    int *neighbors = malloc(N * sizeof(int));
    if (!neighbors) { *out_count = 0; return NULL; }

    int count = 0;
    for (size_t j = 0; j < N; j++) {
        if (dist_sq(X, point, j) <= eps_sq) {
            neighbors[count++] = (int)j;
        }
    }
    *out_count = count;
    return neighbors;
}

int* dbscan_fit(const DBSCAN *model, const Matrix *X) {
    if (!model || !X) return NULL;

    size_t N = X->rows;
    double eps_sq = model->epsilon * model->epsilon;

    int *labels = malloc(N * sizeof(int));
    if (!labels) return NULL;

    /* Initialize all labels to -1 (unvisited/noise) */
    for (size_t i = 0; i < N; i++) labels[i] = -1;

    /* Determine core points */
    int *is_core = calloc(N, sizeof(int));
    if (!is_core) { free(labels); return NULL; }

    /* For each point, compute neighbor count to determine core status */
    int **neighbor_lists = malloc(N * sizeof(int*));
    int  *neighbor_counts = malloc(N * sizeof(int));
    if (!neighbor_lists || !neighbor_counts) {
        free(labels); free(is_core);
        if (neighbor_lists) free(neighbor_lists);
        if (neighbor_counts) free(neighbor_counts);
        return NULL;
    }

    for (size_t i = 0; i < N; i++) {
        neighbor_lists[i] = find_neighbors(X, i, eps_sq, &neighbor_counts[i]);
        if (neighbor_counts[i] >= model->min_samples) {
            is_core[i] = 1;
        }
    }

    /* BFS to expand clusters from core points */
    int cluster_id = 0;

    /* BFS queue */
    int *queue = malloc(N * sizeof(int));
    if (!queue) {
        free(labels); free(is_core);
        for (size_t i = 0; i < N; i++) free(neighbor_lists[i]);
        free(neighbor_lists); free(neighbor_counts);
        return NULL;
    }

    for (size_t i = 0; i < N; i++) {
        if (labels[i] != -1) continue;      /* already visited */
        if (!is_core[i]) continue;           /* not a core point, stays noise for now */

        /* Start a new cluster */
        labels[i] = cluster_id;

        int front = 0, back = 0;
        queue[back++] = (int)i;

        while (front < back) {
            int cur = queue[front++];

            /* For each neighbor of current point */
            for (int n = 0; n < neighbor_counts[cur]; n++) {
                int nb = neighbor_lists[cur][n];

                if (labels[nb] == -1) {
                    /* Was noise/unvisited — assign to cluster */
                    labels[nb] = cluster_id;
                    if (is_core[nb]) {
                        queue[back++] = nb;
                    }
                } else if (labels[nb] == -2) {
                    /* Shouldn't happen, but be safe */
                    labels[nb] = cluster_id;
                }
                /* If already labeled (part of current cluster), skip */
            }
        }

        cluster_id++;
    }

    /* Cleanup */
    free(queue);
    for (size_t i = 0; i < N; i++) free(neighbor_lists[i]);
    free(neighbor_lists);
    free(neighbor_counts);
    free(is_core);

    return labels;
}

int dbscan_count_clusters(const int *labels, int n) {
    if (!labels || n <= 0) return 0;

    int max_label = -1;
    for (int i = 0; i < n; i++) {
        if (labels[i] > max_label) max_label = labels[i];
    }
    /* Clusters are numbered 0..max_label, so count = max_label + 1 */
    return (max_label >= 0) ? max_label + 1 : 0;
}

DBSCAN* dbscan_create(void) {
    DBSCAN *model = calloc(1, sizeof(DBSCAN));
    if (!model) return NULL;
    model->epsilon = 0.5;
    model->min_samples = 5;
    return model;
}

void dbscan_free(DBSCAN *model) {
    if (model) free(model);
}
