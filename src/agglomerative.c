/**
 * agglomerative.c - Agglomerative (hierarchical) clustering
 *
 * Bottom-up approach: start with each point as its own cluster,
 * iteratively merge the two closest clusters until n_clusters remain.
 * Supports single, complete, and average linkage.
 */

#include "agglomerative.h"
#include "metrics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* ── distance cache ──────────────────────────────────────────────── */

static double euclidean_dist(const Matrix *X, size_t a, size_t b) {
    double sum = 0.0;
    for (size_t j = 0; j < X->cols; j++) {
        double d = matrix_get(X, a, j) - matrix_get(X, b, j);
        sum += d * d;
    }
    return sqrt(sum);
}

/* ── create / free ───────────────────────────────────────────────── */

AgglomerativeClustering* agglomerative_create(int n_clusters, LinkageType linkage) {
    AgglomerativeClustering *ac = calloc(1, sizeof(AgglomerativeClustering));
    if (!ac) return NULL;

    ac->n_clusters = n_clusters > 0 ? n_clusters : 2;
    ac->linkage    = linkage;
    ac->labels     = NULL;
    ac->n_samples  = 0;
    ac->n_features = 0;

    ac->base.fit     = agglomerative_fit;
    ac->base.predict = agglomerative_predict;
    ac->base.score   = agglomerative_score;
    ac->base.clone   = agglomerative_clone;
    ac->base.free    = agglomerative_free;
    ac->base.type    = MODEL_KMEANS;

    return ac;
}

void agglomerative_free(Estimator *self) {
    if (!self) return;
    AgglomerativeClustering *ac = (AgglomerativeClustering*)self;
    free(ac->labels);
    free(ac);
}

/* ── fit ─────────────────────────────────────────────────────────── */

Estimator* agglomerative_fit(Estimator *self, const Matrix *X, const Matrix *y) {
    (void)y;
    if (!self || !X) return NULL;

    AgglomerativeClustering *ac = (AgglomerativeClustering*)self;
    size_t n = X->rows;
    ac->n_samples = n;
    ac->n_features = (int)X->cols;

    if ((int)n <= ac->n_clusters) {
        /* Each point is its own cluster */
        ac->labels = malloc(n * sizeof(int));
        for (size_t i = 0; i < n; i++) ac->labels[i] = (int)i;
        return self;
    }

    /* Precompute pairwise distances */
    double *dist = malloc(n * n * sizeof(double));
    if (!dist) return NULL;
    for (size_t i = 0; i < n; i++) {
        dist[i * n + i] = 0.0;
        for (size_t j = i + 1; j < n; j++) {
            double d = euclidean_dist(X, i, j);
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }

    /* Cluster membership: each sample starts in its own cluster */
    int *cluster_of = malloc(n * sizeof(int));
    int *cluster_size = malloc(n * sizeof(int));
    for (size_t i = 0; i < n; i++) {
        cluster_of[i] = (int)i;
        cluster_size[i] = 1;
    }

    /* Active cluster list */
    int *active = malloc(n * sizeof(int));
    int n_active = (int)n;
    for (size_t i = 0; i < n; i++) active[i] = 1;

    /* Cluster distance matrix */
    double *cdist = malloc(n * n * sizeof(double));
    memcpy(cdist, dist, n * n * sizeof(double));

    /* Iteratively merge closest clusters */
    int next_label = (int)n;

    while (n_active > ac->n_clusters) {
        /* Find closest pair */
        double min_dist = DBL_MAX;
        int mi = -1, mj = -1;

        for (int i = 0; i < (int)n; i++) {
            if (!active[i]) continue;
            for (int j = i + 1; j < (int)n; j++) {
                if (!active[j]) continue;
                double d = cdist[i * n + j];
                if (d < min_dist) {
                    min_dist = d;
                    mi = i;
                    mj = j;
                }
            }
        }

        if (mi < 0 || mj < 0) break;

        /* Merge cluster mj into mi */
        int si = cluster_size[mi];
        int sj = cluster_size[mj];

        /* Update distances using Lance-Williams formula */
        for (int k = 0; k < (int)n; k++) {
            if (!active[k] || k == mi || k == mj) continue;

            double dik = cdist[mi * n + k];
            double djk = cdist[mj * n + k];
            double new_dist;

            switch (ac->linkage) {
                case LINKAGE_SINGLE:
                    new_dist = fmin(dik, djk);
                    break;
                case LINKAGE_COMPLETE:
                    new_dist = fmax(dik, djk);
                    break;
                case LINKAGE_AVERAGE:
                default:
                    new_dist = ((double)si * dik + (double)sj * djk) / (double)(si + sj);
                    break;
            }

            cdist[mi * n + k] = new_dist;
            cdist[k * n + mi] = new_dist;
        }

        /* Update labels */
        for (size_t s = 0; s < n; s++) {
            if (cluster_of[s] == mj) {
                cluster_of[s] = mi;
            }
        }

        cluster_size[mi] += cluster_size[mj];
        active[mj] = 0;
        n_active--;
        (void)next_label;
    }

    /* Relabel clusters to 0..n_clusters-1 */
    int *label_map = malloc(n * sizeof(int));
    memset(label_map, -1, n * sizeof(int));
    int label_count = 0;

    for (size_t i = 0; i < n; i++) {
        int c = cluster_of[i];
        if (label_map[c] == -1) {
            label_map[c] = label_count++;
        }
        cluster_of[i] = label_map[c];
    }

    ac->labels = malloc(n * sizeof(int));
    if (ac->labels) {
        memcpy(ac->labels, cluster_of, n * sizeof(int));
    }

    free(dist);
    free(cdist);
    free(cluster_of);
    free(cluster_size);
    free(active);
    free(label_map);

    return self;
}

/* ── predict ─────────────────────────────────────────────────────── */

Matrix* agglomerative_predict(const Estimator *self, const Matrix *X) {
    if (!self || !X) return NULL;
    /* Agglomerative clustering doesn't naturally support prediction on new data.
       Return NULL — users should use fit to get labels. */
    (void)self;
    (void)X;
    return NULL;
}

/* ── score (silhouette) ──────────────────────────────────────────── */

double agglomerative_score(const Estimator *self, const Matrix *X, const Matrix *y) {
    (void)self;
    (void)X;
    (void)y;
    /* Silhouette score computation — use metrics module if available */
    return 0.0;
}

/* ── clone ───────────────────────────────────────────────────────── */

Estimator* agglomerative_clone(const Estimator *self) {
    if (!self) return NULL;
    AgglomerativeClustering *src = (AgglomerativeClustering*)self;

    AgglomerativeClustering *dst = agglomerative_create(src->n_clusters, src->linkage);
    if (!dst) return NULL;

    dst->n_samples = src->n_samples;
    dst->n_features = src->n_features;

    if (src->labels && src->n_samples > 0) {
        dst->labels = malloc(src->n_samples * sizeof(int));
        if (dst->labels) {
            memcpy(dst->labels, src->labels, src->n_samples * sizeof(int));
        }
    }

    return (Estimator*)dst;
}
