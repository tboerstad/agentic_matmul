// Empirical memory SOL: read bandwidth at DRAM / L3 / L2 footprints,
// plus a copy (read+write) stream. AVX-512 nontemporal-friendly read loop.
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__attribute__((noinline))
static double read_sweep(const double *buf, size_t n, int reps) {
    __m512d acc0 = _mm512_setzero_pd(), acc1 = _mm512_setzero_pd();
    __m512d acc2 = _mm512_setzero_pd(), acc3 = _mm512_setzero_pd();
    for (int r = 0; r < reps; r++) {
        for (size_t i = 0; i + 32 <= n; i += 32) {
            acc0 = _mm512_add_pd(acc0, _mm512_load_pd(buf + i));
            acc1 = _mm512_add_pd(acc1, _mm512_load_pd(buf + i + 8));
            acc2 = _mm512_add_pd(acc2, _mm512_load_pd(buf + i + 16));
            acc3 = _mm512_add_pd(acc3, _mm512_load_pd(buf + i + 24));
        }
    }
    acc0 = _mm512_add_pd(_mm512_add_pd(acc0, acc1), _mm512_add_pd(acc2, acc3));
    return _mm512_reduce_add_pd(acc0);
}

typedef struct { size_t bytes; int reps; double gbs; int tid; } job_t;
static volatile double sink;

static void *worker(void *arg) {
    job_t *j = (job_t *)arg;
    size_t n = j->bytes / 8;
    double *buf = aligned_alloc(64, j->bytes);
    for (size_t i = 0; i < n; i++) buf[i] = (double)(i & 1023);
    sink = read_sweep(buf, n, 2); // warm
    double t0 = now();
    sink = read_sweep(buf, n, j->reps);
    double t1 = now();
    j->gbs = ((double)j->bytes * j->reps) / (t1 - t0) / 1e9;
    free(buf);
    return NULL;
}

static void run(const char *label, size_t bytes, int reps, int nthreads) {
    pthread_t th[64];
    job_t jobs[64];
    for (int i = 0; i < nthreads; i++) {
        jobs[i] = (job_t){bytes, reps, 0, i};
        pthread_create(&th[i], NULL, worker, &jobs[i]);
    }
    double total = 0;
    for (int i = 0; i < nthreads; i++) { pthread_join(th[i], NULL); total += jobs[i].gbs; }
    printf("%-28s threads=%d per-buf=%6.1f MB  TOTAL %7.1f GB/s\n",
           label, nthreads, bytes / 1e6, total);
}

int main(void) {
    // DRAM: 512 MB per thread buffer (past 260MB L3 when 4 threads; single thread
    // 512MB also mostly misses since L3 is shared 260MB — use 1GB to be sure)
    run("DRAM read (1 thread)", 1UL << 30, 3, 1);
    run("DRAM read (4 threads)", 512UL << 20, 3, 4);
    // L3-resident: 32 MB per thread (128MB total < 260MB L3)
    run("L3 read (1 thread)", 32UL << 20, 40, 1);
    run("L3 read (4 threads)", 32UL << 20, 40, 4);
    // L2-resident: 1 MB per thread (2MB L2 per core)
    run("L2 read (1 thread)", 1UL << 20, 2000, 1);
    run("L2 read (4 threads)", 1UL << 20, 2000, 4);
    // L1-resident: 32 KB
    run("L1 read (1 thread)", 32UL << 10, 200000, 1);
    return 0;
}
