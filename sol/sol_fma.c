// Empirical compute SOL: AVX-512 FMA throughput, f64 and f32.
// 16 independent accumulator chains hide FMA latency (lat ~4-5, 2 pipes -> need >=10).
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define CHAINS 16
static long ITERS = 20000000L;

__attribute__((noinline))
static double fma_kernel_f64(void) {
    __m512d acc[CHAINS];
    __m512d a = _mm512_set1_pd(1.0000001);
    __m512d b = _mm512_set1_pd(0.9999999);
    for (int i = 0; i < CHAINS; i++) acc[i] = _mm512_set1_pd((double)i * 1e-30);
    for (long it = 0; it < ITERS; it++) {
        #pragma GCC unroll 16
        for (int i = 0; i < CHAINS; i++)
            acc[i] = _mm512_fmadd_pd(a, acc[i], b);
    }
    double sum = 0;
    for (int i = 0; i < CHAINS; i++) sum += ((double*)&acc[i])[0];
    return sum;
}

__attribute__((noinline))
static double fma_kernel_f32(void) {
    __m512 acc[CHAINS];
    __m512 a = _mm512_set1_ps(1.0000001f);
    __m512 b = _mm512_set1_ps(0.9999999f);
    for (int i = 0; i < CHAINS; i++) acc[i] = _mm512_set1_ps((float)i * 1e-30f);
    for (long it = 0; it < ITERS; it++) {
        #pragma GCC unroll 16
        for (int i = 0; i < CHAINS; i++)
            acc[i] = _mm512_fmadd_ps(a, acc[i], b);
    }
    double sum = 0;
    for (int i = 0; i < CHAINS; i++) sum += ((float*)&acc[i])[0];
    return sum;
}

static volatile double sink;
static int use_f32 = 0;

static void *worker(void *arg) {
    double *g = (double *)arg;
    double t0 = now();
    sink = use_f32 ? fma_kernel_f32() : fma_kernel_f64();
    double t1 = now();
    int lanes = use_f32 ? 16 : 8;
    double flops = (double)ITERS * CHAINS * lanes * 2.0;
    *g = flops / (t1 - t0) / 1e9;
    return NULL;
}

int main(int argc, char **argv) {
    int nthreads = argc > 1 ? atoi(argv[1]) : 1;
    if (argc > 2 && argv[2][0] == 'f') use_f32 = 1;
    pthread_t th[64];
    double gf[64];
    // warmup to reach steady turbo
    sink = use_f32 ? fma_kernel_f32() : fma_kernel_f64();
    for (int i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, worker, &gf[i]);
    double total = 0;
    for (int i = 0; i < nthreads; i++) { pthread_join(th[i], NULL); total += gf[i]; }
    printf("%s threads=%d per-thread:", use_f32 ? "f32" : "f64", nthreads);
    for (int i = 0; i < nthreads; i++) printf(" %.1f", gf[i]);
    printf("  TOTAL %.1f GFLOPS\n", total);
    return 0;
}
