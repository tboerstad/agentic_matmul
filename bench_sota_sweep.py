"""NumPy (OpenBLAS) + SciPy dgemm over the SAME shape list as bench_sweep.mojo --full.

Prints per-shape peak GFLOPS so the BLAS libs can be lined up against the Mojo
dispatch / linalg numbers from `mojo bench_sweep.mojo --full`.
"""
import time
import numpy as np
from scipy.linalg import blas

DTYPE = np.float64
N_RUNS = 8
WARMUP = 3


def gflops(m, n, k, secs):
    return (2.0 * m * n * k) / (secs * 1e9)


def peak(fn):
    for _ in range(WARMUP):
        fn()
    best = 1e30
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def bench(label, m, n, k):
    A = np.random.randn(m, k).astype(DTYPE)
    B = np.random.randn(k, n).astype(DTYPE)
    Af = np.asfortranarray(A)
    Bf = np.asfortranarray(B)
    np_t = peak(lambda: np.matmul(A, B))
    sp_t = peak(lambda: blas.dgemm(1.0, Af, Bf))
    np_g = gflops(m, n, k, np_t)
    sp_g = gflops(m, n, k, sp_t)
    print(f"   {label:10s} M= {m:5d} N= {n:6d} K= {k:6d} | numpy {np_g:8.2f} | scipy {sp_g:8.2f}")


MS = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512]


def run_set(title, rows):
    print(f"\n=== {title} ===\n")
    for label, m, n, k in rows:
        bench(label, m, n, k)


def main():
    run_set("square sweep (M = N = K)", [("sq", m, m, m) for m in MS])
    run_set("wide-N sweep: N=8192 K=2048", [("wide", m, 8192, 2048) for m in MS])
    run_set("tall-K sweep: N=2048 K=8192", [("tall", m, 2048, 8192) for m in MS])
    run_set("Qwen up/gate proj (wide-N): N=11008 K=2048", [("up", m, 11008, 2048) for m in MS])
    run_set("Qwen down proj (tall-K): N=2048 K=11008", [("down", m, 2048, 11008) for m in MS])

    general = [
        ("h4k-m128", 128, 4096, 4096), ("h4k-m512", 512, 4096, 4096),
        ("ffn-up8k", 512, 8192, 2048), ("ffn-dn8k", 512, 2048, 8192),
        ("N4000", 512, 4000, 2048), ("N5504", 512, 5504, 2048),
        ("N-odd", 512, 11007, 2048), ("K-odd", 512, 2048, 2047),
        ("M-rem", 100, 11008, 2048), ("M-prime", 333, 4096, 4096),
        ("K128", 512, 2048, 128), ("K256w", 512, 11008, 256),
    ]
    run_set("general (M,N,K) grid", general)

    thin = [
        ("tn-N16", 512, 16, 512), ("tn-N32", 512, 32, 512),
        ("tn-N48", 512, 48, 512), ("tn-N64", 512, 64, 512),
        ("tn-tallK", 2048, 16, 2048), ("tn-bigM", 8192, 16, 512),
        ("tn-N8", 512, 8, 512), ("tn-mid", 128, 32, 2048),
    ]
    run_set("thin-N tall-M grid", thin)

    box = [
        ("sb-sq96", 96, 96, 96), ("sb-sq128", 128, 128, 128),
        ("sb-sq192", 192, 192, 192), ("sb-sq256", 256, 256, 256),
        ("sb-512x128", 512, 128, 512), ("sb-256x128", 256, 128, 512),
        ("sb-512x256", 512, 256, 256), ("sb!sq320", 320, 320, 320),
        ("sb!sq352", 352, 352, 352), ("sb!128x256", 128, 256, 256),
    ]
    run_set("small-box grid", box)


if __name__ == "__main__":
    main()
