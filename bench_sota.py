"""Benchmark SOTA matmul libraries on the Qwen 2.5 VL 3B shapes.

Measures:
  - NumPy (OpenBLAS backend) via np.matmul
  - SciPy BLAS dgemm (OpenBLAS backend) direct call
  - Intel MKL dgemm via ctypes

Shapes (from bench_matmul.mojo):
  - Decode:  M=1,   N=11008, K=2048  (single-token)
  - Prefill: M=96,  N=11008, K=2048  (batch)

All benchmarks use float64 to match the Mojo kernels.
"""

import time
import statistics
import os
import ctypes
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SHAPES = [
    ("decode  (1x11008x2048)", 1, 11008, 2048),
    ("prefill (96x11008x2048)", 96, 11008, 2048),
]
WARMUP = 5
ITERS = 20  # measurement iterations
DTYPE = np.float64


def gflops(m, n, k, secs):
    return (2.0 * m * n * k) / (secs * 1e9)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def bench(fn, warmup=WARMUP, iters=ITERS):
    """Run fn warmup+iters times, return list of durations in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def report(label, m, n, k, times):
    mean_t = statistics.mean(times)
    median_t = statistics.median(times)
    min_t = min(times)
    max_t = max(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0
    g = gflops(m, n, k, mean_t)
    g_peak = gflops(m, n, k, min_t)
    print(f"  {label:30s}  mean={mean_t*1e3:8.3f} ms  "
          f"median={median_t*1e3:8.3f} ms  "
          f"min={min_t*1e3:8.3f} ms  max={max_t*1e3:8.3f} ms  "
          f"std={std_t*1e3:8.3f} ms  "
          f"GFLOPS(mean)={g:7.2f}  GFLOPS(peak)={g_peak:7.2f}")
    return mean_t, g, min_t, g_peak


# ---------------------------------------------------------------------------
# NumPy benchmark (uses OpenBLAS bundled with numpy/scipy)
# ---------------------------------------------------------------------------
def bench_numpy(m, n, k):
    A = np.random.randn(m, k).astype(DTYPE)
    B = np.random.randn(k, n).astype(DTYPE)
    def fn():
        np.matmul(A, B)
    return bench(fn)


# ---------------------------------------------------------------------------
# SciPy dgemm (direct BLAS call via scipy)
# ---------------------------------------------------------------------------
def bench_scipy_dgemm(m, n, k):
    from scipy.linalg import blas
    A = np.asfortranarray(np.random.randn(m, k).astype(DTYPE))
    B = np.asfortranarray(np.random.randn(k, n).astype(DTYPE))
    def fn():
        blas.dgemm(1.0, A, B)
    return bench(fn)


# ---------------------------------------------------------------------------
# Intel MKL dgemm via ctypes
# ---------------------------------------------------------------------------
def load_mkl():
    """Try to load MKL runtime library."""
    search_paths = [
        "libmkl_rt.so.2",
        "libmkl_rt.so",
        "/usr/local/lib/libmkl_rt.so.2",
        "/usr/local/lib/libmkl_rt.so",
    ]
    for p in search_paths:
        try:
            return ctypes.CDLL(p)
        except OSError:
            continue
    return None


def bench_mkl_dgemm(m, n, k):
    mkl = load_mkl()
    if mkl is None:
        return None

    # Set MKL to use all available threads
    try:
        mkl.MKL_Set_Num_Threads(ctypes.c_int(4))
    except Exception:
        pass

    # cblas_dgemm signature:
    # void cblas_dgemm(CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
    #                  MKL_INT M, MKL_INT N, MKL_INT K,
    #                  double alpha, const double *A, MKL_INT lda,
    #                  const double *B, MKL_INT ldb,
    #                  double beta, double *C, MKL_INT ldc)
    cblas_dgemm = mkl.cblas_dgemm
    CblasRowMajor = 101
    CblasNoTrans = 111

    A = np.ascontiguousarray(np.random.randn(m, k).astype(DTYPE))
    B = np.ascontiguousarray(np.random.randn(k, n).astype(DTYPE))
    C = np.zeros((m, n), dtype=DTYPE)

    def fn():
        cblas_dgemm(
            ctypes.c_int(CblasRowMajor),
            ctypes.c_int(CblasNoTrans),
            ctypes.c_int(CblasNoTrans),
            ctypes.c_int(m),
            ctypes.c_int(n),
            ctypes.c_int(k),
            ctypes.c_double(1.0),
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(k),
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
            ctypes.c_double(0.0),
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
        )

    return bench(fn)


# ---------------------------------------------------------------------------
# Also benchmark single-threaded to get per-core performance
# ---------------------------------------------------------------------------
def bench_numpy_single(m, n, k):
    """NumPy matmul with OMP threads pinned to 1."""
    old_threads = os.environ.get("OMP_NUM_THREADS")
    old_openblas = os.environ.get("OPENBLAS_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # Some BLAS libs read env at first call; we use a fresh array to be safe
    A = np.random.randn(m, k).astype(DTYPE)
    B = np.random.randn(k, n).astype(DTYPE)
    def fn():
        np.matmul(A, B)
    result = bench(fn)
    # Restore
    if old_threads is not None:
        os.environ["OMP_NUM_THREADS"] = old_threads
    elif "OMP_NUM_THREADS" in os.environ:
        del os.environ["OMP_NUM_THREADS"]
    if old_openblas is not None:
        os.environ["OPENBLAS_NUM_THREADS"] = old_openblas
    elif "OPENBLAS_NUM_THREADS" in os.environ:
        del os.environ["OPENBLAS_NUM_THREADS"]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 100)
    print("SOTA MatMul Benchmark — Qwen 2.5 VL 3B shapes (float64)")
    print("=" * 100)

    # System info
    import platform
    cpu_model = "N/A"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":")[1].strip()
                    break
    except Exception:
        cpu_model = platform.processor() or "N/A"

    num_cores = os.cpu_count() or 1
    l2_kb = 1024  # default
    try:
        with open("/sys/devices/system/cpu/cpu0/cache/index2/size") as f:
            s = f.read().strip()
            if s.endswith("K"):
                l2_kb = int(s[:-1])
            elif s.endswith("M"):
                l2_kb = int(s[:-1]) * 1024
    except Exception:
        pass

    print(f"\n*** VERIFY THIS HARDWARE MATCHES YOUR SYSTEM BEFORE COMPARING RESULTS ***")
    print(f"CPU: {cpu_model}")
    print(f"Cores: {num_cores}")
    print(f"L2 cache: {l2_kb} KB/core")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy version: {np.__version__}")
    print(f"NumPy BLAS info: ", end="")
    try:
        blas_info = np.show_config("dicts")
        blas_name = blas_info.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown")
        blas_ver = blas_info.get("Build Dependencies", {}).get("blas", {}).get("version", "unknown")
        print(f"{blas_name} {blas_ver}")
    except Exception:
        print("(could not determine)")

    mkl_avail = load_mkl() is not None
    print(f"Intel MKL available: {mkl_avail}")
    print(f"Dtype: float64 | Warmup: {WARMUP} | Iterations: {ITERS}")
    print()

    # Theoretical peak: AVX-512, FMA, float64
    # float64: 512-bit / 64-bit = 8 doubles, 2 ops (FMA) = 16 flops/cycle
    # Read CPU frequency from /proc/cpuinfo
    cpu_ghz = 2.8  # fallback
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "cpu MHz" in line:
                    cpu_ghz = float(line.split(":")[1].strip()) / 1000.0
                    break
    except Exception:
        pass
    peak_1core = cpu_ghz * 16  # GFLOPS
    peak_all = peak_1core * num_cores
    print(f"Theoretical peak ({cpu_model}, AVX-512, float64):")
    print(f"  Single core: {peak_1core:.1f} GFLOPS ({cpu_ghz:.2f} GHz × 8 doubles × 2 FMA)")
    print(f"  {num_cores} cores:    {peak_all:.1f} GFLOPS")
    print()

    summary = []

    for shape_name, M, N, K in SHAPES:
        print("-" * 100)
        print(f"Shape: {shape_name}   (M={M}, N={N}, K={K})")
        print(f"FLOPs per matmul: {2*M*N*K:,}")
        print("-" * 100)

        # NumPy (multi-threaded)
        times = bench_numpy(M, N, K)
        mean_t, g_mean, min_t, g_peak = report("NumPy (OpenBLAS, multi-thread)", M, N, K, times)
        summary.append((shape_name, "NumPy OpenBLAS (multi)", mean_t, g_mean, min_t, g_peak))

        # NumPy (single-threaded)
        times = bench_numpy_single(M, N, K)
        mean_t, g_mean, min_t, g_peak = report("NumPy (OpenBLAS, 1 thread)", M, N, K, times)
        summary.append((shape_name, "NumPy OpenBLAS (1 thread)", mean_t, g_mean, min_t, g_peak))

        # SciPy dgemm
        try:
            times = bench_scipy_dgemm(M, N, K)
            mean_t, g_mean, min_t, g_peak = report("SciPy dgemm (OpenBLAS)", M, N, K, times)
            summary.append((shape_name, "SciPy dgemm", mean_t, g_mean, min_t, g_peak))
        except Exception as e:
            print(f"  SciPy dgemm: FAILED ({e})")

        # MKL dgemm
        if mkl_avail:
            times = bench_mkl_dgemm(M, N, K)
            if times is not None:
                mean_t, g_mean, min_t, g_peak = report("Intel MKL dgemm (multi-thread)", M, N, K, times)
                summary.append((shape_name, "Intel MKL dgemm (multi)", mean_t, g_mean, min_t, g_peak))

        print()

    # Summary table
    print("=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Shape':<30s} {'Library':<30s} {'Mean(ms)':>10s} {'Min(ms)':>10s} "
          f"{'GFLOPS(mean)':>14s} {'GFLOPS(peak)':>14s}")
    print("-" * 100)
    for shape, lib, mean_t, g_mean, min_t, g_peak in summary:
        print(f"{shape:<30s} {lib:<30s} {mean_t*1e3:10.3f} {min_t*1e3:10.3f} "
              f"{g_mean:14.2f} {g_peak:14.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
