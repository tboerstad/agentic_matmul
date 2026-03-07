# SOTA MatMul Benchmark Results

**Date:** 2026-03-07
**Hardware:** Intel Xeon @ 2.80 GHz, 4 cores, AVX-512 (Skylake), KVM virtualized
**Dtype:** float64

## Theoretical Peak Performance (float64, AVX-512 FMA)

| Cores | GFLOPS |
|-------|--------|
| 1     | 44.8   |
| 4     | 179.2  |

Calculation: 2.8 GHz × 8 doubles/cycle (512-bit) × 2 (FMA) = 44.8 GFLOPS/core

## Benchmark Shapes (Qwen 2.5 VL 3B MLP)

| Shape   | M  | N     | K    | FLOPs          |
|---------|----|-------|------|----------------|
| Decode  | 1  | 11008 | 2048 | 45,088,768     |
| Prefill | 96 | 11008 | 2048 | 4,328,521,728  |

## Results Summary

### Decode Shape: 1 × 11008 × 2048

| Implementation                  | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) | vs SOTA |
|---------------------------------|-----------|----------|---------------|---------------|---------|
| **NumPy/OpenBLAS (1 thread)**   | 8.055     | 7.020    | **5.60**      | **6.42**      | 1.00×   |
| NumPy/OpenBLAS (multi-thread)   | 8.060     | 7.646    | 5.59          | 5.90          | 0.92×   |
| SciPy dgemm (OpenBLAS)          | 24.195    | 22.158   | 1.86          | 2.03          | 0.32×   |
| Mojo Comptime (current best)    | 67.18     | ~65.8    | 0.67          | ~0.68         | 0.11×   |
| Mojo SIMD                       | 67.72     | ~67.2    | 0.67          | ~0.67         | 0.10×   |
| Mojo Parallel                   | 69.93     | ~69.5    | 0.64          | ~0.65         | 0.10×   |
| Mojo Register-blocked           | 70.56     | ~69.5    | 0.64          | ~0.65         | 0.10×   |
| Mojo Packed                     | 76.63     | ~76.2    | 0.59          | ~0.59         | 0.09×   |
| Mojo Tiled                      | 97.56     | ~95.7    | 0.46          | ~0.47         | 0.07×   |
| Mojo Naive                      | 285.71    | ~284.4   | 0.16          | ~0.16         | 0.02×   |

**SOTA Winner (Decode): NumPy/OpenBLAS (1 thread) — 6.42 GFLOPS peak (14% of theoretical single-core peak)**

> Note: Intel MKL was not available in this benchmark run. Previous runs with MKL showed 11.37 GFLOPS peak.

### Prefill Shape: 96 × 11008 × 2048

| Implementation                  | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) | vs SOTA |
|---------------------------------|-----------|----------|---------------|---------------|---------|
| **NumPy/OpenBLAS (multi-thread)** | 68.360  | 55.992   | **63.32**     | **77.31**     | 1.00×   |
| SciPy dgemm (OpenBLAS)          | 70.771    | 65.756   | 61.16         | 65.83         | 0.85×   |
| NumPy/OpenBLAS (1 thread)       | 86.993    | 80.004   | 49.76         | 54.10         | 0.70×   |
| Mojo Comptime (current best)    | 123.03    | ~122.0   | 35.18         | ~35.5         | 0.46×   |
| Mojo Packed                     | 141.47    | ~140.3   | 30.60         | ~30.9         | 0.40×   |
| Mojo Register-blocked           | 233.36    | ~231.4   | 18.55         | ~18.7         | 0.24×   |
| Mojo Parallel                   | 239.49    | ~238.6   | 18.07         | ~18.1         | 0.23×   |
| Mojo SIMD                       | 497.69    | ~497.3   | 8.70          | ~8.70         | 0.11×   |
| Mojo Tiled                      | 1246.61   | ~1246.6  | 3.47          | ~3.47         | 0.04×   |
| Mojo Naive                      | 24746.93  | ~24747   | 0.17          | ~0.17         | 0.002×  |

**SOTA Winner (Prefill): NumPy/OpenBLAS (multi-thread) — 77.31 GFLOPS peak (43% of theoretical 4-core peak)**

## Key Takeaways

1. **Decode (M=1):** Memory-bandwidth bound. OpenBLAS peaks at 6.42 GFLOPS — only 14% of
   compute peak because the tiny M=1 means the operation is essentially a matrix-vector product,
   limited by DRAM bandwidth rather than compute.

2. **Prefill (M=96):** Compute-bound. OpenBLAS achieves 77.31 GFLOPS peak (43% of 4-core
   theoretical peak). This shape has enough work to amortize memory access and benefit from
   multi-threading.

3. **Mojo comptime kernel** is now the fastest Mojo implementation on both shapes, reaching
   35.18 GFLOPS mean on prefill (~0.46× SOTA) and 0.67 GFLOPS on decode (~0.11× SOTA).
   The GOTO-style j→k→i loop order, register blocking, comptime unrolling, and software
   prefetching give it a clear edge over the packed kernel.

4. **Mojo progress:** The comptime kernel is within 2× of OpenBLAS on prefill (35.18 vs 63.32
   GFLOPS mean). Remaining gaps are likely due to: missing B-panel packing for large K,
   sub-optimal tile sizes, and lack of architecture-specific micro-kernel tuning.

5. **Decode bottleneck:** All Mojo kernels cluster around 0.6-0.7 GFLOPS on decode — parallelism
   and register blocking don't help when M=1. The bottleneck is memory bandwidth, not compute.

## Libraries Tested

| Library       | Version    | Backend                          |
|---------------|------------|----------------------------------|
| NumPy         | 2.4.2      | scipy-openblas 0.3.31.dev        |
| SciPy         | 1.17.1     | scipy-openblas 0.3.31.dev        |
| Mojo          | 0.26.2-dev | Custom kernels (7 versions)      |
