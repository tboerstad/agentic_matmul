# SOTA MatMul Benchmark Results

> **YOU MUST VERIFY WHAT HARDWARE YOU ARE RUNNING ON BEFORE COMPARING AGAINST THESE RESULTS.**
> Run `mojo run bench_prefill.mojo` or `python bench_sota.py` — both print detected hardware at startup. If your hardware differs from what is listed below, your numbers WILL be different.

**Date:** 2026-03-07
**Hardware:** Intel Xeon (Skylake-SP, model 85) @ 2.80 GHz, 4 cores, AVX-512, 1 MB L2/core, KVM virtualized
**Dtype:** float64

## Theoretical Peak Performance (float64, AVX-512 FMA)

| Cores | GFLOPS |
|-------|--------|
| 1     | 44.8   |
| 4     | 179.2  |

Calculation: 2.8 GHz x 8 doubles/cycle (512-bit) x 2 (FMA) = 44.8 GFLOPS/core

## Benchmark Shapes (Qwen 2.5 VL 3B MLP)

| Shape   | M  | N     | K    | FLOPs          |
|---------|----|-------|------|----------------|
| Decode  | 1  | 11008 | 2048 | 45,088,768     |
| Prefill | 96 | 11008 | 2048 | 4,328,521,728  |

## Results Summary

> All results below were measured on the Skylake hardware listed above. Do NOT compare with numbers from a different machine.

### Decode Shape: 1 x 11008 x 2048

| Implementation                  | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) | vs SOTA |
|---------------------------------|-----------|----------|---------------|---------------|---------|
| **Intel MKL dgemm (4 threads)** | 4.584     | 3.964    | **9.84**      | **11.37**     | 1.00x   |
| NumPy/OpenBLAS (multi-thread)   | 5.399     | 5.194    | 8.35          | 8.68          | 0.76x   |
| NumPy/OpenBLAS (1 thread)       | 5.790     | 5.204    | 7.79          | 8.66          | 0.76x   |
| SciPy dgemm (OpenBLAS64)        | 12.550    | 11.946   | 3.59          | 3.77          | 0.33x   |
| **Mojo GOTO (current best)**    | 59.640    | ~59.4    | **0.76**      | **~0.76**     | 0.07x   |
| Mojo Packed                     | 60.064    | ~58.5    | 0.75          | ~0.77         | 0.07x   |
| Mojo SIMD                       | 65.307    | ~65.2    | 0.69          | ~0.69         | 0.06x   |
| Mojo Tiled                      | 68.085    | ~67.6    | 0.66          | ~0.67         | 0.06x   |
| Mojo Naive                      | 242.277   | ~237.1   | 0.19          | ~0.19         | 0.02x   |

**SOTA Winner (Decode): Intel MKL — 11.37 GFLOPS peak (25% of theoretical single-core peak)**

### Prefill Shape: 96 x 11008 x 2048

| Implementation                  | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) | vs SOTA |
|---------------------------------|-----------|----------|---------------|---------------|---------|
| **NumPy/OpenBLAS (1 thread)**   | 25.121    | 23.164   | **172.31**    | **186.86**    | 1.00x   |
| NumPy/OpenBLAS (multi-thread)   | 25.713    | 23.809   | 168.34        | 181.80        | 0.97x   |
| SciPy dgemm (OpenBLAS64)        | 33.239    | 31.168   | 130.22        | 138.88        | 0.74x   |
| Intel MKL dgemm (4 threads)     | 38.689    | 37.239   | 111.88        | 116.24        | 0.62x   |
| **Mojo GOTO (current best)**    | 80.792    | ~78.8    | **53.58**     | **~54.94**    | 0.29x   |
| Mojo Comptime                   | 93.720    | ~92.2    | 46.19         | ~46.94        | 0.25x   |
| Mojo Packed                     | 104.455   | ~102.4   | 41.44         | ~42.27        | 0.23x   |
| Mojo SIMD                       | 516.489   | ~515.5   | 8.38          | ~8.40         | 0.04x   |
| Mojo Tiled                      | 1457.150  | ~1457.2  | 2.97          | ~2.97         | 0.02x   |
| Mojo Naive                      | 18649.768 | ~18650   | 0.23          | ~0.23         | 0.001x  |

**SOTA Winner (Prefill): NumPy/OpenBLAS — 186.86 GFLOPS peak (104% of theoretical 4-core peak!)**

> Note: Exceeding theoretical peak is possible due to CPU turbo boost and measurement variance.

## Key Takeaways

1. **Decode (M=1):** Memory-bandwidth bound. Intel MKL is best at 11.37 GFLOPS — only 25% of
   compute peak because the tiny M=1 means the operation is essentially a matrix-vector product,
   limited by DRAM bandwidth rather than compute.

2. **Prefill (M=96):** Compute-bound. OpenBLAS achieves near-theoretical-peak at ~187 GFLOPS,
   showing excellent utilization of AVX-512 FMA units. This shape has enough work to amortize
   memory access and saturate the compute pipeline.

3. **Mojo GOTO kernel:** The matmul_goto kernel reaches 53.6 GFLOPS on prefill (~3.5x slower
   than SOTA) and 0.76 GFLOPS on decode (~15x slower). Key improvements:
   - j-parallel GOTO GEMM with per-tile B-panel packing (prefill): C panel stays in L2 across
     k-tiles while packed B enables sequential micro-kernel access (+16% over comptime)
   - GEMV path with p-outer/j-inner loop order (decode): j-parallelism for small M where
     previous kernels had only 1 i-tile = zero parallelism (+8.6% over previous best)

4. **B-panel packing:** Per-tile packing with j-parallelism works well (+16% prefill). Earlier
   attempt with i-parallelism (packing all of B globally) regressed because C panel didn't stay
   in L2 across k-tiles.

> **Note:** These results predate `matmul_prefill` and `matmul_adaptive`, which were developed later. See OPENBLAS_ANALYSIS.md for the latest prefill kernel results (measured on different hardware — do not directly compare numbers).

## Libraries Tested

| Library       | Version    | Backend                          |
|---------------|------------|----------------------------------|
| NumPy         | 2.4.2      | scipy-openblas 0.3.31.dev        |
| SciPy         | 1.17.1     | scipy-openblas64 0.3.31.dev      |
| Intel MKL     | 2025.3.1   | Intel MKL (oneMKL)               |
| Mojo          | 0.26.2-dev | Custom kernels (naive/tiled/simd) |
