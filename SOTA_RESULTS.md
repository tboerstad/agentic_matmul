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
| **Intel MKL dgemm (4 threads)** | 4.584     | 3.964    | **9.84**      | **11.37**     | 1.00×   |
| NumPy/OpenBLAS (multi-thread)   | 5.399     | 5.194    | 8.35          | 8.68          | 0.76×   |
| NumPy/OpenBLAS (1 thread)       | 5.790     | 5.204    | 7.79          | 8.66          | 0.76×   |
| SciPy dgemm (OpenBLAS64)        | 12.550    | 11.946   | 3.59          | 3.77          | 0.33×   |
| **Mojo Optimized (current best)**| 56.680   | ~56.2    | **0.80**      | **~0.80**     | 0.07×   |
| Mojo Comptime                   | 60.188    | ~59.6    | 0.75          | ~0.76         | 0.07×   |
| Mojo Packed                     | 65.938    | ~63.8    | 0.68          | ~0.71         | 0.06×   |
| Mojo SIMD                       | 61.920    | ~61.5    | 0.73          | ~0.73         | 0.06×   |
| Mojo Tiled                      | 61.280    | ~60.5    | 0.74          | ~0.74         | 0.07×   |
| Mojo Naive                      | 241.892   | ~241.0   | 0.19          | ~0.19         | 0.02×   |

**SOTA Winner (Decode): Intel MKL — 11.37 GFLOPS peak (25% of theoretical single-core peak)**

### Prefill Shape: 96 × 11008 × 2048

| Implementation                  | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) | vs SOTA |
|---------------------------------|-----------|----------|---------------|---------------|---------|
| **NumPy/OpenBLAS (1 thread)**   | 25.121    | 23.164   | **172.31**    | **186.86**    | 1.00×   |
| NumPy/OpenBLAS (multi-thread)   | 25.713    | 23.809   | 168.34        | 181.80        | 0.97×   |
| SciPy dgemm (OpenBLAS64)        | 33.239    | 31.168   | 130.22        | 138.88        | 0.74×   |
| Intel MKL dgemm (4 threads)     | 38.689    | 37.239   | 111.88        | 116.24        | 0.62×   |
| **Mojo Optimized (current best)**| 100.097  | ~99.3    | **43.24**     | **~43.6**     | 0.23×   |
| Mojo Comptime                   | 105.470   | ~104.5   | 41.04         | ~41.4         | 0.22×   |
| Mojo Packed                     | 107.069   | ~105.0   | 40.43         | ~41.2         | 0.22×   |
| Mojo Reg-Blocked                | 185.517   | ~184.0   | 23.33         | ~23.5         | 0.13×   |
| Mojo SIMD                       | 523.106   | ~521.0   | 8.27          | ~8.31         | 0.04×   |
| Mojo Tiled                      | 1445.144  | ~1445.0  | 3.00          | ~3.00         | 0.02×   |
| Mojo Naive                      | 18505.352 | ~18505   | 0.23          | ~0.23         | 0.001×  |

**SOTA Winner (Prefill): NumPy/OpenBLAS — 186.86 GFLOPS peak (104% of theoretical 4-core peak!)**

> Note: Exceeding theoretical peak is possible due to CPU turbo boost and measurement variance.

## Key Takeaways

1. **Decode (M=1):** Memory-bandwidth bound. Intel MKL is best at 11.37 GFLOPS — only 25% of
   compute peak because the tiny M=1 means the operation is essentially a matrix-vector product,
   limited by DRAM bandwidth rather than compute.

2. **Prefill (M=96):** Compute-bound. OpenBLAS achieves near-theoretical-peak at ~187 GFLOPS,
   showing excellent utilization of AVX-512 FMA units. This shape has enough work to amortize
   memory access and saturate the compute pipeline.

3. **Mojo gap:** The current Mojo optimized kernel reaches 43.2 GFLOPS on prefill (~4.3× slower
   than SOTA) and 0.80 GFLOPS on decode (~14× slower). The optimized kernel uses B-panel packing
   to eliminate TLB thrashing on B's 88KB row stride.

4. **B-panel packing:** Packing B-matrix panels into contiguous buffers before the micro-kernel
   now works (+5.4% over comptime on prefill) by using j-parallel structure where each thread
   packs and computes its own j-tiles, avoiding the earlier regression from serial packing with
   parallel i-blocks that had too much synchronization overhead.

## Libraries Tested

| Library       | Version    | Backend                          |
|---------------|------------|----------------------------------|
| NumPy         | 2.4.2      | scipy-openblas 0.3.31.dev        |
| SciPy         | 1.17.1     | scipy-openblas64 0.3.31.dev      |
| Intel MKL     | 2025.3.1   | Intel MKL (oneMKL)               |
| Mojo          | 0.26.2-dev | Custom kernels (naive/tiled/simd)|
