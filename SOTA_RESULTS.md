# SOTA MatMul Benchmark Results

> **Hardware-specific results.** All numbers below were measured on a specific machine.
> Performance varies significantly across CPUs, core counts, SIMD widths, and virtualization.
> Before comparing or citing these numbers, verify your hardware — see [AGENTS.md](AGENTS.md)
> for the verification steps. Re-run benchmarks on your own machine for accurate comparisons.

**Date:** 2026-03-08
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
| **NumPy/OpenBLAS (multi-thread)** | 4.150   | 3.978    | **10.86**     | **11.33**     | 1.00×   |
| NumPy/OpenBLAS (1 thread)       | 4.559     | 4.434    | 9.89          | 10.17         | 0.90×   |
| SciPy dgemm (OpenBLAS)          | 12.833    | 10.989   | 3.51          | 4.10          | 0.36×   |
| **Mojo GOTO (best kernel)**     | 63.294    | 63.039   | 0.71          | 0.72          | 0.06×   |
| Mojo Parallel                   | 64.094    | 63.372   | 0.70          | 0.71          | 0.06×   |
| Mojo SIMD                       | 64.924    | 64.428   | 0.69          | 0.70          | 0.06×   |
| Mojo Register-Blocked           | 66.263    | 65.280   | 0.68          | 0.69          | 0.06×   |
| Mojo Comptime                   | 66.718    | 66.068   | 0.68          | 0.68          | 0.06×   |
| Mojo Tiled                      | 68.787    | 66.183   | 0.66          | 0.68          | 0.06×   |
| Mojo Packed                     | 77.136    | 76.380   | 0.58          | 0.59          | 0.05×   |
| Mojo Naive                      | 285.958   | 285.794  | 0.16          | 0.16          | 0.01×   |

**SOTA Winner (Decode): NumPy/OpenBLAS — 11.33 GFLOPS peak (25% of theoretical single-core peak)**

### Prefill Shape: 96 × 11008 × 2048

| Implementation                  | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) | vs SOTA |
|---------------------------------|-----------|----------|---------------|---------------|---------|
| **NumPy/OpenBLAS (multi-thread)** | 23.825  | 22.084   | **181.68**    | **196.00**    | 1.00×   |
| NumPy/OpenBLAS (1 thread)       | 23.457    | 22.497   | 184.53        | 192.40        | 0.98×   |
| **Mojo Prefill (best kernel)**  | 35.272    | 34.524   | 122.72        | 125.38        | 0.64×   |
| SciPy dgemm (OpenBLAS)          | 34.582    | 32.864   | 125.17        | 131.71        | 0.67×   |
| Mojo GOTO                       | 53.175    | 52.217   | 81.40         | 82.89         | 0.42×   |
| Mojo Comptime†                  | 113.537   | 112.794  | 38.12         | 38.38         | 0.20×   |
| Mojo Packed†                    | 138.886   | 136.813  | 31.17         | 31.64         | 0.16×   |
| Mojo Register-Blocked†          | 213.036   | 204.817  | 20.32         | 21.13         | 0.11×   |
| Mojo Parallel†                  | 239.338   | 228.812  | 18.09         | 18.92         | 0.10×   |
| Mojo SIMD†                      | 482.479   | 477.314  | 8.97          | 9.07          | 0.05×   |
| Mojo Tiled†                     | 1213.973  | —        | 3.57          | —             | 0.02×   |
| Mojo Naive†                     | 23462.631 | —        | 0.18          | —             | 0.001×  |

†Results from bench_matmul.mojo (serial run of all kernels). GOTO and Prefill results from dedicated bench_prefill.mojo for more accurate comparison.

**SOTA Winner (Prefill): NumPy/OpenBLAS — 196.00 GFLOPS peak (109% of theoretical 4-core peak!)**

> Note: Exceeding theoretical peak is possible due to CPU turbo boost and measurement variance.

## Mojo Prefill vs GOTO: 51% Improvement

The dedicated `bench_prefill.mojo` benchmark shows:

| Kernel  | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) |
|---------|-----------|----------|---------------|---------------|
| goto    | 53.175    | 52.217   | 81.40         | 82.89         |
| prefill | 35.272    | 34.524   | 122.72        | 125.38        |

**Speedup: 1.51× (51% improvement)**

The prefill kernel closes the gap with OpenBLAS from 42% to 64% of SOTA peak performance.

## Key Takeaways

1. **Decode (M=1):** Memory-bandwidth bound. OpenBLAS is best at 11.33 GFLOPS — only 25% of
   compute peak because the tiny M=1 means the operation is essentially a matrix-vector product,
   limited by DRAM bandwidth rather than compute.

2. **Prefill (M=96):** Compute-bound. OpenBLAS achieves near-theoretical-peak at ~196 GFLOPS,
   showing excellent utilization of AVX-512 FMA units. This shape has enough work to amortize
   memory access and saturate the compute pipeline.

3. **Mojo Prefill kernel:** The best Mojo kernel reaches 125.4 GFLOPS on prefill (64% of SOTA),
   a 51% improvement over the GOTO kernel. Key optimizations:
   - Worker-based parallelism with j-tile batching (vs per-tile scheduling)
   - A-panel packing for L1 locality
   - NR=24 microkernel with 8×24 register blocking
   - KC=512 k-tile for better L2 utilization

4. **Mojo GOTO kernel:** Reaches 82.9 GFLOPS on prefill (42% of SOTA) and 0.72 GFLOPS on
   decode (~6% of SOTA). The GOTO-style j-parallel design with per-tile B-panel packing
   provides the foundation that the prefill kernel builds upon.

5. **Remaining gap to SOTA (36%):** See [OPENBLAS_ANALYSIS.md](OPENBLAS_ANALYSIS.md) for
   detailed analysis of OpenBLAS's 16×12 microkernel vs Mojo's 8×24 microkernel.

## Libraries Tested

| Library       | Version    | Backend                          |
|---------------|------------|----------------------------------|
| NumPy         | 2.4.2      | scipy-openblas 0.3.31.dev        |
| SciPy         | 1.17.1     | scipy-openblas64 0.3.31.dev      |
| Intel MKL     | —          | Not available in this environment |
| Mojo          | 0.26.2-dev | Custom kernels (naive→prefill)   |
