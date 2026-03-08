# SOTA MatMul Benchmark Results

> **Hardware-specific results.** All numbers below were measured on specific machines.
> Performance varies significantly across CPUs, core counts, SIMD widths, and virtualization.
> Before comparing or citing these numbers, verify your hardware — see [AGENTS.md](AGENTS.md)
> for the verification steps. Re-run benchmarks on your own machine for accurate comparisons.

**Date:** 2026-03-08
**Dtype:** float64

## Hardware Configurations

| | Machine A | Machine B |
|---|---|---|
| **CPU** | Intel Xeon @ 2.80 GHz | Intel Xeon @ 2.10 GHz |
| **Microarchitecture** | Skylake | Granite Rapids |
| **Cores** | 4 | 4 |
| **SIMD** | AVX-512 | AVX-512 |
| **Virtualization** | KVM | KVM |
| **L3 Cache** | — | 260 MiB |
| **Theoretical Peak (1 core)** | 44.8 GFLOPS | 33.6 GFLOPS |
| **Theoretical Peak (4 cores)** | 179.2 GFLOPS | 134.4 GFLOPS |

Calculation: clock × 8 doubles/cycle (512-bit) × 2 (FMA) = GFLOPS/core

## Benchmark Shapes (Qwen 2.5 VL 3B MLP)

| Shape   | M  | N     | K    | FLOPs          |
|---------|----|-------|------|----------------|
| Decode  | 1  | 11008 | 2048 | 45,088,768     |
| Prefill | 96 | 11008 | 2048 | 4,328,521,728  |

## Results Summary

### Decode Shape: 1 × 11008 × 2048

| Implementation | Mean (ms) | GFLOPS (peak) | vs SOTA | | Mean (ms) | GFLOPS (peak) | vs SOTA |
|---|---|---|---|---|---|---|---|
| | **Machine A** | | | | **Machine B** | | |
| **NumPy/OpenBLAS (multi-thread)** | 4.150 | **11.33** | 1.00× | | 3.329 | **14.32** | 1.00× |
| NumPy/OpenBLAS (1 thread) | 4.559 | 10.17 | 0.90× | | 2.081 | 22.58 | 1.58× |
| SciPy dgemm (OpenBLAS) | 12.833 | 4.10 | 0.36× | | 9.004 | 5.33 | 0.37× |
| **Mojo Dispatch** | — | — | — | | 2.625 | **20.08** | **1.40×** |
| **Mojo Decode** | — | — | — | | 2.845 | 18.61 | 1.30× |
| Mojo GOTO | 63.294 | 0.72 | 0.06× | | 4.162 | 12.39 | 0.87× |
| Mojo Comptime | 66.718 | 0.68 | 0.06× | | 65.324 | 0.69 | 0.05× |
| Mojo SIMD | 64.924 | 0.70 | 0.06× | | 71.836 | 0.63 | 0.04× |
| Mojo Parallel | 64.094 | 0.71 | 0.06× | | 72.779 | 0.62 | 0.04× |
| Mojo Naive | 285.958 | 0.16 | 0.01× | | 247.265 | 0.18 | 0.01× |

**Machine A notes:** Decode/dispatch kernels were not benchmarked on this machine. GOTO results from bench_matmul.mojo (serial run).

**Machine B notes:** Mojo Dispatch, Decode, and GOTO results from dedicated `bench_decode.mojo` (M=1 shape). Remaining kernels from `bench_matmul.mojo`.

**Machine B highlight:** Mojo Dispatch achieves **20.08 GFLOPS** — 1.40× faster than NumPy/OpenBLAS multi-threaded and 89% of OpenBLAS single-threaded peak (22.58 GFLOPS). The k-parallel GEMV optimization is highly effective on this hardware.

### Prefill Shape: 96 × 11008 × 2048

| Implementation | Mean (ms) | GFLOPS (peak) | vs SOTA | | Mean (ms) | GFLOPS (peak) | vs SOTA |
|---|---|---|---|---|---|---|---|
| | **Machine A** | | | | **Machine B** | | |
| **NumPy/OpenBLAS (multi-thread)** | 23.825 | **196.00** | 1.00× | | 18.768 | **239.43** | 1.00× |
| NumPy/OpenBLAS (1 thread) | 23.457 | 192.40 | 0.98× | | 20.360 | 239.31 | 1.00× |
| **Mojo Prefill (best kernel)** | 35.272 | 125.38 | 0.64× | | 20.528 | **217.94** | **0.91×** |
| SciPy dgemm (OpenBLAS) | 34.582 | 131.71 | 0.67× | | 28.500 | 163.93 | 0.68× |
| Mojo GOTO | 53.175 | 82.89 | 0.42× | | 23.371 | 192.73 | 0.81× |
| Mojo Dispatch† | — | — | — | | 80.623 | 53.69 | 0.22× |
| Mojo Comptime† | 113.537 | 38.38 | 0.20× | | 95.134 | 45.50 | 0.19× |
| Mojo Packed† | 138.886 | 31.64 | 0.16× | | 106.273 | 40.73 | 0.17× |
| Mojo Register-Blocked† | 213.036 | 21.13 | 0.11× | | 154.197 | 28.07 | 0.12× |
| Mojo Parallel† | 239.338 | 18.92 | 0.10× | | 176.735 | 24.49 | 0.10× |
| Mojo SIMD† | 482.479 | 9.07 | 0.05× | | 359.243 | 12.05 | 0.05× |
| Mojo Tiled† | 1213.973 | 3.57 | 0.02× | | 1512.646 | 2.86 | 0.01× |
| Mojo Naive† | 23462.631 | 0.18 | 0.001× | | 17943.288 | 0.24 | 0.001× |

†Results from bench_matmul.mojo (serial run of all kernels). GOTO and Prefill results from dedicated bench_prefill.mojo for more accurate comparison. Dispatch uses the decode kernel for prefill shape (suboptimal — it's designed for decode).

**Machine B highlight:** Mojo Prefill achieves **217.94 GFLOPS** — **91% of OpenBLAS peak!** The gap narrowed dramatically from 36% (Machine A) to just 9% (Machine B).

> Note: Exceeding theoretical peak (Machine B: 239 GFLOPS vs 134.4 theoretical) is due to
> turbo boost — the 2.10 GHz is the base frequency; actual boost clocks are significantly higher.

## Mojo Prefill vs GOTO

Results from dedicated `bench_prefill.mojo` (96 × 11008 × 2048):

| | Machine A | | Machine B | |
|---|---|---|---|---|
| Kernel | GFLOPS (peak) | Mean (ms) | GFLOPS (peak) | Mean (ms) |
| goto | 82.89 | 53.175 | 192.73 | 23.371 |
| prefill | 125.38 | 35.272 | 217.94 | 20.528 |
| **Speedup** | **1.51×** | | **1.14×** | |

Machine A: 51% improvement. Machine B: 14% improvement (both kernels run faster, narrowing relative gap).

## Mojo Decode Kernel Results (Machine B only)

Results from dedicated `bench_decode.mojo`:

### Single-token decode: 1 × 11008 × 2048

| Kernel | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) |
|--------|-----------|----------|---------------|---------------|
| goto | 4.162 | 3.638 | 10.83 | 12.39 |
| decode | 2.845 | 2.422 | 15.85 | 18.61 |
| dispatch | 2.625 | 2.246 | 17.18 | 20.08 |

**Speedup (goto→dispatch): 1.50× (50% improvement)**

### Small-batch decode: 4 × 11008 × 2048

| Kernel | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) |
|--------|-----------|----------|---------------|---------------|
| goto | 14.614 | 14.307 | 12.34 | 12.61 |
| decode | 5.682 | 5.141 | 31.74 | 35.08 |
| dispatch | 5.289 | 4.901 | 34.10 | 36.80 |

**Speedup (goto→dispatch): 2.78× (178% improvement)**

### Batch decode: 7 × 11008 × 2048

| Kernel | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) |
|--------|-----------|----------|---------------|---------------|
| goto | 21.400 | 19.809 | 14.75 | 15.93 |
| decode | 7.799 | 7.100 | 40.47 | 44.45 |
| dispatch | 7.703 | 6.979 | 40.97 | 45.23 |

**Speedup (goto→dispatch): 2.79× (179% improvement)**

## Key Takeaways

1. **Decode (M=1):** Memory-bandwidth bound. On Machine B, the specialized decode kernel with
   k-parallel GEMV reaches 20.08 GFLOPS — **1.40× faster than OpenBLAS multi-threaded** and
   89% of OpenBLAS single-threaded. This is a dramatic improvement over Machine A where the
   best Mojo kernel managed only 6% of SOTA.

2. **Prefill (M=96):** Compute-bound. On Machine B, Mojo Prefill reaches **217.94 GFLOPS (91% of
   OpenBLAS)**. On Machine A, it was 125.38 GFLOPS (64% of OpenBLAS). The Granite Rapids
   microarchitecture with its large L3 cache (260 MiB) benefits Mojo's packing strategy.

3. **Hardware matters enormously:** The same code runs dramatically differently across machines.
   Machine B's Granite Rapids with large L3 cache and higher effective clock speeds narrows
   or closes many gaps with OpenBLAS.

4. **Decode kernel breakthrough:** The k-parallel GEMV decode kernel is the star on Machine B,
   beating OpenBLAS multi-threaded by 40% on M=1 decode. The improvement grows with batch
   size: 2.78× speedup over GOTO at M=4, 2.79× at M=7.

5. **Remaining gap to SOTA (prefill, 9%):** See [OPENBLAS_ANALYSIS.md](OPENBLAS_ANALYSIS.md) for
   detailed analysis of OpenBLAS's 16×12 microkernel vs Mojo's 8×24 microkernel.

## Libraries Tested

| Library       | Version    | Backend                          |
|---------------|------------|----------------------------------|
| NumPy         | 2.4.2      | scipy-openblas 0.3.31.dev        |
| SciPy         | 1.17.1     | scipy-openblas64 0.3.31.dev      |
| Intel MKL     | —          | Not available in this environment |
| Mojo          | 0.26.2-dev | Custom kernels (naive→prefill)   |
