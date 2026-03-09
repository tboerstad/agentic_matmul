# SOTA MatMul Benchmark Results

> **Hardware-specific results.** All numbers below were measured on specific machines.
> Performance varies significantly across CPUs, core counts, SIMD widths, and virtualization.
> Before comparing or citing these numbers, verify your hardware — see [AGENTS.md](AGENTS.md)
> for the verification steps. Re-run benchmarks on your own machine for accurate comparisons.

**Date:** 2026-03-09
**Dtype:** float64

## Hardware Configurations

| | Machine A | Machine B |
|---|---|---|
| **CPU** | Intel Xeon @ 2.80 GHz | Intel(R) Xeon(R) Processor @ 2.10 GHz |
| **Microarchitecture** | Skylake | Granite Rapids (model 207, stepping 2) |
| **Cores** | 4 (1 thread/core) | 4 (1 thread/core) |
| **SIMD** | AVX-512 | AVX-512 (incl. VNNI, BF16, FP16, AMX) |
| **Virtualization** | KVM | KVM |
| **L1d / L1i Cache** | — | 192 KiB / 128 KiB (4 instances) |
| **L2 Cache** | — | 8 MiB (4 instances) |
| **L3 Cache** | — | 260 MiB (shared) |
| **RAM** | — | 16 GiB |
| **Kernel** | — | Linux 6.18.5 |
| **Theoretical Peak (1 core)** | 44.8 GFLOPS | 33.6 GFLOPS |
| **Theoretical Peak (4 cores)** | 179.2 GFLOPS | 134.4 GFLOPS |

Calculation: clock × 8 doubles/cycle (512-bit) × 2 (FMA) = GFLOPS/core

> Note: Exceeding theoretical peak (Machine B: observed up to ~221 GFLOPS vs 134.4 theoretical)
> is due to turbo boost — 2.10 GHz is the base frequency; actual boost clocks are significantly higher.

## Methodology

**Machine B (2026-03-09):** All benchmarks (`bench_prefill.mojo`, `bench_decode.mojo`,
`bench_sota.py`) were run **serially in a single shell session** to avoid CPU contention
and ensure fair cross-implementation comparison. Previous runs used separate sessions which
could introduce bias from varying system load and thermal state.

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
| **NumPy/OpenBLAS (multi-thread)** | 4.150 | **11.33** | 1.00× | | 3.701 | **14.03** | 1.00× |
| NumPy/OpenBLAS (1 thread) | 4.559 | 10.17 | 0.90× | | 3.785 | 13.95 | 0.99× |
| SciPy dgemm (OpenBLAS) | 12.833 | 4.10 | 0.36× | | 8.639 | 5.91 | 0.42× |
| **Mojo Dispatch** | — | — | — | | 3.456 | **14.19** | **1.01×** |
| **Mojo Decode** | — | — | — | | 3.545 | 13.05 | 0.93× |
| Mojo GOTO | 63.294 | 0.72 | 0.06× | | 4.572 | 10.31 | 0.73× |

**Machine A notes:** Decode/dispatch kernels were not benchmarked on this machine.

**Machine B notes:** All Mojo results from `bench_decode.mojo`, all SOTA results from
`bench_sota.py`, run serially in the same session.

**Machine B highlight:** Mojo Dispatch achieves **14.19 GFLOPS peak** — **1.01× of OpenBLAS
multi-threaded** (14.03 GFLOPS). In this serial run the two are essentially tied, with
dispatch edging ahead by 1%.

### Prefill Shape: 96 × 11008 × 2048

| Implementation | Mean (ms) | GFLOPS (peak) | vs SOTA | | Mean (ms) | GFLOPS (peak) | vs SOTA |
|---|---|---|---|---|---|---|---|
| | **Machine A** | | | | **Machine B** | | |
| **NumPy/OpenBLAS (1 thread)** | 23.457 | 192.40 | 0.98× | | 20.357 | **221.60** | **1.00×** |
| **NumPy/OpenBLAS (multi-thread)** | 23.825 | **196.00** | 1.00× | | 21.496 | 212.17 | 0.96× |
| SciPy dgemm (OpenBLAS) | 34.582 | 131.71 | 0.67× | | 33.593 | 138.83 | 0.63× |
| Mojo GOTO | 53.175 | 82.89 | 0.42× | | 27.989 | 158.54 | 0.72× |
| Mojo Prefill | 35.272 | 125.38 | 0.64× | | 28.103 | 157.96 | 0.71× |

**Machine A notes:** Mojo results from dedicated `bench_prefill.mojo`.

**Machine B notes:** Mojo results from `bench_prefill.mojo`, SOTA results from `bench_sota.py`,
run serially in the same session.

**Machine B result:** On Machine B, OpenBLAS 1-thread is the fastest at **221.60 GFLOPS peak** —
faster than multi-threaded (212.17), suggesting thread overhead hurts at this shape/core count.
Mojo GOTO and Prefill are nearly identical (~158 GFLOPS), reaching **71–72% of SOTA**.

## Mojo Prefill vs GOTO

Results from dedicated `bench_prefill.mojo` (96 × 11008 × 2048):

| | Machine A | | Machine B | |
|---|---|---|---|---|
| Kernel | GFLOPS (peak) | Mean (ms) | GFLOPS (peak) | Mean (ms) |
| goto | 82.89 | 53.175 | 158.54 | 27.989 |
| prefill | 125.38 | 35.272 | 157.96 | 28.103 |
| **Speedup** | **1.51×** | | **1.00×** | |

Machine A: 51% improvement from prefill kernel. Machine B (serial run): GOTO and Prefill
are essentially tied — the prefill kernel's packing overhead may offset its microkernel
gains at this cache size and thermal state.

## Mojo Decode Kernel Results (Machine B)

Results from dedicated `bench_decode.mojo` (serial run):

### Single-token decode: 1 × 11008 × 2048

| Kernel | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) |
|--------|-----------|----------|---------------|---------------|
| goto | 4.572 | 4.372 | 9.86 | 10.31 |
| decode | 3.545 | 3.455 | 12.72 | 13.05 |
| dispatch | 3.456 | 3.177 | 13.05 | 14.19 |

**Speedup (goto→dispatch): 1.29× mean, 1.38× peak**

### Batch decode: 7 × 11008 × 2048

| Kernel | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) |
|--------|-----------|----------|---------------|---------------|
| goto | 25.933 | 25.475 | 12.17 | 12.39 |
| decode | 8.075 | 7.870 | 39.08 | 40.10 |
| dispatch | 8.064 | 7.699 | 39.14 | 41.00 |

**Speedup (goto→dispatch): 3.21× mean, 3.31× peak**

## Key Takeaways

1. **Decode (M=1):** Memory-bandwidth bound. Mojo Dispatch matches OpenBLAS multi-threaded
   (14.19 vs 14.03 GFLOPS peak). The k-parallel GEMV approach is effective on Granite Rapids.

2. **Decode (M=7):** Mojo Dispatch reaches **41.00 GFLOPS peak** — a 3.31× speedup over GOTO.
   The batch decode kernel scales well with M.

3. **Prefill (M=96):** Compute-bound. OpenBLAS single-threaded leads at **221.60 GFLOPS**.
   Mojo GOTO and Prefill are tied at ~158 GFLOPS (**71% of SOTA**). The gap is larger than
   in previous non-serial runs, suggesting earlier measurements may have had favorable
   scheduling or thermal conditions.

4. **Hardware matters enormously:** The same code runs dramatically differently across machines.
   Machine B's Granite Rapids with large L3 cache (260 MiB) and higher effective boost clocks
   narrows many gaps with OpenBLAS vs Machine A.

5. **Serial measurement matters:** Running benchmarks serially in a single session gives
   different (and more directly comparable) numbers than running them in parallel or in
   separate sessions. Previous Machine B runs showed higher Mojo prefill numbers (~218
   GFLOPS) which may have benefited from cooler initial thermal state.

## Libraries Tested

| Library       | Version    | Backend                          |
|---------------|------------|----------------------------------|
| NumPy         | 2.4.2      | scipy-openblas 0.3.31.dev        |
| SciPy         | 1.17.1     | scipy-openblas64 0.3.31.dev      |
| Intel MKL     | —          | Not available in this environment |
| Mojo          | 0.26.2-dev | Custom kernels (naive→prefill)   |
