# matmul

An experiment in writing optimized matmul kernels in Mojo using only [Claude Code](https://claude.com/claude-code) on mobile (iOS). The sole goal is maximizing GFLOPS on the two matrix shapes used by Qwen 2.5 VL 3B MLP projections (float64):

- **Decode:** 1 × 11008 × 2048 (memory-bandwidth bound)
- **Prefill:** 96 × 11008 × 2048 (compute-bound)

The project implements 12 progressively optimized kernels — from a naive triple-nested loop to a GOTO-style GEMM with automatic shape dispatch — showcasing cache blocking, SIMD vectorization, thread parallelism, register blocking, data packing, and software prefetching in Mojo.

## Results

Peak GFLOPS by hardware (higher is better):

### Prefill (96 × 11008 × 2048)

| Kernel | Xeon Skylake 2.80 GHz (4c) | Xeon Granite Rapids 2.10 GHz (4c) | Apple M4 Max (14c) |
|---|---|---|---|
| SciPy dgemm | — | — | **538.1** |
| NumPy (Accelerate/OpenBLAS) | 196.0 | 239.4 | 483.1 |
| **Mojo (agentic matmul)** | — | — | **189.9** |
| Mojo linalg (stdlib) | — | — | 104.9 |

### Decode (1 × 11008 × 2048)

| Kernel | Xeon Skylake 2.80 GHz (4c) | Xeon Granite Rapids 2.10 GHz (4c) | Apple M4 Max (14c) |
|---|---|---|---|
| NumPy (Accelerate/OpenBLAS) | 11.3 | 14.3 | **54.3** |
| **Mojo (agentic matmul)** | — | 20.1 | **20.7** |
| Mojo linalg (stdlib) | — | — | 4.8 |

## Kernel evolution

1. **naive** — Triple-nested loop baseline
2. **tiled** — 32×32 cache-blocking
3. **simd** — Tiled + SIMD vectorization
4. **parallel** — Tiled + thread parallelism
5. **register_blocked** — Higher loop unrolling
6. **packed** — A/B buffer packing for sequential access
7. **comptime** — Compile-time parameter specialization
8. **goto** — GOTO-style GEMM: B-panel packing, GEMV/GEMM dispatch
9. **prefill** — Worker-based parallelism, A-panel packing, 8×24 microkernel
10. **prefill_opt** — Optimized v2 microkernel with improved tiling
11. **decode** — k-parallel GEMV with reduction for memory-bandwidth-bound shapes
12. **dispatch** — Auto-selects decode (M < 8) or prefill_opt based on shape

## Project structure

```
├── gemm.mojo            # All 12 GEMM kernel implementations
├── matrix.mojo          # Matrix[dtype] struct (row-major 2D buffer)
├── bench_matmul.mojo    # Benchmark all kernels on decode & prefill shapes
├── bench_linalg.mojo    # Mojo stdlib linalg.matmul baseline
├── bench_sota.py        # NumPy / SciPy dgemm / Intel MKL benchmarks
├── test_gemm.mojo       # Correctness tests against naive reference
├── main.mojo            # Minimal demo (2×2 multiply)
└── setup.sh             # One-command install (uv + Mojo nightly)
```

## Prerequisites

- Python >= 3.12
- An internet connection (setup downloads uv and the Mojo nightly toolchain)
- NumPy / SciPy (optional, only needed for `bench_sota.py`)

## Setup

```bash
bash setup.sh
```

This installs [uv](https://github.com/astral-sh/uv) (if not present), creates a `.venv`, and installs the Mojo nightly compiler via `modular`.

## Run

```bash
source .venv/bin/activate
mojo bench_matmul.mojo        # All 12 kernels on both shapes
mojo bench_linalg.mojo        # Mojo stdlib linalg.matmul baseline
python bench_sota.py           # NumPy/SciPy/MKL benchmarks
mojo test_gemm.mojo           # Correctness tests
```

## Optimization techniques

Each kernel builds on the previous one, adding a new optimization:

| Technique | Kernel(s) | Key idea |
|---|---|---|
| Cache blocking | tiled | 32×32 tiles that fit in L1/L2 |
| SIMD vectorization | simd | `vectorize[]` over the inner dimension |
| Thread parallelism | parallel | `parallelize[]` across tile rows |
| Register blocking | register_blocked, packed | MR×NR micro-tiles held in registers |
| Compile-time specialization | comptime | `comptime` parameters for loop unrolling |
| Data packing | goto, prefill | Repack A/B panels for sequential access |
| Software prefetching | comptime, prefill | `prefetch[]` intrinsic with configurable locality |
| Shape dispatch | dispatch | Auto-selects GEMV (M < 8) vs GEMM path |

## Interpreting results

All performance numbers are hardware-specific. Theoretical peak for double-precision FMA:

```
peak GFLOPS = clock_GHz × doubles_per_SIMD × 2 (FMA) × cores
```

For example: 2.8 GHz × 8 (AVX-512) × 2 × 4 cores = 179.2 GFLOPS.

Identify your hardware before comparing:

```bash
lscpu | grep -E "Model name|CPU\(s\)|MHz|cache|Flags"
```
