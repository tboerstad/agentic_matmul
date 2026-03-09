# matmul

An experiment in writing optimized matmul kernels in Mojo using only [Claude Code](https://claude.com/claude-code) on mobile (iOS). The sole goal is maximizing GFLOPS on the two matrix shapes used by Qwen 2.5 VL 3B MLP projections (float64):

- **Decode:** 1 × 11008 × 2048 (memory-bandwidth bound)
- **Prefill:** 96 × 11008 × 2048 (compute-bound)

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

## Key optimization techniques

- **Cache blocking/tiling** — Partition C into tiles that fit in L1/L2 cache
- **SIMD vectorization** — Process 8 float64 values per instruction (AVX-512) via `vectorize`
- **Register blocking** — Hold MR×NR micro-tiles of C in registers across the k-loop
- **Buffer packing** — Repack A/B panels into contiguous memory to eliminate stride overhead
- **Loop unrolling** — Compile-time `comptime for` with KU=4–8 to reduce loop overhead
- **Software prefetching** — LLVM prefetch intrinsics to hide memory latency
- **Fused multiply-add** — Explicit `fma()` for single-instruction multiply-accumulate
- **Multi-threading** — `parallelize` with per-worker private buffers to avoid synchronization
- **Shape-based dispatch** — Automatically select bandwidth-optimal (GEMV) or compute-optimal (GEMM) kernel

## Project structure

```
gemm.mojo            Core GEMM kernels (all 12 implementations)
matrix.mojo          Generic 2D matrix container (float32/float64)
bench_matmul.mojo    Benchmark all 12 kernels on both shapes
bench_linalg.mojo    Mojo stdlib linalg.matmul baseline
bench_sota.py        NumPy/SciPy/MKL comparison benchmarks
test_gemm.mojo       Correctness tests (2×2, 1×1, non-square)
main.mojo            Minimal entry point
setup.sh             One-command install (uv + Mojo nightly)
```

## Requirements

- Python >= 3.12
- Mojo nightly (installed automatically by `setup.sh`)

For SOTA comparison benchmarks (`bench_sota.py`):
- NumPy (with OpenBLAS or Accelerate backend)
- SciPy (optional, for direct BLAS dgemm)

## Setup

```bash
bash setup.sh
```

This installs [uv](https://github.com/astral-sh/uv) (if needed), creates a virtual environment, and installs the Mojo nightly compiler.

## Run

```bash
source .venv/bin/activate
mojo bench_matmul.mojo        # All 12 kernels on both shapes
mojo bench_linalg.mojo        # Mojo stdlib linalg.matmul baseline
python bench_sota.py           # NumPy/SciPy/MKL benchmarks
mojo test_gemm.mojo           # Correctness tests
```

## Understanding the numbers

Theoretical peak GFLOPS for a given CPU:

```
peak = clock_GHz × doubles_per_SIMD × 2 (FMA) × cores
```

For example, a 4-core Xeon @ 2.8 GHz with AVX-512: 2.8 × 8 × 2 × 4 = **179.2 GFLOPS**.

Results will vary across hardware — clock speed, core count, cache hierarchy, SIMD width, and BLAS backend all affect throughput.
