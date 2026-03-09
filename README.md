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
| **Mojo dispatch** | — | — | **189.9** |
| Mojo prefill_opt | — | — | 190.8 |
| Mojo goto | 82.9 | 192.7 | 144.9 |
| Mojo linalg (stdlib) | — | — | 104.9 |
| Mojo comptime | 38.4 | 45.5 | 92.1 |
| Mojo naive | 0.18 | 0.24 | 1.55 |

### Decode (1 × 11008 × 2048)

| Kernel | Xeon Skylake 2.80 GHz (4c) | Xeon Granite Rapids 2.10 GHz (4c) | Apple M4 Max (14c) |
|---|---|---|---|
| NumPy (Accelerate/OpenBLAS) | 11.3 | 14.3 | **54.3** |
| **Mojo dispatch** | — | 20.1 | **20.7** |
| Mojo goto | 0.72 | 12.4 | 14.2 |
| Mojo linalg (stdlib) | — | — | 4.8 |
| Mojo naive | 0.16 | 0.18 | 1.59 |

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

## Setup

```bash
bash setup.sh
```

## Run

```bash
source .venv/bin/activate
mojo bench_matmul.mojo        # All 12 kernels on both shapes
mojo bench_linalg.mojo        # Mojo stdlib linalg.matmul baseline
python bench_sota.py           # NumPy/SciPy/MKL benchmarks
mojo test_gemm.mojo           # Correctness tests
```
