# matmul

High-performance matrix multiplication kernels in Mojo, benchmarked against SOTA libraries (NumPy/OpenBLAS, SciPy, Intel MKL).

**Target shapes:** Qwen 2.5 VL 3B MLP projections (float64)
- Decode: 1 × 11008 × 2048 (memory-bandwidth bound)
- Prefill: 96 × 11008 × 2048 (compute-bound)

## Results (2026-03-08)

### Prefill (96 × 11008 × 2048) — GFLOPS (higher is better)

| Kernel | GFLOPS (peak) | vs OpenBLAS |
|--------|--------------|-------------|
| OpenBLAS (NumPy) | **196.0** | 1.00× |
| **Mojo prefill** | **125.4** | **0.64×** |
| Mojo goto | 82.9 | 0.42× |
| Mojo comptime | 38.4 | 0.20× |
| Mojo packed | 31.6 | 0.16× |
| Mojo naive | 0.18 | 0.001× |

### Decode (1 × 11008 × 2048) — GFLOPS (higher is better)

| Kernel | GFLOPS (peak) | vs OpenBLAS |
|--------|--------------|-------------|
| OpenBLAS (NumPy) | **11.33** | 1.00× |
| **Mojo goto** | **0.72** | **0.06×** |
| Mojo naive | 0.16 | 0.01× |

See [SOTA_RESULTS.md](SOTA_RESULTS.md) for full benchmark tables and [OPENBLAS_ANALYSIS.md](OPENBLAS_ANALYSIS.md) for technical analysis.

## Kernel Evolution

1. **naive** — Triple-nested loop baseline
2. **tiled** — 32×32 cache-blocking
3. **simd** — Tiled + SIMD vectorization
4. **parallel** — Tiled + thread parallelism
5. **register_blocked** — Higher loop unrolling
6. **packed** — A/B buffer packing for sequential access
7. **comptime** — Compile-time parameter specialization
8. **goto** — GOTO-style GEMM: B-panel packing, GEMV/GEMM dispatch
9. **prefill** — Worker-based parallelism, A-panel packing, 8×24 microkernel, NR=24, KC=512

## Setup

```bash
bash setup.sh
```

## Run

```bash
source .venv/bin/activate
mojo bench_matmul.mojo        # All kernel versions (decode + prefill)
mojo bench_prefill.mojo       # Dedicated goto vs prefill comparison
python bench_sota.py           # SOTA library benchmarks
mojo test_gemm.mojo           # Correctness tests
```
