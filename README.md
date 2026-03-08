# matmul

High-performance matrix multiplication kernels in Mojo, benchmarked against SOTA libraries (NumPy/OpenBLAS, SciPy, Intel MKL).

**Target shapes:** Qwen 2.5 VL 3B MLP projections (float64)
- Decode: 1 × 11008 × 2048 (memory-bandwidth bound)
- Prefill: 96 × 11008 × 2048 (compute-bound)

## Results (2026-03-08)

> **These results are hardware-specific.** Measured on two machines:
> **A:** Intel Xeon @ 2.80 GHz (Skylake), **B:** Intel Xeon @ 2.10 GHz (Granite Rapids).
> Both 4 cores, AVX-512, KVM. Your numbers will differ — see [AGENTS.md](AGENTS.md).

### Prefill (96 × 11008 × 2048) — GFLOPS peak (higher is better)

| Kernel | Machine A | vs SOTA | Machine B | vs SOTA |
|--------|----------|---------|----------|---------|
| OpenBLAS (NumPy) | **196.0** | 1.00× | **239.4** | 1.00× |
| **Mojo prefill** | **125.4** | **0.64×** | **217.9** | **0.91×** |
| Mojo goto | 82.9 | 0.42× | 192.7 | 0.81× |
| Mojo comptime | 38.4 | 0.20× | 45.5 | 0.19× |
| Mojo packed | 31.6 | 0.16× | 40.7 | 0.17× |
| Mojo naive | 0.18 | 0.001× | 0.24 | 0.001× |

### Decode (1 × 11008 × 2048) — GFLOPS peak (higher is better)

| Kernel | Machine A | vs SOTA | Machine B | vs SOTA |
|--------|----------|---------|----------|---------|
| OpenBLAS (NumPy) | **11.33** | 1.00× | **14.32** | 1.00× |
| **Mojo dispatch** | — | — | **20.08** | **1.40×** |
| Mojo decode | — | — | 18.61 | 1.30× |
| Mojo goto | 0.72 | 0.06× | 12.39 | 0.87× |
| Mojo naive | 0.16 | 0.01× | 0.18 | 0.01× |

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
10. **decode** — k-parallel GEMV with reduction for memory-bandwidth-bound shapes
11. **dispatch** — Auto-selects decode (M ≤ 7) or prefill kernel based on shape

## Setup

```bash
bash setup.sh
```

## Run

```bash
source .venv/bin/activate
mojo bench_matmul.mojo        # All kernel versions (decode + prefill)
mojo bench_prefill.mojo       # Dedicated goto vs prefill comparison
mojo bench_decode.mojo        # Dedicated goto vs decode vs dispatch comparison
python bench_sota.py           # SOTA library benchmarks
mojo test_gemm.mojo           # Correctness tests
```
