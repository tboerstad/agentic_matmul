# matmul

An experiment in writing optimized matmul kernels in Mojo using only [Claude Code](https://claude.com/claude-code) on mobile (iOS). The sole goal is maximizing GFLOPS on the two matrix shapes used by Qwen 2.5 VL 3B MLP projections (float64):

- **Decode:** 1 × 11008 × 2048 (memory-bandwidth bound)
- **Prefill:** 96 × 11008 × 2048 (compute-bound)

## Results

Peak GFLOPS by hardware (higher is better):

### Prefill (96 × 11008 × 2048)

| Kernel | Xeon Skylake 2.80 GHz (4c) | Xeon Emerald Rapids 2.10 GHz (4c) | Apple M4 Max (14c) |
|---|---|---|---|
| SciPy dgemm | 144.6 | 200.8 | **538.1** |
| NumPy (Accelerate/OpenBLAS) | 216.9 | 235.6 | **483.1** |
| **Mojo (agentic matmul)** | 208.4 | **256.6** | 189.9 |
| Mojo linalg (stdlib) | 182.4 | **247.5** | 104.9 |

### Decode (1 × 11008 × 2048)

| Kernel | Xeon Skylake 2.80 GHz (4c) | Xeon Emerald Rapids 2.10 GHz (4c) | Apple M4 Max (14c) |
|---|---|---|---|
| SciPy dgemm | **5.5** | 8.4 | — |
| NumPy (Accelerate/OpenBLAS) | 13.4 | 25.0 | **54.3** |
| **Mojo (agentic matmul)** | 13.9 | **28.5** | 20.7 |
| Mojo linalg (stdlib) | 5.9 | 11.4 | 4.8 |

### Tuning history on Skylake AVX-512 (cloud VM, 4 cores)

Recent retune of the prefill register tile from 8×24 (KU=4) to 6×32 (KU=2) on
the cloud Skylake VM:

| Kernel | Prefill peak GFLOPS | Decode peak GFLOPS |
|---|---|---|
| NumPy OpenBLAS                                   | 170.9 | 11.3 |
| dispatch (pre-tune, MR=8 NR=24 KU=4)             | 163.7 | 10.0 |
| **dispatch (this VM SOTA, MR=6 NR=32 KU=2)**     | **175.7** | **10.0** |

(Prefill peaks above 170 GFLOPS beat the OpenBLAS reference for this shape on
this hardware; decode is DRAM-bandwidth bound near ~30 GB/s aggregate.)

### Decode GEMV prefetch + even j-split on Skylake AVX-512 (cloud VM, 4 cores)

Measured on a Skylake-class cloud VM (Xeon @ 2.80 GHz, 4 cores, AVX-512 KVM —
faster than the VM in the table above), peak across ≥10 `matmul_dispatch`
invocations:

| Kernel | Prefill peak GFLOPS | Decode peak GFLOPS |
|---|---|---|
| dispatch (before)                                | 223.2 | 14.9 |
| **dispatch (this VM SOTA, GEMV prefetch + TILE_N=64)** | **229.8** | **16.8** |

Two changes: (1) the decode GEMV now software-prefetches the next KU-block of
B rows — the 8 read streams sit `n*8` bytes apart, too far for the hardware
prefetcher to track, so explicit prefetch is worth +13% peak / +30% mean on
decode; (2) prefill `TILE_N` drops from 128 to 64, giving 172 j-tiles that
split exactly 43 per worker on 4 cores (128 gave 86 tiles split 22/22/22/20,
idling one core ~9% of the time).

### Mojo 1.0.0b2 migration cost on Emerald Rapids 2.10 GHz (cloud VM, 4 cores)

Same VM, same shapes, `std.benchmark.run` peak across 4 invocations. The
pre-migration row builds with the March 13 2026 Mojo nightly
(`26.3.0.dev2026031305`); the post-migration row uses Mojo 1.0.0b2
(`1.0.0b2.dev2026051606`):

| Kernel | Prefill peak GFLOPS | Decode peak GFLOPS |
|---|---|---|
| pre-migration code + Mojo dev2026031305          | **269.0** | **32.0** |
| post-migration code + Mojo 1.0.0b2               | 256.6 | 28.5 |

The migration costs ~5% on prefill and ~11% on decode. About half of the
decode delta is Mojo 1.0.0b2 codegen drift (the unchanged `linalg.matmul`
stdlib kernel regresses ~10% across the same compiler bump); the rest is
residual capture-list / hoisted-binding overhead from the syntax migration.

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
10. **prefill_opt** — v3 microkernel (hoisted B-load + noalias) with 6×32 register tile, KU=2, KC=256, TILE_N=64 — tuned by empirical scan
11. **decode** — j-parallel GEMV with L1-resident column chunks and software prefetch of the next KU-block of B rows
12. **dispatch** — Auto-selects decode (M < 6) or prefill_opt based on shape

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
