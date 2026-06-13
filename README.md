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
12. **dispatch** — Shape-adaptive auto-selection (tuned vs `linalg.matmul`):
    - `M == 1`: decode GEMV (streams B once)
    - `2 ≤ M ≤ 5`: v3 micro-kernel with `MR = M` — packs B once and reuses it
      across all rows (the old GEMV re-streamed B per row, ~2× slower at M=4)
    - `M ≥ 6`, wide-N (`N ≥ K`): 8×24 tile (`M ≤ 192`) or 4×48 tile (`M > 192`)
    - `M ≥ 6`, tall-K (`N < K`): 8×24 tile (`M ≤ 64`) or 6×32 tile, KC=512

### Beating the stdlib `linalg.matmul`

`bench_compare.mojo` measures the agentic kernels and `linalg.matmul`
*interleaved per shape*, so both see the same turbo/thermal state (a long
serial sweep otherwise biases the short-running contestant). On the Skylake
AVX-512 cloud VM (4 cores, float64), the tuned `dispatch` beats `linalg` on all
small/decode shapes — decisively (≈2× at M=1, ≈1.2× at M=4) — and across the
up-projection up to M=128. It still trails `linalg` ~10% on the down-projection
at large M: `_prefill_gemm_v3` parallelizes over N, so each worker re-packs all
of A, which is wasteful when K (and thus A) is large. An M-parallel packing
scheme is the remaining work to close that gap.

## Setup

```bash
bash setup.sh
```

`setup.sh` installs the latest Mojo nightly from the nightly wheel index
(`https://whl.modular.com/nightly/simple/`) — currently MAX 26.5
(Mojo 1.0.0b3, validated against `26.5.0.dev2026061206`). All sources build
warning-free under it. To pin the last stable release instead, install
`modular==26.3` from the stable index (`https://whl.modular.com/simple/`),
which ships Mojo 1.0.0b1.

## Run

```bash
source .venv/bin/activate
mojo bench_matmul.mojo        # All 12 kernels swept across many shapes
mojo bench_linalg.mojo        # Mojo stdlib linalg.matmul baseline
mojo bench_compare.mojo       # Fair head-to-head: our kernels vs linalg, interleaved per shape
python bench_sota.py           # NumPy/SciPy/MKL benchmarks (same shape sweep)
mojo test_gemm.mojo           # Correctness tests
```

### Shape sweep

`bench_matmul.mojo` and `bench_sota.py` both sweep the token dimension
`M ∈ {1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512}` across both MLP projection
orientations — gate/up (`K=2048 → N=11008`) and down (`K=11008 → N=2048`) — so
the comparison spans the full memory-bound (decode, `M=1`) to compute-bound
(prefill, large `M`) range rather than just the two headline shapes. The Mojo
benchmark adapts its timed-run count to the problem size and prints a
peak-GFLOPS summary table at the end (per-shape `decode` / `prefill_opt` /
`dispatch` columns plus the best kernel for that shape); the two headline
shapes additionally report the full 12-kernel breakdown including the naive
baseline. This makes the `matmul_dispatch` decode/prefill crossover (at `M=6`)
and the points where the auto-dispatcher trails the best hand-picked kernel
directly visible.
