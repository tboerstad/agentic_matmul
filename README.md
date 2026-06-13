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
10. **prefill_opt** — v3 microkernel (hoisted B-load + noalias) with 6×32 register tile, KU=2, KC=256, TILE_N=128 — tuned by empirical scan
11. **decode** — j-parallel GEMV with L1-resident column chunks for decode shapes
12. **dispatch** — Shape-adaptive auto-selection (see the comment in
    `matmul_dispatch`):
    - `M == 1`: decode GEMV (streams B once)
    - `2 ≤ M ≤ 5`: v3 micro-kernel with `MR = M` — packs B once and reuses it
      across all rows (the old path routed M<6 to the GEMV, which re-streamed
      all of B once *per row*, ~2× slower at M=4)
    - `M ≥ 6`, wide-N (`N ≥ K`): 8×24 tile (`M ≤ 192`) or 4×48 tile (`M > 192`)
    - `M ≥ 6`, tall-K (`N < K`): 8×24 tile (`M ≤ 64`) or 6×32 tile, KC=512

### Note: dispatch retune vs `linalg.matmul` (Jun 2026)

`matmul_dispatch` was retuned into the shape-adaptive selector above, validated
on the Qwen MLP sweep (M = 1..512, both projection orientations) on a Skylake
AVX-512 cloud VM (4 cores, float64), measured fairly (our kernels and the
contestant interleaved per shape so both see the same turbo/thermal state):

- Beats stdlib `linalg.matmul` on ~14/22 shapes — decisively on every
  decode/small-batch shape (≈2.3× at M=1, ≈1.25–1.5× at M=2–8) and across the
  up-projection through M≈128.
- Beats NumPy (OpenBLAS) and SciPy dgemm on all 22 shapes.
- Still trails `linalg` ~7–11% on the heaviest GEMMs (down-proj at large M,
  up-proj M ≥ 256). That gap is micro-kernel maturity, not memory layout:
  `_prefill_gemm_v3`'s N-parallel scheme already reads the (dominant) B matrix
  from DRAM exactly once into private L2. An M-parallel rewrite was built and
  measured — it was 2–3× *slower* (shared B falls to L3 + per-K-panel launch
  overhead), confirming N-parallel is the right structure here.

### Note: where we still lose, and what does *not* close it (Jun 2026 re-audit)

Re-measured on the Xeon @ 2.10 GHz cloud VM (4 cores, AVX-512, float64) with
`bench_sweep.mojo`, which interleaves `matmul_dispatch` against `linalg.matmul`
per shape. **Both headline shapes win** (decode 1×11008×2048 ≈ 2.0×, prefill
96×11008×2048 ≈ 1.07×). The remaining losses are confined to:

| Shape | ratio (dispatch / linalg) |
|---|---|
| down-proj M=64..512 | 0.89 – 0.92 |
| up-proj M=256 / 512  | 0.97 / 0.93 |

The README previously blamed the down-proj gap on `_prefill_gemm_v3` re-packing
all of A per N-worker, and suggested an M-parallel / shared-A packing scheme as
the fix. **That diagnosis turned out to be wrong on this hardware.** A shared
single-pack-of-A variant (`SHARED_A`) was implemented and measured head-to-head:
it is a *wash* (+1% at M=128, −0.4% at M=512, noise elsewhere). The reason is
that A fits comfortably in the 260 MB L3, so the "redundant" re-reads are L3
traffic, not DRAM — and at these sizes the kernel is **compute-bound**, not
A-packing-bound. An exhaustive sweep of the micro-kernel tile (MR×NR), KC, KU,
and NC_TILES likewise found the current configs already optimal. The residual
gap is genuine micro-kernel maturity vs linalg's hand-tuned AVX-512 kernel.
(Shared A-packing could still pay off on hardware where A exceeds L3.)

## Setup

```bash
bash setup.sh
```

## Run

```bash
source .venv/bin/activate
mojo bench_matmul.mojo        # All 12 kernels on both shapes
mojo bench_linalg.mojo        # Mojo stdlib linalg.matmul baseline
mojo bench_sweep.mojo         # dispatch vs linalg, per-M, both orientations (finds losing dims)
python bench_sota.py           # NumPy/SciPy/MKL benchmarks
mojo test_gemm.mojo           # Correctness tests
mojo verify_dispatch.mojo     # dispatch correctness vs naive (all branches + edge cases)
```
