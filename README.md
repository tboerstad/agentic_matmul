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
10. **prefill_opt** — v3 microkernel (hoisted B-load + noalias), 6×32 register tile, KU=2, KC=256, TILE_N=128
11. **decode** — j-parallel GEMV with L1-resident column chunks for decode shapes
12. **dispatch** — Shape-adaptive auto-selection (see below)

## Dispatch logic

`matmul_dispatch` routes each shape to the kernel and tile that measured best for
it (see the comment in `matmul_dispatch` for the authoritative table):

- `M·N·K < 2^19` (tiny): serial register-blocked `_matmul_small` — no thread
  launch, allocation, or packing. Avoids the fixed parallel-kernel overhead that
  made tiny GEMMs 7–30× slower than `linalg` (sq8 0.03× → 2.37× after).
- `M == 1`: decode GEMV (streams B once).
- `2 ≤ M ≤ 5`: v3 micro-kernel with `MR = M` — packs B once and reuses it across
  all rows (vs the GEMV re-streaming all of B per row, ~2× slower at M=4).
- **Small box, M-dominant, B fits L2** (`M ≥ 64`, `M ≥ N`, `K·N·8 ≤ 512 KB`):
  M-parallel, no-packing `_thin_n_gemm` — each core owns a band of C's rows and
  sweeps the full N, reading A/B straight from source (B stays L2-resident, so
  packing buys nothing). The packed prefill kernel's packing + thread-launch
  overhead dwarfs the tiny compute on these cache-resident shapes; this branch
  pays none of it. Fixes the worst shapes in the general sweep — sq128 0.65→1.0,
  sq96 0.71→1.16, 512×128×512 0.56→0.84. The `M ≥ N` gate keeps it unreachable
  for every wide/headline shape (Qwen up/down proj are `N ≫ M`).
- `M ≥ 6`: parallel `_prefill_gemm_v3` (N-parallel: each worker owns a band of
  j-tiles, reading the dominant B matrix from DRAM once into private L2). Tile and
  KC are selected per regime — narrow NR=16 for `N ≤ 192` (so a small N still
  splits into ≥ num_workers j-tiles), 8×24 only for very-wide-N small-M (Qwen
  up-proj small batch), 6×32 elsewhere; KC scaled up where the L2 allows it.

Tunable parameters (tile MR×NR, KC, KU) are hardware-specific — see notes below.

## Current standing vs `linalg.matmul`

Validated on the Qwen MLP sweep (M = 1..512, both projection orientations) on
the 2.10 GHz Xeon (4 cores, AVX-512, float64), measured interleaved so both
kernels see the same turbo/thermal state:

- **21/22 Qwen shapes WIN.** Decode (M=1) wins ~2.4–2.6×; small/mid-M wins
  1.05–3.4×; the lone exception is down-proj M=512 at ~0.99 (parity).
- Beats NumPy (OpenBLAS) and SciPy dgemm on all 22 shapes.
- General-shape `--full` sweep: a clear majority WIN after the small-N /
  wide-N-small-M / square-large-M / small-box fixes (the exact tally swings
  ±several with VM turbo/thermal, so judge per-shape via interleaved A/B, not the
  single-run count). Two recent levers: (1) the square-ish branch now picks KC
  by detected L2 (KC=512 on the 1 MB Skylake, KC=1024 on the 2 MB Xeon — each
  the measured best on its machine; a single hardcoded KC=1024 had sunk Skylake
  sq2048 to ~0.72) plus a load-balance-aware TILE_N, holding the large squares at
  ~0.84 on Skylake / ~0.88–0.91 on the Xeon; (2) the small-box M-parallel route
  flipped the two worst shapes in the whole sweep — square sq96/sq128, which
  interleaved-A/B (peak/40) lifts 0.65–0.71 → 1.0–1.16 — plus the tall
  cache-resident boxes (512×128×512 0.56→0.84, 256×128×512 0.63→0.90). Residual
  losses are the mid/large squares (sq256 ~0.80, sq512 ~0.81) and low-arithmetic-
  intensity corners (K=128, odd N/K) where `linalg`'s masked AVX-512 remainder
  handling wins.

### Key findings from tuning

The micro-kernel itself is not the bottleneck on most shapes: `_prefill_gemm_v3`
and `linalg.matmul` compile to the same 6×32/8×24 register tile running
`vfmadd231pd` at ~2 FMA/cycle. The wins and losses came from everything *around*
it:

- **Load balance.** The kernel parallelizes over N (j-tiles), so the tile must
  split N evenly across workers. `N=2048` balances perfectly with NR=32
  (TILE_N=64 → 32 even tiles); a small N starves cores unless NR shrinks. Most
  per-shape wins trace back to picking a tile that tiles N cleanly.
- **Masked M-remainder.** Handling `m % MR` leftover rows as one register-blocked
  masked block (instead of row-by-row) freed the tile choice from needing
  `MR | M`, flipping 5 losing shapes to wins. Verified bit-identical
  (`verify_dispatch` max_err 0.0).
- **KC is cache-size-dependent.** On the 1 MB-L2 Skylake, KC=512 is best; on the
  2 MB-L2 Xeon, KC=2048 (fewer k-panels, less C re-traffic) wins large-M — the
  *opposite* conclusion. Every KC/tile pick in the file is hardware-specific; the
  large-M and square-ish branches resolve the split at runtime from the detected
  L2 (a hardcoded square-ish KC=1024 that helped the Xeon cost the Skylake ~15%
  on sq2048), reserving the cpuid probe for shapes big enough to absorb its cost.
- **Packing is pure overhead when the whole problem is cache-resident.** For a
  small box where B fits L2, the packed prefill kernel's A/B packing + per-worker
  buffer alloc + parallelize launch cost more than the matmul itself; routing to
  the M-parallel no-pack kernel won 1.0–1.6× where the packed path lost 0.65–0.79.
- **Don't probe the hardware on the hot path.** `l2_cache_size()` issues ~6
  `cpuid` instructions, which on a KVM guest trap to the hypervisor at **~61 µs
  per call** — fatal on a few-µs GEMM (it sank sq96 to 0.19× when the dispatch
  gate queried it live). The small-box gate uses a compile-time L2 constant; the
  live query stays in the large-M branches where the op is milliseconds.

### Dead ends (measured, do not re-attempt without new evidence)

- **2-D `MC×KC` cache blocking** and **2-D `(M,N)` worker split** — both built and
  measured *slower*. On these large-L3 parts (33–260 MB) A already fits in L3, so
  the "redundant" A re-reads v3 does are cheap L3 traffic, not DRAM; splitting M
  only adds redundant B-packing. Packed-A L2 residency is not the bottleneck.
- **In-kernel packed-B prefetch** — neutral-to-harmful (0.98–1.01); ILP + HW
  prefetcher already hide the L2→L1 latency.
- **Masked-SIMD / zero-edge tiles** — a zero-edge 4×32 lost at every M, so the
  scalar M-remainder is not the bottleneck.
- **`SHARED_A`** (pack A once, share across N-workers) — a wash; A fits in L3.

### Still open

Micro-kernel parity with `linalg` on the heaviest GEMMs (square M ≥ 256, now
~0.88–0.91 after the KC/TILE_N retune, where both kernels sit at ~50–66% of the
358 GFLOPS f64 peak). Closing the remaining gap needs `linalg`-style masked
AVX-512 N-remainder handling and/or pack/compute overlap — substantial and
unproven on this hardware. The thin-N and small-box cache-resident gaps are now
both handled by the M-parallel no-pack route; its L2-fit cut (512 KB) is a
Skylake-tuned constant that a smaller-L2 part may want lowered.

> **Methodology note:** ratios from a single `bench_sweep` run swing ±5–10% at
> M ≥ 128 from turbo/thermal state on shared VMs. Judge micro-kernel changes with
> an interleaved A/B of the two variants (peak GFLOPS over ~15–20 runs), never by
> comparing absolute numbers across runs.

## Setup

```bash
bash setup.sh
```

## Run

```bash
source .venv/bin/activate
mojo bench_matmul.mojo           # All 12 kernels on both shapes
mojo bench_linalg.mojo           # Mojo stdlib linalg.matmul baseline
mojo bench_sweep.mojo --iterate  # FAST: dispatch vs linalg on corner/edge shapes (default)
mojo bench_sweep.mojo --full     # SLOW: full per-M sweep over many aspect ratios + general grid
python bench_sota.py             # NumPy/SciPy/MKL benchmarks
mojo test_gemm.mojo              # Correctness tests
mojo verify_dispatch.mojo        # dispatch correctness vs naive (all branches + edge cases)
```
