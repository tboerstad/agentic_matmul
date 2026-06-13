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
    - `M ≥ 6`, wide-N (`N ≥ K`): 8×24 tile (`M ≤ 192`), 4×48 tile
      (`192 < M ≤ 256`), or 6×32 tile (`M > 256`)
    - `M ≥ 6`, tall-K (`N < K`): 8×24 tile (`M ≤ 64`) or 6×32 tile, KC=512

### Note: dispatch retune vs `linalg.matmul` (Jun 2026)

`matmul_dispatch` was retuned into the shape-adaptive selector above, validated
on the Qwen MLP sweep (M = 1..512, both projection orientations) on a Skylake
AVX-512 cloud VM (4 cores, float64), measured fairly (our kernels and the
contestant interleaved per shape so both see the same turbo/thermal state):

- Beats stdlib `linalg.matmul` on 17/22 shapes — decisively on every
  decode/small-batch shape (≈1.9–2.4× at M=1, ≈1.2–1.3× at M=2–8), across the
  up-projection through M≈128, and on the full down-projection through M=96
  (after the load-balance fix below).
- Beats NumPy (OpenBLAS) and SciPy dgemm on all 22 shapes.
- Still trails `linalg` ~4–11% on the 5 heaviest GEMMs (down-proj M ≥ 128,
  up-proj M ≥ 256). That residual is micro-kernel maturity, not memory layout:
  `_prefill_gemm_v3`'s N-parallel scheme already reads the (dominant) B matrix
  from DRAM exactly once into private L2. An M-parallel rewrite was built and
  measured — it was 2–3× *slower* (shared B falls to L3 + per-K-panel launch
  overhead), confirming N-parallel is the right structure here.

### Note: down-proj mid-M was a load-balance bug, not micro-kernel maturity (Jun 2026)

An assembly-level re-audit on the Skylake @ 2.80 GHz cloud VM (4 cores, AVX-512,
float64) found that `_prefill_gemm_v3` and `linalg.matmul` **compile to the same
micro-kernel** — a 6×32 (or 8×24) register tile with 24 `zmm` accumulators that
never spill, B-loads hoisted, A broadcast, running `vfmadd231pd` at the
theoretical 2 FMA/cycle peak. The compute core is not where we lose.

The real losses were in everything *around* the micro-kernel, and the biggest
one was a pure scheduling artifact. The down-proj `M ≤ 64` branch used
`TILE_N = 9*NELTS = 72`, which splits N=2048 into **29 j-tiles → 8/8/8/5 across
4 workers**, idling one core ~10% of the time. That cost 7–8% at M=32..64 and
showed up as a sharp cliff (down-proj M=8..64 ratios ~0.81–0.83, while the
already-balanced M≥96 path sat near 0.99). Switching that branch to
`TILE_N = 6*NELTS = 48` (43 j-tiles ≈ 11/11/11/10, a clean split) flips all of
M=8..64 from LOSE to **WIN** (≈1.02–1.15× vs linalg) with no micro-kernel change.
This is the same even-split lesson `matmul_prefill_opt` already applied to the
prefill shape — it just hadn't been carried over to the tall-K branch.

After the fix, the remaining losses are confined to the largest GEMMs:

| Shape | ratio (dispatch / linalg) |
|---|---|
| down-proj M=128 / 256 / 512 | 0.96 / 0.89 / 0.90 |
| up-proj   M=256 / 512       | 0.96 / 0.90 |

The assembly suggested two things `linalg` does that we don't: masked AVX-512
load/store on remainder tiles (vs our scalar `vectorize` tail) and a packed-B
prefetch inside the micro-kernel. **Both were implemented and A/B-tested
head-to-head on these shapes, and neither closes the gap on this hardware:**

- *In-kernel packed-B prefetch* — gated behind a comptime flag and measured
  on/off, interleaved: ratio 0.98–1.01 (neutral, slightly *harmful* at M=512).
  The L2→L1 latency is already hidden by the 28-accumulator ILP and the hardware
  prefetcher, so the explicit prefetch only adds port pressure.
- *Edge handling* — if the scalar M-remainder were the bottleneck, a zero-edge
  tile would win. It loses: `4×32` (which divides M=128/256/512 evenly, no
  remainder) ran **slower than the current `6×32`** at every M, because the
  lower compute intensity and extra C-traffic cost more than the remainder saves.
  So the remainder is not where the time goes.
- *KC sweep* — `6×32` at KC ∈ {192,256,384,512}: KC=512 is already
  monotonically best for M ≥ 192; smaller KC helps only marginally at M=96–128
  (within the run-to-run noise floor, which is ±5–10% at these sizes).

So the residual ~9–12% at M ≥ 256 is **not** any of the cheap micro-kernel
tweaks — it is full BLIS-style 2-D cache blocking (an `MC × KC` loop that keeps
the packed-A panel resident in L2 as M grows; we currently pack all of M per
worker). That is a substantial rewrite, deferred until the headline shapes need
it. (An earlier theory blamed `_prefill_gemm_v3` re-packing A per N-worker; a
`SHARED_A` variant was measured and came out a wash — A fits in L3, so the
re-reads are L3 traffic, not DRAM.)

> Methodology note: ratios from a *single* `bench_sweep` run swing ±5–10% at
> M ≥ 128 from turbo/thermal state on this shared VM. Judge micro-kernel changes
> with an interleaved A/B of the two variants (peak GFLOPS over ~15–20 runs),
> never by comparing absolute numbers across runs.

### Remeasurement (Jun 13 2026): down-proj mid-M now sits on the noise floor

Re-ran `bench_sweep.mojo` on the Skylake @ 2.80 GHz cloud VM (4 cores, AVX-512,
float64, Mojo 1.0.0b3 `dev2026061206`). The first sweep was discarded as a cold
start (absolute GFLOPS ~20% below warm state); the table below is the **median
ratio across 3 warm runs** (`dispatch / linalg`, peak GFLOPS per shape):

| M | up-proj (N=11008 K=2048) | down-proj (N=2048 K=11008) |
|---|---|---|
| 1   | 2.63 WIN | 2.44 WIN |
| 2   | 1.21 WIN | 1.13 WIN |
| 4   | 1.18 WIN | 1.10 WIN |
| 8   | 1.18 WIN | 0.99 ~tie |
| 16  | 1.18 WIN | 0.97 LOSE |
| 32  | 1.20 WIN | 0.99 ~tie |
| 64  | 1.09 WIN | 0.96 LOSE |
| 96  | 1.07 WIN | 1.02 WIN |
| 128 | 1.03 WIN | 0.95 LOSE |
| 256 | 0.93 LOSE | 0.88 LOSE |
| 512 | 0.91 LOSE | 0.90 LOSE |

Two things changed versus the post-load-balance-fix snapshot documented above:

- **Up-proj improved at the top of the winning band.** Dispatch now wins cleanly
  through M=128 (1.03–1.07×, previously a slight LOSE at M=96/128). M=256/512 are
  unchanged losses (~0.91–0.93×).
- **Down-proj mid-M (M=8..64) regressed from "decisive WIN" back to the noise
  floor.** The `TILE_N=48` even-split fix claimed ≈1.02–1.15× here; the
  remeasured median is ~0.96–0.99× — a marginal tie that flips WIN/LOSE between
  runs (e.g. M=8: 1.03 / 0.99 / 0.96 across the three warm runs). The even-split
  scheduling win is real but no longer dominates: at these mid-M sizes the gap to
  `linalg` is back inside the ±5–10% run-to-run swing, so this band should be
  read as a tie, not a win. The headline shapes are unaffected — decode (M=1)
  still wins ~2.4–2.6× and prefill (M=96) wins on both orientations (1.07× /
  1.02×).

Net warm-state tally: dispatch clearly beats `linalg` on the full decode/small-M
column and the entire up-proj band through M=128 (13/22 shapes by strict median,
~16/22 counting the down-proj mid-M ties), and trails only on the heaviest GEMMs
(down-proj M≥128, up-proj M≥256) — the same residual large-M gap documented below.

## Future work: how to close the remaining large-M gap

Current losses (all the *largest*, non-headline shapes): down-proj M=128/256/512
≈ 0.94/0.89/0.91 and up-proj M=256/512 ≈ 0.93/0.91 vs `linalg`. Decode (M=1) and
prefill (M=96) already win, so this is strictly the high-batch tail.

### Update (Jun 2026): wide-N M>256 retuned to 6×32 tile + KC=1024

The wide-N (up-proj) large-M branch used a 4×48 register tile for all `M > 192`.
An interleaved 30-run A/B on the Skylake AVX-512 VM shows the 6×32 tile (the
same config the tall-K branch already uses) beats 4×48 by 3–5% from M=288 up
(M=288/320/384/512 ratios 4×48→6×32: 1.045/1.035/1.047/1.030), while 4×48 still
wins at M=256 (0.98). The 4×48 tile's lower MR caps A-broadcast reuse per packed-B
load; the 6-row tile amortizes each B-load over more accumulators once M is large
enough to fill the worker bands. Dispatch now uses 4×48 only for `192 < M ≤ 256`
and 6×32 for `M > 256`.

On top of that, the 6×32 (M>256) branch moved from **KC=512 to KC=1024** for a
further ~1.5–2% at M=512 (1024/512 ratio 1.020/1.016 across two 40-run passes).
This is a *C-traffic* win specific to the wide-N orientation: the micro-kernel
loads+stores its C tile once per k-panel, and up-proj's C is huge (N=11008 →
45 MB at M=512). With K=2048, KC=1024 splits K into **2 even k-panels** instead of
4 (KC=512), halving the C re-traffic; the even split also makes it more stable
than KC=768 (3 uneven panels). KC=2048 (1 panel) over-grows the packed A/B panels
and loses (0.92–0.95). The *down-proj* orientation was swept the same way and
keeps KC=512 — its C is tiny (N=2048 → 8 MB) and K is long (11008), so larger KC
buys no C savings and only spills the B/A panels (KC 768/1024/1536 all ≤1.01,
mostly losses). These two changes narrow but do not close the up-proj M≥288 gap
vs `linalg` — the residual is micro-kernel maturity (see below).

The remaining large-M losses (down-proj M=128/256/512, up-proj M=256) are genuine
micro-kernel maturity vs `linalg`'s hand-tuned AVX-512 kernel, **not** memory
layout — every cheap structural lever has now been measured and rejected (below).

**Dead ends — already measured, do not re-attempt without new evidence:**
- **2-D `MC × KC` cache blocking — implemented and measured *slower* here.** The
  losses scale with M, which *looks* like an A-panel outgrowing L2 (M=512/KC=512
  packs ≈2 MB of A per worker, past the 1 MB L2). A full BLIS-style `ic[MC]` loop
  was built (`_prefill_gemm_mc`: pack the worker's whole B-band once per k-panel,
  then stream MC-row A-blocks against it so packed-A stays L2-resident) and A/B'd
  against v3 on the large-M shapes. It was a **wash-to-loss** at every valid `MC`
  (multiple of MR): e.g. down-proj M=256 mc192=165 vs v3=181 GFLOPS, M=512
  mc192=183 vs v3=190. Reason: on this 33 MB-L3 part A already fits in L3, so the
  "redundant" A re-reads v3 does are cheap L3 traffic, not DRAM; the MC variant's
  larger per-worker B-buffer (whole band vs one tile) plus per-`ic`-block B
  re-reads from L3 cost more than the A-locality it buys. This confirms the
  earlier `SHARED_A` finding and closes the "highest expected payoff" item — MC
  blocking only pays once A exceeds L3, which needs a much smaller-cache machine.
- In-kernel packed-B prefetch (linalg does it): A/B neutral-to-harmful here
  (0.98–1.01); ILP + HW prefetcher already hide the latency.
- Masked-SIMD / zero-edge tiles: a zero-edge `4×32` lost at every M, so the
  scalar M-remainder is *not* the bottleneck.
- `SHARED_A` (pack A once, share across N-workers): a wash — A fits in L3.
- down-proj M>64 tile swap (6×32 ⇄ 8×24-KC512-TN48): non-monotonic across M
  (8×24 won M=128/256 but lost M=192/384 in interleaved A/B) — i.e. inside the
  noise floor, not a real signal. Kept 6×32.

**Still open (needs a different machine or a real micro-kernel rewrite):**
- **M-parallelism for very large M.** Workers split only the N (j-tile) dimension.
  At large M and modest N (down-proj N=2048 → 32 j-tiles / 4 workers) a 2-D
  `(M, N)` split could improve balance. Untested.
- **Micro-kernel parity with `linalg`** on the heaviest GEMMs — the residual
  ~9–11% at M≥256 lives here, not in cache blocking.

**How to validate any of the above:** interleaved A/B of the two kernel variants,
peak GFLOPS over ~15–20 runs (see the methodology note above). The throwaway
`exp_*.mojo` harnesses used for the TILE_N and KC sweeps are the template;
gate the new path behind a comptime flag so on/off can be measured in one binary.

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
