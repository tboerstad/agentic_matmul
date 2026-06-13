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
(down-proj M≥128, up-proj M≥256) — the same residual large-M gap addressed next.

### Large-M retune (Jun 13 2026): unify on the 8×24 tile, drop the KC=1024 pick

Questioning the "residual is unfixable micro-kernel maturity" conclusion above:
an interleaved A/B (peak over 20–25 runs, the blessed methodology) of *several*
tile shapes — not just the 6×32 ⇄ 8×24 pair the earlier note tried — found the
large-M tile selection was simply mistuned, and that the tuning had drifted
across compiler bumps. Two concrete fixes, both validated head-to-head:

- **Up-proj (wide-N) M>192 → 8×24, KC=512** (was 4×48 KC=512 at M=256 and 6×32
  KC=1024 at M>256). The 8×24 tile wins ~+4% at M=256 (0.92 → ~0.96) and ~+3%
  at M=512 (0.91 → ~0.93). MR=8 also divides M=256/512 with no scalar
  remainder. The old KC=1024 "halve the wide-C re-traffic" pick has since
  *regressed* — at current codegen KC=512 beats KC=1024 even for the 6×32 tile —
  so the special case was removed and both orientations now share KC=512.
- **Down-proj (tall-K) M≥128 → 8×24, KC=512** (was 6×32 KC=512). MR=8 divides
  M=128/256/512 evenly; the old MR=6 left a 4-row scalar-`vectorize` tail at
  M=256, which was the single worst-tuned shape (0.85 in isolation). 8×24 wins
  ~+4% at M=256 in controlled A/B, neutral at M=128/512. The `64 < M < 128`
  band keeps 6×32 KC=512 — it wins the M≈96 down-proj prefill batch (~1.01×)
  where 8×24 loses ~5% (both MR divide 96 evenly, but NR=32's wider per-row
  SIMD pays off before the i-panel count grows). So down-proj now crosses tiles
  twice across M; the dispatch comment documents all three bands.

This collapses the old per-M tile zoo (8×24 / 4×48 / 6×32 / KC=1024) down to
"8×24 everywhere, with a single 6×32 band for the down-proj M≈96 batch", and
narrows the up-proj large-M gap to near-parity (M=256 ~0.96, M=512 ~0.93).

**Dead ends re-confirmed on this hardware during the retune (do not re-attempt
without a smaller-L3 machine):**
- *Larger KC to cut down-proj C re-traffic* — K=11008=256×43, so KC∈{688,1376,
  2752} give clean 16/8/4 k-panels (vs 22 uneven at KC=512). All lost: KC=2752
  ran 0.70–0.76. The bigger packed A/B panels spilling cache cost far more than
  the C-traffic saved. KC=512 stays.
- *High-MR / narrow-NR tiles* (10×16, 12×16 — higher FLOP/byte of L1 traffic):
  measured *worse* (~0.78–0.82), so down-proj large-M is not L1-load-bound.

Residual after the retune: up-proj M=256/512 are near-parity (~0.93–0.96) and
down-proj M≥128 still trails ~0.88–0.95. That tail is addressed by the
masked-remainder kernel below; the 2-D (M,N) worker split was built and refuted
(see Dead ends).

### Masked register-blocked M-remainder (Jun 13 2026): 9 losses → 4

The largest unlock came from fixing how the prefill kernel handles the
**M-remainder** — the `m % MR` rows left over when the register tile's `MR`
doesn't divide `M`. The old path swept these row-by-row: each remainder row ran
its own full K-loop with only `NR_VECS` (=4) independent accumulators, far too
shallow to hide the 4-cycle FMA latency. That scalar-ish tax is what forced the
dispatch onto `MR`-divides-`M` tiles even when they tiled `N` badly.

`_prefill_gemm_v3` now handles all `r = m % MR` leftover rows as **one
register-blocked masked block**: a single K-sweep with `r × NR_VECS`
accumulators, reusing the already-packed B panel at full `NR` width, with the
inactive rows masked out of the C load/store (`comptime for mr … if mr < r`).
Verified bit-identical (`verify_dispatch` max_err 0.0 on remainder shapes
M=300/257/130/70/13/7).

With the tail no longer slow, the tile choice flips to whatever balances `N`
best, and `N=2048` balances perfectly only with **NR=32** (TILE_N=64 → 32 even
j-tiles, zero N-remainder). So **down-proj is now a uniform 6×32 tile** at every
M (KC=256 for M≤64, 512 above), and up-proj M>256 also moves to 6×32. Warm
interleaved sweep vs `linalg` (peak/20):

| M | up-proj | down-proj |
|---|---|---|
| 1–64   | WIN (1.05–2.4×) | **WIN (1.05–2.0×)** — mid-M was the noise floor |
| 96, 128| WIN (1.04–1.09×) | **WIN** (1.02–1.03×; M=128 ≈ 1.00) |
| 256    | 0.93 LOSE | 0.94–0.95 LOSE |
| 512    | 0.91–0.94 LOSE | 0.93–0.97 LOSE |

Losing shapes dropped from **9 to 4**. The entire down-proj mid-M band
(M=8..64), previously stuck on the noise floor at ~0.96–0.99, now wins
decisively (e.g. M=8 → 1.28–1.30×, M=16 → 1.19–1.25×), and down-proj M=256
improved 0.88 → 0.94. The only remaining losses are the **four heaviest GEMMs**
(up/down-proj M=256/512), all at ~0.93–0.97.

Those four are genuine micro-kernel maturity: at M≥256 both kernels are
compute-bound at ~66–70% of the 358 GFLOPS f64 peak, and the cheap knobs —
tile (MR×NR), KC, KU, NC_TILES — have all been swept and sit within the ±5–10%
run-to-run noise of each other. Closing the last ~3–7% needs an assembly-level
inner-loop rewrite to match `linalg`'s hand-tuned AVX-512 kernel.

### Large-M retune on the 2.10 GHz Xeon (Jun 13 2026): 4 losses → 0

The "genuine micro-kernel maturity, needs an assembly rewrite" conclusion above
turned out to be **hardware-specific to the Skylake 2.80 GHz VM it was tuned
on**. Re-running the sweep on the cloud 2.10 GHz Xeon (4 cores, AVX-512, **L2
2 MB/core, L3 260 MB** — twice the L2 of the Skylake box) found a different set
of losing shapes and a different optimum. Baseline (warm, peak/8, interleaved):
up-proj M=256 ≈ 0.91–0.93, up-proj M=512 ≈ 0.95–0.97, down-proj M=512 ≈ 0.95
(down-proj M=256 already *won* ~1.01 here). Two tile-selection fixes, each
validated by interleaved A/B vs `linalg` (peak over 20 runs):

- **up-proj (wide-N) M>192 → 6×32 (was 8×24 at M≤256).** On this machine the
  6×32 tile beats 8×24 by a wide margin at every M>192 (M=256: 8×24-KC512 0.93
  → 6×32-KC512 **1.02**) — N=11008's wide per-row SIMD outweighs 8×24's higher
  i-panel count. This alone flips M=256 from LOSE to WIN.
- **Larger KC where the 2 MB L2 allows it.** With double the L2 the C-traffic
  saved by using fewer k-panels now dominates the bigger packed panels:
  - up-proj M>288 → **KC=2048** (single k-panel, K=2048, C written once):
    M=384 0.999→**1.046**, M=512 0.953→**1.038**. (M≤288 keeps KC=512: best at
    M=256/288.)
  - down-proj M>256 → **KC=2048** (6 k-panels over K=11008 vs 22 at KC=512):
    M=384 0.957→**0.975**, M=512 0.936→**0.99** (parity). Pushing KC further
    over-grows the panels and collapses (KC=5504 → 0.75, full-K → 0.45); M≤256
    keeps KC=512 (M=128 needs it: 1.10 vs 1.06).

This is the **opposite** of the Skylake finding ("KC=512 beats KC=1024 even for
6×32") — a direct consequence of the larger L2, and a reminder that every KC/
tile pick in this file is hardware-specific. Warm sweep after the retune: **21
of 22 shapes WIN**, the lone exception being down-proj M=512 at ~0.99 (parity,
flips WIN/LOSE between runs). The four clear large-M losses are gone with no
micro-kernel rewrite — they were a stale tile/KC table carried over from a
smaller-cache machine.

## Future work: how to close the remaining large-M gap

> Superseded on the 2.10 GHz Xeon by the retune above — the four large-M losses
> this section was written to attack are now WINs/parity there. The analysis
> below still holds on the smaller-cache Skylake 2.80 GHz VM, where the larger-KC
> fix does not apply (its 1 MB L2 cannot hold the bigger packed panels).

Current losses (after the masked-remainder kernel above): only the four heaviest
GEMMs — up-proj M=256/512 ≈ 0.93/0.92 and down-proj M=256/512 ≈ 0.94/0.95 vs
`linalg`. Everything M≤128 on both orientations now wins, so this is strictly
the high-batch tail.

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

- **2-D `(M, N)` worker split — built and measured *slower* (Jun 13 2026).** The
  N-parallel scheme splits only j-tiles across the 4 cores, so each worker packs
  *all* M rows of A; at large M that packed-A panel spills L2 (M=512/KC=512 ≈
  2 MB > 1 MB L2). Added a `WM` comptime param to `_prefill_gemm_v3` forming a
  `WM × (cores/WM)` worker grid — each worker owns a row-band so its packed-A
  shrinks ~WM-fold and stays L2-resident. Verified bit-identical to `WM=1`
  (checksum_err 0.0), then A/B'd vs `WM=1` and linalg (peak/20) on down-proj
  large M. **It lost at every M:** down-proj M=256 ratio WM=1 0.95 → WM=2 0.91 →
  WM=4 0.73; M=512 0.94 → 0.93 → 0.76. Same root cause as the MC-blocking loss:
  A fits in L3 so the "redundant" per-worker A re-reads are cheap, and splitting
  M just adds WM-fold redundant **B**-packing (B is the big matrix for down-proj)
  plus B re-reads from L3. The `WM` param was reverted to keep the hot kernel
  clean. Confirms packed-A L2 residency is **not** the down-proj bottleneck.

**Still open (needs a real micro-kernel rewrite, not a memory-layout change):**
- **Micro-kernel parity with `linalg`** on the heaviest GEMMs — the residual
  now lives entirely here. Both kernels are far from the 358 GFLOPS f64 peak
  (linalg ~58%, dispatch ~50% at down-proj M=256), so there is headroom, but the
  N=2048 down-proj shape is a poor fit for register-blocked tiling: it balances
  cleanly across 4 cores only with NR=32 (TILE_N=64 → 32 even tiles), yet the
  register-efficient MR for NR=32 (MR=6) leaves an M-remainder, while the
  remainder-free MR=8 forces NR=24 (TILE_N=48 → 43 tiles, 11/11/11/10 imbalance).
  Closing it needs `linalg`-style **masked AVX-512 N-remainder handling** (so a
  perfectly-balanced NR=32 tiling has no scalar tail) and/or **pack/compute
  overlap** to hide the ~5–8% redundant A/B-packing cost. Both are substantial
  and unproven on this hardware.

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
