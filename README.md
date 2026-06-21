# matmul

An experiment in writing optimized matmul kernels in Mojo using only [Claude Code](https://claude.com/claude-code) on mobile (iOS). The goal is fast matmul: maximizing GFLOPS across shapes and element types, staying competitive with the Mojo stdlib `linalg.matmul` and OpenBLAS. The original target, and still the headline benchmark, is the two float64 matrix shapes from the Qwen 2.5 VL 3B MLP projections:

- **Decode:** 1 × 11008 × 2048 (memory-bandwidth bound)
- **Prefill:** 96 × 11008 × 2048 (compute-bound)

Beyond those, the dispatch kernel and the `bench_focus.mojo` harness now run a
wider shape set and other element types (`--dtype f32/f16/bf16`); see
COMPARISON.md for the float64 and float32 results vs `linalg` and OpenBLAS.
The kernel tiles and cache-blocking constants are tuned for float64, so float32
currently trails `linalg` by a few percent on the compute-bound square band
(an open tuning item, tracked in COMPARISON.md).

## Results

Peak GFLOPS by hardware (higher is better):

### Prefill (96 × 11008 × 2048)

| Kernel | Xeon Skylake 2.80 GHz (4c) | Xeon Emerald Rapids 2.10 GHz (4c) | Apple M4 Max (14c) |
|---|---|---|---|
| SciPy dgemm (Accelerate/SME) | 144.6 | 200.8 | 709 |
| NumPy (Accelerate/SME) | 216.9 | 235.6 | 640 |
| **Mojo (agentic matmul)** | 208.4 | **256.6** | **721** |
| Mojo linalg (stdlib) | 182.4 | **247.5** | 382 |

### Decode (1 × 11008 × 2048)

| Kernel | Xeon Skylake 2.80 GHz (4c) | Xeon Emerald Rapids 2.10 GHz (4c) | Apple M4 Max (14c) |
|---|---|---|---|
| SciPy dgemm (Accelerate) | **5.5** | 8.4 | 52 |
| NumPy (Accelerate) | 13.4 | 25.0 | 57 |
| **Mojo (agentic matmul)** | 13.9 | **28.5** | **57** |
| Mojo linalg (stdlib) | 5.9 | 11.4 | 22 |

On the Apple M4 Max the NumPy/SciPy figures are **Apple Accelerate**, which on
this NumPy (2.5, macOS) drives the **SME matrix coprocessor** (single-thread ==
multi-thread is the tell). SME exceeds the ~515 GFLOPS f64 NEON ceiling of the 10
P-cores, so the earlier NEON-only kernel (prefill 435, decode 37) could not reach
it. The agentic matmul now drives SME itself (see "Apple Silicon" below): a
hand-written f64 FMOPA outer-product GEMM (`sme_kernel.mojo`) takes prefill from
435 to **721** (matches/edges Accelerate, the M4 Max has two SME units so two
threads reach ~1035 GFLOPS aggregate), and a bandwidth-tuned row-split GEMV takes
decode from 37 to **57** (matches Accelerate, both bound by ~228 GB/s of memory
bandwidth). The two Xeon columns predate this toolchain and are NEON/AVX-512 only.

## Kernels

`gemm.mojo` keeps only the state-of-the-art kernels — the earlier evolution
steps (naive, tiled, simd, parallel, register-blocked, packed, comptime, goto,
and the v2 prefill kernel) have been removed. `matmul_dispatch` routes each
shape to the fastest of these four:

Every kernel is built on two shared, zero-cost abstractions:

- **`RegisterTile`** (`gemm.mojo`) — an MR×(NR_VECS·NELTS) block of C held in
  SIMD registers and swept over K with `rank1_update`. The compiler flattens its
  `InlineArray` accumulator into registers, so the FMA/load/store it emits is
  byte-for-byte identical to a hand-numbered register nest.
- **`Tile`** (`tile.mojo`) — a rows×cols window into a row-major buffer (rows
  `stride` apart), generic over origin so one type names both read-only operands
  and the writable C target. Its accessors (`sub`, `tile`, `row`, `addr`) are all
  `@always_inline` over the same offset arithmetic the kernels would write by
  hand, so it is a zero-cost rename of `ptr + i*stride + j` — the kernels speak in
  tiles instead of raw pointer math. Each kernel gets its operands from
  `Matrix.noalias_view()` once at the top and threads those views (not raw
  pointers) through its workers; `Matrix.view()` is the plain (non-noalias)
  variant for callers off the hot path.

The four kernels:

- **`_packed_gemm`** — the workhorse packed GEMM: per-worker A/B-panel
  packing, the `RegisterTile` micro-kernel with hoisted B-loads + noalias,
  a register-tiled masked tail (`_masked_microkernel`) for the M- and
  N-remainders, and an optional shared single pack of A.
- **`_decode_gemv`** — j-parallel GEMV with L1-resident column chunks and
  software prefetch, for decode shapes (M = 1). Streams B exactly once.
- **`_nopack_gemm`** — M-parallel, no-packing register-tiled kernel for the
  two regimes the N-parallel prefill kernel handles badly: thin-N (small N,
  large M·K) and small M-dominant boxes whose B stays L2-resident.
- **`_serial_gemm`** — serial register-tiled kernel for tiny shapes, where
  any thread launch / packing overhead dwarfs the compute.

On Apple Silicon a fifth kernel, **`sme_gemm_ptr`** (`sme_kernel.mojo`), drives the
SME matrix coprocessor for the compute-bound band (see "Apple Silicon" below). It
is the only path that exceeds the NEON ceiling; the four NEON kernels above remain
the fallback for tiny / narrow / non-Apple shapes and the whole x86 build.

## Dispatch logic

`matmul_dispatch` routes each shape to the kernel and tile that measured best for
it (see the comment in `matmul_dispatch` for the authoritative table):

- `M·N·K < 2^19` (tiny): serial register-tiled `_serial_gemm` — no thread
  launch, allocation, or packing. Avoids the fixed parallel-kernel overhead that
  made tiny GEMMs 7–30× slower than `linalg` (sq8 0.03× → 2.37× after).
- `M == 1`: decode GEMV (streams B once).
- `2 ≤ M ≤ 5`: the packed micro-kernel with `MR = M` — packs B once and reuses it
  across all rows (vs the GEMV re-streaming all of B per row, ~2× slower at M=4).
- **Small box, M-dominant, B fits L2** (`M ≥ 64`, `M ≥ N`, B = `K·N·8` fits L2):
  M-parallel, no-packing `_nopack_gemm` — each core owns a band of C's rows and
  sweeps the full N, reading A/B straight from source (B stays L2-resident, so
  packing buys nothing). The packed prefill kernel's packing + thread-launch
  overhead dwarfs the tiny compute on these cache-resident shapes; this branch
  pays none of it. Fixes the worst shapes in the general sweep — sq128 0.65→1.0,
  sq96 0.71→1.16, 512×128×512 0.56→0.84. The tile **MR is picked per shape so it
  divides M with no leftover-row tail**: MR=6 (24 accumulators, deepest ILP) when
  `M % 6 == 0`, else MR=4 (16 accumulators, still deep enough to hide FMA
  latency, and divides every multiple-of-4 box). An M not divisible by MR ran its
  tail one row at a time (a 1×NR tile) far below the MR-row throughput, which made
  sq256 (M%6=4) the single worst shape in the robust peak sweep (~0.84); MR=4
  brings it to ~0.95, 512×128×512 0.89→0.97, 256×128×512 0.97→1.08, while the
  M%6==0 boxes (sq96, sq192) keep MR=6. See DESIGN.md. The B-fits-L2 test is
  **L2-adaptive**:
  a compile-time 512 KB tier (kept cpuid-free for the few-µs tiny boxes) plus a
  `B ≤ L2/3` tier (`_box_l2_budget`, only reached once B > 512 KB, where the op
  is large enough that the one-time memoized cpuid is free). The no-pack route
  re-reads all of B per MR-row block, so B must stay L2-resident *alongside* the
  packed-A micro-panel + C + prefetch headroom across the whole M-sweep — which
  holds only to ~1/3 of L2. On the 2 MB-L2 Xeon that admits up to sq288
  (B≈648 KB) and **excludes** sq320+ (B≥800 KB): re-measured interleaved A/B vs
  the packed square-ish path, no-pack wins only to sq288 (~0.92–1.00) and loses
  badly above it (sq320 ~0.67, sq352 ~0.56, sq384 ~0.42, 640×256×512 ~0.47),
  where the packed path runs 0.78–0.95. (An earlier, more generous `(2·L2)/3`
  ≈ 1.35 MB cut sent sq320–sq416 + tall boxes to no-pack — they were then the
  worst losses in the whole sweep; the old no-pack "wins" there were measured on
  an older Mojo nightly whose `linalg.matmul` was slower and have since flipped.)
  On a 1 MB-L2 part L2/3 = 341 KB sits below the 512 KB tier-1, so it keeps
  no-pack only for B ≤ 512 KB. The `M ≥ N` gate keeps it unreachable for every
  wide/headline shape (Qwen up/down proj are `N ≫ M`).
- `M ≥ 6`: parallel `_packed_gemm` (N-parallel: each worker owns a band of
  j-tiles, reading the dominant B matrix from DRAM once into private L2). Tile and
  KC are selected per regime — narrow NR=16 for `N ≤ 192` (so a small N still
  splits into ≥ num_workers j-tiles), 8×24 only for very-wide-N small-M (Qwen
  up-proj small batch), 6×32 elsewhere; KC scaled up where the L2 allows it.
  For `M ≥ 192` this branch also packs A **once** (`SHARED_A`, see Dead ends)
  instead of once per N-worker — a +3–10% win across the whole large-M band.

Tunable parameters (tile MR×NR, KC, KU) are hardware-specific — see notes below.

### Apple Silicon (M-series) adaptations

The dispatch above was tuned on 4-core AVX-512 Xeons. Apple Silicon is a
big.LITTLE part (P-cores + much slower E-cores) with a large cluster-shared L2,
so the picks below are overridden behind `comptime CompilationTarget.is_apple_silicon()`
(x86 compiles to the byte-for-byte original — Intel is never touched). Full
measured rationale in `DESIGN.md`; in short:

- **SME (matrix coprocessor) for the compute-bound band.** The NEON kernels cap at
  the ~515 GFLOPS f64 NEON peak of the 10 P-cores (measured: 51 GFLOPS/core), but
  Accelerate hits ~700–790 by driving the SME unit (`FEAT_SME_F64F64`), whose f64
  `FMOPA` does an 8×8 outer-product accumulate into a ZA tile. `sme_kernel.mojo`
  drives it from Mojo via inline assembly: a 16×32 micro-kernel (the eight ZA.D
  tiles), GotoBLAS (pc, ic-block, jt, it) blocking, A packed column-major, B read
  in place, and an in-bounds overlap-tile remainder path for any M/N. The M4 Max
  has **two** SME units (one per P-cluster), so two worker threads reach ~1035
  GFLOPS f64 aggregate — above Accelerate. Dispatch routes f64, `m ≥ 64`, `n ≥ 32`,
  `M·N·K ≥ 2^21` shapes here: prefill 435→721 (1.02× Accelerate), the up/down-proj
  and large squares 0.97–1.50× Accelerate, odd-N (11007) 514→822. The crux was
  clobbering the whole SVE register file (`smstart` zeroes z0–z31/p0–p15) so the
  compiler does not lose caller FP state. Tiny (128³) and narrow-N (512×128×512)
  shapes stay below Accelerate's mature small-matrix path and are the open residual.
- **Decode GEMV split by K-rows, not N-columns.** The column-split GEMV read a
  strided slice of row-major B (~132 GB/s of the ~242 the P-cores can read
  sequentially); splitting by K lets each worker stream contiguous B rows into a
  private partial-C, then a parallel reduce. Decode 37→57 GFLOPS (matches
  Accelerate, both bandwidth-bound at ~228 GB/s).
- **Heavy NEON kernels parallelize over P-cores only** (`compute_core_count()`, from
  `hw.perflevel0.physicalcpu`). A static even split hands the E-cores an equal
  share of the compute-bound micro-kernel and they straggle. +24–31% across the
  heavy band; the M=96 prefill headline flips from losing (0.88) to winning
  (1.20) vs the current stdlib `linalg`. The no-pack box kernel stays on all
  cores (its boxes are too small for an E-core to straggle).
- **No-pack box budget capped at 896 KB.** `l2_cache_size()` reports the 16 MB
  *cluster-shared* L2, so the Intel `l2/3` rule over-admits mid boxes to the
  no-pack route; capping it routes B > ~900 KB to the packed P-core path
  (sq512 0.83→0.99).
- **Tiny serial cutoff lowered to 2^18 for `M ≥ 64`.** On a 14-core part the
  parallel launch is cheap enough that box-eligible shapes want threads below
  the 2^19 Xeon cutoff (sq64/sq80 ~0.3–0.55 → 0.7–0.75); small-M shapes keep
  2^19 so they stay serial.
- **Decode GEMV on P-cores** too (bandwidth-bound, but the E-cores contend
  rather than add bandwidth).
- **SHARED_A (pack A once) down to the small-M band.** The M≥192 crossover was
  set for 4 Xeon cores; the per-worker re-pack is `(workers − 1)×` redundant, so
  on 10 P-cores it pays below the headline band. Enabled for the wide-N (M≤192)
  and tall-K small-M branches: both Qwen M=96 headlines improve (up-proj
  1.11→1.20, down-proj 1.20→1.27).

The 6×(4·NELTS) register tile needs no change: `NELTS` auto-scales (2 for f64
NEON, 8 for AVX-512) and the 24-accumulator / KU=2 tile fills the 32-register
NEON file exactly as it fills the 32-register AVX-512 file, so it is
register-optimal on both. Every other knob (KC, the square/prefill TILE_N, the
prefetch distance, sub-P decode workers) was tested and left at the x86 pick
because it sat within the measurement noise on Apple.

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
  single-run count). Two recent levers: (1) the square-ish branch is now a
  **pack-B-only / TileK=K** GEMM matching stdlib linalg (found by reading linalg's
  open source + emit-asm: on a square A is as large as B, so linalg packs only B,
  reads A unpacked, and sweeps the whole K so C is stored once). We do the same
  (PACK_A=False, KC at the rung >= min(K,2048), TileN shrunk to a 512 KB pack
  budget): interleaved A/B vs the old pack-both square path is sq512 +19%, sq768
  +14%, sq1024 +12%, sq1536 +12%, sq2048 +6.5% (all 2σ WIN), bit-identical to
  naive. Small-N square-ish and sq384 (too few j-tiles for the N-parallel
  pack-B-only path) keep the old pack-both fallback, which picks KC by detected L2
  (KC=1024 from a 1 MB/core L2 up, KC=512 below) plus a load-balance-aware TILE_N
  (wide 8·NELTS only at >= 4
  wide j-tiles per worker, else the finer 4·NELTS — the fine tile's extra
  j-tiles smooth the balance on the small squares: sq512, with 2 wide tiles per
  worker, runs +2–4% on the fine tile, while sq1024/sq2048 keep the wide tile;
  see DESIGN.md). The same pack-B-only path also takes the **big boxes** (B ~ 512 KB,
  the top of the small-box window) off the no-pack route — sq256, 512×128×512 and
  256×128×512 lift ~+10–14% (bonly/no-pack) and move from LOSE to parity, while
  the genuinely small boxes (sq96/128/192) keep no-pack (B > 384 KB cut); (2) the
  small-box M-parallel route
  flipped the two worst shapes in the whole sweep — square sq96/sq128, which
  interleaved-A/B (peak/40) lifts 0.65–0.71 → 1.0–1.16 — plus the tall
  cache-resident boxes (512×128×512 0.56→0.84, 256×128×512 0.63→0.90); a later
  MR-divides-M tile pick on that route (MR=6 only when M%6==0, else MR=4) lifted
  the no-tail-divisible boxes another step — sq256, the single worst shape in the
  robust peak sweep at ~0.84, to ~0.95, and 256×128×512 to ~1.08 (see the
  small-box bullet above and DESIGN.md). Residual losses are the mid/large packed
  squares (sq512 ~0.96, sq768/1024 ~0.93–0.97 at robust peak) and low-arithmetic-
  intensity corners (K=128, odd N/K) where `linalg`'s masked AVX-512 remainder
  handling wins. The big-square gap is algorithmic (pack/compute overlap or
  M-blocking), not a tile pick.

### Key findings from tuning

The micro-kernel itself is not the bottleneck on most shapes: `_packed_gemm`
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
- **Masked N-remainder.** The mirror image: the last j-tile of an
  N-not-a-multiple-of-NR shape ends in a partial NR-panel. It used to be computed
  by a row-by-row `vectorize` tail that swept K once per row with only NR_VECS-deep
  ILP — far too shallow to hide FMA latency, so the partial tile ran at a fraction
  of the microkernel's throughput and dragged down the straggler worker. Because
  the packed B panel is already zero-padded to full NR, the fix runs the *same*
  register-blocked MR×NR_VECS microkernel on it (the zero columns contribute
  nothing) and masks only the C store to the valid columns — full-NELTS SIMD for
  each complete lane, a scalar tail for the straddling lane. The cost had scaled
  with the remainder width (worst at a 31-wide remainder): on the 2.10 GHz Xeon
  `512×11007×2048` ran **0.85→1.01** vs `linalg`, with the multiple-of-NR
  `512×11008×2048` already at parity. Every N now holds ~parity regardless of its
  remainder. Bit-identical (`verify_dispatch` max_err 0.0 across remainder widths
  7/8/24/31 on both the wide-N and square-ish branches).
- **K-unroll must not exceed the register file.** The 6×32 micro-kernel holds
  `MR·NR_VECS = 24` SIMD accumulators, and the comptime k-unroll `KU` keeps
  `KU·NR_VECS` B-vectors live per unrolled step. `KU=2` needs `24 + 8 = 32` zmm
  registers — exactly the AVX-512 file — while `KU=4` needs `24 + 16 = 40` and
  spills. The `KU` had drifted to 4 on the shared `_prefill` 6×32 path; restoring
  `KU=2` is bit-identical (codegen-only; `verify_dispatch` max_err 0.0) and
  measured — head-to-head KU2-vs-KU4, peak/30 ×3 on the 2.80 GHz Skylake VM — a
  **uniform win across the heavy-GEMM band**: up-proj/down-proj M=256 **+3 %**
  (flips 0.96–0.99 LOSE→WIN), M=512 **+3–6 %**, h4k/ffn-up8k M=512 +2–5 %, the
  big squares sq1536/sq2048 +2–4 %; and neutral-to-slight-win (never a
  regression) on small-M (M≤64), small-N, and the headline prefill M=96
  (+0.3–1.7 %). The `NR_VECS=2` narrow-N tile (only 12 accumulators) doesn't
  spill at `KU=4`, so it keeps it.
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
- **`SHARED_A`** (pack A once, share across N-workers) — by default every
  N-worker re-packs the full A (M·KC per k-panel), `num_workers` redundant
  copies. This was long held to be **a wash on the wide/tall headline shapes**
  (N ≫ M) — true, but only because the headline is M=96, where A is tiny next
  to the N-sweep so the per-worker re-pack is cheap L3 traffic. The reasoning
  does **not** extend up the M axis: once M grows, A is no longer small and the
  `num_workers`× re-pack is a real cost. Interleaved A/B (peak/30) on the
  2.10 GHz Xeon found a clean crossover at **M ≈ 192** on *every* orientation:
  M ≤ 128 a wash (up-proj M=96 +0.6%, down-proj M=96 −1% — so the headlines
  stay on the per-worker path), M ≥ 192 a clean **+3–10%** that flips the whole
  large-M band LOSE→WIN — up-proj M=512 0.96→0.99 (`ShA/cur` 1.03), h4k-m512
  (512×4096×4096) 0.99→1.04, N4000 0.91→1.01, ffn-up8k 0.98→1.03, down-proj
  M=512 0.93→0.97. `SHARED_A=True` is therefore now enabled from the **square-ish,
  wide-N, AND tall-K branches for M ≥ 192** (previously square-ish only); the
  small-M bands and both Qwen headline shapes keep the byte-for-byte per-worker
  path. Bit-identical (`verify_dispatch` max_err 0.0 across both orientations
  and the gate boundary). On the big squares it had already lifted sq512
  ~0.85→0.90 and sq2048 ~0.86→0.92.

### Still open

Micro-kernel parity with `linalg` on the heaviest GEMMs (square M ≥ 256, now
~0.90–0.94 after the square-ish shared-A pack, where both kernels sit at ~55–66%
of the 358 GFLOPS f64 peak). The large-M wide-N/tall-K band has now **flipped to
a clean WIN** after the `KU=2` register-pressure fix (see *K-unroll* above): on
the 2.80 GHz Skylake VM the full sweep shows the Qwen up-proj winning all 11 M
values (M=256 0.99→1.01, M=512 0.99→1.07), down-proj M=256 0.96→1.00, and
h4k/ffn-up8k M=512 flipping LOSE→WIN — so the residual is mostly the big squares. The odd-N remainder
(N=11007 was ~0.85–0.88) is now **fixed** by the masked-N partial-panel
microkernel (see *Masked N-remainder* above): the partial tile runs at full
microkernel throughput and the shape holds parity (1.01). Closing the square gap
likely needs pack/compute overlap — substantial and unproven on this hardware.
The thin-N and small-box cache-resident gaps are
handled by the M-parallel no-pack route; its L2-fit cut is L2-adaptive
(compile-time 512 KB tier + a `B ≤ L2/3` tier). The no-pack route only wins
while B stays well inside L2 (≈ up to sq288 on the 2 MB Xeon); past that B
spills mid-M-sweep and the packed square-ish path wins, so the cut excludes
sq320+ and the tall boxes. (An earlier `(2·L2)/3` ≈ 1.35 MB cut admitted that
whole band to no-pack — once `linalg.matmul` improved in a newer nightly those
shapes flipped to the *worst* losses in the sweep, sq384 0.42; re-routing them
back to packed lifts them to 0.78–0.95.)

> **Methodology note:** ratios from a single `bench_sweep` run swing ±5–10% at
> M ≥ 128 from turbo/thermal state on shared VMs, and we kept misreading those
> swings as real wins or losses (up-m512 has shown 0.79 in one launch and 1.04 in
> another with byte-identical code). Judge a kernel change with `bench_focus`,
> which is built for exactly this: it runs the shape set for N independent epochs
> (default 10), each a peak over ~12 interleaved A/B reps, and reports the ratio's
> mean ± stdev with a 2σ verdict (WIN / LOSE / tie). A shape is only a real win or
> loss when its 2σ band clears 1.0; a `tie` (band straddles 1.0) is within
> run-to-run noise no matter what a single run printed. Never judge off one ratio,
> and never compare absolute GFLOPS across separate process launches.

## Setup

```bash
bash setup.sh
```

## Run

```bash
source .venv/bin/activate
mojo bench_linalg.mojo           # Mojo stdlib linalg.matmul baseline
mojo bench_sweep.mojo --iterate  # FAST: dispatch vs linalg on corner/edge shapes (default)
mojo bench_sweep.mojo --full     # SLOW: full per-M sweep over many aspect ratios + general grid
mojo bench_focus.mojo            # JUDGE A CHANGE: mean ± stdev + 2σ verdict over 10 epochs
mojo bench_focus.mojo --quick    # fast single-epoch sanity check (NOT for judging)
python bench_sota.py             # NumPy/SciPy/MKL benchmarks
mojo test_gemm.mojo              # Correctness tests
mojo verify_dispatch.mojo        # dispatch correctness vs naive (all branches + edge cases)
```
