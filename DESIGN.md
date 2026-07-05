# Design notes & benchmark rationale

The "why" behind the tuning constants in `gemm.mojo` — relocated out of the
source so the kernels read as code. Numbers are peak GFLOPS ratios vs Mojo's
stdlib `linalg.matmul`, measured on the two headline machines:

- **Machine A:** Intel Xeon Skylake 2.80 GHz, 1 MB/core L2
- **Machine B:** Intel Xeon 2.10 GHz, 2 MB/core L2

Both 4 cores, AVX-512, KVM, f64. **Your numbers WILL differ — see `AGENTS.md`.**

See `README.md` for the per-branch dispatch table and the full results.

## `_packed_gemm` / `SHARED_A`

By default every worker re-packs the FULL A (all i-panels) per k-panel into its
own buffer: `num_workers` redundant copies. On a wide/tall headline shape A is
small next to the N-sweep, so that redundancy is cheap L3 traffic (measured a
wash; the default keeps it). On a big SQUARE, A is as large as B/C and the 4×
re-pack is a real cost (sq2048: ~134 MB of redundant pack traffic/call).
`SHARED_A` packs A once up front (parallelized over i-panels) into a single
buffer keyed `[i-panel][k][MR]`; it halves the packed-A footprint and removes
the redundant packing. Enabled from the square-ish branch (and large-M
wide/tall), so every byte-for-byte headline path is preserved.

## `KU=2` (not 4) on the 6×32 tile

The 6×32 tile holds `MR*NR_VECS = 24` SIMD accumulators, and the comptime
k-unroll keeps `KU*NR_VECS` B-vectors live per step. `KU=2` needs `24 + 8 = 32`
zmm registers — exactly the AVX-512 file — while `KU=4` needs `24 + 16 = 40` and
spills. `KU=2` measured a uniform +2–6% across the heavy-GEMM band, flipping the
Qwen up-proj M≥256, down-proj M=256 and h4k/ffn-up8k M=512 from LOSE to WIN.

## Masked partial-N panel (the old row-by-row tail)

The partial NR-panel and the M-remainder both used to run a row-by-row
`vectorize` tail, sweeping K once per row with only `NR_VECS`-deep ILP — far too
shallow to hide FMA latency. N=11007 (rem=31) ran 0.86 vs linalg while the
multiple-of-NR N=11008 hit parity. Running the full-width register tile
(`_masked_microkernel`) and storing only the valid columns keeps any N (odd / not
a multiple of NR) at ~parity, and removed the MR-not-dividing-M tax (e.g. 6×32 at
M=256).

## `_square_ish` TILE_N: the wide-tiles-per-worker cut

The square-ish branch picks between a wide `TILE_N` (8·NELTS, 64) and a fine one
(4·NELTS, 32). The wide tile packs B in fewer, fatter j-tiles (less packed-A
re-reading); the fine tile doubles the j-tile count, which smooths load balance
when each worker owns only a couple of tiles. The cut is the count of wide
j-tiles per worker: take the wide tile only at >= 4 each, else the fine tile.

Measured interleaved A/B (the two tiles plus linalg in one loop, peak/40 ×4) on
the 1 MB/core Skylake:

| Shape | wide tiles/worker | TN32 vs TN64 | tile |
|---|---|---|---|
| sq512 | 2 | TN32 +2–4% (robust) | fine |
| sq1024 | 4 | wash (±2%) | wide |
| sq2048 | 8 | TN64 +1–2% (robust) | wide |

So the >= 4 cut lifts only sq512 (the worst square) onto the fine tile and keeps
the bigger squares, where the wide tile's lower packed-A traffic wins, on wide.
An earlier `>= 2` cut sent sq512 to the wide tile and left that ~2–4% on the
table. Bit-identical either way (same kernel, different TILE_N).

## `_nopack_gemm` MR: divide M with no tail

The no-pack small-box kernel splits M into MR-row register-tiled blocks; an M
not divisible by MR leaves a tail block that runs one row at a time (a `1 x NR`
tile), far below the MR-row block's throughput. MR=6 (24 SIMD accumulators, the
deepest ILP that still fits the register file) is best when it divides M
cleanly, but the cache-resident box shapes are mostly multiples of 4 that are
not multiples of 6 (128, 256, 512), so MR=6 leaves a 2-to-4-row tail on each.
MR=4 (16 accumulators, still deep enough to hide FMA latency) divides every
multiple-of-4 box with no tail. The dispatch picks MR=6 when `M % 6 == 0`, else
MR=4.

Measured interleaved peak vs linalg on the 1 MB/core Skylake (the worst small
boxes were the worst shapes in the whole sweep at peak):

| Shape | M%6 | MR=6 | MR=4 | MR |
|---|---|---|---|---|
| sq96 | 0 | 1.13-1.17 | 1.09-1.12 | 6 |
| sq192 | 0 | 1.00 | 0.93-0.94 | 6 |
| sq128 | 2 | 1.00-1.06 | 1.01-1.09 | 4 |
| sq256 | 4 | 0.83-0.89 | 0.89-0.97 | 4 |
| 512x128x512 | 2 | 0.88-0.89 | 0.95-0.97 | 4 |
| 256x128x512 | 4 | 0.95-0.97 | 1.08-1.09 | 4 |
| 512x256x256 | 2 | 0.90-0.91 | 0.91-0.93 | 4 |

The `M % 6` rule picks the faster tile in every measured box: MR=6 keeps the
two M%6==0 squares (more ILP, no tail either way), MR=4 lifts the rest by
removing the slow one-row tail. sq256 was the single worst shape in the robust
peak sweep (~0.84); MR=4 brings it to ~0.95. The big-square gap (packed sq512
and up, ~0.93-0.97) is untouched: that residual is algorithmic (it needs
pack/compute overlap or M-blocking), not a tile pick.

## `_box_l2_budget`: the L2/3 no-pack cut

The no-pack route re-reads ALL of B once per MR-row block, so B must stay
L2-resident across the whole M-sweep — which holds only to ~1/3 of L2
(re-measured interleaved A/B vs linalg, Machine B, peak over 50 runs ×3):

| Shape | B size | no-pack | packed | route |
|---|---|---|---|---|
| sq288 | 648 KB | ~0.92–1.00 | ~0.92 | no-pack (admit) |
| sq320 | 800 KB | ~0.64–0.71 | ~0.82–0.86 | PACKED |
| sq352 | 968 KB | ~0.55–0.58 | ~0.77–0.95 | PACKED |
| sq384 | 1.15 MB | ~0.42 | ~0.89–0.92 | PACKED |
| 640×256×512 | 1 MB | 0.47 | 0.86–0.91 | PACKED |
| 512×320×512 | 1.28 MB | 0.46–0.50 | 0.83–0.87 | PACKED |

A previous `(2*L2)/3` ≈ 1.33 MB cut admitted sq320..sq416 + tall boxes to
no-pack, where they were the worst losses in the whole sweep (sq384 0.42); their
earlier no-pack "wins" were on an older Mojo nightly whose linalg was slower and
have since flipped. On the 1 MB Skylake, L2/3 = 341 KB sits below the
compile-time 512 KB tier-1 cut, so that part simply keeps no-pack for B ≤ 512 KB
(sq288/sq320 there already preferred packed).

## Removed: `_square_ish_kc` (the pack-both fallback's L2-adaptive KC)

The square-ish branch once had a *pack-both* fallback for small-N squares (too
few wide j-tiles to fill the cores), and `_square_ish_kc` picked that path's KC
from the detected L2 (KC=1024 from a 1 MB/core L2 up, KC=512 below — a deeper KC
sweeps more of K per C residency, so each L2-resident C micro-tile is loaded and
stored fewer times). That fallback turned out to be the worst code in the suite
on a 2 MB/core part (sq384 0.82, sq300 0.77): its redundant A-pack was pure
overhead. The fallback is now pack-B-only at TileK = min(K, 2048) (see "Small-N
square" above), so the helper and its L2 probe are gone — KC is just the single
C-stored-once panel, the same the wide rungs use.

## `_l2_resident_kc`: a single C-stored-once panel (KC = min(K, 2048))

The large-M **wide-N / tall-K** band (the `_wide_n` / `_tall_k` `m > 192` SHARED_A
branches) takes its KC from `_l2_resident_kc`. This used to size the packed-B tile
(TILE_N x KC) at *half* the per-core L2 (the BLIS rule) — KC=1024 on a 1 MB/core
L2, KC=2048 on a 2 MB/core L2 — and the `_wide_n` `m <= 288` rung was a hardcoded
KC=512. On the current nightly that left the whole band 2-4% behind linalg:

| Shape | M×N×K | old KC | old | KC=2048 |
|---|---|---|---|---|
| up-m256 | 256×11008×2048 | 512 | 0.964 LOSE | 1.001 tie |
| up-m512 | 512×11008×2048 | 1024 | 0.970 LOSE | 0.997 tie |
| odd-N | 512×11007×2048 | 1024 | 0.964 LOSE | 0.996 tie |
| 512×4096×4096 | | 1024 | 0.986 tie | **1.020 WIN** |

(interleaved A/B vs linalg, 8 epochs, peak-of-12, Machine A 1 MB/core L2.)

The fix is the same "C stored once" insight the square-ish branch already uses:
set **KC = min(K, 2048)** — a single k-panel for the benchmark K's (<= 2048), so
each L2-resident C micro-tile is loaded/stored exactly once instead of once per
k-panel. This is also linalg's `calculate_tile_n_k` TileK. The half-L2 KC=1024
swept K in two panels and paid the extra C reload/store per panel; once linalg got
faster across nightlies, that re-traffic was the whole margin. (The README already
noted KC=2048 wins large-M on the 2 MB part for "fewer k-panels, less C re-traffic"
— the same mechanism now reaches the 1 MB part.)

Implemented by sizing the budget from the per-core L2 and **capping at 2048**
(the single-panel depth): 1 MB/core → KC=2048 (was 1024), 2 MB/core → KC=2048
(unchanged, so Machine B is byte-for-byte identical). The `m <= 288` wide rung
folds into the `m > 192` cache-aware branch instead of its hardcoded KC=512.
Bit-identical (`verify_dispatch` max_err 0.0). The headline prefill (M=96) is in
the `m <= 192` band (KC=256) and is untouched.

The cap is now **byte-based** (16 KB per packed column), so narrower dtypes
get the same cache footprint the band was tuned at in f64: KC=2048 in f64
(identical values to the old element cap on every branch, by construction)
and KC=4096 in f32. `_prefill_kc` gained the matching 4096 rung. See "Wide-N
heavy band" below for the f32 measurements that motivated it. The L2 term is
sized at **half** the per-core L2 (see "Half-L2 packed-B tile" below): every
f64 pick and every 2 MB/core pick lands on the cap either way, and on the
1 MB/core part the whole-L2 f32 tile it used to allow was a measured 15-17%
cliff.

## Wide-N heavy band: finer TILE_N + byte-based KC cap (the f32 tuning item)

The compute-bound f32 wide-N band (up-proj M >= 256, 512x4096x4096, odd-N) was
the open tuning item: a stable 3-7% loss (up-m512 0.970, M512-g 0.964, oddN
0.961, up-m256 0.945, all interleaved A/B on Machine B, 2 MB/core L2). The
tiles and KC caps were element counts tuned in f64, and f32 paid twice:

1. **Load balance.** The wide TILE_N = 8*NELTS doubles to 128 columns in f32,
   so N=11008 splits into 86 j-tiles, 21.5 per core: a ceildiv(86,4)=22
   makespan where the ideal is 21.5, a ~2-3% straggler tax on every call. The
   f64 sweeps never saw it because 8*NELTS=64 splits the same N into 172
   tiles, 43 per core exactly.
2. **Panel depth.** The KC cap of 2048 *elements* is 16 KB deep per packed
   column in f64 and only 8 KB in f32, so an f32 K=4096 swept in two k-panels
   (and paid the C reload/store per panel) where the byte budget f64 was tuned
   at allows a single C-stored-once panel.

The fix is the same treatment the tall-K heavy band already got: the
`_wide_n m > 192` branch drops to the finer **TILE_N = 4*NELTS** (2x more
j-tiles, tighter makespan, packed-B tile at half the L2 instead of all of it),
and `_l2_resident_kc`'s cap becomes **byte-based** (16 KB per column: KC=2048
f64, KC=4096 f32; the tall-K band passed 8 KB until its second pass retired
the extra cap, see "Tall-K large-M, second pass").
Neither lever alone closes M512-g: KC=4096 at the wide tile is a 2 MB packed-B
tile (the whole L2) and measured *worse* (0.936); at the finer tile it is 1 MB
and wins.

Interleaved A/B vs linalg (6 epochs x peak-of-8, f32, Machine B):

| Shape | M×N×K | old (tn8, cap 2048) | new (tn4, byte cap) |
|---|---|---|---|
| up-m512 | 512×11008×2048 | 0.970 LOSE | 0.999 |
| M512-g | 512×4096×4096 | 0.964 LOSE | 1.001 |
| oddN | 512×11007×2048 | 0.961 | 1.004 |
| up-m256 | 256×11008×2048 | 0.945 LOSE | 0.984 |
| dn-m512 | 512×2048×11008 | 0.958 | 0.977 |

The finer tile was also measured in f64 on the same branch (Machine B): a
tie-or-slight-win (up-m512 1.005 vs 0.996, M512-g 0.997 vs 0.991, up-m256
0.982 vs 0.976), and the f64 KC values are unchanged by construction. Machine
A has not re-measured the finer wide-N tile; the change follows the tall-K
precedent (the same tile narrowing lifted that band on Machine A), and on a
1 MB/core L2 it shrinks the packed-B tile from the whole L2 to half of it. Full bench_focus (10 epochs, 2-sigma): every f32 wide-N
shape moves to tie (up-m256 1.001, up-m512 1.002, M512-g 0.978, oddN 0.985)
and the f64 set holds (up-m512 1.003, oddN 0.979, M512-g 0.979, all tie).
Bit-identical (`verify_dispatch` max_err 0.0). The headline prefill (M=96) is
in the `m <= 192` band and untouched.

## Tall-K large-M: cap KC at 1024, finer TILE_N

The `KC = min(K, 2048)` single-panel pick above is right for **wide-N** (K=2048
*does* fit one panel, so C really is stored once). On **tall K** (down-proj,
K=11008) it backfires, and the down-proj large-M band was the worst residual in the
sweep — `dn-m512` swung as low as 0.708 in a single `bench_sweep` run and sat at a
noisy 0.90 ± 0.073 mean in `bench_focus`.

Two things go wrong when KC=2048 meets K=11008:

1. **The "C stored once" benefit is unreachable.** K=11008 needs ⌈11008/2048⌉ = 6
   k-panels no matter what, so C is re-read and re-stored across panels regardless;
   the property that justified KC=2048 on wide-N simply does not hold here.
2. **The M·KC packed-A panel overflows L2.** With M=512, KC=2048 the per-worker
   packed-A panel is 512×2048×8 = 8 MB, far past the 1 MB/core L2. The kernel
   sweeps j-tiles in the inner loop and re-reads that A panel once per j-tile, so an
   8 MB panel re-streams from L3 on every j-tile instead of staying L2-resident.

A smaller KC keeps the A panel closer to L2 and was measured uniformly faster on
tall K, even though it adds k-panels (more C re-traffic) — the A-reuse effect
dominates. The `_tall_k` `m > 256` branch now passes `_l2_resident_kc` an
8 KB per-column cap (KC=1024 in f64, half the default `_wide_n` gets; the cap
argument has since become byte-based, see above), and the whole heavy
band (`m >= 192`) drops to the finer `TILE_N = 4·NELTS` (one NR-panel per j-tile),
which splits N into 2× more j-tiles for a tighter worker makespan and a smaller
packed-B panel. The `192 <= m <= 256` rung keeps KC=512 (it already used a single
L1-resident panel) and only narrows its TILE_N, which retains the K=8192 M≤256 win
while lifting the K=11008 M=256 loss.

Measured interleaved A/B vs linalg, Machine A (1 MB/core L2), down-proj N=2048:

| Shape | M×N×K | old (KC2048, tn8) | new (KC≤1024, tn4) |
|---|---|---|---|
| dn-m256 | 256×2048×11008 | ~0.95 | ~0.97 |
| dn-m384 | 384×2048×11008 | 0.94 | 0.98 |
| dn-m512 | 512×2048×11008 | 0.90 (0.71 tail) | 0.95 |
| ffn-dn8k | 512×2048×8192 | 1.04 | 1.08 |
| dn-k4096 | 512×2048×4096 | 0.97 | 1.01 |

In `bench_focus` (10 epochs, 2σ) `dn-m512` goes from 0.90 ± 0.073 (a 0.71..0.95
spread that read as a noisy "tie") to **0.956 ± 0.011** (a stable 0.94..0.97 band):
the mean lifts ~5.6% and the catastrophic low-end collapse is gone. Bit-identical
(`verify_dispatch` max_err 0.0; KC/TILE_N are codegen-only levers).

A second pass (next section) closed the remaining few percent by dropping the
A-pack from this rung entirely, which also retired the 8 KB KC cap this
section introduced.

## Tall-K large-M, second pass: pack-B-only, KC back to 2048

After the KC-cap fix above, the tall-K `m > 256` band was the last stable
documented f64 loss (`dn-m512` 0.956 ± 0.011). The remaining gap was the
A-pack itself, the same overhead the square-ish branch had already shed: on a
tall K the A matrix is the *dominant* operand (dn-m512's A is 512×11008×8 =
45 MB, past the 33 MB L3), so the SHARED_A pre-pack is a full extra DRAM
sweep (read A, write packed A) before any FLOP runs, and at ~80 ms per call
that sweep is a few percent of the whole GEMM.

The fix routes the rung to the square-ish treatment: **PACK_A=False** (the
micro-kernel reads A from source as strided column broadcasts; the HW
prefetcher handles the MR=6 concurrent row streams) on the same finer
TILE_N=4·NELTS. That also retires the rung's special 8 KB KC cap: the cap
existed to keep the *packed-A* panel near L2, and with no packed panel the
default 16 KB cap (KC=2048 in f64) applies, halving the per-panel C
reload/store traffic. Note the "A panel overflows L2" argument from the first
pass does not return: the unpacked source-A re-read streams from L3 the same
way at either KC, and the deeper panel simply means fewer C round trips.

Interleaved A/B vs linalg (6 epochs × peak-of-8, f64, Machine A), old
(SHARED_A, KC≤1024) → new (pack-B-only, KC≤2048):

| Shape | M×N×K | old | new | new/old |
|---|---|---|---|---|
| dn-m384 | 384×2048×11008 | 0.974 | 1.005 | +3.3% |
| dn-m512 | 512×2048×11008 | 0.968 | 0.995 | +2.8% |
| dn-k4096 | 512×2048×4096 | 0.955 | 0.998 | +4.5% |
| ffn-dn8k | 512×2048×8192 | 1.075 | 1.053 | −2.0% (stays a WIN) |

The same A/B in f32 (old KC=2048 in elements there) is a uniform win with no
give-back: dn-m384 0.974 → 1.016, dn-m512 0.976 → 1.011, ffn-dn8k 0.962 →
1.006, dn-k4096 0.952 → 1.013 (+3.6-6.5%), provided the packed-B tile obeys
the half-L2 rule below.

In full `bench_focus` (10 epochs, 2σ) `dn-m512` goes **0.956 ± 0.011 LOSE →
1.002 ± 0.006 tie** (dead parity with linalg), and no other f64 shape moves
outside its band. The wide-N band measured a wash under the same treatment
(up-m512 0.998×, oddN 0.996×) and keeps SHARED_A. Bit-identical
(`verify_dispatch` max_err 0.0, including tall-K M=257 N=575 K=1100).

## Half-L2 packed-B tile (the f32 KC=4096 cliff)

The first tall-K pack-B-only attempt in f32 regressed 8-14% because
`_l2_resident_kc` sized the packed-B tile (TILE_N × KC) to the *whole*
per-core L2: in f32 the finer tile is 64 columns and the 16 KB cap allows
KC=4096, a 64×4096×4 = 1 MB tile that fills the entire 1 MB L2 and evicts
the A stream and C tiles sweeping through it. The same overflow was latent
in the wide-N SHARED_A band wherever K reaches 4096: an interleaved A/B on
f32 512×4096×4096 measured **0.844 (KC=4096) vs 0.991 (KC=2048)** against
linalg, a 17% cliff, with the K=2048 control shape a dead 0.999 either way.
(The earlier f32 wide-N tuning that picked KC=4096 was measured on Machine
B, where the same 1 MB tile is only *half* the 2 MB L2 and wins; the
byte-based cap carried the byte footprint across machines, but the right
invariant is the *fraction* of L2.)

`_l2_resident_kc` now sizes the L2 term at **half** the per-core L2. Every
f64 pick and every Machine B pick already landed on the 16 KB cap, so those
are unchanged by construction; the only picks that move are the 1 MB/core
f32 K≥4096 ones, off the measured cliff.

## Square-ish: pack-B-only / TileK=K (matching stdlib linalg)

The square-ish branch was the residual loss (sq512 0.80, sq2048 0.93 vs linalg).
emit-asm showed our FMA micro-kernel is byte-identical to linalg's (the same
6x32 / 24-accumulator `vfmadd231pd` nest, no spills), so the gap is entirely in
the orchestration. Reading linalg's open source (`max/kernels/src/linalg/`:
`matmul/cpu/impl.mojo` + `utils.mojo`) pinned down two coupled choices its
`TiledMatmul` makes that ours did not:

1. **It packs only B, never A.** The inner kernel reads A straight from the
   source as strided column broadcasts (the `vbroadcastsd (%r13,%r9)` with the
   `addq %r13` row stride in the asm). On a square A is as large as B, so our
   A-pack was a full M*K copy of pure overhead, and the packed-A buffer competed
   with B and C for L3.
2. **TileK = min(K, 2048)** (`calculate_tile_n_k`, from a 512 KB pack budget /
   kernel_cols=32). For every benchmark square (K <= 2048) that is the *whole* K
   in one panel, so each C micro-tile is swept over all of K and **stored once**,
   versus our KC-panel splitting that re-loaded and re-stored C per panel. TileN
   is then shrunk (128 / 64 / 32 as K = 512 / 1024 / 2048) to keep the packed-B
   tile (TileN x TileK) within the 512 KB budget.

Neither half works alone, which is why the earlier single-lever attempts all
missed (KC sweep = C-once without dropping the A-pack; M-blocking = A-residency
at KC=512; 2D = a tie; each had only one piece). They are also coupled the wrong
way if you keep A-packing: TileK=K with a narrow TileN and a *packed* A explodes
the packed-A re-read (measured 0.43). Dropping the A-pack removes that, because
the unpacked source-A re-read streams from L3 and the HW prefetcher hides it.

Implemented as `PACK_A=False` on `_packed_gemm` (skip the A-pack; the micro-kernel
reads A via `load_a_col[MR](a_view.addr(i, pc+step), k)` at row-stride k) at the
narrow TileN = 4*NELTS, KC = min(K, 2048). Interleaved A/B vs the old pack-both
square path: sq512 1.188, sq768 1.141, sq1024 1.118, sq1536 1.117, sq2048 1.065
(all 2-sigma WIN); bit-identical to naive (verify_dispatch max_err 0.0). The branch
once stepped TileN by K (128/64/32) to keep the packed-B tile near 512 KB, but the
narrow TileN fits that budget at every K and load-balances better (see "Small-N
square" below), so the rungs collapsed to this one call. The headline
wide/tall/decode shapes never enter square-ish, so they are untouched.

### Small-N square: narrow the TileN so the j-tiles balance the cores

This is the indisputable-win change. The square-ish branch used to step TileN by K
(128 / 64 / 32 as K <= 512 / 1024 / 2048) to keep the packed-B tile near 512 KB.
That was tuned for the big squares (sq512/1024/2048) and quietly mis-served the
small ones. sq384 (K=384) took the K<=1024 rung's TileN=64, which splits N=384
into ceildiv(384,64) = **6 j-tiles across 4 cores** — a [2,2,1,1] distribution
whose makespan is 2 k-sweeps where the ideal is 1.5, so two cores idle through the
last round. Measured that was sq384 0.82, the worst loss in the whole suite (and
sq300 0.77, sq320 0.71 the same way). It looked for a while like a pack-both
fallback bug, but instrumenting the actual route showed sq384 never reached the
fallback at all: rung 2 caught it and handed it the unbalanced TileN=64.

The fix is to stop stepping TileN by K and always use the **narrow TileN = 4*NELTS**
(one NR-panel per j-tile, the finest the N-parallel kernel offers). The makespan of
an N-parallel GEMM is ceildiv(j_tiles, num_workers) k-sweeps, so the finest TileN
maximizes j_tiles and minimizes the rounding waste. sq384 then splits into 12 tiles
(3/core, perfectly balanced). The narrow TileN keeps the packed-B tile within
budget at every K (32 x 2048 x 8 = 512 KB at the KC=2048 cap).

Measured in the full bench_focus harness (10 epochs, peak-of-12, 2-sigma verdict —
the trustworthy one per AGENTS.md), baseline -> fixed:

| Shape | before | after |
|---|---|---|
| sq384 | 0.828 **LOSE** | 1.007 tie |
| sq512 | 0.962 tie | 0.987 tie |
| sq1024 | 0.974 **LOSE** | 0.983 tie |
| sq2048 | 1.007 tie | 1.008 tie |
| sq256 | 1.007 tie | 1.022 tie |

The worst loss in the suite is gone and nothing in the band regresses (the headline
decode/prefill/box512 wins and the wide/tall shapes are on other code paths and
unmoved). Bit-identical (verify_dispatch max_err 0.0); the whole branch collapses
from three K-rungs + a pack-both fallback to one line. Two squares (sq300, sq320)
lift a lot but kept a residual loss: their N gives 10 columns, which 4 cores cannot
split evenly (makespan 3 vs ideal 2.5), a limit no TileN choice removes because the
work unit is a whole column. That residual is fixed next.

### Small-N square residual: 2D parallelism for the uneven column count

The narrow TileN balances the band only when the column count ceildiv(N, NR) is a
multiple of the worker count. sq384 gives 12 columns / 4 cores, sq512/1024/2048 give
16/32/64, all even. sq300 and sq320 give 10 columns: ceildiv(10, 4) = 3 columns on
the busiest core where the balanced ideal is 2.5, a [3, 3, 3, 1] makespan. The
column path parallelizes over N only, so a whole column (every M row of it) belongs
to one core; no TileN choice splits that last round.

`_pack_b_only_2d` breaks the column into MR-row blocks, so the unit of work is one
MR x NR C tile and the parallel grid is columns x row_blocks. sq320 becomes 10 x 54
= 540 tiles across 4 cores (135 each, balanced) instead of a 3-column makespan. Each
worker takes a contiguous column-major slice of the grid and packs into a private
[k][NR] buffer the columns its slice touches, reusing it down that column's rows, so
B is still packed about once total (only the columns on a worker boundary get packed
twice) with no global pack barrier and the buffer stays L2-hot. A reads from source
unpacked (PACK_A=False), so there is no packed-A buffer to coordinate across the row
split, and the path is gated to k <= 2048 so each tile is a single C-stored-once
K-panel.

The 2D path costs something real (boundary repacks, more masked tiles), so it is
gated to where it pays: a grid makespan that beats the column makespan by at least
1/8 (ceildiv(columns * row_blocks, workers) vs ceildiv(columns, workers) *
row_blocks). At 10 columns that is a 1/6 cut and the 2D path runs; at 11 columns
only 1/12, and the column path is kept (an interleaved A/B measured 2D a wash-to-
loss there, sq336 0.93). Interleaved A/B vs the column path (12 epochs, peak-of-14):
sq320 1.18, sq300 1.09 (both 2-sigma 2D wins). Full bench_focus (10 epochs, 2-sigma
verdict), baseline -> fixed:

| Shape | before | after |
|---|---|---|
| sq320 | 0.883 **LOSE** | 1.021 tie (221 -> 261 GFLOPS) |
| sq300 | 1.021 tie | 1.080 **WIN** |

The confident LOSE is gone and nothing else moves (the in-band squares keep the
column path, their column counts being even). Bit-identical (verify_dispatch
max_err 0.0).

The same pack-B-only path also took the **big boxes** off the no-pack route. The
small-box branch (`_small_box`, M-parallel no-pack: re-reads all of B per MR-row
block) is right only while B is genuinely small; at the top of its admission
window (B ~ 512 KB) packing B once and reading A unpacked wins, given N splits
into >= num_workers NR-tiles. A pack-B-only/no-pack sweep: sq256 1.14,
512x128x512 1.10, 256x128x512 1.11 (2-sigma WIN at B=512 KB), versus sq192 0.76,
sq128 0.95, sq96 0.82 (no-pack wins below). The cut is B > 384 KB (with the
>= num_workers NR-tile guard); `_small_box` routes those to the same
`_prefill[..., PACK_A=False]` at TileN=4*NELTS, KC=512 (k <= 512 there, so a
single whole-K panel). The small-N squares (sq384/sq300/sq320) take the
square-ish branch's narrow pack-B-only rung (see "Small-N square" above).

## Small-box f32 routing: byte gates vs NR granularity

The small-box admission and route picks are byte-based (`b_bytes = k*n*elem`),
which is right for the L2-residency physics but silently re-routes shapes when
the element type shrinks: an f32 square carries half the bytes of its f64
twin, so squares that in f64 go to `_square_ish` (and its balanced 2D grid)
were landing in `_small_box`'s no-pack / 1D-column routes in f32. Meanwhile
the *granularity* hazards those routes carry scale with `NR = 4*NELTS`, which
**doubles** in f32 (64 lanes). So f32 hit both hazards at once, on shapes the
f64 sweeps never see:

- **sq300 f32** (B = 360 KB, admitted by the 512 KB tier): the no-pack route's
  N-remainder (300 % 64 = 44 columns) runs one row at a time on a single
  accumulator chain, latency-bound at ~1/8 of the register-tile throughput,
  and at 15% of N it dominated the whole GEMM: **0.25 vs linalg** (91 GFLOPS),
  the worst shape in the suite in any dtype.
- **sq320 f32** (B = 400 KB, the pack-B-only branch): ceildiv(320, 64) = 5
  columns across 4 cores on the 1D column path, a [2,1,1,1] makespan:
  **0.73 vs linalg**.

A route A/B (interleaved, peak-of-12, 4 epochs, f32) across the band:

| Shape | B | no-pack | pack-B 1D | pack-B 2D | route |
|---|---|---|---|---|---|
| sq256 | 256 KB | **0.99** | 0.96 | 0.98 | no-pack (keep) |
| sq288 | 324 KB | **0.88** | 0.78 | 0.90 | no-pack (keep) |
| sq300 | 360 KB | 0.25 | 0.85 | **0.95** | 2D |
| sq320 | 400 KB | 1.02 | 0.72 | **1.10** | 2D |
| sq352 | 484 KB | 0.75 | 0.70 | **0.73** | 2D (all ~tie) |
| 512×128×512 | 256 KB | **1.15** | 0.59 | 0.83 | no-pack (keep) |
| 256×128×512 | 256 KB | **1.16** | 0.59 | 1.01 | no-pack (keep) |

Two dispatch changes, both dtype-generic (they fire wherever the arithmetic
says so, it just takes f32's NR=64 to reach them on square-ish shapes):

1. The `_small_box` pack-B-only branch (B > 384 KB) now calls `_square_ish`
   instead of the 1D `_prefill` directly, so its measured makespan gate can
   pick the balanced 2D grid when the column count splits the cores unevenly
   (sq320's 5 columns). For every f64 bench shape the gate resolves to the
   same 1D path and KC rung as before, so f64 routing is unchanged.
2. Shapes whose no-pack N-remainder is at least N/8 (`(n % NR) * 8 >= n`, the
   point where the ~8x-slower row-by-row tail costs more than everything it
   saves) are evicted from no-pack to `_square_ish` too; its masked
   partial-panel microkernel runs the remainder at full register-tile
   throughput. sq288 (remainder 32/288 = 11%) stays no-pack, matching the
   measured wash.

Full `bench_focus --dtype f32` (10 epochs, 2σ verdict), before → after:

| Shape | before | after |
|---|---|---|
| sq300 | 0.253 ± 0.026 **LOSE** (91 GFLOPS) | 0.934 ± 0.032 (349 GFLOPS, 3.8×) |
| sq320 | 0.728 ± 0.016 **LOSE** (337 GFLOPS) | **1.035 ± 0.014 WIN** (509 GFLOPS) |

Nothing else in the f32 set moves outside its noise band (box512 keeps its
1.07 WIN, sq256 1.05 WIN, the wide-N band's small losses were the separate
f32-tiling item, fixed in "Wide-N heavy band" above), and the f64 set is
unchanged (verified with a full 10-epoch f64 run; `verify_dispatch` max_err
0.0). sq300's residual (the 2D grid's masked 44-wide column) is fixed by the
partial-N hot kernel, next section.

## Partial-N panels at full throughput (`_partial_n_microkernel`)

`_masked_microkernel` handles two leftovers with one loop: M-remainder rows
and partial NR-columns. It is deliberately a cold path (no K-unroll, and the
A gather runs a per-row `mr < rows` guard on every K-step), which is the
right trade for the M-tail: at most MR-1 rows of the whole GEMM. It is the
wrong trade for a partial *column*, because every row block of that column is
masked: an N-not-multiple-of-NR shape runs `ceildiv(m, MR)` masked tiles per
K-panel, and once the remainder column is a real fraction of N the whole GEMM
drops with it.

f32 made this visible. NR doubles to 64 lanes, so sq300 (300 = 4*64 + 44) has
1 of 5 columns masked, 20% of all tiles, and sat at **0.831 +/- 0.006 LOSE**
(interleaved, Machine B) while the remainder-free sq320 ran 1.07 WIN at 44%
more GFLOPS. Tile-shape experiments confirmed the masked column (and nothing
else) was the cliff: a narrower NR=32 grid (12-wide remainder) lifted sq300 to
0.878 while an MR=12/NR=32 tile (same padded-FLOP waste, different shape) sank
to 0.63, so the waste arithmetic could not explain the loss; the per-K-step
masking overhead did.

The fix uses what the pack already guarantees: the packed B panel is
zero-padded to full NR, so when all MR rows are live the K-sweep needs no
masking at all. `_partial_n_microkernel` is the same KU-unrolled loop as
`_full_microkernel` (same packed/unpacked A addressing), masking only the C
load and store, once per tile instead of once per K-step. `_packed_gemm` and
`_pack_b_only_2d` route full-rows partial-N tiles there; M-remainder tails
keep the masked kernel (their masked work really is a thin slice).

Full bench_focus (10 epochs, 2-sigma verdict), before -> after:

| Shape | dtype | before | after |
|---|---|---|---|
| sq300 | f32 | 0.812 +/- 0.043 **LOSE** (395 GFLOPS) | 0.985 +/- 0.049 tie (492 GFLOPS) |
| sq300 | f64 | 0.998 tie (251 GFLOPS) | **1.084 +/- 0.038 WIN** (271 GFLOPS) |
| sq320 | f32 | 1.056 WIN | 1.053 WIN (unchanged, no remainder) |

oddN (remainder 63 of N=11007 in f32, 0.6% of its columns) stays at parity as
before; the kernel matters exactly where the remainder fraction is large.
Bit-identical: same FMA order per element, the zero columns contribute
nothing (`verify_dispatch` max_err 0.0 across remainder widths, including the
N=519..575 sweep that exercises every masked-column width on the square-ish
branch).

## Dead end: x86 M-blocking (GotoBLAS loop 3) for the squares

MC blocking (block M so an MC-tall packed-A block stays L2-resident across a
worker's j-tiles, instead of re-reading the full M x KC panel per j-tile) is the
classic GotoBLAS third loop. Tried on the x86 `_packed_gemm` (a comptime-gated BLOCK_M variant that pre-packs
the worker's j-tile B slabs, then sweeps MC-row blocks outside the j-tile loop) it
did not pay. An interleaved A/B blocked-vs-flat (10 epochs, peak-of-12) measured
sq1024 0.973 and sq1536 0.982 (2-sigma LOSE), sq384/sq512/sq768/sq2048 ~1.0 (tie):
a tie-to-slight-loss, never a win. The full M x KC packed-A panel (8 MB on sq2048)
fits the 33 MB L3, and at the ~290 GFLOPS AVX-512 ceiling L3 bandwidth absorbs the
per-j-tile re-read the blocking removes, while pre-packing all the worker's B adds
its own L3 traffic. A separate clean 2D (M-block x N-block) decomposition A/B was
also a dead tie (sq384 1.02, sq2048 0.99). The actual gap was not A-panel reuse
or 2D at all: it was the A-pack overhead + per-panel C reloads, fixed by the
pack-B-only / TileK=K path above (which reads linalg's source rather than guessing
at the structure from the asm).

