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

## `_square_ish_kc`: opposite picks per machine

Interleaved A/B (peak/25) gives Machine A (1 MB) KC512 > KC1024 (sq2048 0.84 vs
0.72, sq1024 0.84 vs 0.80), while Machine B (2 MB) measured KC1024 > KC512
(sq768..2048 +3–6%) — opposite conclusions a single hardcoded KC can't satisfy.
The `l2_cache_size()` probe is ~6 cpuid → ~61 µs VM-exit on a KVM guest, < 2.5%
of a multi-ms square-ish op; smaller shapes skip it entirely.
