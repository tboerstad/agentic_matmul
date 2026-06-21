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

## `_square_ish_kc`: KC=1024 from 1 MB L2 up

KC sets how many times each L2-resident C micro-tile is loaded and stored for
k-panel accumulation: a deeper KC sweeps more of K per residency, so C is
touched fewer times. The cut is where the KC-deep packed-B tile (wide TILE_N x
KC = 512 KB at KC=1024) still coexists in L2 with the packed-A panel and the C
accumulators. The probe (`l2_cache_size()`, ~6 cpuid → ~61 µs VM-exit on a KVM
guest, < 2.5% of a multi-ms square-ish op) picks KC=1024 from a 1 MB/core L2 up,
KC=512 below; smaller shapes skip it (k <= 512 is a single panel either way).

The 1 MB cut replaced an earlier 1.5 MB one. On the 1 MB Skylake an older
nightly measured KC512 > KC1024 (then sq2048 0.84 vs 0.72), so the cut kept that
part on KC512. A current-nightly interleaved A/B (12 epochs, peak-of-15, KC the
only lever at each shape's tile width) flipped it: KC1024/KC512 is sq1024 1.036
+/- 0.007 and sq2048 1.029 +/- 0.007 (both 2-sigma WIN), sq1536 1.015 (tie). The
linalg/codegen the ratio is judged against also moved across nightlies, so the
crossover is a measured, nightly-specific number, not a fixed property of the
part (AGENTS.md warns picks flip across nightlies). Machine B (2 MB) was already
on KC1024 and is unchanged. (This KC pick now governs only the square-ish
*fallback* path; the balanced squares take the pack-B-only path below.)

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
reads A via `load_a_col[MR](a_view.addr(i, pc+step), k)` at row-stride k) plus the
budget (KC, TileN) rungs. Interleaved A/B vs the old pack-both square path: sq512
1.188, sq768 1.141, sq1024 1.118, sq1536 1.117, sq2048 1.065 (all 2-sigma WIN);
bit-identical to naive (verify_dispatch max_err 0.0). The path parallelizes over
N, so it is gated to shapes with >= num_workers j-tiles at the budget TileN; a
small-N square-ish shape (or sq384, whose TileN=128 leaves 3 tiles) keeps the
pack-both fallback. The headline wide/tall/decode shapes never enter square-ish,
so they are untouched.

The same pack-B-only path also took the **big boxes** off the no-pack route. The
small-box branch (`_small_box`, M-parallel no-pack: re-reads all of B per MR-row
block) is right only while B is genuinely small; at the top of its admission
window (B ~ 512 KB) packing B once and reading A unpacked wins, given N splits
into >= num_workers NR-tiles. A pack-B-only/no-pack sweep: sq256 1.14,
512x128x512 1.10, 256x128x512 1.11 (2-sigma WIN at B=512 KB), versus sq192 0.76,
sq128 0.95, sq96 0.82 (no-pack wins below). The cut is B > 384 KB (with the
>= num_workers NR-tile guard); `_small_box` routes those to the same
`_prefill[..., PACK_A=False]` at TileN=4*NELTS, KC=512 (k <= 512 there, so a
single whole-K panel). sq384 stays put (pack-B-only measured 0.97-0.98 vs its
pack-both fallback: at K=384 the avoided A-pack is too small to pay).

## Dead end: x86 M-blocking (GotoBLAS loop 3) for the squares

The SME path's MC blocking (block M so an MC-tall packed-A block stays L2-resident
across a worker's j-tiles, instead of re-reading the full M x KC panel per j-tile)
lifts the squares a lot on Apple, where SME's ~1000 GFLOPS makes memory the wall.
Ported to the x86 `_packed_gemm` (a comptime-gated BLOCK_M variant that pre-packs
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

## Apple Silicon (M-series): big.LITTLE / shared-cache NEON adaptations

The kernels were tuned on 4-core AVX-512 Xeons; the constants above are x86.
Apple Silicon differs on several axes that the original picks get wrong, so the
five NEON adaptations below (and the SME / decode rewrites in the next two
sections) are each behind `comptime CompilationTarget.is_apple_silicon()`
and compile away to the byte-for-byte x86 path off Apple. NELTS auto-scales
(2 for f64 NEON vs 8 for AVX-512), so the register tile is already correct: the
6×(4·NELTS) tile is 24 SIMD accumulators on both, and with KU=2 it needs 32
registers — exactly the NEON file, exactly the AVX-512 file — so the
micro-kernel is register-optimal on Apple unchanged. The wins are all *around*
the kernel.

Measured on an M4 Max (10 P-cores + 4 E-cores, 16 MB cluster-shared L2 reported
by `hw.perflevel0.l2cachesize`, 14 returned by `num_physical_cores()`). Peak
GFLOPS over 40+ interleaved runs vs stdlib `linalg`; the ratio is the
thermal-normalized metric (see `bench_focus.mojo`).

1. **Parallelize the heavy kernels over P-cores only.** The packed GEMM splits
   its j-tiles in a static even share per worker. Handing an E-core (which runs
   the compute-bound micro-kernel at a fraction of a P-core's throughput) an
   equal share makes it the straggler the whole region waits on.
   `compute_core_count()` returns the P-core count (`hw.perflevel0.physicalcpu`)
   on Apple, `num_physical_cores()` elsewhere. Routing `_packed_gemm` and the
   decode GEMV through it lifted the heavy band +24–31% (prefill M=96 ratio
   0.88→1.17, sq1024 0.78→1.12, sq2048 1.03→1.39, down-proj M=512 0.86→1.19) and
   the decode shapes ~10–30%. `_nopack_gemm` stays on all cores: its boxes are
   tiny and cache-resident, the per-block compute never lets an E-core straggle,
   and capping there only idled 4 cores (sq128/sq256 lost 30–44%).

2. **Box-budget cut for the cluster-shared L2.** `l2_cache_size()` reports the
   16 MB *cluster-shared* L2, not a per-core figure, so the Intel `l2/3` no-pack
   admission rule computes a 5.6 MB budget and wrongly admits mid boxes. The
   no-pack route re-reads all of B per MR-row block on all cores, so it only
   beats the packed P-core path for genuinely small boxes: it wins to ~sq320
   (B 800 KB, 1.04) and loses above (sq512 B 2 MB 0.76, box768 B 1 MB 0.73).
   P-core no-pack is worse still (sq512 0.64): no-pack is algorithmically wrong
   for larger B regardless of cores. Capping the Apple budget at a measured
   896 KB routes B > ~900 KB to the packed path: sq512 0.83→0.99 (281→513
   GFLOPS).

3. **Lower the serial cutoff to 2^18 for m ≥ 64.** The 2^19-MAC tiny cutoff was
   tuned on a 4-core Xeon; on a 14-core part the parallel launch is cheap enough
   that a shape with enough M-row blocks to fill the cores wants threads sooner.
   sq64/sq80 (just under 2^19) ran serial at 0.29–0.55; at the 2^18 cutoff they
   take the no-pack box path (0.70–0.75). The `m ≥ 64` guard (the box branch's
   own floor) keeps small-M shapes — too few row blocks to parallelize — on the
   2^19 cutoff, so 8×8×4096 stays serial.

4. **Decode GEMV on P-cores.** The j-parallel GEMV is bandwidth-bound; the open
   question was whether the E-cores add bandwidth or just contend while issuing
   loads slowly. Measured: they contend. P-cores beat all 14 on every decode
   shape (1×11008×2048 ratio 1.52→1.6–1.8). Sub-P-core counts are no better
   (nw=4 dropped to ~half), so the residual gap to OpenBLAS's 1-thread GEMV is
   per-core micro-kernel efficiency, not the worker count.

5. **SHARED_A down to the small-M band.** SHARED_A packs the full A once instead
   of having every N-worker re-pack it; the M≥192 crossover where that starts to
   pay was measured on a 4-core Xeon. The per-worker re-pack is
   `(num_workers − 1)×` redundant, so on 10 P-cores it is ~10× rather than ~4×,
   and the crossover drops below the small-M headline band. Enabled for the
   wide-N (M≤192) and tall-K (65≤M<192) small-M branches on Apple. Positive on
   every small-M shape measured (peak over 150 runs, the effect was inside the
   noise at n=60 and only resolved at n=150): up-proj M=96 1.11→1.20, M=192
   1.12→1.16; down-proj M=96 1.20→1.27, M=160 1.10→1.18. Both Qwen headline
   shapes are M=96, so this is a direct headline win on top of adaptation 1.

What did **not** move on Apple (all within the noise floor, so left at the x86
pick): the large-band KC (the 16 MB shared L2 keeps even KC=2048 resident), the
square-ish wide-vs-fine TILE_N, the prefill-band TILE_N, the decode worker count
below P, the thin-N tile width, and the software-prefetch distance (the HW
prefetcher is strong enough that `PREFETCH_B_DIST` is irrelevant).

## SME (matrix coprocessor): breaking the NEON ceiling (`sme_kernel.mojo`)

The five adaptations above are all *NEON* wins, and NEON is the wall: a memory-free
f64 FMA microbenchmark measures **51 GFLOPS/core**, so the 10 P-cores cap near
**515 GFLOPS** f64. Yet Accelerate hits ~700–790 on the heavy GEMMs and its decode
is single-thread == multi-thread — the signature of the **SME matrix coprocessor**
(`hw.optional.arm.FEAT_SME_F64F64 = 1`, SVL = 512 bit). f64 `FMOPA` does an 8×8
outer-product accumulate into a ZA tile (128 flops/instr). A throughput sweep finds
**two** SME units on the M4 Max (one per P-cluster): 1 thread → 490, 2 → 985,
plateau ~1035 GFLOPS. So NEON cannot reach Accelerate, but a two-thread SME GEMM can
beat it.

`sme_kernel.mojo` drives SME from Mojo via `inlined_assembly`:

- **Micro-kernel:** a 16×32 C block held in all eight ZA.D tiles (a 2×4 grid of
  8×8). Per K-step: 2 A-loads + 4 B-loads + 8 FMOPA. A is packed column-major
  (16-row panels); B is read in place (its 32 row-columns are contiguous), advancing
  one row (`ldb`) per step — only A needs packing.
- **Blocking:** GotoBLAS (pc, ic-block, jt, it). pc blocks K by KC, ic-block blocks
  M by MC so the MC-tall A block stays L2-resident across the worker's j-tiles (A
  read from DRAM once per worker, not once per j-strip — this is what lifts sq2048
  534→757 and dn-m512 439→781). C accumulates across k-panels in ZA: the first panel
  zeroes ZA (`_sme_micro_z`), later panels reload C into ZA, add, store
  (`_sme_micro_a`). Config: no blocking when A fits L2 and N is wide (prefill,
  up-proj); KC=512/MC=128 for squares; KC=384/MC=256 for K ≥ 4096.
- **Two SME units:** parallelize the N j-strips across `nw=2` workers.
- **Any M/N:** N % 32 uses a column overlap tile (a full 32-wide tile shifted to start
  at N−32, separate pass, overwrites the small overlap with identical full-sum values).
  M % 16 is **folded into the main sweep** as a partial last i-tile — A's panel is
  zero-padded past row M, the full tile is computed, and only the rM valid rows are
  written straight to C by row-limited micro-kernels (`_sme_micro_z_part`/`_a_part`,
  two bounded asm loops over za0–3 / za4–7). Keeping it in the blocked sweep preserves
  B-reuse/pipelining: a separate M-remainder pass had left M100 at 0.81×; folded it is
  0.92× (and large M%16 ≈ 0.98×). Odd-N large shapes (512×11007×2048: 514→822) stay on
  SME instead of dropping to the NEON ceiling.
- **The register-file trap:** `smstart` zeroes the whole SVE register file (z0–z31,
  p0–p15). The inline asm MUST clobber all of them, or the compiler assumes the
  callee-saved v8–v15 survive and silently corrupts the caller's FP — a heisenbug
  that made a folded constant `2.0*1000/4` evaluate to 0.0.

Routed for f64, `m ≥ 64`, `n ≥ 32`, `M·N·K ≥ 2^21`. Measured dispatch-vs-Accelerate
peak: prefill 1.02×, sq512 1.50×, sq1024 1.27×, sq2048 1.10×, sq256 1.38×, sq384
0.97×, M512-g 0.97×, up-proj 1.06–1.12×, dn-proj 0.98×, odd-N 1.01×.

**Open residual:** tiny (128³, ~0.77×), narrow-N box (512×128×512, ~0.90× at peak),
and sub-16-M batch (M=6, ~0.68×) shapes. Each needs a specialized micro-kernel a
production BLAS keeps on hand — a low-overhead small-matrix path (sq128: too few
j-tiles for two SME units, per-tile `smstart`/`smstop` bites), a narrow-N tile
orientation (box512), or an 8-row tile (M<16, where the 16-row tile wastes >half its
FMOPA). These sit below the NEON ceiling too, so NEON does not reach them either, and
Accelerate's mature AMX small/narrow path wins. The square decode GEMV (1×4096×4096)
and M%16 batches (M100) are now handled (over-decomposed K / folded partial tile).

Decode (M = 1) is bandwidth-bound, not an SME problem: a memory-free read sweep hits
242 GB/s on the P-cores with plain NEON, *above* Accelerate's ~228 GB/s decode, but
the old column-split GEMV only sustained 132 (it read a strided column slice of
row-major B). Splitting by K-rows — each worker streams contiguous B rows into a
private partial-C, then a parallel reduce — fixes the access pattern. A fixed nw-way
K split, though, left ~1/3 of bandwidth on the table when K did not divide evenly
across the P-cores (1×4096×4096 ran 0.68× at nw=10); **over-decomposing K into ~4×
more chunks than workers** lets `parallelize` work-steal a balanced share, taking both
the headline 1×11008×2048 and the square 1×4096×4096 to ~1.0× Accelerate (37→57,
40→58). (Apple-gated: the partial reduction reorders the f64 sum, ~1e-12, within the
dispatch's 1e-7 tolerance but not bit-identical to the x86 column-split.)
