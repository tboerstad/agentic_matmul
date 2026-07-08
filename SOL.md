# Speed-of-light (SOL) and SOTA analysis

Measured 2026-07-06 on the 2.10 GHz Xeon (Machine B class: 4 cores, AVX-512,
2 MB/core L2, 260 MB shared L3, KVM). The CPU also reports `amx_bf16`,
`amx_tile`, `amx_int8`, `avx512_fp16`, and `avx512_bf16`, which matter for the
low-precision ideas below. All compute and bandwidth ceilings here are
**empirical**, measured in the same boot as the kernel benchmarks with the
microbenchmarks in `sol/` (`bash sol/run.sh`). Re-measure on your machine
before comparing; every number is hardware- and even boot-specific.

## Compute SOL: measured FMA peak beats the paper formula

The paper formula in AGENTS.md uses the base clock:

```
2.1 GHz x 8 lanes x 2 (FMA) x 2 units x 4 cores = 268.8 GFLOPS f64
```

Measured with a 16-chain AVX-512 FMA microbenchmark (`sol/sol_fma.c`), the
cores turbo well above base under full-width FMA load:

| | 1 core | 4 cores | implied clock |
|---|---|---|---|
| f64 FMA peak | 82 GFLOPS | **310 GFLOPS** | 2.57 / 2.43 GHz |
| f32 FMA peak | 166 GFLOPS | **666 GFLOPS** | same |

The measured all-core peak is 15% above the paper number. Use the measured
peak as the denominator for efficiency claims, and re-measure it in the same
boot as the kernel run (turbo policy varies across VM placements).

Derived ceilings for the narrow types on this part:

- **f16 (avx512_fp16 native FMA): ~1330 GFLOPS** (2x the f32 peak: 32 lanes).
- **bf16 via f32-convert: ~666 GFLOPS** (compute in f32 after upconvert).
- **bf16 via AMX (`tdpbf16ps`): ~9.8 TFLOPS paper** (1024 flops/cycle/core at
  ~2.4 GHz). Unvalidated on this machine; even 30% of it is ~3 TFLOPS.

## Memory SOL

Measured with an AVX-512 read sweep at cache-sized footprints (`sol/sol_bw.c`):

| Level | 1 thread | 4 threads |
|---|---|---|
| DRAM read | 14.6 GB/s | **57 GB/s** |
| L3 read | 27.7 GB/s | **108 GB/s** |
| L2 read | ~116 GB/s | ~513 GB/s |
| L1 read | ~205 GB/s (ILP-limited) | |

Roofline ridge points (f64, 4 cores): compute peak / DRAM BW = 5.4 flops/byte,
compute peak / L3 BW = 2.9 flops/byte. Every `bench_focus` shape except decode
has arithmetic intensity 10 to 170 flops/byte, so **the whole suite except
decode is compute-bound** with SOL = 310 GFLOPS (f64). Decode
(1x11008x2048) has AI = 0.25 flops/byte; its B matrix is 180 MB, which fits
the 260 MB L3, so its roofline is **27 GFLOPS at L3 bandwidth** (14 GFLOPS if
B ever streams from DRAM).

## Where the kernels stand vs SOL (f64, this boot)

`bench_focus` 10 epochs, mean of per-epoch peak GFLOPS, same boot as the
310 GFLOPS FMA measurement:

| Shape | dispatch | linalg | % of SOL (dispatch) |
|---|---|---|---|
| decode | 18 | 9 | 67% of L3 roofline (27) |
| **prefill** | 237 | 217 | **76%** |
| sq512 | 285 | 290 | 92% |
| sq1024 | 299 | 302 | 96% |
| sq2048 | 302 | 301 | **97%** |
| M512-g | 268 | 281 | 86% |
| up-m256 | 255 | 264 | 82% |
| up-m512 | 277 | 284 | 89% |
| dn-m512 | 282 | 283 | 91% |
| sq128 | 157 | 159 | 51% (few-µs op, overhead-bound) |
| sq256 | 270 | 264 | 87% |
| sq384 | 284 | 283 | 92% |
| box512 | 268 | 258 | 86% |
| oddN | 275 | 282 | 89% |
| sq300 | 258 | 236 | 83% |
| sq320 | 278 | 269 | 90% |

The SOL lens inverts the linalg-relative picture. The big squares, long
treated as the open problem because they show ~0.96-0.99 vs linalg, sit at
96-97% of the machine's measured FMA peak: **both kernels are at the wall
there, and there is almost nothing left to win.** The headline prefill, a
1.09 WIN vs linalg, is the largest real compute-bound gap in the suite at
76% of SOL (~70 GFLOPS on the table). Decode is at 72 GB/s effective
bandwidth vs the measured 108 GB/s L3 read ceiling.

## Per-dtype snapshot (quick single-epoch, absolute GFLOPS)

| Shape | f64 disp | f32 disp | f16 disp | f16 linalg | bf16 disp | bf16 linalg |
|---|---|---|---|---|---|---|
| prefill | 237 | 484 | 850 | 327 | **5** | 263 |
| sq1024 | 299 | 589 | 1021 | 355 | **5** | 286 |
| sq2048 | 302 | 617 | 1014 | 353 | **5** | 296 |
| decode | 18 | 36 | 93 | 43 | **5** | 33 |
| dtype SOL | 310 | 666 | ~1330 | | ~666 (AVX-512) / ~9800 (AMX) | |

Two headlines:

- **f16 is a large undocumented win**: the generic microkernel lowers to
  native `avx512_fp16` FMA and runs 2.4-2.9x faster than linalg (which stays
  around 350 GFLOPS) at ~77% of the 2x-f32 ceiling.
- **bf16 was broken at the time of this table**: the generic microkernel
  dropped to 4-5 GFLOPS, 0.02x linalg, because LLVM has no bf16 SIMD FMA to
  lower to and emulated element-wise. Fixed since (idea 2 below, DONE): bf16
  now computes in f32 and every shape is a WIN or tie vs linalg. The bf16
  columns above are the historical broken numbers. On AMX parts (this
  machine), bf16 shapes with M % 32 == N % 16 == K % 32 == 0 now dispatch to
  the tdpbf16ps tile kernel (idea 3 below, DONE) and run 1.1-2.4 TFLOPS,
  past even the f16 column's AVX-512 ceiling.

## SOTA baselines (bench_sota.py, this boot, separate processes)

Prefill (96x11008x2048, f64), peak GFLOPS: MKL 233, NumPy multi 216,
NumPy 1-thread 209, SciPy dgemm 159. Dispatch's 237 interleaved mean-of-peaks
still leads, and MKL is now the strongest external baseline (bench_sota
gained MKL since the README tables were written).

Decode (1x11008x2048, f64), peak GFLOPS: MKL 24.7, NumPy 1-thread 23.4,
NumPy multi 9.4, SciPy 6.3. **MKL's 24.7 GFLOPS is 99 GB/s effective, near
the 108 GB/s measured L3 read ceiling. Dispatch measured 18 GFLOPS (72 GB/s)
in the same boot.** Cross-process comparisons are noisy (see the methodology
note in README.md), yet the gap to the measured roofline is consistent:
decode has ~1.5x headroom on this part, and OpenBLAS reaches ~23 GFLOPS with
a single thread.

---

# Five ideas for the next agents

Ordered by expected value. Judge every kernel change with `mojo
bench_focus.mojo` (2-sigma verdict), never a single ratio.

## 1. SOL-aware verdicts: report % of roofline, not just ratio-vs-linalg — DONE

Implemented. `sol.mojo` is a Mojo port of `sol/sol_fma.c` and `sol/sol_bw.c`
(plus an L3-size probe added to `cpu_cache.mojo`): it self-measures the
all-core FMA peak (in the dtype under test), L3 and DRAM read bandwidth, and
the detected L3 size in the same process as the kernel run. `bench_focus` now
measures those ceilings once up front, prints a SOL banner, and adds a **%SOL**
column (dispatch GFLOPS / roofline, where roofline = min(FMA peak, BW ×
arithmetic intensity) with the bandwidth chosen as L3 when the working set fits
the detected L3, else DRAM) plus a `compute`/`bw` bound tag on every shape.

The lens change is immediate: on a Machine-A-class box (2.80 GHz, 1 MB/core L2,
33 MB L3), `bench_focus` reports `sq2048` as a LOSE vs linalg (0.987) yet at
92% of SOL (at the wall), and `prefill` as a WIN vs linalg (1.117) at only 63%
of SOL (the real gap). Decode reads above 100% of its cold-DRAM roofline
because its 180 MB B cannot fit this 33 MB L3, so the harness's keep-B-hot reps
beat a true cold pass, which is exactly the honesty gap idea 5 calls out.

Because the numbers are re-measured every run, they transfer across machines
and Mojo nightlies, where the linalg denominator keeps moving. The original
motivation held: the linalg-relative lens repeatedly pointed agents at the
wrong work (the near-wall squares vs the real prefill gap). The tables above in
this file are from a specific Machine-B boot and will differ from any given
run; trust the `bench_focus` banner for the machine you are on.

## 2. Fix bf16: 4-5 GFLOPS today, 0.02x linalg, a 60x bug-class win — DONE

Implemented as prescribed: bf16 keeps bf16 storage for A, B and C and
computes in f32 (`gemm._compute_dtype`). The upconversion rides the pack for
the packed kernels (`_pack_a_panel`/`_pack_b_slab` widen as they copy, so
packed panels are f32 and the hot K-sweep is exactly the f32 microkernel),
the no-pack and GEMV paths widen on load (`load_a_col`/`load_b_row` take a
cast target), and C narrows once per register-tile store. The decode GEMV
accumulates into an f32 staging row (a bf16 accumulator would round to 8
mantissa bits every KU steps) while B streams in bf16 at half the f32 bytes.
Tile geometry (NELTS/NR/TILE_N) and the byte-based routing gates follow the
compute dtype, so bf16 rides the f32-measured dispatch map; the storage-byte
gates initially routed bf16 sq320 onto the no-pack path at 0.54 vs linalg,
and routing by compute bytes brought it back (DESIGN.md "bf16: keep bf16
storage, compute in f32"). Every cast folds away when storage equals
compute: verify_dispatch (f64) still prints max_err 0.0.

Measured on a Machine-A-class box (2.80 GHz Skylake, 4 cores, no
avx512_bf16), bench_focus 10 epochs, 2-sigma verdicts, before -> after:

| Shape | before (disp / ratio) | after (disp / ratio) |
|---|---|---|
| decode | 2 / 0.29 | **42 / 3.87 WIN** |
| prefill | 1 / 0.006 | **216 / 1.22 WIN** |
| sq2048 | 1 / 0.005 | **242 / 1.10 WIN** |
| up-m512 | 1 / 0.005 | **405 / 1.85 WIN** |
| dn-m512 | 0 / 0.005 | **242 / 1.06 WIN** |
| sq300 | 0 / 0.013 | **161 / 1.43 WIN** |

All 16 shapes: 14 WIN, 2 tie (sq128, sq256), no losses. The wide-N band
lands at 360-405 GFLOPS, past the ~350 the f16 path gets from linalg and
approaching the f32 kernels on the same shapes. Correctness is gated by
`verify_f32_routes.mojo`, which now covers every dispatch route in bf16
against a naive f64 reference: max_rel ~0.006, the single f32-to-bf16
rounding of C. `bench_focus` measures its SOL banner in the compute dtype,
so the bf16 %SOL column is a real f32-peak roofline instead of the
emulated-FMA rate. `avx512_bf16`'s `vdpbf16ps` remains an optional second
step on machines that have it (this one does not).

## 3. AMX tile microkernel for bf16: raise the ceiling ~15x — DONE

Implemented as prescribed (`amx.mojo`, dispatched from `matmul_dispatch`;
design details in DESIGN.md "AMX bf16"). The xstate request is the one
`arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` syscall via
`external_call`, memoized with the cpuid AMX-TILE/AMX-BF16 check; the tile
ops (`ldtilecfg`, `tileloadd`, `tdpbf16ps`, `tilestored`, `tilerelease`) are
`inlined_assembly` blocks because Mojo exposes no AMX intrinsics. The kernel
is N-parallel over 32-column j-tiles: B is VNNI pair-interleaved per j-tile
(`SIMD.interleave`), A is read unpacked by `tileloadd`'s strided row gather,
and each 32x32 C block is a 2x2 grid of f32 accumulator tiles that stays in
tile registers across the whole K sweep (C stored once, then narrowed to
bf16). Gated to bf16 with m % 32 == n % 16 == k % 32 == 0 above the tiny
cutoff on a machine that grants AMX; everything else falls through to the
existing cascade unchanged (verify_dispatch f64 max_err 0.0, and the f64
suite re-ran clean).

Measured on this Machine-B-class box (2.10 GHz Granite Rapids, 4 cores),
bench_focus --dtype bf16, 10 epochs, 2-sigma verdicts, mean dispatch GFLOPS:

| Shape | before (f32-compute path) | after (AMX) | vs linalg |
|---|---|---|---|
| sq2048 | ~296 | **2438** | 6.67 WIN |
| sq384 | ~250 | **2262** | 6.67 WIN |
| sq1024 | ~286 | **2247** | 6.20 WIN |
| dn-m512 | ~300 | **1923** | 5.46 WIN |
| M512-g | ~330 | **1912** | 5.36 WIN |
| up-m512 | ~405 | **1829** | 5.19 WIN |
| prefill | ~500 | **1051** | 3.21 WIN |

All 16 shapes WIN (decode, oddN and sq300 stay on the AVX-512 routes: M=1,
n % 16 != 0, k % 32 != 0). The squares land at ~25% of the ~9.8 TFLOPS AMX
paper peak — inside the "even 25-30%" band above — and ~3.7x past the 666
GFLOPS AVX-512 f32 ceiling, so the bf16 %SOL column now reads far past 100%
on AMX shapes: the banner still measures the AVX-512 f32 FMA peak. Two
follow-ups for the next agents: measure a `tdpbf16ps` peak in `sol.mojo` so
the bf16 roofline is honest again, and tune the AMX wide-N band (prefill at
1.05 TFLOPS is well under the squares' 2.4; the pack/stream traffic per flop
is higher there, and nothing AMX-side has been tuned yet).

## 4. Take prefill from 76% to ~90% of SOL: C traffic and pack overlap — DEAD END (measured)

Both halves of this idea were implemented and measured on a Machine-B-class
box, and neither pays; the full experiment tables are in DESIGN.md "Dead
end: the prefill-band C-traffic and pack-overlap ideas". Summary: the
C-stored-once deeper k-panel (pack-B-only KC=2048, the m > 192 treatment)
runs 0.83 vs the current rung at M=96 — at small M the deep panel breaks the
L2 residency of the packed-A panel + B slab + C tile, which the KC=256
geometry is the only one to preserve (every KC/TILE_N/pack-choice neighbor
was swept; all lose). A phase-split measurement (pack-only / compute-only
variants of the same kernel) shows the real structure: the B pack is 16% of
runtime at M=96 and runs at its own per-core copy-throughput floor, compute
alone reaches 92% of the FMA peak, and the two are strictly serialized.
Prefetch-based overlap (full-row pack prefetch at any distance; bursts of
next-slab prefetches between microkernel sweeps at t0/t1/t2 locality) is a
wash to a 2-5% LOSS. The rung's ~76% of SOL equals ~96% of the serialized
pack+compute model, so ~90% of SOL is unreachable with this kernel
architecture; the levers left are a second memory agent per core (SMT /
dedicated pack thread) or fewer B bytes per flop (narrower dtypes — which is
what idea 3's AMX path now delivers for bf16).

## 5. Decode: close the gap to the L3 roofline, and make the benchmark honest

UPDATE (2026-07-08, Machine-B-class boot): the performance half no longer
reproduces. Decode re-measured 25 GFLOPS = ~101 GB/s effective, ~94% of the
same-process 108 GB/s L3 read ceiling and above the MKL 24.7 cited below —
the 18 GFLOPS / 72 GB/s standing was boot-specific. The idea's tuning
suggestions all measured worse or flat (2 workers 0.50x, 3 workers 0.76x,
KU=16 0.97x, doubled prefetch distance 0.98x; DESIGN.md "SOL.md idea 5
note"). What remains open from this idea is the benchmark-honesty half (the
cold-B variant below) and fewer-byte weight formats (bf16 decode now rides
idea 2's path at ~2.1x linalg; int8 weights would halve bytes again).

Original analysis: decode sits at 72 GB/s effective vs a measured 108 GB/s
L3 read ceiling, and
MKL demonstrates 99 GB/s (24.7 GFLOPS) on the same machine, so ~1.4x is on
the table. Profile where the bytes go: per-worker chunk size vs L1
(the current L1-resident column chunks), prefetch distance, and whether
4-way j-parallelism self-interferes in L3 (NumPy reaches 23 GFLOPS with ONE
thread; try 2-3 workers). Second, the benchmark keeps B hot in L3 across
reps, which a real single-pass decode never sees; add a cold-B variant
(rotate through several 180 MB B buffers so each rep starts uncached) so
wins are real. Past the roofline, the only lever is fewer bytes: f16 decode
already measures 93 GFLOPS, and the bf16 path (idea 2, now DONE: bf16 decode
measured 42 GFLOPS / 3.9x linalg on a Machine-A box) halves bytes again. An
int8-weight GEMV with f32 accumulate would quadruple effective decode
throughput vs f64 at the same bandwidth.

## Honorable mentions

- Add MKL to the interleaved harness as a second baseline (it is now the
  strongest external SOTA on this machine, and bench_sota's cross-process
  numbers are not comparable to bench_focus ratios).
- A one-shot per-machine autotune cache for the tunables the docs call
  hardware-specific (KC, TILE_N, MR, the L2/3 cut): measure once, store, load
  at dispatch. Machines A and B already want opposite KC picks.
- sq128-class few-µs shapes idle at ~51% of SOL on launch/sync overhead; a
  persistent worker pool (avoid per-call `parallelize` setup) is the known
  fix if anyone cares about that band.
