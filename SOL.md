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
- **bf16 is broken**: the same generic microkernel drops to 4-5 GFLOPS,
  0.02x linalg. LLVM has no bf16 SIMD FMA to lower to, so it emulates
  element-wise. Every bf16 shape is a 20-60x LOSE.

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

## 1. SOL-aware verdicts: report % of roofline, not just ratio-vs-linalg

Add a tiny FMA-peak + bandwidth self-measurement (port `sol/sol_fma.c` /
`sol_bw.c` to Mojo, or shell out to them) and have `bench_focus` print each
shape as a % of its roofline SOL next to the linalg ratio. Motivation: the
linalg-relative lens has repeatedly pointed agents at the wrong work. The
"open problem" squares are at 96-97% of the machine peak (tuning them is
wasted effort), while prefill, a comfortable WIN vs linalg, hides the
suite's largest real gap at 76% of SOL. A % SOL column also transfers across
machines and Mojo nightlies, where the linalg denominator keeps moving.
Cheap, pure tooling, no kernel risk, and it compounds: every later agent
makes better decisions.

## 2. Fix bf16: 4-5 GFLOPS today, 0.02x linalg, a 60x bug-class win

The dtype-generic microkernel emulates bf16 FMA element-wise (no native bf16
SIMD FMA exists). Fix: keep bf16 storage and convert to f32 in registers,
FMA in f32, convert C back once. The packing stage already touches every
element, so upconversion can ride the pack for B and A (packed panels become
f32), with only load-side conversion in the no-pack/GEMV paths (a bf16 load
plus a 16-bit shift makes an f32). Ceiling is the f32 peak (666 GFLOPS);
linalg's ~290 shows conversion-in-loop already sustains ~44% of it, so
pack-time conversion should beat linalg. `avx512_bf16`'s `vdpbf16ps` is an
optional second step. Verify vs naive with a tolerance (bf16 accumulation
differs by construction); `verify_f32_routes.mojo` shows the pattern.

## 3. AMX tile microkernel for bf16: raise the ceiling ~15x

This part has `amx_bf16`/`amx_tile`: `tdpbf16ps` does a 16x32 by 32x16 tile
FMA into f32 accumulators, ~1024 flops/cycle/core, ~9.8 TFLOPS paper across
4 cores (vs 666 GFLOPS AVX-512 f32). linalg does not use it (bf16 linalg
measured ~290 GFLOPS), so even 25-30% of the AMX paper peak is ~10x the best
bf16 number on the machine today. Ingredients: request the AMX xstate from
the OS (`arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA=18)`, one
syscall via `external_call`), `ldtilecfg`, then `tdpbf16ps` through
`llvm_intrinsic`. B must be packed K-pairs-interleaved, which the packed
kernels' pack stage can produce. Start with one serial 32x32 C tile kernel
vs naive (tolerance check), then graft onto `_packed_gemm`'s tiling. High
effort, highest ceiling in the repo. Gate it on `os_is_linux()` plus a cpuid
check so other machines keep the current path. Watch for AMX clock/license
throttling: measure the f64 suite before/after to confirm no regression.

## 4. Take prefill from 76% to ~90% of SOL: C traffic and pack overlap

Prefill (M=96) runs in the m <= 192 wide-N band with KC=256, so the 8.5 MB C
is loaded and stored K/KC = 8 times per call (~135 MB of extra C traffic,
roughly 12% of runtime at L3 bandwidth). The heavy bands already won their
last 2-6% by moving to a single C-stored-once k-panel (see DESIGN.md
`_l2_resident_kc`), and the same treatment is untried here: with pack-B-only
and a deeper KC, the per-worker packed-B tile (TILE_N x KC) still fits half
the L2 while A (1.6 MB) streams from L3 unpacked. If the C-traffic fix
lands, the remaining gap is pack/compute serialization; try double-buffering
the B j-tile pack (pack tile j+1 while computing tile j, or prefetch the
next source panel during the current microkernel sweep). Prefill is the
headline shape; +10% of SOL is ~30 GFLOPS, the largest available compute win
in f64/f32.

## 5. Decode: close the gap to the L3 roofline, and make the benchmark honest

Decode sits at 72 GB/s effective vs a measured 108 GB/s L3 read ceiling, and
MKL demonstrates 99 GB/s (24.7 GFLOPS) on the same machine, so ~1.4x is on
the table. Profile where the bytes go: per-worker chunk size vs L1
(the current L1-resident column chunks), prefetch distance, and whether
4-way j-parallelism self-interferes in L3 (NumPy reaches 23 GFLOPS with ONE
thread; try 2-3 workers). Second, the benchmark keeps B hot in L3 across
reps, which a real single-pass decode never sees; add a cold-B variant
(rotate through several 180 MB B buffers so each rep starts uncached) so
wins are real. Past the roofline, the only lever is fewer bytes: f16 decode
already measures 93 GFLOPS, and a working bf16 path (ideas 2/3) halves bytes
again. An int8-weight GEMV with f32 accumulate would quadruple effective
decode throughput vs f64 at the same bandwidth.

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
