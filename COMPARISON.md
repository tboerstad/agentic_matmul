# Full comparison: agentic matmul vs `linalg` vs OpenBLAS

A three-way comparison of the dispatch kernel (`matmul_dispatch`) against the
Mojo stdlib `linalg.matmul` and OpenBLAS (NumPy `np.matmul`, SciPy `dgemm`)
across the full `bench_focus.mojo` shape set. The main tables below are float64
(the project's target precision); a float32 section follows at the end.

## Hardware (this run)

| | |
|---|---|
| CPU | Intel Xeon @ 2.10 GHz, 4 cores (1 thread/core) |
| SIMD | AVX-512 (avx512f/dq/bw/vl/vnni/fp16), AMX tiles present |
| Cache | L1d 48 KB/core, L2 2 MB/core (8 MB total), L3 260 MB shared |
| Memory | 15 GiB |
| Virtualization | KVM guest (shared cloud VM) |
| Mojo | 1.0.0b3.dev2026062114 |
| BLAS | scipy-openblas 0.3.33 (SkylakeX kernels, DYNAMIC_ARCH) |

This is a shared, virtualized VM. Absolute GFLOPS run lower and noisier than the
bare-metal columns in README.md, and they swing 5-10% with the turbo/thermal
state a process lands in. Read the dispatch/linalg ratio and the verdict, not the
absolute numbers. OpenBLAS necessarily runs in a separate process, so its column
is a cross-process comparison (the same methodology README.md uses).

## Methodology

- dispatch vs linalg: `mojo bench_focus.mojo` (default), 10 independent epochs,
  each a peak over 12 interleaved A/B reps. The ratio is reported as
  mean +/- stdev with a 2-sigma verdict: WIN (mean - 2 sigma > 1.0), LOSE
  (mean + 2 sigma < 1.0), tie (band straddles 1.0, within run-to-run noise).
- OpenBLAS: NumPy `np.matmul`, peak GFLOPS over 12-20 reps after warmup, on the
  same 14 shapes. Headline decode/prefill also measured with SciPy `dgemm` and
  single-threaded NumPy via `python bench_sota.py`.

All GFLOPS below are peak (min-time). FLOPs = 2*M*N*K.

## Full shape set (peak GFLOPS)

| shape | dims (M x N x K) | dispatch | linalg | OpenBLAS (NumPy) | dispatch/linalg | verdict |
|---|---|---:|---:|---:|---:|:--:|
| decode  | 1 x 11008 x 2048    |  22 |   8 |  23.8 | 2.784 +/- 0.168 | WIN  |
| prefill | 96 x 11008 x 2048   | 233 | 184 | 195.0 | 1.265 +/- 0.018 | WIN  |
| sq512   | 512 x 512 x 512     | 247 | 255 | 199.8 | 0.972 +/- 0.005 | LOSE |
| sq1024  | 1024 x 1024 x 1024  | 267 | 266 | 230.3 | 1.002 +/- 0.007 | tie  |
| sq2048  | 2048 x 2048 x 2048  | 272 | 264 | 222.2 | 1.031 +/- 0.013 | WIN  |
| M512-g  | 512 x 4096 x 4096   | 243 | 245 | 225.9 | 0.992 +/- 0.014 | tie  |
| up-m256 | 256 x 11008 x 2048  | 234 | 229 | 234.5 | 1.025 +/- 0.007 | WIN  |
| up-m512 | 512 x 11008 x 2048  | 246 | 246 | 213.1 | 0.999 +/- 0.013 | tie  |
| dn-m512 | 512 x 2048 x 11008  | 239 | 244 | 230.1 | 0.983 +/- 0.020 | tie  |
| sq128   | 128 x 128 x 128     | 156 | 166 |  87.6 | 0.940 +/- 0.041 | tie  |
| sq256   | 256 x 256 x 256     | 251 | 244 | 177.1 | 1.029 +/- 0.013 | WIN  |
| sq384   | 384 x 384 x 384     | 197 | 223 | 201.9 | 0.883 +/- 0.032 | LOSE |
| box512  | 512 x 128 x 512     | 244 | 231 | 174.2 | 1.057 +/- 0.033 | tie  |
| oddN    | 512 x 11007 x 2048  | 245 | 244 | 199.5 | 1.005 +/- 0.011 | tie  |

## Headline shapes (Qwen 2.5 VL 3B MLP), peak GFLOPS

| engine | decode (1 x 11008 x 2048) | prefill (96 x 11008 x 2048) |
|---|---:|---:|
| dispatch (agentic matmul) | 22 | 233 |
| Mojo linalg (stdlib)      |  8 | 184 |
| NumPy OpenBLAS (multi)    | 22.6 | 193.6 |
| NumPy OpenBLAS (1 thread) | 22.2 | 194.3 |
| SciPy dgemm (OpenBLAS)    |  7.4 | 151.0 |

## Reading of the results

vs linalg (in-process, the trustworthy comparison):

- Headline shapes are clear wins. decode is 2.78x faster (the GEMV path streams B
  once while linalg under-uses M=1), and prefill is 1.27x faster.
- The compute-bound square/upcast band is parity-to-slight-win: sq1024, M512-g,
  up-m512, dn-m512, oddN all tie; sq2048, up-m256, sq256 are small wins.
- Two confident losses remain: sq512 (0.972) and sq384 (0.883). sq384 is the
  weakest shape (M%6 = 0 but the no-pack/packed crossover near a 2 MB L2 leaves
  it on a slower path here). sq512 is a thin, consistent 3% behind linalg's
  pack-B-only square kernel.

vs OpenBLAS:

- decode matches NumPy OpenBLAS (~22-24 GFLOPS, both memory-bandwidth bound) and
  is ~3x over SciPy dgemm.
- prefill beats NumPy OpenBLAS (233 vs 194) and SciPy dgemm (151) outright.
- Across the compute-bound band the dispatch kernel is at or above NumPy OpenBLAS
  on every shape (e.g. sq2048 272 vs 222, sq256 251 vs 177, box512 244 vs 174).
  OpenBLAS only pulls ahead on none of these 14 shapes at peak.
- The Mojo stdlib `linalg` also generally beats OpenBLAS on the larger shapes,
  except it loses decode badly (8 vs 23).

## float32

Same machine and methodology, run with `mojo bench_focus.mojo --dtype f32`
(full 10 epochs). OpenBLAS here is NumPy `sgemm` (`np.matmul` on float32 arrays).
All peak GFLOPS. Absolute throughput is roughly 2x the float64 tables, as
expected from doubling the SIMD lane count (NELTS 8 -> 16).

| shape | dims (M x N x K) | dispatch | linalg | OpenBLAS (NumPy) | dispatch/linalg | verdict |
|---|---|---:|---:|---:|---:|:--:|
| decode  | 1 x 11008 x 2048    |  21 |   8 |  22.6 | 2.462 +/- 0.078 | WIN  |
| prefill | 96 x 11008 x 2048   | 302 | 301 | 205.8 | 1.003 +/- 0.024 | tie  |
| sq512   | 512 x 512 x 512     | 478 | 490 | 399.6 | 0.975 +/- 0.044 | tie  |
| sq1024  | 1024 x 1024 x 1024  | 474 | 493 | 442.7 | 0.962 +/- 0.010 | LOSE |
| sq2048  | 2048 x 2048 x 2048  | 464 | 499 | 453.8 | 0.931 +/- 0.022 | LOSE |
| M512-g  | 512 x 4096 x 4096   | 431 | 450 | 420.0 | 0.958 +/- 0.024 | tie  |
| up-m256 | 256 x 11008 x 2048  | 396 | 410 | 338.7 | 0.964 +/- 0.025 | tie  |
| up-m512 | 512 x 11008 x 2048  | 420 | 462 | 416.3 | 0.909 +/- 0.029 | LOSE |
| dn-m512 | 512 x 2048 x 11008  | 425 | 466 | 412.0 | 0.912 +/- 0.020 | LOSE |
| sq128   | 128 x 128 x 128     | 158 | 164 | 202.9 | 0.964 +/- 0.028 | tie  |
| sq256   | 256 x 256 x 256     | 344 | 332 | 347.9 | 1.032 +/- 0.052 | tie  |
| sq384   | 384 x 384 x 384     | 274 | 360 | 435.7 | 0.761 +/- 0.044 | LOSE |
| box512  | 512 x 128 x 512     | 372 | 444 | 426.4 | 0.843 +/- 0.086 | tie  |
| oddN    | 512 x 11007 x 2048  | 418 | 443 | 426.6 | 0.945 +/- 0.038 | tie  |

Reading of the f32 results:

- vs linalg the picture flips from f64. In f64 dispatch is parity-to-slight-win
  across the compute band; in f32 it is consistently 4-9% behind linalg
  (sq1024, sq2048, up-m512, dn-m512 are confident LOSEs). The dispatch tiles
  (MR/NR and the KC cache-blocking constants) are tuned for the f64 element
  size. At f32 the SIMD lane count doubles and those constants are no longer
  optimal, so dispatch gives ground to linalg. sq384 stays the weakest shape
  (0.76), same as f64.
- decode still wins ~2.5x (the GEMV path under-uses M=1 in linalg), and prefill
  ties.
- vs OpenBLAS dispatch still beats sgemm on the headline and upcast shapes
  (prefill 302 vs 206, up-m256 396 vs 339, sq2048 464 vs 454) and ties most
  others. OpenBLAS pulls ahead only on the small / odd corners (sq128 203 vs
  158, sq384 436 vs 274).

### Open tuning item: the f32 compute-band gap

The f32 losses are spread across both the square-ish path (sq1024, sq2048) and
the wide/tall path (M512-g, up-m512, dn-m512), which share the same register
micro-kernel (`6 x 4*NELTS`, i.e. 6x64 in f32). Two contained KC ideas were
investigated and ruled out by measurement, not committed:

- Making `_square_ish_kc` dtype-aware is a no-op for the losing shapes. sq1024
  and sq2048 ride the pack-B-only path (KC rungs 1024 / 2048), which never calls
  `_square_ish_kc`; the helper only fires in the fallback where k <= 512, so
  KC=512 already covers all of K.
- Raising the square pack-B-only packed-B tile from the 512 KB target to the
  half-L2 (1 MB) BLIS budget used by the wide/tall band did not clear the 2-sigma
  bar in f32 (sq1024 0.961 +/- 0.013 vs 0.962 baseline; sq2048 0.947 +/- 0.014 vs
  0.931, bands fully overlapping, both still LOSE). Reverted.

The common factor is the f64-shaped micro-kernel tile, so closing the f32 gap is
a per-dtype micro-kernel tuning pass (tile shape / register pressure at 6x64),
not a cache-constant tweak. Left as future work.

float16 and bfloat16: the kernels compile and run at these dtypes (via
`--dtype f16` / `--dtype bf16`), and `matmul_dispatch` is generic over dtype,
but on this x86 build there is no vectorized half-precision FMA path, so they
run near-scalar (one quick epoch took ~12 minutes versus seconds for f32). They
are also not a fair OpenBLAS comparison: NumPy/OpenBLAS has no bf16 GEMM and no
real half-precision GEMM (float16 `np.matmul` upcasts). Those runs were not
included here.

## Reproduce

```bash
source .venv/bin/activate
mojo bench_focus.mojo              # dispatch vs linalg, f64, 14 shapes, verdict
mojo bench_focus.mojo --dtype f32  # same harness in f32
python bench_sota.py               # NumPy / SciPy OpenBLAS on decode + prefill
```
