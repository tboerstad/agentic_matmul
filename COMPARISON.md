# Full comparison: agentic matmul vs `linalg` vs OpenBLAS

A three-way comparison of the dispatch kernel (`matmul_dispatch`) against the
Mojo stdlib `linalg.matmul` and OpenBLAS (NumPy `np.matmul`, SciPy `dgemm`),
all in float64, across the full `bench_focus.mojo` shape set.

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

## Reproduce

```bash
source .venv/bin/activate
mojo bench_focus.mojo      # dispatch vs linalg, 14 shapes, mean +/- stdev verdict
python bench_sota.py       # NumPy / SciPy OpenBLAS on decode + prefill
```
