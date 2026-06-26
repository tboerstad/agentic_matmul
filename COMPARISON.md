# Full comparison: agentic matmul vs OpenBLAS vs Mojo `linalg`

Measured run of the agentic kernel (`matmul_dispatch`) against the two reference
implementations the project tracks: **OpenBLAS** (via NumPy/SciPy) and the **Mojo
stdlib `linalg.matmul`**. All numbers are float64.

## Test machine

| | |
|---|---|
| CPU | Intel Xeon @ 2.80 GHz, 4 cores (1 thread/core), AVX-512 (`avx512f/dq/cd/bw/vl/vnni`) |
| Mojo | 1.0.0b3.dev2026062606 (MAX 26.5 nightly) |
| NumPy / BLAS | NumPy 2.5.0 → scipy-openblas **0.3.33.112.0** (DYNAMIC_ARCH, Haswell kernel, `OPENBLAS_NUM_THREADS` default) |
| Date | 2026-06-26 (UTC) |

This is the **Xeon Skylake 2.80 GHz (4c)** column from the README, re-measured on
the current toolchain. Numbers differ from the README table (shared-VM turbo/thermal
state and a newer OpenBLAS/Mojo nightly), so treat this as a self-contained run.

Harnesses used:
- `python bench_sota.py` — OpenBLAS via NumPy `matmul` and SciPy `dgemm`.
- `mojo bench_linalg.mojo` — stdlib `linalg.matmul`, standalone.
- `mojo bench_focus.mojo` — **agentic vs `linalg`, interleaved in one process**,
  10 epochs × peak-of-12, reported as mean ratio ± stdev with a 2σ verdict. This is
  the only trustworthy agentic-vs-`linalg` comparison (same turbo/thermal state per
  rep). Cross-process absolute GFLOPS (the OpenBLAS rows) are inherently noisier.

---

## Headline Qwen shapes (float64), peak GFLOPS

| Kernel | Decode (1×11008×2048) | Prefill (96×11008×2048) |
|---|---|---|
| **Agentic matmul (dispatch)** | **~12.0** | **~209** |
| Mojo `linalg` (stdlib) | 7.0 | 195 |
| NumPy (OpenBLAS, multi-thread) | 12.3 | 178 |
| NumPy (OpenBLAS, 1 thread) | 12.6 | 178 |
| SciPy `dgemm` (OpenBLAS) | 4.8 | 135 |

Derived ratios on the headline shapes:

| | Agentic ÷ `linalg` | Agentic ÷ NumPy(OpenBLAS) |
|---|---|---|
| **Decode** | **2.20× WIN** (interleaved, 2σ) | ~0.98 (parity) |
| **Prefill** | **1.11× WIN** (interleaved, 2σ) | **~1.17×** |

- **Decode (memory-bandwidth bound):** agentic ≈ OpenBLAS (~12 GFLOPS, both bound by
  DRAM bandwidth streaming B once), and both are ~1.75× the stdlib `linalg` GEMV.
  SciPy's `dgemm` path is far slower here (4.8) — its single-token GEMV overhead.
- **Prefill (compute bound):** agentic is fastest — **+17% over OpenBLAS** and
  **+11% over `linalg`** (the latter measured interleaved, 2σ confirmed).

---

## Agentic vs `linalg` — full shape set (interleaved, 2σ verdict)

`mojo bench_focus.mojo`, 10 epochs × peak-of-12. `WIN`/`LOSE` mean the 2σ band
clears 1.0; `tie` means it straddles 1.0 (within run-to-run noise).

| Shape | M×N×K | Agentic GFLOPS | linalg GFLOPS | Ratio (mean ± stdev) | Verdict |
|---|---|---|---|---|---|
| decode  | 1×11008×2048 | 12 | 5 | 2.196 ± 0.079 | **WIN** |
| prefill | 96×11008×2048 | 209 | 188 | 1.109 ± 0.014 | **WIN** |
| sq512   | 512×512×512 | 271 | 285 | 0.950 ± 0.013 | LOSE |
| sq1024  | 1024×1024×1024 | 277 | 285 | 0.970 ± 0.006 | LOSE |
| sq2048  | 2048×2048×2048 | 277 | 276 | 1.002 ± 0.010 | tie |
| M512-g  | 512×4096×4096 | 260 | 268 | 0.972 ± 0.010 | LOSE |
| up-m256 | 256×11008×2048 | 237 | 246 | 0.963 ± 0.009 | LOSE |
| up-m512 | 512×11008×2048 | 265 | 271 | 0.979 ± 0.005 | LOSE |
| dn-m512 | 512×2048×11008 | 257 | 272 | 0.947 ± 0.013 | LOSE |
| sq128   | 128×128×128 | 159 | 155 | 1.055 ± 0.130 | tie |
| sq256   | 256×256×256 | 232 | 236 | 0.969 ± 0.090 | tie |
| sq384   | 384×384×384 | 189 | 184 | 1.033 ± 0.109 | tie |
| box512  | 512×128×512 | 266 | 259 | 1.025 ± 0.030 | tie |
| oddN    | 512×11007×2048 | 259 | 266 | 0.973 ± 0.006 | LOSE |

**Tally:** 2 WIN (both headline shapes), 7 LOSE, 5 tie. The losses are 2–5%, all on
the heavy compute-bound squares / wide-N large-M band — consistent with the
"Still open" section of the README (micro-kernel parity with `linalg` on the
heaviest GEMMs is the unclosed gap). Small cache-resident boxes (sq128/256/384,
box512) are ties within noise.

### Wider corner/edge sweep (single run — noise-prone, directional only)

`mojo bench_sweep.mojo --iterate`. A single run swings ±5–10% at M ≥ 128, so read
these as directional, not verdicts:

| Shape | Ratio vs linalg | | Shape | Ratio vs linalg |
|---|---|---|---|---|
| M1-gemv (1×4096×4096) | 2.16 WIN | | N4000 (512×4000×2048) | 0.93 |
| M6-batch (6×4096×4096) | 1.37 WIN | | N3072 (512×3072×2048) | 0.96 |
| M512-g (512×4096×4096) | 0.97 | | N-odd (512×11007×2048) | 0.99 |
| sq256 | 1.06 | | K-odd (512×2048×2047) | 0.95 |
| sq512 | 1.06 | | K128 (512×2048×128) | 0.92 |
| sq1024 | 0.96 | | sq128 | 0.98 |
| sq2048 | 1.02 | | box512 (512×128×512) | 0.98 |

Low-M shapes (M=1 GEMV, M=6 batch) win decisively (2.16× / 1.37×). Low-arithmetic-
intensity corners (K=128, odd K) lean to `linalg`, matching its better AVX-512
masked-remainder handling.

---

## OpenBLAS detail (`bench_sota.py`)

float64, warmup 5, 20 iters.

| Shape | Library | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) |
|---|---|---|---|---|---|
| decode | NumPy OpenBLAS (multi) | 4.27 | 3.67 | 10.6 | 12.3 |
| decode | NumPy OpenBLAS (1 thread) | 3.83 | 3.59 | 11.8 | 12.6 |
| decode | SciPy dgemm | 9.83 | 9.44 | 4.6 | 4.8 |
| prefill | NumPy OpenBLAS (multi) | 25.04 | 24.30 | 172.9 | 178.1 |
| prefill | NumPy OpenBLAS (1 thread) | 26.44 | 24.28 | 163.7 | 178.3 |
| prefill | SciPy dgemm | 35.07 | 32.07 | 123.4 | 135.0 |

Note OpenBLAS prefill is single-thread == multi-thread at peak (178), i.e. it does
not scale past one core on this 96-row shape; the agentic kernel parallelizes over
the four cores and reaches ~209.

---

## Bottom line

- **vs OpenBLAS:** agentic **wins prefill by ~17%** and is **at parity on decode**
  (both bandwidth-bound). It beats SciPy `dgemm` on both shapes by a wide margin.
- **vs `linalg`:** agentic **wins both headline shapes** (decode 2.20×, prefill
  1.11×, 2σ confirmed) and trails by 2–5% on the heavy compute-bound square /
  wide-N band — the known open tuning item.

### Reproduce

```bash
source .venv/bin/activate
python bench_sota.py        # OpenBLAS (NumPy/SciPy)
mojo bench_linalg.mojo      # linalg standalone
mojo bench_focus.mojo       # agentic vs linalg, 2σ verdict (the trustworthy one)
mojo bench_sweep.mojo --iterate   # wider corner/edge sweep
```
