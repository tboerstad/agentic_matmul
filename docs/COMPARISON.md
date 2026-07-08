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
- `python bench/sota.py` — OpenBLAS via NumPy `matmul` and SciPy `dgemm`.
- `mojo -I . bench/linalg_baseline.mojo` — stdlib `linalg.matmul`, standalone.
- `mojo -I . bench/focus.mojo` — **agentic vs `linalg`, interleaved in one process**,
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

`mojo -I . bench/focus.mojo`, 10 epochs × peak-of-12. `WIN`/`LOSE` mean the 2σ band
clears 1.0; `tie` means it straddles 1.0 (within run-to-run noise). The numbers
below are **after** the large-M wide-N KC fix described in the next section; the
"was" column is the pre-fix baseline.

| Shape | M×N×K | Agentic GFLOPS | linalg GFLOPS | Ratio (mean ± stdev) | Verdict | was |
|---|---|---|---|---|---|---|
| decode  | 1×11008×2048 | 13 | 6 | 2.250 ± 0.074 | **WIN** | 2.196 WIN |
| prefill | 96×11008×2048 | 211 | 189 | 1.117 ± 0.015 | **WIN** | 1.109 WIN |
| sq512   | 512×512×512 | 272 | 283 | 0.960 ± 0.012 | LOSE | 0.950 LOSE |
| sq1024  | 1024×1024×1024 | 276 | 282 | 0.978 ± 0.007 | LOSE | 0.970 LOSE |
| sq2048  | 2048×2048×2048 | 279 | 279 | 1.002 ± 0.015 | tie | 1.002 tie |
| M512-g  | 512×4096×4096 | 263 | 266 | 0.986 ± 0.016 | tie | 0.972 **LOSE** |
| up-m256 | 256×11008×2048 | 242 | 249 | 0.972 ± 0.090 | tie | 0.963 **LOSE** |
| up-m512 | 512×11008×2048 | 267 | 272 | 0.979 ± 0.047 | tie | 0.979 **LOSE** |
| dn-m512 | 512×2048×11008 | 258 | 272 | 0.947 ± 0.008 | LOSE | 0.947 LOSE |
| sq128   | 128×128×128 | 166 | 168 | 0.985 ± 0.054 | tie | 1.055 tie |
| sq256   | 256×256×256 | 246 | 247 | 0.997 ± 0.020 | tie | 0.969 tie |
| sq384   | 384×384×384 | 186 | 182 | 1.022 ± 0.060 | tie | 1.033 tie |
| box512  | 512×128×512 | 261 | 257 | 1.016 ± 0.039 | tie | 1.025 tie |
| oddN    | 512×11007×2048 | 263 | 265 | 0.991 ± 0.013 | tie | 0.973 **LOSE** |

**Tally (after fix):** 2 WIN · **3 LOSE** · 9 tie — down from **7 LOSE** before. The
KC fix flipped four wide-N large-M shapes (M512-g, up-m256, up-m512, oddN) from a
confident 2–4% LOSE to parity (in isolation they measure 0.997–1.020; the full
14-shape run adds thermal contention and wider variance, hence `tie`). The 3
remaining losses are the heavy squares (sq512/sq1024) and the tall-K down-proj
(dn-m512, K=11008) — the documented algorithmic gap that needs pack/compute
overlap. Small cache-resident boxes (sq128/256/384, box512) are ties within noise.

## Improvement made on this branch: large-M wide-N KC

Investigating the gap, the large-M wide-N/tall-K band (`_wide_n`/`_tall_k`,
`m > 192`) was sweeping K in a half-L2 `KC=1024` panel (and a hardcoded `KC=512`
for `m <= 288`), storing each C micro-tile twice. Switching to a single
*C-stored-once* panel — `KC = min(K, 2048)` in `_l2_resident_kc`, the same TileK
`linalg` uses — lifts the band to parity/WIN. Measured in isolation (interleaved
A/B vs linalg, 8 epochs, peak-of-12):

| Shape | before | after |
|---|---|---|
| up-m256 (256×11008×2048) | 0.964 LOSE | 1.001 tie |
| up-m512 (512×11008×2048) | 0.970 LOSE | 0.997 tie |
| odd-N (512×11007×2048) | 0.964 LOSE | 0.996 tie |
| 512×4096×4096 | 0.986 tie | **1.020 WIN** |

Bit-identical (`test_dispatch` max_err 0.0); the 2 MB-L2 machine is unchanged
(its KC was already 2048); the headline prefill (M=96, KC=256 band) is untouched.
See `DESIGN.md` → *`_l2_resident_kc`*.

### Wider corner/edge sweep (single run — noise-prone, directional only)

`mojo -I . bench/sweep.mojo --iterate`. A single run swings ±5–10% at M ≥ 128, so read
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

## OpenBLAS detail (`bench/sota.py`)

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
python bench/sota.py        # OpenBLAS (NumPy/SciPy)
mojo -I . bench/linalg_baseline.mojo      # linalg standalone
mojo -I . bench/focus.mojo       # agentic vs linalg, 2σ verdict (the trustworthy one)
mojo -I . bench/sweep.mojo --iterate   # wider corner/edge sweep
```
