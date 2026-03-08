# OpenBLAS vs Mojo DGEMM: Prefill Shape Analysis

> **Hardware-specific analysis.** The benchmark numbers and performance gaps described here are
> specific to the machine listed below. Results will differ on other CPUs — different SIMD widths
> (AVX2 vs AVX-512), core counts, cache sizes, and clock speeds all change absolute GFLOPS and
> relative rankings. Always verify your hardware first (see [AGENTS.md](AGENTS.md)) and re-run
> benchmarks before drawing conclusions.

**Date:** 2026-03-08
**Hardware:** Intel Xeon @ 2.80 GHz, 4 cores, AVX-512 (Skylake), KVM virtualized
**Shape:** M=96, N=11008, K=2048 (float64)

## Latest Benchmark Results

| Implementation | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) |
|---|---|---|---|---|
| OpenBLAS (multi-thread via NumPy) | 23.825 | 22.084 | 181.68 | **196.00** |
| OpenBLAS (1 thread via NumPy) | 23.457 | 22.497 | 184.53 | 192.40 |
| **Mojo prefill** | 35.272 | 34.524 | 122.72 | **125.38** |
| Mojo goto | 53.175 | 52.217 | 81.40 | 82.89 |
| SciPy dgemm (OpenBLAS64) | 34.582 | 32.864 | 125.17 | 131.71 |

**Theoretical peak:** 32 FLOPs/cycle × 2.8 GHz = 89.6 GFLOPS/core, 358.4 GFLOPS/4 cores
(with turbo: actual observed ~196 GFLOPS suggests effective ~2.45 GHz sustained across cores)

**Gap: Mojo prefill is ~36% slower than OpenBLAS (125.4 vs 196.0 GFLOPS peak)**

> Note: OpenBLAS single-thread shows similar performance to multi-thread here because M=96
> is small enough that parallelization overhead limits scaling. OpenBLAS's "1 thread" run
> may still use internal threading via its level3 threading layer.

## OpenBLAS DGEMM Architecture (SkylakeX/Sapphire Rapids)

### Kernel: `dgemm_kernel_16x2_skylakex.c`

OpenBLAS uses a **16×12 microkernel** (UNROLL_M=16, UNROLL_N=2, processing 12 columns at once):

```
Accumulator registers (24 zmm):
  zmm8-zmm9   = C[0:16, col0:col1]    (n-pair 1)
  zmm10-zmm11 = C[0:16, col2:col3]    (n-pair 2)
  zmm12-zmm13 = C[0:16, col4:col5]    (n-pair 3)
  zmm14-zmm15 = C[0:16, col6:col7]    (n-pair 4)  (only for n>6)
  ...
  zmm28-zmm31 = C[0:16, col10:col11]  (n-pair 6)

Scratch registers:
  zmm0 = alpha
  zmm1-zmm4 = A loads (interleaved even/odd via vmovddup)
  zmm5 = B broadcast (vbroadcastf32x4 for 2 values)

Total: 30/32 zmm registers used
```

### Key inner loop technique: vmovddup + vbroadcastf32x4

Instead of broadcasting individual B scalars, OpenBLAS uses a clever interleaving trick:
1. Load A with `vmovddup` (duplicate even/odd doubles across lanes)
2. Broadcast B pair with `vbroadcastf32x4` (128-bit → 512-bit)
3. This gives 4 FMAs per B-pair load, computing 2 columns simultaneously

Per k step: **24 FMA + 4 A loads + 6 B loads + prefetches = ~34 instructions**
FMA cycles: 24 / 2 = 12 → **384 FLOPs / 12 cycles = 32 FLOPs/cycle** ✓

### Level3 Driver: 3-level cache-oblivious tiling

OpenBLAS `level3.c` tiles the full problem into **MC × NC × KC** blocks:
- **KC** (k-tile): sized so A-panel (MC×KC) fits in L2 cache
- **MC** (m-tile): equals problem M when small (96 here), or ~512 for large M
- **NC** (n-tile): large chunk of N processed per thread, B-panel in L3
- **A is packed once per (MC, KC) pair**, reused across ALL NC columns
- **B is packed once per (NC, KC) pair**, reused across ALL MC rows

## Mojo Prefill Kernel Architecture

### Kernel: `_prefill_gemm` in gemm.mojo

Uses an **8×24 microkernel** (MR=8, NR=24):

```
Accumulator registers (24 zmm):
  zmm0-zmm2   = C[row0, 0:24]   (3 vectors of 8 doubles)
  zmm3-zmm5   = C[row1, 0:24]
  ...
  zmm21-zmm23 = C[row7, 0:24]

Scratch registers:
  zmm24 = A broadcast for current row
  zmm25-zmm27 = B vectors (cols 0-7, 8-15, 16-23)
  zmm28-zmm31 = additional A broadcasts

Total: 32/32 zmm registers used (fully packed)
```

Per k step: **24 FMA + 8 broadcast + 3 load + 1 prefetch = 36 instructions**
FMA cycles: 24 / 2 = 12 → **384 FLOPs / 12 cycles = 32 FLOPs/cycle** ✓

### Generated assembly inner loop (verified via objdump)

```asm
; Per k step (KU=8 unrolled, so this repeats 8× per iteration):
prefetcht0  B_panel[pk+4]
vbroadcastsd A_packed[pk, row0] -> zmm16     ; A scalar for row 0
vmovupd      B_packed[pk, 0:8]  -> zmm17     ; B vector cols 0-7
vmovupd      B_packed[pk, 8:16] -> zmm18     ; B vector cols 8-15
vfmadd231pd  zmm17, zmm16, zmm15             ; C[row0, 0:8]  += A[0] * B[0:8]
vbroadcastsd A_packed[pk, row1] -> zmm19     ; A scalar for row 1
vfmadd231pd  zmm17, zmm19, zmm13             ; C[row1, 0:8]  += A[1] * B[0:8]
vbroadcastsd A_packed[pk, row2] -> zmm20
vfmadd231pd  zmm17, zmm20, zmm11             ; C[row2, 0:8]  += A[2] * B[0:8]
... (rows 3-7 similarly)
vfmadd231pd  zmm17, zmm25, zmm1              ; C[row7, 0:8]  += A[7] * B[0:8]
; Then repeat for zmm18 (cols 8-15):
vfmadd231pd  zmm16, zmm18, zmm14             ; C[row0, 8:16] += A[0] * B[8:16]
vfmadd231pd  zmm19, zmm18, zmm12             ; C[row1, 8:16] += A[1] * B[8:16]
... (rows 2-7)
vfmadd231pd  zmm18, zmm25, zmm0              ; C[row7, 8:16] += A[7] * B[8:16]
```

The assembly is clean — the compiler correctly interleaves broadcasts with FMAs and reuses B loads across all rows.

## Why Mojo is ~36% Slower: Root Causes

### 1. A-loading overhead: 8 broadcasts vs 4 loads (MEDIUM impact)

Mojo broadcasts each A element individually:
- **8 `vbroadcastsd` per k step** (one per row)
- Each goes through load ports 2,3

OpenBLAS uses `vmovddup` to load 2 adjacent doubles and duplicate:
- **4 loads per k step** for 16 rows (using interleaved even/odd trick)
- 50% fewer A-load port pressure

This doesn't bottleneck throughput (FMA is the limiter at 8 cycles), but it:
- Increases instruction count, increasing frontend pressure
- Reduces instruction-level parallelism headroom

### 2. Micro-kernel shape: 8×16 vs 16×12 (MEDIUM impact)

| | Mojo 8×24 | OpenBLAS 16×12 |
|---|---|---|
| FMAs per k step | 24 | 24 |
| Cycles per k step | 12 | 12 |
| Accumulator regs | 24 | 24 |
| **FLOPs per iteration** | **384** | **384** |
| M coverage per tile | 8 | 16 |

Both kernels now achieve the same FLOPs per inner loop iteration (384), but OpenBLAS
processes 2× more M-rows per micro-tile (16 vs 8), meaning:
- **Fewer M-tiles** for a given problem (6 tiles for M=96 vs 12)
- **Better A-reuse** — each A element serves 12 N-columns vs 24, but loaded once for 16 rows
- The larger MR gives OpenBLAS a structural advantage in A-packing efficiency

### 3. B-packing strategy (LOW impact for this shape)

Mojo packs B per j-tile (72 cols × KC rows per tile). For N=11008:
- ~153 tiles, each packs 72 × KC bytes
- With 4 workers processing ~38 tiles each, B is packed ~153 times total

OpenBLAS packs B in larger NC-chunks through the level3 driver.
For this specific small shape (M=96), this difference is small because
B-packing cost is dominated by memory bandwidth, not count.

### 4. Thread scheduling (LOW-MEDIUM impact)

OpenBLAS's level3 driver uses a sophisticated threading model:
- Work decomposition adapts to problem shape
- Thread-local buffers are pre-allocated and reused
- Barrier synchronization between k-tiles minimizes false sharing

Mojo uses `parallelize` which creates tasks per j-tile:
- More granular work items (172 j-tiles vs 4 thread chunks)
- Higher scheduling overhead per unit of work
- `matmul_prefill` improves this by chunking j-tiles per worker

### 5. Missing optimization: C-prefetching in kernel (LOW impact)

OpenBLAS's `COMPUTE_m16n12` macro includes explicit C-panel prefetches
during the k-loop (`prefetcht1 (%3)`) to warm up the next C-write
destination. Mojo only prefetches B-panels.

## Summary: Where the 36% Goes

| Factor | Estimated Impact |
|---|---|
| Micro-kernel shape (8×24 vs 16×12, less work per overhead) | ~10-15% |
| A-load overhead (8 broadcasts vs 4 loads) | ~5-8% |
| B-packing & tiling granularity | ~5-8% |
| Thread scheduling overhead | ~3-5% |
| C-prefetching & cache management | ~2-3% |

## Recommendations for Closing the Gap

1. **Increase MR to 16** — Match OpenBLAS's 16×12 shape. This doubles A-reuse per B load
   and puts more work per loop iteration. Requires careful register allocation
   (24 accumulators + scratch = 30 registers, very tight for zmm).

2. **Use vmovddup trick** — Instead of 8 separate `vbroadcastsd` for A, load pairs of
   adjacent A elements with `vmovddup` and use `vshufpd` to route them. This halves
   A-loading pressure but requires A to be packed in pairs (which it already is).

3. **Add C-panel prefetching** — Add `prefetcht1` hints for the next C-store destination
   during the k-loop to reduce writeback latency.

4. **Consider MR=16, NR=12** — OpenBLAS's choice of MR=16, NR=12 processes 2× more M-rows
   per micro-tile. This halves the number of M-tiles (6 vs 12 for M=96) and improves
   A-loading efficiency via the vmovddup trick. The current 8×24 kernel matches FLOPs
   per iteration but needs more M-tiles to cover the same problem.
