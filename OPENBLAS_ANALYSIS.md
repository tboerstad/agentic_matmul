# OpenBLAS vs Mojo DGEMM: Prefill Shape Analysis

> **YOU MUST VERIFY WHAT HARDWARE YOU ARE RUNNING ON BEFORE COMPARING AGAINST THESE RESULTS.**
> The numbers below were measured on specific hardware. If your CPU, cache sizes, or core count differ, your results WILL be different. Run `mojo run bench_prefill.mojo` to see your detected hardware.

**Date:** 2026-03-08
**Hardware:** Intel Xeon (Sapphire Rapids) @ 2.10 GHz, 4 cores, AVX-512, 2 MB L2/core
**Shape:** M=96, N=11008, K=2048 (float64)

> **Note:** These results are from a DIFFERENT machine than SOTA_RESULTS.md (which used Skylake @ 2.80 GHz, 1 MB L2/core). Do NOT directly compare absolute numbers between the two documents.

## Benchmark Results (Sapphire Rapids @ 2.10 GHz, 4 cores, 2 MB L2/core)

| Implementation | Mean (ms) | Min (ms) | GFLOPS (mean) | GFLOPS (peak) |
|---|---|---|---|---|
| OpenBLAS (1 thread via NumPy) | 22.09 | 20.50 | 196.0 | **211.2** |
| OpenBLAS (multi-thread via NumPy) | 22.82 | 21.68 | 189.7 | 199.7 |
| **Mojo prefill** | 23.09 | 22.33 | 187.5 | **193.8** |
| Mojo goto | 27.58 | 25.55 | 156.9 | 169.4 |
| SciPy dgemm (OpenBLAS64) | 31.33 | 30.00 | 138.2 | 144.3 |

**Theoretical peak:** 32 FLOPs/cycle × 2.1 GHz = 67.2 GFLOPS/core, 268.8 GFLOPS/4 cores

**Gap: Mojo prefill is ~8% slower than OpenBLAS (193.8 vs 211.2 GFLOPS peak)**

> Note: OpenBLAS single-thread is faster than multi-thread here because M=96 is small
> enough that parallelization overhead hurts more than it helps. OpenBLAS's "1 thread"
> run is likely still using all cores internally via its level3 threading layer.

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

Uses an **8×16 microkernel** (MR=8, NR=16):

```
Accumulator registers (16 zmm):
  zmm0-zmm1   = C[row0, 0:16]
  zmm2-zmm3   = C[row1, 0:16]
  ...
  zmm14-zmm15 = C[row7, 0:16]

Scratch registers:
  zmm16 = A broadcast for row 0
  zmm17 = B vector (first 8 cols)
  zmm18 = B vector (next 8 cols)
  zmm19-zmm25 = A broadcasts for rows 1-7

Total: 26/32 zmm registers used
```

Per k step: **16 FMA + 8 broadcast + 2 load + 1 prefetch = 27 instructions**
FMA cycles: 16 / 2 = 8 → **256 FLOPs / 8 cycles = 32 FLOPs/cycle** ✓

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

## Why Mojo is ~8% Slower: Root Causes

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

| | Mojo 8×16 | OpenBLAS 16×12 |
|---|---|---|
| FMAs per k step | 16 | 24 |
| Cycles per k step | 8 | 12 |
| Accumulator regs | 16 | 24 |
| **FLOPs per iteration** | **256** | **384** |
| Work per branch/overhead | Less | More |

OpenBLAS processes 50% more FLOPs per inner loop iteration, meaning:
- **Less loop overhead** relative to useful work
- **Better instruction-level parallelism** (24 independent FMA chains vs 16)
- The out-of-order engine can overlap more work

With KU=8 unrolling, Mojo's loop does 128 FMAs per iteration vs OpenBLAS's ~192,
but the key issue is the ratio of overhead (pointer arithmetic, loop control, prefetches)
to useful work.

### 3. B-packing strategy (LOW impact for this shape)

Mojo packs B per j-tile (64 cols × KC rows per tile). For N=11008:
- 172 tiles, each packs 64 × KC bytes
- With 4 workers processing ~43 tiles each, B is packed 172 times total

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

## Summary: Where the 8% Goes

| Factor | Estimated Impact |
|---|---|
| A-load overhead (8 broadcasts vs 4 loads) | ~2-3% |
| Smaller micro-tile (less work per overhead) | ~2-3% |
| B-packing & tiling granularity | ~1-2% |
| Thread scheduling overhead | ~1-2% |
| C-prefetching | <1% |

## Recommendations for Closing the Gap

1. **Increase MR to 16** — Match OpenBLAS's 16×12 shape. This doubles A-reuse per B load
   and puts more work per loop iteration. Requires careful register allocation
   (24 accumulators + scratch = 30 registers, very tight for zmm).

2. **Use vmovddup trick** — Instead of 8 separate `vbroadcastsd` for A, load pairs of
   adjacent A elements with `vmovddup` and use `vshufpd` to route them. This halves
   A-loading pressure but requires A to be packed in pairs (which it already is).

3. **Add C-panel prefetching** — Add `prefetcht1` hints for the next C-store destination
   during the k-loop to reduce writeback latency.

4. **Consider NR=12** — OpenBLAS's choice of NR=12 (6 pairs of 2) is specifically tuned
   to maximize register usage: 6×4=24 accumulator zmm + 5 scratch = 29 total.
   NR=16 with MR=8 only uses 16 accumulators, wasting register budget.
