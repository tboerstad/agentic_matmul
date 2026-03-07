# Performance Gap Analysis: Mojo GEMM vs SOTA

## Current State

| Shape | Mojo Packed | SOTA | Gap |
|-------|-------------|------|-----|
| Decode (1×11008×2048) | 0.70 GFLOPS | 11.37 GFLOPS (MKL) | 16× |
| Prefill (96×11008×2048) | 35.62 GFLOPS | 186.86 GFLOPS (OpenBLAS) | 5.2× |

Theoretical 4-core peak: 179.2 GFLOPS (float64, AVX-512 FMA).

## #1 Missing Optimization: 2D Register-Tile Micro-Kernel (MR × NR)

This is the single biggest win available and the foundation of every high-perf GEMM
(GotoBLAS/OpenBLAS/MKL).

### The Problem

The current micro-kernel uses MR=4 rows × NR=1 SIMD vector (8 doubles). Per inner-loop
iteration of the k-loop:

- **Loads:** 1 B vector (8 doubles) + 4 A scalars = 5 loads
- **FMAs:** 4
- **Compute intensity:** 0.8 FMAs/load

The CPU has 2 FMA units that can each execute 1 FMA/cycle, but can only do ~2 loads/cycle.
With 0.8 FMAs/load, the FMA units are idle most of the time waiting for data.

### The Fix

Use a 2D accumulator tile: MR rows × NR SIMD-vectors wide. Each B vector loaded is reused
across MR rows, and each A scalar broadcast is reused across NR column-vectors.

**Example: MR=6, NR=2 (12 accumulators, 20 registers total)**

```
Loads: 2 B vectors + 6 A scalars = 8 loads
FMAs:  6 × 2 = 12
Compute intensity: 1.5 FMAs/load  (1.9× better)
```

**Example: MR=6, NR=3 (18 accumulators, 27 registers total)**

```
Loads: 3 B vectors + 6 A scalars = 9 loads
FMAs:  6 × 3 = 18
Compute intensity: 2.0 FMAs/load  (2.5× better)
```

AVX-512 has 32 zmm registers, so MR=6 NR=3 fits comfortably (18 acc + 3 B + 6 A = 27).

### Expected Impact

The prefill shape should go from ~35 GFLOPS to ~70-90 GFLOPS just from this change
(2-2.5× improvement from better compute intensity alone).

## #2 Missing: Hierarchical Cache Blocking (MC/KC/NC)

Current code uses a single `TILE=32` for all three loop dimensions. BLAS libraries use
separate blocking parameters tuned to the cache hierarchy:

- **NC** (~4096): N-dimension block sized for L3 cache
- **KC** (~256-512): K-dimension block sized for L2 cache
- **MC** (~72-144): M-dimension block sized for L1 cache

The micro-kernel (MR × NR) sits inside these nested loops. Proper sizing ensures each
level of cache is fully utilized.

## #3 Missing: B-Panel Packing (requires NR > 1 first)

B-panel packing was attempted and regressed because:
1. The micro-kernel only uses NR=1, so the packing cost isn't amortized
2. M=96 only gives 3 i-tiles, not enough reuse of packed B

With a proper MR×NR micro-kernel and hierarchical blocking, B-panel packing becomes
beneficial because packed B is reused across all MC/MR row-blocks.

## #4 Missing: Software Prefetching

Pre-loading the next k-iteration's B-panel data into cache while the current FMAs execute.
Hides memory latency. Worth ~10-20% once the micro-kernel is compute-bound.

## #5 Missing: Decode-Specific GEMV Path

M=1 is fundamentally a matrix-vector product, not a matrix-matrix product. It needs a
completely different strategy: stream B row-by-row through cache, accumulate with vector
reductions. The GEMM micro-kernel structure is wrong for this shape.

## Priority Order

1. **2D micro-kernel (MR×NR)** — Expected 2-2.5× improvement (biggest single win)
2. **Hierarchical blocking** — Expected 1.3-1.5× on top of #1
3. **B-panel packing** — Expected 1.2-1.3× (only works after #1 and #2)
4. **Prefetching** — Expected 1.1-1.2×
5. **GEMV path for decode** — Separate optimization for M=1

Combined, these should close most of the 5× gap on prefill.
