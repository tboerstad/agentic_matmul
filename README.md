# matmul

High-performance matrix multiplication kernels in Mojo, benchmarked against SOTA libraries (OpenBLAS, Intel MKL) on Qwen 2.5 VL 3B MLP shapes.

> **YOU MUST VERIFY WHAT HARDWARE YOU ARE RUNNING ON BEFORE COMPARING AGAINST THESE RESULTS.**
> Performance numbers are highly hardware-dependent (CPU microarchitecture, cache sizes, clock speed, core count). All benchmark results in this repo were measured on specific machines — see each document for exact hardware specs.

## Setup

```bash
bash setup.sh
source .venv/bin/activate
```

## Implementations

The `gemm.mojo` file contains 10 matmul implementations, from naive to near-SOTA:

| Implementation | Key technique |
|---|---|
| `matmul_naive` | Triple-nested loop |
| `matmul_tiled` | Cache-blocked tiles |
| `matmul_simd` | Tiled + SIMD vectorization |
| `matmul_parallel` | Tiled + SIMD + multi-threaded |
| `matmul_register_blocked` | Register blocking (MR=4) |
| `matmul_packed` | B-matrix packing |
| `matmul_comptime` | Compile-time optimized tiles |
| `matmul_goto` | GOTO-style GEMM with B-panel packing + GEMV path |
| `matmul_prefill` | A+B packing, worker-based parallelism (MR=8, NR=24) |
| `matmul_adaptive` | **Hardware-adaptive** — auto-tunes KC based on detected L2 cache size |

### Hardware-adaptive matmul

`matmul_adaptive` reads `/sys/devices/system/cpu/cpu0/cache/index2/size` at runtime to detect the per-core L2 cache size and picks the KC (k-tile) parameter accordingly. This means the same binary performs well on different hardware without recompilation.

Helper functions `detect_l2_cache_kb()`, `detect_cpu_model()`, and `print_hw_info()` are exported from `gemm.mojo` for use in benchmarks.

## Benchmarks

All benchmarks print detected hardware info at startup. **Verify this matches your system before comparing results.**

```bash
# All implementations on decode + prefill shapes
mojo run bench_matmul.mojo

# goto vs prefill vs adaptive (prefill shape only)
mojo run bench_prefill.mojo

# SOTA libraries (NumPy/OpenBLAS, SciPy, Intel MKL)
python bench_sota.py
```

### Benchmark shapes (Qwen 2.5 VL 3B MLP)

| Shape | M | N | K |
|---|---|---|---|
| Decode (single token) | 1 | 11008 | 2048 |
| Prefill (batch) | 96 | 11008 | 2048 |

## Results

See detailed results with hardware specs in:

- **SOTA_RESULTS.md** — Full comparison of all implementations vs NumPy/OpenBLAS/MKL
- **OPENBLAS_ANALYSIS.md** — Deep-dive into why OpenBLAS is fast and where Mojo's gap is

> **IMPORTANT:** Each results document specifies the exact hardware it was measured on. Do NOT compare numbers across documents measured on different hardware.

## Testing

```bash
mojo run test_gemm.mojo
```
