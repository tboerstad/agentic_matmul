"""Optimized CPU matmul kernels in Mojo.

Public API: `matmul_dispatch` (shape-routed C = A * B) and the `Matrix` it
operates on. Everything else is internal layering:

  * `microkernel.mojo`  the register-tile micro-kernel and operand loaders
  * `packed.mojo`       the packed GEMM drivers (prefill workhorse, 2D grid)
  * `gemv.mojo`         the M = 1 decode GEMV
  * `nopack.mojo`       serial and M-parallel kernels that skip packing
  * `amx.mojo`          the Intel AMX bf16 tile kernel
  * `dispatch.mojo`     shape gates, cache heuristics, per-regime tile picks
  * `matrix.mojo` / `tile.mojo`  the data types the kernels speak in
  * `cpu_cache.mojo`    cpuid cache-size detection (memoized)
  * `sol.mojo`          speed-of-light self-measurement for the benchmarks
"""

from matmul.dispatch import matmul_dispatch
from matmul.matrix import Matrix
from matmul.tile import Tile
