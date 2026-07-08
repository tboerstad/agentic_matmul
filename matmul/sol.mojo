"""Empirical speed-of-light (SOL) self-measurement, in Mojo.

The kernel benchmarks in this repo have long been judged only against the
stdlib linalg ratio, and that denominator keeps moving (it drifts with the
Mojo nightly and with the host) while pointing agents at the wrong work: the
big squares sit at ~0.96 vs linalg yet are already at ~97% of the machine's
FMA peak, and prefill looks like a comfortable win vs linalg while hiding the
suite's largest gap to the roofline. See docs/SOL.md.

This module measures the machine's own ceilings in the same process as the
kernel run, so `bench/focus.mojo` can print each shape as a % of its roofline SOL.
It is a Mojo port of `sol/sol_fma.c` and `sol/sol_bw.c`, kept in-tree so the
number transfers across machines and nightlies with no external toolchain.

    from matmul.sol import fma_peak_gflops, l3_read_bw_gbs, dram_read_bw_gbs

    var peak = fma_peak_gflops[DType.float64]()  # all-core FMA peak, GFLOPS
    var l3 = l3_read_bw_gbs()                     # all-core L3 read BW, GB/s
    var dram = dram_read_bw_gbs()                 # all-core DRAM read BW, GB/s

Every number is hardware- and boot-specific; re-measure in the same boot as
any kernel benchmark you want to state as a % of SOL.
"""

from matmul.amx import (
    amx_bf16_usable,
    _amx_configure,
    _amx_release,
    _tile_zero,
    _tile_load,
    _tile_dpbf16ps,
    _tile_store,
)
from matmul.cpu_cache import l3_cache_size
from std.algorithm.functional import parallelize
from std.collections import InlineArray
from std.memory.unsafe_pointer import alloc
from std.sys import CompilationTarget, num_physical_cores, simd_width_of
from std.time import perf_counter_ns


# 16 independent FMA chains hide the FMA latency (latency ~4-5 cycles, 2 issue
# ports, so >=10 chains are needed to saturate both pipes). Matches sol_fma.c.
comptime _CHAINS = 16


def fma_peak_gflops[dtype: DType](iters: Int = 20_000_000) -> Float64:
    """All-core AVX-512 FMA peak for `dtype`, in GFLOPS (the sum over cores).

    Runs `_CHAINS` register-resident SIMD accumulator chains per core, each an
    `acc = a * acc + b` recurrence the compiler cannot close (the trip count is
    a runtime value), so the loop executes and the FMA pipes stay full. The
    reduced accumulators escape through `sinks` so the work is not dead-code
    eliminated. A single warm chain runs first to reach steady turbo, as the C
    benchmark does. Pass the dtype the kernels actually FMA in: the bf16
    kernels compute in f32 (microkernel.compute_dtype, docs/SOL.md idea 2), so their
    ceiling is the f32 peak; a raw bf16 chain would measure LLVM's
    element-wise emulation instead, which nothing in the repo runs anymore."""
    comptime W = simd_width_of[dtype]()
    var nw = num_physical_cores()
    var gfs = alloc[Float64](nw)
    var sinks = alloc[Scalar[dtype]](nw)

    # Runtime-sourced multipliers just above and below 1 so the recurrence stays
    # bounded across 20M steps (matches sol_fma.c's 1.0000001 / 0.9999999).
    var a0 = Scalar[dtype](1.0) + Scalar[dtype](0.0000001)
    var b0 = Scalar[dtype](1.0) - Scalar[dtype](0.0000001)

    def chain(wid: Int) {read iters, read a0, read b0, mut gfs, mut sinks}:
        var a = SIMD[dtype, W](a0)
        var b = SIMD[dtype, W](b0)
        var acc = InlineArray[SIMD[dtype, W], _CHAINS](uninitialized=True)
        comptime for i in range(_CHAINS):
            acc[i] = SIMD[dtype, W](b0) * Scalar[dtype](i + 1)
        var t0 = perf_counter_ns()
        for _ in range(iters):
            comptime for i in range(_CHAINS):
                acc[i] = a.fma(acc[i], b)
        var t1 = perf_counter_ns()
        var s = SIMD[dtype, W](0)
        comptime for i in range(_CHAINS):
            s += acc[i]
        sinks[wid] = s.reduce_add()
        var flops = Float64(iters) * Float64(_CHAINS) * Float64(W) * 2.0
        # bytes/flops per ns are GB/s / GFLOPS: (t1 - t0) is in ns.
        gfs[wid] = flops / Float64(t1 - t0)

    chain(0)  # warm to steady turbo before the timed all-core run
    parallelize(chain, nw, nw)

    var total = Float64(0)
    var guard = Scalar[dtype](0)
    for i in range(nw):
        total += gfs[i]
        guard += sinks[i]
    gfs.free()
    sinks.free()
    # NaN never happens here; the compare just keeps `guard` (and the chains
    # feeding it) live so the timed loop is not optimized away.
    if guard != guard:
        return -1.0
    return total


def amx_bf16_peak_gflops(iters: Int = 250_000) -> Float64:
    """All-core AMX bf16 tile-FMA peak in GFLOPS, or 0 when the AMX kernel is
    not usable on this machine (so callers can fall back to the AVX-512
    peak). Each core runs four independent tdpbf16ps accumulator chains
    (tmm0-3, fed from preloaded tmm4-7): four chains at ~16-cycle throughput
    cover the instruction's accumulation latency, the same reasoning as the
    16 FMA chains above. One tile op is 16x16x32 MACs = 16384 flops. The
    source tiles hold small values so 250k accumulations stay far from
    overflow, and a stored tile feeds the sink so the loop stays live."""
    comptime if not CompilationTarget.has_intel_amx():
        return 0.0
    if not amx_bf16_usable():
        return 0.0

    var nw = num_physical_cores()
    var gfs = alloc[Float64](nw)
    var sinks = alloc[Float64](nw)
    var src = alloc[BFloat16](16 * 32)
    for i in range(16 * 32):
        src[i] = BFloat16(0.001)
    var scratch = alloc[Float32](nw * 256)

    def worker(wid: Int) {read iters, mut gfs, mut sinks, read src, mut scratch}:
        _amx_configure()
        _tile_zero[0]()
        _tile_zero[1]()
        _tile_zero[2]()
        _tile_zero[3]()
        _tile_load[4](src, 64)
        _tile_load[5](src, 64)
        _tile_load[6](src, 64)
        _tile_load[7](src, 64)
        var t0 = perf_counter_ns()
        for _ in range(iters):
            _tile_dpbf16ps[0, 4, 6]()
            _tile_dpbf16ps[1, 4, 7]()
            _tile_dpbf16ps[2, 5, 6]()
            _tile_dpbf16ps[3, 5, 7]()
        var t1 = perf_counter_ns()
        var sc = scratch + wid * 256
        _tile_store[0](sc, 64)
        _amx_release()
        var s = Float64(0)
        for i in range(256):
            s += Float64(sc[i])
        sinks[wid] = s
        var flops = Float64(iters) * 4.0 * 16.0 * 16.0 * 32.0 * 2.0
        gfs[wid] = flops / Float64(t1 - t0)

    worker(0)  # warm to steady turbo (and take the first-use xstate trap)
    parallelize(worker, nw, nw)

    var total = Float64(0)
    var guard = Float64(0)
    for i in range(nw):
        total += gfs[i]
        guard += sinks[i]
    gfs.free()
    sinks.free()
    src.free()
    scratch.free()
    if guard != guard:
        return -1.0
    return total


def _read_sweep(buf: UnsafePointer[Float64, _], n: Int, reps: Int) -> Float64:
    """Four AVX-512 accumulator chains reading `n` doubles `reps` times; returns
    the reduced sum so the loads escape. Matches sol_bw.c's read loop."""
    var a0 = SIMD[DType.float64, 8](0)
    var a1 = SIMD[DType.float64, 8](0)
    var a2 = SIMD[DType.float64, 8](0)
    var a3 = SIMD[DType.float64, 8](0)
    for _ in range(reps):
        var i = 0
        while i + 32 <= n:
            a0 += buf.load[width=8](i)
            a1 += buf.load[width=8](i + 8)
            a2 += buf.load[width=8](i + 16)
            a3 += buf.load[width=8](i + 24)
            i += 32
    return ((a0 + a1) + (a2 + a3)).reduce_add()


def read_bw_gbs(bytes_per_thread: Int, reps: Int) -> Float64:
    """All-core read bandwidth in GB/s: one private buffer per core, swept
    `reps` times after a warm pass. The footprint that decides which cache level
    this measures is `bytes_per_thread` (per core) times the core count."""
    var nw = num_physical_cores()
    var n = bytes_per_thread // 8
    if n < 32:
        n = 32
    var bufs = alloc[Float64](nw * n)
    for i in range(nw * n):
        bufs[i] = Float64(i & 1023)
    var gfs = alloc[Float64](nw)
    var sinks = alloc[Float64](nw)

    def worker(wid: Int) {read n, read reps, mut bufs, mut gfs, mut sinks}:
        var buf = bufs + wid * n
        sinks[wid] = _read_sweep(buf, n, 2)  # warm
        var t0 = perf_counter_ns()
        var s = _read_sweep(buf, n, reps)
        var t1 = perf_counter_ns()
        sinks[wid] += s
        var bytes = Float64(n * 8) * Float64(reps)
        # bytes per ns is GB/s.
        gfs[wid] = bytes / Float64(t1 - t0)

    parallelize(worker, nw, nw)

    var total = Float64(0)
    var guard = Float64(0)
    for i in range(nw):
        total += gfs[i]
        guard += sinks[i]
    bufs.free()
    gfs.free()
    sinks.free()
    if guard != guard:
        return -1.0
    return total


def l3_read_bw_gbs() -> Float64:
    """All-core L3 read bandwidth, GB/s. Sizes the aggregate footprint to ~3/4
    of the detected L3 so it stays resident across all cores; falls back to
    24 MB total when L3 is undetectable."""
    var l3 = l3_cache_size()
    if l3 == 0:
        l3 = 24 << 20
    var per = (l3 * 3 // 4) // num_physical_cores()
    per = (per // 256) * 256  # whole 32-double (256-byte) sweep steps
    if per < 256:
        per = 256
    return read_bw_gbs(per, 300)


def dram_read_bw_gbs() -> Float64:
    """All-core DRAM read bandwidth, GB/s. 256 MB per core overflows any L3 on
    a 4+ core part, so every read misses to memory."""
    return read_bw_gbs(256 << 20, 4)


struct MachineSol(Copyable, Movable):
    """A one-shot snapshot of this machine's measured ceilings, for the roofline
    the benchmark reports each shape against."""

    var fma_peak: Float64  # all-core FMA peak for the measured dtype, GFLOPS
    var amx_peak: Float64  # all-core AMX bf16 tile peak, GFLOPS (0 = no AMX)
    var l3_bw: Float64  # all-core L3 read bandwidth, GB/s
    var dram_bw: Float64  # all-core DRAM read bandwidth, GB/s
    var l3_bytes: Int  # detected L3 size, bytes (0 if undetectable)
    var cores: Int

    def __init__(
        out self,
        fma_peak: Float64,
        l3_bw: Float64,
        dram_bw: Float64,
        l3_bytes: Int,
        cores: Int,
        amx_peak: Float64 = 0.0,
    ):
        self.fma_peak = fma_peak
        self.amx_peak = amx_peak
        self.l3_bw = l3_bw
        self.dram_bw = dram_bw
        self.l3_bytes = l3_bytes
        self.cores = cores

    def _compute_peak(self, use_amx: Bool) -> Float64:
        """The compute ceiling for one shape: the tile-unit peak when the
        shape runs on the AMX kernel, else the AVX-512 FMA peak."""
        if use_amx and self.amx_peak > 0:
            return self.amx_peak
        return self.fma_peak

    def roofline(
        self, m: Int, n: Int, k: Int, elem_bytes: Int, use_amx: Bool = False
    ) -> Float64:
        """The speed-of-light GFLOPS for one GEMM shape: min of the compute
        peak and the bandwidth roofline BW * arithmetic-intensity. Arithmetic
        intensity uses the compulsory traffic (read A + read B + write C
        once); the bandwidth is L3 when that working set fits the detected L3,
        else DRAM. This reproduces docs/SOL.md's per-shape rooflines from measured
        ceilings: a big square lands compute-bound at the FMA peak, and M=1
        decode lands bandwidth-bound at 2/elem flops-per-byte times the
        relevant BW. `use_amx` selects the tile-unit compute peak for shapes
        the bf16 dispatch routes to the AMX kernel."""
        var flops = 2.0 * Float64(m) * Float64(n) * Float64(k)
        var bytes = Float64((m * k + k * n + m * n) * elem_bytes)
        var ai = flops / bytes
        var footprint = (m * k + k * n + m * n) * elem_bytes
        var bw = self.dram_bw
        if self.l3_bytes > 0 and footprint <= self.l3_bytes:
            bw = self.l3_bw
        var bw_roof = bw * ai
        return min(self._compute_peak(use_amx), bw_roof)

    def bound(
        self, m: Int, n: Int, k: Int, elem_bytes: Int, use_amx: Bool = False
    ) -> String:
        """"compute" or "bw": which ceiling the shape's roofline hit."""
        var flops = 2.0 * Float64(m) * Float64(n) * Float64(k)
        var bytes = Float64((m * k + k * n + m * n) * elem_bytes)
        var footprint = (m * k + k * n + m * n) * elem_bytes
        var bw = self.dram_bw
        if self.l3_bytes > 0 and footprint <= self.l3_bytes:
            bw = self.l3_bw
        if bw * (flops / bytes) < self._compute_peak(use_amx):
            return "bw"
        return "compute"


def measure_sol[dtype: DType, WITH_AMX_BF16: Bool = False]() -> MachineSol:
    """Measure this machine's ceilings for `dtype` in the current process: the
    all-core FMA peak, L3 and DRAM read bandwidth, and the detected L3 size.
    WITH_AMX_BF16 additionally measures the tdpbf16ps tile peak (bf16 storage
    runs the AMX kernel on eligible shapes, so its compute ceiling is the tile
    units, not the AVX-512 f32 pipes it computes in elsewhere)."""
    var amx = Float64(0)
    comptime if WITH_AMX_BF16:
        amx = amx_bf16_peak_gflops()
    return MachineSol(
        fma_peak=fma_peak_gflops[dtype](),
        l3_bw=l3_read_bw_gbs(),
        dram_bw=dram_read_bw_gbs(),
        l3_bytes=l3_cache_size(),
        cores=num_physical_cores(),
        amx_peak=amx,
    )
