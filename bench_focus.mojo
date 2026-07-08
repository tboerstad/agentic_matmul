# Focused A/B harness: dispatch vs stdlib linalg, judged by mean +/- stdev.
#
# WHY THIS EXISTS: a single dispatch/linalg ratio is not trustworthy. The peak
# (min-time) GFLOPS of one process launch still swings +/-5-10% with the turbo /
# thermal state that launch happened to land in, and we kept reading those
# swings as real kernel wins or losses. sq512 has been seen at 1.15 and at 0.79
# with byte-identical code; up-m512 looked like a 0.95 LOSE in one launch and a
# dead tie across ten. A regression gate keyed off one number flips on noise.
#
# So the DEFAULT here is not one ratio per shape: it runs the whole shape set for
# EPOCHS independent epochs (epoch loop OUTER, shapes INNER, so every shape is
# sampled across the full thermal envelope rather than one shape paying for a
# late-run throttle), each epoch a peak over RUNS interleaved reps, and reports
# the ratio's MEAN, STDEV, MIN and MAX across epochs plus a 2-sigma verdict:
#
#   WIN   mean - 2*stdev > 1.0   (confidently faster than linalg)
#   LOSE  mean + 2*stdev < 1.0   (confidently slower)
#   tie   the 2-sigma band straddles 1.0  (within run-to-run noise; do NOT claim
#         a win or a loss here, no matter what a single run printed)
#
# Judge a kernel edit by whether a shape's verdict or mean moves outside its
# stdev band, never by a single ratio. Still compare the dispatch/linalg RATIO,
# never absolute GFLOPS across separate process launches.
#
# Flags:
#   (no flag)        EPOCHS=10 epochs x RUNS=12 reps  -- the trustworthy default
#   --epochs N       override epoch count (>= 2 for a meaningful stdev)
#   --runs M         override reps-per-epoch (the peak is taken over these)
#   --quick          one epoch, prints bare ratios with no stdev (old behavior;
#                    fast edit-loop sanity check, NOT for judging a change)
#   --dtype T        element type: f64 (default), f32, f16, or bf16. The whole
#                    A/B/C path and both kernels run in T. f16 accumulates in
#                    f16 and bf16 dispatch computes in f32 (storage stays bf16;
#                    see gemm._compute_dtype), so dispatch and linalg diverge
#                    numerically; the ratio still measures throughput, not accuracy.
from amx import amx_shape_ok
from gemm import matmul_dispatch, _compute_dtype
from matrix import Matrix
from sol import MachineSol, measure_sol
from linalg.matmul import matmul as linalg_matmul
from layout import Coord, TileTensor, row_major
from std.collections import List
from std.math import sqrt
from std.sys import argv, size_of
from std.time import perf_counter_ns


def gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


def fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


def round3(x: Float64) -> Float64:
    # Round to 3 decimals for readable output (display only).
    return Float64(Int(x * 1000.0 + 0.5)) / 1000.0


# The % of the machine's roofline SOL that `gflops` reaches for one shape.
# See sol.MachineSol.roofline: the SOL is min(compute peak, bandwidth *
# arithmetic intensity), all self-measured this process. This is the number to
# judge a kernel by; the linalg ratio moves with the nightly and points at the
# wrong work (SOL.md), while a shape's % of SOL says how much of the machine is
# left. A bf16 shape that the dispatch routes to the AMX tile kernel is judged
# against the measured tdpbf16ps peak, not the AVX-512 f32 peak it would
# otherwise compute in (the same amx_shape_ok gate the dispatch uses).
def _uses_amx[dtype: DType](sol: MachineSol, m: Int, n: Int, k: Int) -> Bool:
    return (
        dtype == DType.bfloat16
        and sol.amx_peak > 0.0
        and amx_shape_ok(m, n, k)
    )


def pct_of_sol[
    dtype: DType
](sol: MachineSol, m: Int, n: Int, k: Int, gflops: Float64) -> Float64:
    var roof = sol.roofline(
        m, n, k, size_of[Scalar[dtype]](), _uses_amx[dtype](sol, m, n, k)
    )
    if roof <= 0.0:
        return 0.0
    return 100.0 * gflops / roof


def print_sol_banner[dtype: DType](sol: MachineSol):
    print(
        "SOL (measured this process, dtype", dtype, "):",
        "FMA peak", Int(sol.fma_peak), "GFLOPS |",
        "L3 read", Int(sol.l3_bw), "GB/s |",
        "DRAM read", Int(sol.dram_bw), "GB/s |",
        "L3", sol.l3_bytes // (1 << 20), "MB |",
        sol.cores, "cores",
    )
    if sol.amx_peak > 0.0:
        print(
            "  AMX bf16 tile peak", Int(sol.amx_peak), "GFLOPS:",
            "the compute ceiling for shapes the bf16 dispatch routes to the",
            "tdpbf16ps kernel (m % 32 == k % 32 == 0).",
        )
    print(
        "  (%SOL = dispatch GFLOPS / roofline; roofline = min(FMA peak,"
        " BW x arithmetic-intensity). Judge by %SOL, not the linalg ratio.)"
    )
    print(
        "  (a bw-bound shape reading > 100% means the harness held its operand"
        " cache-hot across reps vs the cold-DRAM roofline: SOL.md idea 5.)"
    )


# One epoch for one shape: peak (min-time) GFLOPS over n_runs interleaved reps,
# returned as (dispatch_gflops, linalg_gflops). Generic over dtype so the same
# harness compares dispatch vs linalg in f64, f32, f16, or bf16.
def measure[
    dtype: DType
](m: Int, n: Int, k: Int, n_runs: Int) raises -> Tuple[Float64, Float64]:
    var a = Matrix[dtype](m, k)
    var b = Matrix[dtype](k, n)
    var c = Matrix[dtype](m, n)
    fill(a, 17)
    fill(b, 13)

    var c_data = List[Scalar[dtype]](capacity=m * n)
    for _ in range(m * n):
        c_data.append(0)
    var a_ptr = a.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()
    var c_ptr = c_data.unsafe_ptr()
    var c_tile = TileTensor(c_ptr, row_major(Coord(m, n)))
    var a_tile = TileTensor(a_ptr, row_major(Coord(m, k)))
    var b_tile = TileTensor(b_ptr, row_major(Coord(k, n)))

    # Warm both before timing.
    matmul_dispatch(c, a, b)
    linalg_matmul[target="cpu"](c_tile, a_tile, b_tile, None)

    var t0: UInt
    var t1: UInt
    var dt: Float64
    var d_min = Float64(1e30)
    var l_min = Float64(1e30)
    for _ in range(n_runs):
        t0 = perf_counter_ns(); matmul_dispatch(c, a, b); t1 = perf_counter_ns()
        dt = Float64(t1 - t0)
        if dt < d_min: d_min = dt
        t0 = perf_counter_ns()
        linalg_matmul[target="cpu"](c_tile, a_tile, b_tile, None)
        t1 = perf_counter_ns()
        dt = Float64(t1 - t0)
        if dt < l_min: l_min = dt

    return (gflops(m, n, k, d_min / 1e9), gflops(m, n, k, l_min / 1e9))


# The shape set: headline Qwen shapes, the heavy compute-bound GEMMs where the
# P-core cap bites, and a few small cache-resident boxes.
def shapes() -> Tuple[List[String], List[Int], List[Int], List[Int]]:
    var labels = [
        String("decode  "), "prefill ",
        "sq512   ", "sq1024  ", "sq2048  ",
        "M512-g  ", "up-m256 ", "up-m512 ", "dn-m512 ",
        "sq128   ", "sq256   ", "sq384   ", "box512  ", "oddN    ",
        "sq300   ", "sq320   ",
    ]
    var ms = [
        1, 96,
        512, 1024, 2048,
        512, 256, 512, 512,
        128, 256, 384, 512, 512,
        300, 320,
    ]
    var ns = [
        11008, 11008,
        512, 1024, 2048,
        4096, 11008, 11008, 2048,
        128, 256, 384, 128, 11007,
        300, 320,
    ]
    var ks = [
        2048, 2048,
        512, 1024, 2048,
        4096, 2048, 2048, 11008,
        128, 256, 384, 512, 2048,
        300, 320,
    ]
    return (labels^, ms^, ns^, ks^)


# Old single-epoch behavior: one peak ratio per shape, no stdev. Fast, but a
# single ratio is noise-prone, so this is a sanity check, not a judgment.
def run_quick[dtype: DType](sol: MachineSol, runs: Int) raises:
    print("=== focused bench --quick (single epoch, peak over", runs, "runs; NOT for judging) | dtype", dtype, "===")
    print_sol_banner[dtype](sol)
    var sh = shapes()
    var labels = sh[0].copy(); var ms = sh[1].copy(); var ns = sh[2].copy(); var ks = sh[3].copy()
    for s in range(len(ms)):
        var r = measure[dtype](ms[s], ns[s], ks[s], runs)
        print(
            "  ", labels[s], "M=", ms[s], "N=", ns[s], "K=", ks[s],
            "| dispatch", Int(r[0]), "| linalg", Int(r[1]),
            "| ratio", round3(r[0] / r[1]),
            "| %SOL", Int(pct_of_sol[dtype](sol, ms[s], ns[s], ks[s], r[0])),
            sol.bound(
                ms[s], ns[s], ks[s], size_of[Scalar[dtype]](),
                _uses_amx[dtype](sol, ms[s], ns[s], ks[s]),
            ),
        )


# The default: EPOCHS independent epochs, report mean +/- stdev + 2-sigma verdict.
def run_stats[dtype: DType](sol: MachineSol, epochs: Int, runs: Int) raises:
    print(
        "=== focused bench:", epochs, "epochs x peak-of-", runs,
        "; ratio mean +/- stdev, 2-sigma verdict | dtype", dtype, "===",
    )
    print("(epoch loop is outer so every shape is sampled across the full thermal envelope)")
    print_sol_banner[dtype](sol)
    var sh = shapes()
    var labels = sh[0].copy(); var ms = sh[1].copy(); var ns = sh[2].copy(); var ks = sh[3].copy()
    var num = len(ms)

    # Per-shape ratio samples, one per epoch.
    var ratios = List[List[Float64]]()
    var dgf_sum = List[Float64]()
    var lgf_sum = List[Float64]()
    for _ in range(num):
        ratios.append(List[Float64]())
        dgf_sum.append(0.0)
        lgf_sum.append(0.0)

    for e in range(epochs):
        for s in range(num):
            var r = measure[dtype](ms[s], ns[s], ks[s], runs)
            ratios[s].append(r[0] / r[1])
            dgf_sum[s] += r[0]
            lgf_sum[s] += r[1]
        print("  epoch", e + 1, "of", epochs, "done")

    print("\n--- summary (mean ratio +/- stdev over", epochs, "epochs) ---")
    for s in range(num):
        var samples = ratios[s].copy()
        var mean = Float64(0.0)
        for v in samples: mean += v
        mean /= Float64(epochs)

        var sd = Float64(0.0)
        if epochs >= 2:
            var ss = Float64(0.0)
            for v in samples:
                var d = v - mean
                ss += d * d
            sd = sqrt(ss / Float64(epochs - 1))  # sample stdev

        var lo = mean - 2.0 * sd
        var hi = mean + 2.0 * sd
        var tag = String("tie ")
        if lo > 1.0: tag = "WIN "
        elif hi < 1.0: tag = "LOSE"

        var rmin = samples[0]
        var rmax = samples[0]
        for v in samples:
            if v < rmin: rmin = v
            if v > rmax: rmax = v

        var d_mean = dgf_sum[s] / Float64(epochs)
        print(
            "  ", labels[s],
            "| dispatch", Int(d_mean),
            "| linalg", Int(lgf_sum[s] / Float64(epochs)),
            "| ratio", round3(mean), "+/-", round3(sd),
            "[", round3(rmin), "..", round3(rmax), "]",
            tag,
            "| %SOL", Int(pct_of_sol[dtype](sol, ms[s], ns[s], ks[s], d_mean)),
            sol.bound(
                ms[s], ns[s], ks[s], size_of[Scalar[dtype]](),
                _uses_amx[dtype](sol, ms[s], ns[s], ks[s]),
            ),
        )


# Run the chosen mode (quick vs stats) for one compile-time dtype. The machine
# SOL is measured once up front so every shape can be reported as a % of its
# roofline. The FMA peak is measured in the dtype the kernels actually compute
# in (f32 for bf16 storage), which is the ceiling the kernels can reach; the
# bandwidth rooflines still use the storage element size via size_of.
def run[dtype: DType](quick: Bool, epochs: Int, runs: Int) raises:
    var sol = measure_sol[
        _compute_dtype[dtype](), dtype == DType.bfloat16
    ]()
    if quick:
        run_quick[dtype](sol, runs)
    else:
        run_stats[dtype](sol, epochs, runs)


def main() raises:
    var args = argv()
    var quick = False
    var epochs = 10
    var runs = 12
    var dt = String("f64")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--quick":
            quick = True
        elif a == "--epochs" and i + 1 < len(args):
            epochs = Int(String(args[i + 1]))
            i += 1
        elif a == "--runs" and i + 1 < len(args):
            runs = Int(String(args[i + 1]))
            i += 1
        elif a == "--dtype" and i + 1 < len(args):
            dt = String(args[i + 1])
            i += 1
        i += 1

    # dtype is a compile-time parameter, so dispatch the string to the matching
    # instantiation. Default is f64, so a bare `mojo bench_focus.mojo` is unchanged.
    if dt == "f32":
        run[DType.float32](quick, epochs, runs)
    elif dt == "f16":
        run[DType.float16](quick, epochs, runs)
    elif dt == "bf16":
        run[DType.bfloat16](quick, epochs, runs)
    elif dt == "f64":
        run[DType.float64](quick, epochs, runs)
    else:
        print("unknown --dtype", dt, "(use f64, f32, f16, or bf16)")
