# General-purpose matmul benchmark: matmul_dispatch vs stdlib linalg.matmul.
#
# Compares our shape-adaptive matmul_dispatch against the Mojo stdlib
# linalg.matmul across a range of (M, N, K) shapes and prints the dispatch/
# linalg GFLOPS ratio per shape with a WIN/LOSE tag. The two kernels are
# interleaved per repetition so both see the same turbo/thermal state.
#
# Two modes (pick one on the command line):
#
#   mojo bench_sweep.mojo --iterate   (default)
#       Fast, surgical pass over the CORNERS & EDGES of the (M,N,K) space —
#       the shapes most likely to expose a dispatch/tiling weakness or a
#       regression: square GEMMs (N==K branch boundary), N that doesn't tile
#       evenly across the workers, odd N/K (remainder paths), small K (low
#       arithmetic intensity), and the M dispatch thresholds (1, just-past
#       small-batch, large). ~A few seconds of runtime — built for the
#       edit-kernel -> remeasure dev loop.
#
#   mojo bench_sweep.mojo --full
#       The complete picture: a per-M sweep across several aspect ratios
#       (square, wide-N, tall-K, plus the two Qwen MLP orientations) and a
#       broad general (M,N,K) shape grid. Slower; run before committing a
#       kernel change to confirm nothing regressed off the corners.
#
# The goal is a GENERAL-PURPOSE matmul, not a Qwen-specific one: the Qwen
# shapes are included as two aspect ratios among many, not as the headline.
from gemm import matmul_dispatch
from matrix import Matrix
from linalg.matmul import matmul as linalg_matmul
from layout import Coord, TileTensor, row_major
from std.collections import List
from std.sys import argv
from std.time import perf_counter_ns


def gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


def fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


def bench_shape(label: String, m: Int, n: Int, k: Int, n_runs: Int) raises:
    var a = Matrix(m, k)
    var b = Matrix(k, n)
    var c = Matrix(m, n)
    fill(a, 17)
    fill(b, 13)

    var c_data = List[Float64](capacity=m * n)
    for _ in range(m * n):
        c_data.append(0.0)
    var a_ptr = a.data.unsafe_ptr()
    var b_ptr = b.data.unsafe_ptr()
    var c_ptr = c_data.unsafe_ptr()
    var c_tile = TileTensor(c_ptr, row_major(Coord(m, n)))
    var a_tile = TileTensor(a_ptr, row_major(Coord(m, k)))
    var b_tile = TileTensor(b_ptr, row_major(Coord(k, n)))

    # Warmup both
    matmul_dispatch(c, a, b)
    linalg_matmul[target="cpu"](c_tile, a_tile, b_tile, None)

    var t0: UInt
    var t1: UInt
    var dt: Float64

    # Interleave per run so both kernels see the same turbo/thermal state.
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

    var d_gf = gflops(m, n, k, d_min / 1e9)
    var l_gf = gflops(m, n, k, l_min / 1e9)
    var ratio = d_gf / l_gf
    var tag = String("WIN ") if ratio >= 1.0 else String("LOSE")
    print(
        "  ", label, "M=", m, "N=", n, "K=", k,
        "| dispatch", d_gf, "| linalg", l_gf,
        "| ratio", ratio, tag,
    )


def run_set(
    title: String,
    labels: List[String],
    ms: List[Int],
    ns: List[Int],
    ks: List[Int],
    n_runs: Int,
) raises:
    print("\n===", title, "===\n")
    for i in range(len(ms)):
        bench_shape(labels[i], ms[i], ns[i], ks[i], n_runs)


# --------------------------------------------------------------------------
# --iterate: corners & edges only. Fast. The shapes here are deliberately the
# ones that stress the dispatch's decision boundaries and remainder paths.
# --------------------------------------------------------------------------
def run_iterate() raises:
    print("mode: --iterate (corners & edges, fast)")
    var n_runs = 5
    var labels = [
        # M dispatch thresholds on a neutral general shape (N==K==4096).
        String("M1-gemv "), "M6-batch", "M512-g ",
        # Square GEMMs (N==K -> wide-N branch boundary; currently the worst).
        "sq256  ", "sq512  ", "sq1024 ", "sq2048 ",
        # N that does NOT tile evenly across 4 workers (load imbalance).
        "N4000  ", "N3072  ",
        # Odd N / K -> SIMD remainder + scalar tail paths.
        "N-odd  ", "K-odd  ",
        # Small K -> low FLOP/byte, packing overhead dominates.
        "K128   ",
    ]
    var ms = [
        1, 6, 512,
        256, 512, 1024, 2048,
        512, 512,
        512, 512,
        512,
    ]
    var ns = [
        4096, 4096, 4096,
        256, 512, 1024, 2048,
        4000, 3072,
        11007, 2048,
        2048,
    ]
    var ks = [
        4096, 4096, 4096,
        256, 512, 1024, 2048,
        2048, 2048,
        2048, 2047,
        128,
    ]
    run_set("corners & edges", labels, ms, ns, ks, n_runs)


# --------------------------------------------------------------------------
# --full: the complete picture. Per-M sweeps across several aspect ratios,
# plus a broad general (M,N,K) grid.
# --------------------------------------------------------------------------
def run_full() raises:
    print("mode: --full (complete sweep, slow)")
    var n_runs = 8
    var ms = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512]

    # Per-M sweeps over four aspect ratios. Two are general (square, and a
    # 4x wide-N / tall-K pair); two are the Qwen MLP orientations, kept as
    # examples rather than the headline.
    var sq_l = List[String]()
    var sq_m = List[Int](); var sq_n = List[Int](); var sq_k = List[Int]()
    for i in range(len(ms)):
        sq_l.append("sq  "); sq_m.append(ms[i]); sq_n.append(ms[i]); sq_k.append(ms[i])
    run_set("square sweep (M = N = K)", sq_l, sq_m, sq_n, sq_k, n_runs)

    var w_l = List[String]()
    for i in range(len(ms)): w_l.append("wide")
    run_set("wide-N sweep: N=8192 K=2048", w_l, ms,
            [8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192],
            [2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048], n_runs)

    var t_l = List[String]()
    for i in range(len(ms)): t_l.append("tall")
    run_set("tall-K sweep: N=2048 K=8192", t_l, ms,
            [2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048],
            [8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192], n_runs)

    var up_l = List[String]()
    for i in range(len(ms)): up_l.append("up  ")
    run_set("Qwen up/gate proj (wide-N): N=11008 K=2048", up_l, ms,
            [11008,11008,11008,11008,11008,11008,11008,11008,11008,11008,11008],
            [2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048], n_runs)

    var dn_l = List[String]()
    for i in range(len(ms)): dn_l.append("down")
    run_set("Qwen down proj (tall-K): N=2048 K=11008", dn_l, ms,
            [2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048],
            [11008,11008,11008,11008,11008,11008,11008,11008,11008,11008,11008], n_runs)

    # Broad general (M,N,K) grid — corner/edge shapes plus a few larger GEMMs.
    var g_l = [
        String("h4k-m128"), "h4k-m512", "ffn-up8k", "ffn-dn8k",
        "N4000   ", "N5504   ", "N-odd   ", "K-odd   ",
        "M-rem   ", "M-prime ", "K128    ", "K256w   ",
    ]
    var g_m = [
        128, 512, 512, 512,
        512, 512, 512, 512,
        100, 333, 512, 512,
    ]
    var g_n = [
        4096, 4096, 8192, 2048,
        4000, 5504, 11007, 2048,
        11008, 4096, 2048, 11008,
    ]
    var g_k = [
        4096, 4096, 2048, 8192,
        2048, 2048, 2048, 2047,
        2048, 4096, 128, 256,
    ]
    run_set("general (M,N,K) grid", g_l, g_m, g_n, g_k, n_runs)


def main() raises:
    var args = argv()
    var full = False
    var iterate = False
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--full":
            full = True
        elif a == "--iterate":
            iterate = True

    if full and not iterate:
        run_full()
    elif iterate and not full:
        run_iterate()
    elif full and iterate:
        print("pick one of --full / --iterate, not both")
    else:
        # Default to the fast iterate loop.
        print("(no mode flag given; defaulting to --iterate. Use --full for the complete sweep.)")
        run_iterate()
