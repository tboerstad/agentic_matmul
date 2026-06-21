# Focused A/B harness: dispatch vs stdlib linalg over many runs (peak GFLOPS).
#
# Complements bench_sweep: where --iterate scans many corner shapes at n_runs=5
# (fast, but far too noisy to trust a single sub-millisecond shape -- sq512 was
# seen at ratio 1.15 then 0.95 with byte-identical code), this runs a small set
# of headline + heavy shapes for N_RUNS reps and reports the PEAK (min-time)
# GFLOPS, the thermal-robust statistic the README mandates for judging a kernel
# change. Interleaved per rep so dispatch and linalg see the same turbo/thermal
# state; still compare the dispatch/linalg RATIO across edits, never absolute
# GFLOPS across separate process launches.
from gemm import matmul_dispatch
from matrix import Matrix
from linalg.matmul import matmul as linalg_matmul
from layout import Coord, TileTensor, row_major
from std.collections import List
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

    var d_gf = gflops(m, n, k, d_min / 1e9)
    var l_gf = gflops(m, n, k, l_min / 1e9)
    print(
        "  ", label, "M=", m, "N=", n, "K=", k,
        "| dispatch", Int(d_gf), "| linalg", Int(l_gf),
        "| ratio", d_gf / l_gf,
    )


def main() raises:
    var n_runs = 40
    print("=== focused bench (peak GFLOPS over", n_runs, "runs) ===")
    # Headline Qwen shapes.
    bench_shape("decode  ", 1, 11008, 2048, n_runs)
    bench_shape("prefill ", 96, 11008, 2048, n_runs)
    # Heavy compute-bound shapes where the P-core cap matters most.
    bench_shape("sq512   ", 512, 512, 512, n_runs)
    bench_shape("sq1024  ", 1024, 1024, 1024, n_runs)
    bench_shape("sq2048  ", 2048, 2048, 2048, n_runs)
    bench_shape("M512-g  ", 512, 4096, 4096, n_runs)
    bench_shape("up-m256 ", 256, 11008, 2048, n_runs)
    bench_shape("up-m512 ", 512, 11008, 2048, n_runs)
    bench_shape("dn-m512 ", 512, 2048, 11008, n_runs)
    # Small cache-resident boxes (noisy; on all cores via _nopack_gemm).
    bench_shape("sq128   ", 128, 128, 128, n_runs)
    bench_shape("sq256   ", 256, 256, 256, n_runs)
    bench_shape("sq384   ", 384, 384, 384, n_runs)
    bench_shape("box512  ", 512, 128, 512, n_runs)
    bench_shape("oddN    ", 512, 11007, 2048, n_runs)
