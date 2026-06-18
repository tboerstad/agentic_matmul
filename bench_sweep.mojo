# Per-M shape sweep: matmul_dispatch vs stdlib linalg.matmul, interleaved.
#
# Walks M = 1..512 over both Qwen 2.5 VL 3B MLP projection orientations
# (wide-N up/gate proj and tall-K down proj) and prints the dispatch/linalg
# GFLOPS ratio per shape with a WIN/LOSE tag. The two kernels are interleaved
# per repetition so both see the same turbo/thermal state — this is the tool
# for spotting exactly which dims we trail linalg on.
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


def bench_shape(label: String, m: Int, n: Int, k: Int) raises:
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

    var n_runs = 8
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


def main() raises:
    var ms = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512]

    print("=== up/gate proj (wide-N): N=11008 K=2048 ===\n")
    for i in range(len(ms)):
        bench_shape("up  ", ms[i], 11008, 2048)

    print("\n=== down proj (tall-K): N=2048 K=11008 ===\n")
    for i in range(len(ms)):
        bench_shape("down", ms[i], 2048, 11008)

    # --- Extra (M, N, K) probe shapes ---------------------------------------
    # The dispatch hard-codes TILE_N / KC for the two Qwen orientations above
    # (N=11008 and N=2048 split evenly across 4 workers). These shapes leave
    # that comfort zone to surface where we trail linalg: square GEMMs (which
    # sit on the N>=K branch boundary), other N values that tile unevenly into
    # 4 workers, odd N/K (remainder paths), small-K (low arithmetic intensity),
    # and larger compute-bound GEMMs.
    var lbl = [
        # Square GEMM (N == K -> wide-N branch). Classic compute-bound case.
        String("sq256 "), "sq512 ", "sq1024", "sq2048",
        # Larger square-ish / other-model FFN shapes (hidden 4096, Llama-ish).
        "h4k-m128", "h4k-m512", "ffn-up8k", "ffn-dn8k",
        # N values that do NOT divide cleanly across 4 workers at TILE_N=64/72.
        "N3072 ", "N4000 ", "N5504 ",
        # Odd dims -> exercise the SIMD N-remainder and K tail directly.
        "N-odd ", "K-odd ", "M-rem ", "M-prime",
        # Small K: low FLOP/byte, packing overhead dominates.
        "K128  ", "K256w ",
    ]
    var em = [
        256, 512, 1024, 2048,
        128, 512, 512, 512,
        512, 512, 512,
        512, 512, 100, 333,
        512, 512,
    ]
    var en = [
        256, 512, 1024, 2048,
        4096, 4096, 8192, 2048,
        3072, 4000, 5504,
        11007, 2048, 11008, 4096,
        2048, 11008,
    ]
    var ek = [
        256, 512, 1024, 2048,
        4096, 4096, 2048, 8192,
        2048, 2048, 2048,
        2048, 2047, 2048, 4096,
        128, 256,
    ]

    print("\n=== extra (M,N,K) probe shapes ===\n")
    for i in range(len(em)):
        bench_shape(lbl[i], em[i], en[i], ek[i])
