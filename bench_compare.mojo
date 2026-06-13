from gemm import matmul_dispatch, matmul_prefill, matmul_prefill_opt
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


def fmt1(x: Float64) -> String:
    var scaled = Int(x * 10.0 + 0.5)
    return String(scaled // 10) + "." + String(scaled % 10)


def rjust(s: String, width: Int) -> String:
    var out = s
    while out.byte_length() < width:
        out = " " + out
    return out


# Measures our dispatch, our best-of (dispatch/prefill/prefill_opt), and linalg
# *interleaved per shape*, so all contestants see the same turbo/thermal state.
def compare(label: String, M: Int, N: Int, K: Int) raises:
    var flops = 2.0 * Float64(M) * Float64(N) * Float64(K)
    var runs: Int
    if flops < 1e9: runs = 8
    elif flops < 1e10: runs = 4
    else: runs = 3

    var a = Matrix(M, K); var b = Matrix(K, N); var c = Matrix(M, N)
    fill(a, 17); fill(b, 13)

    # linalg needs TileTensor views over raw buffers.
    var c_data = List[Float64](capacity=M * N)
    for _ in range(M * N): c_data.append(0.0)
    var c_tile = TileTensor(c_data.unsafe_ptr(), row_major(Coord(M, N)))
    var a_tile = TileTensor(a.data.unsafe_ptr(), row_major(Coord(M, K)))
    var b_tile = TileTensor(b.data.unsafe_ptr(), row_major(Coord(K, N)))

    var t0: UInt; var t1: UInt; var dt: Float64

    # dispatch
    for _ in range(2): matmul_dispatch(c, a, b)
    var disp = Float64(1e30)
    for _ in range(runs):
        t0 = perf_counter_ns(); matmul_dispatch(c, a, b); t1 = perf_counter_ns()
        dt = Float64(t1 - t0)
        if dt < disp: disp = dt

    # prefill (8x24 KC512 KU8)
    for _ in range(2): matmul_prefill(c, a, b)
    var pre = Float64(1e30)
    for _ in range(runs):
        t0 = perf_counter_ns(); matmul_prefill(c, a, b); t1 = perf_counter_ns()
        dt = Float64(t1 - t0)
        if dt < pre: pre = dt

    # prefill_opt (6x32 KC256 KU2 T64)
    for _ in range(2): matmul_prefill_opt(c, a, b)
    var preo = Float64(1e30)
    for _ in range(runs):
        t0 = perf_counter_ns(); matmul_prefill_opt(c, a, b); t1 = perf_counter_ns()
        dt = Float64(t1 - t0)
        if dt < preo: preo = dt

    # linalg
    for _ in range(2): linalg_matmul[target="cpu"](c_tile, a_tile, b_tile, None)
    var lin = Float64(1e30)
    for _ in range(runs):
        t0 = perf_counter_ns()
        linalg_matmul[target="cpu"](c_tile, a_tile, b_tile, None)
        t1 = perf_counter_ns()
        dt = Float64(t1 - t0)
        if dt < lin: lin = dt

    var g_disp = gflops(M, N, K, disp / 1e9)
    var g_pre = gflops(M, N, K, pre / 1e9)
    var g_preo = gflops(M, N, K, preo / 1e9)
    var g_lin = gflops(M, N, K, lin / 1e9)
    var g_best = max(g_disp, max(g_pre, g_preo))
    var verdict = "WIN " if g_best >= g_lin else "lose"
    var ratio = g_best / g_lin

    print(
        rjust(label, 12), "|",
        rjust(fmt1(g_disp), 8), rjust(fmt1(g_pre), 8), rjust(fmt1(g_preo), 8), "|",
        rjust(fmt1(g_best), 8), "(ours) vs", rjust(fmt1(g_lin), 8), "(linalg)  ",
        verdict, fmt1(ratio * 100.0) + "%",
    )


def main() raises:
    print("shape        |  dispatch  prefill  prefopt |     best         linalg     verdict")
    print("-" * 92)
    var m_sweep = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512]
    for ref m in m_sweep:
        compare("up   M=" + String(m), m, 11008, 2048)
    print("-" * 92)
    for ref m in m_sweep:
        compare("down M=" + String(m), m, 2048, 11008)
