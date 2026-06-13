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


def bench_one(label: String, m: Int, n: Int, k: Int) raises:
    var a = Matrix(m, k)
    var b = Matrix(k, n)
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

    # Warmup
    for _ in range(2):
        linalg_matmul[target="cpu"](c_tile, a_tile, b_tile, None)

    # Measure peak across N runs
    var n_runs = 5
    var min_ns = Float64(1e30)
    var sum_ns = Float64(0)
    for _ in range(n_runs):
        var t0 = perf_counter_ns()
        linalg_matmul[target="cpu"](c_tile, a_tile, b_tile, None)
        var t1 = perf_counter_ns()
        var dt = Float64(t1 - t0)
        sum_ns += dt
        if dt < min_ns:
            min_ns = dt

    var mean_s = sum_ns / Float64(n_runs) / 1e9
    var min_s = min_ns / 1e9
    print(
        "  ", label, ":",
        mean_s * 1e3, "ms (mean) |",
        gflops(m, n, k, mean_s), "GFLOPS (mean) |",
        gflops(m, n, k, min_s), "GFLOPS (peak)",
    )


def main() raises:
    print("=" * 50)
    print("  linalg.matmul benchmark (Mojo stdlib)")
    print("=" * 50)
    print("")

    # Same shape grid as bench_matmul.mojo / bench_sota.py: sweep the token
    # dimension M across both Qwen 2.5 VL 3B MLP projection orientations.
    var m_sweep = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512]

    print("--- up-proj sweep  (M x 11008 x 2048) ---\n")
    for ref m in m_sweep:
        bench_one("up   M=" + String(m), m, 11008, 2048)
    print("")

    print("--- down-proj sweep (M x 2048 x 11008) ---\n")
    for ref m in m_sweep:
        bench_one("down M=" + String(m), m, 2048, 11008)
