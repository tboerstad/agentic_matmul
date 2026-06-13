from gemm import (
    matmul_naive,
    matmul_tiled,
    matmul_simd,
    matmul_parallel,
    matmul_register_blocked,
    matmul_packed,
    matmul_comptime,
    matmul_goto,
    matmul_prefill,
    matmul_prefill_opt,
    matmul_decode,
    matmul_dispatch,
)
from matrix import Matrix
from std.collections import List
from std.time import perf_counter_ns


def gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


def fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


def report(label: String, m: Int, n: Int, k: Int, sum_ns: Float64, min_ns: Float64, n_runs: Int) -> Float64:
    # Prints one kernel line; returns the *peak* GFLOPS (from the min time).
    var mean_s = sum_ns / Float64(n_runs) / 1e9
    var min_s = min_ns / 1e9
    var peak = gflops(m, n, k, min_s)
    print(
        "  ", label, ":",
        mean_s * 1e3, "ms |",
        gflops(m, n, k, mean_s), "GFLOPS (mean) |",
        peak, "GFLOPS (peak)",
    )
    return peak


# A per-shape result row, collected for the final comparison table.
struct ShapeResult(Copyable, Movable):
    var label: String
    var m: Int
    var n: Int
    var k: Int
    var decode_peak: Float64
    var prefill_opt_peak: Float64
    var dispatch_peak: Float64
    var best_peak: Float64
    var best_name: String

    def __init__(out self, label: String, m: Int, n: Int, k: Int):
        self.label = label
        self.m = m
        self.n = n
        self.k = k
        self.decode_peak = 0.0
        self.prefill_opt_peak = 0.0
        self.dispatch_peak = 0.0
        self.best_peak = 0.0
        self.best_name = String("-")

    def consider(mut self, name: String, peak: Float64):
        if peak > self.best_peak:
            self.best_peak = peak
            self.best_name = name


def bench_shape(label: String, M: Int, N: Int, K: Int, include_slow: Bool) -> ShapeResult:
    # Benchmarks every kernel on one M×N×K shape and returns a summary row.
    #
    # The number of timed runs adapts to the problem size so the total wall
    # time stays bounded even as we sweep up to large M. `include_slow` adds
    # the single-threaded baselines (naive/tiled/simd) and the speedup-vs-naive
    # table — only worth it for the small headline shapes.
    var flops = 2.0 * Float64(M) * Float64(N) * Float64(K)
    var runs: Int
    if flops < 1e8:
        runs = 10
    elif flops < 1e9:
        runs = 5
    elif flops < 1e10:
        runs = 3
    else:
        runs = 2
    var warmup = 2 if runs > 2 else 1

    var a = Matrix(M, K)
    var b = Matrix(K, N)
    var c = Matrix(M, N)
    fill(a, 17)
    fill(b, 13)

    print("---", label, "(", M, "x", N, "x", K, ") — runs:", runs, "---\n")

    var res = ShapeResult(label, M, N, K)

    var t0: UInt
    var t1: UInt
    var dt: Float64
    var min_ns: Float64
    var sum_ns: Float64

    var s_naive: Float64 = 0.0
    var have_naive = False

    if include_slow:
        # naive — one run (very slow, single-threaded)
        t0 = perf_counter_ns(); matmul_naive(c, a, b); t1 = perf_counter_ns()
        sum_ns = Float64(t1 - t0); min_ns = sum_ns
        s_naive = report("naive       ", M, N, K, sum_ns, min_ns, 1)
        res.consider("naive", s_naive)
        have_naive = True

        # tiled
        matmul_tiled(c, a, b)
        min_ns = 1e30; sum_ns = 0.0
        for _ in range(runs):
            t0 = perf_counter_ns(); matmul_tiled(c, a, b); t1 = perf_counter_ns()
            dt = Float64(t1 - t0); sum_ns += dt
            if dt < min_ns: min_ns = dt
        res.consider("tiled", report("tiled       ", M, N, K, sum_ns, min_ns, runs))

        # simd
        matmul_simd(c, a, b)
        min_ns = 1e30; sum_ns = 0.0
        for _ in range(runs):
            t0 = perf_counter_ns(); matmul_simd(c, a, b); t1 = perf_counter_ns()
            dt = Float64(t1 - t0); sum_ns += dt
            if dt < min_ns: min_ns = dt
        res.consider("simd", report("simd        ", M, N, K, sum_ns, min_ns, runs))

    # parallel
    for _ in range(warmup): matmul_parallel(c, a, b)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(runs):
        t0 = perf_counter_ns(); matmul_parallel(c, a, b); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    res.consider("parallel", report("parallel    ", M, N, K, sum_ns, min_ns, runs))

    # register_blocked
    for _ in range(warmup): matmul_register_blocked(c, a, b)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(runs):
        t0 = perf_counter_ns(); matmul_register_blocked(c, a, b); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    res.consider("regblk", report("regblk      ", M, N, K, sum_ns, min_ns, runs))

    # packed
    for _ in range(warmup): matmul_packed(c, a, b)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(runs):
        t0 = perf_counter_ns(); matmul_packed(c, a, b); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    res.consider("packed", report("packed      ", M, N, K, sum_ns, min_ns, runs))

    # comptime
    for _ in range(warmup): matmul_comptime(c, a, b)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(runs):
        t0 = perf_counter_ns(); matmul_comptime(c, a, b); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    res.consider("comptime", report("comptime    ", M, N, K, sum_ns, min_ns, runs))

    # goto
    for _ in range(warmup): matmul_goto(c, a, b)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(runs):
        t0 = perf_counter_ns(); matmul_goto(c, a, b); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    res.consider("goto", report("goto        ", M, N, K, sum_ns, min_ns, runs))

    # prefill
    for _ in range(warmup): matmul_prefill(c, a, b)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(runs):
        t0 = perf_counter_ns(); matmul_prefill(c, a, b); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    res.consider("prefill", report("prefill     ", M, N, K, sum_ns, min_ns, runs))

    # prefill_opt
    for _ in range(warmup): matmul_prefill_opt(c, a, b)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(runs):
        t0 = perf_counter_ns(); matmul_prefill_opt(c, a, b); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    res.prefill_opt_peak = report("prefill_opt ", M, N, K, sum_ns, min_ns, runs)
    res.consider("prefill_opt", res.prefill_opt_peak)

    # decode
    for _ in range(warmup): matmul_decode(c, a, b)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(runs):
        t0 = perf_counter_ns(); matmul_decode(c, a, b); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    res.decode_peak = report("decode      ", M, N, K, sum_ns, min_ns, runs)
    res.consider("decode", res.decode_peak)

    # dispatch
    for _ in range(warmup): matmul_dispatch(c, a, b)
    min_ns = 1e30; sum_ns = 0.0
    for _ in range(runs):
        t0 = perf_counter_ns(); matmul_dispatch(c, a, b); t1 = perf_counter_ns()
        dt = Float64(t1 - t0); sum_ns += dt
        if dt < min_ns: min_ns = dt
    res.dispatch_peak = report("dispatch    ", M, N, K, sum_ns, min_ns, runs)
    res.consider("dispatch", res.dispatch_peak)

    if have_naive and s_naive > 0.0:
        # Speedup vs naive in terms of peak GFLOPS (higher kernel peak == faster).
        print("\n  speedup vs naive (peak):")
        print("    dispatch    :", res.dispatch_peak / s_naive, "x")

    print("")
    return res^


def main() raises:
    var t_start = perf_counter_ns()
    print("=== matmul benchmark: kernel sweep across many shapes ===")
    print("    (Qwen 2.5 VL 3B MLP projection shapes, float64)\n")

    # MLP projection shapes for Qwen 2.5 VL 3B:
    #   hidden_size = 2048, intermediate_size = 11008
    #   - gate/up proj: K=2048  -> N=11008   ("up"   orientation)
    #   - down   proj: K=11008 -> N=2048     ("down" orientation)
    # We sweep the M dimension (tokens) from decode (M=1) up to large prefill
    # batches so the comparison spans the memory-bound -> compute-bound range.
    var m_sweep = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512]

    var results = List[ShapeResult]()

    # ---- headline shapes: full kernel set incl. naive baseline --------------
    print("######## headline shapes (full kernel set) ########\n")
    results.append(bench_shape("decode  up  ", 1, 11008, 2048, True))
    results.append(bench_shape("prefill up  ", 96, 11008, 2048, True))

    # ---- up-projection sweep (K=2048 -> N=11008) ----------------------------
    print("######## up-proj sweep  (M x 11008 x 2048) ########\n")
    for ref m in m_sweep:
        if m == 1 or m == 96:
            continue  # already covered as headline shapes
        results.append(bench_shape("up   M=" + String(m), m, 11008, 2048, False))

    # ---- down-projection sweep (K=11008 -> N=2048) --------------------------
    print("######## down-proj sweep (M x 2048 x 11008) ########\n")
    for ref m in m_sweep:
        results.append(bench_shape("down M=" + String(m), m, 2048, 11008, False))

    # ---- comparison table ---------------------------------------------------
    print("=" * 92)
    print("SUMMARY — peak GFLOPS per shape (higher is better)")
    print("=" * 92)
    print(
        rjust("shape", 14), "|",
        rjust("M", 5), rjust("N", 7), rjust("K", 7), "|",
        rjust("decode", 9), rjust("prefill_opt", 12), rjust("dispatch", 9), "|",
        rjust("best", 9), " best-kernel",
    )
    print("-" * 92)
    for ref r in results:
        print(
            rjust(r.label, 14), "|",
            rjust(String(r.m), 5), rjust(String(r.n), 7), rjust(String(r.k), 7), "|",
            rjust(fmt1(r.decode_peak), 9),
            rjust(fmt1(r.prefill_opt_peak), 12),
            rjust(fmt1(r.dispatch_peak), 9), "|",
            rjust(fmt1(r.best_peak), 9), " " + r.best_name,
        )
    print("=" * 92)

    var t_end = perf_counter_ns()
    var elapsed_s = Float64(t_end - t_start) / 1e9
    print("\n=== total benchmark wall time:", elapsed_s, "s ===")


def fmt1(x: Float64) -> String:
    # One-decimal fixed-point string for table cells.
    var scaled = Int(x * 10.0 + 0.5)
    return String(scaled // 10) + "." + String(scaled % 10)


def rjust(s: String, width: Int) -> String:
    var out = s
    while out.byte_length() < width:
        out = " " + out
    return out
