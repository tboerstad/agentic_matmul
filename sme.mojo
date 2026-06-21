# Benchmark + correctness harness for the SME f64 GEMM (sme_kernel.sme_gemm).
from sme_kernel import sme_gemm, MR
from matrix import Matrix
from std.time import perf_counter_ns
from std.collections import InlineArray

# --------------------------------------------------------------------------------
# Test + benchmark
# --------------------------------------------------------------------------------
from gemm import matmul_dispatch


def gflops(m: Int, n: Int, k: Int, secs: Float64) -> Float64:
    return (2.0 * Float64(m) * Float64(n) * Float64(k)) / (secs * 1e9)


def fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


def _peak(mut c: Matrix[DType.float64], a: Matrix[DType.float64], b: Matrix[DType.float64], nw: Int, KC: Int, MC: Int, n_runs: Int) -> Float64:
    var best_ns = UInt(0)
    for r in range(n_runs):
        var t0 = perf_counter_ns()
        sme_gemm(c, a, b, nw, KC, MC)
        var t1 = perf_counter_ns()
        var dt = t1 - t0
        if r == 0 or dt < best_ns: best_ns = dt
    # GFLOPS = 2*m*n*k / best_ns (best_ns in ns cancels the 1e9).
    return 2.0 * Float64(c.rows) * Float64(c.cols) * Float64(a.cols) / Float64(best_ns)


def bench(label: String, M: Int, N: Int, K: Int, n_runs: Int, nw: Int) raises:
    var a = Matrix(M, K)
    var b = Matrix(K, N)
    var c = Matrix(M, N)
    var cref = Matrix(M, N)
    fill(a, 17)
    fill(b, 13)
    matmul_dispatch(cref, a, b)

    var BIG = 1 << 30
    # (KC, MC) candidate configs. BIG = no blocking on that axis.
    var kcs = [BIG, 512, 512, 512, 256, 256, 1024, 384, 768, 512]
    var mcs = [BIG, 256, 512, 128, 256, 128, 256, 256, 128, 384]
    var best = Float64(0.0)
    var best_kc = 0
    var best_mc = 0
    # correctness once (a blocked config exercises both micro-kernels)
    for i in range(M * N): c.data[i] = 0.0
    sme_gemm(c, a, b, nw, 256, 256)
    var max_err = Float64(0.0)
    for i in range(M * N):
        var e = abs(c.data[i] - cref.data[i])
        if e > max_err: max_err = e
    for cfg in range(len(kcs)):
        var kc = kcs[cfg]
        var mc = mcs[cfg]
        var g = _peak(c, a, b, nw, kc, mc, n_runs)
        if g > best:
            best = g
            best_kc = kc
            best_mc = mc
    print("  ", label, "M=", M, "N=", N, "K=", K, "num_i=", M // MR,
          "| best", Int(best), "@KC=", best_kc, "MC=", best_mc, "| max_err", max_err)


def main() raises:
    var n_runs = 30
    print("=== SME f64 GEMM (jc-pc-ic, KC-blocked): best KC per shape ===")
    bench("prefill ", 96, 11008, 2048, n_runs, 2)
    bench("sq512   ", 512, 512, 512, n_runs, 2)
    bench("sq1024  ", 1024, 1024, 1024, n_runs, 2)
    bench("sq2048  ", 2048, 2048, 2048, n_runs, 2)
    bench("M512-g  ", 512, 4096, 4096, n_runs, 2)
    bench("up-m256 ", 256, 11008, 2048, n_runs, 2)
    bench("up-m512 ", 512, 11008, 2048, n_runs, 2)
    bench("dn-m512 ", 512, 2048, 11008, n_runs, 2)
    # Small/medium divisible shapes to decide the SME-vs-NEON dispatch gate.
    bench("sq128   ", 128, 128, 128, n_runs, 2)
    bench("sq256   ", 256, 256, 256, n_runs, 2)
    bench("sq384   ", 384, 384, 384, n_runs, 2)
    bench("box512  ", 512, 128, 512, n_runs, 2)
    bench("sq64    ", 64, 64, 64, n_runs, 2)
    # Non-divisible shapes (exercise the remainder regions).
    bench("oddN11007", 512, 11007, 2048, n_runs, 2)
    bench("M100    ", 100, 11008, 2048, n_runs, 2)
    bench("N4001   ", 512, 4001, 2048, n_runs, 2)
    bench("M100N99 ", 100, 4099, 2048, n_runs, 2)
    bench("sq127   ", 127, 127, 512, n_runs, 2)
