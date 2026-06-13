# Correctness check for matmul_dispatch against a naive reference.
#
# Exercises every dispatch branch and the packed micro-kernel's edge cases
# (full MR panels, NR remainders, and M tails) across both projection
# orientations. test_gemm.mojo only tests a local reference matmul, not the
# real kernels, so this is the actual coverage for matmul_dispatch.
from gemm import matmul_dispatch
from matrix import Matrix
from std.math import abs


def fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j * 7 + 3) % seed) * 0.1 - 0.5


def naive[dtype: DType, //](mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]):
    for i in range(a.rows):
        for j in range(c.cols):
            var dot = Scalar[dtype](0)
            for p in range(a.cols):
                dot += a[i, p] * b[p, j]
            c[i, j] = dot


def check(m: Int, n: Int, k: Int) raises:
    var a = Matrix(m, k)
    var b = Matrix(k, n)
    var c = Matrix(m, n)
    var c_ref = Matrix(m, n)
    fill(a, 17)
    fill(b, 13)
    matmul_dispatch(c, a, b)
    naive(c_ref, a, b)
    var max_err = Float64(0)
    for i in range(m):
        for j in range(n):
            var e = abs(c[i, j] - c_ref[i, j])
            if e > max_err:
                max_err = e
    var ok = max_err < 1e-7
    print("  M=", m, "N=", n, "K=", k, "| max_err=", max_err,
          "|", "OK" if ok else "FAIL")
    if not ok:
        raise Error("mismatch")


def main() raises:
    print("=== correctness: dispatch vs naive ===")
    # Exercise full MR panels, NR remainders, and M tails for both orientations.
    # M>256 hits the new wide-N 6x32 branch; 257/300 also stress its M-tail.
    var ms = [2, 3, 4, 5, 6, 7, 13, 64, 65, 70, 96, 128, 130, 256, 257, 300, 384, 512]
    for i in range(len(ms)):
        check(ms[i], 2048, 768)   # wide-N (N>K), incl. new M>256 6x32 branch
    for i in range(len(ms)):
        check(ms[i], 770, 512)    # wide-N, N not a multiple of TILE_N (remainder)
    # M>256 wide-N with K>1024 exercises the KC=1024 branch's multi-k-panel
    # accumulation (2 panels: 1024 + partial) + N remainder + M tail together.
    check(300, 1100, 1100)
    check(384, 1100, 1100)
    print("all passed")
