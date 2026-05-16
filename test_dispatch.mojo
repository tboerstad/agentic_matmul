from gemm import matmul_naive, matmul_dispatch
from matrix import Matrix


fn fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


def assert_close(c: Matrix[DType.float64], refm: Matrix[DType.float64], name: String, tol: Float64) raises:
    for i in range(c.rows):
        for j in range(c.cols):
            var d = Float64(c[i, j] - refm[i, j])
            if d < 0:
                d = -d
            if d > tol:
                print("MISMATCH in", name, "at", i, j, ":", c[i, j], "vs", refm[i, j])
                raise Error("mismatch")
    print("  ", name, ": OK")


def main() raises:
    # Tiny irregular shape — exercises remainder paths
    var a = Matrix(17, 41)
    var b = Matrix(41, 53)
    var ref_small = Matrix(17, 53)
    var c_small = Matrix(17, 53)
    fill(a, 11)
    fill(b, 7)
    matmul_naive(ref_small, a, b)
    matmul_dispatch(c_small, a, b)
    assert_close(c_small, ref_small, "dispatch 17x53x41", tol=1e-9)

    # Decode shape (M=1): exercises decode path
    var ad = Matrix(1, 128)
    var bd = Matrix(128, 64)
    var ref_d = Matrix(1, 64)
    var c_d = Matrix(1, 64)
    fill(ad, 11)
    fill(bd, 7)
    matmul_naive(ref_d, ad, bd)
    matmul_dispatch(c_d, ad, bd)
    assert_close(c_d, ref_d, "dispatch 1x64x128", tol=1e-9)

    # Prefill-ish shape: M=96 divides MR=6 evenly, exercises full microkernel
    var ap = Matrix(96, 256)
    var bp = Matrix(256, 128)
    var ref_p = Matrix(96, 128)
    var c_p = Matrix(96, 128)
    fill(ap, 13)
    fill(bp, 17)
    matmul_naive(ref_p, ap, bp)
    matmul_dispatch(c_p, ap, bp)
    assert_close(c_p, ref_p, "dispatch 96x128x256", tol=1e-6)

    # Odd MR boundary: M=5 (one below MR=6) — must go through decode/GEMV
    var ar = Matrix(5, 64)
    var br = Matrix(64, 32)
    var ref_r = Matrix(5, 32)
    var c_r = Matrix(5, 32)
    fill(ar, 11)
    fill(br, 7)
    matmul_naive(ref_r, ar, br)
    matmul_dispatch(c_r, ar, br)
    assert_close(c_r, ref_r, "dispatch 5x32x64 (decode path)", tol=1e-9)

    # Odd MR boundary: M=7 (just above MR=6) — must handle 1 remainder row
    var ar2 = Matrix(7, 64)
    var br2 = Matrix(64, 32)
    var ref_r2 = Matrix(7, 32)
    var c_r2 = Matrix(7, 32)
    fill(ar2, 11)
    fill(br2, 7)
    matmul_naive(ref_r2, ar2, br2)
    matmul_dispatch(c_r2, ar2, br2)
    assert_close(c_r2, ref_r2, "dispatch 7x32x64 (1 remainder row)", tol=1e-9)

    # Real prefill shape — compare dispatch against naive on a slice
    var a_full = Matrix(96, 2048)
    var b_full = Matrix(2048, 11008)
    fill(a_full, 17)
    fill(b_full, 13)
    var c_full = Matrix(96, 11008)
    matmul_dispatch(c_full, a_full, b_full)
    # Spot check: recompute one row via naive
    var row = 47
    var col = 5003
    var expected = Float64(0)
    for p in range(2048):
        expected += Float64(a_full[row, p]) * Float64(b_full[p, col])
    var got = Float64(c_full[row, col])
    var d = expected - got
    if d < 0:
        d = -d
    if d > 1e-6:
        print("MISMATCH full prefill at", row, col, ":", got, "vs", expected)
        raise Error("mismatch")
    print("   dispatch 96x11008x2048 spot check (47,5003): OK (", got, "≈", expected, ")")
