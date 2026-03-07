"""Quick correctness check: compare matmul_nr_blocked against matmul_naive."""
from gemm import matmul_naive, matmul_nr_blocked
from matrix import Matrix
from std.testing import assert_almost_equal


fn fill(mut m: Matrix, seed: Int):
    for i in range(m.rows):
        for j in range(m.cols):
            m[i, j] = Scalar[m.dtype]((i * m.cols + j) % seed) * 0.1


fn main() raises:
    # Test 1: small square
    print("Test 1: 8x8x8 ...")
    var a1 = Matrix(8, 8)
    var b1 = Matrix(8, 8)
    var c_ref1 = Matrix(8, 8)
    var c_nr1 = Matrix(8, 8)
    fill(a1, 7)
    fill(b1, 11)
    matmul_naive(c_ref1, a1, b1)
    matmul_nr_blocked(c_nr1, a1, b1)
    for i in range(8):
        for j in range(8):
            assert_almost_equal(c_nr1[i, j], c_ref1[i, j], atol=1e-10)
    print("  PASS")

    # Test 2: non-square, non-tile-aligned
    print("Test 2: 13x37x19 ...")
    var a2 = Matrix(13, 19)
    var b2 = Matrix(19, 37)
    var c_ref2 = Matrix(13, 37)
    var c_nr2 = Matrix(13, 37)
    fill(a2, 7)
    fill(b2, 11)
    matmul_naive(c_ref2, a2, b2)
    matmul_nr_blocked(c_nr2, a2, b2)
    for i in range(13):
        for j in range(37):
            assert_almost_equal(c_nr2[i, j], c_ref2[i, j], atol=1e-10)
    print("  PASS")

    # Test 3: LLM prefill shape
    print("Test 3: 96x11008x2048 (prefill) ...")
    var a3 = Matrix(96, 2048)
    var b3 = Matrix(2048, 11008)
    var c_ref3 = Matrix(96, 11008)
    var c_nr3 = Matrix(96, 11008)
    fill(a3, 17)
    fill(b3, 13)
    matmul_naive(c_ref3, a3, b3)
    matmul_nr_blocked(c_nr3, a3, b3)
    var max_err: Float64 = 0.0
    for i in range(96):
        for j in range(11008):
            var err = abs(c_nr3[i, j] - c_ref3[i, j])
            if err > max_err:
                max_err = err
            assert_almost_equal(c_nr3[i, j], c_ref3[i, j], atol=1e-6)
    print("  PASS (max error:", max_err, ")")

    print("\nAll tests passed!")
