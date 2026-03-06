from std.testing import assert_almost_equal, TestSuite
from matrix import Matrix


fn matmul[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype],
    a: Matrix[dtype],
    b: Matrix[dtype],
):
    var m = a.rows
    var k = a.cols
    var n = c.cols

    for i in range(m):
        for j in range(n):
            var dot = Scalar[dtype](0)
            for p in range(k):
                var a_val = a[i, p]

                var b_val: Scalar[dtype]

                comptime if transpose_b:
                    b_val = b[j, p]
                else:
                    b_val = b[p, j]

                dot += a_val * b_val

            c[i, j] = dot


# A = [[0, 1], [2, 3]], B = [[5, 6], [7, 8]], expected C = [[7, 8], [31, 36]]
def test_basic_2x2() raises:
    var a = Matrix(2, 2)
    a[0, 0] = 0.0; a[0, 1] = 1.0
    a[1, 0] = 2.0; a[1, 1] = 3.0

    var b = Matrix(2, 2)
    b[0, 0] = 5.0; b[0, 1] = 6.0
    b[1, 0] = 7.0; b[1, 1] = 8.0

    var c = Matrix(2, 2)
    matmul(c, a, b)
    assert_almost_equal(c[0, 0], 7.0)
    assert_almost_equal(c[0, 1], 8.0)
    assert_almost_equal(c[1, 0], 31.0)
    assert_almost_equal(c[1, 1], 36.0)


# trans_b=True: op(B) = B^T
# A = identity, B = [[1, 2], [3, 4]] -> B^T = [[1, 3], [2, 4]]
def test_trans_b() raises:
    var a = Matrix(2, 2)
    a[0, 0] = 1.0; a[0, 1] = 0.0
    a[1, 0] = 0.0; a[1, 1] = 1.0

    var b = Matrix(2, 2)
    b[0, 0] = 1.0; b[0, 1] = 2.0
    b[1, 0] = 3.0; b[1, 1] = 4.0

    var c = Matrix(2, 2)
    matmul[transpose_b=True](c, a, b)
    assert_almost_equal(c[0, 0], 1.0)
    assert_almost_equal(c[0, 1], 3.0)
    assert_almost_equal(c[1, 0], 2.0)
    assert_almost_equal(c[1, 1], 4.0)


# 1x1 degenerate: scalar multiply
def test_1x1() raises:
    var a = Matrix(1, 1)
    a[0, 0] = 3.0

    var b = Matrix(1, 1)
    b[0, 0] = 7.0

    var c = Matrix(1, 1)
    matmul(c, a, b)
    assert_almost_equal(c[0, 0], 21.0)


# Non-square: A is 2x3, B is 3x2, C is 2x2
def test_non_square() raises:
    # A = [[1, 2, 3], [4, 5, 6]]
    var a = Matrix(2, 3)
    a[0, 0] = 1.0; a[0, 1] = 2.0; a[0, 2] = 3.0
    a[1, 0] = 4.0; a[1, 1] = 5.0; a[1, 2] = 6.0

    # B = [[7, 8], [9, 10], [11, 12]]
    var b = Matrix(3, 2)
    b[0, 0] = 7.0; b[0, 1] = 8.0
    b[1, 0] = 9.0; b[1, 1] = 10.0
    b[2, 0] = 11.0; b[2, 1] = 12.0

    var c = Matrix(2, 2)
    matmul(c, a, b)
    # C[0,0] = 1*7 + 2*9 + 3*11 = 58
    # C[0,1] = 1*8 + 2*10 + 3*12 = 64
    # C[1,0] = 4*7 + 5*9 + 6*11 = 139
    # C[1,1] = 4*8 + 5*10 + 6*12 = 154
    assert_almost_equal(c[0, 0], 58.0)
    assert_almost_equal(c[0, 1], 64.0)
    assert_almost_equal(c[1, 0], 139.0)
    assert_almost_equal(c[1, 1], 154.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
