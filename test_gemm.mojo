from std.testing import assert_almost_equal, TestSuite


fn matmul[*, transpose_b: Bool = False](
    mut c: List[Float64],
    a: List[Float64],
    b: List[Float64],
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for j in range(n):
            var dot: Float64 = 0.0
            for p in range(k):
                var a_val = a[i * k + p]

                var b_val: Float64

                comptime if transpose_b:
                    b_val = b[j * k + p]
                else:
                    b_val = b[p * n + j]

                dot += a_val * b_val

            c[i * n + j] = dot


# A = [[0, 1], [2, 3]], B = [[5, 6], [7, 8]], expected C = [[7, 8], [31, 36]]
def test_basic_2x2() raises:
    var a: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var b: List[Float64] = [5.0, 6.0, 7.0, 8.0]
    var c: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    matmul(c, a, b, m=2, n=2, k=2)
    assert_almost_equal(c[0], 7.0)
    assert_almost_equal(c[1], 8.0)
    assert_almost_equal(c[2], 31.0)
    assert_almost_equal(c[3], 36.0)


# trans_b=True: op(B) = B^T
# A = identity, B = [[1, 2], [3, 4]] -> B^T = [[1, 3], [2, 4]]
def test_trans_b() raises:
    var a: List[Float64] = [1.0, 0.0, 0.0, 1.0]  # identity
    var b: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var c: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    matmul[transpose_b=True](c, a, b, m=2, n=2, k=2)
    assert_almost_equal(c[0], 1.0)
    assert_almost_equal(c[1], 3.0)
    assert_almost_equal(c[2], 2.0)
    assert_almost_equal(c[3], 4.0)


# 1x1 degenerate: scalar multiply
def test_1x1() raises:
    var a: List[Float64] = [3.0]
    var b: List[Float64] = [7.0]
    var c: List[Float64] = [0.0]
    matmul(c, a, b, m=1, n=1, k=1)
    assert_almost_equal(c[0], 21.0)


# Non-square: A is 2x3, B is 3x2, C is 2x2
def test_non_square() raises:
    # A = [[1, 2, 3], [4, 5, 6]]
    var a: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # B = [[7, 8], [9, 10], [11, 12]]
    var b: List[Float64] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    var c: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    matmul(c, a, b, m=2, n=2, k=3)
    # C[0,0] = 1*7 + 2*9 + 3*11 = 58
    # C[0,1] = 1*8 + 2*10 + 3*12 = 64
    # C[1,0] = 4*7 + 5*9 + 6*11 = 139
    # C[1,1] = 4*8 + 5*10 + 6*12 = 154
    assert_almost_equal(c[0], 58.0)
    assert_almost_equal(c[1], 64.0)
    assert_almost_equal(c[2], 139.0)
    assert_almost_equal(c[3], 154.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
