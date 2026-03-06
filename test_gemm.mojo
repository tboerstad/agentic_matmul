from std.testing import assert_almost_equal, TestSuite


struct Matrix[cols: Int](Movable):
    var data: List[Float64]
    var rows: Int

    fn __init__(out self, rows: Int):
        self.rows = rows
        self.data = List[Float64](length=rows * Self.cols, fill=0.0)

    fn __getitem__(self, row: Int) -> SIMD[DType.float64, Self.cols]:
        return (self.data.unsafe_ptr() + row * Self.cols).load[width=Self.cols]()

    fn __setitem__(mut self, row: Int, val: SIMD[DType.float64, Self.cols]):
        (self.data.unsafe_ptr() + row * Self.cols).store(val)


fn gemm[N: Int, K: Int](
    trans_a: Bool,
    trans_b: Bool,
    m: Int,
    alpha: Float64,
    beta: Float64,
    a: Matrix[K],
    b: Matrix[N],
    c: Matrix[N],
) -> Matrix[N]:
    var result = Matrix[N](rows=m)

    if not trans_a and not trans_b:
        for i in range(m):
            var acc = SIMD[DType.float64, N](0)
            for p in range(K):
                acc += a[i][p] * b[p]
            result[i] = alpha * acc + beta * c[i]
    elif trans_a and not trans_b:
        for i in range(m):
            var acc = SIMD[DType.float64, N](0)
            for p in range(K):
                acc += a[p][i] * b[p]
            result[i] = alpha * acc + beta * c[i]
    elif not trans_a and trans_b:
        for i in range(m):
            var acc = SIMD[DType.float64, N](0)
            for p in range(K):
                for j in range(N):
                    acc[j] += a[i][p] * b[j][p]
            result[i] = alpha * acc + beta * c[i]
    else:
        for i in range(m):
            var acc = SIMD[DType.float64, N](0)
            for p in range(K):
                for j in range(N):
                    acc[j] += a[p][i] * b[j][p]
            result[i] = alpha * acc + beta * c[i]

    return result^


# A = [[0, 1], [2, 3]], B = [[5, 6], [7, 8]], expected C = [[7, 8], [31, 36]]
def test_basic_2x2() raises:
    var a = Matrix[2](rows=2)
    a[0] = SIMD[DType.float64, 2](0.0, 1.0)
    a[1] = SIMD[DType.float64, 2](2.0, 3.0)

    var b = Matrix[2](rows=2)
    b[0] = SIMD[DType.float64, 2](5.0, 6.0)
    b[1] = SIMD[DType.float64, 2](7.0, 8.0)

    var c = Matrix[2](rows=2)

    var r = gemm[2, 2](False, False, 2, 1.0, 0.0, a, b, c)
    assert_almost_equal(r[0][0], 7.0)
    assert_almost_equal(r[0][1], 8.0)
    assert_almost_equal(r[1][0], 31.0)
    assert_almost_equal(r[1][1], 36.0)


# alpha=2 should scale every output element by 2
def test_alpha_scaling() raises:
    var a = Matrix[2](rows=2)
    a[0] = SIMD[DType.float64, 2](1.0, 0.0)
    a[1] = SIMD[DType.float64, 2](0.0, 1.0)

    var b = Matrix[2](rows=2)
    b[0] = SIMD[DType.float64, 2](3.0, 4.0)
    b[1] = SIMD[DType.float64, 2](5.0, 6.0)

    var c = Matrix[2](rows=2)

    var r = gemm[2, 2](False, False, 2, 2.0, 0.0, a, b, c)
    assert_almost_equal(r[0][0], 6.0)
    assert_almost_equal(r[0][1], 8.0)
    assert_almost_equal(r[1][0], 10.0)
    assert_almost_equal(r[1][1], 12.0)


# beta=1 should accumulate into an existing C
def test_beta_accumulate() raises:
    var a = Matrix[2](rows=2)
    a[0] = SIMD[DType.float64, 2](1.0, 0.0)
    a[1] = SIMD[DType.float64, 2](0.0, 1.0)

    var b = Matrix[2](rows=2)
    b[0] = SIMD[DType.float64, 2](1.0, 0.0)
    b[1] = SIMD[DType.float64, 2](0.0, 1.0)

    var c = Matrix[2](rows=2)
    c[0] = SIMD[DType.float64, 2](10.0, 20.0)
    c[1] = SIMD[DType.float64, 2](30.0, 40.0)

    var r = gemm[2, 2](False, False, 2, 1.0, 1.0, a, b, c)
    # I*I = I, so result = 1*I + 1*C = I + C
    assert_almost_equal(r[0][0], 11.0)
    assert_almost_equal(r[0][1], 20.0)
    assert_almost_equal(r[1][0], 30.0)
    assert_almost_equal(r[1][1], 41.0)


# trans_a=True: op(A) = A^T
# A = [[1, 2], [3, 4]] stored row-major -> trans gives [[1, 3], [2, 4]]
# B = identity -> result = A^T
def test_trans_a() raises:
    var a = Matrix[2](rows=2)
    a[0] = SIMD[DType.float64, 2](1.0, 2.0)
    a[1] = SIMD[DType.float64, 2](3.0, 4.0)

    var b = Matrix[2](rows=2)
    b[0] = SIMD[DType.float64, 2](1.0, 0.0)
    b[1] = SIMD[DType.float64, 2](0.0, 1.0)

    var c = Matrix[2](rows=2)

    var r = gemm[2, 2](True, False, 2, 1.0, 0.0, a, b, c)
    assert_almost_equal(r[0][0], 1.0)
    assert_almost_equal(r[0][1], 3.0)
    assert_almost_equal(r[1][0], 2.0)
    assert_almost_equal(r[1][1], 4.0)


# trans_b=True: op(B) = B^T
# A = identity, B = [[1, 2], [3, 4]] -> B^T = [[1, 3], [2, 4]]
def test_trans_b() raises:
    var a = Matrix[2](rows=2)
    a[0] = SIMD[DType.float64, 2](1.0, 0.0)
    a[1] = SIMD[DType.float64, 2](0.0, 1.0)

    var b = Matrix[2](rows=2)
    b[0] = SIMD[DType.float64, 2](1.0, 2.0)
    b[1] = SIMD[DType.float64, 2](3.0, 4.0)

    var c = Matrix[2](rows=2)

    var r = gemm[2, 2](False, True, 2, 1.0, 0.0, a, b, c)
    assert_almost_equal(r[0][0], 1.0)
    assert_almost_equal(r[0][1], 3.0)
    assert_almost_equal(r[1][0], 2.0)
    assert_almost_equal(r[1][1], 4.0)


# 1x1 degenerate: scalar multiply
def test_1x1() raises:
    var a = Matrix[1](rows=1)
    a[0] = SIMD[DType.float64, 1](3.0)

    var b = Matrix[1](rows=1)
    b[0] = SIMD[DType.float64, 1](7.0)

    var c = Matrix[1](rows=1)

    var r = gemm[1, 1](False, False, 1, 1.0, 0.0, a, b, c)
    assert_almost_equal(r[0][0], 21.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
