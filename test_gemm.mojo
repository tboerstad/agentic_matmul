from std.testing import assert_almost_equal, TestSuite


fn gemm(
    trans_a: Bool,
    trans_b: Bool,
    m: Int,
    n: Int,
    k: Int,
    alpha: Float64,
    beta: Float64,
    a: List[Float64],
    lda: Int,
    b: List[Float64],
    ldb: Int,
    c: List[Float64],
    ldc: Int,
) -> List[Float64]:
    var result = List[Float64](capacity=m * ldc)
    for i in range(m * ldc):
        result.append(c[i])

    for i in range(m):
        for j in range(n):
            var dot: Float64 = 0.0
            for p in range(k):
                var a_val: Float64
                if trans_a:
                    a_val = a[p * lda + i]
                else:
                    a_val = a[i * lda + p]

                var b_val: Float64
                if trans_b:
                    b_val = b[j * ldb + p]
                else:
                    b_val = b[p * ldb + j]

                dot += a_val * b_val

            result[i * ldc + j] = alpha * dot + beta * c[i * ldc + j]

    return result^


# A = [[0, 1], [2, 3]], B = [[5, 6], [7, 8]], expected C = [[7, 8], [31, 36]]
def test_basic_2x2() raises:
    var a: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var b: List[Float64] = [5.0, 6.0, 7.0, 8.0]
    var c: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    var r = gemm(False, False, 2, 2, 2, 1.0, 0.0, a, 2, b, 2, c, 2)
    assert_almost_equal(r[0], 7.0)
    assert_almost_equal(r[1], 8.0)
    assert_almost_equal(r[2], 31.0)
    assert_almost_equal(r[3], 36.0)


# alpha=2 should scale every output element by 2
def test_alpha_scaling() raises:
    var a: List[Float64] = [1.0, 0.0, 0.0, 1.0]  # identity
    var b: List[Float64] = [3.0, 4.0, 5.0, 6.0]
    var c: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    var r = gemm(False, False, 2, 2, 2, 2.0, 0.0, a, 2, b, 2, c, 2)
    assert_almost_equal(r[0], 6.0)
    assert_almost_equal(r[1], 8.0)
    assert_almost_equal(r[2], 10.0)
    assert_almost_equal(r[3], 12.0)


# beta=1 should accumulate into an existing C
def test_beta_accumulate() raises:
    var a: List[Float64] = [1.0, 0.0, 0.0, 1.0]  # identity
    var b: List[Float64] = [1.0, 0.0, 0.0, 1.0]  # identity
    var c: List[Float64] = [10.0, 20.0, 30.0, 40.0]
    var r = gemm(False, False, 2, 2, 2, 1.0, 1.0, a, 2, b, 2, c, 2)
    # I*I = I, so result = 1*I + 1*C = I + C
    assert_almost_equal(r[0], 11.0)
    assert_almost_equal(r[1], 20.0)
    assert_almost_equal(r[2], 30.0)
    assert_almost_equal(r[3], 41.0)


# trans_a=True: op(A) = A^T
# A = [[1, 2], [3, 4]] stored row-major -> trans gives [[1, 3], [2, 4]]
# B = identity -> result = A^T
def test_trans_a() raises:
    var a: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var b: List[Float64] = [1.0, 0.0, 0.0, 1.0]  # identity
    var c: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    var r = gemm(True, False, 2, 2, 2, 1.0, 0.0, a, 2, b, 2, c, 2)
    assert_almost_equal(r[0], 1.0)
    assert_almost_equal(r[1], 3.0)
    assert_almost_equal(r[2], 2.0)
    assert_almost_equal(r[3], 4.0)


# trans_b=True: op(B) = B^T
# A = identity, B = [[1, 2], [3, 4]] -> B^T = [[1, 3], [2, 4]]
def test_trans_b() raises:
    var a: List[Float64] = [1.0, 0.0, 0.0, 1.0]  # identity
    var b: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var c: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    var r = gemm(False, True, 2, 2, 2, 1.0, 0.0, a, 2, b, 2, c, 2)
    assert_almost_equal(r[0], 1.0)
    assert_almost_equal(r[1], 3.0)
    assert_almost_equal(r[2], 2.0)
    assert_almost_equal(r[3], 4.0)


# 1x1 degenerate: scalar multiply
def test_1x1() raises:
    var a: List[Float64] = [3.0]
    var b: List[Float64] = [7.0]
    var c: List[Float64] = [0.0]
    var r = gemm(False, False, 1, 1, 1, 1.0, 0.0, a, 1, b, 1, c, 1)
    assert_almost_equal(r[0], 21.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
