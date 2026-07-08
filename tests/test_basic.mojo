# Unit tests for matmul_dispatch on small hand-checked shapes.
#
# These shapes all land in the tiny serial branch, so they run in
# milliseconds and make a fast smoke test. The exhaustive per-branch
# coverage lives in test_dispatch.mojo (f64) and test_dtypes.mojo
# (f32/bf16).
#
# Run from the repo root: mojo -I . tests/test_basic.mojo
from matmul import Matrix, matmul_dispatch
from std.testing import assert_almost_equal, TestSuite


# A = [[0, 1], [2, 3]], B = [[5, 6], [7, 8]], expected C = [[7, 8], [31, 36]]
def test_basic_2x2() raises:
    var a = Matrix.from_rows([[0.0, 1.0], [2.0, 3.0]])
    var b = Matrix.from_rows([[5.0, 6.0], [7.0, 8.0]])
    var c = Matrix(2, 2)
    matmul_dispatch(c, a, b)
    assert_almost_equal(c[0, 0], 7.0)
    assert_almost_equal(c[0, 1], 8.0)
    assert_almost_equal(c[1, 0], 31.0)
    assert_almost_equal(c[1, 1], 36.0)


# 1x1 degenerate: scalar multiply
def test_1x1() raises:
    var a = Matrix.from_rows([[3.0]])
    var b = Matrix.from_rows([[7.0]])
    var c = Matrix(1, 1)
    matmul_dispatch(c, a, b)
    assert_almost_equal(c[0, 0], 21.0)


# Non-square: A is 2x3, B is 3x2, C is 2x2
def test_non_square() raises:
    var a = Matrix.from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var b = Matrix.from_rows([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    var c = Matrix(2, 2)
    matmul_dispatch(c, a, b)
    # C[0,0] = 1*7 + 2*9 + 3*11 = 58
    # C[0,1] = 1*8 + 2*10 + 3*12 = 64
    # C[1,0] = 4*7 + 5*9 + 6*11 = 139
    # C[1,1] = 4*8 + 5*10 + 6*12 = 154
    assert_almost_equal(c[0, 0], 58.0)
    assert_almost_equal(c[0, 1], 64.0)
    assert_almost_equal(c[1, 0], 139.0)
    assert_almost_equal(c[1, 1], 154.0)


# float32 storage takes the same tiny serial route.
def test_float32() raises:
    var a = Matrix[DType.float32].from_rows([[1.0, 2.0], [3.0, 4.0]])
    var b = Matrix[DType.float32].from_rows([[5.0, 6.0], [7.0, 8.0]])
    var c = Matrix[DType.float32](2, 2)
    matmul_dispatch(c, a, b)
    assert_almost_equal(c[0, 0], 19.0)
    assert_almost_equal(c[0, 1], 22.0)
    assert_almost_equal(c[1, 0], 43.0)
    assert_almost_equal(c[1, 1], 50.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
