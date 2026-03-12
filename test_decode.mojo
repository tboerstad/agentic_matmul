from decode import matmul_decode
from matrix import Matrix
from std.testing import assert_almost_equal, TestSuite


def test_decode_basic() raises:
    # A = [[1, 2, 3]], B = [[1, 0], [0, 1], [1, 1]], C = [[4, 5]]
    var a = Matrix(1, 3)
    a[0, 0] = 1.0; a[0, 1] = 2.0; a[0, 2] = 3.0
    var b = Matrix(3, 2)
    b[0, 0] = 1.0; b[0, 1] = 0.0
    b[1, 0] = 0.0; b[1, 1] = 1.0
    b[2, 0] = 1.0; b[2, 1] = 1.0
    var c = Matrix(1, 2)
    matmul_decode(c, a, b)
    assert_almost_equal(c[0, 0], 4.0)
    assert_almost_equal(c[0, 1], 5.0)


def test_decode_2x2() raises:
    var a = Matrix(2, 2)
    a[0, 0] = 1.0; a[0, 1] = 2.0; a[1, 0] = 3.0; a[1, 1] = 4.0
    var b = Matrix(2, 2)
    b[0, 0] = 5.0; b[0, 1] = 6.0; b[1, 0] = 7.0; b[1, 1] = 8.0
    var c = Matrix(2, 2)
    matmul_decode(c, a, b)
    assert_almost_equal(c[0, 0], 19.0)
    assert_almost_equal(c[0, 1], 22.0)
    assert_almost_equal(c[1, 0], 43.0)
    assert_almost_equal(c[1, 1], 50.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
