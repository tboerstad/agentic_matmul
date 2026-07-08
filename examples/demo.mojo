# A minimal tour of the library: build matrices, multiply, print.
#
# Also the post-setup smoke test: if this runs, the toolchain and the
# package are wired up correctly.
#
# Run from the repo root: mojo -I . examples/demo.mojo
from matmul import Matrix, matmul_dispatch


def main():
    # A = [[0, 1], [2, 3]], B = [[5, 6], [7, 8]]
    var a = Matrix.from_rows([[0.0, 1.0], [2.0, 3.0]])
    var b = Matrix.from_rows([[5.0, 6.0], [7.0, 8.0]])
    var c = Matrix(2, 2)

    matmul_dispatch(c, a, b)

    print("A =")
    a.print()
    print("B =")
    b.print()
    print("C = A * B =")
    c.print()

    # The same entry point handles every dtype and shape; a large shape
    # routes to the packed parallel kernels instead of the tiny serial one.
    var big_a = Matrix[DType.float32](96, 2048, fill=0.5)
    var big_b = Matrix[DType.float32](2048, 1024, fill=0.25)
    var big_c = Matrix[DType.float32](96, 1024)
    matmul_dispatch(big_c, big_a, big_b)
    print("96x2048 @ 2048x1024 f32, C[0,0] =", big_c[0, 0], "(expect 256.0)")
