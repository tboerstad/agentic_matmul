from matrix import Matrix
from gemm import matmul


fn main():
    print("Hello from Mojo!")

    # matmul demo: A = [[0, 1], [2, 3]], B = [[5, 6], [7, 8]]
    var a = Matrix(2, 2)
    a[0, 0] = 0.0; a[0, 1] = 1.0
    a[1, 0] = 2.0; a[1, 1] = 3.0

    var b = Matrix(2, 2)
    b[0, 0] = 5.0; b[0, 1] = 6.0
    b[1, 0] = 7.0; b[1, 1] = 8.0

    var c = Matrix(2, 2)
    matmul(c, a, b)

    print("A = [[0, 1], [2, 3]]")
    print("B = [[5, 6], [7, 8]]")
    print("C = A * B =")
    print("  [[", c[0, 0], ",", c[0, 1], "],")
    print("   [", c[1, 0], ",", c[1, 1], "]]")
