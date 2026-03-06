from matrix import Matrix


fn matmul[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype],
    a: Matrix[dtype],
    b: Matrix[dtype],
):
    # Computes C = A * op(B)
    # op(B) = B if transpose_b == False, B^T if transpose_b == True
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


fn main():
    print("Hello from Mojo!")

    # matmul demo
    # A = [[0, 1], [2, 3]], B = [[5, 6], [7, 8]]
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
    c.print()
