fn matmul[*, transpose_b: Bool = False](
    inout c: List[Float64],
    a: List[Float64],
    b: List[Float64],
    m: Int,
    n: Int,
    k: Int,
):
    # Computes C = A * op(B)
    # op(B) = B if transpose_b == False, B^T if transpose_b == True
    # All matrices are row-major flat buffers.
    for i in range(m):
        for j in range(n):
            var dot: Float64 = 0.0
            for p in range(k):
                var a_val = a[i * k + p]

                var b_val: Float64

                @parameter
                if transpose_b:
                    b_val = b[j * k + p]
                else:
                    b_val = b[p * n + j]

                dot += a_val * b_val

            c[i * n + j] = dot


fn main():
    print("Hello from Mojo!")

    # matmul demo
    # A = [[0, 1], [2, 3]], B = [[5, 6], [7, 8]]
    var a = List[Float64](0.0, 1.0, 2.0, 3.0)
    var b = List[Float64](5.0, 6.0, 7.0, 8.0)
    var c = List[Float64](0.0, 0.0, 0.0, 0.0)

    matmul(c, a, b, m=2, n=2, k=2)

    print("A = [[0, 1], [2, 3]]")
    print("B = [[5, 6], [7, 8]]")
    print("C = A * B =")
    print("  [[", c[0], ",", c[1], "],")
    print("   [", c[2], ",", c[3], "]]")
