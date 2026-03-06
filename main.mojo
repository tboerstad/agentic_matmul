struct Matrix[cols: Int](Movable):
    """Contiguous row-major matrix with comptime column width.

    Backed by a flat List[Float64] of rows*cols elements.
    Rows are loaded/stored as SIMD[DType.float64, cols] vectors via the
    underlying pointer, so row-wide arithmetic is fully vectorized.
    M (number of rows) is runtime-sized.
    """

    var data: List[Float64]
    var rows: Int

    fn __init__(out self, rows: Int):
        self.rows = rows
        self.data = List[Float64](length=rows * Self.cols, fill=0.0)

    fn __getitem__(self, row: Int) -> SIMD[DType.float64, Self.cols]:
        return (self.data.unsafe_ptr() + row * Self.cols).load[width=Self.cols]()

    fn __setitem__(mut self, row: Int, val: SIMD[DType.float64, Self.cols]):
        (self.data.unsafe_ptr() + row * Self.cols).store(val)


fn matmul2x2(
    a: SIMD[DType.float64, 4],
    b: SIMD[DType.float64, 4],
) -> SIMD[DType.float64, 4]:
    # a and b are row-major 2x2 matrices: [a00, a01, a10, a11]
    var c00 = a[0] * b[0] + a[1] * b[2]
    var c01 = a[0] * b[1] + a[1] * b[3]
    var c10 = a[2] * b[0] + a[3] * b[2]
    var c11 = a[2] * b[1] + a[3] * b[3]
    return SIMD[DType.float64, 4](c00, c01, c10, c11)


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
    """C(m x N) = alpha * op(A) * op(B) + beta * C.

    N and K are comptime — each row is a fixed-width SIMD vector.
    M (number of output rows) is runtime.

    Layout assumptions depending on transpose flags:
      trans_a=False: A is m x K  (a.rows == m)
      trans_a=True:  A is K x m  (a.rows == K), read transposed
      trans_b=False: B is K x N  (b.rows == K)
      trans_b=True:  B is N x K stored; logical B^T is K x N
                     b.rows == N, b[j][p] gives element (j, p)
    """
    var result = Matrix[N](rows=m)

    if not trans_a and not trans_b:
        # Fast path: broadcast A[i][p] * full SIMD row B[p]
        for i in range(m):
            var acc = SIMD[DType.float64, N](0)
            for p in range(K):
                acc += a[i][p] * b[p]
            result[i] = alpha * acc + beta * c[i]
    elif trans_a and not trans_b:
        # A^T[i][p] = A[p][i]; B rows still SIMD
        for i in range(m):
            var acc = SIMD[DType.float64, N](0)
            for p in range(K):
                acc += a[p][i] * b[p]
            result[i] = alpha * acc + beta * c[i]
    elif not trans_a and trans_b:
        # B stored N x K, B^T[p][j] = B[j][p]
        for i in range(m):
            var acc = SIMD[DType.float64, N](0)
            for p in range(K):
                for j in range(N):
                    acc[j] += a[i][p] * b[j][p]
            result[i] = alpha * acc + beta * c[i]
    else:
        # Both transposed
        for i in range(m):
            var acc = SIMD[DType.float64, N](0)
            for p in range(K):
                for j in range(N):
                    acc[j] += a[p][i] * b[j][p]
            result[i] = alpha * acc + beta * c[i]

    return result^


fn main():
    print("Hello from Mojo!")

    # matmul2x2 demo
    # A = [[0, 1], [2, 3]]
    # B = [[5, 6], [7, 8]]
    var a_simd = SIMD[DType.float64, 4](0.0, 1.0, 2.0, 3.0)
    var b_simd = SIMD[DType.float64, 4](5.0, 6.0, 7.0, 8.0)

    var c_simd = matmul2x2(a_simd, b_simd)

    print("A = [[0, 1], [2, 3]]")
    print("B = [[5, 6], [7, 8]]")
    print("C = A * B =")
    print("  [[", c_simd[0], ",", c_simd[1], "],")
    print("   [", c_simd[2], ",", c_simd[3], "]]")

    # gemm demo: same multiplication via Matrix-based GEMM
    var a = Matrix[2](rows=2)
    a[0] = SIMD[DType.float64, 2](0.0, 1.0)
    a[1] = SIMD[DType.float64, 2](2.0, 3.0)

    var b = Matrix[2](rows=2)
    b[0] = SIMD[DType.float64, 2](5.0, 6.0)
    b[1] = SIMD[DType.float64, 2](7.0, 8.0)

    var c = Matrix[2](rows=2)

    var gr = gemm[2, 2](False, False, 2, 1.0, 0.0, a, b, c)

    print("\nGEMM: C = 1.0 * A * B + 0.0 * C")
    print("  [[", gr[0][0], ",", gr[0][1], "],")
    print("   [", gr[1][0], ",", gr[1][1], "]]")
