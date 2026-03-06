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
    # Computes C = alpha * op(A) * op(B) + beta * C
    # op(X) = X if trans == False, X^T if trans == True
    # All matrices are row-major flat buffers.
    var result = List[Float64](capacity=m * ldc)
    for i in range(m * ldc):
        result.append(c[i])

    for i in range(m):
        for j in range(n):
            var dot: Float64 = 0.0
            for p in range(k):
                var a_val: Float64
                if trans_a:
                    a_val = a[p * lda + i]  # A^T: row p, col i -> A[p][i]
                else:
                    a_val = a[i * lda + p]  # A: row i, col p

                var b_val: Float64
                if trans_b:
                    b_val = b[j * ldb + p]  # B^T: row j, col p -> B[j][p]
                else:
                    b_val = b[p * ldb + j]  # B: row p, col j

                dot += a_val * b_val

            result[i * ldc + j] = alpha * dot + beta * c[i * ldc + j]

    return result


fn main():
    print("Hello from Mojo!")

    # matmul2x2 demo
    # A = [[0, 1], [2, 3]]
    # B = [[5, 6], [7, 8]]
    var a = SIMD[DType.float64, 4](0.0, 1.0, 2.0, 3.0)
    var b = SIMD[DType.float64, 4](5.0, 6.0, 7.0, 8.0)

    var c = matmul2x2(a, b)

    print("A = [[0, 1], [2, 3]]")
    print("B = [[5, 6], [7, 8]]")
    print("C = A * B =")
    print("  [[", c[0], ",", c[1], "],")
    print("   [", c[2], ",", c[3], "]]")

    # gemm demo: same multiplication via GEMM
    # C = 1.0 * A * B + 0.0 * C
    var ga = List[Float64](0.0, 1.0, 2.0, 3.0)
    var gb = List[Float64](5.0, 6.0, 7.0, 8.0)
    var gc = List[Float64](0.0, 0.0, 0.0, 0.0)

    var gr = gemm(False, False, 2, 2, 2, 1.0, 0.0, ga, 2, gb, 2, gc, 2)

    print("\nGEMM: C = 1.0 * A * B + 0.0 * C")
    print("  [[", gr[0], ",", gr[1], "],")
    print("   [", gr[2], ",", gr[3], "]]")
