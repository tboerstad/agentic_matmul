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


fn main():
    print("Hello from Mojo!")

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
