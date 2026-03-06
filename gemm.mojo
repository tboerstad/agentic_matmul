fn matmul[*, transpose_b: Bool = False](
    mut c: List[Float64],
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

                comptime if transpose_b:
                    b_val = b[j * k + p]
                else:
                    b_val = b[p * n + j]

                dot += a_val * b_val

            c[i * n + j] = dot
