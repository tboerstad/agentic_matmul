from matrix import Matrix


fn matmul_naive[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    var m = a.rows
    var n = c.cols
    var k = a.cols
    for i in range(m):
        for j in range(n):
            var dot = Scalar[dtype](0)
            for p in range(k):
                var a_val = a[i, p]

                comptime if transpose_b:
                    dot += a_val * b[j, p]
                else:
                    dot += a_val * b[p, j]

            c[i, j] = dot


fn matmul_tiled[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], mut a: Matrix[dtype], mut b: Matrix[dtype]
):
    comptime TILE = 32

    var m = a.rows
    var n = c.cols
    var k = a.cols
    var num_i = (m + TILE - 1) // TILE
    var num_p = (k + TILE - 1) // TILE
    var num_j = (n + TILE - 1) // TILE

    # Zero out C (tiles accumulate with +=)
    for idx in range(m * n):
        c.data[idx] = Scalar[dtype](0)

    for i0 in range(num_i):
        for p0 in range(num_p):
            var a_tile = a.tile(TILE, TILE, i0, p0)
            for j0 in range(num_j):
                var c_tile = c.tile(TILE, TILE, i0, j0)
                comptime if transpose_b:
                    var b_tile = b.tile(TILE, TILE, j0, p0)
                    for i in range(c_tile.rows):
                        for p in range(a_tile.cols):
                            var a_val = a_tile[i, p]
                            for j in range(c_tile.cols):
                                c_tile[i, j] = c_tile[i, j] + a_val * b_tile[j, p]
                else:
                    var b_tile = b.tile(TILE, TILE, p0, j0)
                    for i in range(c_tile.rows):
                        for p in range(a_tile.cols):
                            var a_val = a_tile[i, p]
                            for j in range(c_tile.cols):
                                c_tile[i, j] = c_tile[i, j] + a_val * b_tile[p, j]


fn matmul[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], mut a: Matrix[dtype], mut b: Matrix[dtype]
):
    matmul_tiled[dtype, transpose_b=transpose_b](c, a, b)
