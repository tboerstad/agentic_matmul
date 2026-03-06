from matrix import Matrix


fn matmul_naive[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    # Computes C = A * op(B)  —  simple triple-nested loop (ijk order).
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
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    # Computes C = A * op(B)  —  tiled / cache-blocked version.
    #
    # Key optimizations over naive:
    #   1. Loop tiling: process TILE x TILE sub-blocks so data fits in L1/L2 cache
    #   2. Accumulate into a local register variable before writing back to C
    #   3. Loop order i→p→j inside tiles keeps A[i,p] reads sequential and
    #      reuses each loaded A element across the full j-tile
    comptime TILE = 32

    var m = a.rows
    var n = c.cols
    var k = a.cols

    # Zero out C (tiles accumulate with +=)
    for idx in range(m * n):
        c.store(idx, Scalar[dtype](0))

    # Tile over all three dimensions
    for i0 in range(0, m, TILE):
        var i_end = i0 + TILE
        if i_end > m:
            i_end = m
        for p0 in range(0, k, TILE):
            var p_end = p0 + TILE
            if p_end > k:
                p_end = k
            for j0 in range(0, n, TILE):
                var j_end = j0 + TILE
                if j_end > n:
                    j_end = n

                # Micro-kernel: multiply the (i0:i_end, p0:p_end) block of A
                # with the (p0:p_end, j0:j_end) block of B, accumulating into C
                for i in range(i0, i_end):
                    for p in range(p0, p_end):
                        var a_val = a[i, p]
                        for j in range(j0, j_end):
                            comptime if transpose_b:
                                c[i, j] = c[i, j] + a_val * b[j, p]
                            else:
                                c[i, j] = c[i, j] + a_val * b[p, j]


# Default matmul points to the tiled version
fn matmul[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    matmul_tiled[dtype, transpose_b=transpose_b](c, a, b)
