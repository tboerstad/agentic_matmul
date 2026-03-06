from matrix import Matrix
from buffer import NDBuffer, DimList
from layout import TileTensor


fn matmul_naive[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    # Computes C = A * op(B)  —  simple triple-nested loop (ijk order).
    # Uses TileTensor views over the raw matrix data for element access.
    var m = a.rows
    var n = c.cols
    var k = a.cols

    var tt_a = TileTensor(NDBuffer[dtype, 2](a.ptr, DimList(m, k)))
    var tt_b = TileTensor(NDBuffer[dtype, 2](b.ptr, DimList(b.rows, b.cols)))
    var tt_c = TileTensor(NDBuffer[dtype, 2](c.ptr, DimList(m, n)))

    for i in range(m):
        for j in range(n):
            var dot = Scalar[dtype](0)
            for p in range(k):
                var a_val = tt_a[i, p]

                comptime if transpose_b:
                    dot += a_val * tt_b[j, p]
                else:
                    dot += a_val * tt_b[p, j]

            tt_c[i, j] = dot


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
    #
    # Uses TileTensor views for element access.
    comptime TILE = 32

    var m = a.rows
    var n = c.cols
    var k = a.cols

    var tt_a = TileTensor(NDBuffer[dtype, 2](a.ptr, DimList(m, k)))
    var tt_b = TileTensor(NDBuffer[dtype, 2](b.ptr, DimList(b.rows, b.cols)))
    var tt_c = TileTensor(NDBuffer[dtype, 2](c.ptr, DimList(m, n)))

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
                        var a_val = tt_a[i, p]
                        for j in range(j0, j_end):
                            comptime if transpose_b:
                                tt_c[i, j] = tt_c[i, j] + a_val * tt_b[j, p]
                            else:
                                tt_c[i, j] = tt_c[i, j] + a_val * tt_b[p, j]


# Default matmul points to the tiled version
fn matmul[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    matmul_tiled[dtype, transpose_b=transpose_b](c, a, b)
