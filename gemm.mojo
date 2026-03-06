from matrix import Matrix, make_tile_tensor
from buffer import NDBuffer, DimList
from layout import TileTensor


fn matmul_naive[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    """C = A * op(B) — simple triple-nested loop via TileTensor views."""
    var tt_a = make_tile_tensor(a.ptr, a.rows, a.cols)
    var tt_b = make_tile_tensor(b.ptr, b.rows, b.cols)
    var tt_c = make_tile_tensor(c.ptr, c.rows, c.cols)

    var m = tt_a.dim(0)
    var k = tt_a.dim(1)
    var n = tt_c.dim(1)

    for i in range(m):
        for j in range(n):
            var dot = Scalar[dtype](0)
            for p in range(k):
                comptime if transpose_b:
                    dot += tt_a[i, p] * tt_b[j, p]
                else:
                    dot += tt_a[i, p] * tt_b[p, j]
            tt_c[i, j] = dot


fn matmul_tiled[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    """C = A * op(B) — cache-blocked using TileTensor.slice() for sub-tiles.

    Uses TileTensor abstractions:
      - fill() to zero-initialize C
      - slice() to extract sub-tile views for each block
      - dim() to query tile dimensions (handles edge tiles automatically)
      - 0-based local indexing within each sub-tile
    """
    comptime TILE = 32

    var tt_a = make_tile_tensor(a.ptr, a.rows, a.cols)
    var tt_b = make_tile_tensor(b.ptr, b.rows, b.cols)
    var tt_c = make_tile_tensor(c.ptr, c.rows, c.cols)

    var m = tt_a.dim(0)
    var k = tt_a.dim(1)
    var n = tt_c.dim(1)

    # Zero C via TileTensor fill
    tt_c.fill(Scalar[dtype](0))

    for i0 in range(0, m, TILE):
        var i1 = min(i0 + TILE, m)
        for p0 in range(0, k, TILE):
            var p1 = min(p0 + TILE, k)
            for j0 in range(0, n, TILE):
                var j1 = min(j0 + TILE, n)

                # Slice out sub-tile views — local indices are 0-based
                var a_tile = tt_a.slice((i0, i1), (p0, p1))
                var c_tile = tt_c.slice((i0, i1), (j0, j1))

                var ti = c_tile.dim(0)
                var tp = a_tile.dim(1)
                var tj = c_tile.dim(1)

                comptime if transpose_b:
                    # B is (N, K); slice rows=j, cols=p
                    var b_tile = tt_b.slice((j0, j1), (p0, p1))
                    for i in range(ti):
                        for p in range(tp):
                            var a_val = a_tile[i, p]
                            for j in range(tj):
                                c_tile[i, j] += a_val * b_tile[j, p]
                else:
                    # B is (K, N); slice rows=p, cols=j
                    var b_tile = tt_b.slice((p0, p1), (j0, j1))
                    for i in range(ti):
                        for p in range(tp):
                            var a_val = a_tile[i, p]
                            for j in range(tj):
                                c_tile[i, j] += a_val * b_tile[p, j]


fn matmul[dtype: DType = DType.float64, *, transpose_b: Bool = False](
    mut c: Matrix[dtype], a: Matrix[dtype], b: Matrix[dtype]
):
    matmul_tiled[dtype, transpose_b=transpose_b](c, a, b)
