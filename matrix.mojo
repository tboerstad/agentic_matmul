from std.memory import UnsafePointer


struct TileTensor[dtype: DType = DType.float64, origin: MutOrigin = MutAnyOrigin]:
    """A lightweight view into a rectangular sub-block of a Matrix.

    Provides local (row, col) indexing within the tile while mapping
    to the correct position in the parent matrix's storage.  Boundary
    tiles are automatically clamped so rows/cols never exceed the
    parent dimensions.
    """

    var _ptr: UnsafePointer[Scalar[Self.dtype], origin=Self.origin]
    var rows: Int
    var cols: Int
    var _stride: Int

    fn __init__(out self, ptr: UnsafePointer[Scalar[Self.dtype], origin=Self.origin], rows: Int, cols: Int, stride: Int):
        self._ptr = ptr
        self.rows = rows
        self.cols = cols
        self._stride = stride

    fn __getitem__(self, row: Int, col: Int) -> Scalar[Self.dtype]:
        return self._ptr[row * self._stride + col]

    fn __setitem__(self, row: Int, col: Int, val: Scalar[Self.dtype]):
        self._ptr[row * self._stride + col] = val


struct Matrix[dtype: DType = DType.float64]:
    """A simple 2D CPU matrix backed by a flat row-major buffer."""

    var data: List[Scalar[Self.dtype]]
    var rows: Int
    var cols: Int

    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = List[Scalar[Self.dtype]](capacity=rows * cols)
        for _ in range(rows * cols):
            self.data.append(Scalar[Self.dtype](0))

    fn __getitem__(self, row: Int, col: Int) -> Scalar[Self.dtype]:
        return self.data[row * self.cols + col]

    fn __setitem__(mut self, row: Int, col: Int, val: Scalar[Self.dtype]):
        self.data[row * self.cols + col] = val

    fn tile(mut self, tile_h: Int, tile_w: Int, i: Int, j: Int) -> TileTensor[Self.dtype]:
        """Return a TileTensor view of the (i, j)-th tile of size tile_h x tile_w.

        Boundary tiles are clamped to the matrix dimensions.
        """
        var row0 = i * tile_h
        var col0 = j * tile_w
        return TileTensor[Self.dtype](
            self.data.unsafe_ptr() + row0 * self.cols + col0,
            min(tile_h, self.rows - row0),
            min(tile_w, self.cols - col0),
            self.cols,
        )

    fn numel(self) -> Int:
        return self.rows * self.cols

    fn print(self):
        for i in range(self.rows):
            var line = String("[")
            for j in range(self.cols):
                if j > 0:
                    line += ", "
                line += String(self.data[i * self.cols + j])
            line += "]"
            print(line)
