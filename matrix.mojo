struct Matrix[dtype: DType = DType.float64]:
    """A simple 2D CPU matrix backed by a flat row-major buffer.

    Inspired by NDBuffer but stripped to essentials:
    - Always rank-2 (rows x cols)
    - Parameterized dtype (defaults to float64)
    - Row-major layout
    - CPU only
    """

    var data: List[Scalar[Self.dtype]]
    var rows: Int
    var cols: Int

    # --- constructors -----------------------------------------------------------

    fn __init__(out self, rows: Int, cols: Int):
        """Allocate a zero-filled rows x cols matrix."""
        self.rows = rows
        self.cols = cols
        self.data = List[Scalar[Self.dtype]](capacity=rows * cols)
        for _ in range(rows * cols):
            self.data.append(Scalar[Self.dtype](0))

    # --- element access ---------------------------------------------------------

    fn __getitem__(self, row: Int, col: Int) -> Scalar[Self.dtype]:
        return self.data[row * self.cols + col]

    fn __setitem__(mut self, row: Int, col: Int, val: Scalar[Self.dtype]):
        self.data[row * self.cols + col] = val

    # --- flat buffer access (for matmul kernels) --------------------------------

    fn load(self, idx: Int) -> Scalar[Self.dtype]:
        return self.data[idx]

    fn store(mut self, idx: Int, val: Scalar[Self.dtype]):
        self.data[idx] = val

    # --- SIMD vector access ----------------------------------------------------

    fn simd_load[width: Int](self, idx: Int) -> SIMD[Self.dtype, width]:
        """Load `width` contiguous elements as a SIMD vector via pointer."""
        return (self.data.unsafe_ptr() + idx).load[width=width]()

    fn simd_store[width: Int](mut self, idx: Int, val: SIMD[Self.dtype, width]):
        """Store a SIMD vector of `width` elements via pointer."""
        (self.data.unsafe_ptr() + idx).store(val)

    # --- properties -------------------------------------------------------------

    fn numel(self) -> Int:
        return self.rows * self.cols

    # --- display ----------------------------------------------------------------

    fn print(self):
        for i in range(self.rows):
            var line = String("[")
            for j in range(self.cols):
                if j > 0:
                    line += ", "
                line += String(self.data[i * self.cols + j])
            line += "]"
            print(line)


fn main():
    # float64 (default)
    var m = Matrix(2, 3)
    m[0, 0] = 1.0
    m[0, 1] = 2.0
    m[0, 2] = 3.0
    m[1, 0] = 4.0
    m[1, 1] = 5.0
    m[1, 2] = 6.0
    print("Matrix[float64] 2x3:")
    m.print()

    # float32
    var m32 = Matrix[DType.float32](2, 2)
    m32[0, 0] = 1.0
    m32[0, 1] = 2.0
    m32[1, 0] = 3.0
    m32[1, 1] = 4.0
    print("Matrix[float32] 2x2:")
    m32.print()
