from memory import UnsafePointer, memset_zero
from buffer import NDBuffer, DimList
from layout import TileTensor


struct Matrix[dtype: DType = DType.float64]:
    """A 2D matrix backed by UnsafePointer with TileTensor views for computation.

    - Always rank-2 (rows x cols)
    - Parameterized dtype (defaults to float64)
    - Row-major layout via TileTensor from max
    - CPU only
    """

    var ptr: UnsafePointer[Scalar[Self.dtype]]
    var rows: Int
    var cols: Int

    # --- constructors -----------------------------------------------------------

    fn __init__(out self, rows: Int, cols: Int):
        """Allocate a zero-filled rows x cols matrix."""
        self.rows = rows
        self.cols = cols
        self.ptr = UnsafePointer[Scalar[Self.dtype]].alloc(rows * cols)
        memset_zero(self.ptr, rows * cols)

    fn __copyinit__(out self, other: Self):
        self.rows = other.rows
        self.cols = other.cols
        var n = self.rows * self.cols
        self.ptr = UnsafePointer[Scalar[Self.dtype]].alloc(n)
        for i in range(n):
            self.ptr[i] = other.ptr[i]

    fn __moveinit__(out self, owned other: Self):
        self.rows = other.rows
        self.cols = other.cols
        self.ptr = other.ptr
        other.ptr = UnsafePointer[Scalar[Self.dtype]]()

    fn __del__(owned self):
        if self.ptr:
            self.ptr.free()

    # --- element access via TileTensor ------------------------------------------

    fn __getitem__(self, row: Int, col: Int) -> Scalar[Self.dtype]:
        return self.ptr[row * self.cols + col]

    fn __setitem__(mut self, row: Int, col: Int, val: Scalar[Self.dtype]):
        self.ptr[row * self.cols + col] = val

    # --- flat buffer access (for matmul kernels) --------------------------------

    fn load(self, idx: Int) -> Scalar[Self.dtype]:
        return self.ptr[idx]

    fn store(mut self, idx: Int, val: Scalar[Self.dtype]):
        self.ptr[idx] = val

    # --- TileTensor view --------------------------------------------------------

    fn tile_tensor(self) -> TileTensor[mut=False, dtype=Self.dtype]:
        """Return an immutable TileTensor view over this matrix's data."""
        var buf = NDBuffer[Self.dtype, 2](self.ptr, DimList(self.rows, self.cols))
        return TileTensor(buf)

    fn mut_tile_tensor(mut self) -> TileTensor[mut=True, dtype=Self.dtype]:
        """Return a mutable TileTensor view over this matrix's data."""
        var buf = NDBuffer[Self.dtype, 2](self.ptr, DimList(self.rows, self.cols))
        return TileTensor(buf)

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
                line += String(self.ptr[i * self.cols + j])
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
