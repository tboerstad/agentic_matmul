struct Matrix:
    """A simple 2D CPU matrix backed by a flat row-major buffer.

    Inspired by NDBuffer but stripped to essentials:
    - Always rank-2 (rows x cols), Float64
    - Row-major layout
    - CPU only
    """

    var data: List[Float64]
    var rows: Int
    var cols: Int

    # --- constructors -----------------------------------------------------------

    fn __init__(out self, rows: Int, cols: Int):
        """Allocate a zero-filled rows x cols matrix."""
        self.rows = rows
        self.cols = cols
        self.data = List[Float64](capacity=rows * cols)
        for _ in range(rows * cols):
            self.data.append(0.0)

    # --- element access ---------------------------------------------------------

    fn __getitem__(self, row: Int, col: Int) -> Float64:
        return self.data[row * self.cols + col]

    fn __setitem__(mut self, row: Int, col: Int, val: Float64):
        self.data[row * self.cols + col] = val

    # --- flat buffer access (for matmul kernels) --------------------------------

    fn load(self, idx: Int) -> Float64:
        return self.data[idx]

    fn store(mut self, idx: Int, val: Float64):
        self.data[idx] = val

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
    # Quick smoke test
    var m = Matrix(2, 3)
    m[0, 0] = 1.0
    m[0, 1] = 2.0
    m[0, 2] = 3.0
    m[1, 0] = 4.0
    m[1, 1] = 5.0
    m[1, 2] = 6.0
    print("Matrix 2x3:")
    m.print()
    print("Element [1,2]:", m[1, 2])
    print("numel:", m.numel())
