from std.collections import List
from tile import Tile


struct Matrix[dtype: DType = DType.float64](Movable):
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

    def __init__(out self, rows: Int, cols: Int):
        """Allocate a zero-filled rows x cols matrix."""
        self = Self(rows, cols, fill=Scalar[Self.dtype](0))

    def __init__(out self, rows: Int, cols: Int, *, fill: Scalar[Self.dtype]):
        """Allocate a rows x cols matrix, every element set to `fill`."""
        self.rows = rows
        self.cols = cols
        self.data = List[Scalar[Self.dtype]](capacity=rows * cols)
        for _ in range(rows * cols):
            self.data.append(fill)

    @staticmethod
    def from_rows(rows: List[List[Scalar[Self.dtype]]]) -> Self:
        """Build a matrix from a list of equal-length rows — for tests and demos
        that would otherwise assign element by element."""
        var r = len(rows)
        var c = len(rows[0]) if r > 0 else 0
        var m = Self(r, c)
        for i in range(r):
            for j in range(c):
                m.data[i * c + j] = rows[i][j]
        return m^

    # --- element access ---------------------------------------------------------

    def __getitem__(self, row: Int, col: Int) -> Scalar[Self.dtype]:
        return self.data[row * self.cols + col]

    def __setitem__(mut self, row: Int, col: Int, val: Scalar[Self.dtype]):
        self.data[row * self.cols + col] = val

    # --- flat buffer access (for matmul kernels) --------------------------------

    def load(self, idx: Int) -> Scalar[Self.dtype]:
        return self.data[idx]

    def store(mut self, idx: Int, val: Scalar[Self.dtype]):
        self.data[idx] = val

    # --- views ------------------------------------------------------------------

    def view(ref self) -> Tile[Self.dtype, origin_of(self.data)]:
        """The whole matrix as a `Tile` (row-major, stride = cols). A convenience
        for callers outside the hot path; the kernels build their Tiles from the
        `as_noalias_ptr()` local instead, to keep the noalias the loops rely on."""
        return Tile(self.data.unsafe_ptr(), self.rows, self.cols, self.cols)

    # --- properties -------------------------------------------------------------

    def numel(self) -> Int:
        return self.rows * self.cols

    # --- display ----------------------------------------------------------------

    def print(self):
        for i in range(self.rows):
            var line = String("[")
            for j in range(self.cols):
                if j > 0:
                    line += ", "
                line += String(self.data[i * self.cols + j])
            line += "]"
            print(line)


def main():
    # float64 (default), built row-by-row
    var m = Matrix.from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("Matrix[float64] 2x3:")
    m.print()

    # float32
    var m32 = Matrix[DType.float32].from_rows([[1.0, 2.0], [3.0, 4.0]])
    print("Matrix[float32] 2x2:")
    m32.print()
