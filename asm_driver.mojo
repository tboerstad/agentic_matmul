"""Driver to generate assembly for all matmul kernels.
Compile with: mojo build --emit asm -O3 -o gemm.s asm_driver.mojo
"""
from gemm import matmul_naive, matmul_tiled, matmul_simd, matmul_parallel, matmul_register_blocked, matmul_packed
from matrix import Matrix


fn main():
    var a = Matrix(96, 2048)
    var b = Matrix(2048, 11008)
    var c = Matrix(96, 11008)

    matmul_naive(c, a, b)
    matmul_tiled(c, a, b)
    matmul_simd(c, a, b)
    matmul_parallel(c, a, b)
    matmul_register_blocked(c, a, b)
    matmul_packed(c, a, b)
