"""CPU architecture detection using Mojo's CompilationTarget.

This shows everything we can detect at compile time to auto-tune GEMM parameters.
"""

from std.sys.info import (
    CompilationTarget,
    num_physical_cores,
    num_logical_cores,
    num_performance_cores,
    simd_bit_width,
    simd_byte_width,
    simd_width_of,
)

comptime target = CompilationTarget


fn main():
    print("========================================")
    print("  Mojo CPU Detection Report")
    print("========================================")

    # ---- Architecture ----
    print("\n--- Architecture ---")
    print("is_x86:", target.is_x86())

    # ---- OS ----
    print("\n--- Operating System ---")
    print("is_linux:", target.is_linux())
    print("is_macos:", target.is_macos())

    # ---- x86 SIMD Feature Flags ----
    print("\n--- x86 SIMD Features ---")
    print("has_sse4:", target.has_sse4())
    print("has_avx:", target.has_avx())
    print("has_avx2:", target.has_avx2())
    print("has_avx512f:", target.has_avx512f())
    print("has_fma:", target.has_fma())
    print("has_vnni:", target.has_vnni())
    print("has_intel_amx:", target.has_intel_amx())

    # ---- ARM Features ----
    print("\n--- ARM Features ---")
    print("has_neon:", target.has_neon())
    print("has_neon_int8_dotprod:", target.has_neon_int8_dotprod())
    print("has_neon_int8_matmul:", target.has_neon_int8_matmul())

    # ---- Apple Silicon ----
    print("\n--- Apple Silicon ---")
    print("is_apple_silicon:", target.is_apple_silicon())
    print("is_apple_m1:", target.is_apple_m1())
    print("is_apple_m2:", target.is_apple_m2())
    print("is_apple_m3:", target.is_apple_m3())
    print("is_apple_m4:", target.is_apple_m4())

    # ---- ARM Server ----
    print("\n--- ARM Server ---")
    print("is_neoverse_n1:", target.is_neoverse_n1())

    # ---- SIMD Widths ----
    print("\n--- SIMD Configuration ---")
    print("simd_bit_width:", simd_bit_width())
    print("simd_byte_width:", simd_byte_width())
    print("simd_width_of[float64]:", simd_width_of[DType.float64]())
    print("simd_width_of[float32]:", simd_width_of[DType.float32]())

    # ---- Core Counts ----
    print("\n--- Core Counts ---")
    print("num_physical_cores:", num_physical_cores())
    print("num_logical_cores:", num_logical_cores())
    print("num_performance_cores:", num_performance_cores())

    # ---- Derived GEMM Parameters ----
    print("\n========================================")
    print("  Auto-tuned GEMM Parameters")
    print("========================================")

    comptime NELTS = simd_width_of[DType.float64]()
    comptime BITS = simd_bit_width()
    print("NELTS (float64/vec):", NELTS)
    print("Vector width (bits):", BITS)

    comptime if target.has_avx512f():
        # AVX-512: 32 zmm registers
        # Microkernel: 8x24 (MR=8, NR=3*NELTS=24)
        # Register budget: 8*3=24 accumulators + 8 A-loads + 3 B-bcasts = 35
        comptime MR = 8
        comptime NR_VECS = 3
        comptime NR = NR_VECS * NELTS  # 24
        comptime KC = 512
        comptime KU = 8
        comptime TILE_N = NR * 3  # 72
        print("\nProfile: AVX-512 (32 zmm registers)")
        print("  MR:", MR)
        print("  NR:", NR, "(", NR_VECS, "vectors)")
        print("  KC:", KC)
        print("  KU:", KU)
        print("  TILE_N:", TILE_N)
        print("  B panel:", KC * NR * 8, "bytes")
    elif target.has_avx2():
        # AVX2: 16 ymm registers, NELTS=4 for float64
        # Microkernel: 6x8 (MR=6, NR=2*NELTS=8)
        # Register budget: 6*2=12 accumulators + 1 A-bcast + 2 B-loads = 15
        comptime MR = 6
        comptime NR_VECS = 2
        comptime NR = NR_VECS * NELTS  # 8
        comptime KC = 256
        comptime KU = 4
        comptime TILE_N = NR * 6  # 48
        print("\nProfile: AVX2 (16 ymm registers)")
        print("  MR:", MR)
        print("  NR:", NR, "(", NR_VECS, "vectors)")
        print("  KC:", KC)
        print("  KU:", KU)
        print("  TILE_N:", TILE_N)
        print("  B panel:", KC * NR * 8, "bytes")
    elif target.has_neon():
        # NEON: 32 v-registers, NELTS=2 for float64
        # Microkernel: 8x12 (MR=8, NR=6*NELTS=12) or 8x6 (NR=3*NELTS=6)
        comptime MR = 8
        comptime NR_VECS = 3
        comptime NR = NR_VECS * NELTS  # 6
        comptime KC = 512
        comptime KU = 4
        comptime TILE_N = NR * 4  # 24
        print("\nProfile: NEON/AArch64 (32 v-registers)")
        print("  MR:", MR)
        print("  NR:", NR, "(", NR_VECS, "vectors)")
        print("  KC:", KC)
        print("  KU:", KU)
        print("  TILE_N:", TILE_N)
        print("  B panel:", KC * NR * 8, "bytes")
    elif target.has_sse4():
        # SSE4: 16 xmm registers, NELTS=2 for float64
        comptime MR = 4
        comptime NR_VECS = 2
        comptime NR = NR_VECS * NELTS  # 4
        comptime KC = 256
        comptime KU = 4
        comptime TILE_N = NR * 4  # 16
        print("\nProfile: SSE4 (16 xmm registers)")
        print("  MR:", MR)
        print("  NR:", NR, "(", NR_VECS, "vectors)")
        print("  KC:", KC)
        print("  KU:", KU)
        print("  TILE_N:", TILE_N)
        print("  B panel:", KC * NR * 8, "bytes")
    else:
        print("\nProfile: Generic (fallback)")
        print("  MR: 4, NR: 4, KC: 256")

    print("\n========================================")
    print("  Summary: What's available at comptime")
    print("========================================")
    print("CompilationTarget methods (all comptime Bool):")
    print("  Architecture: is_x86")
    print("  OS: is_linux, is_macos")
    print("  x86 SIMD: has_sse4, has_avx, has_avx2, has_avx512f, has_fma, has_vnni, has_intel_amx")
    print("  ARM SIMD: has_neon, has_neon_int8_dotprod, has_neon_int8_matmul")
    print("  Apple: is_apple_silicon, is_apple_m1/m2/m3/m4")
    print("  ARM Server: is_neoverse_n1")
    print("Other comptime values:")
    print("  simd_bit_width(), simd_byte_width(), simd_width_of[dtype]()")
    print("Runtime only:")
    print("  num_physical_cores(), num_logical_cores(), num_performance_cores()")
