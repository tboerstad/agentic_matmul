# Assembly Verification of GEMM Optimizations

Generated with `mojo build --emit asm -o gemm.s asm_driver.mojo` (default -O3).
Target: x86-64 with AVX-512 (512-bit zmm registers, 8 doubles/vector).

## Summary Table

| Kernel | SIMD Width | FMA Type | Register Blocking | Parallelized | Accumulation |
|--------|-----------|----------|-------------------|-------------|--------------|
| naive | scalar (xmm) | `vfmadd231sd` | No | No | Scalar dot |
| tiled | scalar (xmm) | `vfmadd213sd` | No | No | Scalar per element |
| simd | 512-bit (zmm) | `vfmadd213pd` | No | No | Load/FMA/store per k |
| parallel | 512-bit (zmm) | `vfmadd213pd` | No | Yes | Load/FMA/store per k |
| register_blocked | 512-bit (zmm) | `vfmadd231pd` x4 | MR=4 (4 C rows) | Yes | Load/FMA/store per k |
| packed | 512-bit (zmm) | `vfmadd231pd` x4 | MR=4 (4 C rows) | Yes | Register-resident across k |

---

## 1. Naive Kernel (lines ~147-153 in gemm.s)

**Inner loop (`.LBB0_17`):**
```asm
vmovsd   (%rdx,%r10,8), %xmm1        # scalar load: A[i,p]
vfmadd231sd (%r11), %xmm1, %xmm0     # dot += A[i,p] * B[p,j]  (scalar FMA)
addq     $88064, %r11                 # stride B by N*8 = 11008*8
incq     %r10
cmpq     $2048, %r10                  # k = 2048
jne      .LBB0_17
vmovsd   %xmm0, (%rsi,%r9,8)         # store dot product to C[i,j]
```

**Verdict: CORRECT** - Pure scalar operations (`sd` = scalar double). No SIMD as expected.
The loop counts to 2048 (K dimension). Each iteration does exactly 1 FMA.
Stride of 88064 = 11008 * 8 bytes confirms column-striding through B in row-major layout.

---

## 2. Tiled Kernel (lines ~207-218 in gemm.s)

**Inner loop (`.LBB0_26`):**
```asm
vmovsd   (%r10,%rdi,8), %xmm1            # scalar load: B[p, j0+j]
vfmadd213sd (%r13,%rdi,8), %xmm0, %xmm1  # C[i,j] + A[i,p]*B[p,j]
vmovsd   %xmm1, (%r13,%rdi,8)            # store back to C[i,j]
incq     %rdi
cmpq     $32, %rdi                        # TILE = 32
jne      .LBB0_26
```

**Verdict: CORRECT but NOT AUTO-VECTORIZED** - Still scalar (`sd`) operations.
The compiler did not auto-vectorize this loop. The tile size 32 and sequential
j-loop (stride 1) are present. Cache blocking is structurally correct (inner
loop counts to 32 = TILE), but without explicit SIMD this only benefits from
improved cache locality, not vectorization.

The loop order i->p->j is confirmed: A[i,p] is loaded once (in `%xmm0` before
the j-loop at `.LBB0_25`) and reused across all 32 j iterations. This is the
intended broadcast-and-reuse pattern.

---

## 3. SIMD Kernel (lines ~287-298 in gemm.s)

**Inner loop (`.LBB0_38`):**
```asm
vbroadcastsd (%rbp,%rax,8), %zmm0        # broadcast A[i,p] -> all 8 lanes
...
vmovupd  64(%r15,%r8,8), %zmm1           # 512-bit load: 8 doubles from B[p, j0:j0+8]
vfmadd213pd 64(%r13,%r8,8), %zmm0, %zmm1 # C[i, j0:j0+8] += A[i,p] * B[p, j0:j0+8]
vmovupd  %zmm1, 64(%r13,%r8,8)           # 512-bit store back to C
addq     $8, %r8                          # advance by 8 doubles (NELTS)
cmpq     $24, %r8                         # loop bound (processes 32 elts: 4 iters of 8)
jb       .LBB0_38
```

**Verdict: CORRECT** - Full 512-bit AVX-512 vectorization confirmed.
- `vbroadcastsd` broadcasts scalar A[i,p] across all 8 lanes of zmm0.
- `vmovupd` + `vfmadd213pd` operate on 8 doubles simultaneously (512-bit).
- The inner loop does 4 iterations (0, 8, 16 → cmpq $24) processing 32 elements = TILE.
- `vectorize[NELTS]()` correctly generated the SIMD loop with remainder handling.

---

## 4. Parallel Kernel (process_i_tile, lines ~2496-2828 in gemm.s)

**Parallelization confirmed:**
- `process_i_tile` is emitted as a **separate function** (not inlined into main).
- `KGEN_CompilerRT_AsyncRT_Execute@PLT` dispatches tasks to the thread pool.
- `KGEN_CompilerRT_NumPhysicalCores@PLT` queries core count at runtime.
- `KGEN_CompilerRT_AsyncRT_Wait@PLT` joins all tasks.

**SIMD inner loop (`.LBB10_35` / `.LBB10_28`):**
```asm
vbroadcastsd (%r8,%r12,8), %zmm0          # broadcast A[i,p]
...
vmovupd  (%r12,%r8,8), %zmm1              # 512-bit load from B
vfmadd213pd (%rcx,%r8,8), %zmm0, %zmm1    # C += A * B (8 doubles)
vmovupd  %zmm1, (%rcx,%r8,8)              # 512-bit store
addq     $8, %r8
cmpq     %rdi, %r8                         # tile_n iterations
jl       .LBB10_35
```

**Scalar remainder (`.LBB10_20`):**
```asm
vmovsd   (%r11,%r8,8), %xmm1
vfmadd213sd (%rcx,%r8,8), %xmm0, %xmm1   # scalar FMA for remainders
```

**Verdict: CORRECT** - Identical SIMD micro-kernel to `matmul_simd`, now dispatched
across threads via the async runtime. Scalar tail loop handles non-multiple-of-8 remainders.

---

## 5. Register-Blocked Kernel (process_i_tile, lines ~3227-3718 in gemm.s)

**MR=4 SIMD inner loop (`.LBB15_32`):**
```asm
# A values for 4 rows loaded into xmm0-3 then broadcast to zmm4-7:
vbroadcastsd %xmm0, %zmm4    # A[i+0, p] broadcast
vbroadcastsd %xmm1, %zmm5    # A[i+1, p] broadcast
vbroadcastsd %xmm2, %zmm6    # A[i+2, p] broadcast
vbroadcastsd %xmm3, %zmm7    # A[i+3, p] broadcast
...
vmovupd  (%rcx,%r9,8), %zmm8            # load B[p, j:j+8]  <-- loaded ONCE
vmovupd  (%r8,%r9,8), %zmm9             # load C[i+0, j:j+8]
vfmadd231pd %zmm8, %zmm4, %zmm9         # C[i+0] += A[i+0,p] * B[p,j]
vmovupd  %zmm9, (%r8,%r9,8)             # store C[i+0]
vmovupd  (%r11,%r9,8), %zmm9            # load C[i+1, j:j+8]
vfmadd231pd %zmm8, %zmm5, %zmm9         # C[i+1] += A[i+1,p] * B[p,j]
vmovupd  %zmm9, (%r11,%r9,8)            # store C[i+1]
vmovupd  (%rbp,%r9,8), %zmm9            # load C[i+2, j:j+8]
vfmadd231pd %zmm8, %zmm6, %zmm9         # C[i+2] += A[i+2,p] * B[p,j]
vmovupd  %zmm9, (%rbp,%r9,8)            # store C[i+2]
vfmadd213pd (%r14,%r9,8), %zmm7, %zmm8  # C[i+3] += A[i+3,p] * B[p,j]
vmovupd  %zmm8, (%r14,%r9,8)            # store C[i+3]
addq     $8, %r9
cmpq     %rbx, %r9
jl       .LBB15_32
```

**Scalar MR=4 remainder (`.LBB15_16`):**
```asm
# Same 4-row pattern but with scalar xmm operations for remainder elements
vfmadd231sd %xmm0, %xmm4, %xmm5     # row 0
vfmadd231sd %xmm1, %xmm4, %xmm5     # row 1
vfmadd231sd %xmm2, %xmm4, %xmm5     # row 2
vfmadd213sd (%r14,%rcx,8), %xmm3, %xmm4  # row 3
```

**Verdict: CORRECT** - MR=4 register blocking verified. B vector (`zmm8`) is loaded
once and reused across all 4 FMA instructions for rows 0-3. This gives 4x B-reuse
as designed. The compiler even optimized the last row to use `vfmadd213pd` (fused
load-from-memory) instead of a separate load+FMA pair.

**Issue: C vectors are still loaded/stored every k-iteration.** Each of the 4 C rows
does a load-FMA-store per p iteration. This is the expected behavior for this kernel —
the packed kernel is supposed to fix this.

---

## 6. Packed Kernel (process_i_tile, lines ~4118-4527 in gemm.s)

**MR=4 register-accumulation inner loop (`.LBB20_27`):**
```asm
# C accumulators loaded ONCE before the k-loop:
vmovupd  (%r10,%rcx,8), %zmm3   # acc0 = C[i+0, j:j+8]
vmovupd  (%rbp,%rcx,8), %zmm2   # acc1 = C[i+1, j:j+8]
vmovupd  (%r13,%rcx,8), %zmm1   # acc2 = C[i+2, j:j+8]
vmovupd  (%r14,%rcx,8), %zmm0   # acc3 = C[i+3, j:j+8]
...
# Inner k-loop — C stays in registers:
.LBB20_27:
    vmovupd  (%rax), %zmm4                     # load B[p, j:j+8]
    vfmadd231pd (%r8,%rdi,8){1to8}, %zmm4, %zmm3  # acc0 += A[i+0,p] * B  (embedded broadcast!)
    vfmadd231pd (%r10,%rdi,8){1to8}, %zmm4, %zmm2  # acc1 += A[i+1,p] * B
    vfmadd231pd (%r12,%rdi,8){1to8}, %zmm4, %zmm1  # acc2 += A[i+2,p] * B
    vfmadd231pd (%rdx,%rdi,8){1to8}, %zmm4, %zmm0  # acc3 += A[i+3,p] * B
    incq     %rdi
    addq     %rsi, %rax                         # advance B pointer by row stride
    cmpq     %rdi, %r15                          # loop over tile_k
    jne      .LBB20_27
# C accumulators stored ONCE after k-loop:
vmovupd  %zmm3, (%r10,%rcx,8)   # store acc0 -> C[i+0]
vmovupd  %zmm2, (%rbp,%rcx,8)   # store acc1 -> C[i+1]
vmovupd  %zmm1, (%r13,%rcx,8)   # store acc2 -> C[i+2]
vmovupd  %zmm0, (%r14,%rcx,8)   # store acc3 -> C[i+3]
```

**Verdict: CORRECT and EXCELLENT** - This is the most optimized inner loop.

Key observations:
1. **Register accumulation confirmed**: C vectors (`zmm0-zmm3`) are loaded before
   the k-loop and stored after — not touched during the k-iterations. For TILE=32,
   this eliminates 31 of 32 load/store pairs per tile = **~32x reduction in C traffic**.

2. **AVX-512 embedded broadcast** (`{1to8}`): The compiler used AVX-512's embedded
   broadcast addressing mode — `vfmadd231pd (%r8,%rdi,8){1to8}, %zmm4, %zmm3` loads
   a scalar A[i,p] from memory and broadcasts it to all 8 zmm lanes **within the FMA
   instruction itself**. This eliminates the separate `vbroadcastsd` instruction,
   saving 4 instructions per k-iteration vs the register-blocked kernel.

3. **Tight inner loop**: The k-loop body is only 7 instructions: 1 B-load + 4 FMAs
   (with embedded broadcast) + 1 increment + 1 compare/branch. This is very close
   to optimal.

4. **Register pressure**: Uses zmm0-zmm4 (5 zmm registers) for the hot loop:
   4 accumulators + 1 B vector. Well within the 32 zmm register budget.

---

## Cross-Kernel Findings

### What's Working Well

1. **Progressive SIMD widening**: scalar (naive/tiled) -> 512-bit zmm (simd onward).
2. **AVX-512 FMA**: All vectorized kernels use fused multiply-add (`vfmadd2xxpd`),
   achieving 2 FLOPS/element/cycle (vs 1 with separate mul+add).
3. **Embedded broadcast in packed kernel**: The compiler exploited AVX-512's `{1to8}`
   memory broadcast, fusing the A-scalar broadcast into the FMA instruction.
4. **Parallelization**: The async runtime correctly dispatches `process_i_tile` as
   independent tasks with no synchronization needed.
5. **Register blocking (MR=4)**: B vector loaded once, reused 4 times confirmed.
6. **Register accumulation**: C accumulators stay in zmm registers across the full
   k-tile loop in the packed kernel.

### Issues and Missed Optimizations

1. **No software prefetch instructions**: Zero `prefetch` instructions in the entire
   binary. For the prefill shape (96x11008x2048), data significantly exceeds L2
   cache. Adding `prefetcht0`/`prefetcht1` for upcoming B-tiles could improve
   performance by hiding memory latency.

2. **Tiled kernel not auto-vectorized**: The tiled kernel (`matmul_tiled`) uses
   scalar operations despite having a contiguous j-loop. The compiler chose not to
   auto-vectorize. Only the explicit `vectorize[NELTS]()` call in `matmul_simd`
   triggers SIMD codegen. This is expected but means the tiled->simd performance
   jump is entirely due to SIMD, not just a `vectorize` overhead.

3. **Register-blocked kernel: load-store per k**: In `matmul_register_blocked`,
   C rows are loaded and stored on every k-iteration (the loop at `.LBB15_32`).
   This is the correct behavior per the source code — and exactly what the packed
   kernel was designed to fix.

4. **No loop unrolling of k-loop in packed kernel**: The inner k-loop (`.LBB20_27`)
   processes one k-value per iteration. Unrolling by 2-4 could improve instruction
   throughput by reducing loop overhead and enabling better out-of-order execution.

5. **Alignment**: All vector loads/stores use `vmovupd` (unaligned). If the matrix
   buffers were 64-byte aligned, `vmovapd` could be used. Modern CPUs handle
   unaligned loads with no penalty when the address happens to be aligned, so this
   is typically a non-issue in practice.
