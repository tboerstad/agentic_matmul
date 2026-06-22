# SME (Scalable Matrix Extension) f64 GEMM for Apple Silicon M4 (FEAT_SME_F64F64).
#
# The NEON kernels in gemm.mojo top out at the ~515 GFLOPS f64 NEON ceiling of the
# 10 P-cores. Apple Accelerate beats that (prefill ~709 GFLOPS) by driving the SME
# matrix coprocessor, which one thread saturates per P-cluster. The M4 Max has two
# SME units, so an aggregate of ~1035 GFLOPS f64 is reachable from two threads.
#
# This kernel computes C = A*B with the SME f64 outer-product instruction FMOPA:
# each FMOPA does an 8x8 outer-product accumulate into a ZA tile (SVL=512b => 8 f64).
# The micro-kernel holds a 16x32 block of C in all eight ZA.D tiles (a 2x4 grid of
# 8x8) and sweeps a K-panel with two A-loads + four B-loads + eight FMOPA per step.
#
# Loop order is GotoBLAS jc-pc-ic: for each N j-strip (parallelized across the two
# SME-bearing P-clusters), for each KC-deep k-panel, for each 16-row i-tile. Holding
# pc outside ic keeps the B k-panel L1-resident and reused across every i-tile, so B
# is streamed from DRAM only once regardless of M. C accumulates across k-panels in
# the ZA tiles: the first panel overwrites (zeroed ZA), later panels load C back into
# ZA, add, and store. A is packed once column-major; B is read in place.
from std.sys.intrinsics import inlined_assembly
from std.memory.unsafe_pointer import alloc
from std.algorithm.functional import parallelize
from std.math import ceildiv
from std.time import perf_counter_ns
from matrix import Matrix

alias MR = 16   # micro-tile rows  (2 ZA row-blocks of 8)
alias NR = 32   # micro-tile cols  (4 ZA col-blocks of 8)


# ---------------------------------------------------------------------------
# Shared SME assembly fragments.
#
# The four 16x32 micro-kernels below are the same instruction stream with two
# independent choices: seed ZA from zero vs load C (the first vs a later k-panel),
# and store all 16 rows vs only the `vr` valid rows of a partial last i-tile. The
# FMA K-sweep (_SME_KSWEEP16) was identical in all four, the full ZA->C store
# (_SME_TAIL_FULL) in two, the masked store (_SME_TAIL_PART) in two. Naming each
# once and composing the template per kernel is pure comptime string
# concatenation, so every kernel emits the exact instruction stream it did when
# spelled out in full -- the operand slots ($0=pA $1=pB $2=pC $3=ldc $4=kc $5=ldb
# $6=vr) line up because every kernel passes its args in that order.
# ---------------------------------------------------------------------------

comptime _SME_HEAD_ZERO = """
        smstart
        ptrue p0.d
        zero {za}
        lsl x13, $3, #3
        lsl x16, $5, #3
"""

comptime _SME_HEAD_A = """
        smstart
        ptrue p0.d
        lsl x13, $3, #3
        lsl x16, $5, #3
        mov x14, $2
        mov w12, #0
        mov x15, x14
        ld1d {za0h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za1h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za2h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za3h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #1
        mov x15, x14
        ld1d {za0h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za1h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za2h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za3h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #2
        mov x15, x14
        ld1d {za0h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za1h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za2h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za3h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #3
        mov x15, x14
        ld1d {za0h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za1h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za2h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za3h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #4
        mov x15, x14
        ld1d {za0h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za1h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za2h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za3h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #5
        mov x15, x14
        ld1d {za0h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za1h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za2h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za3h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #6
        mov x15, x14
        ld1d {za0h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za1h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za2h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za3h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #7
        mov x15, x14
        ld1d {za0h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za1h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za2h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za3h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #0
        mov x15, x14
        ld1d {za4h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za5h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za6h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za7h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #1
        mov x15, x14
        ld1d {za4h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za5h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za6h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za7h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #2
        mov x15, x14
        ld1d {za4h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za5h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za6h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za7h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #3
        mov x15, x14
        ld1d {za4h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za5h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za6h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za7h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #4
        mov x15, x14
        ld1d {za4h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za5h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za6h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za7h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #5
        mov x15, x14
        ld1d {za4h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za5h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za6h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za7h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #6
        mov x15, x14
        ld1d {za4h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za5h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za6h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za7h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        mov w12, #7
        mov x15, x14
        ld1d {za4h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za5h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za6h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za7h.d[w12, 0]}, p0/z, [x15]
"""

comptime _SME_HEAD_A_PART = """
        smstart
        ptrue p0.d
        zero {za}
        lsl x13, $3, #3
        lsl x16, $5, #3
        mov x9, #8
        mov x14, $2
        mov x17, $6
        cmp x17, #8
        csel x19, x17, x9, lt
        mov x18, #0
    7:
        cmp x18, x19
        b.ge 8f
        mov w12, w18
        mov x15, x14
        ld1d {za0h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za1h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za2h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za3h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        add x18, x18, #1
        b 7b
    8:
        cmp x17, #8
        b.le 9f
        mov x18, #8
    10:
        cmp x18, x17
        b.ge 9f
        sub w12, w18, #8
        mov x15, x14
        ld1d {za4h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za5h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za6h.d[w12, 0]}, p0/z, [x15]
        add x15, x15, #64
        ld1d {za7h.d[w12, 0]}, p0/z, [x15]
        add x14, x14, x13
        add x18, x18, #1
        b 10b
    9:
        lsl x16, $5, #3
"""

comptime _SME_KSWEEP16 = """        mov x9, $0
        mov x10, $1
        mov x11, $4
    1:
        ld1d {z0.d}, p0/z, [x9]
        ld1d {z1.d}, p0/z, [x9, #1, mul vl]
        ld1d {z2.d}, p0/z, [x10]
        ld1d {z3.d}, p0/z, [x10, #1, mul vl]
        ld1d {z4.d}, p0/z, [x10, #2, mul vl]
        ld1d {z5.d}, p0/z, [x10, #3, mul vl]
        fmopa za0.d, p0/m, p0/m, z0.d, z2.d
        fmopa za1.d, p0/m, p0/m, z0.d, z3.d
        fmopa za2.d, p0/m, p0/m, z0.d, z4.d
        fmopa za3.d, p0/m, p0/m, z0.d, z5.d
        fmopa za4.d, p0/m, p0/m, z1.d, z2.d
        fmopa za5.d, p0/m, p0/m, z1.d, z3.d
        fmopa za6.d, p0/m, p0/m, z1.d, z4.d
        fmopa za7.d, p0/m, p0/m, z1.d, z5.d
        add x9, x9, #128
        add x10, x10, x16
        subs x11, x11, #1
        b.ne 1b
"""

comptime _SME_TAIL_FULL = """        mov x14, $2
        mov w12, #0
        mov x15, x14
        st1d {za0h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za1h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za2h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za3h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #1
        mov x15, x14
        st1d {za0h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za1h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za2h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za3h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #2
        mov x15, x14
        st1d {za0h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za1h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za2h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za3h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #3
        mov x15, x14
        st1d {za0h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za1h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za2h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za3h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #4
        mov x15, x14
        st1d {za0h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za1h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za2h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za3h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #5
        mov x15, x14
        st1d {za0h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za1h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za2h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za3h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #6
        mov x15, x14
        st1d {za0h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za1h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za2h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za3h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #7
        mov x15, x14
        st1d {za0h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za1h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za2h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za3h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #0
        mov x15, x14
        st1d {za4h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za5h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za6h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za7h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #1
        mov x15, x14
        st1d {za4h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za5h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za6h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za7h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #2
        mov x15, x14
        st1d {za4h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za5h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za6h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za7h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #3
        mov x15, x14
        st1d {za4h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za5h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za6h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za7h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #4
        mov x15, x14
        st1d {za4h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za5h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za6h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za7h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #5
        mov x15, x14
        st1d {za4h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za5h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za6h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za7h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #6
        mov x15, x14
        st1d {za4h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za5h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za6h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za7h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        mov w12, #7
        mov x15, x14
        st1d {za4h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za5h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za6h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za7h.d[w12, 0]}, p0, [x15]
        smstop
        """

comptime _SME_TAIL_PART = """        mov x9, #8
        mov x14, $2
        mov x17, $6
        cmp x17, #8
        csel x19, x17, x9, lt
        mov x18, #0
    7:
        cmp x18, x19
        b.ge 8f
        mov w12, w18
        mov x15, x14
        st1d {za0h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za1h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za2h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za3h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        add x18, x18, #1
        b 7b
    8:
        cmp x17, #8
        b.le 9f
        mov x18, #8
    10:
        cmp x18, x17
        b.ge 9f
        sub w12, w18, #8
        mov x15, x14
        st1d {za4h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za5h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za6h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za7h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        add x18, x18, #1
        b 10b
    9:
        smstop
        """

comptime _SME_ASM_Z = _SME_HEAD_ZERO + _SME_KSWEEP16 + _SME_TAIL_FULL
comptime _SME_ASM_A = _SME_HEAD_A + _SME_KSWEEP16 + _SME_TAIL_FULL
comptime _SME_ASM_Z_PART = _SME_HEAD_ZERO + _SME_KSWEEP16 + _SME_TAIL_PART
comptime _SME_ASM_A_PART = _SME_HEAD_A_PART + _SME_KSWEEP16 + _SME_TAIL_PART



@always_inline
def _sme_micro_z(
    pA: UnsafePointer[Float64, ...],
    pB: UnsafePointer[Float64, ...],
    pC: UnsafePointer[Float64, ...],
    ldc: Int,
    kc: Int,
    ldb: Int,
):
    """16x32 SME micro-kernel over a kc-deep K-panel. zeroes ZA (first k-panel: C is overwritten). pA is packed
    column-major (16-row panels); B is read in place from pB with row stride ldb
    (the 32 columns of a B row are contiguous). C is row-major, row stride ldc."""
    inlined_assembly[
        _SME_ASM_Z,
        NoneType,
        constraints="r,r,r,r,r,r,~{z0},~{z1},~{z2},~{z3},~{z4},~{z5},~{z6},~{z7},~{z8},~{z9},~{z10},~{z11},~{z12},~{z13},~{z14},~{z15},~{z16},~{z17},~{z18},~{z19},~{z20},~{z21},~{z22},~{z23},~{z24},~{z25},~{z26},~{z27},~{z28},~{z29},~{z30},~{z31},~{p0},~{p1},~{p2},~{p3},~{p4},~{p5},~{p6},~{p7},~{p8},~{p9},~{p10},~{p11},~{p12},~{p13},~{p14},~{p15},~{x9},~{x10},~{x11},~{x13},~{x14},~{x15},~{x16},~{w12},~{memory}",
        has_side_effect=True,
    ](pA, pB, pC, ldc, kc, ldb)


@always_inline
def _sme_micro_a(
    pA: UnsafePointer[Float64, ...],
    pB: UnsafePointer[Float64, ...],
    pC: UnsafePointer[Float64, ...],
    ldc: Int,
    kc: Int,
    ldb: Int,
):
    """16x32 SME micro-kernel over a kc-deep K-panel. loads the current C block into ZA (subsequent k-panels accumulate). pA is packed
    column-major (16-row panels); B is read in place from pB with row stride ldb
    (the 32 columns of a B row are contiguous). C is row-major, row stride ldc."""
    inlined_assembly[
        _SME_ASM_A,
        NoneType,
        constraints="r,r,r,r,r,r,~{z0},~{z1},~{z2},~{z3},~{z4},~{z5},~{z6},~{z7},~{z8},~{z9},~{z10},~{z11},~{z12},~{z13},~{z14},~{z15},~{z16},~{z17},~{z18},~{z19},~{z20},~{z21},~{z22},~{z23},~{z24},~{z25},~{z26},~{z27},~{z28},~{z29},~{z30},~{z31},~{p0},~{p1},~{p2},~{p3},~{p4},~{p5},~{p6},~{p7},~{p8},~{p9},~{p10},~{p11},~{p12},~{p13},~{p14},~{p15},~{x9},~{x10},~{x11},~{x13},~{x14},~{x15},~{x16},~{w12},~{memory}",
        has_side_effect=True,
    ](pA, pB, pC, ldc, kc, ldb)


@always_inline
def _sme_micro_z_part(
    pA: UnsafePointer[Float64, ...],
    pB: UnsafePointer[Float64, ...],
    pC: UnsafePointer[Float64, ...],
    ldc: Int,
    kc: Int,
    ldb: Int,
    vr: Int,
):
    """16x32 SME micro-kernel storing only the first `vr` rows to C (zeroes ZA). For the
    partial last i-tile when M % 16 != 0: A's panel is zero-padded past row M, the
    full tile is computed, and only the vr valid rows are written (in-bounds, no
    recompute, no scratch)."""
    inlined_assembly[
        _SME_ASM_Z_PART,
        NoneType,
        constraints="r,r,r,r,r,r,r,~{z0},~{z1},~{z2},~{z3},~{z4},~{z5},~{z6},~{z7},~{z8},~{z9},~{z10},~{z11},~{z12},~{z13},~{z14},~{z15},~{z16},~{z17},~{z18},~{z19},~{z20},~{z21},~{z22},~{z23},~{z24},~{z25},~{z26},~{z27},~{z28},~{z29},~{z30},~{z31},~{p0},~{p1},~{p2},~{p3},~{p4},~{p5},~{p6},~{p7},~{p8},~{p9},~{p10},~{p11},~{p12},~{p13},~{p14},~{p15},~{x9},~{x10},~{x11},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{w12},~{memory}",
        has_side_effect=True,
    ](pA, pB, pC, ldc, kc, ldb, vr)


@always_inline
def _sme_micro_a_part(
    pA: UnsafePointer[Float64, ...],
    pB: UnsafePointer[Float64, ...],
    pC: UnsafePointer[Float64, ...],
    ldc: Int,
    kc: Int,
    ldb: Int,
    vr: Int,
):
    """16x32 SME micro-kernel storing only the first `vr` rows to C (loads vr rows of C into ZA, accumulates). For the
    partial last i-tile when M % 16 != 0: A's panel is zero-padded past row M, the
    full tile is computed, and only the vr valid rows are written (in-bounds, no
    recompute, no scratch)."""
    inlined_assembly[
        _SME_ASM_A_PART,
        NoneType,
        constraints="r,r,r,r,r,r,r,~{z0},~{z1},~{z2},~{z3},~{z4},~{z5},~{z6},~{z7},~{z8},~{z9},~{z10},~{z11},~{z12},~{z13},~{z14},~{z15},~{z16},~{z17},~{z18},~{z19},~{z20},~{z21},~{z22},~{z23},~{z24},~{z25},~{z26},~{z27},~{z28},~{z29},~{z30},~{z31},~{p0},~{p1},~{p2},~{p3},~{p4},~{p5},~{p6},~{p7},~{p8},~{p9},~{p10},~{p11},~{p12},~{p13},~{p14},~{p15},~{x9},~{x10},~{x11},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{w12},~{memory}",
        has_side_effect=True,
    ](pA, pB, pC, ldc, kc, ldb, vr)



@always_inline
def _sme_micro8(
    pA: UnsafePointer[Float64, ...],
    pB: UnsafePointer[Float64, ...],
    pC: UnsafePointer[Float64, ...],
    ldc: Int,
    kc: Int,
    ldb: Int,
    vr: Int,
):
    """8x32 SME micro-kernel (one ZA row-block, za0-3, 4 FMOPA/step), storing vr<=8
    rows. For M < 16 the 16-row tile wastes over half its FMOPA; the 8-row tile halves
    that. A is packed in 8-row column-major panels; B read in place; vr valid rows
    written straight to C."""
    inlined_assembly[
        """
        smstart
        ptrue p0.d
        zero {za}
        lsl x13, $3, #3
        lsl x16, $5, #3
        mov x9, $0
        mov x10, $1
        mov x11, $4
    1:
        ld1d {z0.d}, p0/z, [x9]
        ld1d {z2.d}, p0/z, [x10]
        ld1d {z3.d}, p0/z, [x10, #1, mul vl]
        ld1d {z4.d}, p0/z, [x10, #2, mul vl]
        ld1d {z5.d}, p0/z, [x10, #3, mul vl]
        fmopa za0.d, p0/m, p0/m, z0.d, z2.d
        fmopa za1.d, p0/m, p0/m, z0.d, z3.d
        fmopa za2.d, p0/m, p0/m, z0.d, z4.d
        fmopa za3.d, p0/m, p0/m, z0.d, z5.d
        add x9, x9, #64
        add x10, x10, x16
        subs x11, x11, #1
        b.ne 1b
        mov x14, $2
        mov x17, $6
        mov x18, #0
    7:
        cmp x18, x17
        b.ge 8f
        mov w12, w18
        mov x15, x14
        st1d {za0h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za1h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za2h.d[w12, 0]}, p0, [x15]
        add x15, x15, #64
        st1d {za3h.d[w12, 0]}, p0, [x15]
        add x14, x14, x13
        add x18, x18, #1
        b 7b
    8:
        smstop
        """,
        NoneType,
        constraints="r,r,r,r,r,r,r,~{z0},~{z1},~{z2},~{z3},~{z4},~{z5},~{z6},~{z7},~{z8},~{z9},~{z10},~{z11},~{z12},~{z13},~{z14},~{z15},~{z16},~{z17},~{z18},~{z19},~{z20},~{z21},~{z22},~{z23},~{z24},~{z25},~{z26},~{z27},~{z28},~{z29},~{z30},~{z31},~{p0},~{p1},~{p2},~{p3},~{p4},~{p5},~{p6},~{p7},~{p8},~{p9},~{p10},~{p11},~{p12},~{p13},~{p14},~{p15},~{x9},~{x10},~{x11},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{w12},~{memory}",
        has_side_effect=True,
    ](pA, pB, pC, ldc, kc, ldb, vr)


def sme_gemm_small_ptr(
    c_ptr: UnsafePointer[Float64, ...],
    a_ptr: UnsafePointer[Float64, ...],
    b_ptr: UnsafePointer[Float64, ...],
    M: Int,
    N: Int,
    K: Int,
    nw: Int = 2,
):
    """f64 GEMM for small M (M < 16), using the 8x32 tile. The 16-row tile would
    waste over half its FMOPA on so few rows; the 8-row tile (ceildiv(M,8) <= 2
    i-tiles, KC=K since there is no i-reuse to block for) wastes far less. A packed
    into 8-row column-major panels; B read in place; each tile stores its vr valid
    rows. N % 32 handled by a shifted overlap column tile."""
    var num_i8 = ceildiv(M, 8)
    var num_j_full = N // NR
    var rN = N - num_j_full * NR
    var pA = alloc[Float64](num_i8 * K * 8)
    def pack_a(it: Int) {mut pA, read a_ptr, read K, read M}:
        var base = it * K * 8
        var i0 = it * 8
        for k in range(K):
            for r in range(8):
                var row = i0 + r
                pA[base + k * 8 + r] = a_ptr[row * K + k] if row < M else 0.0
    parallelize(pack_a, num_i8, num_i8)
    var tiles_pw = ceildiv(num_j_full, nw) if num_j_full > 0 else 1
    def worker(wid: Int) {read pA, read b_ptr, read c_ptr, read N, read K, read M, read num_i8, read num_j_full, read tiles_pw}:
        var jt0 = wid * tiles_pw
        var jt1 = jt0 + tiles_pw
        if jt1 > num_j_full: jt1 = num_j_full
        for jt in range(jt0, jt1):
            for it in range(num_i8):
                var vr = M - it * 8
                if vr > 8: vr = 8
                _sme_micro8(pA + it * K * 8, b_ptr + jt * NR, c_ptr + (it * 8) * N + jt * NR, N, K, N, vr)
    if num_j_full > 0:
        parallelize(worker, nw, nw)
    if rN > 0:
        var j0 = N - NR
        def col_worker(it: Int) {read pA, read b_ptr, read c_ptr, read N, read K, read M, read j0}:
            var vr = M - it * 8
            if vr > 8: vr = 8
            _sme_micro8(pA + it * K * 8, b_ptr + j0, c_ptr + (it * 8) * N + j0, N, K, N, vr)
        parallelize(col_worker, num_i8, num_i8)
    pA.free()


def sme_gemm(
    mut C: Matrix[DType.float64],
    A: Matrix[DType.float64],
    B: Matrix[DType.float64],
    nw: Int = 2,
    KC: Int = 1 << 30,
    MC: Int = 1 << 30,
):
    """Matrix wrapper around `sme_gemm_ptr` (see it for the algorithm)."""
    sme_gemm_ptr(
        C.data.unsafe_ptr(), A.data.unsafe_ptr(), B.data.unsafe_ptr(),
        C.rows, C.cols, A.cols, nw, KC, MC,
    )


def sme_gemm_ptr(
    c_ptr: UnsafePointer[Float64, ...],
    a_ptr: UnsafePointer[Float64, ...],
    b_ptr: UnsafePointer[Float64, ...],
    M: Int,
    N: Int,
    K: Int,
    nw: Int = 2,
    KC: Int = 1 << 30,
    MC: Int = 1 << 30,
):
    """Full f64 GEMM C = A*B via the SME micro-kernel, on raw row-major pointers.
    Handles any M >= 16, N >= 32 (M % 16 / N % 32 need not be zero). Parallelizes the
    N j-strips across `nw` workers; nw=2 puts one worker on each of the M4 Max's two
    SME-bearing P-clusters.

    The divisible bulk (rows [0, 16*(M//16)), cols [0, 32*(N//32))) runs the blocked
    main loop, order (pc, ic-block, jt, it): pc blocks K by KC, ic-block blocks M by
    MC. For a fixed k-panel the MC-tall A block (MC x KC) is reused across the
    worker's j-tiles and stays L2-resident, so each A byte is read from DRAM once per
    worker instead of once per j-strip. The B k-panel (KC x 32) stays L1-resident
    across the inner i-tile loop. C accumulates across k-panels in the ZA tiles (first
    panel zeroes ZA, later panels reload C, add, store).

    Remainders need no masking and no scratch. The M % 16 partial last i-tile is
    folded into the main sweep as a normal i-tile (A's panel zero-padded past row M)
    whose store writes only its rM valid rows straight to C (`_sme_micro_z_part` /
    `_sme_micro_a_part`) — in-bounds, no recompute, and B-reuse / blocking / pipelining
    intact (this is why it matches divisible M, where a separate remainder pass did
    not). The N % 32 partial last column is covered by a full 32-wide tile shifted to
    start at N-32 (every byte in-bounds; it overwrites the small column overlap with
    identical full-sum values, in its own sequential pass)."""
    var num_i_full = M // MR
    var num_j_full = N // NR
    var rM = M - num_i_full * MR  # partial row count (0 if divisible)
    var rN = N - num_j_full * NR  # partial col count (0 if divisible)
    # i-tiles incl a partial last one when rM != 0; its A panel is zero-padded.
    var num_i = num_i_full + (1 if rM > 0 else 0)
    var i_last = num_i_full * MR  # first row of the partial tile
    var mc_t = MC // MR
    if mc_t < 1: mc_t = 1
    if mc_t > num_i: mc_t = num_i

    # Pack A column-major into aligned 16-row panels; the last panel (rows
    # [i_last, M)) is zero-padded to 16 rows when M % 16 != 0.
    var pA = alloc[Float64](num_i * K * MR)
    var pack_workers = num_i if num_i < 10 else 10
    def pack_a(it: Int) {mut pA, read a_ptr, read K, read M}:
        var base = it * K * MR
        var i0 = it * MR
        for k in range(K):
            for r in range(MR):
                var row = i0 + r
                pA[base + k * MR + r] = a_ptr[row * K + k] if row < M else 0.0
    parallelize(pack_a, num_i, pack_workers)

    # --- Region 1: bulk (all i-tiles incl the partial last), blocked, into C -------
    var tiles_pw = ceildiv(num_j_full, nw) if num_j_full > 0 else 1
    def worker(wid: Int) {read pA, read b_ptr, read c_ptr, read N, read K, read num_i, read num_i_full, read num_j_full, read tiles_pw, read KC, read mc_t, read rM}:
        var jt0 = wid * tiles_pw
        var jt1 = jt0 + tiles_pw
        if jt1 > num_j_full: jt1 = num_j_full
        for pc in range(0, K, KC):
            var kc = KC
            if pc + kc > K: kc = K - pc
            var first = pc == 0
            for ic0 in range(0, num_i, mc_t):
                var ic1 = ic0 + mc_t
                if ic1 > num_i: ic1 = num_i
                for jt in range(jt0, jt1):
                    var pb = b_ptr + pc * N + jt * NR
                    for it in range(ic0, ic1):
                        var pa = pA + it * K * MR + pc * MR
                        var pcptr = c_ptr + (it * MR) * N + jt * NR
                        if it < num_i_full:
                            if first:
                                _sme_micro_z(pa, pb, pcptr, N, kc, N)
                            else:
                                _sme_micro_a(pa, pb, pcptr, N, kc, N)
                        else:
                            if first:
                                _sme_micro_z_part(pa, pb, pcptr, N, kc, N, rM)
                            else:
                                _sme_micro_a_part(pa, pb, pcptr, N, kc, N, rM)
    if num_j_full > 0:
        parallelize(worker, nw, nw)

    # --- Region 2: right column strip, a full 32-wide tile shifted to start N-32 ---
    if rN > 0:
        var j0 = N - NR
        def col_worker(it: Int) {read pA, read b_ptr, read c_ptr, read N, read K, read num_i_full, read j0, read rM}:
            var pa = pA + it * K * MR
            var pcptr = c_ptr + (it * MR) * N + j0
            if it < num_i_full:
                _sme_micro_z(pa, b_ptr + j0, pcptr, N, K, N)
            else:
                _sme_micro_z_part(pa, b_ptr + j0, pcptr, N, K, N, rM)
        parallelize(col_worker, num_i, num_i if num_i < 10 else 10)

    pA.free()
