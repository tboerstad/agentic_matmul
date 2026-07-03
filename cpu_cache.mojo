"""Query the per-core L2 cache size for matmul cache blocking.

Matmul performance is dominated by how well the working set fits in cache, so
the blocking parameters (KC and the no-pack routing budget) are functions of
the host CPU's L2 size. Rather than hard-coding that number, this module reads
it from the hardware via the x86 `cpuid` instruction: Intel enumerates the
cache hierarchy through the deterministic cache parameters leaf (EAX = 4), and
AMD mirrors the identical encoding at leaf 0x8000001D.

    from cpu_cache import l2_cache_size

    var l2 = l2_cache_size()   # bytes, or 0 if undetectable

Cache geometry is fixed for the life of the process, so the probe runs at most
once: the first call walks `cpuid` and memoizes the result in a process-global
(`_Global`, thread-safe, lazily initialized). This matters because the matmul
dispatcher queries the L2 size on each call, and a live `cpuid` on every
matmul would sink the tiny shapes.
"""

from std.ffi import _Global
from std.sys import CompilationTarget
from std.sys.intrinsics import inlined_assembly


struct CpuidResult(TrivialRegisterPassable):
    """The four 32-bit registers returned by a `cpuid` call."""

    var eax: UInt32
    var ebx: UInt32
    var ecx: UInt32
    var edx: UInt32


def cpuid(leaf: UInt32, subleaf: UInt32 = 0) -> CpuidResult:
    """Execute the x86 `cpuid` instruction with `leaf` in EAX and `subleaf` in
    ECX. Only valid on x86 hosts, so guard callers with
    `CompilationTarget.is_x86()`."""
    return inlined_assembly[
        "cpuid",
        CpuidResult,
        constraints="={eax},={ebx},={ecx},={edx},{eax},{ecx}",
        has_side_effect=True,
    ](leaf, subleaf)


def _is_amd() -> Bool:
    """Return True if the vendor string from leaf 0 is "AuthenticAMD"."""
    var r = cpuid(0)
    # Vendor string is EBX, EDX, ECX (12 ASCII bytes). "AuthenticAMD" packs to
    # EBX=0x68747541 ('htuA'), EDX=0x69746E65 ('itne'), ECX=0x444D4163 ('DMAc').
    return r.ebx == 0x68747541 and r.edx == 0x69746E65 and r.ecx == 0x444D4163


def _detect_l2_size() -> Int:
    """Walk the `cpuid` cache-parameter leaf and return the L2 data/unified
    cache size in bytes, or 0 when it can't be determined (non-x86 hosts, or an
    unsupported leaf)."""
    comptime if not CompilationTarget.is_x86():
        return 0

    var leaf = UInt32(0x8000001D) if _is_amd() else UInt32(4)

    # Walk sub-leaves until cpuid reports a null cache type (no more caches).
    var idx = 0
    while idx < 64:
        var r = cpuid(leaf, UInt32(idx))
        var ctype = Int(r.eax & 0x1F)
        if ctype == 0:  # null: end of the cache list
            break

        var level = Int((r.eax >> 5) & 0x7)
        # Cache type 1 is data, 3 is unified; skip instruction caches.
        if level == 2 and (ctype == 1 or ctype == 3):
            var line_size = Int((r.ebx & 0xFFF) + 1)
            var partitions = Int(((r.ebx >> 12) & 0x3FF) + 1)
            var ways = Int(((r.ebx >> 22) & 0x3FF) + 1)
            var sets = Int(r.ecx) + 1
            return ways * partitions * line_size * sets
        idx += 1

    return 0


# Process-global, thread-safe, lazily initialized: the first read runs the
# `cpuid` walk, every later read returns the memoized value.
comptime _L2_SIZE = _Global["agentic_matmul_l2_size", _detect_l2_size]


def l2_cache_size() -> Int:
    """Per-core L2 cache size in bytes, memoized; 0 if undetectable."""
    try:
        return _L2_SIZE.get_or_create_ptr()[]
    except:
        # Detection itself never raises; this only guards the global accessor.
        return 0
