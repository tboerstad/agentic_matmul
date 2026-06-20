"""Query CPU cache sizes (L1/L2/L3) and line size for matmul cache blocking.

Matmul performance is dominated by how well the working set fits in cache, so
the blocking parameters (MC/KC/NC in BLIS terminology) and packing alignment
are really functions of the host CPU's cache sizes and line size. Rather than
hard-coding those numbers, this module reads them from the hardware.

On x86 the values come from the `cpuid` instruction: Intel enumerates the cache
hierarchy via the *deterministic cache parameters* leaf (`EAX = 4`), and AMD
mirrors the identical encoding at leaf `0x8000001D`. On macOS (where Apple
Silicon has no `cpuid` at all) the sizes come from `sysctlbyname`.

Trivial API for the kernels:

    from cpu_cache import (
        l1_data_cache_size, l2_cache_size, l3_cache_size, cache_line_size
    )

    var l1 = l1_data_cache_size()   # bytes, or 0 if undetectable
    var line = cache_line_size()    # bytes (64 on x86, 128 on Apple Silicon)

All accessors return 0 when the value can't be determined, so callers should
fall back to a sensible default in that case.

Cache geometry is fixed for the life of the process, so the (relatively heavy)
hardware probe runs at most once. That probe is a `cpuid` enumeration on x86 and
a handful of `sysctlbyname` calls on macOS. The first accessor call detects the whole
L1/L2/L3/line geometry in a single pass and memoizes it in a process-global via
`_Global` (thread-safe, lazily initialized); every later call just reads the
cached struct. This matters because the matmul dispatcher queries the L2 size
on each call, and without memoization a decode/prefill loop would re-run the
`cpuid` walk on every matmul.
"""

from std.collections import List
from std.ffi import _Global, external_call
from std.memory.unsafe_pointer import alloc
from std.sys import CompilationTarget
from std.sys.intrinsics import inlined_assembly


# --- raw cpuid --------------------------------------------------------------


struct CpuidResult(TrivialRegisterPassable):
    """The four 32-bit registers returned by a `cpuid` call."""

    var eax: UInt32
    var ebx: UInt32
    var ecx: UInt32
    var edx: UInt32


def cpuid(leaf: UInt32, subleaf: UInt32 = 0) -> CpuidResult:
    """Execute the x86 `cpuid` instruction.

    `leaf` is loaded into EAX and `subleaf` into ECX before the instruction;
    the EAX/EBX/ECX/EDX outputs are returned. Only valid on x86 hosts, so guard
    callers with `CompilationTarget.is_x86()`.
    """
    return inlined_assembly[
        "cpuid",
        CpuidResult,
        constraints="={eax},={ebx},={ecx},={edx},{eax},{ecx}",
        has_side_effect=True,
    ](leaf, subleaf)


# --- cpuid cache enumeration (x86) ------------------------------------------


# Cache type values reported in EAX[4:0] of the cache-parameter leaf.
comptime CACHE_TYPE_NULL = 0
comptime CACHE_TYPE_DATA = 1
comptime CACHE_TYPE_INSTRUCTION = 2
comptime CACHE_TYPE_UNIFIED = 3


struct CacheInfo(Copyable, Movable):
    """Geometry of a single CPU cache, as decoded from `cpuid`."""

    var level: Int
    """Cache level: 1 for L1, 2 for L2, 3 for L3, ..."""
    var cache_type: Int
    """One of CACHE_TYPE_{DATA, INSTRUCTION, UNIFIED}."""
    var size_bytes: Int
    """Total size of the cache in bytes."""
    var line_size: Int
    """Coherency line size in bytes (typically 64)."""

    def __init__(
        out self,
        level: Int,
        cache_type: Int,
        size_bytes: Int,
        line_size: Int,
    ):
        self.level = level
        self.cache_type = cache_type
        self.size_bytes = size_bytes
        self.line_size = line_size


def _is_amd() -> Bool:
    """Return True if the vendor string from leaf 0 is "AuthenticAMD"."""
    var r = cpuid(0)
    # Vendor string is EBX, EDX, ECX (12 ASCII bytes). "AuthenticAMD" packs to
    # EBX=0x68747541 ('htuA'), EDX=0x69746E65 ('itne'), ECX=0x444D4163 ('DMAc').
    return r.ebx == 0x68747541 and r.edx == 0x69746E65 and r.ecx == 0x444D4163


def _query_caches() -> List[CacheInfo]:
    """Enumerate every CPU cache reported by `cpuid` (L1d, L1i, L2, L3, ...).

    Returns an empty list on non-x86 hosts or when the cache-parameter leaf is
    unsupported.
    """
    var caches = List[CacheInfo]()

    comptime if not CompilationTarget.is_x86():
        return caches^

    # Intel exposes deterministic cache parameters at leaf 4; AMD mirrors the
    # exact same encoding at leaf 0x8000001D.
    var leaf = UInt32(0x8000001D) if _is_amd() else UInt32(4)

    # Walk sub-leaves until cpuid reports a null cache type (no more caches).
    var idx = 0
    while idx < 64:
        var r = cpuid(leaf, UInt32(idx))
        var ctype = Int(r.eax & 0x1F)
        if ctype == CACHE_TYPE_NULL:
            break

        var level = Int((r.eax >> 5) & 0x7)
        var line_size = Int((r.ebx & 0xFFF) + 1)
        var partitions = Int(((r.ebx >> 12) & 0x3FF) + 1)
        var ways = Int(((r.ebx >> 22) & 0x3FF) + 1)
        var sets = Int(r.ecx) + 1
        var size_bytes = ways * partitions * line_size * sets

        caches.append(CacheInfo(level, ctype, size_bytes, line_size))
        idx += 1

    return caches^


def _cpuid_cache_size(caches: List[CacheInfo], level: Int) -> Int:
    """Size in bytes of the data/unified cache at `level` (x86), or 0.

    Takes the already-enumerated cache list so a single `_query_caches()` pass
    can answer every level (see `_detect_cache_geometry`).
    """
    var best = 0
    var best_is_data = False
    for c in caches:
        if c.level != level:
            continue
        var is_data = (
            c.cache_type == CACHE_TYPE_DATA
            or c.cache_type == CACHE_TYPE_UNIFIED
        )
        # Prefer a data/unified cache over the split instruction cache at L1.
        if best == 0 or (is_data and not best_is_data):
            best = c.size_bytes
            best_is_data = is_data
    return best


def _cpuid_line_size(caches: List[CacheInfo]) -> Int:
    """Coherency line size in bytes from the first reported cache (x86), or 0."""
    for c in caches:
        if c.line_size > 0:
            return c.line_size
    return 0


# --- macOS sysctl fallback --------------------------------------------------


def _sysctl_size(name: String) -> Int:
    """Read an integer-valued `sysctl` by name (macOS), in bytes.

    Returns 0 on non-macOS hosts or if the key is missing / not an integer.
    The whole body is elided at compile time off macOS, so the `sysctlbyname`
    symbol is never linked on other platforms.
    """
    comptime if not CompilationTarget.is_macos():
        return 0

    var value = alloc[UInt64](1)
    value[0] = 0
    var length = alloc[UInt](1)
    length[0] = 8  # sizeof(UInt64)

    # int sysctlbyname(const char *name, void *oldp, size_t *oldlenp,
    #                  const void *newp, size_t newlen);
    var rc = external_call["sysctlbyname", Int32](
        name.unsafe_ptr(), value, length, Int(0), UInt(0)
    )

    var result = 0
    if rc == 0 and (length[0] == 8 or length[0] == 4):
        result = Int(value[0])

    value.free()
    length.free()
    return result


def _macos_cache_size(primary: String, fallback: String) -> Int:
    """Try the performance-core-specific sysctl key first, then the generic
    one. On Apple Silicon `hw.perflevel0.*` reports the P-core cache, which is
    the relevant figure for compute; Intel Macs only have the generic key.
    """
    var s = _sysctl_size(primary)
    if s > 0:
        return s
    return _sysctl_size(fallback)


# --- memoized geometry -------------------------------------------------------


struct CacheGeometry(Copyable, Movable):
    """The host's resolved cache geometry, detected once and then cached.

    All fields are in bytes and are 0 when the corresponding value can't be
    determined on this host.
    """

    var l1: Int
    """Per-core L1 data (or unified) cache size."""
    var l2: Int
    """Per-core L2 cache size."""
    var l3: Int
    """L3 (last-level) cache size."""
    var line: Int
    """Coherency line size."""

    def __init__(out self, l1: Int, l2: Int, l3: Int, line: Int):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.line = line


def _detect_cache_geometry() -> CacheGeometry:
    """Probe the hardware for the full cache geometry in a single pass.

    Runs the platform-specific detection exactly once. `_Global` below makes
    this the lazily-initialized backing store, so the `cpuid` walk and `sysctl`
    calls happen on the first accessor and never again.
    """
    comptime if CompilationTarget.is_macos():
        return CacheGeometry(
            _macos_cache_size("hw.perflevel0.l1dcachesize", "hw.l1dcachesize"),
            _macos_cache_size("hw.perflevel0.l2cachesize", "hw.l2cachesize"),
            _macos_cache_size("hw.perflevel0.l3cachesize", "hw.l3cachesize"),
            _sysctl_size("hw.cachelinesize"),
        )

    # x86: enumerate the cache hierarchy once and read every level from it.
    var caches = _query_caches()
    return CacheGeometry(
        _cpuid_cache_size(caches, 1),
        _cpuid_cache_size(caches, 2),
        _cpuid_cache_size(caches, 3),
        _cpuid_line_size(caches),
    )


# Process-global, thread-safe, lazily initialized: the first read triggers
# `_detect_cache_geometry`, every later read returns the cached struct.
comptime _CACHE_GEOMETRY = _Global["agentic_matmul_cpu_cache", _detect_cache_geometry]


def _cached_geometry() -> CacheGeometry:
    """Return the memoized cache geometry, detecting it on first use."""
    try:
        return _CACHE_GEOMETRY.get_or_create_ptr()[].copy()
    except:
        # Detection itself never raises; this only guards the global accessor.
        return CacheGeometry(0, 0, 0, 0)


# --- public API -------------------------------------------------------------


def l1_data_cache_size() -> Int:
    """Per-core L1 data (or unified) cache size in bytes; 0 if undetectable."""
    return _cached_geometry().l1


def l2_cache_size() -> Int:
    """Per-core L2 cache size in bytes; 0 if undetectable."""
    return _cached_geometry().l2


def l3_cache_size() -> Int:
    """L3 (last-level) cache size in bytes; 0 if undetectable.

    Apple Silicon has no per-core L3 (it uses a shared system-level cache that
    macOS does not surface here), so this typically returns 0 there.
    """
    return _cached_geometry().l3


def cache_line_size() -> Int:
    """Cache line size in bytes; 0 if undetectable.

    Typically 64 on x86 and 128 on Apple Silicon. Useful for choosing packing
    alignment and the inner-kernel stride in the matmul.
    """
    return _cached_geometry().line
