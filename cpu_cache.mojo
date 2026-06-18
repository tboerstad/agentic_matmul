"""Query CPU cache sizes (L1/L2/L3 ...) via the x86 `cpuid` instruction.

Matmul performance is dominated by how well the working set fits in cache, so
the blocking parameters (MC/KC/NC in BLIS terminology) are really functions of
the L1/L2/L3 sizes of the host CPU. Rather than hard-coding those numbers, this
module reads them straight from the hardware using the `cpuid` intrinsic.

On Intel CPUs the cache hierarchy is enumerated via the *deterministic cache
parameters* leaf (`EAX = 4`); on AMD CPUs with the topology-extensions feature
the identical layout is exposed via leaf `0x8000001D`. We pick the right leaf
from the vendor string returned by leaf 0.

`cpuid` is an x86 instruction, so the detailed `query_caches()` enumeration is
x86-only. On macOS the size accessors fall back to `sysctlbyname`, which works
on both Intel and Apple Silicon (the latter has no `cpuid` at all).

Usage:
    from cpu_cache import query_caches, l1_data_cache_size

    var caches = query_caches()      # List[CacheInfo], one per cache (x86)
    var l1 = l1_data_cache_size()    # bytes, or 0 if undetectable
"""

from std.collections import List
from std.ffi import external_call
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
    the EAX/EBX/ECX/EDX outputs are returned. Only valid on x86 hosts — guard
    callers with `CompilationTarget.is_x86()`.
    """
    return inlined_assembly[
        "cpuid",
        CpuidResult,
        constraints="={eax},={ebx},={ecx},={edx},{eax},{ecx}",
        has_side_effect=True,
    ](leaf, subleaf)


# --- cache descriptor -------------------------------------------------------


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
    var associativity: Int
    """Number of ways of set-associativity."""
    var sets: Int
    """Number of sets."""

    def __init__(
        out self,
        level: Int,
        cache_type: Int,
        size_bytes: Int,
        line_size: Int,
        associativity: Int,
        sets: Int,
    ):
        self.level = level
        self.cache_type = cache_type
        self.size_bytes = size_bytes
        self.line_size = line_size
        self.associativity = associativity
        self.sets = sets

    def type_name(self) -> String:
        if self.cache_type == CACHE_TYPE_DATA:
            return String("Data")
        elif self.cache_type == CACHE_TYPE_INSTRUCTION:
            return String("Instruction")
        elif self.cache_type == CACHE_TYPE_UNIFIED:
            return String("Unified")
        return String("Unknown")

    def label(self) -> String:
        """Short label like "L1d", "L1i", "L2", "L3"."""
        var s = String("L") + String(self.level)
        if self.cache_type == CACHE_TYPE_DATA:
            s += "d"
        elif self.cache_type == CACHE_TYPE_INSTRUCTION:
            s += "i"
        return s


# --- enumeration ------------------------------------------------------------


def _is_amd() -> Bool:
    """Return True if the vendor string from leaf 0 is "AuthenticAMD"."""
    var r = cpuid(0)
    # Vendor string is EBX, EDX, ECX (12 ASCII bytes). "AuthenticAMD" packs to
    # EBX=0x68747541 ('htuA'), EDX=0x69746E65 ('itne'), ECX=0x444D4163 ('DMAc').
    return r.ebx == 0x68747541 and r.edx == 0x69746E65 and r.ecx == 0x444D4163


def query_caches() -> List[CacheInfo]:
    """Enumerate every CPU cache reported by `cpuid`, ordered as the hardware
    lists them (L1d, L1i, L2, L3, ...).

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

        caches.append(
            CacheInfo(
                level=level,
                cache_type=ctype,
                size_bytes=size_bytes,
                line_size=line_size,
                associativity=ways,
                sets=sets,
            )
        )
        idx += 1

    return caches^


# --- macOS sysctl fallback --------------------------------------------------


def _sysctl_size(name: String) -> Int:
    """Read an integer-valued `sysctl` by name (macOS), in bytes.

    Returns 0 on non-macOS hosts or if the key is missing / not an integer.
    The whole body is elided at compile time off macOS, so the
    `sysctlbyname` symbol is never linked on other platforms.
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
    # rc == 0 and a full 8-byte (or 4-byte) integer was written.
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


# --- convenience accessors --------------------------------------------------


def _cache_size(level: Int, prefer_data: Bool) -> Int:
    """Size in bytes of the cache at `level`, or 0 if none was found.

    When `prefer_data` is True a data/unified cache is preferred over an
    instruction cache at the same level (relevant only for split L1).
    """
    var best = 0
    var best_is_data = False
    for c in query_caches():
        if c.level != level:
            continue
        var is_data = (
            c.cache_type == CACHE_TYPE_DATA
            or c.cache_type == CACHE_TYPE_UNIFIED
        )
        if best == 0 or (prefer_data and is_data and not best_is_data):
            best = c.size_bytes
            best_is_data = is_data
    return best


def l1_data_cache_size() -> Int:
    """Per-core L1 data (or unified) cache size in bytes; 0 if undetectable."""
    comptime if CompilationTarget.is_macos():
        return _macos_cache_size(
            "hw.perflevel0.l1dcachesize", "hw.l1dcachesize"
        )
    return _cache_size(1, prefer_data=True)


def l2_cache_size() -> Int:
    """Per-core L2 cache size in bytes; 0 if undetectable."""
    comptime if CompilationTarget.is_macos():
        return _macos_cache_size("hw.perflevel0.l2cachesize", "hw.l2cachesize")
    return _cache_size(2, prefer_data=True)


def l3_cache_size() -> Int:
    """L3 (last-level) cache size in bytes; 0 if undetectable.

    Apple Silicon has no per-core L3 (it uses a shared system-level cache that
    macOS does not surface here), so this typically returns 0 there.
    """
    comptime if CompilationTarget.is_macos():
        return _macos_cache_size("hw.perflevel0.l3cachesize", "hw.l3cachesize")
    return _cache_size(3, prefer_data=True)


# --- demo -------------------------------------------------------------------


def _pad(s: String, width: Int) -> String:
    var out = s
    while out.byte_length() < width:
        out += " "
    return out


def main():
    comptime if CompilationTarget.is_x86():
        print("CPU cache hierarchy (via cpuid):")
        print(
            " ",
            _pad("cache", 6),
            _pad("type", 12),
            _pad("size", 10),
            _pad("line", 6),
            _pad("ways", 5),
            "sets",
        )
        for c in query_caches():
            var kib = c.size_bytes // 1024
            print(
                " ",
                _pad(c.label(), 6),
                _pad(c.type_name(), 12),
                _pad(String(kib) + " KiB", 10),
                _pad(String(c.line_size) + " B", 6),
                _pad(String(c.associativity), 5),
                String(c.sets),
            )
        print()
    else:
        comptime if CompilationTarget.is_macos():
            print("Detailed cpuid enumeration is x86-only; using sysctl sizes.")
        else:
            print("Detailed enumeration unavailable on this platform.")
        print()

    print("Quick accessors:")
    print("  L1d:", l1_data_cache_size() // 1024, "KiB")
    print("  L2: ", l2_cache_size() // 1024, "KiB")
    print("  L3: ", l3_cache_size() // 1024, "KiB")
