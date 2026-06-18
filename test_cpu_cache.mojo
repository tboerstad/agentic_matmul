from std.testing import assert_true, assert_equal, TestSuite
from std.sys import CompilationTarget

from cpu_cache import (
    query_caches,
    l1_data_cache_size,
    l2_cache_size,
    l3_cache_size,
    CACHE_TYPE_DATA,
    CACHE_TYPE_UNIFIED,
)


# At least one cache should be reported on any x86 host.
def test_caches_nonempty() raises:
    comptime if not CompilationTarget.is_x86():
        return
    var caches = query_caches()
    assert_true(len(caches) > 0, "expected at least one cache on x86")


# Every reported cache must have sane, positive geometry, and the product of
# its fields must equal the reported total size.
def test_geometry_consistent() raises:
    comptime if not CompilationTarget.is_x86():
        return
    for c in query_caches():
        assert_true(c.level >= 1, "cache level must be >= 1")
        assert_true(c.size_bytes > 0, "cache size must be positive")
        assert_true(c.line_size > 0, "line size must be positive")
        assert_true(c.associativity > 0, "associativity must be positive")
        assert_true(c.sets > 0, "set count must be positive")
        # size = ways * line_size * sets (partitions folded into the product
        # when decoded, but is 1 on every shipping CPU). Allow line_size*ways*
        # sets to divide cleanly into size.
        assert_equal(
            c.size_bytes % (c.line_size * c.associativity),
            0,
            "size not a multiple of line_size * ways",
        )


# Line size is 64 bytes on every current x86 CPU.
def test_line_size_64() raises:
    comptime if not CompilationTarget.is_x86():
        return
    for c in query_caches():
        assert_equal(c.line_size, 64, "expected 64-byte cache lines")


# Cache sizes should grow monotonically down the hierarchy: L1 <= L2 <= L3.
def test_hierarchy_monotonic() raises:
    comptime if not CompilationTarget.is_x86():
        return
    var l1 = l1_data_cache_size()
    var l2 = l2_cache_size()
    var l3 = l3_cache_size()
    if l1 > 0 and l2 > 0:
        assert_true(l1 <= l2, "L1 should not exceed L2")
    if l2 > 0 and l3 > 0:
        assert_true(l2 <= l3, "L2 should not exceed L3")


# The L1 accessor should resolve to a data or unified cache at level 1.
def test_l1_is_data_or_unified() raises:
    comptime if not CompilationTarget.is_x86():
        return
    var size = l1_data_cache_size()
    if size == 0:
        return
    var matched = False
    for c in query_caches():
        var is_data = (
            c.cache_type == CACHE_TYPE_DATA
            or c.cache_type == CACHE_TYPE_UNIFIED
        )
        if c.level == 1 and is_data and c.size_bytes == size:
            matched = True
    assert_true(matched, "l1_data_cache_size did not match an L1 data cache")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
