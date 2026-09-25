"""End-to-end tests for the experimental CPython CrossJIT provider."""

import os
import json
import subprocess
import sys
import warnings

# Unit tests exercise lowering independently of the production profitability
# policy.  Dedicated subprocess tests below verify the default threshold.
os.environ.setdefault("NPY_CROSSJIT_MIN_WORK", "0")

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("PYTHON_JIT_CROSSJIT") != "1",
    reason="CrossJIT requires an explicitly enabled Tier-2 build",
)


def _runtime():
    internal = pytest.importorskip("_testinternalcapi")
    if not hasattr(internal, "crossjit_stats"):
        pytest.skip("CPython was built without the CrossJIT test API")
    return internal


def test_host_region_hit_and_guard_miss():
    """A two-node DAG fuses, while changed shape resumes normal NumPy."""
    internal = _runtime()
    internal.crossjit_reset_stats()

    def fused(left, right, scale, count):
        for _ in range(count):
            result = (left + right) * scale
        return result

    left = np.arange(32.0)
    right = left + 1
    scale = left + 2
    actual = fused(left, right, scale, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(actual, (left + right) * scale)

    changed = np.arange(7.0)
    actual = fused(changed, changed + 1, changed + 2, 2)
    np.testing.assert_array_equal(actual, (changed + changed + 1) * (changed + 2))

    stats = internal.crossjit_stats()
    assert stats["recognized_nodes"] == 2
    assert stats["formed_regions"] == 1
    assert stats["rewritten_regions"] == 1
    assert stats["virtual_values"] == 1
    assert stats["guard_misses"] >= 1
    assert stats["exec_hits"] >= 1


def test_floating_exception_soft_miss():
    """A fused FPE is discarded so NumPy owns warning behavior."""
    internal = _runtime()
    internal.crossjit_reset_stats()

    def fused(left, right, count):
        for _ in range(count):
            result = (left / right) * right
        return result

    left = np.arange(1.0, 33.0)
    right = left + 1
    fused(left, right, internal.TIER2_THRESHOLD * 2)

    with np.errstate(all="warn"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fused(left, np.zeros_like(left), 3)
    assert np.isnan(result).all()
    assert caught
    assert internal.crossjit_stats()["exec_misses"] >= 1


def test_rewrite_releases_trace_observations():
    """A published region must not retain the arrays observed while tracing."""
    internal = _runtime()
    internal.crossjit_reset_stats()

    def fused(left, right, scale, count):
        for _ in range(count):
            result = (left + right) * scale
        return result

    left = np.arange(16.0)
    right = left + 1
    scale = left + 2
    before = tuple(map(sys.getrefcount, (left, right, scale)))
    result = fused(left, right, scale, internal.TIER2_THRESHOLD * 2)
    after = tuple(map(sys.getrefcount, (left, right, scale)))

    np.testing.assert_array_equal(result, (left + right) * scale)
    assert after == before
    assert internal.crossjit_stats()["rewritten_regions"] == 1


def test_unary_node_participates_in_host_region():
    """Unary negative shares Host value identity with a binary producer."""
    internal = _runtime()
    internal.crossjit_reset_stats()

    def fused(left, right, count):
        for _ in range(count):
            result = -(left + right)
        return result

    left = np.arange(24.0)
    right = left + 1
    result = fused(left, right, internal.TIER2_THRESHOLD * 2)

    np.testing.assert_array_equal(result, -(left + right))
    stats = internal.crossjit_stats()
    assert stats["recognized_nodes"] == 2
    assert stats["formed_regions"] == 1
    assert stats["rewritten_regions"] == 1
    assert stats["exec_hits"] >= 1


def test_ufunc_calls_build_the_same_semantic_dag():
    """Exact ufunc calls expose their arguments without provider trace scans."""
    internal = _runtime()
    internal.crossjit_reset_stats()

    def called(left, right, scale, count):
        for _ in range(count):
            result = np.multiply(np.add(left, right), scale)
        return result

    left = np.arange(20.0)
    right = left + 1
    scale = left + 2
    before = tuple(map(sys.getrefcount, (left, right, scale)))
    result = called(left, right, scale, internal.TIER2_THRESHOLD * 2)
    after = tuple(map(sys.getrefcount, (left, right, scale)))

    np.testing.assert_array_equal(result, (left + right) * scale)
    assert after == before
    stats = internal.crossjit_stats()
    assert stats["recognized_nodes"] == 2
    # v3 keeps domain facts out of the semantic plane: this call spelling
    # reaches the same provider-neutral Host proof as operator spelling.
    assert stats["host_proof_created"] >= 1
    assert stats["formed_regions"] == 1


def test_provider_rejection_retries_a_shorter_host_region():
    """An unsupported tail cannot discard a legal elementwise prefix."""
    internal = _runtime()
    internal.crossjit_reset_stats()

    def partially_supported(left, right, scale, exponent, count):
        for _ in range(count):
            result = ((left + right) * scale) ** exponent
        return result

    left = np.arange(1.0, 17.0)
    right = left + 1
    scale = left + 2
    exponent = np.full_like(left, 2.0)
    result = partially_supported(
        left, right, scale, exponent, internal.TIER2_THRESHOLD * 2)

    np.testing.assert_array_equal(result, ((left + right) * scale) ** exponent)
    stats = internal.crossjit_stats()
    assert stats["recognized_nodes"] == 3
    assert stats["rejected_regions"] >= 1
    assert stats["formed_regions"] == 1
    assert stats["rewritten_regions"] == 1
    assert stats["exec_hits"] >= 1


def test_semantics_only_ablation_does_not_rewrite():
    """Semantic analysis can be measured without publishing execution plans."""
    program = r"""
import json
import numpy as np
import _testinternalcapi as internal

internal.crossjit_reset_stats()
left = np.arange(32.0)
right = left + 1
scale = left + 2
for _ in range(internal.TIER2_THRESHOLD * 2):
    result = (left + right) * scale
np.testing.assert_array_equal(result, (left + right) * scale)
print(json.dumps(internal.crossjit_stats()))
"""
    env = os.environ.copy()
    env["PYTHON_JIT_CROSSJIT"] = "semantics"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    stats = json.loads(completed.stdout)
    assert stats["recognized_nodes"] == 2
    assert stats["formed_regions"] == 0
    assert stats["rewritten_regions"] == 0
    assert stats["guard_hits"] == 0
    assert stats["exec_hits"] == 0


def test_operator_comparison_is_modeled_but_not_lowered():
    """Comparison spelling reaches the semantic provider and stays a barrier."""
    internal = _runtime()
    internal.crossjit_reset_stats()

    def compare(left, right, count):
        for _ in range(count):
            result = left < right
        return result

    left = np.arange(24.0)
    right = left + 1
    before = tuple(map(sys.getrefcount, (left, right)))
    result = compare(left, right, internal.TIER2_THRESHOLD * 2)
    after = tuple(map(sys.getrefcount, (left, right)))

    np.testing.assert_array_equal(result, np.less(left, right))
    assert after == before
    stats = internal.crossjit_stats()
    assert stats["recognized_nodes"] == 1
    assert stats["formed_regions"] == 0
    assert stats["rewritten_regions"] == 0


def test_strided_scalar_dag_executes_like_adi_expression():
    """An ADI-shaped DAG consumes a deeper temporary and strided columns."""
    internal = _runtime()
    internal.crossjit_reset_stats()

    def fused(left, right, scale, bias, count):
        for _ in range(count):
            result = scale * left + bias * right
        return result

    matrix = np.arange(64.0).reshape(8, 8)
    left = matrix[:, 1]
    right = matrix[:, 3]
    assert not left.flags.c_contiguous
    result = fused(
        left, right, -0.25, 1.5, internal.TIER2_THRESHOLD * 2)

    np.testing.assert_allclose(result, -0.25 * left + 1.5 * right)
    stats = internal.crossjit_stats()
    assert stats["recognized_nodes"] == 3
    assert stats["formed_regions"] >= 1
    assert stats["rewritten_regions"] >= 1
    assert stats["guard_hits"] >= 1
    assert stats["exec_hits"] >= 1


def test_multidimensional_strided_dag_executes():
    """Stencil-shaped inner slices use guarded per-axis input strides."""
    internal = _runtime()
    internal.crossjit_reset_stats()

    def fused(left, center, right, count):
        for _ in range(count):
            result = left + center * right
        return result

    volume = np.arange(10 * 11 * 12.0).reshape(10, 11, 12)
    left = volume[1:-1, 1:-1, :-2]
    center = volume[1:-1, 1:-1, 1:-1]
    right = volume[1:-1, 1:-1, 2:]
    assert not left.flags.c_contiguous
    result = fused(left, center, right, internal.TIER2_THRESHOLD * 2)

    np.testing.assert_allclose(result, left + center * right)
    stats = internal.crossjit_stats()
    assert stats["recognized_nodes"] == 2
    assert stats["formed_regions"] >= 1
    assert stats["rewritten_regions"] >= 1
    assert stats["exec_hits"] >= 1


def test_broadcast_dag_uses_guarded_zero_strides():
    """Provider lowering owns broadcast-domain guards and zero-stride axes."""
    internal = _runtime()
    internal.crossjit_reset_stats()

    def fused(column, row, scale, count):
        for _ in range(count):
            result = (column + row) * scale
        return result

    column = np.arange(9.0).reshape(9, 1)
    row = np.arange(7.0).reshape(1, 7)
    result = fused(column, row, 0.25, internal.TIER2_THRESHOLD * 2)

    np.testing.assert_allclose(result, (column + row) * 0.25)
    stats = internal.crossjit_stats()
    assert stats["recognized_nodes"] == 2
    assert stats["formed_regions"] >= 1
    assert stats["rewritten_regions"] >= 1
    assert stats["exec_hits"] >= 1


def test_long_dag_survives_host_value_arena_growth():
    """Stable semantic ids repair opaque node references after arena growth."""
    internal = _runtime()
    internal.crossjit_reset_stats()

    def fused(a, b, c, d, e, count):
        for _ in range(count):
            result = (((a + b) * c) + d) * e
        return result

    values = tuple(np.arange(32.0) + offset for offset in range(1, 6))
    result = fused(*values, internal.TIER2_THRESHOLD * 2)

    expected = (((values[0] + values[1]) * values[2]) + values[3]) * values[4]
    np.testing.assert_allclose(result, expected)
    stats = internal.crossjit_stats()
    assert stats["recognized_nodes"] == 4
    assert stats["formed_regions"] >= 1
    assert stats["rewritten_regions"] >= 1
    assert stats["exec_hits"] >= 1


def test_six_node_provider_loop_executes():
    """The NumPy provider guards and lowers the largest specialized loop."""
    internal = _runtime()
    internal.crossjit_reset_stats()

    def fused(a, b, c, d, e, f, g, count):
        for _ in range(count):
            result = (((((a + b) * c) + d) * e) - f) + g
        return result

    values = tuple(np.arange(48.0) + offset for offset in range(1, 8))
    result = fused(*values, internal.TIER2_THRESHOLD * 2)

    expected = (((((values[0] + values[1]) * values[2]) + values[3]) *
                 values[4]) - values[5]) + values[6]
    np.testing.assert_allclose(result, expected)
    stats = internal.crossjit_stats()
    assert stats["recognized_nodes"] == 6
    assert stats["formed_regions"] >= 1
    assert stats["rewritten_regions"] >= 1
    assert stats["guard_hits"] >= 1
    assert stats["exec_hits"] >= 1


def test_default_short_region_threshold_declines_tiny_arrays():
    """The production policy leaves an unprofitable tiny DAG on NumPy ufuncs."""
    program = r"""
import json
import numpy as np
import _testinternalcapi as internal

internal.crossjit_reset_stats()
left = np.arange(32.0)
right = left + 1
scale = left + 2
for _ in range(internal.TIER2_THRESHOLD * 2):
    result = (left + right) * scale
np.testing.assert_array_equal(result, (left + right) * scale)
print(json.dumps(internal.crossjit_stats()))
"""
    env = os.environ.copy()
    env.pop("NPY_CROSSJIT_MIN_WORK", None)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    stats = json.loads(completed.stdout)
    assert stats["recognized_nodes"] == 2
    assert stats["formed_regions"] == 0
    assert stats["rewritten_regions"] == 0
    assert stats["exec_hits"] == 0


def test_exact_python_int_scalars_and_range_guard():
    """Safe integer scalars lower; a wider value side-exits before execution."""
    internal = _runtime()
    internal.crossjit_reset_stats()

    def fused(array, scale, factor, count):
        for _ in range(count):
            result = (array + scale) * factor
        return result

    array = np.arange(24.0)
    result = fused(array, 3, 2, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_allclose(result, (array + 3) * 2)
    before = internal.crossjit_stats()
    assert before["exec_hits"] >= 1

    wide = 2**54
    fallback = fused(array, wide, 2, 2)
    np.testing.assert_allclose(fallback, (array + wide) * 2)
    after = internal.crossjit_stats()
    assert after["guard_misses"] > before["guard_misses"]
