"""End-to-end tests for the experimental CPython CrossJIT provider."""

import os
import json
import subprocess
import sys
import warnings

# Unit tests exercise lowering independently of the production profitability
# policy.  Dedicated subprocess tests below verify the default threshold.
os.environ.setdefault("NPY_CROSSJIT_MIN_WORK", "0")
os.environ.setdefault("NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK", "1")

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


def _provider_runtime():
    from numpy._core import _multiarray_umath as umath
    if not hasattr(umath, "_crossjit_stats"):
        pytest.skip("NumPy was built without the CrossJIT provider API")
    return umath


def test_provider_optimization_funnel_telemetry(monkeypatch):
    """Host and Provider counters distinguish every optimization stage."""
    internal = _runtime()
    provider = _provider_runtime()
    monkeypatch.setenv("NPY_CROSSJIT_DIAGNOSTICS", "1")
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()

    def fused(left, right, scale, count):
        for _ in range(count):
            result = (left + right) * scale
        return result

    left = np.arange(64.0)
    result = fused(
        left, left + 1, left + 2, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_allclose(result, (left + left + 1) * (left + 2))

    host = internal.crossjit_stats()
    stats = provider._crossjit_stats()
    assert host["recognized_nodes"] == 2
    assert host["host_regions_proven"] == 1
    assert stats["regions_prepare_called"] == 1
    assert stats["regions_provider_accepted"] == 1
    assert stats["guard_plans_lowered"] == 1
    assert stats["execution_plans_lowered"] == 1
    assert host["rewritten_regions"] == 1
    assert host["published_regions"] == 1
    assert stats["regions_executed"] >= 1
    assert stats["guard_hit"] >= 1
    assert stats["exec_hit"] >= 1
    assert host["published_regions"] <= stats["regions_provider_accepted"]
    assert (stats["regions_provider_accepted"] <=
            stats["regions_prepare_called"])
    assert (stats["exec_hit"] + stats["exec_miss"] + stats["exec_error"] ==
            stats["regions_executed"])
    assert stats["diagnostics_enabled"] is True
    assert len(stats["candidates"]) == 1
    assert len(stats["observed_signatures"]) == 1

    observed = stats["observed_signatures"][0]
    assert observed["prepare_count"] == 1
    assert observed["accepted_count"] == 1
    assert observed["execution_count"] == stats["exec_hit"]
    assert observed["generated_hit_count"] == stats["generated_exec_hit"]
    assert (observed["generic_fallback_count"] ==
            stats["generic_fallback_hit"])
    assert observed["min_elements"] == 64
    assert observed["max_elements"] == 64
    assert len(observed["element_buckets"]) == 14
    assert sum(observed["element_buckets"]) == observed["execution_count"]
    assert len(observed["prepare_element_buckets"]) == 14
    assert sum(observed["prepare_element_buckets"]) == 1
    assert observed["refined_layout_class"] == 1  # FULL_CONTIGUOUS
    assert observed["iteration_rank_class"] == 1
    assert observed["iteration_descriptor_version"] == 1
    assert observed["outer_shape_class"] == 0  # NONE for flat iteration
    assert observed["outer_count"] == 1
    assert observed["inner_extent"] == 64
    assert observed["access_kinds"] == (1, 1, 1)  # CONTIGUOUS
    assert "benchmark" not in json.dumps(observed).lower()

    candidate = stats["last_candidate"]
    assert candidate["region_length"] == 2
    assert candidate["element_count"] == 64
    assert candidate["layout_class"] == 1  # CONTIGUOUS_SAME_SHAPE
    assert candidate["dtype_class"] == 2  # float64
    assert candidate["operand_count"] == 3
    assert candidate["result_count"] == 1
    assert candidate["internal_value_count"] == 1
    assert candidate["eliminable_materialization_count"] == 1
    assert candidate["guard_count"] == 3
    assert candidate["unique_guard_count"] == 3
    assert candidate["profitability_score"] == 64
    assert candidate["profitability_threshold"] == 0
    assert candidate["prepare_decision"] == 1
    assert stats["candidates"][0] == candidate


def test_provider_detailed_diagnostics_are_opt_in(monkeypatch):
    """The ordinary counter path does not retain candidate records or timers."""
    internal = _runtime()
    provider = _provider_runtime()
    monkeypatch.delenv("NPY_CROSSJIT_DIAGNOSTICS", raising=False)
    monkeypatch.setenv("NPY_CROSSJIT_TELEMETRY_TIMING", "1")
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()

    def fused(left, right, scale, count):
        for _ in range(count):
            result = (left + right) * scale
        return result

    left = np.arange(64.0)
    fused(left, left + 1, left + 2, internal.TIER2_THRESHOLD * 2)
    stats = provider._crossjit_stats()
    assert stats["regions_provider_accepted"] == 1
    assert stats["exec_hit"] >= 1
    assert stats["diagnostics_enabled"] is False
    assert stats["timing_enabled"] is False
    assert stats["last_candidate"] is None
    assert stats["candidates"] == ()
    assert stats["guard_checks"] == 0
    assert stats["prepare_time_ns"] == 0
    assert stats["guard_lower_time_ns"] == 0
    assert stats["execution_lower_time_ns"] == 0
    assert stats["guard_time_ns"] == 0
    assert stats["region_time_ns"] == 0


def test_host_region_hit_and_guard_miss():
    """A two-node DAG fuses, while changed shape resumes normal NumPy."""
    internal = _runtime()
    provider = _provider_runtime()
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()

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
    provider_stats = provider._crossjit_stats()
    assert provider_stats["guard_hit"] >= 1
    assert provider_stats["guard_miss"] >= 1
    assert provider_stats["exec_hit"] >= 1


def test_floating_exception_soft_miss():
    """A fused FPE is discarded so NumPy owns warning behavior."""
    internal = _runtime()
    provider = _provider_runtime()
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()

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
    assert provider._crossjit_stats()["exec_miss"] >= 1


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
    provider = _provider_runtime()
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()

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
    provider_stats = provider._crossjit_stats()
    assert provider_stats["regions_rejected_unsupported_node"] >= 1
    assert provider_stats["regions_provider_accepted"] == 1


def test_profile_telemetry_identifies_unsupported_operation(monkeypatch):
    """Diagnostic coverage retains unsupported atomic operation identity."""
    internal = _runtime()
    provider = _provider_runtime()
    monkeypatch.setenv("NPY_CROSSJIT_DIAGNOSTICS", "1")
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()

    def partially_supported(left, right, scale, exponent, count):
        for _ in range(count):
            result = ((left + right) * scale) ** exponent
        return result

    left = np.arange(1.0, 17.0)
    result = partially_supported(
        left, left + 1, left + 2, np.full_like(left, 2.0),
        internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(
        result, ((left + left + 1) * (left + 2)) ** 2.0)

    stats = provider._crossjit_stats()
    assert stats["unsupported_node_records_dropped"] == 0
    assert len(stats["unsupported_nodes"]) == 1
    record = stats["unsupported_nodes"][0]
    assert record["semantic_id"] == 5  # numpy.power
    assert record["occurrence_count"] >= 1
    assert record["dtype_class"] == 2
    assert record["refined_layout_class"] == 1
    assert record["region_length"] == 3


def test_provider_reports_unsupported_contract():
    """A semantically legal DAG can remain outside NumPy's fixed contract."""
    internal = _runtime()
    provider = _provider_runtime()
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()

    def mixed(array, scalar, scale, count):
        for _ in range(count):
            result = (array + scalar) * scale
        return result

    array = np.arange(32, dtype=np.int64)
    result = mixed(array, 0.5, array + 1, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(result, (array + 0.5) * (array + 1))
    stats = provider._crossjit_stats()
    assert stats["regions_prepare_called"] >= 1
    assert stats["regions_rejected_unsupported_contract"] >= 1
    assert stats["regions_provider_accepted"] == 0
    assert stats["regions_executed"] == 0


def test_provider_execution_error_is_observable(monkeypatch):
    """Provider ERROR is counted and propagates without publishing a result."""
    internal = _runtime()
    provider = _provider_runtime()
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()

    def fused(left, right, scale, count):
        for _ in range(count):
            result = (left + right) * scale
        return result

    left = np.arange(32.0)
    right = left + 1
    scale = left + 2
    fused(left, right, scale, internal.TIER2_THRESHOLD * 2)
    before = provider._crossjit_stats()
    monkeypatch.setenv("NPY_CROSSJIT_TEST_EXEC_ERROR", "1")
    with pytest.raises(RuntimeError, match="injected NumPy CrossJIT"):
        fused(left, right, scale, 2)
    after = provider._crossjit_stats()
    assert after["exec_error"] == before["exec_error"] + 1
    assert after["regions_executed"] == before["regions_executed"] + 1


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


def test_profiled_inner_contiguous_kernel_is_descriptor_selected(monkeypatch):
    """PF4 uses the generated descriptor registry and guards exact strides."""
    internal = _runtime()
    provider = _provider_runtime()
    monkeypatch.setenv("NPY_CROSSJIT_PROFILED_FORCE_GENERATED", "1")
    monkeypatch.setenv("NPY_CROSSJIT_DIAGNOSTICS", "1")
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()

    def fused(left, right, scale, count):
        for _ in range(count):
            result = scale * (left - right)
        return result

    backing = np.arange(3 * 59 * 60.0).reshape(3, 59, 60)
    left, right, scale = (item[:, :59] for item in backing)
    assert all(not item.flags.c_contiguous for item in (left, right, scale))
    before = tuple(map(sys.getrefcount, (left, right, scale)))
    result = fused(
        left, right, scale, internal.TIER2_THRESHOLD * 2)
    after = tuple(map(sys.getrefcount, (left, right, scale)))

    np.testing.assert_array_equal(result, scale * (left - right))
    assert result.flags.c_contiguous
    assert after == before
    stats = provider._crossjit_stats()
    assert stats["generated_registry_hit"] == 1
    assert stats["generated_exec_hit"] == stats["exec_hit"]
    assert stats["output_allocations"] == stats["exec_hit"]
    assert stats["internal_allocations"] == 0
    host_stats = internal.crossjit_stats()
    assert host_stats["virtual_values"] == 1
    assert host_stats["materialized_values"] == 1
    observed = stats["observed_signatures"][0]
    assert observed["refined_layout_class"] == 2  # INNER_CONTIGUOUS
    assert observed["iteration_rank_class"] == 2  # OUTER_INNER
    assert observed["outer_shape_class"] == 1  # AFFINE_ROWS
    assert observed["access_kinds"] == (1, 1, 1)

    contiguous = tuple(np.ascontiguousarray(item) for item in
                       (left, right, scale))
    changed = fused(*contiguous, 2)
    np.testing.assert_array_equal(
        changed, contiguous[2] * (contiguous[0] - contiguous[1]))
    assert provider._crossjit_stats()["guard_miss"] >= 1


def test_profiled_inner_production_policy_is_specialization_specific():
    """PF4 production admission is keyed by the execution descriptor."""
    program = r"""
import json
import numpy as np
import _testinternalcapi as internal
from numpy._core import _multiarray_umath as umath

def observed(left, right, scale, count):
    for _ in range(count):
        result = scale * (left - right)
    return result

def validation(left, right, scale, count):
    for _ in range(count):
        result = (left + right) * scale
    return result

def unmeasured(left, right, scale, count):
    for _ in range(count):
        result = scale * (left - right)
    return result

def run(function, values, expected):
    internal.crossjit_reset_stats()
    umath._crossjit_reset_stats()
    result = function(*values, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(result, expected)
    return {
        "host": internal.crossjit_stats(),
        "provider": umath._crossjit_stats(),
    }

backing = np.arange(3 * 59 * 60.0).reshape(3, 59, 60)
left, right, scale = (item[:, :59] for item in backing)
observed_stats = run(observed, (left, right, scale), scale * (left - right))
validation_stats = run(
    validation, (left, right, scale), (left + right) * scale)

other = np.arange(3 * 58 * 60.0).reshape(3, 58, 60)
other_left, other_right, other_scale = (item[:, :59] for item in other)
unmeasured_stats = run(
    unmeasured,
    (other_left, other_right, other_scale),
    other_scale * (other_left - other_right),
)
print(json.dumps({
    "observed": observed_stats,
    "validation": validation_stats,
    "unmeasured": unmeasured_stats,
}))
"""
    env = os.environ.copy()
    for name in (
        "NPY_CROSSJIT_MIN_WORK",
        "NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK",
        "NPY_CROSSJIT_PROFILED_FORCE_GENERATED",
    ):
        env.pop(name, None)
    env["NPY_CROSSJIT_DIAGNOSTICS"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    results = json.loads(completed.stdout)

    host = results["observed"]["host"]
    provider = results["observed"]["provider"]
    assert host["rewritten_regions"] == 1
    assert host["virtual_values"] == 1
    assert host["materialized_values"] == 1
    assert provider["generated_registry_hit"] == 1
    assert provider["generated_exec_hit"] == provider["exec_hit"]
    assert provider["output_allocations"] == provider["exec_hit"]
    assert provider["internal_allocations"] == 0

    # PF9 keeps this zero-frequency seed in validation generation but removes
    # it from the production binary.
    validation_stats = results["validation"]
    assert validation_stats["host"]["rewritten_regions"] == 0
    assert validation_stats["provider"]["regions_rejected_unprofitable"] == 1
    assert validation_stats["provider"]["regions_executed"] == 0

    unmeasured_stats = results["unmeasured"]
    assert unmeasured_stats["host"]["rewritten_regions"] == 0
    assert (
        unmeasured_stats["provider"]["regions_rejected_unprofitable"] == 1
    )
    assert unmeasured_stats["provider"]["regions_executed"] == 0


def test_profiled_inner_special_values_and_fpe_soft_miss(monkeypatch):
    """The generated outer/inner kernel preserves FP and replay semantics."""
    internal = _runtime()
    provider = _provider_runtime()
    monkeypatch.setenv("NPY_CROSSJIT_PROFILED_FORCE_GENERATED", "1")

    def fused(left, right, scale, count):
        for _ in range(count):
            result = scale * (left - right)
        return result

    backing = np.ones((3, 59, 60), dtype=np.float64)
    left, right, scale = (item[:, :59] for item in backing)
    left.flat[:6] = [np.nan, np.inf, -np.inf, -0.0, 0.0, 3.0]
    right.flat[:6] = [1.0, np.inf, -2.0, 0.0, -0.0, np.nan]
    scale.flat[:6] = [2.0, 1.0, -1.0, -1.0, 1.0, 0.0]

    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    with np.errstate(all="ignore"):
        result = fused(left, right, scale, internal.TIER2_THRESHOLD * 2)
        expected = scale * (left - right)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    stats = provider._crossjit_stats()
    assert stats["generated_registry_hit"] == 1
    # Invalid arithmetic is detected after the generated attempt; the Host
    # then replays ordinary NumPy so it retains warning/FPE authority.
    assert stats["exec_miss"] >= 1
    assert internal.crossjit_stats()["exec_misses"] >= 1

    safe = np.arange(3 * 59 * 60.0).reshape(3, 59, 60)
    safe_left, safe_right, safe_scale = (item[:, :59] for item in safe)
    fused(safe_left, safe_right, safe_scale, internal.TIER2_THRESHOLD * 2)
    with np.errstate(all="warn"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fpe_result = fused(
            safe_left,
            safe_left,
            np.full_like(safe_scale, np.inf),
            2,
        )
    assert np.isnan(fpe_result).all()
    assert caught
    assert provider._crossjit_stats()["exec_miss"] >= 1


def test_profiled_scalar_and_zero_stride_production_policy():
    """PF5 admits only exact measured scalar/broadcast descriptors."""
    program = r"""
import json
import numpy as np
import _testinternalcapi as internal
from numpy._core import _multiarray_umath as umath

def scalar_inner(left, scalar, right, count):
    for _ in range(count):
        result = right + left * scalar
    return result

def zero_inner(column, left, right, count):
    for _ in range(count):
        result = (column + left) * right
    return result

def zero_outer(row, left, right, count):
    for _ in range(count):
        result = (row + left) * right
    return result

def scalar_unmeasured(left, scalar, right, count):
    for _ in range(count):
        result = right + left * scalar
    return result

def run(function, values, expected):
    internal.crossjit_reset_stats()
    umath._crossjit_reset_stats()
    before = tuple(map(__import__("sys").getrefcount, values))
    result = function(*values, internal.TIER2_THRESHOLD * 2)
    after = tuple(map(__import__("sys").getrefcount, values))
    np.testing.assert_allclose(result, expected)
    assert before == after
    return internal.crossjit_stats(), umath._crossjit_stats()

backing = np.arange(2 * 59 * 60.0).reshape(2, 59, 60)
left, right = (item[:, :59] for item in backing)
scalar = run(scalar_inner, (left, 0.25, right), right + left * 0.25)
column = np.arange(59.0).reshape(59, 1)
zero = run(zero_inner, (column, left, right), (column + left) * right)
row = np.arange(59.0).reshape(1, 59)
mismatch = run(zero_outer, (row, left, right), (row + left) * right)

other = np.arange(2 * 58 * 60.0).reshape(2, 58, 60)
other_left, other_right = (item[:, :59] for item in other)
unmeasured = run(
    scalar_unmeasured,
    (other_left, 0.25, other_right),
    other_right + other_left * 0.25,
)
print(json.dumps({
    "scalar": scalar,
    "zero": zero,
    "mismatch": mismatch,
    "unmeasured": unmeasured,
}))
"""
    env = os.environ.copy()
    for name in (
        "NPY_CROSSJIT_MIN_WORK",
        "NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK",
        "NPY_CROSSJIT_PROFILED_FORCE_GENERATED",
    ):
        env.pop(name, None)
    env["NPY_CROSSJIT_DIAGNOSTICS"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", program], check=True,
        capture_output=True, text=True, env=env)
    results = json.loads(completed.stdout)

    host, provider = results["scalar"]
    assert host["rewritten_regions"] == 1
    assert host["virtual_values"] == 1
    assert host["materialized_values"] == 1
    assert provider["generated_registry_hit"] == 1
    assert provider["generated_exec_hit"] == provider["exec_hit"]
    assert provider["output_allocations"] == provider["exec_hit"]
    assert provider["internal_allocations"] == 0
    assert provider["observed_signatures"][0]["access_kinds"] == [1, 2, 1]

    # The zero-stride candidate is a zero-frequency PF5 validation artifact,
    # and PF9 deliberately excludes it from the production artifact set.
    for name in ("zero", "mismatch", "unmeasured"):
        host, provider = results[name]
        assert host["rewritten_regions"] == 0
        assert provider["regions_rejected_unprofitable"] == 1
        assert provider["regions_executed"] == 0


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


@pytest.mark.parametrize(
    ("node_count", "expression"),
    [
        (2, lambda v: (v[0] + v[1]) * v[2]),
        (3, lambda v: (v[0] + v[1]) * v[2] - v[3]),
        (4, lambda v: ((v[0] + v[1]) * v[2] - v[3]) / v[4]),
        (5, lambda v: ((v[0] + v[1]) * v[2] - v[3]) / v[4] + v[5]),
        (6, lambda v: (((v[0] + v[1]) * v[2] - v[3]) / v[4] +
                       v[5]) * v[6]),
    ],
)
def test_contiguous_float64_straight_line_validation_fallbacks(
        node_count, expression):
    """Excluded zero-frequency validation chains retain generic correctness."""
    internal = _runtime()
    provider = _provider_runtime()

    def fused_2(a, b, c, count):
        for _ in range(count):
            result = (a + b) * c
        return result

    def fused_3(a, b, c, d, count):
        for _ in range(count):
            result = (a + b) * c - d
        return result

    def fused_4(a, b, c, d, e, count):
        for _ in range(count):
            result = ((a + b) * c - d) / e
        return result

    def fused_5(a, b, c, d, e, f, count):
        for _ in range(count):
            result = ((a + b) * c - d) / e + f
        return result

    def fused_6(a, b, c, d, e, f, g, count):
        for _ in range(count):
            result = (((a + b) * c - d) / e + f) * g
        return result

    functions = {2: fused_2, 3: fused_3, 4: fused_4, 5: fused_5, 6: fused_6}
    values = tuple(np.arange(257.0) / 257 + i + 1
                   for i in range(node_count + 1))
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    result = functions[node_count](
        *values, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(result, expression(values))
    stats = provider._crossjit_stats()
    assert stats["exec_hit"] >= 1
    assert stats["fixed_fast_path_hit"] == 0
    assert stats["fixed_generic_path_hit"] == stats["exec_hit"]
    assert stats["fixed_fast_plans_built"] == 0
    assert stats["fixed_generic_plans_built"] == 1
    assert stats["generated_registry_hit"] == 0
    assert stats["generated_registry_miss"] == 1
    assert stats["generated_exec_hit"] == 0
    assert stats["generic_fallback_hit"] == stats["exec_hit"]
    assert stats["output_allocations"] == stats["exec_hit"]
    assert stats["internal_allocations"] == 0


def test_contiguous_float64_branched_and_scalar_validation_fallbacks():
    """Excluded validation artifacts retain generic fallback correctness."""
    internal = _runtime()
    provider = _provider_runtime()

    def branched(a, b, c, d, count):
        for _ in range(count):
            result = a * b + c * d
        return result

    def array_scalar(a, scalar, b, count):
        for _ in range(count):
            result = (a + scalar) * b
        return result

    def scalar_array(scalar, a, b, count):
        for _ in range(count):
            result = (scalar * a) + b
        return result

    a = np.arange(193.0) / 193 + 1
    arrays = (a, a + 1, a + 2, a + 3)
    cases = (
        (branched, arrays, arrays[0] * arrays[1] + arrays[2] * arrays[3], False),
        (array_scalar, (arrays[0], 0.25, arrays[1]),
         (arrays[0] + 0.25) * arrays[1], False),
        (scalar_array, (0.25, arrays[0], arrays[1]),
         0.25 * arrays[0] + arrays[1], False),
    )
    for function, args, expected, generated in cases:
        internal.crossjit_reset_stats()
        provider._crossjit_reset_stats()
        result = function(*args, internal.TIER2_THRESHOLD * 2)
        np.testing.assert_array_equal(result, expected)
        stats = provider._crossjit_stats()
        if generated:
            assert stats["generated_registry_hit"] == 1
            assert stats["generated_exec_hit"] == stats["exec_hit"]
            assert stats["generic_fallback_hit"] == 0
        else:
            assert stats["generated_registry_miss"] == 1
            assert stats["generated_exec_hit"] == 0
            assert stats["generic_fallback_hit"] == stats["exec_hit"]
        assert stats["output_allocations"] == stats["exec_hit"]
        assert stats["internal_allocations"] == 0


def test_fixed_generic_fallback_preserves_special_values():
    """An unsupported signature stays on the fixed fallback exactly."""
    internal = _runtime()
    provider = _provider_runtime()

    def add_negative(left, right, count):
        for _ in range(count):
            result = -(left + right)
        return result

    finite = np.arange(6.0) + 1
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    finite_result = add_negative(
        finite, finite + 1, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(finite_result, -(finite + finite + 1))
    hit_stats = provider._crossjit_stats()
    assert hit_stats["exec_hit"] >= 1
    assert hit_stats["fixed_fast_path_hit"] == 0
    assert hit_stats["fixed_generic_path_hit"] == hit_stats["exec_hit"]
    assert hit_stats["fixed_fast_plans_built"] == 0
    assert hit_stats["fixed_generic_plans_built"] == 1
    assert hit_stats["generated_registry_hit"] == 0
    assert hit_stats["generated_registry_miss"] == 1
    assert hit_stats["generated_exec_hit"] == 0
    assert hit_stats["generic_fallback_hit"] == hit_stats["exec_hit"]

    left = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 3.0])
    right = np.array([1.0, np.inf, -2.0, -0.0, 0.0, np.nan])
    result = add_negative(left, right, 2)
    expected = -(left + right)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    stats = provider._crossjit_stats()
    assert stats["fixed_fast_path_hit"] == 0
    assert stats["fixed_generic_path_hit"] >= stats["exec_hit"]
    assert stats["output_allocations"] >= stats["exec_hit"]
    assert stats["internal_allocations"] == 0


def test_minimum_maximum_special_values_remain_correct():
    """Recognized min/max operations retain NaN and signed-zero behavior."""
    internal = _runtime()

    def minmax(left, right, tail, count):
        for _ in range(count):
            result = np.maximum(np.minimum(left, right), tail)
        return result

    left = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 3.0])
    right = np.array([1.0, np.inf, 2.0, 0.0, -0.0, np.nan])
    tail = np.array([-2.0, 4.0, -3.0, -0.0, 0.0, 1.0])
    internal.crossjit_reset_stats()
    result = minmax(left, right, tail, internal.TIER2_THRESHOLD * 2)
    expected = np.maximum(np.minimum(left, right), tail)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    assert internal.crossjit_stats()["recognized_nodes"] == 2


def test_production_decisiondb_policy_without_forced_admission():
    """Generated, generic, and original choices follow measured envelopes."""
    program = r"""
import json
import numpy as np
import _testinternalcapi as internal
from numpy._core import _multiarray_umath as umath

def fused(left, scalar, right, count):
    for _ in range(count):
        result = right + left * scalar
    return result

def fused_tiny(left, scalar, right, count):
    for _ in range(count):
        result = right + left * scalar
    return result

def array_scalar(left, scalar, right, count):
    for _ in range(count):
        result = (left + scalar) * right
    return result

def scalar_array(scalar, left, right, count):
    for _ in range(count):
        result = (scalar * left) + right
    return result

def run(function, values, expected):
    internal.crossjit_reset_stats()
    umath._crossjit_reset_stats()
    result = function(*values, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(result, expected)
    return {
        "host": internal.crossjit_stats(),
        "provider": umath._crossjit_stats(),
    }

left = np.arange(32.0)
generated = run(fused, (left, 0.25, left + 1),
                (left + 1) + left * 0.25)

tiny = np.arange(8.0)
unmeasured = run(fused_tiny, (tiny, 0.25, tiny + 1),
                 (tiny + 1) + tiny * 0.25)

medium = np.arange(256.0)
generic_profitable = run(array_scalar, (medium, 0.25, medium + 1),
                         (medium + 0.25) * (medium + 1))
generic_regressive = run(scalar_array, (0.25, medium, medium + 1),
                         0.25 * medium + medium + 1)
print(json.dumps({
    "generated": generated,
    "unmeasured": unmeasured,
    "generic_profitable": generic_profitable,
    "generic_regressive": generic_regressive,
}))
"""
    env = os.environ.copy()
    env.pop("NPY_CROSSJIT_MIN_WORK", None)
    env.pop("NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK", None)
    env["NPY_CROSSJIT_DIAGNOSTICS"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    stats = json.loads(completed.stdout)

    generated = stats["generated"]
    assert generated["host"]["rewritten_regions"] == 1
    assert generated["provider"]["generated_registry_hit"] == 1
    assert generated["provider"]["generated_exec_hit"] >= 1

    unmeasured = stats["unmeasured"]
    assert unmeasured["host"]["rewritten_regions"] == 0
    assert unmeasured["provider"]["regions_rejected_unprofitable"] == 1
    assert unmeasured["provider"]["regions_executed"] == 0

    generic = stats["generic_profitable"]
    assert generic["host"]["rewritten_regions"] == 1
    assert generic["provider"]["generated_registry_hit"] == 0
    assert generic["provider"]["generic_fallback_hit"] >= 1

    regressive = stats["generic_regressive"]
    assert regressive["host"]["rewritten_regions"] == 0
    assert regressive["provider"]["regions_rejected_unprofitable"] == 1
    assert regressive["provider"]["generic_fallback_hit"] == 0


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
