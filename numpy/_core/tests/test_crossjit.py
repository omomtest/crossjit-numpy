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


@pytest.mark.parametrize("operation", ["matmul", "add", "mean", "reduce"])
@pytest.mark.parametrize("force_generic", [False, True])
def test_scalar_publication_preserves_result_type_observers(operation, force_generic):
    """Conditional projections cannot narrow observers on original execution."""
    _runtime()
    _provider_runtime()
    program = r'''
import json
import numpy as np
import _testinternalcapi as internal
from numpy._core import _multiarray_umath as provider
operation = OPERATION
if operation == 'matmul':
    def exercise(a, b, count):
        for _ in range(count):
            result = a @ b
            observed = isinstance(result, np.ndarray)
        return observed, type(result)
    args = np.arange(3.0), np.arange(3.0)
elif operation == 'add':
    def exercise(a, b, count):
        for _ in range(count):
            result = a + b
            observed = isinstance(result, np.ndarray)
        return observed, type(result)
    args = np.array(1.0), np.array(2.0)
elif operation == 'mean':
    def exercise(a, b, count):
        for _ in range(count):
            result = np.mean(a)
            observed = isinstance(result, np.ndarray)
        return observed, type(result)
    args = np.arange(3.0), None
else:
    def exercise(a, b, count):
        for _ in range(count):
            result = np.maximum.reduce(a)
            observed = isinstance(result, np.ndarray)
        return observed, type(result)
    args = np.arange(3.0), None
expected = exercise(*args, 1)
assert expected == (False, np.float64), expected
provider._crossjit_reset_stats()
actual = exercise(*args, internal.TIER2_THRESHOLD * 3)
assert actual == expected, (expected, actual)
stats = provider._crossjit_stats()
assert stats['exec_hit'] == 0, stats
assert stats['exec_miss'] == stats['exec_error'] == 0, stats
print(json.dumps({'scalar_type': actual[1].__name__}))
'''.replace("OPERATION", repr(operation))
    env = {k: v for k, v in os.environ.items() if not k.startswith("NPY_CROSSJIT")}
    if force_generic:
        env.update(NPY_CROSSJIT_MIN_WORK="0", NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK="1")
    result = subprocess.run([sys.executable, "-c", program], env=env,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["scalar_type"] == "float64"


def test_declined_projection_preserves_later_array_rank_change():
    """An ndarray exemplar cannot prove later zero-dimensional result typing."""
    _runtime()
    _provider_runtime()
    program = r'''
import numpy as np
import _testinternalcapi as internal
def exercise(a, b, count):
    for _ in range(count):
        result = a + b
        observed = isinstance(result, np.ndarray)
    return observed, type(result)
assert exercise(np.arange(3.0), np.arange(3.0), internal.TIER2_THRESHOLD*3) == (True, np.ndarray)
assert exercise(np.array(1.0), np.array(2.0), 20) == (False, np.float64)
'''
    env = {k: v for k, v in os.environ.items() if not k.startswith("NPY_CROSSJIT")}
    env["NPY_CROSSJIT_MIN_WORK"] = "1000000000"
    subprocess.run([sys.executable, "-c", program], env=env,
                   capture_output=True, text=True, check=True)


def test_long_diagnostic_dag_survives_a_following_signature_record():
    """The diagnostic node storage must cover the advertised 12-node domain."""
    _runtime()
    _provider_runtime()
    program = r'''
import json
import numpy as np
import _testinternalcapi as internal
from numpy._core import _multiarray_umath as provider
def nine(a, b, c, count):
    for _ in range(count):
        result = a+b+c+a+b+c+a+b+c+a
    return result
def two(a, b, count):
    for _ in range(count):
        result = a+b
    return result
args = tuple(np.arange(24.0)+i for i in (1,2,3))
expected = nine(*args, 1).tobytes()
provider._crossjit_reset_stats()
assert nine(*args, internal.TIER2_THRESHOLD*2).tobytes() == expected
two(args[0], args[1], internal.TIER2_THRESHOLD*2)
assert nine(*args, 4).tobytes() == expected
stats = provider._crossjit_stats()
records = [r for r in stats['observed_signatures'] if r['node_count'] == 9]
assert records, stats
assert stats['exec_miss'] == stats['exec_error'] == 0
for record in records:
    assert len(record['nodes']) == 9
    assert all(node['output'] == record['input_count']+i
               for i, node in enumerate(record['nodes'])), record
    assert record['generic_fallback_count'] > 0
print(json.dumps({'records': len(records)}))
'''
    env = {k:v for k,v in os.environ.items() if not k.startswith("NPY_CROSSJIT")}
    env.update(NPY_CROSSJIT_DIAGNOSTICS="1", NPY_CROSSJIT_MIN_WORK="0",
               NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK="1")
    result = subprocess.run([sys.executable, "-c", program], env=env,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["records"] > 0


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
    # Optional combined execution consumes the Host-scheduled guard batch once.
    assert stats["guard_hit"] == stats["exec_hit"]
    assert host["combined_guard_exec_calls"] == stats["exec_hit"]
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
    assert observed["sample_operand_strides"] == ((8, 0), (8, 0), (8, 0))
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


def test_exact_numpy_float64_scalar_generated_execution_and_type_guard(monkeypatch):
    """Strong scalar promotion and exact runtime scalar type stay Provider-owned."""
    internal = _runtime()
    provider = _provider_runtime()
    monkeypatch.delenv("NPY_CROSSJIT_MIN_WORK", raising=False)
    monkeypatch.delenv("NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK", raising=False)
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()

    def fused(scalar, array, bias, count):
        for _ in range(count):
            result = bias + scalar * array
        return result

    array = np.arange(128.0)
    bias = array + 2
    scalar = np.float64(.375)
    actual = fused(scalar, array, bias, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(actual, bias + scalar * array)
    assert provider._crossjit_stats()["generated_exec_hit"] > 0
    before = internal.crossjit_stats()["guard_misses"]
    actual = fused(.5, array, bias, 2)
    np.testing.assert_array_equal(actual, bias + .5 * array)
    assert internal.crossjit_stats()["guard_misses"] > before


def test_numpy_float64_scalar_does_not_use_weak_float32_promotion():
    internal = _runtime()
    provider = _provider_runtime()
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()

    def fused(scalar, array, bias, count):
        for _ in range(count):
            result = bias + scalar * array
        return result

    array = np.arange(128, dtype=np.float32)
    scalar = np.float64(1.0 + 2.0**-30)
    expected = array + scalar * array
    actual = fused(scalar, array, array, internal.TIER2_THRESHOLD * 2)
    assert actual.dtype == expected.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)
    assert provider._crossjit_stats()["regions_rejected_unsupported_contract"] > 0
    assert provider._crossjit_stats()["exec_hit"] == 0


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


@pytest.mark.parametrize("reverse", [False, True])
def test_affine_generated_flat_runtime_strides_and_guard(monkeypatch, reverse):
    internal = _runtime()
    provider = _provider_runtime()
    monkeypatch.setenv("NPY_CROSSJIT_PROFILED_FORCE_GENERATED", "1")

    namespace = {}
    exec("def fused(scale, array, bias, count):\n"
         "    for _ in range(count):\n"
         "        result = scale * array + bias\n"
         "    return result\n", namespace)
    fused = namespace["fused"]

    # One normalized artifact, with concrete byte strides supplied by the plan.
    values = np.linspace(1., 2., 40000).reshape(400, 100)[:, 2]
    if reverse:
        values = values[::-1]
    scale, bias = np.float64(.25), np.float64(.5)
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    result = fused(scale, values, bias, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(result, scale * values + bias)
    stats = provider._crossjit_stats()
    assert stats["generated_exec_hit"] > 0
    assert stats["generic_fallback_hit"] == 0
    assert stats["internal_allocations"] == 0
    replacement = values.copy()
    result = fused(scale, replacement, bias, 2)
    np.testing.assert_array_equal(result, scale * replacement + bias)
    assert provider._crossjit_stats()["guard_miss"] > 0


@pytest.mark.parametrize("reverse", [False, True])
def test_affine_generated_mixed_rows_and_fp_replay(monkeypatch, reverse):
    internal = _runtime()
    provider = _provider_runtime()
    monkeypatch.setenv("NPY_CROSSJIT_PROFILED_FORCE_GENERATED", "1")

    namespace = {}
    exec("def fused(left, right, scale, count):\n"
         "    for _ in range(count):\n"
         "        result = scale * (left - right)\n"
         "    return result\n", namespace)
    fused = namespace["fused"]

    backing = np.linspace(1., 2., 4 * 29 * 90).reshape(4, 29, 90)
    left, right = backing[0, :, :87:3], backing[1, :, :87:3]
    if reverse:
        left, right = left[::-1, ::-1], right[::-1, ::-1]
    scale = backing[2, :, :29].copy()
    before = tuple(map(sys.getrefcount, (left, right, scale)))
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    result = fused(left, right, scale, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(result, (left - right) * scale)
    assert tuple(map(sys.getrefcount, (left, right, scale))) == before
    stats = provider._crossjit_stats()
    assert stats["generated_exec_hit"] > 0
    assert stats["generic_fallback_hit"] == 0
    assert stats["internal_allocations"] == 0
    left[0, :4] = [-0., 0., np.inf, np.nan]
    right[0, :4] = [0., -0., np.inf, 1.]
    with np.errstate(all="warn"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fused(left, right, scale, 2)
        expected = (left - right) * scale
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    assert caught
    assert provider._crossjit_stats()["exec_miss"] > 0


def test_affine_production_policy_requires_measured_iteration_cell():
    program = r'''
import json, numpy as np
import _testinternalcapi as internal
from numpy._core import _multiarray_umath as umath
def run(outer, inner):
    namespace = {}
    exec("def fused(left, right, scale, count):\n"
         "    for _ in range(count):\n"
         "        result = scale * (left - right)\n"
         "    return result\n", namespace)
    backing = np.linspace(1., 2., 3 * outer * (2 * inner + 3)).reshape(
        3, outer, 2 * inner + 3)
    left, right = (item[:, :2 * inner:2] for item in backing[:2])
    scale = backing[2, :, :inner].copy()
    internal.crossjit_reset_stats()
    umath._crossjit_reset_stats()
    result = namespace['fused'](left, right, scale, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(result, scale * (left - right))
    return umath._crossjit_stats()
print(json.dumps({'measured': run(60, 60), 'unmeasured': run(90, 40)}))
'''
    env = os.environ.copy()
    for name in ("NPY_CROSSJIT_MIN_WORK", "NPY_CROSSJIT_PROFILED_FORCE_GENERATED",
                 "NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK"):
        env.pop(name, None)
    result = subprocess.run([sys.executable, "-c", program], env=env,
                            check=True, capture_output=True, text=True)
    stats = json.loads(result.stdout)
    assert stats['measured']['generated_exec_hit'] > 0
    assert stats['measured']['generic_fallback_hit'] == 0
    assert stats['unmeasured']['generated_exec_hit'] == 0
    assert stats['unmeasured']['regions_rejected_unprofitable'] > 0


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_affine_preserves_keeporder_result_layout(monkeypatch, reverse, mixed):
    internal = _runtime()
    provider = _provider_runtime()
    monkeypatch.setenv("NPY_CROSSJIT_PROFILED_FORCE_GENERATED", "1")
    namespace = {}
    exec("def fused(scale, left, right, count):\n"
         "    for _ in range(count):\n"
         "        result = scale * (left + right)\n"
         "    return result\n", namespace)
    left = np.array(np.linspace(1., 2., 3600).reshape(60, 60), order='F')
    right = left.copy(order='C' if mixed else 'F')
    if reverse:
        left, right = left[::-1, ::-1], right[::-1, ::-1]
    scale = np.float64(.25)
    expected = scale * (left + right)
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    result = namespace['fused'](
        scale, left, right, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(result, expected)
    assert result.strides == expected.strides
    assert result.flags.c_contiguous == expected.flags.c_contiguous
    assert result.flags.f_contiguous == expected.flags.f_contiguous
    # Conflicting layout preferences are conservatively unsupported.  The
    # positive affine tests above separately require actual generated HITs.
    stats = provider._crossjit_stats()
    assert stats['generated_exec_hit'] == 0
    assert stats['generic_fallback_hit'] == 0
    assert stats['regions_rejected_unsupported_contract'] > 0


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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("expression", ["a * b - c", "(a - b) * c",
                                         "c * (a - b)", "(a + b) * c",
                                         "(a + b) + c"])
def test_general_strided_pair_preserves_both_operations(dtype, reverse, expression):
    """The non-contiguous pair driver must evaluate its actual opcodes."""
    internal = _runtime()
    provider = _provider_runtime()
    namespace = {}
    exec("def fused(a, b, c, count):\n"
         "    for _ in range(count):\n"
         f"        result = {expression}\n"
         "    return result\n", namespace)
    fused = namespace["fused"]
    base = np.arange(5 * 7 * 3, dtype=dtype).reshape(5, 7, 3) / 8
    a, b, c = base[:, :, 0], base[:, :, 1], base[:, :, 2]
    if reverse:
        a, b, c = a[::-1, ::-1], b[::-1, ::-1], c[::-1, ::-1]
    expected = fused(a, b, c, 1)
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    actual = fused(a, b, c, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(actual, expected)
    stats = provider._crossjit_stats()
    assert stats["generic_fallback_hit"] > 0
    assert stats["exec_miss"] == 0
    assert stats["exec_error"] == 0


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
    assert validation_stats["provider"]["regions_rejected_unprofitable"] >= 1
    # A separately measured one-node prefix may execute. The rejected
    # two-node validation seed itself must remain absent from execution.
    assert all(record["generated_hit_count"] == 0
               for record in validation_stats["provider"]["observed_signatures"]
               if record["node_count"] == 2)

    unmeasured_stats = results["unmeasured"]
    assert unmeasured_stats["host"]["rewritten_regions"] == 0
    assert (
        unmeasured_stats["provider"]["regions_rejected_unprofitable"] >= 1
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
        assert provider["regions_rejected_unprofitable"] >= 1
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


def _new_long_arithmetic(node_count, middle_call=False):
    names = tuple(f"v{i}" for i in range(node_count + 1))
    expression = names[0]
    for i in range(node_count):
        if middle_call and i == node_count // 2:
            expression = f"np.sin({expression})"
        operation = ("+", "*", "-")[i % 3]
        expression = f"({expression} {operation} {names[i + 1]})"
    namespace = {"np": np}
    exec(f"def fused({', '.join(names)}, count):\n"
         "    for _ in range(count):\n"
         f"        result = {expression}\n"
         "    return result\n", namespace)
    return namespace["fused"]


def _long_reference(values, middle_call=False):
    result = values[0]
    for i, operand in enumerate(values[1:]):
        if middle_call and i == (len(values) - 1) // 2:
            result = np.sin(result)
        if i % 3 == 0:
            result = result + operand
        elif i % 3 == 1:
            result = result * operand
        else:
            result = result - operand
    return result


@pytest.mark.parametrize("node_count", [4, 5, 8, 12])
@pytest.mark.parametrize("dtype", [np.float64, np.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_long_dag_order_layout_fresh_output_and_shape_deopt(node_count, dtype, strided, monkeypatch):
    """The full Host/Provider path handles long DAGs, not just a C microkernel."""
    internal, provider = _runtime(), _provider_runtime()
    monkeypatch.setenv("NPY_CROSSJIT_DIAGNOSTICS", "1")
    fused = _new_long_arithmetic(node_count)
    storage = tuple(np.arange(96, dtype=dtype) + offset
                    for offset in range(1, node_count + 2))
    values = tuple(v.reshape(8, 12)[:, ::-2] if strided else v[:48]
                   for v in storage)
    expected = _long_reference(values)
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    actual = fused(*values, internal.TIER2_THRESHOLD * 2)
    host, stats = internal.crossjit_stats(), provider._crossjit_stats()
    np.testing.assert_array_equal(actual, expected)
    assert host["recognized_nodes"] == node_count
    assert host["rewritten_regions"] >= 1
    assert stats["last_candidate"]["region_length"] == node_count
    assert stats["internal_allocations"] == 0
    assert stats["exec_hit"] > 0
    assert stats["exec_error"] == 0
    retained = actual.copy()
    second = fused(*values, 2)
    assert actual is not second and not np.shares_memory(actual, second)
    np.testing.assert_array_equal(actual, retained)

    # Failure of a pre-execution shape guard reconstructs original execution.
    changed = tuple(v[..., :1] for v in values)
    misses = internal.crossjit_stats()["guard_misses"]
    actual = fused(*changed, 2)
    np.testing.assert_array_equal(actual, _long_reference(changed))
    assert internal.crossjit_stats()["guard_misses"] > misses


def test_long_dag_cannot_cross_unsupported_middle_call(monkeypatch):
    internal, provider = _runtime(), _provider_runtime()
    monkeypatch.setenv("NPY_CROSSJIT_DIAGNOSTICS", "1")
    fused = _new_long_arithmetic(8, middle_call=True)
    values = tuple(np.arange(48.0) / 100 + i for i in range(1, 10))
    expected = _long_reference(values, middle_call=True)
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    actual = fused(*values, internal.TIER2_THRESHOLD * 2)
    stats = provider._crossjit_stats()
    np.testing.assert_array_equal(actual, expected)
    assert stats["exec_error"] == 0
    assert stats["region_length_histogram"][8] == 0


def test_eight_node_ieee_values_and_signed_zero():
    internal, provider = _runtime(), _provider_runtime()
    fused = _new_long_arithmetic(8)
    first = np.resize(np.array([-0., 0., np.nan, np.inf, -np.inf, 1.]), 48)
    values = (first,) + tuple(np.full(48, -0. if i % 3 == 0 else 1.)
                              for i in range(1, 9))
    with np.errstate(all="ignore"):
        expected = _long_reference(values)
        internal.crossjit_reset_stats()
        provider._crossjit_reset_stats()
        actual = fused(*values, internal.TIER2_THRESHOLD * 2)
    stats = provider._crossjit_stats()
    assert stats["exec_hit"] > 0 and stats["exec_error"] == 0
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual[expected == 0]),
                                  np.signbit(expected[expected == 0]))


def test_eight_node_fpe_replays_before_output_publication():
    internal, provider = _runtime(), _provider_runtime()
    namespace = {}
    exec("def fused(a, b, z, count):\n"
         "    for _ in range(count):\n"
         "        result = (((((((a + b) * b) - b) + b) * b) - b) + b) / z\n"
         "    return result\n", namespace)
    fused = namespace["fused"]
    a, b, z = np.ones(48), np.ones(48), np.ones(48)
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    retained = fused(a, b, z, internal.TIER2_THRESHOLD * 2)
    saved = retained.copy()
    before = provider._crossjit_stats()
    z.fill(0.)
    with np.errstate(divide="warn"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = fused(a, b, z, 2)
    after = provider._crossjit_stats()
    assert caught and np.isinf(actual).all()
    assert after["exec_miss"] > before["exec_miss"]
    assert after["exec_error"] == before["exec_error"]
    with np.errstate(divide="raise"):
        with pytest.raises(FloatingPointError, match="divide by zero"):
            fused(a, b, z, 2)
    after = provider._crossjit_stats()
    # The generated/generic result is still private when its FPE is detected.
    assert after["exec_error"] == before["exec_error"]
    assert after["exec_miss"] > before["exec_miss"]
    np.testing.assert_array_equal(retained, saved)


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
    assert unmeasured["provider"]["regions_rejected_unprofitable"] >= 1
    assert unmeasured["provider"]["regions_executed"] == 0

    generic = stats["generic_profitable"]
    assert generic["host"]["rewritten_regions"] == 1
    assert generic["provider"]["generated_registry_hit"] == 0
    assert generic["provider"]["generic_fallback_hit"] >= 1

    regressive = stats["generic_regressive"]
    assert regressive["host"]["rewritten_regions"] == 0
    assert regressive["provider"]["regions_rejected_unprofitable"] >= 1
    assert regressive["provider"]["generic_fallback_hit"] == 0


@pytest.mark.parametrize('symbol', ['+', '-', '*', '/'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('reverse', [False, True])
def test_single_node_capability_order_and_shape_replay(symbol, dtype, reverse, monkeypatch):
    internal = _runtime()
    provider = _provider_runtime()
    monkeypatch.setenv('NPY_CROSSJIT_MIN_WORK', '0')
    monkeypatch.setenv('NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK', '1')
    namespace = {}
    exec(f'def fused(a, b, count):\n'
         f'    for _ in range(count):\n'
         f'        result = a {symbol} b\n'
         f'    return result\n', namespace)
    fused = namespace['fused']
    a = np.linspace(-3, 3, 256, dtype=dtype)[::2]
    b = np.linspace(1, 2, 256, dtype=dtype)[::2]
    if reverse:
        a, b = a[::-1], b[::-1]
    expected = fused(a, b, 1)
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    actual = fused(a, b, internal.TIER2_THRESHOLD * 2)
    assert actual.tobytes() == expected.tobytes()
    assert provider._crossjit_stats()['exec_hit'] > 0
    assert internal.crossjit_stats()['rewritten_regions'] > 0
    with pytest.raises(ValueError):
        fused(a, b[:-1], 2)
    assert fused(a, b, 2).tobytes() == expected.tobytes()


def test_production_long_dag_measured_cells_without_force():
    """Long generated artifacts admit measured cells, including loop shape."""
    program = r'''
import json
import numpy as np
import _testinternalcapi as internal
from numpy._core import _multiarray_umath as provider
results = []
for nodes, outer, inner, accepted in (
        (8, 4, 16, True), (8, 4, 64, True), (8, 4, 256, True),
        (12, 4, 16, True), (12, 4, 64, True), (12, 4, 256, True),
        (12, 4, 1024, True), (8, 4, 1024, False),
        (8, 64, 4, False), (12, 64, 4, False)):
    names = [f'v{i}' for i in range(nodes + 1)]
    expression = names[0]
    for i in range(nodes):
        expression = f"({expression} {('+', '*', '-')[i % 3]} {names[i + 1]})"
    namespace = {}
    exec(f"def fused({', '.join(names)}, count):\n"
         "    for _ in range(count):\n"
         f"        result = {expression}\n"
         "    return result\n", namespace)
    values = tuple((1 + np.arange(outer * (inner + 3), dtype=np.float64)
                    / (outer * inner) + i / 8).reshape(outer, inner + 3)[:, :inner]
                   for i in range(nodes + 1))
    expected = namespace['fused'](*values, 1)
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    actual = namespace['fused'](*values, internal.TIER2_THRESHOLD * 2)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    results.append(dict(nodes=nodes, outer=outer, inner=inner, accepted=accepted,
                        host=internal.crossjit_stats(), provider=provider._crossjit_stats()))
print(json.dumps(results))
'''
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(('NPY_CROSSJIT_', 'PYTHON_JIT_')):
            env.pop(name)
    env.update(PYTHON_JIT_STRESS='1', PYTHON_JIT='1', PYTHON_JIT_CROSSJIT='1', NPY_CROSSJIT_DIAGNOSTICS='1')
    completed = subprocess.run([sys.executable, '-c', program], env=env,
                               check=True, capture_output=True, text=True)
    for result in json.loads(completed.stdout):
        stats = result['provider']
        if result['accepted']:
            assert result['host']['rewritten_regions'] == 1, result
            assert stats['generated_exec_hit'] > 0, result
            assert stats['output_allocations'] == stats['generated_exec_hit']
            assert stats['internal_allocations'] == 0
        else:
            assert result['host']['rewritten_regions'] == 0, result
            assert stats['regions_rejected_unprofitable'] > 0, result
            assert stats['generated_exec_hit'] == 0, result
        assert stats['exec_miss'] == stats['exec_error'] == 0


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


@pytest.mark.parametrize("nodes", [4, 5, 8])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("scalar_kind", [float, int])
def test_provider_first_long_dag_and_replay(nodes, strided, scalar_kind, monkeypatch):
    """External recognition avoids scalar speculation, with Provider replay."""
    internal, provider = _runtime(), _provider_runtime()
    monkeypatch.setenv("NPY_CROSSJIT_DIAGNOSTICS", "1")
    names = [f"scale{i}" for i in range(nodes - 1)] + ["left", "right"]
    expression = "(left + right)"
    for name in names[:-2]:
        expression = f"({name} / {expression})"
    namespace = {}
    exec(f"def fused({', '.join(names)}, count):\n"
         "    for _ in range(count):\n"
         f"        result = {expression}\n"
         "    return result\n", namespace)
    fused = namespace["fused"]
    base = np.arange(96.0).reshape(8, 12) + 2
    arrays = (base[:, ::-2], (base + 1)[:, ::-2]) if strided else (base[0], base[1])
    values = tuple(scalar_kind(2 + i) for i in range(nodes - 1)) + arrays

    def exercise(args, count):
        return fused(*args, count)

    expected = exercise(values, 1)
    ownership = tuple(sys.getrefcount(value) for value in values)
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    actual = exercise(values, internal.TIER2_THRESHOLD * 2)
    host, stats = internal.crossjit_stats(), provider._crossjit_stats()
    assert actual.tobytes() == expected.tobytes()
    assert host["provider_first_recognized"] >= nodes
    if scalar_kind is float:
        assert host["scalar_speculations_avoided"] > 0
        assert host["type_guard_regions"] == 0
    assert stats["last_candidate"]["region_length"] == nodes
    assert stats["internal_allocations"] == 0
    assert stats["exec_error"] == stats["exec_miss"] == 0
    retained = actual.copy()
    second = exercise(values, 2)
    assert actual is not second and not np.shares_memory(actual, second)
    assert actual.tobytes() == retained.tobytes()
    assert tuple(sys.getrefcount(value) for value in values) == ownership

    # Strong scalars have the same numeric values but fail the Provider's
    # representation guard. Fallback starts at the original arithmetic op.
    changed = tuple(np.float64(value) for value in values[:-2]) + arrays
    before = provider._crossjit_stats()["guard_miss"]
    actual = exercise(changed, 20)
    assert actual.tobytes() == expected.tobytes()
    assert provider._crossjit_stats()["guard_miss"] > before

    class ObservedFloat(float):
        def __truediv__(self, other):
            raise RuntimeError("original scalar operation observed")

    # An unsupported observer must run through original Python execution,
    # including the already-proven intermediate operations before it.
    changed = (*values[:-3], ObservedFloat(values[-3]), *arrays)
    with pytest.raises(RuntimeError, match="original scalar operation observed"):
        exercise(changed, 2)


def test_provider_first_decline_preserves_scalar_and_observer_semantics(monkeypatch):
    """A declined region cannot leave unguarded scalar-narrowing facts."""
    internal, provider = _runtime(), _provider_runtime()
    monkeypatch.delenv("NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK", raising=False)
    monkeypatch.delenv("NPY_CROSSJIT_PROFILED_FORCE_GENERATED", raising=False)
    monkeypatch.setenv("NPY_CROSSJIT_MIN_WORK", "1000000000")
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()

    def fused(left, right, scale, count):
        for _ in range(count):
            result = scale / (left + right)
            following = scale + 1
        return result, following

    left, right = np.arange(24.0) + 2, np.arange(24.0) + 3
    for scale in (2.0, 2, np.float64(2)):
        result, following = fused(left, right, scale, internal.TIER2_THRESHOLD * 2)
        assert result.tobytes() == (scale / (left + right)).tobytes()
        assert following == scale + 1
    assert internal.crossjit_stats()["scalar_speculations_avoided"] > 0
    assert provider._crossjit_stats()["exec_hit"] == 0

    class Observer(float):
        def __truediv__(self, other):
            raise RuntimeError("declined original scalar observed")

    with pytest.raises(RuntimeError, match="declined original scalar observed"):
        fused(left, right, Observer(2), 2)


def test_profiled_provider_value_class_admission_and_guard():
    """Measured scalar classes admit a DAG; magnitude changes replay safely."""
    program = r"""
import json
import sys
from itertools import product
import numpy as np
import _testinternalcapi as internal
from numpy._core import _multiarray_umath as provider

def fresh():
    namespace = {}
    exec('def fused(scale, left, offset, numerator, count):\n'
         '    for _ in range(count):\n'
         '        result = numerator / (scale * left + offset)\n'
         '    return result\n', namespace)
    return namespace['fused']

def run(function, args, count):
    provider._crossjit_reset_stats()
    before = tuple(map(sys.getrefcount, args))
    actual = function(*args, count)
    expected = args[3] / (args[0] * args[1] + args[2])
    assert actual.tobytes() == expected.tobytes()
    assert tuple(map(sys.getrefcount, args)) == before
    return actual, provider._crossjit_stats()

admissions = []
for stride in (2, 100):
    left = (np.arange(98 * stride, dtype=np.float64) + 2)[::stride]
    for kinds in product((float, np.float64, int), repeat=3):
        args = (kinds[0](2), left, kinds[1](3), kinds[2](4))
        function = fresh()
        output, stats = run(function, args, internal.TIER2_THRESHOLD * 2)
        assert stats['generated_exec_hit'] > 0, stats
        assert stats['exec_miss'] == stats['exec_error'] == 0
        assert stats['internal_allocations'] == 0
        assert stats['output_allocations'] == stats['generated_exec_hit']
        saved = output.tobytes()
        other, _ = run(function, args, 2)
        assert other is not output and not np.shares_memory(other, output)
        assert output.tobytes() == saved
        admissions.append(stats['generated_exec_hit'])

function = fresh()
args = (2, left, 3, 4)
run(function, args, internal.TIER2_THRESHOLD * 2)
changed = (2**31, left, 3, 4)
_, guard = run(function, changed, 2)
assert guard['guard_miss'] > 0, guard
assert guard['exec_miss'] == guard['exec_error'] == 0
_, declined = run(fresh(), changed, internal.TIER2_THRESHOLD * 2)
# A separately measured two-node prefix may still be admitted. The new
# class-sensitive complete DAG must remain rejected for this magnitude.
complete = [s for s in declined['observed_signatures'] if s['node_count'] == 3]
assert complete and all(s['accepted_count'] == s['execution_count'] == 0 for s in complete)
print(json.dumps({'admitted_cells': len(admissions)}))
"""
    env = os.environ.copy()
    for name in ("NPY_CROSSJIT_MIN_WORK", "NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK",
                 "NPY_CROSSJIT_PROFILED_FORCE_GENERATED",
                 "NPY_CROSSJIT_DIAGNOSTIC_DISABLE_GENERATED"):
        env.pop(name, None)
    env["NPY_CROSSJIT_DIAGNOSTICS"] = "1"
    completed = subprocess.run([sys.executable, "-c", program], env=env,
                               capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["admitted_cells"] == 54


def test_late_literal_constant_uses_measured_complete_dag():
    """Host constant binding enables an already measured normalized DAG."""
    program = r"""
import json
import sys
import numpy as np
import _testinternalcapi as internal
from numpy._core import _multiarray_umath as provider

def fresh():
    namespace = {}
    exec('def fused(left, count):\n'
         '    for _ in range(count):\n'
         '        result = 4.0 / (2.0 * left + 3.0)\n'
         '    return result\n', namespace)
    return namespace['fused']

for stride in (2, 100):
    left = (np.arange(98 * stride, dtype=np.float64) + 2)[::stride]
    expected = 4.0 / (2.0 * left + 3.0)
    function = fresh()
    refs = sys.getrefcount(left)
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    result = function(left, internal.TIER2_THRESHOLD * 2)
    assert result.tobytes() == expected.tobytes()
    stats = provider._crossjit_stats()
    assert stats['generated_exec_hit'] > 0, stats
    assert stats['last_candidate']['region_length'] == 3, stats
    assert stats['output_allocations'] == stats['generated_exec_hit']
    assert stats['internal_allocations'] == stats['exec_miss'] == stats['exec_error'] == 0
    assert sys.getrefcount(left) == refs
    saved = result.tobytes()
    other = function(left, 2)
    assert other is not result and not np.shares_memory(other, result)
    assert result.tobytes() == saved
    # Provider shape validation fails before generated execution; the Host
    # restores ordinary execution including the original literal loads.
    changed = np.arange(99.0) + 2
    actual = function(changed, 2)
    assert actual.tobytes() == (4.0 / (2.0 * changed + 3.0)).tobytes()
    assert provider._crossjit_stats()['guard_miss'] > 0
print(json.dumps({'correct': True}))
"""
    env = os.environ.copy()
    for name in ("NPY_CROSSJIT_MIN_WORK", "NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK",
                 "NPY_CROSSJIT_PROFILED_FORCE_GENERATED",
                 "NPY_CROSSJIT_DIAGNOSTIC_DISABLE_GENERATED"):
        env.pop(name, None)
    env["NPY_CROSSJIT_DIAGNOSTICS"] = "1"
    completed = subprocess.run([sys.executable, "-c", program], env=env,
                               capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["correct"]


@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("step", [1, 2])
def test_affine_coalesced_trailing_domain(monkeypatch, rank, step):
    """Provider coalesces accesses while retaining full object/layout guards."""
    internal, provider = _runtime(), _provider_runtime()
    monkeypatch.setenv("NPY_CROSSJIT_PROFILED_FORCE_GENERATED", "1")
    monkeypatch.setenv("NPY_CROSSJIT_DIAGNOSTICS", "1")
    namespace = {}
    exec("def fused(left, right, scale, count):\n"
         "    for _ in range(count):\n"
         "        result = scale * (left - right)\n"
         "    return result\n", namespace)
    fused = namespace["fused"]
    shape = (8, 9, 60) if rank == 3 else (8, 9, 5, 60)
    backing = np.linspace(1., 2., np.prod(shape)).reshape(shape)
    selection = (slice(1, 7), slice(1, 8)) + (
        (slice(None),) if rank == 4 else ()) + (slice(None, None, step),)
    left = backing[selection]
    right = (backing * .5)[selection]
    scale = left.copy() * .25
    before = tuple(map(sys.getrefcount, (left, right, scale)))
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    result = fused(left, right, scale, internal.TIER2_THRESHOLD * 2)
    expected = scale * (left - right)
    assert result.shape == left.shape
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))
    assert tuple(map(sys.getrefcount, (left, right, scale))) == before
    stats = provider._crossjit_stats()
    assert stats["generated_exec_hit"] > 0
    assert stats["generic_fallback_hit"] == 0
    assert stats["internal_allocations"] == 0
    observed = stats["observed_signatures"][0]
    assert observed["outer_shape_class"] == 1
    assert observed["iteration_descriptor_version"] == (1 if step == 1 else 2)
    assert observed["outer_count"] == left.shape[0]
    assert observed["inner_extent"] == np.prod(left.shape[1:])
    assert observed["sample_operand_strides"][0] == (
        left.strides[-1], left.strides[0])
    # Full original strides, including the folded axes, remain exact guards.
    replacement = left.copy()
    changed = fused(replacement, right, scale, 2)
    np.testing.assert_array_equal(changed, scale * (replacement - right))
    assert provider._crossjit_stats()["guard_miss"] > 0
    left.flat[:4] = [-0., 0., np.inf, np.nan]
    right.flat[:4] = [0., -0., np.inf, 1.]
    with np.errstate(all="warn"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fused(left, right, scale, 2)
        expected = scale * (left - right)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    assert caught
    assert provider._crossjit_stats()["exec_miss"] > 0


@pytest.mark.parametrize("layout", ["gap", "broadcast", "singleton", "reverse"])
def test_affine_coalescing_rejects_unproven_domains(monkeypatch, layout):
    internal, provider = _runtime(), _provider_runtime()
    monkeypatch.setenv("NPY_CROSSJIT_PROFILED_FORCE_GENERATED", "1")
    monkeypatch.setenv("NPY_CROSSJIT_DIAGNOSTICS", "1")
    namespace = {}
    exec("def fused(left, right, scale, count):\n"
         "    for _ in range(count):\n"
         "        result = scale * (left - right)\n"
         "    return result\n", namespace)
    left = np.arange(4 * 7 * 60.).reshape(4, 7, 60)
    if layout == "gap":
        left = left[:, :, :59]
    elif layout == "singleton":
        left = left[:, :1, :]
    elif layout == "reverse":
        left = left[:, :, ::-1]
    right = left.copy()
    if layout == "broadcast":
        right = right[:, :1, :]
    scale = np.ones(left.shape)
    provider._crossjit_reset_stats()
    result = namespace["fused"](
        left, right, scale, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(result, scale * (left - right))
    stats = provider._crossjit_stats()
    assert stats["generated_exec_hit"] == 0
    assert stats["exec_error"] == 0
    assert stats["generic_fallback_hit"] > 0
    assert all(item["outer_shape_class"] == 2
               for item in stats["observed_signatures"])


def test_coalesced_domain_production_requires_measured_cell():
    program = r'''
import json, numpy as np
import _testinternalcapi as internal
from numpy._core import _multiarray_umath as umath
namespace = {}
exec("def fused(left, right, scale, count):\n"
     "    for _ in range(count):\n"
     "        result = scale * (left - right)\n"
     "    return result\n", namespace)
backing = np.linspace(1., 2., 8 * 9 * 60).reshape(8, 9, 60)
left = backing[1:7, 1:8, :]
right, scale = left.copy(), left.copy()
umath._crossjit_reset_stats()
result = namespace['fused'](left, right, scale, internal.TIER2_THRESHOLD * 2)
np.testing.assert_array_equal(result, scale * (left - right))
print(json.dumps(umath._crossjit_stats()))
'''
    env = os.environ.copy()
    for name in ("NPY_CROSSJIT_MIN_WORK", "NPY_CROSSJIT_PROFILED_FORCE_GENERATED",
                 "NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK"):
        env.pop(name, None)
    result = subprocess.run([sys.executable, "-c", program], env=env,
                            check=True, capture_output=True, text=True)
    stats = json.loads(result.stdout)
    assert stats["generated_exec_hit"] == 0
    assert stats["regions_rejected_unprofitable"] > 0
