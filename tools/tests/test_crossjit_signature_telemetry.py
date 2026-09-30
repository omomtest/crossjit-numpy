import importlib.util
import json
from pathlib import Path


def _module():
    path = (Path(__file__).parents[2] /
            "numpy/_core/code_generators/crossjit_signature_telemetry.py")
    spec = importlib.util.spec_from_file_location(
        "numpy_crossjit_signature_telemetry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(prepare=1, execute=10, scalar_mask=0):
    buckets = [0] * 14
    buckets[4] = execute
    return {
        "prepare_count": prepare,
        "accepted_count": prepare,
        "execution_count": execute,
        "generated_hit_count": execute - 2,
        "generic_fallback_count": 2,
        "min_elements": 16,
        "max_elements": 4096,
        "element_buckets": buckets,
        "input_count": 3,
        "node_count": 2,
        "dtype_class": 2,
        "layout_class": 1,
        "scalar_mask": scalar_mask,
        "nodes": [
            {"op": 0, "left": 0, "right": 1, "output": 3},
            {"op": 2, "left": 3, "right": 2, "output": 4},
        ],
    }


def test_provider_snapshot_becomes_stable_observation():
    observed = _module().collect_snapshot({
        "observed_signatures": [_record(), _record(2, 20)]})
    records = observed.observations()
    assert len(records) == 1
    assert records[0].prepare_count == 3
    assert records[0].accepted_count == 3
    assert records[0].execution_count == 30
    assert records[0].generated_hit_count == 26
    assert records[0].generic_fallback_count == 4
    assert sum(records[0].element_buckets) == 30
    data = json.loads(observed.serialize())
    signature = data["observations"][0]["signature"]
    assert [item["operation_id"] for item in signature["operations"]] == [
        "numpy.add", "numpy.multiply"]
    assert "benchmark" not in observed.serialize()


def test_scalar_relation_is_part_of_signature():
    observed = _module().collect_snapshot({
        "observed_signatures": [_record(), _record(scalar_mask=2)]})
    assert len(observed.observations()) == 2


def test_coverage_diagnostic_wrapper_is_accepted():
    observed = _module().collect_snapshot({
        "schema_version": 1,
        "workload_id": "offline-report-only",
        "host_counters": {"recognized_nodes": 2},
        "provider_counters": {"observed_signatures": [_record()]},
    })
    records = observed.observations()
    assert len(records) == 1
    assert records[0].execution_count == 10
    assert "workload" not in observed.serialize()
