import importlib.util
from pathlib import Path

import pytest


def _lowering_module():
    path = (Path(__file__).parents[2] / "numpy/_core/code_generators" /
            "crossjit_lowering.py")
    spec = importlib.util.spec_from_file_location("numpy_crossjit_lowering", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _linear_manifest(length):
    operation_names = ("add", "mul", "sub", "true_divide", "add", "mul")
    inputs = [chr(ord("a") + index) for index in range(length + 1)]
    operations = []
    prior = None
    for index, operation in enumerate(operation_names[:length]):
        result = "out" if index == length - 1 else f"t{index}"
        operations.append({
            "id": f"n{index}",
            "operation": operation,
            "operands": [inputs[0] if index == 0 else prior, inputs[index + 1]],
            "result": result,
        })
        prior = result
    return {
        "provider": "numpy",
        "host_semantic_proof_digest": f"proof-linear-{length}",
        "inputs": inputs,
        "operations": operations,
        "outputs": ["out"],
        "virtual_values": [operation["result"] for operation in operations[:-1]],
        "materialized_values": ["out"],
        "specialization": {
            "dtype_class": "float64",
            "layout_class": "contiguous_same_shape",
            "error_mode": "numpy_fpe_fallback",
            "attributes": {"rank_class": "one_dimensional"},
        },
    }


@pytest.mark.parametrize("length", range(2, 7))
def test_numpy_l2_through_l6_lower_deterministically(length):
    lower = _lowering_module().lower_manifest
    first = lower(_linear_manifest(length))
    second = lower(_linear_manifest(length))
    assert first.serialize() == second.serialize()
    assert first.stable_hash() == second.stable_hash()
    assert len(first.operations) == length
    assert first.dtype_class == "float64"
    assert first.layout_class == "contiguous_same_shape"


def test_numpy_branched_dag_lowering():
    manifest = _linear_manifest(2)
    manifest.update({
        "host_semantic_proof_digest": "proof-branched",
        "inputs": ["a", "b", "c", "d"],
        "operations": [
            {"id": "n0", "operation": "mul", "operands": ["a", "b"],
             "result": "t0"},
            {"id": "n1", "operation": "mul", "operands": ["c", "d"],
             "result": "t1"},
            {"id": "n2", "operation": "add", "operands": ["t0", "t1"],
             "result": "out"},
        ],
        "virtual_values": ["t0", "t1"],
    })
    ir = _lowering_module().lower_manifest(manifest)
    assert ir.operations[2].operands == ("t0", "t1")


def test_benchmark_identity_is_rejected():
    manifest = _linear_manifest(2)
    manifest["benchmark_name"] = "not-an-input"
    with pytest.raises(ValueError, match="benchmark identity"):
        _lowering_module().lower_manifest(manifest)


def test_affine_rows_require_explicit_two_axis_fact():
    lower = _lowering_module().iteration_descriptor
    descriptor = lower(3, ("SCALAR", "GENERAL_STRIDED", "CONTIGUOUS"), 2, 1)
    assert descriptor.schema_version == 2
    assert descriptor.operands[1].inner_stride == "runtime_inner_stride"
    with pytest.raises(ValueError, match="two-axis"):
        lower(3, ("SCALAR", "FIXED_STRIDE", "CONTIGUOUS"), 2, 2)
    with pytest.raises(ValueError, match="two-axis"):
        lower(3, ("SCALAR", "GENERAL_STRIDED", "CONTIGUOUS"), 2, 2)
