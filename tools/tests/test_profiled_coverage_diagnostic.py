import importlib.util
from pathlib import Path

import pytest


def _diagnostic_module():
    root = Path(__file__).parents[2]
    path = (root / "benchmarks/benchmarks/crossjit/"
            "run_profiled_coverage_diagnostic.py")
    spec = importlib.util.spec_from_file_location(
        "crossjit_profiled_coverage_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _signature(operation, recipe_hash):
    return {
        "operations": [{
            "operation_id": operation,
            "recipe_hash": recipe_hash,
        }],
    }


def test_recipe_catalog_includes_pruned_kernel_policy_recipes():
    module = _diagnostic_module()
    metadata = {
        "kernels": [{
            "optimization_signature": _signature("numpy.add", "a" * 64),
        }],
        "production_policies": [{
            "signature": _signature("numpy.true_divide", "d" * 64),
        }],
    }

    assert module._recipe_hash_catalog(metadata) == {
        "numpy.add": "a" * 64,
        "numpy.true_divide": "d" * 64,
    }


def test_recipe_catalog_rejects_conflicting_frozen_hashes():
    module = _diagnostic_module()
    metadata = {
        "kernels": [{
            "optimization_signature": _signature("numpy.add", "a" * 64),
        }],
        "production_policies": [{
            "signature": _signature("numpy.add", "b" * 64),
        }],
    }

    with pytest.raises(ValueError, match="conflicting recipe hashes"):
        module._recipe_hash_catalog(metadata)
