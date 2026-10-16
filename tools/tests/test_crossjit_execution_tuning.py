"""Scalar representation fixtures for conservative complete-path calibration."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _module():
    path = (Path(__file__).parents[2] / "benchmarks/benchmarks/crossjit/"
            "run_profiled_execution_tuning.py")
    spec = importlib.util.spec_from_file_location("execution_tuning", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.np = np
    return module


def _case():
    return {"entry_abi": "affine_v2", "iteration_descriptor": {
        "iteration_rank_class": "FLAT", "operands": [
            {"access_kind": kind} for kind in
            ("SCALAR", "FIXED_STRIDE", "SCALAR", "SCALAR")]}}


def test_mixed_scalar_fixture_preserves_representations_and_affine_access():
    values = _module()._inputs(_case(), 1, 98, 100,
                              scalar_pattern="python,numpy,integer")
    assert type(values[0]) is float
    assert type(values[2]) is np.float64
    assert type(values[3]) is int
    assert values[1].shape == (98,)
    assert values[1].strides == (800,)


@pytest.mark.parametrize("pattern", ["python,numpy", "python,numpy,unknown"])
def test_incomplete_or_unsupported_scalar_pattern_rejected(pattern):
    with pytest.raises(ValueError, match="every scalar representation"):
        _module()._inputs(_case(), 1, 98, scalar_pattern=pattern)


def test_scalar_pattern_propagates_to_all_worker_modes():
    args = SimpleNamespace(python="python", generated_metadata="metadata.json",
        layouts="SCALAR_OPERAND", shapes="1x98", specializations="hash",
        samples=7, target_elements=300000, min_iterations=100,
        max_iterations=10000, affine_stride=100, scalar_kind="legacy",
        scalar_pattern="python,numpy,integer")
    for mode in ("baseline", "generated", "generic"):
        command = _module()._command(args, mode)
        assert command[command.index("--scalar-pattern") + 1] == args.scalar_pattern
