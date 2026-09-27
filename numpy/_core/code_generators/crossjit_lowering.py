"""NumPy-owned lowering from Provider manifests to generic ElementwiseIR."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from crossjit_generator.ir import (
    HostOperation,
    HostProvenRegion,
    ProviderSpecialization,
    SpecializationAttribute,
    build_elementwise_ir,
)

try:
    from .crossjit_recipes import RECIPES
except ImportError:  # Direct build-tool loading outside the NumPy package.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).with_name("crossjit_recipes.py")
    spec = importlib.util.spec_from_file_location("numpy_crossjit_recipes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    RECIPES = module.RECIPES


_OPERATION_IDS = {
    "add": "numpy.add",
    "sub": "numpy.subtract",
    "subtract": "numpy.subtract",
    "mul": "numpy.multiply",
    "multiply": "numpy.multiply",
    "true_divide": "numpy.true_divide",
}
_FORBIDDEN_POLICY_KEYS = {"benchmark", "benchmark_name", "workload_id", "source_file"}


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def lower_manifest(manifest: Mapping[str, Any]):
    """Consume Host proof and NumPy C_P projections; do not establish either."""

    forbidden = _FORBIDDEN_POLICY_KEYS.intersection(manifest)
    if forbidden:
        raise ValueError(
            f"benchmark identity is not a lowering input: {sorted(forbidden)}")
    if manifest.get("provider") != "numpy":
        raise ValueError("manifest Provider must be numpy")
    operations = tuple(
        HostOperation(
            node_id=operation["id"],
            operation_id=_OPERATION_IDS[operation["operation"]],
            operands=tuple(operation["operands"]),
            result=operation["result"],
        )
        for operation in manifest["operations"]
    )
    region = HostProvenRegion(
        provider="numpy",
        semantic_proof_digest=manifest["host_semantic_proof_digest"],
        inputs=tuple(manifest["inputs"]),
        operations=operations,
        outputs=tuple(manifest["outputs"]),
        virtual_values=tuple(manifest["virtual_values"]),
        materialized_values=tuple(manifest["materialized_values"]),
    )
    contract = manifest["specialization"]
    attributes = tuple(
        SpecializationAttribute(name, value)
        for name, value in contract.get("attributes", {}).items()
    )
    specialization = ProviderSpecialization(
        contract_digest=contract.get("contract_digest") or _digest(contract),
        dtype_class=contract["dtype_class"],
        layout_class=contract["layout_class"],
        error_mode=contract["error_mode"],
        attributes=attributes,
    )
    return build_elementwise_ir(region, RECIPES, specialization)


__all__ = ["lower_manifest"]
