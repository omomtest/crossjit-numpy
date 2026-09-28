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
from crossjit_generator.signature import OptimizationSignature

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


def lower_signature(signature: OptimizationSignature):
    """Reconstruct the supported Provider IR from structural telemetry."""

    if (not isinstance(signature, OptimizationSignature) or
            signature.provider != "numpy" or
            signature.domain != "elementwise"):
        raise ValueError("unsupported OptimizationSignature")
    values = [f"input_{index}" for index in range(signature.input_count)]
    operations = []
    for index, operation in enumerate(signature.operations):
        recipe = RECIPES.get(operation.operation_id)
        if recipe is None or recipe.stable_hash() != operation.recipe_hash:
            raise ValueError(
                f"missing or stale NumPy recipe: {operation.operation_id}")
        result = f"value_{index}"
        operations.append(HostOperation(
            node_id=f"node_{index}",
            operation_id=operation.operation_id,
            operands=tuple(values[slot] for slot in operation.operands),
            result=result,
        ))
        values.append(result)
    final = operations[-1].result
    region = HostProvenRegion(
        provider="numpy",
        semantic_proof_digest=f"observed:{signature.stable_hash()}",
        inputs=tuple(values[:signature.input_count]),
        operations=tuple(operations),
        outputs=(final,),
        virtual_values=tuple(item.result for item in operations[:-1]),
        materialized_values=(final,),
    )
    specialization = ProviderSpecialization(
        contract_digest=f"signature:{signature.stable_hash()}",
        dtype_class=signature.dtype_class,
        layout_class=signature.layout_class,
        error_mode=signature.error_mode,
        attributes=(SpecializationAttribute(
            "scalar_mask", signature.scalar_mask),),
    )
    return build_elementwise_ir(region, RECIPES, specialization)


__all__ = ["lower_manifest", "lower_signature"]
