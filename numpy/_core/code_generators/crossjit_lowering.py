"""NumPy-owned lowering from Provider manifests to generic ElementwiseIR."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from crossjit_generator.ir import (
    HostOperation,
    HostProvenRegion,
    ProviderSpecialization,
    SpecializationAttribute,
    build_elementwise_ir,
)
from crossjit_generator.signature import OptimizationSignature
from crossjit_generator.iteration import (
    IterationDescriptor,
    IterationRankClass,
    OperandAccessKind,
    OperandDescriptor,
    OutputDescriptor,
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


def lower_profile_record(record: Mapping[str, Any]):
    """Lower an offline refined profile variant to IR plus iteration C_P."""

    forbidden = _FORBIDDEN_POLICY_KEYS.intersection(record)
    if forbidden:
        raise ValueError(
            f"workload identity is not a lowering input: {sorted(forbidden)}")
    signature = OptimizationSignature.from_data(record["signature"])
    signature_hash = signature.stable_hash()
    if record.get("canonical_signature_hash") != signature_hash:
        raise ValueError("profile record changed canonical structural signature")
    variant_descriptor = {
        "canonical_signature_hash": signature_hash,
        "iteration_rank_class": record["iteration_rank_class"],
        "operand_access_classes": record["operand_access_classes"],
        "refined_layout_class": record["layout_class"],
    }
    if record.get("profile_variant_hash") != _digest(variant_descriptor):
        raise ValueError("profile execution-specialization hash mismatch")
    access_names = tuple(record["operand_access_classes"])
    if len(access_names) != signature.input_count:
        raise ValueError("profile access count does not match signature")
    ir = lower_signature(signature)
    descriptor = iteration_descriptor(
        signature.input_count, access_names, record["iteration_rank_class"])
    return ir, descriptor


def iteration_descriptor(
        input_count: int, access_names: tuple[str, ...], rank_code: int,
) -> IterationDescriptor:
    """Build the versioned Provider iteration projection from layout facts."""

    if len(access_names) != input_count:
        raise ValueError("profile access count does not match input count")
    access_kinds = tuple(OperandAccessKind(name) for name in access_names)
    if rank_code not in (1, 2):
        raise ValueError("unsupported profile iteration rank class")
    rank = (IterationRankClass.FLAT if rank_code == 1
            else IterationRankClass.OUTER_INNER)
    outer = 1 if rank is IterationRankClass.FLAT else "runtime_outer_count"
    outer_shape = "NONE" if rank is IterationRankClass.FLAT else "AFFINE_ROWS"
    operands = []
    for slot, access in enumerate(access_kinds):
        zero = access in (OperandAccessKind.SCALAR,
                          OperandAccessKind.ZERO_STRIDE)
        outer_stride = None
        if rank is IterationRankClass.OUTER_INNER:
            outer_stride = (0 if access is OperandAccessKind.SCALAR else
                            "runtime_outer_stride")
        operands.append(OperandDescriptor(
            slot, access, 0 if zero else "element_size",
            outer_stride,
            ("SCALAR" if access is OperandAccessKind.SCALAR else
             "BROADCAST_INNER" if access is OperandAccessKind.ZERO_STRIDE else
             "NONE"),
        ))
    return IterationDescriptor(
        1, rank, outer, outer_shape, "runtime_inner_extent",
        tuple(operands),
        OutputDescriptor(
            OperandAccessKind.CONTIGUOUS, "element_size",
            ("runtime_output_outer_stride"
             if rank is IterationRankClass.OUTER_INNER else None),
            "FRESH_PROVIDER_OUTPUT"),
    )


def lower_inner_validation_seed(manifest: Mapping[str, Any]):
    """Reuse a structural validation recipe with an inner-row descriptor."""

    variant = deepcopy(manifest)
    specialization = variant["specialization"]
    specialization["layout_class"] = "inner_contiguous"
    specialization["contract_digest"] = (
        "numpy-fp64-inner-contiguous-affine-rows-v1")
    ir = lower_manifest(variant)
    descriptor = iteration_descriptor(
        len(variant["inputs"]),
        ("CONTIGUOUS",) * len(variant["inputs"]), 2)
    return ir, descriptor


def lower_zero_stride_validation_seed(manifest: Mapping[str, Any]):
    """Apply an inner-broadcast execution descriptor to a structural seed."""

    ir = lower_manifest(deepcopy(manifest))
    access_names = ("ZERO_STRIDE",) + (
        ("CONTIGUOUS",) * (len(manifest["inputs"]) - 1))
    return ir, iteration_descriptor(len(manifest["inputs"]), access_names, 2)


__all__ = [
    "iteration_descriptor", "lower_inner_validation_seed", "lower_manifest",
    "lower_profile_record", "lower_signature",
    "lower_zero_stride_validation_seed",
]
