"""Translate NumPy Provider telemetry into generic observed signatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crossjit_generator.signature import (
    ELEMENT_BUCKET_UPPER_BOUNDS,
    ObservedSignatureProfile,
    ObservedSignatures,
    OptimizationSignature,
    SignatureOperation,
)

try:
    from .crossjit_recipes import RECIPES
except ImportError:
    import importlib.util

    path = Path(__file__).with_name("crossjit_recipes.py")
    spec = importlib.util.spec_from_file_location(
        "numpy_crossjit_recipes_for_telemetry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    RECIPES = module.RECIPES


_OPERATIONS = {
    0: ("numpy.add", 2),
    1: ("numpy.subtract", 2),
    2: ("numpy.multiply", 2),
    3: ("numpy.true_divide", 2),
    4: ("numpy.negative", 1),
    5: ("numpy.minimum", 2),
    6: ("numpy.maximum", 2),
}
_DTYPES = {
    0: "other",
    1: "float32",
    2: "float64",
    3: "int32",
    4: "int64",
}
_LAYOUTS = {
    1: "contiguous_same_shape",
    2: "contiguous_broadcast",
    3: "zero_stride",
    4: "inner_contiguous",
    5: "general_strided",
}


def collect_snapshot(stats: dict) -> ObservedSignatures:
    result = ObservedSignatures()
    for record in stats.get("observed_signatures", ()):
        operations = []
        input_count = record["input_count"]
        for index, node in enumerate(record["nodes"]):
            operation_id, arity = _OPERATIONS[node["op"]]
            if node["output"] != input_count + index:
                raise ValueError("non-canonical NumPy telemetry output slot")
            slots = (node["left"],) if arity == 1 else (
                node["left"], node["right"])
            operations.append(SignatureOperation(
                operation_id, slots, RECIPES[operation_id].stable_hash()))
        signature = OptimizationSignature(
            provider="numpy",
            provider_abi_version=3,
            domain="elementwise",
            input_count=input_count,
            operations=tuple(operations),
            dtype_class=_DTYPES[record["dtype_class"]],
            layout_class=_LAYOUTS[record["layout_class"]],
            error_mode="numpy_fpe_fallback",
            materialization_class="virtual_intermediates_single_output",
            scalar_mask=record["scalar_mask"],
        )
        buckets = tuple(record["element_buckets"])
        if len(buckets) != len(ELEMENT_BUCKET_UPPER_BOUNDS):
            raise ValueError("unexpected NumPy element-count buckets")
        result.merge(ObservedSignatureProfile(
            signature=signature,
            prepare_count=record["prepare_count"],
            accepted_count=record["accepted_count"],
            execution_count=record["execution_count"],
            generated_hit_count=record["generated_hit_count"],
            generic_fallback_count=record["generic_fallback_count"],
            min_elements=record["min_elements"],
            max_elements=record["max_elements"],
            element_buckets=buckets,
        ))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    combined = ObservedSignatures()
    for path in args.snapshots:
        snapshot = collect_snapshot(json.loads(path.read_text()))
        for observation in snapshot.observations():
            combined.merge(observation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(combined.serialize())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
