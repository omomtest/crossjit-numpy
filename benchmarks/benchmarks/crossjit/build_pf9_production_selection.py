#!/usr/bin/env python3
"""Build the deterministic PF9 production-artifact selection manifest.

This tool consumes only the frozen candidate metadata, structural/refined
frequency counts, generic-signature DecisionDB evidence, compiled artifact
costs, and developer policy.  It deliberately never reads workload identity
or end-to-end timing fields from the coverage reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text()
    return json.loads(text), _sha256_bytes(text.encode("utf-8"))


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _function_source_bytes(source: str) -> dict[str, int]:
    starts = list(re.finditer(r"(?m)^void (crossjit_elem_[A-Za-z0-9_]+)\(", source))
    result: dict[str, int] = {}
    for index, match in enumerate(starts):
        if index + 1 < len(starts):
            end = starts[index + 1].start()
        else:
            end = source.find("\nconst NumpyCrossJitGeneratedKernel", match.start())
            if end < 0:
                end = len(source)
        result[match.group(1)] = len(source[match.start():end].encode("utf-8"))
    return result


def _symbol_text_bytes(object_path: Path) -> dict[str, int]:
    output = subprocess.check_output(
        ["nm", "-S", "--size-sort", str(object_path)], text=True)
    result: dict[str, int] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[2].upper() == "T":
            result[fields[3]] = int(fields[1], 16)
    return result


def _measurement_summary(record: dict[str, Any] | None, *, layer: str) -> dict[str, Any]:
    if record is None:
        return {
            "artifact_decision": "UNMEASURED",
            "confidence": 0.0,
            "generated_profitable_point_count": 0,
            "layer": layer,
            "max_cv": None,
            "profitability_envelope": {"points": []},
            "provenance": None,
            "reason": "no exact frozen DecisionDB record for this candidate identity",
        }
    points = record["profitability"]["points"]
    return {
        "artifact_decision": record["artifact_decision"],
        "confidence": record["confidence"],
        "generated_profitable_point_count": sum(
            point["decision"] == "GENERATED" for point in points),
        "layer": layer,
        "max_cv": max((point["max_cv"] for point in points), default=None),
        "profitability_envelope": record["profitability"],
        "provenance": record["provenance"],
        "reason": record["reason"],
    }


def _candidate_cost(
    kernel: dict[str, Any] | None,
    source_bytes: dict[str, int],
    text_bytes: dict[str, int],
) -> dict[str, int]:
    if kernel is None:
        return {
            "generated_source_bytes": 0,
            "metadata_bytes": 0,
            "object_text_bytes": 0,
        }
    return {
        "generated_source_bytes": source_bytes[kernel["symbol"]],
        "metadata_bytes": len(_canonical_bytes(kernel)),
        "object_text_bytes": text_bytes[kernel["symbol"]],
    }


def _runtime_decision(record: dict[str, Any] | None, element_count: int) -> str:
    if record is None:
        return "ORIGINAL"
    for point in record["profitability"]["points"]:
        if point["min_elements"] <= element_count <= point["max_elements"]:
            return point["decision"]
    return "ORIGINAL"


def _record_decision(
    *,
    artifact_present: bool,
    generation_source: str,
    measurement: dict[str, Any],
    frequency: int,
    expected_generated_executions: int,
    shadowed_by_measured_structural_artifact: bool,
) -> tuple[str, str, str]:
    state = measurement["artifact_decision"]
    if generation_source == "validation_seed" and frequency == 0:
        return (
            "REJECT",
            "DISABLE",
            "validation-only specialization has zero frozen production frequency; retain in validation generation only",
        )
    if state == "DISABLED":
        return "REJECT", "DISABLE", "developer DISABLE policy"
    if state == "FORCED":
        if not artifact_present:
            raise ValueError("FORCED production candidate has no frozen artifact")
        return "SELECT", "FORCE", "developer FORCE policy with a valid frozen measurement record"
    if state == "AUTO_REJECTED":
        return "REJECT", "AUTO", "preserve frozen AUTO_REJECTED profitability outcome"
    if state in ("STALE", "UNMEASURED"):
        reason = "stale measurement" if state == "STALE" else "no exact specialization measurement"
        if shadowed_by_measured_structural_artifact:
            reason += "; frozen registry precedence routes these executions to the measured structural artifact"
        return "REJECT", "AUTO", reason
    if state != "AUTO_SELECTED":
        raise ValueError(f"unsupported artifact decision: {state}")
    if expected_generated_executions == 0:
        suffix = (
            "; frozen registry precedence routes these executions to the "
            "measured structural artifact"
            if shadowed_by_measured_structural_artifact else ""
        )
        return (
            "REJECT",
            "AUTO",
            "valid measured artifact has zero expected generated executions "
            "under the frozen profile and runtime DecisionDB" + suffix,
        )
    return (
        "SELECT",
        "AUTO",
        "exact measured AUTO_SELECTED artifact covers frozen-profile executions authorized by the unchanged runtime DecisionDB",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--numpy-root", type=Path, required=True)
    parser.add_argument("--pf6-metadata", type=Path, required=True)
    parser.add_argument("--pf6-source", type=Path, required=True)
    parser.add_argument("--pf6-object", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--opportunity-report", type=Path, required=True)
    parser.add_argument("--production-report", type=Path, required=True)
    parser.add_argument("--decision-db", type=Path, required=True)
    parser.add_argument("--execution-decision-db", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.generator_root.resolve()))
    from crossjit_generator.domains.elementwise import ElementwiseOptimizer

    sys.path.insert(0, str((args.numpy_root / "numpy/_core/code_generators").resolve()))
    import crossjit_lowering as lowering

    metadata, metadata_hash = _read_json(args.pf6_metadata)
    observations, observations_hash = _read_json(args.observations)
    opportunity, opportunity_hash = _read_json(args.opportunity_report)
    production, production_hash = _read_json(args.production_report)
    decision_db, decision_db_hash = _read_json(args.decision_db)
    execution_inputs = [_read_json(path) for path in args.execution_decision_db]

    source_text = args.pf6_source.read_text()
    source_sizes = _function_source_bytes(source_text)
    text_sizes = _symbol_text_bytes(args.pf6_object)

    structural_records = {
        record["signature_hash"]: record for record in decision_db["records"]
    }
    execution_records = {
        record["specialization_hash"]: record
        for data, _ in execution_inputs for record in data["records"]
    }
    if len(execution_records) != sum(
            len(data["records"]) for data, _ in execution_inputs):
        raise ValueError("duplicate execution specialization across DecisionDBs")

    opportunity_by_variant = {
        record["profile_variant_hash"]: record
        for record in opportunity["signatures"]
    }
    structural_frequency: dict[str, int] = {}
    structural_generated: dict[str, int] = {}
    refined_generated: dict[str, int] = {}
    kernels = metadata["kernels"]
    if len(kernels) != 28:
        raise ValueError("PF9 expects the exact frozen 28-artifact PF6 universe")
    kernel_by_candidate = {kernel["candidate_hash"]: kernel for kernel in kernels}
    if len(kernel_by_candidate) != len(kernels):
        raise ValueError("duplicate candidate hash in PF6 artifact universe")
    nonprofiled_by_signature = {
        kernel["optimization_signature_hash"]: kernel
        for kernel in kernels if "profiled_layout_class" not in kernel
    }
    profiled_outer_by_variant = {
        kernel["profile_variant_hash"]: kernel
        for kernel in kernels
        if kernel.get("entry_abi") == "outer_inner_v1" and
        kernel.get("profile_variant_hash") is not None
    }

    # PF6 profiles do not encode enough physical-stride detail to prove that
    # a refined OUTER_INNER artifact is dominated by a structural FLAT
    # artifact.  Conservatively record potential coverage for every measured
    # identity whose unchanged DecisionDB authorizes the observed count.  The
    # final expected-coverage total below is a union over profile variants, so
    # overlapping potential coverage is never double-counted.
    for profile in opportunity["signatures"]:
        signature_hash = profile["canonical_signature_hash"]
        if (profile["legacy_layout_class"] == "contiguous_same_shape" and
                signature_hash in nonprofiled_by_signature):
            structural_frequency[signature_hash] = (
                structural_frequency.get(signature_hash, 0) +
                profile["dynamic_execution_opportunity_count"])
    for profile in production["signatures"]:
        generated_count = profile["generated_execution_count"]
        if generated_count == 0:
            continue
        outer = profile["outer_count_samples"]
        inner = profile["inner_extent_samples"]
        if len(outer) != 1 or len(inner) != 1:
            raise ValueError(
                "PF9 requires an exact frozen element count per generated variant")
        element_count = outer[0] * inner[0]
        signature_hash = profile["canonical_signature_hash"]
        structural_kernel = nonprofiled_by_signature.get(signature_hash)
        structural_record = structural_records.get(signature_hash)
        if (profile["legacy_layout_class"] == "contiguous_same_shape" and
                structural_kernel is not None and
                _runtime_decision(structural_record, element_count) ==
                "GENERATED"):
            structural_generated[signature_hash] = (
                structural_generated.get(signature_hash, 0) + generated_count)
        kernel = profiled_outer_by_variant.get(profile["profile_variant_hash"])
        if kernel is not None:
            specialization_hash = kernel["execution_specialization_hash"]
            if _runtime_decision(
                    execution_records.get(specialization_hash),
                    element_count) == "GENERATED":
                refined_generated[specialization_hash] = (
                    refined_generated.get(specialization_hash, 0) +
                    generated_count)

    records: list[dict[str, Any]] = []
    optimizer = ElementwiseOptimizer()
    for signature_hash, decision_record in sorted(structural_records.items()):
        kernel = nonprofiled_by_signature.get(signature_hash)
        if kernel is None:
            try:
                ir = lowering.lower_signature(decision_record["signature"])
                candidate = optimizer.legal_candidates(ir)[0]
            except (KeyError, ValueError):
                candidate = None
            if candidate is None:
                candidate_hash = None
                specialization_data = None
                specialization_hash = None
            else:
                specialization = candidate.execution_specialization(
                    signature_hash)
                candidate_hash = candidate.stable_hash()
                specialization_data = specialization.to_data()
                specialization_hash = specialization.stable_hash()
            generation_source = "observed"
            generation_mode = "AUTO"
        else:
            candidate_hash = kernel["candidate_hash"]
            specialization_data = kernel["execution_specialization"]
            specialization_hash = kernel["execution_specialization_hash"]
            generation_source = kernel["generation_source"]
            generation_mode = kernel["generation_mode"]
        measurement = _measurement_summary(decision_record, layer="DecisionDB")
        frequency = structural_frequency.get(signature_hash, 0)
        expected = structural_generated.get(signature_hash, 0)
        decision, developer_policy, reason = _record_decision(
            artifact_present=kernel is not None,
            generation_source=generation_source,
            measurement=measurement,
            frequency=frequency,
            expected_generated_executions=expected,
            shadowed_by_measured_structural_artifact=False,
        )
        records.append({
            "artifact_cost": _candidate_cost(kernel, source_sizes, text_sizes),
            "artifact_present_in_pf6": kernel is not None,
            "candidate_hash": candidate_hash,
            "developer_policy": developer_policy,
            "execution_specialization": specialization_data,
            "execution_specialization_hash": specialization_hash,
            "expected_generated_execution_count": expected,
            "generation_mode": generation_mode,
            "generation_source": generation_source,
            "identity_layer": "structural",
            "measurement": measurement,
            "production_artifact_decision": decision,
            "profile_dynamic_opportunity_count": frequency,
            "reason": reason,
            "structural_signature": decision_record["signature"],
            "structural_signature_hash": signature_hash,
        })

    for kernel in sorted(
            (item for item in kernels if "profiled_layout_class" in item),
            key=lambda item: item["execution_specialization_hash"]):
        specialization_hash = kernel["execution_specialization_hash"]
        decision_record = execution_records.get(specialization_hash)
        measurement = _measurement_summary(
            decision_record, layer="ExecutionDecisionDB")
        variant_hash = kernel.get("profile_variant_hash")
        if variant_hash is None:
            frequency = 0
            expected = 0
        else:
            frequency = opportunity_by_variant[variant_hash][
                "dynamic_execution_opportunity_count"]
            expected = refined_generated.get(specialization_hash, 0)
        selection_source = kernel.get(
            "profiled_source", kernel["generation_source"])
        decision, developer_policy, reason = _record_decision(
            artifact_present=True,
            generation_source=selection_source,
            measurement=measurement,
            frequency=frequency,
            expected_generated_executions=expected,
            shadowed_by_measured_structural_artifact=False,
        )
        records.append({
            "artifact_cost": _candidate_cost(kernel, source_sizes, text_sizes),
            "artifact_present_in_pf6": True,
            "candidate_hash": kernel["candidate_hash"],
            "developer_policy": developer_policy,
            "execution_specialization": kernel["execution_specialization"],
            "execution_specialization_hash": specialization_hash,
            "expected_generated_execution_count": expected,
            "generation_mode": kernel["generation_mode"],
            "generation_source": selection_source,
            "identity_layer": "refined_execution",
            "measurement": measurement,
            "production_artifact_decision": decision,
            "profile_dynamic_opportunity_count": frequency,
            "profile_variant_hash": variant_hash,
            "reason": reason,
            "structural_signature": kernel["optimization_signature"],
            "structural_signature_hash": kernel[
                "optimization_signature_hash"],
        })

    records.sort(key=lambda item: (
        item["structural_signature_hash"],
        item["execution_specialization_hash"] or "",
        item["candidate_hash"] or "",
    ))
    artifact_records = [item for item in records
                        if item["artifact_present_in_pf6"]]
    selected = [item for item in artifact_records
                if item["production_artifact_decision"] == "SELECT"]
    if len(records) != 30 or len(artifact_records) != 28 or len(selected) != 9:
        raise ValueError(
            "unexpected frozen selection counts: "
            f"{len(records)}/{len(artifact_records)}/{len(selected)}")
    selected_candidate_hashes = {item["candidate_hash"] for item in selected}
    expected_coverage = 0
    for profile in production["signatures"]:
        generated_count = profile["generated_execution_count"]
        if generated_count == 0:
            continue
        outer = profile["outer_count_samples"]
        inner = profile["inner_extent_samples"]
        if len(outer) != 1 or len(inner) != 1:
            raise ValueError(
                "PF9 requires an exact frozen element count per generated variant")
        element_count = outer[0] * inner[0]
        signature_hash = profile["canonical_signature_hash"]
        structural_kernel = nonprofiled_by_signature.get(signature_hash)
        structural_covered = (
            structural_kernel is not None and
            structural_kernel["candidate_hash"] in selected_candidate_hashes and
            _runtime_decision(
                structural_records.get(signature_hash), element_count) ==
            "GENERATED"
        )
        refined_kernel = profiled_outer_by_variant.get(
            profile["profile_variant_hash"])
        refined_covered = (
            refined_kernel is not None and
            refined_kernel["candidate_hash"] in selected_candidate_hashes and
            _runtime_decision(
                execution_records.get(
                    refined_kernel["execution_specialization_hash"]),
                element_count) == "GENERATED"
        )
        if not (structural_covered or refined_covered):
            raise ValueError(
                "selected artifacts do not cover a frozen generated variant")
        expected_coverage += generated_count
    total_opportunities = opportunity["coverage"][
        "total_dynamic_execution_opportunities"]
    if expected_coverage != 2_698_694:
        raise ValueError("selected artifacts changed frozen generated coverage")

    universe_identity = [{
        "candidate_hash": item["candidate_hash"],
        "execution_specialization_hash": item[
            "execution_specialization_hash"],
        "identity_layer": item["identity_layer"],
        "structural_signature_hash": item["structural_signature_hash"],
    } for item in records]
    candidate_universe_hash = _sha256_bytes(_canonical_bytes(universe_identity))
    execution_hashes = [digest for _, digest in execution_inputs]
    result = {
        "candidate_universe_hash": candidate_universe_hash,
        "compiler_artifact_input": {
            "pf6_generated_metadata_sha256": metadata_hash,
            "pf6_generated_object_sha256": _sha256_bytes(
                args.pf6_object.read_bytes()),
            "pf6_generated_source_sha256": _sha256_bytes(
                source_text.encode("utf-8")),
        },
        "decision_inputs": {
            "decision_db_sha256": decision_db_hash,
            "execution_decision_db_sha256": execution_hashes,
            "observations_sha256": observations_hash,
            "opportunity_report_sha256": opportunity_hash,
            "production_report_sha256": production_hash,
        },
        "policy_name": "GEN-PROFILED-PF9",
        "records": records,
        "runtime_profitability_retuned": False,
        "schema_version": 1,
        "selection_rule": {
            "kind": "deterministic_evidence_and_pareto",
            "rules": [
                "DISABLE rejects and FORCE selects only with an exact valid frozen measurement record",
                "AUTO_REJECTED, STALE, and UNMEASURED identities reject",
                "AUTO_SELECTED requires nonzero frozen-profile opportunity and at least one observed count authorized as GENERATED by its unchanged exact DecisionDB",
                "when PF6 profiles cannot prove physical-stride dominance, measured structural and refined identities are both retained rather than assigned arbitrary overlap weights",
                "zero-frequency validation seeds remain validation-only",
            ],
            "weights": None,
        },
        "summary": {
            "expected_generated_dynamic_coverage": (
                expected_coverage / total_opportunities),
            "expected_generated_execution_count": expected_coverage,
            "frozen_candidate_count": len(records),
            "pf6_generated_artifact_count": len(artifact_records),
            "production_rejected_artifact_count": (
                len(artifact_records) - len(selected)),
            "production_selected_artifact_count": len(selected),
            "selected_generated_source_function_bytes": sum(
                item["artifact_cost"]["generated_source_bytes"]
                for item in selected),
            "selected_object_text_function_bytes": sum(
                item["artifact_cost"]["object_text_bytes"]
                for item in selected),
            "total_dynamic_execution_opportunities": total_opportunities,
        },
    }

    serialized = _canonical_bytes(result)
    lowered = serialized.lower()
    for forbidden in (b'"benchmark"', b'"workload"', b'"source_path"'):
        if forbidden in lowered:
            raise ValueError("selection manifest contains forbidden identity input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(serialized)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
