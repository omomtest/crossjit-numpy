#!/usr/bin/env python3
"""Tune structural CrossJIT signatures with generic microbenchmarks."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys


GRID = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536)
CANDIDATE_CASES = (
    "linear_2", "linear_3", "linear_4", "linear_5", "linear_6",
    "branched_3",
)
FALLBACK_CONTROLS = ("array_scalar_2", "scalar_array_2")
CASE_TO_SIGNATURE_ID = {
    "linear_2": "linear_2_add_multiply",
    "linear_3": "linear_3_add_multiply_subtract",
    "linear_4": "linear_4_add_multiply_subtract_divide",
    "linear_5": "linear_5_add_multiply_subtract_divide_add",
    "linear_6": "linear_6_add_multiply_subtract_divide_add_multiply",
    "branched_3": "branched_3_multiply_multiply_add",
}


def _cv(values):
    return (statistics.stdev(values) / statistics.mean(values)
            if len(values) > 1 and statistics.mean(values) else 0.0)


def _worker_command(args, implementation):
    mode = "baseline" if implementation == "baseline" else "crossjit"
    return [
        args.python, str(args.harness), "--worker", "--mode", mode,
        "--n-grid", ",".join(map(str, GRID)),
        "--cases", ",".join((*CANDIDATE_CASES, *FALLBACK_CONTROLS)),
        "--dtype", "float64", "--layout", "CONTIGUOUS_SAME_SHAPE",
        "--samples", str(args.samples),
        "--target-elements", str(args.target_elements),
        "--min-iterations", str(args.min_iterations),
        "--max-iterations", str(args.max_iterations),
        "--cost-cells", "",
    ]


def _run_process(args, implementation):
    env = os.environ.copy()
    env.update({"PYTHON_JIT": "1", "OPENBLAS_NUM_THREADS": "1"})
    env.pop("NPY_CROSSJIT_DIAGNOSTICS", None)
    env.pop("NPY_CROSSJIT_TELEMETRY_TIMING", None)
    env.pop("NPY_CROSSJIT_DIAGNOSTIC_DISABLE_GENERATED", None)
    if implementation == "baseline":
        env["PYTHON_JIT_CROSSJIT"] = "0"
        env.pop("NPY_CROSSJIT_MIN_WORK", None)
    else:
        env["PYTHON_JIT_CROSSJIT"] = "1"
        env["NPY_CROSSJIT_MIN_WORK"] = "0"
        if implementation == "generic":
            env["NPY_CROSSJIT_DIAGNOSTIC_DISABLE_GENERATED"] = "1"
    completed = subprocess.run(
        _worker_command(args, implementation), check=True,
        capture_output=True, text=True, env=env)
    records = json.loads(completed.stdout)
    for record in records:
        record["implementation"] = implementation
    return records


def _measure(args):
    schedule = [(process, implementation)
                for process in range(args.processes)
                for implementation in ("baseline", "generated", "generic")]
    random.Random(args.seed).shuffle(schedule)
    raw = []
    for process, implementation in schedule:
        for record in _run_process(args, implementation):
            record["independent_process_index"] = process
            raw.append(record)

    groups = defaultdict(lambda: defaultdict(list))
    for record in raw:
        groups[(record["case_identifier"], record["element_count"])][
            record["implementation"]].append(record)
    rows = []
    for (case_id, n), implementations in sorted(groups.items()):
        medians = {}
        cvs = {}
        process_values = {}
        for implementation in ("baseline", "generated", "generic"):
            records = implementations[implementation]
            values = [record["process_median_ns"] for record in records]
            if len(values) != args.processes:
                raise RuntimeError("missing independent process measurement")
            if not all(record["correct"] for record in records):
                raise RuntimeError("candidate correctness failure")
            if implementation != "baseline":
                expected_path = ("fixed_fast" if implementation == "generated"
                                 and case_id in CANDIDATE_CASES else
                                 "fixed_generic")
                observed_paths = set()
                for record in records:
                    evidence = record["execution_evidence"]
                    if evidence["fixed_fast_path_hit"] and not evidence[
                            "fixed_generic_path_hit"]:
                        observed_paths.add("fixed_fast")
                    elif evidence["fixed_generic_path_hit"] and not evidence[
                            "fixed_fast_path_hit"]:
                        observed_paths.add("fixed_generic")
                    else:
                        observed_paths.add("mixed")
                if observed_paths != {expected_path}:
                    raise RuntimeError(
                        f"unexpected {implementation} path for {case_id}: "
                        f"{observed_paths}")
            medians[implementation] = statistics.median(values)
            cvs[implementation] = _cv(values)
            process_values[implementation] = values
        rows.append({
            "case_identifier": case_id,
            "element_count": n,
            "candidate": case_id in CANDIDATE_CASES,
            "baseline_median_ns": medians["baseline"],
            "generated_median_ns": medians["generated"],
            "generic_median_ns": medians["generic"],
            "generated_speedup": medians["baseline"] / medians["generated"],
            "generic_speedup": medians["baseline"] / medians["generic"],
            "baseline_cv": cvs["baseline"],
            "generated_cv": cvs["generated"],
            "generic_cv": cvs["generic"],
            "process_medians_ns": process_values,
            "correct": True,
        })
    return rows


def _runtime_choice(row, threshold, max_cv):
    generated_noise = max(row["baseline_cv"], row["generated_cv"])
    generic_noise = max(row["baseline_cv"], row["generic_cv"])
    if (generated_noise <= max_cv and row["generated_speedup"] >= threshold and
            row["generated_median_ns"] < row["generic_median_ns"]):
        return "GENERATED", "stable generated speedup above threshold"
    if generic_noise <= max_cv and row["generic_speedup"] >= threshold:
        return "GENERIC_FALLBACK", "stable generic speedup above threshold"
    return "ORIGINAL", "no measured-profitable Provider path"


def _code_size(args, kernel_count):
    source_bytes = args.generated_source.stat().st_size
    object_bytes = args.generated_object.stat().st_size
    section_output = subprocess.run(
        ["size", "-A", str(args.generated_object)], check=True,
        capture_output=True, text=True).stdout
    text_bytes = 0
    for line in section_output.splitlines():
        fields = line.split()
        if fields and fields[0].startswith(".text") and len(fields) >= 2:
            text_bytes += int(fields[1])
    return {
        "generated_kernel_count": kernel_count,
        "generated_source_bytes": source_bytes,
        "generated_object_file_bytes": object_bytes,
        "generated_object_text_bytes": text_bytes,
        "mean_source_bytes_per_kernel": source_bytes / kernel_count,
        "mean_object_text_bytes_per_kernel": text_bytes / kernel_count,
    }


def _build_decision_db(args, rows, metadata, code_size):
    sys.path.insert(0, str(args.generator_root))
    import crossjit_generator
    from crossjit_generator.backend.c import CBackend
    from crossjit_generator.decision import (
        ArtifactDecision, DecisionDB, DecisionEnvironment, DecisionRecord,
        MeasurementPoint, MeasurementProvenance, ProfitabilityEnvelope,
        RuntimeDecision,
    )
    from crossjit_generator.domains.elementwise import ElementwiseOptimizer
    from crossjit_generator.signature import (
        OptimizationSignature, SignatureOperation)

    kernels = {kernel["signature_id"]: kernel
               for kernel in metadata["kernels"]}
    records = []
    summary = []
    for case_id in CANDIDATE_CASES:
        kernel = kernels[CASE_TO_SIGNATURE_ID[case_id]]
        signature = OptimizationSignature.from_data(
            kernel["optimization_signature"])
        case_rows = sorted(
            (row for row in rows if row["case_identifier"] == case_id),
            key=lambda row: row["element_count"])
        if not case_rows:
            raise RuntimeError(
                f"missing tuning rows for {case_id}; available="
                f"{sorted({row['case_identifier'] for row in rows})}")
        points = []
        generated_ranges = []
        for index, row in enumerate(case_rows):
            decision_name, reason = _runtime_choice(
                row, args.speedup_threshold, args.max_cv)
            low = row["element_count"] if index == 0 else (
                case_rows[index - 1]["element_count"] + 1)
            high = row["element_count"]
            # Interpolated bucket ranges require both boundary measurements to
            # agree.  Otherwise only the measured upper point is admitted.
            if index and decision_name != _runtime_choice(
                    case_rows[index - 1], args.speedup_threshold,
                    args.max_cv)[0]:
                low = high
            decision = RuntimeDecision(decision_name)
            if decision is RuntimeDecision.GENERATED:
                generated_ranges.append((low, high))
            noise = max(row["baseline_cv"], row["generated_cv"],
                        row["generic_cv"])
            points.append(MeasurementPoint(
                low, high, row["baseline_median_ns"],
                row["generated_median_ns"], row["generic_median_ns"],
                row["generated_speedup"], row["generic_speedup"], noise,
                max(0.0, min(1.0, 1.0 - noise / args.max_cv)),
                decision, reason,
            ))
        artifact_decision = (ArtifactDecision.AUTO_SELECTED
                             if generated_ranges else
                             ArtifactDecision.AUTO_REJECTED)
        decision_reason = (
            "at least one stable generated-profitable element range"
            if generated_ranges else
            "no stable generated-profitable element range")
        environment = DecisionEnvironment(
            provider="numpy", provider_abi_version=3,
            recipe_hashes=tuple(operation.recipe_hash
                                for operation in signature.operations),
            generator_version=crossjit_generator.__version__,
            optimizer_version=ElementwiseOptimizer.VERSION,
            backend_version=CBackend.VERSION,
            target_triple=args.target_triple,
            target_features=tuple(args.target_features.split(",")),
            compiler_id=args.compiler_id,
            compiler_version=args.compiler_version,
            compiler_flags=tuple(args.compiler_flags.split()),
            artifact_format_version=1,
        )
        records.append(DecisionRecord(
            signature, environment, artifact_decision,
            MeasurementProvenance(
                "generic_signature_microbenchmark", args.processes,
                args.samples, "2026-09-28T00:00:00+08:00"),
            ProfitabilityEnvelope(tuple(points)),
            math.ceil(code_size["generated_source_bytes"] / len(CANDIDATE_CASES)),
            math.ceil(code_size["generated_object_file_bytes"] /
                      len(CANDIDATE_CASES)),
            0, kernel["candidate_hash"],
            min(point.confidence for point in points), decision_reason,
        ))
        summary.append({
            "case_identifier": case_id,
            "signature_hash": signature.stable_hash(),
            "artifact_decision": artifact_decision.value,
            "generated_ranges": generated_ranges,
            "reason": decision_reason,
        })

    recipe_hashes = {
        operation.operation_id: operation.recipe_hash
        for record in records for operation in record.signature.operations
    }
    controls = {
        "array_scalar_2": (
            2,
            (("numpy.add", (0, 1)), ("numpy.multiply", (3, 2))),
        ),
        "scalar_array_2": (
            1,
            (("numpy.multiply", (0, 1)), ("numpy.add", (3, 2))),
        ),
    }
    base_environment = records[0].environment
    for case_id, (scalar_mask, operations) in controls.items():
        signature = OptimizationSignature(
            provider="numpy", provider_abi_version=3, domain="elementwise",
            input_count=3,
            operations=tuple(SignatureOperation(
                operation_id, operands, recipe_hashes[operation_id])
                for operation_id, operands in operations),
            dtype_class="float64", layout_class="contiguous_same_shape",
            error_mode="numpy_fpe_fallback",
            materialization_class="virtual_intermediates_single_output",
            scalar_mask=scalar_mask,
        )
        environment = replace(
            base_environment,
            recipe_hashes=tuple(operation.recipe_hash
                                for operation in signature.operations))
        case_rows = sorted(
            (row for row in rows if row["case_identifier"] == case_id),
            key=lambda row: row["element_count"])
        points = []
        generic_ranges = []
        for index, row in enumerate(case_rows):
            generic_noise = max(row["baseline_cv"], row["generic_cv"])
            profitable = (generic_noise <= args.max_cv and
                          row["generic_speedup"] >= args.speedup_threshold)
            decision = (RuntimeDecision.GENERIC_FALLBACK if profitable
                        else RuntimeDecision.ORIGINAL)
            low = row["element_count"] if index == 0 else (
                case_rows[index - 1]["element_count"] + 1)
            high = row["element_count"]
            if index:
                prior = case_rows[index - 1]
                prior_profitable = (
                    max(prior["baseline_cv"], prior["generic_cv"]) <=
                    args.max_cv and prior["generic_speedup"] >=
                    args.speedup_threshold)
                if prior_profitable != profitable:
                    low = high
            if profitable:
                generic_ranges.append((low, high))
            points.append(MeasurementPoint(
                low, high, row["baseline_median_ns"], None,
                row["generic_median_ns"], None, row["generic_speedup"],
                generic_noise,
                max(0.0, min(1.0, 1.0 - generic_noise / args.max_cv)),
                decision,
                ("stable generic speedup above threshold" if profitable else
                 "generic fallback is unprofitable or noisy"),
            ))
        records.append(DecisionRecord(
            signature, environment, ArtifactDecision.AUTO_REJECTED,
            MeasurementProvenance(
                "generic_signature_microbenchmark", args.processes,
                args.samples, "2026-09-28T00:00:00+08:00"),
            ProfitabilityEnvelope(tuple(points)), 0, 0, 0, None,
            min(point.confidence for point in points),
            "no generated artifact; generic fallback measured separately",
        ))
        summary.append({
            "case_identifier": case_id,
            "signature_hash": signature.stable_hash(),
            "artifact_decision": ArtifactDecision.AUTO_REJECTED.value,
            "generic_ranges": generic_ranges,
            "reason": "fallback-only control; no generated artifact",
        })
    database = DecisionDB(records)
    for _ in range(8):
        size = len(database.serialize().encode("utf-8"))
        updated = [replace(record, decision_db_bytes=size)
                   for record in database.records()]
        next_database = DecisionDB(updated)
        if next_database.serialize() == database.serialize():
            break
        database = next_database
    return database, summary


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--harness", type=Path, default=(
        Path(__file__).with_name("run_fixed_fusion_break_even.py")))
    parser.add_argument("--generator-root", type=Path,
                        default=root.parent / "crossjit-generator")
    parser.add_argument("--generated-metadata", type=Path, required=True)
    parser.add_argument("--generated-source", type=Path, required=True)
    parser.add_argument("--generated-object", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-elements", type=int, default=2_000_000)
    parser.add_argument("--min-iterations", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--speedup-threshold", type=float, default=1.05)
    parser.add_argument("--max-cv", type=float, default=0.15)
    parser.add_argument("--target-triple", default="x86_64-linux-gnu")
    parser.add_argument("--target-features", default=(
        "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,sahf,cx16"))
    parser.add_argument("--compiler-id", default="gcc")
    parser.add_argument("--compiler-version", default="13.3.0")
    parser.add_argument("--compiler-flags", default=(
        "-std=c11 -O3 -fno-strict-aliasing -msse -msse2 -msse3 "
        "-mssse3 -msse4.1 -msse4.2 -mpopcnt -msahf -mcx16 -fPIC"))
    args = parser.parse_args()

    rows = _measure(args)
    metadata = json.loads(args.generated_metadata.read_text())
    code_size = _code_size(args, len(metadata["kernels"]))
    database, decisions = _build_decision_db(
        args, rows, metadata, code_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decision_path = args.output_dir / "g12_decision_db.json"
    result_path = args.output_dir / "g12_candidate_tuning.json"
    decision_path.write_text(database.serialize())
    document = {
        "schema_version": 1,
        "method": "generic_signature_microbenchmark",
        "configuration": {
            "element_grid": list(GRID),
            "independent_processes": args.processes,
            "samples_per_process": args.samples,
            "speedup_threshold": args.speedup_threshold,
            "max_cv": args.max_cv,
            "forced_admission": True,
            "formal_npbench_evaluation": False,
            "frequency_used_for_selection": False,
        },
        "candidate_decisions": decisions,
        "code_size": code_size,
        "decision_db_bytes": decision_path.stat().st_size,
        "rows": rows,
    }
    result_path.write_text(json.dumps(
        document, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    counts = defaultdict(int)
    for decision in decisions:
        counts[decision["artifact_decision"]] += 1
    print(json.dumps({
        "candidate_count": len(decisions),
        "decision_counts": counts,
        "decision_db": str(decision_path),
        "results": str(result_path),
        "code_size": code_size,
    }, default=dict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
