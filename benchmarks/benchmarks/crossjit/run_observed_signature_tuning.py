#!/usr/bin/env python3
"""Tune observed structural signatures with generic microbenchmarks only."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from itertools import permutations


GRID = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536)
SYMBOLS = {
    "numpy.add": "+",
    "numpy.subtract": "-",
    "numpy.multiply": "*",
    "numpy.true_divide": "/",
}


def _cv(values):
    return (statistics.stdev(values) / statistics.mean(values)
            if len(values) > 1 and statistics.mean(values) else 0.0)


def _load_cases(observations, metadata):
    generated = {
        kernel["optimization_signature_hash"]
        for kernel in metadata["kernels"]
        if kernel["generation_source"] == "observed"
    }
    result = []
    for record in observations["observations"]:
        if record["signature_hash"] in generated:
            result.append(record)
    return sorted(result, key=lambda item: item["signature_hash"])


def _array(n, dtype, layout, offset):
    if layout == "contiguous_same_shape":
        return 1 + np.arange(n, dtype=dtype) / max(n, 1) + offset
    if layout == "general_strided":
        base = 1 + np.arange(n * 2, dtype=dtype) / max(n, 1) + offset
        return base[::2]
    if layout == "inner_contiguous":
        if n % 2:
            raise ValueError("inner-contiguous fixture requires even N")
        base = (1 + np.arange(n * 2, dtype=dtype) / max(n, 1) + offset
                ).reshape(4, n // 2)
        return base[::2, :]
    raise ValueError(f"unsupported tuning layout: {layout}")


def _inputs(signature, n, permutation):
    values = [None] * signature["input_count"]
    array_index = 0
    for structural_index in range(signature["input_count"]):
        if signature["scalar_mask"] & (1 << structural_index):
            value = 0.25 + structural_index * 0.125
        else:
            value = _array(
                n, np.float64, signature["layout_class"],
                array_index * 0.125)
            array_index += 1
        values[permutation[structural_index]] = value
    return tuple(values)


def _functions(case_hash, signature, serial, permutation):
    input_count = signature["input_count"]
    names = [f"v{index}" for index in range(input_count)]
    arguments = ", ".join((*names, "count"))
    expressions = [names[permutation[index]] for index in range(input_count)]
    for index, operation in enumerate(signature["operations"]):
        left, right = operation["operands"]
        expressions.append(
            f"({expressions[left]} {SYMBOLS[operation['operation_id']]} "
            f"{expressions[right]})")
    result_expression = expressions[-1]
    array_name = names[permutation[next(
        index for index in range(input_count)
        if not signature["scalar_mask"] & (1 << index))]]
    source = (
        f"def expression_{serial}({arguments}):\n"
        "    for _ in range(count):\n"
        f"        result = {result_expression}\n"
        "    return result\n"
        f"def control_{serial}({arguments}):\n"
        "    for _ in range(count):\n"
        f"        result = {array_name}\n"
        "    return result\n"
    )
    namespace = {}
    exec(compile(source, f"<crossjit-signature-{case_hash}-{serial}>", "exec"),
         namespace)
    return namespace[f"expression_{serial}"], namespace[f"control_{serial}"]


def _measure_cell(case, n, args, serial, permutation):
    signature = case["signature"]
    values = _inputs(signature, n, permutation)
    expression, control = _functions(
        case["signature_hash"], signature, serial, permutation)
    expected = expression(*values, 1)
    internal.crossjit_reset_stats()
    umath._crossjit_reset_stats()
    warm_count = internal.TIER2_THRESHOLD * 2
    actual = expression(*values, warm_count)
    control(*values, warm_count)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    iterations = max(
        args.min_iterations,
        min(args.max_iterations, args.target_elements // max(n, 1)),
    )
    before = umath._crossjit_stats()
    samples = []
    controls = []
    for sample in range(args.samples):
        order = ((expression, samples), (control, controls))
        if sample % 2:
            order = tuple(reversed(order))
        for function, results in order:
            started = time.perf_counter_ns()
            value = function(*values, iterations)
            results.append((time.perf_counter_ns() - started) / iterations)
            if function is expression:
                actual = value
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    after = umath._crossjit_stats()
    deltas = {name: after[name] - before[name] for name in (
        "exec_hit", "exec_miss", "exec_error", "regions_executed",
        "generated_exec_hit", "generic_fallback_hit",
        "fixed_fast_path_hit", "fixed_generic_path_hit")}
    if args.mode == "generated":
        valid_path = (deltas["generated_exec_hit"] > 0 and
                      deltas["generic_fallback_hit"] == 0)
    elif args.mode == "generic":
        valid_path = (deltas["generic_fallback_hit"] > 0 and
                      deltas["generated_exec_hit"] == 0)
    else:
        valid_path = True
    if not valid_path:
        raise RuntimeError(
            f"unexpected {args.mode} path for {case['signature_hash']} N={n}: "
            f"{deltas}; provider={after}")
    return {
        "signature_hash": case["signature_hash"],
        "element_count": n,
        "mode": args.mode,
        "process_median_ns": statistics.median(samples),
        "process_cv": _cv(samples),
        "control_median_ns": statistics.median(controls),
        "correct": True,
        "execution_evidence": deltas,
    }


def _discover_mappings(cases):
    mappings = {}
    rejected = []
    serial = 100000
    for case in cases:
        signature = case["signature"]
        found = None
        for permutation in permutations(range(signature["input_count"])):
            serial += 1
            values = _inputs(signature, 16, permutation)
            expression, _ = _functions(
                case["signature_hash"], signature, serial, permutation)
            internal.crossjit_reset_stats()
            umath._crossjit_reset_stats()
            expected = expression(*values, 1)
            actual = expression(*values, internal.TIER2_THRESHOLD * 2)
            np.testing.assert_allclose(
                actual, expected, rtol=1e-12, atol=1e-12)
            stats = umath._crossjit_stats()
            if stats["generated_exec_hit"] > 0:
                found = permutation
                break
        if found is None:
            rejected.append({
                "signature_hash": case["signature_hash"],
                "reason": "generic harness could not reconstruct exact Host region",
            })
        else:
            mappings[case["signature_hash"]] = list(found)
    return {"mappings": mappings, "rejected": rejected}


def _worker(args):
    global np, internal, umath
    import numpy as np
    import _testinternalcapi as internal
    from numpy._core import _multiarray_umath as umath

    observations = json.loads(args.observations.read_text())
    metadata = json.loads(args.generated_metadata.read_text())
    cases = _load_cases(observations, metadata)
    if args.discover_mappings:
        print(json.dumps(_discover_mappings(cases), sort_keys=True))
        return 0
    mappings = json.loads(args.mappings_json)
    cases = [case for case in cases if case["signature_hash"] in mappings]
    selected = None
    if args.cells:
        selected = {
            (cell.split(":", 1)[0], int(cell.split(":", 1)[1]))
            for cell in args.cells.split(",")
        }
    records = []
    serial = 0
    for case in cases:
        for n in GRID:
            if selected is not None and (case["signature_hash"], n) not in selected:
                continue
            serial += 1
            records.append(_measure_cell(
                case, n, args, serial,
                tuple(mappings[case["signature_hash"]])))
    print(json.dumps(records, separators=(",", ":")))
    return 0


def _worker_command(args, mode):
    return [
        args.python, str(Path(__file__).resolve()), "--worker", "--mode", mode,
        "--observations", str(args.observations),
        "--generated-metadata", str(args.generated_metadata),
        "--samples", str(args.samples),
        "--target-elements", str(args.target_elements),
        "--min-iterations", str(args.min_iterations),
        "--max-iterations", str(args.max_iterations),
        "--cells", args.cells,
        "--mappings-json", args.mappings_json,
    ]


def _run_process(args, mode):
    env = os.environ.copy()
    env.update({"PYTHON_JIT": "1", "OPENBLAS_NUM_THREADS": "1"})
    env.pop("NPY_CROSSJIT_DIAGNOSTICS", None)
    env.pop("NPY_CROSSJIT_TELEMETRY_TIMING", None)
    env.pop("NPY_CROSSJIT_DIAGNOSTIC_DISABLE_GENERATED", None)
    if mode == "baseline":
        env["PYTHON_JIT_CROSSJIT"] = "0"
        env.pop("NPY_CROSSJIT_MIN_WORK", None)
    else:
        env["PYTHON_JIT_CROSSJIT"] = "1"
        env["NPY_CROSSJIT_MIN_WORK"] = "0"
        if mode == "generic":
            env["NPY_CROSSJIT_DIAGNOSTIC_DISABLE_GENERATED"] = "1"
    completed = subprocess.run(
        _worker_command(args, mode), check=True, capture_output=True,
        text=True, env=env)
    return json.loads(completed.stdout)


def _measure(args):
    discovery_env = os.environ.copy()
    discovery_env.update({
        "PYTHON_JIT": "1", "PYTHON_JIT_CROSSJIT": "1",
        "NPY_CROSSJIT_MIN_WORK": "0", "OPENBLAS_NUM_THREADS": "1",
    })
    discovery_env.pop("NPY_CROSSJIT_DIAGNOSTIC_DISABLE_GENERATED", None)
    discovery_command = _worker_command(args, "generated")
    discovery_command.append("--discover-mappings")
    discovered = subprocess.run(
        discovery_command, check=True, capture_output=True, text=True,
        env=discovery_env)
    discovery = json.loads(discovered.stdout)
    args.reconstruction_rejections = discovery["rejected"]
    args.mappings_json = json.dumps(
        discovery["mappings"], sort_keys=True, separators=(",", ":"))
    print("structural mappings=" + args.mappings_json, flush=True)
    print("reconstruction rejections=" + json.dumps(
        args.reconstruction_rejections, sort_keys=True), flush=True)
    schedule = [(process, mode) for process in range(args.processes)
                for mode in ("baseline", "generated", "generic")]
    random.Random(args.seed).shuffle(schedule)
    raw = []
    for index, (process, mode) in enumerate(schedule, 1):
        print(f"[{index}/{len(schedule)}] process={process} mode={mode}",
              flush=True)
        records = _run_process(args, mode)
        for record in records:
            record["independent_process_index"] = process
            raw.append(record)
    groups = defaultdict(lambda: defaultdict(list))
    for record in raw:
        groups[(record["signature_hash"], record["element_count"])][
            record["mode"]].append(record)
    rows = []
    for (signature_hash, n), modes in sorted(groups.items()):
        values = {}
        cvs = {}
        process_values = {}
        for mode in ("baseline", "generated", "generic"):
            records = modes[mode]
            samples = [record["process_median_ns"] for record in records]
            if len(samples) != args.processes or not all(
                    record["correct"] for record in records):
                raise RuntimeError("incomplete generic signature measurement")
            values[mode] = statistics.median(samples)
            cvs[mode] = _cv(samples)
            process_values[mode] = samples
        rows.append({
            "signature_hash": signature_hash,
            "element_count": n,
            "baseline_median_ns": values["baseline"],
            "generated_median_ns": values["generated"],
            "generic_median_ns": values["generic"],
            "generated_speedup": values["baseline"] / values["generated"],
            "generic_speedup": values["baseline"] / values["generic"],
            "baseline_cv": cvs["baseline"],
            "generated_cv": cvs["generated"],
            "generic_cv": cvs["generic"],
            "process_medians_ns": process_values,
            "correct": True,
        })
    return rows


def _choice(row, threshold, max_cv):
    generated_noise = max(row["baseline_cv"], row["generated_cv"])
    generic_noise = max(row["baseline_cv"], row["generic_cv"])
    if (generated_noise <= max_cv and
            row["generated_speedup"] >= threshold and
            row["generated_median_ns"] < row["generic_median_ns"]):
        return "GENERATED", "stable generated speedup above threshold"
    if generic_noise <= max_cv and row["generic_speedup"] >= threshold:
        return "GENERIC_FALLBACK", "stable generic speedup above threshold"
    return "ORIGINAL", "no measured-profitable Provider path"


def _build_database(args, rows):
    sys.path.insert(0, str(args.generator_root))
    import crossjit_generator
    from crossjit_generator.backend.c import CBackend
    from crossjit_generator.decision import (
        ArtifactDecision, DecisionDB, DecisionEnvironment, DecisionRecord,
        MeasurementPoint, MeasurementProvenance, ProfitabilityEnvelope,
        RuntimeDecision,
    )
    from crossjit_generator.domains.elementwise import ElementwiseOptimizer
    from crossjit_generator.signature import ObservedSignatures

    base = DecisionDB.from_data(json.loads(args.base_decision_db.read_text()))
    observations = ObservedSignatures.from_data(
        json.loads(args.observations.read_text()))
    metadata = json.loads(args.generated_metadata.read_text())
    environment_data = json.loads(args.decision_environment.read_text())
    measured_hashes = {row["signature_hash"] for row in rows}
    supported = {
        kernel["optimization_signature_hash"]: kernel
        for kernel in metadata["kernels"]
        if (kernel["generation_source"] == "observed" and
            kernel["optimization_signature_hash"] in measured_hashes)
    }
    signatures = {
        profile.signature.stable_hash(): profile.signature
        for profile in observations.observations()
    }
    source_bytes = args.generated_source.stat().st_size
    object_bytes = args.generated_object.stat().st_size
    per_source = math.ceil(source_bytes / len(metadata["kernels"]))
    per_object = math.ceil(object_bytes / len(metadata["kernels"]))
    records = list(base.records())
    decisions = []
    for signature_hash in sorted(supported):
        signature = signatures[signature_hash]
        case_rows = sorted(
            (row for row in rows if row["signature_hash"] == signature_hash),
            key=lambda row: row["element_count"])
        points = []
        generated_ranges = []
        generic_ranges = []
        for index, row in enumerate(case_rows):
            name, reason = _choice(row, args.speedup_threshold, args.max_cv)
            low = row["element_count"] if index == 0 else (
                case_rows[index - 1]["element_count"] + 1)
            high = row["element_count"]
            if index and name != _choice(
                    case_rows[index - 1], args.speedup_threshold,
                    args.max_cv)[0]:
                low = high
            if name == "GENERATED":
                generated_ranges.append((low, high))
            elif name == "GENERIC_FALLBACK":
                generic_ranges.append((low, high))
            noise = max(row["baseline_cv"], row["generated_cv"],
                        row["generic_cv"])
            points.append(MeasurementPoint(
                low, high, row["baseline_median_ns"],
                row["generated_median_ns"], row["generic_median_ns"],
                row["generated_speedup"], row["generic_speedup"], noise,
                max(0.0, min(1.0, 1.0 - noise / args.max_cv)),
                RuntimeDecision(name), reason,
            ))
        artifact = (ArtifactDecision.AUTO_SELECTED if generated_ranges else
                    ArtifactDecision.AUTO_REJECTED)
        environment = DecisionEnvironment(
            provider=signature.provider,
            provider_abi_version=signature.provider_abi_version,
            recipe_hashes=tuple(op.recipe_hash for op in signature.operations),
            generator_version=crossjit_generator.__version__,
            optimizer_version=ElementwiseOptimizer.VERSION,
            backend_version=CBackend.VERSION,
            target_triple=environment_data["target_triple"],
            target_features=tuple(environment_data["target_features"]),
            compiler_id=environment_data["compiler_id"],
            compiler_version=environment_data["compiler_version"],
            compiler_flags=tuple(environment_data["compiler_flags"]),
            artifact_format_version=environment_data["artifact_format_version"],
        )
        records.append(DecisionRecord(
            signature, environment, artifact,
            MeasurementProvenance(
                "generic_signature_microbenchmark", args.processes,
                args.samples, "2026-09-28T10:50:00Z"),
            ProfitabilityEnvelope(tuple(points)),
            per_source if generated_ranges else 0,
            per_object if generated_ranges else 0,
            0,
            supported[signature_hash]["candidate_hash"]
            if generated_ranges else None,
            min(point.confidence for point in points),
            ("at least one stable generated-profitable element range"
             if generated_ranges else
             "no stable generated-profitable element range"),
        ))
        decisions.append({
            "signature_hash": signature_hash,
            "artifact_decision": artifact.value,
            "generated_ranges": generated_ranges,
            "generic_ranges": generic_ranges,
        })
    database = DecisionDB(records)
    for _ in range(8):
        size = len(database.serialize().encode("utf-8"))
        next_database = DecisionDB(
            replace(record, decision_db_bytes=size)
            for record in database.records())
        if next_database.serialize() == database.serialize():
            break
        database = next_database
    return database, decisions


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("baseline", "generated", "generic"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--generator-root", type=Path,
                        default=root.parent / "crossjit-generator")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--generated-metadata", type=Path, required=True)
    parser.add_argument("--generated-source", type=Path)
    parser.add_argument("--generated-object", type=Path)
    parser.add_argument("--base-decision-db", type=Path)
    parser.add_argument("--decision-environment", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-elements", type=int, default=2_000_000)
    parser.add_argument("--min-iterations", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--speedup-threshold", type=float, default=1.05)
    parser.add_argument("--max-cv", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1414)
    parser.add_argument("--cells", default="")
    parser.add_argument("--mappings-json", default="{}")
    parser.add_argument("--discover-mappings", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.mode is None:
            parser.error("--worker requires --mode")
        return _worker(args)
    for name in ("generated_source", "generated_object", "base_decision_db",
                 "decision_environment", "output_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    rows = _measure(args)
    database, decisions = _build_database(args, rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    database_path = args.output_dir / "g14_decision_db.json"
    result_path = args.output_dir / "g14_signature_tuning.json"
    database_path.write_text(database.serialize())
    result = {
        "schema_version": 1,
        "method": "generic_signature_microbenchmark",
        "configuration": {
            "element_grid": list(GRID),
            "independent_processes": args.processes,
            "samples_per_process": args.samples,
            "speedup_threshold": args.speedup_threshold,
            "max_cv": args.max_cv,
            "formal_npbench_evaluation": False,
            "benchmark_identity_used_for_selection": False,
        },
        "candidate_decisions": decisions,
        "reconstruction_rejections": args.reconstruction_rejections,
        "rows": rows,
    }
    result_path.write_text(json.dumps(
        result, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "supported_observed_candidates": len(decisions),
        "selected": sum(d["artifact_decision"] == "AUTO_SELECTED"
                        for d in decisions),
        "rejected": sum(d["artifact_decision"] == "AUTO_REJECTED"
                        for d in decisions),
        "decision_db": str(database_path),
        "results": str(result_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
