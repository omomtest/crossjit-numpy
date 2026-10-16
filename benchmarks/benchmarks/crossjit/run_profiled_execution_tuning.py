#!/usr/bin/env python3
"""Generic execution-specialization microbenchmarks for PF4/PF5.

The controller compares ordinary NumPy, the existing generic CrossJIT
interpreter, and a generated descriptor-selected artifact.  Inputs are
structural operation/DAG signatures plus iteration fixtures; no workload or
benchmark identity participates in selection or runtime policy.
"""

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
import time


DEFAULT_SHAPES = (
    (2, 8), (2, 16), (4, 16), (4, 64), (8, 128), (16, 256),
    (59, 59), (32, 512), (64, 1024),
)
OPERATORS = {
    "numpy.add": "+",
    "numpy.subtract": "-",
    "numpy.multiply": "*",
    "numpy.true_divide": "/",
}


def _cv(values):
    return (statistics.stdev(values) / statistics.mean(values)
            if len(values) > 1 and statistics.mean(values) else 0.0)


def _cases(metadata, layouts):
    return sorted((
        kernel for kernel in metadata["kernels"]
        if kernel.get("profiled_layout_class") in layouts and
        kernel.get("entry_abi") in ("flat_v1", "outer_inner_v1", "affine_v2") and
        "iteration_descriptor" in kernel
    ), key=lambda item: item["execution_specialization_hash"])


def _expression(signature, prepared_slot=None, prepared_operation="mul"):
    values = [f"v{index}" for index in range(signature["input_count"])]
    if prepared_slot is not None:
        op = {"add": "+", "sub": "-", "mul": "*"}[prepared_operation]
        values[prepared_slot] = f"(v{prepared_slot} {op} preparation_factor)"
    for operation in signature["operations"]:
        left, right = operation["operands"]
        values.append(
            f"({values[left]} {OPERATORS[operation['operation_id']]} "
            f"{values[right]})")
    return values[-1]


def _functions(case, serial, prepared_slot=None, prepared_operation="mul"):
    signature = case["optimization_signature"]
    names = [f"v{index}" for index in range(signature["input_count"])]
    if prepared_slot is not None:
        names.append("preparation_factor")
    arguments = ", ".join((*names, "count"))
    expression = _expression(signature, prepared_slot, prepared_operation)
    array_name = next(
        names[index] for index, descriptor in enumerate(
            case["iteration_descriptor"]["operands"])
        if descriptor["access_kind"] not in ("SCALAR", "ZERO_STRIDE"))
    source = (
        f"def expression_{serial}({arguments}):\n"
        "    for _ in range(count):\n"
        f"        result = {expression}\n"
        "    return result\n"
        f"def control_{serial}({arguments}):\n"
        "    for _ in range(count):\n"
        f"        result = {array_name}\n"
        "    return result\n"
    )
    namespace = {}
    exec(compile(source, "<crossjit-execution-specialization>", "exec"),
         namespace)
    return namespace[f"expression_{serial}"], namespace[f"control_{serial}"]


def _array(outer, inner, offset, padding=3):
    padded = inner + padding
    base = (1.0 + np.arange(outer * padded, dtype=np.float64) /
            max(outer * inner, 1) + offset).reshape(outer, padded)
    value = base[:, :inner]
    assert padding == 0 or not value.flags.c_contiguous or outer == 1
    return value


def _inputs(case, outer, inner, stride=2, scalar_kind="legacy", scalar_pattern="", contiguous_padding=""):
    result = []
    paddings = [int(item) for item in contiguous_padding.split(",")] if contiguous_padding else []
    contiguous_count = sum(item["access_kind"] == "CONTIGUOUS" for item in case["iteration_descriptor"]["operands"])
    if paddings and (len(paddings) != contiguous_count or min(paddings) < 0):
        raise ValueError("padding must specify every unit-stride array")
    contiguous_index = 0
    array_index = 0
    pattern = scalar_pattern.split(",") if scalar_pattern else []
    scalar_count = sum(descriptor["access_kind"] == "SCALAR" for descriptor in
                       case["iteration_descriptor"]["operands"])
    if pattern and (len(pattern) != scalar_count or
                    any(kind not in ("python", "numpy", "integer", "prepared") for kind in pattern)):
        raise ValueError("scalar pattern must specify every scalar representation")
    scalar_index = 0
    for descriptor in case["iteration_descriptor"]["operands"]:
        access = descriptor["access_kind"]
        if access == "SCALAR":
            scalar = 0.25 + len(result) * 0.125
            kind = pattern[scalar_index] if pattern else scalar_kind
            scalar_index += 1
            use_numpy = (kind == "numpy" or
                         (kind == "legacy" and
                          case.get("entry_abi") == "affine_v2"))
            result.append(2 + len(result) if kind == "integer" else
                          np.float64(scalar) if use_numpy else scalar)
        elif access == "ZERO_STRIDE":
            per_row = (
                0.5 + len(result) * 0.125 +
                np.arange(outer, dtype=np.float64) / max(outer, 1)
            ).reshape(outer, 1)
            result.append(per_row)
        elif access == "CONTIGUOUS":
            padding = paddings[contiguous_index] if paddings else 3
            result.append(_array(outer, inner, array_index * 0.125, padding))
            contiguous_index += 1
            array_index += 1
        elif access == "FIXED_STRIDE":
            padded = inner * stride + 3
            base = (1.0 + np.arange(outer * padded, dtype=np.float64) /
                    max(outer * inner, 1) + array_index * 0.125).reshape(
                        outer, padded)
            result.append(base[:, :inner * stride:stride])
            array_index += 1
        else:
            raise ValueError(f"unsupported tuning access kind: {access}")
    if case["iteration_descriptor"]["iteration_rank_class"] == "FLAT":
        if outer != 1:
            raise ValueError("flat descriptor requires outer count one")
        result = [value[0] if isinstance(value, np.ndarray) else value
                  for value in result]
    return tuple(result)


def _measure_cell(case, outer, inner, args, serial):
    values = _inputs(case, outer, inner, args.affine_stride, args.scalar_kind,
                     args.scalar_pattern, args.contiguous_padding)
    scalar_slots = [i for i, operand in enumerate(case["iteration_descriptor"]["operands"])
                    if operand["access_kind"] == "SCALAR"]
    pattern = args.scalar_pattern.split(",") if args.scalar_pattern else []
    prepared_slots = [slot for slot, kind in zip(scalar_slots, pattern)
                      if kind == "prepared"]
    if len(prepared_slots) > 1:
        raise ValueError("one Host scalar preparation per region is supported")
    prepared_slot = prepared_slots[0] if prepared_slots else None
    if prepared_slot is not None:
        uses = sum(prepared_slot in operation["operands"] for operation in
                   case["optimization_signature"]["operations"])
        if uses != 1:
            raise ValueError("prepared fixture must have exactly one scalar consumer")
        values += (1.0,)
    expression, control = _functions(case, serial, prepared_slot, args.prepared_operation)
    expected = expression(*values, 1)
    # Inspect the prepared full-DAG candidate only during untimed warmup.
    # Production counters deliberately hide candidate detail when diagnostics
    # are disabled, so retain this proof before disabling diagnostics for timing.
    os.environ["NPY_CROSSJIT_DIAGNOSTICS"] = "1" if args.mode != "baseline" else "0"
    internal.crossjit_reset_stats()
    umath._crossjit_reset_stats()
    warm = internal.TIER2_THRESHOLD * 2
    actual = expression(*values, warm)
    control(*values, warm)
    expected_nodes = len(case["optimization_signature"]["operations"])
    operations = case["optimization_signature"]["operations"]
    op_ids = {"numpy.add": 0, "numpy.subtract": 1,
              "numpy.multiply": 2, "numpy.true_divide": 3}
    expected_topology = tuple(dict(op=op_ids[op["operation_id"]],
        left=op["operands"][0], right=op["operands"][1],
        output=case["optimization_signature"]["input_count"] + index)
        for index, op in enumerate(operations))
    reached = [record for record in umath._crossjit_stats()["observed_signatures"]
               if record["execution_count"]]
    full_candidate = (len(reached) == 1 and
        reached[0]["node_count"] == expected_nodes and
        reached[0]["outer_count"] == outer and
        reached[0]["inner_extent"] == inner and
        tuple(reached[0]["nodes"]) == expected_topology)
    observed_class_mask = reached[0].get("sample_value_class_mask", 0) if len(reached) == 1 else 0
    if prepared_slot is not None and args.mode != "baseline":
        expected_classes = {"python": 1, "numpy": 2, "integer": 3, "prepared": 5}
        expected_mask = sum(expected_classes[kind] << (4 * slot)
                            for slot, kind in zip(scalar_slots, pattern))
        if observed_class_mask != expected_mask or not internal.crossjit_stats()["prepared_scalar_values"]:
            raise RuntimeError("fixture did not reach the preparation-inclusive scalar class")
    observed_nodes = reached[0]["node_count"] if len(reached) == 1 else 0
    # These finite strict-order fixtures must be identical. NumPy testing
    # helpers contain arithmetic DAGs themselves and can pollute global
    # generated-path counters during complete-path calibration.
    if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
        raise AssertionError("strict-order fixture output changed")
    elements = outer * inner
    iterations = max(
        args.min_iterations,
        min(args.max_iterations, args.target_elements // max(elements, 1)))
    # A backedge executor may leave one ordinary first iteration per call.
    # Determine this entry cost outside timing, rather than assuming every
    # iteration enters Tier 2 or permitting arbitrary missing executions.
    entry_ordinary_iterations = 0
    if args.mode != "baseline":
        hit_field = ("generated_exec_hit" if args.mode == "generated"
                     else "generic_fallback_hit")
        other_field = ("generic_fallback_hit" if args.mode == "generated"
                       else "generated_exec_hit")
        observations = []
        for count in (1, 2, 10, iterations, iterations):
            umath._crossjit_reset_stats()
            expression(*values, count)
            stats = umath._crossjit_stats()
            observations.append((count, stats[hit_field]))
            if stats[other_field] or stats["exec_miss"] or stats["exec_error"]:
                raise RuntimeError("probe used a foreign or replaying execution path")
        offsets = {count - hits for count, hits in observations}
        if not full_candidate or len(offsets) != 1 or not offsets <= {0, 1}:
            raise RuntimeError(f"unstable or incomplete DAG entry: {observations}; "
                               f"reached={reached}")
        entry_ordinary_iterations = offsets.pop()
    os.environ["NPY_CROSSJIT_DIAGNOSTICS"] = "0"
    umath._crossjit_reset_stats()
    before = umath._crossjit_stats()
    prepared_before = internal.crossjit_stats().get("prepared_scalar_values", 0)
    samples = []
    controls = []
    for sample in range(args.samples):
        order = ((expression, samples), (control, controls))
        if sample % 2:
            order = tuple(reversed(order))
        for function, target in order:
            started = time.perf_counter_ns()
            value = function(*values, iterations)
            target.append((time.perf_counter_ns() - started) / iterations)
            if function is expression:
                actual = value
        if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
            raise AssertionError("strict-order fixture output changed")
    after = umath._crossjit_stats()
    deltas = {name: after[name] - before[name] for name in (
        "exec_hit", "exec_miss", "exec_error", "regions_executed",
        "generated_exec_hit", "generic_fallback_hit",
        "output_allocations")}
    expected_executions = (iterations - entry_ordinary_iterations) * args.samples
    prepared_executions = internal.crossjit_stats().get("prepared_scalar_values", 0) - prepared_before
    replay_free = deltas["exec_miss"] == deltas["exec_error"] == 0
    if args.mode == "generated":
        valid = (deltas["generated_exec_hit"] == expected_executions and
                 deltas["generic_fallback_hit"] == 0 and full_candidate and replay_free)
    elif args.mode == "generic":
        valid = (deltas["generic_fallback_hit"] == expected_executions and
                 deltas["generated_exec_hit"] == 0 and full_candidate and replay_free)
    else:
        valid = deltas["regions_executed"] == 0
    if not valid:
        raise RuntimeError(
            f"unexpected {args.mode} path for "
            f"{case['execution_specialization_hash']} {outer}x{inner}: "
            f"{deltas}")
    return {
        "execution_specialization_hash":
            case["execution_specialization_hash"],
        "structural_signature_hash": case["optimization_signature_hash"],
        "profiled_source": case["profiled_source"],
        "outer_count": outer,
        "inner_extent": inner,
        "affine_stride": args.affine_stride,
        "scalar_kind": args.scalar_kind,
        "scalar_pattern": args.scalar_pattern,
        "prepared_operation": args.prepared_operation,
        "contiguous_padding": args.contiguous_padding,
        "observed_scalar_class_mask": observed_class_mask,
        "prepared_scalar_values": prepared_executions,
        "expected_region_nodes": expected_nodes,
        "observed_region_nodes": observed_nodes,
        "expected_executions": expected_executions,
        "timed_iterations": iterations * args.samples,
        "entry_ordinary_iterations_per_call": entry_ordinary_iterations,
        "element_count": elements,
        "mode": args.mode,
        "process_median_ns": statistics.median(samples),
        "process_cv": _cv(samples),
        "control_median_ns": statistics.median(controls),
        "execution_evidence": deltas,
        "correct": True,
    }


def _worker(args):
    global np, internal, umath
    import numpy as np
    import _testinternalcapi as internal
    from numpy._core import _multiarray_umath as umath

    metadata = json.loads(args.generated_metadata.read_text())
    layouts = set(args.layouts.split(","))
    cases = _cases(metadata, layouts)
    selected = None
    if args.specializations:
        selected = set(args.specializations.split(","))
        cases = [case for case in cases
                 if case["execution_specialization_hash"] in selected]
    if not cases:
        raise ValueError("no selected descriptor can execute the calibration")
    shapes = tuple(
        tuple(map(int, item.split("x", 1)))
        for item in args.shapes.split(",") if item)
    records = []
    serial = 0
    for case in cases:
        for outer, inner in shapes:
            if (case["iteration_descriptor"]["iteration_rank_class"] == "FLAT"
                    and outer != 1):
                continue
            serial += 1
            records.append(_measure_cell(
                case, outer, inner, args, serial))
    print(json.dumps(records, separators=(",", ":")))
    return 0


def _command(args, mode):
    return [
        args.python, str(Path(__file__).resolve()), "--worker", "--mode", mode,
        "--generated-metadata", str(args.generated_metadata),
        "--layouts", args.layouts, "--shapes", args.shapes,
        "--specializations", args.specializations,
        "--samples", str(args.samples),
        "--target-elements", str(args.target_elements),
        "--min-iterations", str(args.min_iterations),
        "--max-iterations", str(args.max_iterations),
        "--affine-stride", str(args.affine_stride),
        "--scalar-kind", args.scalar_kind,
        "--scalar-pattern", args.scalar_pattern,
        "--prepared-operation", args.prepared_operation,
        "--contiguous-padding", args.contiguous_padding,
    ]


def _run(args, mode):
    env = os.environ.copy()
    env.update({"PYTHON_JIT": "1", "PYTHON_JIT_STRESS": "1", "OPENBLAS_NUM_THREADS": "1"})
    for key in list(env):
        if key.startswith("PYTHON_JIT_") and key not in ("PYTHON_JIT_STRESS", "PYTHON_JIT_CROSSJIT"):
            env.pop(key)
    for name in (
        "NPY_CROSSJIT_DIAGNOSTICS", "NPY_CROSSJIT_TELEMETRY_TIMING",
        "NPY_CROSSJIT_DIAGNOSTIC_DISABLE_GENERATED",
        "NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK",
        "NPY_CROSSJIT_PROFILED_FORCE_GENERATED",
    ):
        env.pop(name, None)
    if mode == "baseline":
        env["PYTHON_JIT_CROSSJIT"] = "0"
    else:
        env["PYTHON_JIT_CROSSJIT"] = "1"
        env["NPY_CROSSJIT_MIN_WORK"] = "0"
        env["NPY_CROSSJIT_PROFILED_FORCE_GENERATED"] = "1"
        if mode == "generic":
            env["NPY_CROSSJIT_DIAGNOSTIC_DISABLE_GENERATED"] = "1"
    result = subprocess.run(
        _command(args, mode), capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError(f"{mode} worker failed:\n{result.stderr}")
    return json.loads(result.stdout)


def _measure(args):
    schedule = [(process, mode) for process in range(args.processes)
                for mode in ("baseline", "generated", "generic")]
    random.Random(args.seed).shuffle(schedule)
    raw = []
    for index, (process, mode) in enumerate(schedule, 1):
        print(f"[{index}/{len(schedule)}] process={process} mode={mode}",
              flush=True)
        for record in _run(args, mode):
            record["independent_process_index"] = process
            raw.append(record)
    groups = defaultdict(lambda: defaultdict(list))
    for record in raw:
        key = (record["execution_specialization_hash"],
               record["outer_count"], record["inner_extent"])
        groups[key][record["mode"]].append(record)
    rows = []
    for key, modes in sorted(groups.items()):
        specialization, outer, inner = key
        values = {}
        cvs = {}
        process_values = {}
        for mode in ("baseline", "generated", "generic"):
            samples = [item["process_median_ns"] for item in modes[mode]]
            if len(samples) != args.processes:
                raise RuntimeError("incomplete independent-process result")
            values[mode] = statistics.median(samples)
            cvs[mode] = _cv(samples)
            process_values[mode] = samples
        sample = modes["generated"][0]
        rows.append({
            "execution_specialization_hash": specialization,
            "structural_signature_hash": sample["structural_signature_hash"],
            "profiled_source": sample["profiled_source"],
            "outer_count": outer,
            "inner_extent": inner,
            "element_count": outer * inner,
            "baseline_median_ns": values["baseline"],
            "generated_median_ns": values["generated"],
            "generic_median_ns": values["generic"],
            "generated_speedup": values["baseline"] / values["generated"],
            "generic_speedup": values["baseline"] / values["generic"],
            "generated_vs_generic": values["generic"] / values["generated"],
            "baseline_cv": cvs["baseline"],
            "generated_cv": cvs["generated"],
            "generic_cv": cvs["generic"],
            "process_medians_ns": process_values,
            "execution_validation": {mode: [{
                "independent_process_index": item["independent_process_index"],
                "expected_region_nodes": item["expected_region_nodes"],
                "observed_region_nodes": item["observed_region_nodes"],
                "expected_executions": item["expected_executions"],
                "timed_iterations": item["timed_iterations"],
                "entry_ordinary_iterations_per_call": item["entry_ordinary_iterations_per_call"],
                "observed_scalar_class_mask": item["observed_scalar_class_mask"],
                "prepared_scalar_values": item["prepared_scalar_values"],
                "prepared_operation": item["prepared_operation"],
                "execution_evidence": item["execution_evidence"],
            } for item in modes[mode]] for mode in ("baseline", "generated", "generic")},
            "correct": True,
        })
    return rows


def _build_database(args, metadata, rows):
    sys.path.insert(0, str(args.generator_root))
    import crossjit_generator
    from crossjit_generator.candidate import ExecutionSpecialization
    from crossjit_generator.backend.c import CBackend
    from crossjit_generator.decision import (
        ArtifactDecision, DecisionEnvironment, ExecutionDecisionDB,
        ExecutionDecisionRecord, MeasurementPoint, MeasurementProvenance,
        ProfitabilityEnvelope, RuntimeDecision,
    )
    from crossjit_generator.domains.elementwise import ElementwiseOptimizer
    from crossjit_generator.signature import OptimizationSignature

    environment_data = json.loads(args.decision_environment.read_text())
    kernels = {
        item["execution_specialization_hash"]: item
        for item in _cases(metadata, set(args.layouts.split(",")))
        if not args.specializations or item["execution_specialization_hash"]
        in set(args.specializations.split(","))
    }
    records = []
    for specialization_hash, kernel in sorted(kernels.items()):
        signature = OptimizationSignature.from_data(
            kernel["optimization_signature"])
        specialization = ExecutionSpecialization.from_data(
            kernel["execution_specialization"])
        environment = DecisionEnvironment(
            provider=signature.provider,
            provider_abi_version=signature.provider_abi_version,
            recipe_hashes=tuple(
                operation.recipe_hash for operation in signature.operations),
            generator_version=crossjit_generator.__version__,
            optimizer_version=ElementwiseOptimizer.VERSION,
            backend_version=(CBackend.AFFINE_VERSION
                             if kernel["entry_abi"] == "affine_v2"
                             else CBackend.VERSION),
            target_triple=environment_data["target_triple"],
            target_features=tuple(environment_data["target_features"]),
            compiler_id=environment_data["compiler_id"],
            compiler_version=environment_data["compiler_version"],
            compiler_flags=tuple(environment_data["compiler_flags"]),
            artifact_format_version=environment_data[
                "artifact_format_version"],
        )
        case_rows = [item for item in rows
                     if item["execution_specialization_hash"] ==
                     specialization_hash]
        points = []
        selected = False
        for row in sorted(case_rows, key=lambda item: item["element_count"]):
            noise = max(row["baseline_cv"], row["generated_cv"],
                        row["generic_cv"])
            profitable = (
                noise <= args.max_cv and
                row["generated_speedup"] >= args.speedup_threshold and
                row["generated_median_ns"] < row["generic_median_ns"])
            selected |= profitable
            points.append(MeasurementPoint(
                row["element_count"], row["element_count"],
                row["baseline_median_ns"], row["generated_median_ns"],
                row["generic_median_ns"], row["generated_speedup"],
                row["generic_speedup"], noise,
                max(0.0, min(1.0, 1.0 - noise / args.max_cv)),
                (RuntimeDecision.GENERATED if profitable else
                 RuntimeDecision.ORIGINAL),
                ("stable descriptor-specific generated speedup" if profitable
                 else "unprofitable or noisy descriptor specialization"),
            ))
        records.append(ExecutionDecisionRecord(
            signature, specialization, environment,
            (ArtifactDecision.AUTO_SELECTED if selected else
             ArtifactDecision.AUTO_REJECTED),
            MeasurementProvenance(
                "generic_signature_microbenchmark", args.processes,
                args.samples, args.measured_at_utc),
            ProfitabilityEnvelope(tuple(points)),
            kernel["candidate_hash"],
            min((point.confidence for point in points), default=0.0),
            ("at least one measured-profitable specialization cell" if selected
             else "no measured-profitable specialization cell"),
        ))
    profile_hash = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    universe = hashlib.sha256("\n".join(sorted(kernels)).encode()).hexdigest()
    return ExecutionDecisionDB(
        records, profile_identity_hash=profile_hash,
        candidate_universe_hash=universe)


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("baseline", "generated", "generic"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--generator-root", type=Path,
                        default=root.parent / "crossjit-generator")
    parser.add_argument("--generated-metadata", type=Path, required=True)
    parser.add_argument("--decision-environment", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--layouts", default="INNER_CONTIGUOUS")
    parser.add_argument("--specializations", default="")
    parser.add_argument("--shapes", default=",".join(
        f"{outer}x{inner}" for outer, inner in DEFAULT_SHAPES))
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-elements", type=int, default=2_000_000)
    parser.add_argument("--min-iterations", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--affine-stride", type=int, default=2)
    parser.add_argument("--scalar-kind", choices=("legacy", "python", "numpy", "integer"),
                        default="legacy")
    parser.add_argument("--scalar-pattern", default="",
                        help="comma-separated representation for each scalar live-in")
    parser.add_argument("--contiguous-padding", default="", help="comma-separated row padding for each unit-stride array")
    parser.add_argument("--prepared-operation", choices=("add", "sub", "mul"), default="mul")
    parser.add_argument("--speedup-threshold", type=float, default=1.05)
    parser.add_argument("--max-cv", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2909)
    parser.add_argument("--measured-at-utc", default="2026-09-29T14:00:00Z")
    args = parser.parse_args()
    if args.worker:
        # Startup policy also applies when a worker is launched directly.
        env = dict(os.environ)
        for key in list(env):
            if key.startswith("PYTHON_JIT_") and not key.startswith("PYTHON_JIT_CROSSJIT"):
                env.pop(key)
        env.update(PYTHON_JIT="1", PYTHON_JIT_STRESS="1")
        if env != dict(os.environ):
            os.execve(sys.executable, [sys.executable, *sys.argv], env)
        if args.mode is None:
            parser.error("--worker requires --mode")
        return _worker(args)
    for name in ("decision_environment", "profile", "output_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    metadata = json.loads(args.generated_metadata.read_text())
    rows = _measure(args)
    database = _build_database(args, metadata, rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "schema_version": 1,
        "method": "generic_signature_microbenchmark",
        "configuration": {
            "independent_processes": args.processes,
            "samples_per_process": args.samples,
            "speedup_threshold": args.speedup_threshold,
            "max_cv": args.max_cv,
            "benchmark_identity_used_for_selection": False,
            "formal_npbench_evaluation": False,
            "run_mode": "development", "python_jit_stress": True,
            "scalar_kind": args.scalar_kind,
            "scalar_pattern": args.scalar_pattern,
            "prepared_operation": args.prepared_operation,
            "contiguous_padding": args.contiguous_padding,
        },
        "rows": rows,
    }
    (args.output_dir / "execution_specialization_tuning.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "execution_decision_db.json").write_text(
        database.serialize())
    print(json.dumps({
        "cells": len(rows),
        "selected_specializations": sum(
            record.artifact_decision.value == "AUTO_SELECTED"
            for record in database.records()),
        "decision_db": str(
            args.output_dir / "execution_decision_db.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
