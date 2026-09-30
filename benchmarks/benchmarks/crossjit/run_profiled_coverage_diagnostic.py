#!/usr/bin/env python3
"""Run and summarize the non-timing GEN-PROFILED coverage diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


LAYOUTS = {
    1: "FULL_CONTIGUOUS",
    2: "INNER_CONTIGUOUS",
    3: "ZERO_STRIDE_BROADCAST",
    4: "SCALAR_OPERAND",
    5: "FIXED_AFFINE_STRIDE",
    6: "TRULY_GENERAL_STRIDED",
}
ACCESS_KINDS = {
    1: "CONTIGUOUS",
    2: "SCALAR",
    3: "ZERO_STRIDE",
    4: "FIXED_STRIDE",
    5: "GENERAL_STRIDED",
}
DTYPES = {0: "other", 1: "float32", 2: "float64", 3: "int32", 4: "int64"}
OPERATIONS = {
    0: "numpy.add",
    1: "numpy.subtract",
    2: "numpy.multiply",
    3: "numpy.true_divide",
    4: "numpy.negative",
    5: "numpy.minimum",
    6: "numpy.maximum",
}
SEMANTIC_OPERATIONS = {
    1: "numpy.add",
    2: "numpy.subtract",
    3: "numpy.multiply",
    4: "numpy.true_divide",
    5: "numpy.power",
    6: "numpy.negative",
    7: "numpy.minimum",
    8: "numpy.maximum",
    9: "numpy.less",
    10: "numpy.less_equal",
    11: "numpy.logical_and",
    12: "numpy.matmul",
    13: "numpy.add.outer",
    14: "numpy.mean",
    15: "numpy.maximum.reduce",
    16: "numpy.boolean_subscript",
}
ELEMENT_BOUNDS = (16, 32, 64, 128, 256, 512, 1024, 2048,
                  4096, 8192, 16384, 32768, 65536, None)


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(value):
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _worker(args):
    sys.path.insert(0, str(args.npbench_src))
    from npbench.infrastructure import Benchmark, Test, generate_framework
    import _testinternalcapi as internal
    from numpy._core import _multiarray_umath as umath

    internal.crossjit_reset_stats()
    umath._crossjit_reset_stats()
    benchmark = Benchmark(args.workload)
    framework = generate_framework("numpy")
    Test(benchmark, framework, framework).run(
        args.preset, True, args.repeat, args.timeout, ignore_errors=False)
    result = {
        "schema_version": 1,
        "workload_id": args.workload,
        "host_counters": internal.crossjit_stats(),
        "provider_counters": umath._crossjit_stats(),
    }
    args.worker_output.write_text(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _command(args, workload, output):
    return [
        args.python, str(Path(__file__).resolve()), "--worker",
        "--npbench-src", str(args.npbench_src),
        "--workload", workload,
        "--worker-output", str(output),
        "--preset", args.preset,
        "--repeat", str(args.repeat),
        "--timeout", str(args.timeout),
    ]


def _canonical_signature(record, recipes):
    operations = []
    for node in record["nodes"]:
        operation_id = OPERATIONS[node["op"]]
        arity = 1 if operation_id == "numpy.negative" else 2
        operands = [node["left"]] if arity == 1 else [
            node["left"], node["right"]]
        operations.append({
            "operation_id": operation_id,
            "operands": operands,
            "recipe_hash": recipes[operation_id],
        })
    signature = {
        "schema_version": 1,
        "provider": "numpy",
        "provider_abi_version": 3,
        "domain": "elementwise",
        "input_count": record["input_count"],
        "operations": operations,
        "dtype_class": DTYPES[record["dtype_class"]],
        "layout_class": {
            1: "contiguous_same_shape",
            2: "contiguous_broadcast",
            3: "zero_stride",
            4: "inner_contiguous",
            5: "general_strided",
        }[record["layout_class"]],
        "error_mode": "numpy_fpe_fallback",
        "materialization_class": "virtual_intermediates_single_output",
        "scalar_mask": record["scalar_mask"],
    }
    return signature, _sha256(signature)


def _merge_profile(groups, workload, record, recipes, unsupported):
    signature, signature_hash = _canonical_signature(record, recipes)
    access = [ACCESS_KINDS[item] for item in record["access_kinds"]]
    variant_descriptor = {
        "canonical_signature_hash": signature_hash,
        "refined_layout_class": LAYOUTS[record["refined_layout_class"]],
        "iteration_rank_class": record["iteration_rank_class"],
        "operand_access_classes": access,
    }
    variant_hash = _sha256(variant_descriptor)
    item = groups.get(variant_hash)
    if item is None:
        reason = unsupported.get(signature_hash)
        item = {
            "canonical_signature_hash": signature_hash,
            "profile_variant_hash": variant_hash,
            "signature": signature,
            "operation_sequence": [
                operation["operation_id"] for operation in signature["operations"]],
            "dag_topology": [
                {"operation": operation["operation_id"],
                 "operands": operation["operands"]}
                for operation in signature["operations"]],
            "dtype_class": signature["dtype_class"],
            "legacy_layout_class": signature["layout_class"],
            "layout_class": variant_descriptor["refined_layout_class"],
            "operand_access_classes": access,
            "iteration_rank_class": record["iteration_rank_class"],
            "outer_count_samples": [],
            "inner_extent_samples": [],
            "region_length": len(signature["operations"]),
            "scalar_mask": signature["scalar_mask"],
            "dynamic_prepare_count": 0,
            "dynamic_execution_opportunity_count": 0,
            "generated_execution_count": 0,
            "generic_execution_count": 0,
            "prepare_element_count_distribution": [0] * len(ELEMENT_BOUNDS),
            "execution_element_count_distribution": [0] * len(ELEMENT_BOUNDS),
            "optimizer_status": "REJECTED" if reason else "SUPPORTED",
            "optimizer_rejection_reason": reason,
            "workloads": [],
        }
        groups[variant_hash] = item
    item["dynamic_prepare_count"] += record["prepare_count"]
    item["dynamic_execution_opportunity_count"] += record["execution_count"]
    item["generated_execution_count"] += record["generated_hit_count"]
    item["generic_execution_count"] += record["generic_fallback_count"]
    for target, source in (
            (item["prepare_element_count_distribution"],
             record["prepare_element_buckets"]),
            (item["execution_element_count_distribution"],
             record["element_buckets"])):
        for index, value in enumerate(source):
            target[index] += value
    item["outer_count_samples"].append(record["outer_count"])
    item["inner_extent_samples"].append(record["inner_extent"])
    item["workloads"].append(workload)


def _unsupported_atomic_fit(operation, inplace):
    if inplace:
        return False
    if operation == "numpy.power":
        return True
    return False


def _summarize(args, workloads):
    metadata = json.loads(args.generated_metadata.read_text())
    unsupported = {
        item["signature_hash"]: item["reason"]
        for item in metadata["unsupported_observed_signatures"]
    }
    recipes = {
        operation["operation_id"]: operation["recipe_hash"]
        for kernel in metadata["kernels"]
        for operation in kernel["optimization_signature"]["operations"]
    }
    groups = {}
    unsupported_groups = {}
    totals = defaultdict(int)
    records = []
    for workload in workloads:
        raw = json.loads((args.output_dir / "raw" / f"{workload}.json").read_text())
        provider = raw["provider_counters"]
        host = raw["host_counters"]
        for name, value in provider.items():
            if isinstance(value, int) and name not in {
                    "diagnostics_enabled", "timing_enabled"}:
                totals[name] += value
        for record in provider["observed_signatures"]:
            _merge_profile(groups, workload, record, recipes, unsupported)
        for record in provider["unsupported_nodes"]:
            operation = SEMANTIC_OPERATIONS.get(
                record["semantic_id"], f"semantic_id:{record['semantic_id']}")
            key = _canonical_json({
                "operation": operation,
                "inplace": bool(record["inplace"]),
                "node_index": record["node_index"],
                "region_length": record["region_length"],
                "dtype_class": record["dtype_class"],
                "layout_class": record["refined_layout_class"],
                "scalar_mask": record["scalar_mask"],
            })
            item = unsupported_groups.setdefault(key, {
                "operation_identity": operation,
                "inplace": bool(record["inplace"]),
                "occurrence_count": 0,
                "workloads": [],
                "dtype_class": DTYPES[record["dtype_class"]],
                "layout_class": LAYOUTS[record["refined_layout_class"]],
                "scalar_mask": record["scalar_mask"],
                "region_length": record["region_length"],
                "node_index": record["node_index"],
                "min_elements": record["min_elements"],
                "max_elements": record["max_elements"],
                "is_elementwise": operation.startswith("numpy.") and operation not in {
                    "numpy.matmul", "numpy.add.outer", "numpy.mean",
                    "numpy.maximum.reduce", "numpy.boolean_subscript"},
                "fits_existing_atomic_recipe_model":
                    _unsupported_atomic_fit(operation, bool(record["inplace"])),
            })
            item["occurrence_count"] += record["occurrence_count"]
            item["min_elements"] = min(item["min_elements"], record["min_elements"])
            item["max_elements"] = max(item["max_elements"], record["max_elements"])
            item["workloads"].append(workload)
        records.append({
            "workload": workload,
            "correct": True,
            "recognized_nodes": host.get("recognized_nodes", 0),
            "prepared_regions": provider["regions_prepare_called"],
            "accepted_regions": provider["regions_provider_accepted"],
            "execution_hits": provider["exec_hit"],
            "unsupported_node_rejections":
                provider["regions_rejected_unsupported_node"],
        })

    for item in groups.values():
        item["workloads"] = sorted(set(item["workloads"]))
        item["outer_count_samples"] = sorted(set(item["outer_count_samples"]))
        item["inner_extent_samples"] = sorted(set(item["inner_extent_samples"]))
        item["prepare_element_count_distribution"] = [
            {"upper_bound": bound, "count": count}
            for bound, count in zip(
                ELEMENT_BOUNDS, item["prepare_element_count_distribution"])]
        item["execution_element_count_distribution"] = [
            {"upper_bound": bound, "count": count}
            for bound, count in zip(
                ELEMENT_BOUNDS, item["execution_element_count_distribution"])]
    for item in unsupported_groups.values():
        item["workloads"] = sorted(set(item["workloads"]))

    layouts = []
    for layout in LAYOUTS.values():
        members = [item for item in groups.values() if item["layout_class"] == layout]
        workloads_for_layout = sorted({
            workload for item in members for workload in item["workloads"]})
        layouts.append({
            "layout_class": layout,
            "distinct_signature_count": len(members),
            "dynamic_prepare_count": sum(
                item["dynamic_prepare_count"] for item in members),
            "dynamic_execution_opportunity_count": sum(
                item["dynamic_execution_opportunity_count"] for item in members),
            "workload_count": len(workloads_for_layout),
            "workloads": workloads_for_layout,
            "estimated_recoverable_dynamic_coverage": sum(
                item["dynamic_execution_opportunity_count"] for item in members),
        })
    signatures = sorted(groups.values(), key=lambda item: item["profile_variant_hash"])
    total_dynamic = sum(
        item["dynamic_execution_opportunity_count"] for item in signatures)
    supported_dynamic = sum(
        item["dynamic_execution_opportunity_count"] for item in signatures
        if item["optimizer_status"] == "SUPPORTED")
    supported_signatures = sum(
        item["optimizer_status"] == "SUPPORTED" for item in signatures)
    return {
        "schema_version": 1,
        "configuration": {
            "preset": args.preset,
            "repeat": args.repeat,
            "python": args.python,
            "profile_guided_analysis_only": True,
            "benchmark_identity_in_canonical_signature": False,
            "execution_mode": args.execution_mode,
            "forced_generic_profile_execution": (
                args.execution_mode == "opportunity"),
            "excluded_workloads": sorted(args.exclude_workload),
            "formal_timing": False,
        },
        "workload_count": len(workloads),
        "correct_count": len(workloads),
        "provider_totals": dict(sorted(totals.items())),
        "coverage": {
            "observed_profile_variant_count": len(signatures),
            "optimizer_supported_variant_count": supported_signatures,
            "signature_coverage": (supported_signatures / len(signatures)
                                   if signatures else 0.0),
            "total_dynamic_execution_opportunities": total_dynamic,
            "optimizer_supported_dynamic_opportunities": supported_dynamic,
            "dynamic_coverage": (supported_dynamic / total_dynamic
                                 if total_dynamic else 0.0),
        },
        "layout_summary": layouts,
        "signatures": signatures,
        "unsupported_nodes": sorted(
            unsupported_groups.values(),
            key=lambda item: (item["operation_identity"], item["layout_class"])),
        "records": records,
    }


def _controller(args):
    suite = json.loads(args.suite.read_text())
    excluded = set(args.exclude_workload)
    workloads = [
        item["name"] for item in suite["experiments"]
        if item["name"] not in excluded
    ]
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summary = _summarize(args, workloads)
        args.output_dir.joinpath("coverage_report.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n")
        print(json.dumps(summary["coverage"], sort_keys=True, indent=2))
        return 0
    env = os.environ.copy()
    env.update({
        "PYTHON_JIT": "1",
        "PYTHON_JIT_CROSSJIT": "1",
        "NPY_CROSSJIT_DIAGNOSTICS": "1",
        "OPENBLAS_NUM_THREADS": str(args.openblas_threads),
    })
    if args.execution_mode == "opportunity":
        env.update({
            "NPY_CROSSJIT_MIN_WORK": "0",
            "NPY_CROSSJIT_DIAGNOSTIC_DISABLE_GENERATED": "1",
            "NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK": "1",
        })
    else:
        for name in (
                "NPY_CROSSJIT_MIN_WORK",
                "NPY_CROSSJIT_DIAGNOSTIC_DISABLE_GENERATED",
                "NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK"):
            env.pop(name, None)
    env.pop("NPY_CROSSJIT_TELEMETRY_TIMING", None)
    started = time.monotonic()
    for index, workload in enumerate(workloads, 1):
        output = (raw_dir / f"{workload}.json").resolve()
        print(f"[{index}/{len(workloads)}] {workload}", flush=True)
        with tempfile.TemporaryDirectory(prefix="crossjit-profile-") as temp_dir:
            subprocess.run(
                _command(args, workload, output), check=True, env=env,
                cwd=temp_dir, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True)
        checkpoint = {
            "schema_version": 1,
            "completed_workloads": workloads[:index],
            "elapsed_seconds": time.monotonic() - started,
        }
        (args.output_dir / "checkpoint.json").write_text(
            json.dumps(checkpoint, sort_keys=True, indent=2) + "\n")
    summary = _summarize(args, workloads)
    summary["elapsed_seconds"] = time.monotonic() - started
    args.output_dir.joinpath("coverage_report.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps(summary["coverage"], sort_keys=True, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--npbench-src", type=Path, required=True)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--generated-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workload")
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--preset", default="S")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--openblas-threads", type=int, default=6)
    parser.add_argument(
        "--execution-mode", choices=("opportunity", "production"),
        default="opportunity",
        help=("force the generic executor to count the complete opportunity "
              "universe, or exercise the frozen production policy"))
    parser.add_argument("--exclude-workload", action="append", default=[])
    args = parser.parse_args()
    if args.worker:
        return _worker(args)
    required = (args.suite, args.generated_metadata, args.output_dir)
    if any(value is None for value in required):
        parser.error("controller requires --suite, --generated-metadata, and --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
