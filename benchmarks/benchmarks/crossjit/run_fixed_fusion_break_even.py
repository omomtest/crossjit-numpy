#!/usr/bin/env python3
"""Paired-process CrossJIT fixed-fusion break-even calibration.

The controller launches independent baseline and CrossJIT workers in
randomized pairs.  Each worker runs the same expressions, inputs, correctness
checks, Python JIT, and NumPy build.  The only execution-path difference is
``PYTHON_JIT_CROSSJIT``; CrossJIT workers explicitly force admission with
``NPY_CROSSJIT_MIN_WORK=0`` and every recorded row must prove ``exec_hit``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


DEFAULT_N = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
             16384, 32768, 65536)


@dataclass(frozen=True)
class Case:
    identifier: str
    family: str
    region_length: int
    expression: str
    argument_kinds: tuple[str, ...]


CASES = (
    Case("linear_2", "linear_chain", 2, "(a + b) * c",
         ("array", "array", "array")),
    Case("linear_3", "linear_chain", 3, "(a + b) * c - d",
         ("array", "array", "array", "array")),
    Case("linear_4", "linear_chain", 4, "((a + b) * c - d) / e",
         ("array", "array", "array", "array", "array")),
    Case("linear_5", "linear_chain", 5, "((a + b) * c - d) / e + f",
         ("array", "array", "array", "array", "array", "array")),
    Case("linear_6", "linear_chain", 6, "(((a + b) * c - d) / e + f) * g",
         ("array", "array", "array", "array", "array", "array", "array")),
    Case("branched_3", "branched_dag", 3, "a * b + c * d",
         ("array", "array", "array", "array")),
    Case("array_scalar_2", "array_scalar", 2, "(a + scalar) * b",
         ("array", "scalar", "array")),
    Case("scalar_array_2", "scalar_array", 2, "(scalar * a) + b",
         ("scalar", "array", "array")),
)
CASE_BY_ID = {case.identifier: case for case in CASES}


def _cv(values):
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def _parse_ints(text):
    return tuple(int(value) for value in text.split(",") if value)


def _array_for_layout(n, dtype, layout, offset):
    if layout == "CONTIGUOUS_SAME_SHAPE":
        return (1 + np.arange(n, dtype=dtype) / max(n, 1) + offset)
    if layout == "CONTIGUOUS_BROADCAST":
        operand_index = round(offset / 0.125)
        size = n if operand_index == 0 else 1
        return (1 + np.arange(size, dtype=dtype) / max(n, 1) + offset
                ).reshape(1, size)
    if layout == "ZERO_STRIDE":
        base = np.asarray([1.0 + offset], dtype=dtype)
        return np.broadcast_to(base, (n,))
    if layout == "INNER_CONTIGUOUS":
        if n % 2:
            raise ValueError("INNER_CONTIGUOUS fixture requires even N")
        columns = n // 2
        base = (1 + np.arange(4 * columns, dtype=dtype) / max(n, 1)
                + offset).reshape(4, columns)
        return base[::2, :]
    if layout == "GENERAL_STRIDED":
        base = 1 + np.arange(n * 2, dtype=dtype) / max(n, 1) + offset
        return base[::2]
    raise ValueError(f"unsupported layout fixture: {layout}")


def _inputs(case, n, dtype, layout):
    values = []
    array_index = 0
    for kind in case.argument_kinds:
        if kind == "scalar":
            values.append(0.25)
        else:
            values.append(_array_for_layout(
                n, dtype, layout, array_index * 0.125))
            array_index += 1
    return tuple(values)


def _compiled_functions(case, serial):
    names = tuple(chr(ord("a") + index)
                  if kind == "array" else "scalar"
                  for index, kind in enumerate(case.argument_kinds))
    # Scalar cases have the scalar in a fixed semantic position, while names
    # for arrays remain stable and readable in emitted diagnostic source.
    if case.identifier == "array_scalar_2":
        names = ("a", "scalar", "b")
    elif case.identifier == "scalar_array_2":
        names = ("scalar", "a", "b")
    arguments = ", ".join((*names, "count"))
    source = (
        f"def expression_{serial}({arguments}):\n"
        "    for _ in range(count):\n"
        f"        result = {case.expression}\n"
        "    return result\n"
        f"def control_{serial}({arguments}):\n"
        "    for _ in range(count):\n"
        f"        result = {next(name for name in names if name != 'scalar')}\n"
        "    return result\n"
    )
    namespace = {}
    exec(compile(source, f"<crossjit-{case.identifier}-{serial}>", "exec"),
         namespace)
    return (namespace[f"expression_{serial}"],
            namespace[f"control_{serial}"], names)


def _expected(case, names, values):
    namespace = dict(zip(names, values, strict=True))
    return eval(case.expression, {"__builtins__": {}}, namespace)


def _counter_delta(after, before, name):
    return after[name] - before[name]


def _measure_cell(case, n, args, serial):
    dtype = np.dtype(args.dtype)
    values = _inputs(case, n, dtype, args.layout)
    expression, control, names = _compiled_functions(case, serial)
    expected = _expected(case, names, values)

    internal.crossjit_reset_stats()
    umath._crossjit_reset_stats()
    warm_count = internal.TIER2_THRESHOLD * 2
    warm_started = time.perf_counter_ns()
    actual = expression(*values, warm_count)
    warmup_ns = time.perf_counter_ns() - warm_started
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    control(*values, warm_count)

    iterations = max(
        args.min_iterations,
        min(args.max_iterations, args.target_elements // max(n, 1)),
    )
    before = umath._crossjit_stats()
    expression_samples = []
    control_samples = []
    for sample in range(args.samples):
        first, second = ((expression, expression_samples),
                         (control, control_samples))
        if sample % 2:
            first, second = second, first
        for function, samples in (first, second):
            started = time.perf_counter_ns()
            timed_result = function(*values, iterations)
            samples.append((time.perf_counter_ns() - started) / iterations)
            if function is expression:
                actual = timed_result
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    after = umath._crossjit_stats()
    host = internal.crossjit_stats()
    exec_hit = _counter_delta(after, before, "exec_hit")
    executed = _counter_delta(after, before, "regions_executed")
    valid_execution = (
        args.mode == "baseline" or
        (exec_hit > 0 and executed > 0 and
         after["regions_provider_accepted"] > 0 and
         host["published_regions"] > 0)
    )
    return {
        "case_identifier": case.identifier,
        "operation_pattern": case.family,
        "dtype": dtype.name,
        "layout": args.layout,
        "region_length": case.region_length,
        "element_count": n,
        "eliminated_materializations": case.region_length - 1,
        "mode": args.mode,
        "forced_admission": args.mode == "crossjit",
        "iterations_per_sample": iterations,
        "sample_count": args.samples,
        "samples_ns": expression_samples,
        "control_samples_ns": control_samples,
        "process_median_ns": statistics.median(expression_samples),
        "process_cv": _cv(expression_samples),
        "control_median_ns": statistics.median(control_samples),
        "warmup_ns": warmup_ns,
        "correct": True,
        "valid_execution": valid_execution,
        "diagnostics_enabled": after["diagnostics_enabled"],
        "execution_evidence": {
            "exec_hit": exec_hit,
            "exec_miss": _counter_delta(after, before, "exec_miss"),
            "exec_error": _counter_delta(after, before, "exec_error"),
            "regions_executed": executed,
            "provider_accepted": after["regions_provider_accepted"],
            "published": host["published_regions"],
        },
        "cost_counters": {
            name: after[name]
            for name in (
                "prepare_time_ns", "guard_lower_time_ns",
                "execution_lower_time_ns", "guard_time_ns",
                "region_time_ns", "allocation_time_ns", "kernel_time_ns",
                "guard_hit", "guard_miss", "regions_prepare_called",
                "guard_plans_lowered", "execution_plans_lowered",
            )
        },
    }


def _worker(args):
    global np, internal, umath
    import numpy as np
    import _testinternalcapi as internal
    from numpy._core import _multiarray_umath as umath

    selected = CASES if args.cases == "all" else tuple(
        CASE_BY_ID[name] for name in args.cases.split(","))
    cost_cells = None
    if args.cost_cells:
        cost_cells = {
            (case_id, int(n))
            for case_id, n in (cell.split(":", 1)
                               for cell in args.cost_cells.split(","))
        }
    records = []
    serial = 0
    for case in selected:
        for n in _parse_ints(args.n_grid):
            if cost_cells is not None and (case.identifier, n) not in cost_cells:
                continue
            serial += 1
            records.append(_measure_cell(case, n, args, serial))
    print(json.dumps(records, separators=(",", ":")))
    return 0


def _run_worker(args, mode, detailed, cost_cells=None):
    command = [
        args.python, str(Path(__file__).resolve()), "--worker",
        "--mode", mode, "--n-grid", args.n_grid,
        "--cases", args.cases, "--dtype", args.dtype,
        "--layout", args.layout, "--samples", str(args.samples),
        "--target-elements", str(args.target_elements),
        "--min-iterations", str(args.min_iterations),
        "--max-iterations", str(args.max_iterations),
        "--cost-cells", cost_cells or "",
    ]
    env = os.environ.copy()
    env.update({
        "PYTHON_JIT": "1",
        "PYTHON_JIT_CROSSJIT": "1" if mode == "crossjit" else "0",
        "OPENBLAS_NUM_THREADS": str(args.openblas_threads),
    })
    env.pop("PYTHON_JIT_CROSSJIT_DEBUG", None)
    env.pop("NPY_CROSSJIT_DIAGNOSTICS", None)
    env.pop("NPY_CROSSJIT_TELEMETRY_TIMING", None)
    env.pop("NPY_CROSSJIT_TEST_EXEC_ERROR", None)
    if mode == "crossjit":
        env["NPY_CROSSJIT_MIN_WORK"] = "0"
    else:
        env.pop("NPY_CROSSJIT_MIN_WORK", None)
    if detailed:
        env["NPY_CROSSJIT_DIAGNOSTICS"] = "1"
        env["NPY_CROSSJIT_TELEMETRY_TIMING"] = "1"
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, env=env)
    return json.loads(completed.stdout)


def _aggregate_cost(records):
    result = []
    groups = defaultdict(list)
    for record in records:
        groups[(record["case_identifier"], record["element_count"])].append(record)
    for (case_id, n), rows in sorted(groups.items()):
        components = defaultdict(list)
        for row in rows:
            counters = row["cost_counters"]
            evidence = row["execution_evidence"]
            executed = max(1, evidence["regions_executed"])
            guard_calls = max(1, counters["guard_hit"] + counters["guard_miss"])
            guard = counters["guard_time_ns"] / guard_calls
            region = counters["region_time_ns"] / executed
            allocation = counters["allocation_time_ns"] / executed
            kernel = counters["kernel_time_ns"] / executed
            setup = max(0.0, region - guard - allocation - kernel)
            residual = max(
                0.0,
                row["process_median_ns"] - row["control_median_ns"]
                - guard - region,
            )
            components["provider_guard_ns"].append(guard)
            components["output_allocation_ns"].append(allocation)
            components["fixed_fused_loop_ns"].append(kernel)
            components["metadata_setup_ns"].append(setup)
            components["host_ext_dispatch_residual_ns"].append(residual)
            for total_name, count_name, output_name in (
                ("prepare_time_ns", "regions_prepare_called",
                 "plan_prepare_ns"),
                ("guard_lower_time_ns", "guard_plans_lowered",
                 "guard_lower_ns"),
                ("execution_lower_time_ns", "execution_plans_lowered",
                 "execution_lower_ns"),
            ):
                count = max(1, counters[count_name])
                components[output_name].append(counters[total_name] / count)
        result.append({
            "case_identifier": case_id,
            "element_count": n,
            "independent_process_count": len(rows),
            **{name: statistics.median(values)
               for name, values in components.items()},
            "method_note": (
                "dispatch residual subtracts the Python control loop, one "
                "Provider guard, and Provider region callback from total; "
                "fixed_fused_loop includes per-chunk/node dispatch"
            ),
        })
    return result


def _controller(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    rng = random.Random(args.seed)
    for process_index in range(args.processes):
        pair = ["baseline", "crossjit"]
        rng.shuffle(pair)
        jobs.extend((process_index, mode) for mode in pair)
    raw = []
    for process_index, mode in jobs:
        for record in _run_worker(args, mode, detailed=False):
            record["independent_process_index"] = process_index
            raw.append(record)

    groups = defaultdict(lambda: {"baseline": [], "crossjit": []})
    for record in raw:
        key = (record["case_identifier"], record["element_count"])
        groups[key][record["mode"]].append(record)
    rows = []
    for (case_id, n), modes in sorted(groups.items()):
        baseline = modes["baseline"]
        crossjit = modes["crossjit"]
        baseline_values = [row["process_median_ns"] for row in baseline]
        crossjit_values = [row["process_median_ns"] for row in crossjit]
        baseline_median = statistics.median(baseline_values)
        crossjit_median = statistics.median(crossjit_values)
        valid = (
            len(baseline) == args.processes == len(crossjit) and
            all(row["correct"] for row in baseline + crossjit) and
            all(not row["diagnostics_enabled"]
                for row in baseline + crossjit) and
            all(row["valid_execution"] and
                row["execution_evidence"]["exec_hit"] > 0
                for row in crossjit)
        )
        case = CASE_BY_ID[case_id]
        rows.append({
            "case_identifier": case_id,
            "dtype": args.dtype,
            "layout": args.layout,
            "region_length": case.region_length,
            "element_count": n,
            "operation_pattern": case.family,
            "eliminated_materializations": case.region_length - 1,
            "baseline_median_ns": baseline_median,
            "baseline_cv": _cv(baseline_values),
            "baseline_process_medians_ns": baseline_values,
            "crossjit_median_ns": crossjit_median,
            "crossjit_cv": _cv(crossjit_values),
            "crossjit_process_medians_ns": crossjit_values,
            "speedup": baseline_median / crossjit_median,
            "independent_process_count": len(crossjit),
            "provider_exec_hit_min": min(
                row["execution_evidence"]["exec_hit"] for row in crossjit),
            "provider_regions_executed_min": min(
                row["execution_evidence"]["regions_executed"]
                for row in crossjit),
            "provider_exec_hits": [
                row["execution_evidence"]["exec_hit"] for row in crossjit
            ],
            "provider_regions_executed": [
                row["execution_evidence"]["regions_executed"]
                for row in crossjit
            ],
            "forced_admission": True,
            "valid_crossjit_execution": valid,
        })

    break_even = {}
    for case in CASES:
        case_rows = [row for row in rows
                     if row["case_identifier"] == case.identifier]
        if not case_rows:
            continue
        profitable = [row["element_count"] for row in case_rows
                      if row["valid_crossjit_execution"] and
                      row["speedup"] >= 1.0]
        sustained = None
        for index, row in enumerate(case_rows):
            tail = case_rows[index:]
            if all(item["valid_crossjit_execution"] and
                   item["speedup"] >= 1.0 for item in tail):
                sustained = row["element_count"]
                break
        intervals = []
        interval_start = None
        previous_n = None
        for row in case_rows:
            profitable_here = (row["valid_crossjit_execution"] and
                               row["speedup"] >= 1.0)
            if profitable_here and interval_start is None:
                interval_start = row["element_count"]
            elif not profitable_here and interval_start is not None:
                intervals.append([interval_start, previous_n])
                interval_start = None
            previous_n = row["element_count"]
        if interval_start is not None:
            intervals.append([interval_start, previous_n])
        break_even[case.identifier] = {
            "first_observed_profitable_n": min(profitable) if profitable else None,
            "sustained_profitable_n": sustained,
            "profitable_intervals_on_grid": intervals,
            "regressing_element_counts": [
                row["element_count"] for row in case_rows
                if row["valid_crossjit_execution"] and row["speedup"] < 1.0
            ],
            "measurement_only": True,
        }

    cost_raw = []
    for process_index in range(args.cost_processes):
        for record in _run_worker(
                args, "crossjit", detailed=True,
                cost_cells=args.cost_cells):
            record["independent_process_index"] = process_index
            cost_raw.append(record)
    costs = _aggregate_cost(cost_raw)

    document = {
        "schema_version": 1,
        "configuration": {
            "python": args.python,
            "dtype": args.dtype,
            "layout": args.layout,
            "n_grid": list(_parse_ints(args.n_grid)),
            "cases": args.cases,
            "independent_processes": args.processes,
            "samples_per_process": args.samples,
            "forced_admission": True,
            "baseline_crossjit_mode": "0",
            "crossjit_mode": "1",
            "detailed_diagnostics_in_main_timing": False,
            "production_profitability_changed": False,
        },
        "rows": rows,
        "break_even_observations": break_even,
        "cost_decomposition": costs,
        "evidence_note": (
            "Per-process medians and execution counts are retained in each "
            "row; per-sample raw records are intentionally not persisted."
        ),
    }
    json_path = args.output_dir / f"{args.output_stem}.json"
    csv_path = args.output_dir / f"{args.output_stem}.csv"
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid = sum(row["valid_crossjit_execution"] for row in rows)
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "cells": len(rows),
        "valid_cells": valid,
        "break_even_observations": break_even,
        "cost_decomposition": costs,
    }, indent=2, sort_keys=True))
    return 0 if valid == len(rows) else 2


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("baseline", "crossjit"),
                        default="baseline")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("crossjit-break-even-results"))
    parser.add_argument("--output-stem", default="p1_float64_contiguous")
    parser.add_argument("--n-grid", default=",".join(map(str, DEFAULT_N)))
    parser.add_argument("--cases", default="all")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--layout", default="CONTIGUOUS_SAME_SHAPE")
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--cost-processes", type=int, default=5)
    parser.add_argument(
        "--cost-cells", default="linear_2:256,linear_4:4096,linear_6:65536")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-elements", type=int, default=2_000_000)
    parser.add_argument("--min-iterations", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--openblas-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=310)
    return parser


if __name__ == "__main__":
    parsed = _parser().parse_args()
    raise SystemExit(_worker(parsed) if parsed.worker else _controller(parsed))
