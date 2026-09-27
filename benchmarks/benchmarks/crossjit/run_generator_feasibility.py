#!/usr/bin/env python3
"""Run the G0 benchmark-only static-C feasibility gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import statistics
import subprocess


CASES = ("linear_2", "linear_4", "linear_6")
DEFAULT_N = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
             16384, 32768, 65536)
COMPILER_FLAGS = (
    "-std=c11", "-O3", "-fno-strict-aliasing", "-ffp-contract=off",
    "-msse", "-msse2", "-msse3", "-mssse3", "-msse4.1", "-msse4.2",
    "-mpopcnt", "-msahf", "-mcx16",
)


def _cv(values):
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if len(values) > 1 and mean else 0.0


def _load(path):
    with path.open() as stream:
        return json.load(stream)


def _compile(args):
    args.build_dir.mkdir(parents=True, exist_ok=True)
    executable = args.build_dir / "g0_static_oracle"
    diagnostics = args.build_dir / "g0_vectorization.txt"
    assembly = args.build_dir / "g0_static_oracle.s"
    common = [args.cc, *COMPILER_FLAGS, str(args.source), "-lm"]
    command = [*common, f"-fopt-info-vec-all={diagnostics}", "-o", str(executable)]
    subprocess.run(command, check=True)
    subprocess.run(
        [args.cc, *COMPILER_FLAGS, "-fverbose-asm", "-S", str(args.source),
         "-o", str(assembly)], check=True)
    return executable, diagnostics, assembly, command


def _oracle_rows(args, executable):
    jobs = [(process, case, n) for process in range(args.processes)
            for case in CASES for n in args.n_grid]
    random.Random(args.seed).shuffle(jobs)
    raw = []
    for process, case, n in jobs:
        completed = subprocess.run(
            [str(executable), case, str(n), str(args.samples),
             str(args.target_elements), str(args.max_iterations)],
            check=True, capture_output=True, text=True)
        row = json.loads(completed.stdout)
        row["independent_process_index"] = process
        raw.append(row)
    grouped = {}
    for case in CASES:
        for n in args.n_grid:
            records = [row for row in raw
                       if row["case_identifier"] == case and
                       row["element_count"] == n]
            values = [row["process_median_ns"] for row in records]
            grouped[(case, n)] = {
                "oracle_loop_median_ns": statistics.median(values),
                "oracle_loop_cv": _cv(values),
                "oracle_process_medians_ns": values,
                "oracle_correct": all(row["correct"] for row in records),
                "oracle_fpe_parity": all(row["fpe_parity"] for row in records),
            }
    return grouped


def _cost_map(document):
    return {(row["case_identifier"], row["element_count"]): row
            for row in document.get("cost_decomposition", [])}


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--source", type=Path, default=here / "g0_static_oracle.c")
    parser.add_argument("--p1-results", type=Path,
                        default=here / "results/p1_float64_contiguous.json")
    parser.add_argument("--integrated-static-results", type=Path,
                        default=here / "results/p3_float64_contiguous.json")
    parser.add_argument("--output-dir", type=Path, default=here / "results")
    parser.add_argument("--build-dir", type=Path, default=here / "g0-build")
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-elements", type=int, default=100_000_000)
    parser.add_argument("--max-iterations", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=610)
    parser.add_argument("--n-grid", type=lambda value: tuple(map(int, value.split(","))),
                        default=DEFAULT_N)
    args = parser.parse_args()

    executable, diagnostics, assembly, compile_command = _compile(args)
    oracle = _oracle_rows(args, executable)
    p1 = _load(args.p1_results)
    p1_cost = _cost_map(p1)
    integrated = (_load(args.integrated_static_results)
                  if args.integrated_static_results.exists() else None)
    integrated_rows = ({(row["case_identifier"], row["element_count"]): row
                        for row in integrated["rows"]}
                       if integrated is not None else {})
    integrated_cost = _cost_map(integrated or {})

    rows = []
    for p1_row in p1["rows"]:
        key = (p1_row["case_identifier"], p1_row["element_count"])
        if key not in oracle:
            continue
        row = {
            "case_identifier": key[0],
            "operation_sequence": {
                "linear_2": "add,mul",
                "linear_4": "add,mul,sub,true_divide",
                "linear_6": "add,mul,sub,true_divide,add,mul",
            }[key[0]],
            "region_length": p1_row["region_length"],
            "element_count": key[1],
            "dtype": "float64",
            "layout": "CONTIGUOUS_SAME_SHAPE",
            "p1_baseline_median_ns": p1_row["baseline_median_ns"],
            "p1_baseline_cv": p1_row["baseline_cv"],
            "p1_generic_crossjit_median_ns": p1_row["crossjit_median_ns"],
            "p1_generic_crossjit_cv": p1_row["crossjit_cv"],
            **oracle[key],
        }
        row["p1_generic_full_to_oracle_loop_ratio"] = (
            row["p1_generic_crossjit_median_ns"] / row["oracle_loop_median_ns"])
        cost = p1_cost.get(key)
        if cost is not None:
            row["p1_generic_fixed_loop_ns"] = cost["fixed_fused_loop_ns"]
            row["p1_generic_loop_to_oracle_loop_ratio"] = (
                cost["fixed_fused_loop_ns"] / row["oracle_loop_median_ns"])
        static_row = integrated_rows.get(key)
        if static_row is not None:
            row["integrated_static_crossjit_median_ns"] = static_row["crossjit_median_ns"]
            row["integrated_static_backend_path"] = static_row.get("backend_path")
        static_cost = integrated_cost.get(key)
        if static_cost is not None:
            row["integrated_static_fixed_loop_ns"] = static_cost["fixed_fused_loop_ns"]
        rows.append(row)

    diag_text = diagnostics.read_text()
    vectorized = {case: any(
        "optimized: loop vectorized" in line and function in diag_text[:diag_text.find(line) + len(line)]
        for line in diag_text.splitlines())
        for case, function in (
            ("linear_2", "g0_oracle_linear_2"),
            ("linear_4", "g0_oracle_linear_4"),
            ("linear_6", "g0_oracle_linear_6"),
        )}
    # GCC diagnostics identify source loop lines rather than function names;
    # retain the full file as evidence and derive per-case status from known lines.
    source_lines = {"linear_2": 28, "linear_4": 44, "linear_6": 64}
    vectorized = {
        case: any(f"{args.source.name}:{line}:" in item and
                  "loop vectorized" in item
                  for item in diag_text.splitlines())
        for case, line in source_lines.items()
    }

    representative = {}
    for key in (("linear_2", 256), ("linear_4", 4096),
                ("linear_6", 65536)):
        representative[f"{key[0]}_n{key[1]}"] = next(
            row for row in rows
            if (row["case_identifier"], row["element_count"]) == key)
    decision = (
        all(vectorized.values()) and
        representative["linear_4_n4096"]["p1_generic_loop_to_oracle_loop_ratio"] >= 2.0 and
        representative["linear_6_n65536"]["p1_generic_loop_to_oracle_loop_ratio"] >= 2.0 and
        all(row["oracle_correct"] and row["oracle_fpe_parity"] for row in rows)
    )
    document = {
        "schema_version": 1,
        "gate": "G0",
        "decision": "PASS" if decision else "FAIL",
        "configuration": {
            "compiler": args.cc,
            "compiler_flags": list(COMPILER_FLAGS),
            "unsafe_fast_math": False,
            "independent_processes": args.processes,
            "samples_per_process": args.samples,
            "target_elements": args.target_elements,
            "n_grid": list(args.n_grid),
            "p1_results": str(args.p1_results),
            "integrated_static_confirmation": str(args.integrated_static_results),
            "compile_command": compile_command,
        },
        "compiler_evidence": {
            "vectorized": vectorized,
            "diagnostics": str(diagnostics),
            "assembly": str(assembly),
        },
        "representative": representative,
        "rows": rows,
        "method_note": (
            "P1 ordinary-NumPy and generic-CrossJIT values are immutable "
            "five-process measurements from P1. The benchmark-only oracle "
            "measures the compiler-visible loop into a preallocated output; "
            "the P1 cost decomposition provides the like-for-like generic "
            "fixed-loop comparison. Pre-existing integrated-static P3 data "
            "is included only as an end-to-end confirmation."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "g0_generator_feasibility.json"
    csv_path = args.output_dir / "g0_generator_feasibility.csv"
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(
            {name for row in rows for name in row}
        ), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "decision": document["decision"],
        "json": str(json_path),
        "csv": str(csv_path),
        "representative": representative,
        "vectorized": vectorized,
    }, indent=2, sort_keys=True))
    return 0 if decision else 2


if __name__ == "__main__":
    raise SystemExit(main())
