#!/usr/bin/env python3
"""Compare generated PF4 loops with handwritten static inner-row oracles."""

from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import ctypes
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time


SHAPES = ((2, 8), (4, 64), (59, 59), (16, 256), (64, 1024))


def _cv(values):
    return (statistics.stdev(values) / statistics.mean(values)
            if len(values) > 1 and statistics.mean(values) else 0.0)


def _compile(args, directory):
    diagnostics = directory / "vectorization.txt"
    command = [
        args.compiler, "-std=c11", "-O3", "-fno-strict-aliasing",
        "-ffp-contract=off", "-fPIC", "-shared",
        "-fopt-info-vec-optimized=" + str(diagnostics),
        str(args.generated_source), str(args.oracle_source),
        "-I", str(args.generated_source.parent),
        "-o", str(directory / "pf4_parity.so"),
    ]
    subprocess.run(command, check=True)
    return directory / "pf4_parity.so", diagnostics


def _measure(args, library_path, metadata):
    library = ctypes.CDLL(str(library_path))
    rows = []
    for kernel in metadata["kernels"]:
        if (kernel.get("profiled_layout_class") != "INNER_CONTIGUOUS" or
                kernel.get("entry_abi") != "outer_inner_v1"):
            continue
        operations = [item["operation_id"] for item in
                      kernel["optimization_signature"]["operations"]]
        oracle_name = {
            ("numpy.add", "numpy.multiply"):
                "crossjit_pf4_oracle_add_multiply",
            ("numpy.subtract", "numpy.multiply"):
                "crossjit_pf4_oracle_subtract_multiply",
        }.get(tuple(operations))
        if oracle_name is None:
            continue
        generated = getattr(library, kernel["symbol"])
        oracle = getattr(library, oracle_name)
        for function in (generated, oracle):
            function.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_ssize_t),
                ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                ctypes.c_size_t,
            ]
        for outer, inner in SHAPES:
            padded = inner + 3
            inputs = [array("d", (
                1.0 + slot * 0.125 + index / (outer * padded)
                for index in range(outer * padded))) for slot in range(3)]
            buffers = [(ctypes.c_double * len(value)).from_buffer(value)
                       for value in inputs]
            pointers = (ctypes.c_void_p * 3)(*(
                ctypes.cast(value, ctypes.c_void_p).value for value in buffers))
            strides = (ctypes.c_ssize_t * 3)(*(
                (padded * ctypes.sizeof(ctypes.c_double),) * 3))
            outputs = {
                "generated": (ctypes.c_double * (outer * inner))(),
                "oracle": (ctypes.c_double * (outer * inner))(),
            }
            iterations = max(16, min(
                20000, args.target_elements // (outer * inner)))
            samples = defaultdict(list)
            functions = (("generated", generated), ("oracle", oracle))
            for sample in range(args.samples):
                order = functions if sample % 2 == 0 else tuple(reversed(functions))
                for name, function in order:
                    started = time.perf_counter_ns()
                    for _ in range(iterations):
                        function(pointers, strides, outputs[name], outer, inner)
                    samples[name].append(
                        (time.perf_counter_ns() - started) / iterations)
            if list(outputs["generated"]) != list(outputs["oracle"]):
                raise RuntimeError("generated/oracle numerical mismatch")
            rows.append({
                "execution_specialization_hash":
                    kernel["execution_specialization_hash"],
                "profiled_source": kernel["profiled_source"],
                "outer_count": outer,
                "inner_extent": inner,
                "element_count": outer * inner,
                "generated_median_ns": statistics.median(samples["generated"]),
                "oracle_median_ns": statistics.median(samples["oracle"]),
                "generated_over_oracle": (
                    statistics.median(samples["generated"]) /
                    statistics.median(samples["oracle"])),
                "correct": True,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--generated-source", type=Path, required=True)
    parser.add_argument("--generated-metadata", type=Path, required=True)
    parser.add_argument("--oracle-source", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--target-elements", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.generated_metadata.read_text())
    if args.worker:
        print(json.dumps(_measure(args, args.library, metadata),
                         separators=(",", ":")))
        return 0
    if args.output is None:
        parser.error("controller requires --output")
    with tempfile.TemporaryDirectory(prefix="crossjit-pf4-") as name:
        directory = Path(name)
        library, diagnostics = _compile(args, directory)
        schedule = list(range(args.processes))
        random.Random(2909).shuffle(schedule)
        raw = []
        for process in schedule:
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--generated-source", str(args.generated_source),
                "--generated-metadata", str(args.generated_metadata),
                "--oracle-source", str(args.oracle_source),
                "--library", str(library), "--samples", str(args.samples),
                "--target-elements", str(args.target_elements),
            ]
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True)
            for row in json.loads(completed.stdout):
                row["independent_process_index"] = process
                raw.append(row)
        groups = defaultdict(list)
        for row in raw:
            groups[(row["execution_specialization_hash"],
                    row["outer_count"], row["inner_extent"])].append(row)
        rows = []
        for key, values in sorted(groups.items()):
            ratios = [item["generated_over_oracle"] for item in values]
            rows.append({
                **{name: values[0][name] for name in (
                    "execution_specialization_hash", "profiled_source",
                    "outer_count", "inner_extent", "element_count")},
                "generated_over_oracle_median": statistics.median(ratios),
                "generated_over_oracle_cv": _cv(ratios),
                "process_ratios": ratios,
                "correct": True,
            })
        document = {
            "schema_version": 1,
            "independent_processes": args.processes,
            "samples_per_process": args.samples,
            "rows": rows,
            "vectorization_diagnostics": diagnostics.read_text(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            document, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "cells": len(rows),
            "maximum_ratio": max(item["generated_over_oracle_median"]
                                 for item in rows),
            "output": str(args.output),
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
