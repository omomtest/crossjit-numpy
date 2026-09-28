#!/usr/bin/env python3
"""Replay observed signatures directly from canonical ElementwiseIR.

This PF2 harness is deliberately independent of Host source reconstruction.
It synthesizes values in canonical logical-slot order, checks ordinary NumPy,
generated, and generic-fixed implementations independently, and records a
portable replay envelope without Python or ndarray object identities.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


GRID = (16, 128, 1024, 8192, 65536)
OPERATORS = {
    "add": (np.add, 0),
    "subtract": (np.subtract, 1),
    "multiply": (np.multiply, 2),
    "true_divide": (np.true_divide, 3),
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _selected_hashes(previous):
    hashes = {item["signature_hash"]
              for item in previous["candidate_decisions"]}
    hashes.update(item["signature_hash"]
                  for item in previous["reconstruction_rejections"])
    return hashes


def _inputs(record, n, special=False):
    values = []
    for item in record.logical_inputs:
        if item.access_kind == "SCALAR":
            values.append(np.array([0.375 + item.logical_slot * 0.125],
                                   dtype=np.float64))
        else:
            value = (1.0 + np.arange(n, dtype=np.float64) / max(n, 1) +
                     item.logical_slot * 0.125)
            values.append(value)
    if special:
        patterns = (
            np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 1.0, -1.0]),
            np.array([-0.0, 0.0, 1.0, -1.0, np.inf, np.nan, 2.0]),
        )
        for index, item in enumerate(record.logical_inputs):
            if item.access_kind != "SCALAR":
                values[index] = np.resize(patterns[index % 2], n).astype(
                    np.float64, copy=False)
    return tuple(values)


def _evaluate(record, inputs):
    values = list(inputs)
    with np.errstate(all="ignore"):
        for operation, dependencies in zip(
                record.elementwise_ir.operations,
                record.operation_dependencies, strict=True):
            operator = operation.semantic_expression["operator"]
            function = OPERATORS[operator][0]
            values.append(function(values[dependencies[0]],
                                   values[dependencies[1]]))
    result = values[record.output_map[0]]
    if np.ndim(result) == 0 or np.size(result) == 1:
        array_length = next(np.size(value) for value in inputs
                            if np.size(value) > 1)
        result = np.full(array_length, np.ravel(result)[0], dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def _assert_equal(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12,
                               equal_nan=True)
    zeros = (actual == 0) & (expected == 0)
    if np.any(zeros):
        np.testing.assert_array_equal(np.signbit(actual[zeros]),
                                      np.signbit(expected[zeros]))


def _generic_source():
    return r'''
void crossjit_replay_generic(const void *const *inputs,
                             double *restrict output, size_t size,
                             size_t input_count, size_t operation_count,
                             const int *operators, const int *left,
                             const int *right, unsigned long long scalar_mask)
{
    for (size_t i = 0; i < size; i++) {
        double values[16];
        for (size_t slot = 0; slot < input_count; slot++) {
            const double *input = (const double *)inputs[slot];
            values[slot] = input[(scalar_mask & (1ULL << slot)) ? 0 : i];
        }
        for (size_t node = 0; node < operation_count; node++) {
            const double a = values[left[node]];
            const double b = values[right[node]];
            double result;
            switch (operators[node]) {
                case 0: result = a + b; break;
                case 1: result = a - b; break;
                case 2: result = a * b; break;
                case 3: result = a / b; break;
                default: result = 0.0; break;
            }
            values[input_count + node] = result;
        }
        output[i] = values[input_count + operation_count - 1];
    }
}
'''


def _compile(candidates, generator, output_dir, cc, cflags):
    bundle = generator.generate_bundle(
        tuple(candidates), header_name="pf2_replay_kernels.h")
    output_dir.mkdir(parents=True, exist_ok=True)
    header = output_dir / "pf2_replay_kernels.h"
    source = output_dir / "pf2_replay_kernels.c"
    library = output_dir / "pf2_replay_kernels.so"
    header.write_text(bundle.header)
    source.write_text(bundle.source + "\n" + _generic_source())
    command = [cc, *cflags.split(), "-shared", "-fPIC", "-o",
               str(library), str(source)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return bundle, library, command


def _pointers(inputs):
    pointer_array = ctypes.c_void_p * len(inputs)
    return pointer_array(*(int(value.ctypes.data) for value in inputs))


def _descriptors(record):
    operators = []
    left = []
    right = []
    for operation, dependencies in zip(record.elementwise_ir.operations,
                                       record.operation_dependencies,
                                       strict=True):
        operators.append(OPERATORS[operation.semantic_expression["operator"]][1])
        left.append(dependencies[0])
        right.append(dependencies[1])
    array = ctypes.c_int * len(operators)
    scalar_mask = sum(1 << item.logical_slot
                      for item in record.logical_inputs
                      if item.access_kind == "SCALAR")
    return array(*operators), array(*left), array(*right), scalar_mask


def _configure_functions(library, artifacts):
    pointer_type = ctypes.POINTER(ctypes.c_void_p)
    functions = {}
    for artifact in artifacts:
        function = getattr(library, artifact.symbol)
        function.argtypes = [pointer_type, ctypes.POINTER(ctypes.c_double),
                             ctypes.c_size_t]
        function.restype = None
        functions[artifact.manifest] = function
    generic = library.crossjit_replay_generic
    generic.argtypes = [
        pointer_type, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_ulonglong,
    ]
    generic.restype = None
    return functions, generic


def _call_generated(function, inputs, output):
    function(_pointers(inputs), output.ctypes.data_as(
        ctypes.POINTER(ctypes.c_double)), output.size)


def _call_generic(function, record, inputs, output, descriptors):
    operators, left, right, scalar_mask = descriptors
    function(_pointers(inputs), output.ctypes.data_as(
        ctypes.POINTER(ctypes.c_double)), output.size,
        len(inputs), len(record.elementwise_ir.operations),
        operators, left, right, scalar_mask)


def _median_ns(callback, iterations, samples):
    values = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        for _ in range(iterations):
            callback()
        values.append((time.perf_counter_ns() - started) / iterations)
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser()
    workspace = Path(__file__).resolve().parents[4]
    parser.add_argument("--generator-root", type=Path,
                        default=workspace / "crossjit-generator")
    parser.add_argument("--observations", type=Path, default=Path(__file__).with_name(
        "results") / "g14" / "g14_observed_signatures.json")
    parser.add_argument("--previous-tuning", type=Path, default=Path(__file__).with_name(
        "results") / "g14" / "g14_signature_tuning.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name(
        "results") / "profiled" / "pf2_replay")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--target-elements", type=int, default=1_000_000)
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--cflags", default="-std=c11 -O3 -fno-strict-aliasing")
    args = parser.parse_args()

    sys.path.insert(0, str(args.generator_root))
    from crossjit_generator.backend import c as generator
    from crossjit_generator.domains.elementwise import ElementwiseOptimizer
    from crossjit_generator.replay import CanonicalIRReplay
    from crossjit_generator.signature import OptimizationSignature

    lowering = _load_module(
        "numpy_crossjit_replay_lowering",
        workspace / "numpy" / "numpy" / "_core" / "code_generators" /
        "crossjit_lowering.py")
    observations = json.loads(args.observations.read_text())["observations"]
    previous = json.loads(args.previous_tuning.read_text())
    selected = _selected_hashes(previous)
    cases = sorted((item for item in observations
                    if item["signature_hash"] in selected),
                   key=lambda item: item["signature_hash"])
    if len(cases) != 8:
        raise RuntimeError(f"expected eight accepted observed signatures, got {len(cases)}")

    records = []
    candidates = []
    signature_to_candidate = {}
    optimizer = ElementwiseOptimizer()
    for case in cases:
        signature = OptimizationSignature.from_data(case["signature"])
        ir = lowering.lower_signature(signature)
        access = tuple("SCALAR" if signature.scalar_mask & (1 << index)
                       else "CONTIGUOUS"
                       for index in range(signature.input_count))
        record = CanonicalIRReplay.from_ir(
            ir, access_kinds=access,
            iteration_descriptor={
                "descriptor_status": "PF2_REPLAY_FACTS_PRE_PF3",
                "inner_extent": "runtime_element_count",
                "iteration_rank_class": "FLAT",
                "operand_access": list(access),
                "outer_count": 1,
            },
        )
        # Exercise persistence: tuning below consumes the decoded record.
        record = CanonicalIRReplay.from_data(json.loads(record.serialize()))
        candidate = optimizer.legal_candidates(record.elementwise_ir)[0]
        records.append((case, record, candidate))
        candidates.append(candidate)
        signature_to_candidate[case["signature_hash"]] = candidate.stable_hash()

    bundle, library_path, compile_command = _compile(
        candidates, generator, args.output_dir, args.cc, args.cflags)
    metadata = json.loads(bundle.metadata)
    candidate_to_kernel = {item["candidate_hash"]: item
                           for item in metadata["kernels"]}
    artifacts = tuple(generator.CBackend().generate(candidate)
                      for candidate in candidates)
    library = ctypes.CDLL(str(library_path))
    functions, generic = _configure_functions(library, artifacts)
    manifest_to_function = {json.loads(key)["candidate_hash"]: value
                            for key, value in functions.items()}

    old_failures = {item["signature_hash"]
                    for item in previous["reconstruction_rejections"]}
    rows = []
    outcomes = []
    for case, record, candidate in records:
        signature_hash = case["signature_hash"]
        generated = manifest_to_function[candidate.stable_hash()]
        descriptors = _descriptors(record)
        try:
            for special in (False, True):
                inputs = _inputs(record, 128, special=special)
                expected = _evaluate(record, inputs)
                generated_output = np.empty(128, dtype=np.float64)
                generic_output = np.empty(128, dtype=np.float64)
                _call_generated(generated, inputs, generated_output)
                _call_generic(generic, record, inputs, generic_output,
                              descriptors)
                _assert_equal(generated_output, expected)
                _assert_equal(generic_output, expected)
            for n in GRID:
                inputs = _inputs(record, n)
                expected = _evaluate(record, inputs)
                generated_output = np.empty(n, dtype=np.float64)
                generic_output = np.empty(n, dtype=np.float64)
                iterations = max(3, min(2000, args.target_elements // n))
                baseline_ns = _median_ns(
                    lambda: _evaluate(record, inputs), iterations,
                    args.samples)
                generated_ns = _median_ns(
                    lambda: _call_generated(
                        generated, inputs, generated_output),
                    iterations, args.samples)
                generic_ns = _median_ns(
                    lambda: _call_generic(
                        generic, record, inputs, generic_output, descriptors),
                    iterations, args.samples)
                _assert_equal(generated_output, expected)
                _assert_equal(generic_output, expected)
                rows.append({
                    "baseline_median_ns": baseline_ns,
                    "correct": True,
                    "element_count": n,
                    "generated_median_ns": generated_ns,
                    "generic_fixed_median_ns": generic_ns,
                    "signature_hash": signature_hash,
                })
            outcomes.append({
                "classification": None,
                "correct": True,
                "old_source_reconstruction_failure": signature_hash in old_failures,
                "replay_hash": record.stable_hash(),
                "signature_hash": signature_hash,
                "status": "TUNABLE",
            })
        except Exception as error:
            outcomes.append({
                "classification": "tooling defect",
                "correct": False,
                "error": repr(error),
                "old_source_reconstruction_failure": signature_hash in old_failures,
                "replay_hash": record.stable_hash(),
                "signature_hash": signature_hash,
                "status": "FAILED",
            })

    replay_manifest = {
        "records": [record.to_data() for _, record, _ in records],
        "schema_version": 1,
        "signature_to_candidate_hash": signature_to_candidate,
    }
    result = {
        "compile_command": compile_command,
        "configuration": {
            "element_grid": list(GRID),
            "formal_npbench_evaluation": False,
            "host_legality_validation": False,
            "samples": args.samples,
            "source_reconstruction": False,
        },
        "generated_metadata": metadata,
        "method": "canonical_elementwise_ir_synthetic_replay",
        "old_reconstruction_failure_hashes": sorted(old_failures),
        "outcomes": outcomes,
        "rows": rows,
        "schema_version": 1,
        "summary": {
            "failed": sum(item["status"] != "TUNABLE" for item in outcomes),
            "old_failures_recovered": sum(
                item["status"] == "TUNABLE" and
                item["old_source_reconstruction_failure"]
                for item in outcomes),
            "tunable": sum(item["status"] == "TUNABLE" for item in outcomes),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pf2_replay_manifest.json").write_text(
        json.dumps(replay_manifest, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "pf2_replay_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0 if result["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
