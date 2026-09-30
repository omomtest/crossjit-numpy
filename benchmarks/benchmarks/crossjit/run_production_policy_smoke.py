#!/usr/bin/env python3
"""Small G13 production-policy diagnostic; this is not NPBench evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _run(function, values, expected, internal, provider, np):
    internal.crossjit_reset_stats()
    provider._crossjit_reset_stats()
    result = function(*values, internal.TIER2_THRESHOLD * 2)
    np.testing.assert_array_equal(result, expected)
    return {
        "host": internal.crossjit_stats(),
        "provider": provider._crossjit_stats(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-db", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTHON_JIT_CROSSJIT") != "1":
        raise RuntimeError("production smoke requires PYTHON_JIT_CROSSJIT=1")
    for override in (
        "NPY_CROSSJIT_MIN_WORK",
        "NPY_CROSSJIT_DIAGNOSTIC_FORCE_FALLBACK",
        "NPY_CROSSJIT_DIAGNOSTIC_DISABLE_GENERATED",
    ):
        if os.environ.get(override) is not None:
            raise RuntimeError(f"forced-admission override is set: {override}")

    import numpy as np
    import _testinternalcapi as internal
    from numpy._core import _multiarray_umath as provider

    def generated(left, right, scale, count):
        for _ in range(count):
            result = (left + right) * scale
        return result

    def unmeasured(left, right, scale, count):
        for _ in range(count):
            result = (left + right) * scale
        return result

    def generated_large(left, right, scale, offset, divisor, count):
        for _ in range(count):
            result = (((left + right) * scale) - offset) / divisor
        return result

    def generic_profitable(left, scalar, right, count):
        for _ in range(count):
            result = (left + scalar) * right
        return result

    def generic_regressive(scalar, left, right, count):
        for _ in range(count):
            result = (scalar * left) + right
        return result

    left = np.arange(32.0)
    generated_stats = _run(
        generated, (left, left + 1, left + 2),
        (left + left + 1) * (left + 2), internal, provider, np)
    tiny = np.arange(8.0)
    unmeasured_stats = _run(
        unmeasured, (tiny, tiny + 1, tiny + 2),
        (tiny + tiny + 1) * (tiny + 2), internal, provider, np)
    large = np.arange(65536.0) + 1.0
    generated_large_stats = _run(
        generated_large,
        (large, large + 1, large + 2, large + 3, large + 4),
        (((large + large + 1) * (large + 2)) - (large + 3)) / (large + 4),
        internal, provider, np)
    medium = np.arange(256.0)
    generic_stats = _run(
        generic_profitable, (medium, 0.25, medium + 1),
        (medium + 0.25) * (medium + 1), internal, provider, np)
    regressive_stats = _run(
        generic_regressive, (0.25, medium, medium + 1),
        0.25 * medium + medium + 1, internal, provider, np)

    assert generated_stats["provider"]["generated_exec_hit"] > 0
    assert generated_large_stats["provider"]["generated_exec_hit"] > 0
    assert unmeasured_stats["provider"]["regions_rejected_unprofitable"] == 1
    assert generic_stats["provider"]["generic_fallback_hit"] > 0
    assert regressive_stats["provider"]["regions_rejected_unprofitable"] == 1
    assert regressive_stats["provider"]["generic_fallback_hit"] == 0

    document = {
        "schema_version": 1,
        "formal_npbench_evaluation": False,
        "forced_admission": False,
        "decision_db_sha256": hashlib.sha256(
            args.decision_db.read_bytes()).hexdigest(),
        "cases": {
            "selected_generated_profitable_n32": generated_stats,
            "remeasured_generated_linear_l4_n65536": generated_large_stats,
            "selected_generated_unmeasured_n8": unmeasured_stats,
            "registry_miss_generic_profitable_n256": generic_stats,
            "registry_miss_generic_regressive_n256": regressive_stats,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        document, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
