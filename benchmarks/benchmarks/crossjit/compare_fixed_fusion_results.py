#!/usr/bin/env python3
"""Build a machine-readable before/after fixed-fusion comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(path):
    document = json.loads(path.read_text())
    return {
        (row["case_identifier"], row["element_count"]): row
        for row in document["rows"]
    }


def _compare(matrix, old_path, new_path):
    old = _load(old_path)
    new = _load(new_path)
    if old.keys() != new.keys():
        missing_new = sorted(old.keys() - new.keys())
        missing_old = sorted(new.keys() - old.keys())
        raise ValueError(
            f"matrix mismatch: missing_new={missing_new}, "
            f"missing_old={missing_old}")
    rows = []
    for key in sorted(old):
        before = old[key]
        after = new[key]
        rows.append({
            "matrix": matrix,
            "case_identifier": key[0],
            "element_count": key[1],
            "dtype": after["dtype"],
            "layout": after["layout"],
            "region_length": after["region_length"],
            "operation_pattern": after["operation_pattern"],
            "old_baseline_median_ns": before["baseline_median_ns"],
            "old_crossjit_median_ns": before["crossjit_median_ns"],
            "old_speedup": before["speedup"],
            "old_baseline_cv": before["baseline_cv"],
            "old_crossjit_cv": before["crossjit_cv"],
            "new_baseline_median_ns": after["baseline_median_ns"],
            "new_crossjit_median_ns": after["crossjit_median_ns"],
            "new_speedup": after["speedup"],
            "new_baseline_cv": after["baseline_cv"],
            "new_crossjit_cv": after["crossjit_cv"],
            "baseline_runtime_ratio": (
                before["baseline_median_ns"] / after["baseline_median_ns"]),
            "crossjit_runtime_improvement": (
                before["crossjit_median_ns"] / after["crossjit_median_ns"]),
            "speedup_delta": after["speedup"] - before["speedup"],
            "backend_path": after["backend_path"],
            "valid_crossjit_execution": after["valid_crossjit_execution"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-primary", type=Path, required=True)
    parser.add_argument("--new-primary", type=Path, required=True)
    parser.add_argument("--old-transition", type=Path, required=True)
    parser.add_argument("--new-transition", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    rows = (
        _compare("primary", args.old_primary, args.new_primary) +
        _compare("transition", args.old_transition, args.new_transition)
    )
    document = {
        "schema_version": 1,
        "old_primary": str(args.old_primary),
        "new_primary": str(args.new_primary),
        "old_transition": str(args.old_transition),
        "new_transition": str(args.new_transition),
        "rows": rows,
    }
    json_path = args.output_stem.with_suffix(".json")
    csv_path = args.output_stem.with_suffix(".csv")
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "rows": len(rows),
        "valid": sum(row["valid_crossjit_execution"] for row in rows),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
