#!/usr/bin/env python3
"""Select reviewable AOT manifest candidates from CrossJIT telemetry.

This tool intentionally writes a candidate manifest instead of modifying the
checked-in manifest.  Selection is workload-agnostic: graphs are ranked by
observed native uses and saved element time across every supplied telemetry
file; benchmark names and source locations are neither read nor emitted.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-entries", type=int, default=32)
    parser.add_argument("--min-native-uses", type=int, default=3)
    parser.add_argument("--min-gain-percent", type=float, default=3.0)
    return parser.parse_args()


def rows(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise SystemExit(f"{path}:{line_number}: {error}") from error


def key(row: dict[str, Any]) -> tuple[Any, ...]:
    nodes = tuple(
        (node["op"], node["left"], node["right"],
         node.get("third", 0), node["out"])
        for node in row["nodes"]
    )
    return (
        row["graph_hash"], row["dtype_name"], row["scalar_mask"],
        row["input_count"], row["output_slot"], nodes,
    )


def main() -> int:
    args = parse_args()
    if args.max_entries < 1 or args.min_native_uses < 1:
        raise SystemExit("entry and use limits must be positive")
    grouped: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
        lambda: {"uses": 0, "saved": 0.0, "bucket_mask": 0})
    for row in rows(args.telemetry):
        required = {
            "graph_hash", "dtype_name", "scalar_mask", "size_bucket",
            "native_uses", "old_median_ns_per_element_x1024",
            "native_median_ns_per_element_x1024", "visited_elements",
            "input_count", "output_slot", "nodes",
        }
        if not required.issubset(row) or not row["nodes"]:
            continue
        if row["dtype_name"] not in {"float32", "float64", "int32", "int64"}:
            continue
        if not 2 <= len(row["nodes"]) <= 8:
            continue
        if row["input_count"] > 8:
            continue
        if row["dtype_name"].startswith("int") and any(
                node["op"] == 3 for node in row["nodes"]):
            continue
        old = row["old_median_ns_per_element_x1024"]
        native = row["native_median_ns_per_element_x1024"]
        uses = row["native_uses"]
        if old <= 0 or native <= 0 or uses <= 0:
            continue
        gain = 100.0 * (old - native) / old
        if gain < args.min_gain_percent:
            continue
        aggregate = grouped[key(row)]
        aggregate["uses"] += uses
        aggregate["saved"] += max(0, old - native) * row["visited_elements"]
        aggregate["bucket_mask"] |= 1 << row["size_bucket"]

    ranked = sorted(
        ((aggregate["saved"], graph, aggregate)
         for graph, aggregate in grouped.items()
         if aggregate["uses"] >= args.min_native_uses),
        reverse=True,
    )[:args.max_entries]
    entries = []
    for _, graph, aggregate in ranked:
        graph_hash, dtype_name, scalar_mask, input_count, output_slot, nodes = graph
        entries.append({
            "name": f"telemetry_{graph_hash:016x}",
            "graph_hash": graph_hash,
            "dtype": dtype_name,
            "scalar_mask": scalar_mask,
            "accepted_bucket_mask": aggregate["bucket_mask"],
            "input_count": input_count,
            "output_slot": output_slot,
            "observed_native_uses": aggregate["uses"],
            "nodes": [
                {"op": op, "left": left, "right": right,
                 "third": third, "out": out}
                for op, left, right, third, out in nodes
            ],
        })
    output = {
        "dialect_version": 1,
        "selection": {
            "source": "crossjit_numpy_llvm.ndjson",
            "min_native_uses": args.min_native_uses,
            "min_gain_percent": args.min_gain_percent,
        },
        "entries": entries,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
