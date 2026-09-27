#!/usr/bin/env python3
"""Run one Python workload and persist CrossJIT funnel telemetry as JSON.

The wrapper resets counters after importing NumPy, executes the workload with
its normal ``__main__`` semantics, and samples counters only after completion.
Set ``--detailed`` to retain bounded per-candidate Provider diagnostics.  The
ordinary mode keeps only the low-overhead cumulative counters.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import runpy
import sys


LAYOUT_NAMES = {
    1: "CONTIGUOUS_SAME_SHAPE",
    2: "CONTIGUOUS_BROADCAST",
    3: "ZERO_STRIDE",
    4: "INNER_CONTIGUOUS",
    5: "GENERAL_STRIDED",
}
DTYPE_NAMES = {
    0: "OTHER",
    1: "float32",
    2: "float64",
    3: "int32",
    4: "int64",
}
REJECTION_NAMES = {
    0: "none",
    1: "live_ins",
    2: "input_kind",
    3: "dtype_layout",
    4: "dtype_mismatch",
    5: "no_array",
    6: "broadcast",
    7: "scalar_promotion",
    8: "scalar_range",
    9: "node_count",
    10: "unprofitable",
    11: "kernel",
    12: "value_mapping",
    13: "other",
}


def _distribution(candidates: list[dict[str, int]], field: str, names=None):
    counts = Counter(candidate[field] for candidate in candidates)
    if names is None:
        return {str(key): value for key, value in sorted(counts.items())}
    return {
        names.get(key, f"UNKNOWN_{key}"): value
        for key, value in sorted(counts.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("workload", type=Path)
    parser.add_argument("workload_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.detailed:
        os.environ["NPY_CROSSJIT_DIAGNOSTICS"] = "1"
    else:
        os.environ.pop("NPY_CROSSJIT_DIAGNOSTICS", None)
        os.environ.pop("NPY_CROSSJIT_TELEMETRY_TIMING", None)

    import _testinternalcapi as internal
    from numpy._core import _multiarray_umath as umath

    internal.crossjit_reset_stats()
    umath._crossjit_reset_stats()
    sys.argv = [str(args.workload), *args.workload_args]
    runpy.run_path(str(args.workload), run_name="__main__")

    host = internal.crossjit_stats()
    provider = umath._crossjit_stats()
    candidates = list(provider["candidates"])
    report = {
        "schema_version": 1,
        "workload_id": args.workload_id,
        "detailed_diagnostics": provider["diagnostics_enabled"],
        "funnel": {
            "recognized_nodes": host["recognized_nodes"],
            "host_proven_regions": host["host_regions_proven"],
            "provider_prepare_calls": provider["regions_prepare_called"],
            "provider_accepted_regions": provider[
                "regions_provider_accepted"
            ],
            "published_regions": host["published_regions"],
            "executed_regions": provider["regions_executed"],
            "guard_hit": host["guard_hits"],
            "guard_miss": host["guard_misses"],
            "provider_guard_error": provider["guard_error"],
            "exec_hit": provider["exec_hit"],
            "exec_miss": provider["exec_miss"],
            "exec_error": provider["exec_error"],
        },
        "rejections": {
            "unsupported_node": provider[
                "regions_rejected_unsupported_node"
            ],
            "unsupported_contract": provider[
                "regions_rejected_unsupported_contract"
            ],
            "unprofitable": provider["regions_rejected_unprofitable"],
            "other": provider["regions_rejected_other"],
        },
        "candidate_record_count": len(candidates),
        "candidate_records_dropped": provider[
            "candidate_records_dropped"
        ],
        "candidate_distributions": {
            "region_length": _distribution(candidates, "region_length"),
            "element_count": _distribution(candidates, "element_count"),
            "layout": _distribution(
                candidates, "layout_class", LAYOUT_NAMES
            ),
            "dtype": _distribution(candidates, "dtype_class", DTYPE_NAMES),
            "rejection_reason": _distribution(
                candidates, "rejection_reason", REJECTION_NAMES
            ),
        },
        "host_counters": host,
        "provider_counters": provider,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
