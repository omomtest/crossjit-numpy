#!/usr/bin/env python3
"""Turn the reviewed hot-DAG manifest into a C-only lookup descriptor.

Code is never generated from runtime telemetry.  The manifest is reviewed,
versioned source; this tool makes the build fail if its structural contract is
incomplete or duplicated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(manifest_path: Path, output_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("entries", [])
    seen: set[tuple[int, str, int]] = set()
    rows: list[str] = []
    for entry in entries:
        required = {"name", "graph_hash", "dtype", "scalar_mask",
                    "accepted_bucket_mask", "nodes"}
        missing = required - set(entry)
        if missing:
            raise SystemExit(f"manifest entry missing: {sorted(missing)}")
        key = (entry["graph_hash"], entry["dtype"], entry["scalar_mask"])
        if key in seen:
            raise SystemExit(f"duplicate AOT key: {key}")
        seen.add(key)
        if entry["dtype"] != "float64" or len(entry["nodes"]) != 2:
            raise SystemExit("first AOT generator supports reviewed f64 two-node entries")
        nodes = entry["nodes"]
        rows.append(
            "    { UINT64_C(%d), %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, \"%s\" },"
            % (entry["graph_hash"], entry["scalar_mask"],
               entry["accepted_bucket_mask"],
               nodes[0]["op"], nodes[0]["left"], nodes[0]["right"],
               nodes[0]["out"], nodes[1]["op"], nodes[1]["left"],
               nodes[1]["right"], nodes[1]["out"], len(nodes), entry["name"])
        )
    output_path.write_text(
        "/* Generated from jit_aot_manifest.json; do not edit. */\n"
        "#define NUMPY_JIT_AOT_MANIFEST_COUNT %d\n"
        "static const NumpyJitAOTManifestRecord numpy_jit_aot_manifest[] = {\n%s\n};\n"
        % (len(rows), "\n".join(rows))
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
