#!/usr/bin/env python3
"""Generate reviewed, straight-line AOT Region kernels and their lookup table.

The generated kernels deliberately preserve node order and do not reassociate
floating-point expressions.  Each manifest entry supports 2--8 nodes, the
float32/float64 native types, and wrapping int32/int64 add/sub/mul/neg/square.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


DTYPES = {
    "float32": ("NPY_FLOAT", "float", False),
    "float64": ("NPY_DOUBLE", "double", False),
    "int32": ("NPY_INT32", "npy_uint32", True),
    "int64": ("NPY_INT64", "npy_uint64", True),
}
OPS = {
    0: lambda left, right: f"{left} + {right}",
    1: lambda left, right: f"{left} - {right}",
    2: lambda left, right: f"{left} * {right}",
    3: lambda left, right: f"{left} / {right}",
    4: lambda left, right: f"{left} * {left}",
    5: lambda left, right: f"0 - {left}",
}


def identifier(name: str, index: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return f"numpy_jit_aot_{index}_{clean}"


def validate(entry: dict, seen: set[tuple[int, str, int]]) -> tuple:
    required = {
        "name", "graph_hash", "dtype", "scalar_mask",
        "accepted_bucket_mask", "nodes",
    }
    missing = required - set(entry)
    if missing:
        raise SystemExit(f"manifest entry missing: {sorted(missing)}")
    dtype = entry["dtype"]
    if dtype not in DTYPES:
        raise SystemExit(f"unsupported AOT dtype: {dtype}")
    nodes = entry["nodes"]
    if not 2 <= len(nodes) <= 8:
        raise SystemExit("AOT entries must contain 2--8 nodes")
    input_count = entry.get("input_count", nodes[0]["out"])
    output_slot = entry.get("output_slot", nodes[-1]["out"])
    if not 1 <= input_count <= 8:
        raise SystemExit("AOT native ABI supports 1--8 inputs")
    if entry["scalar_mask"] >> input_count:
        raise SystemExit("scalar_mask refers to a non-input slot")
    available = set(range(input_count))
    normalized_nodes = []
    integer = DTYPES[dtype][2]
    for offset, node in enumerate(nodes):
        op = node["op"]
        if op not in OPS or (integer and op == 3):
            raise SystemExit(f"unsupported AOT operation {op} for {dtype}")
        left = node["left"]
        right = node.get("right", left)
        out = node["out"]
        if left not in available or (op not in (4, 5) and right not in available):
            raise SystemExit("AOT node reads a value before it is produced")
        if out != input_count + offset:
            raise SystemExit("AOT node outputs must be dense and execution ordered")
        available.add(out)
        normalized_nodes.append((op, left, right, node.get("third", 0), out))
    if output_slot not in available:
        raise SystemExit("AOT output_slot is not produced by the graph")
    key = (entry["graph_hash"], dtype, entry["scalar_mask"])
    if key in seen:
        raise SystemExit(f"duplicate AOT key: {key}")
    seen.add(key)
    return input_count, output_slot, normalized_nodes


def function_source(entry: dict, index: int, input_count: int,
                    output_slot: int, nodes: list[tuple]) -> str:
    _, ctype, _ = DTYPES[entry["dtype"]]
    stem = identifier(entry["name"], index)
    scalar_mask = entry["scalar_mask"]
    loads = []
    for slot in range(input_count):
        if scalar_mask & (1 << slot):
            loads.append(
                f"        memcpy(&values[{slot}], &args->scalars[{slot}], "
                f"sizeof(values[{slot}]));")
        else:
            loads.append(
                f"        values[{slot}] = ((const {ctype} *)args->data[{slot}])[i];")
    expressions = []
    for op, left, right, _third, out in nodes:
        expressions.append(
            f"        values[{out}] = {OPS[op](f'values[{left}]', f'values[{right}]')};")
    body = "\n".join(loads + expressions + [
        f"        (({ctype} *)args->output)[i] = values[{output_slot}];"
    ])
    slots = input_count + len(nodes)
    return f"""
static void
{stem}_scalar(const NumpyJitKernelArgs *args)
{{
    for (npy_intp i = 0; i < args->size; i++) {{
        {ctype} values[{slots}];
{body}
    }}
}}

#if NUMPY_JIT_AOT_X86
__attribute__((target("avx2"), optimize("no-fast-math")))
static void
{stem}_avx2(const NumpyJitKernelArgs *args)
{{
    for (npy_intp i = 0; i < args->size; i++) {{
        {ctype} values[{slots}];
{body}
    }}
}}
#endif
"""


def main(manifest_path: Path, output_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dialect_version") != 1:
        raise SystemExit("unsupported AOT manifest dialect")
    entries = manifest.get("entries", [])
    if len(entries) > 32:
        raise SystemExit("AOT manifest is limited to 32 entries")
    seen: set[tuple[int, str, int]] = set()
    functions: list[str] = []
    rows: list[str] = []
    for index, entry in enumerate(entries):
        input_count, output_slot, nodes = validate(entry, seen)
        functions.append(function_source(
            entry, index, input_count, output_slot, nodes))
        stem = identifier(entry["name"], index)
        dtype_macro = DTYPES[entry["dtype"]][0]
        node_rows = ", ".join(
            "{%d, %d, %d, %d, %d}" % node for node in nodes)
        code_size = 96 + len(nodes) * 32
        rows.append(f"""    {{
        UINT64_C({entry['graph_hash']}), {dtype_macro},
        {entry['scalar_mask']}, {entry['accepted_bucket_mask']},
        {input_count}, {output_slot}, {len(nodes)},
        {{ {node_rows} }},
        {{ {stem}_scalar, {entry['accepted_bucket_mask']}, 0,
           {code_size}, "{entry['name']}.scalar" }},
#if NUMPY_JIT_AOT_X86
        {{ {stem}_avx2, {entry['accepted_bucket_mask']}, 256,
           {code_size + 32}, "{entry['name']}.avx2" }},
#else
        {{ NULL, 0, 0, 0, "none" }},
#endif
    }},""")
    output_path.write_text(
        "/* Generated from jit_aot_manifest.json; do not edit. */\n"
        + "\n".join(functions)
        + f"\n#define NUMPY_JIT_AOT_MANIFEST_COUNT {len(rows)}\n"
        + "static const NumpyJitAOTManifestRecord numpy_jit_aot_manifest[] = {\n"
        + "\n".join(rows)
        + "\n};\n"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
