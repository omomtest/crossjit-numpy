#!/usr/bin/env python3
"""Generate NumPy CrossJIT kernels during the ordinary Meson build."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-c", type=Path, required=True)
    parser.add_argument("--output-h", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    generator_root = args.generator_root.resolve()
    sys.path.insert(0, str(generator_root))
    import crossjit_generator
    from crossjit_generator.backend.c import generate_bundle
    from crossjit_generator.domains.elementwise import ElementwiseOptimizer

    lowering = _load_module(
        "numpy_crossjit_lowering", Path(__file__).with_name("crossjit_lowering.py"))
    manifest_text = args.manifest.read_text()
    manifest = json.loads(manifest_text)
    if manifest.get("schema_version") != 1 or manifest.get("provider") != "numpy":
        raise ValueError("unsupported NumPy CrossJIT generation manifest")
    optimizer = ElementwiseOptimizer()
    candidates = []
    signature_ids = {}
    for signature in manifest["signatures"]:
        ir = lowering.lower_manifest(signature["region"])
        candidate = optimizer.legal_candidates(ir)[0]
        candidates.append(candidate)
        signature_ids[candidate.stable_hash()] = signature["id"]
    bundle = generate_bundle(
        tuple(candidates), header_name=args.output_h.name)

    metadata = json.loads(bundle.metadata)
    metadata.update({
        "generator_version": crossjit_generator.__version__,
        "manifest_sha256": hashlib.sha256(
            manifest_text.encode("utf-8")).hexdigest(),
        "provider": "numpy",
        "recipe_set_hash": hashlib.sha256(
            "\n".join(
                lowering.RECIPES[name].stable_hash()
                for name in sorted(lowering.RECIPES)
            ).encode("ascii")
        ).hexdigest(),
    })
    for kernel in metadata["kernels"]:
        kernel["signature_id"] = signature_ids[kernel["candidate_hash"]]
    metadata_text = json.dumps(
        metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n"

    for output in (args.output_c, args.output_h, args.output_json):
        output.parent.mkdir(parents=True, exist_ok=True)
    _write_if_changed(args.output_c, bundle.source)
    _write_if_changed(args.output_h, bundle.header)
    _write_if_changed(args.output_json, metadata_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
