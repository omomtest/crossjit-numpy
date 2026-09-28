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


def _registry_name(signature_id: str) -> str:
    normalized = "".join(
        character.upper() if character.isalnum() else "_"
        for character in signature_id
    )
    return f"NPY_CROSSJIT_GENERATED_{normalized}"


def _add_numpy_registry(header: str, source: str, kernels: list[dict]):
    """Emit a NumPy-owned static registry around generic generated kernels."""

    ordered = sorted(kernels, key=lambda kernel: kernel["signature_id"])
    enum_rows = [
        f"    {_registry_name(kernel['signature_id'])} = {index},"
        for index, kernel in enumerate(ordered)
    ]
    declarations = "\n".join((
        "typedef void (*NumpyCrossJitGeneratedKernelEntry)(",
        "    const void *const *inputs, double *restrict output, size_t size);",
        "",
        "typedef enum {",
        *enum_rows,
        f"    NPY_CROSSJIT_GENERATED_COUNT = {len(ordered)}",
        "} NumpyCrossJitGeneratedSignature;",
        "",
        "typedef struct {",
        "    const char *signature_id;",
        "    unsigned char input_count;",
        "    NumpyCrossJitGeneratedKernelEntry entry;",
        "} NumpyCrossJitGeneratedKernel;",
        "",
        "extern const NumpyCrossJitGeneratedKernel",
        "    numpy_crossjit_generated_registry[NPY_CROSSJIT_GENERATED_COUNT];",
        "",
    ))
    marker = "#endif /* GENERATED_CROSSJIT_KERNELS_H */"
    if marker not in header:
        raise ValueError("unexpected generated header guard")
    header = header.replace(marker, declarations + marker)
    registry_rows = [
        f'    [{_registry_name(kernel["signature_id"])}] = '
        f'{{"{kernel["signature_id"]}", {kernel["input_count"]}, '
        f'{kernel["symbol"]}}},'
        for kernel in ordered
    ]
    source += "\n" + "\n".join((
        "const NumpyCrossJitGeneratedKernel",
        "numpy_crossjit_generated_registry[NPY_CROSSJIT_GENERATED_COUNT] = {",
        *registry_rows,
        "};",
        "",
    ))
    return header, source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-c", type=Path, required=True)
    parser.add_argument("--output-h", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--observations", type=Path, action="append", default=[])
    args = parser.parse_args()

    generator_root = args.generator_root.resolve()
    sys.path.insert(0, str(generator_root))
    import crossjit_generator
    from crossjit_generator.backend.c import generate_bundle
    from crossjit_generator.domains.elementwise import ElementwiseOptimizer
    from crossjit_generator.signature import (
        GenerationMode,
        GenerationRequest,
        GenerationSource,
        ObservedSignatures,
        optimization_signature_from_ir,
        resolve_generation_requests,
    )

    lowering = _load_module(
        "numpy_crossjit_lowering", Path(__file__).with_name("crossjit_lowering.py"))
    manifest_text = args.manifest.read_text()
    manifest = json.loads(manifest_text)
    if manifest.get("schema_version") != 1 or manifest.get("provider") != "numpy":
        raise ValueError("unsupported NumPy CrossJIT generation manifest")
    optimizer = ElementwiseOptimizer()
    requests = []
    payloads = {}
    for signature in manifest["signatures"]:
        mode = GenerationMode(signature["mode"])
        source = GenerationSource(signature["source"])
        ir = lowering.lower_manifest(signature["region"])
        candidate = optimizer.legal_candidates(ir)[0]
        optimization_signature = optimization_signature_from_ir(
            ir, provider_abi_version=3)
        request = GenerationRequest(
            optimization_signature, mode, source, signature["id"])
        requests.append(request)
        payloads[request.request_id] = candidate

    unsupported_observed = []
    for observations_path in args.observations:
        observations = ObservedSignatures.from_data(
            json.loads(observations_path.read_text()))
        for profile in observations.observations():
            signature_hash = profile.signature.stable_hash()
            request_id = f"observed_{signature_hash[:24]}"
            try:
                ir = lowering.lower_signature(profile.signature)
                candidate = optimizer.legal_candidates(ir)[0]
            except (ValueError, KeyError) as error:
                unsupported_observed.append({
                    "reason": str(error),
                    "signature_hash": signature_hash,
                })
                continue
            request = GenerationRequest(
                profile.signature, GenerationMode.AUTO,
                GenerationSource.OBSERVED, request_id)
            requests.append(request)
            payloads[request.request_id] = candidate

    resolution = resolve_generation_requests(requests)
    enabled = tuple(
        (request, payloads[request.request_id])
        for request in resolution.generated
    )
    candidates = tuple(candidate for _, candidate in enabled)
    signature_info = {
        candidate.stable_hash(): request
        for request, candidate in enabled
    }
    bundle = generate_bundle(
        candidates, header_name=args.output_h.name)

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
        "generation_universe": {
            "actual_generated_candidate_count": len(candidates),
            "disabled_signature_count": len(resolution.disabled_hashes),
            "forced_signature_count": resolution.forced_count,
            "observed_signature_count": resolution.observed_count,
            "unsupported_observed_signature_count": len(unsupported_observed),
            "validation_seed_count": resolution.validation_seed_count,
        },
        "unsupported_observed_signatures": sorted(
            unsupported_observed, key=lambda item: item["signature_hash"]),
    })
    for kernel in metadata["kernels"]:
        request = signature_info[kernel["candidate_hash"]]
        kernel["generation_mode"] = request.mode.value
        kernel["generation_source"] = request.source.value
        kernel["optimization_signature"] = request.signature.to_data()
        kernel["optimization_signature_hash"] = (
            request.signature.stable_hash())
        kernel["signature_id"] = request.request_id
    header_text, source_text = _add_numpy_registry(
        bundle.header, bundle.source, metadata["kernels"])
    metadata_text = json.dumps(
        metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n"

    for output in (args.output_c, args.output_h, args.output_json):
        output.parent.mkdir(parents=True, exist_ok=True)
    _write_if_changed(args.output_c, source_text)
    _write_if_changed(args.output_h, header_text)
    _write_if_changed(args.output_json, metadata_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
