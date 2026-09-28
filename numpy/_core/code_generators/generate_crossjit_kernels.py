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


def _add_numpy_registry(
        header: str, source: str, kernels: list[dict], policies: list[dict]):
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
        "typedef struct {",
        "    unsigned char op, left, right, output;",
        "} NumpyCrossJitGeneratedPolicyNode;",
        "typedef struct {",
        "    unsigned long long min_elements, max_elements;",
        "    unsigned char decision;",
        "} NumpyCrossJitGeneratedPolicyRange;",
        "typedef struct {",
        "    const char *signature_hash;",
        "    unsigned char input_count, node_count;",
        "    unsigned short scalar_mask;",
        "    unsigned char output_slot, range_count;",
        "    NumpyCrossJitGeneratedPolicyNode nodes[6];",
        "    NumpyCrossJitGeneratedPolicyRange ranges[13];",
        "    NumpyCrossJitGeneratedKernelEntry entry;",
        "} NumpyCrossJitGeneratedPolicy;",
        f"#define NPY_CROSSJIT_PRODUCTION_POLICY_COUNT {len(policies)}",
        "extern const NumpyCrossJitGeneratedPolicy",
        "    numpy_crossjit_production_policies[",
        "        NPY_CROSSJIT_PRODUCTION_POLICY_COUNT];",
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
    operation_codes = {
        "numpy.add": 0,
        "numpy.subtract": 1,
        "numpy.multiply": 2,
        "numpy.true_divide": 3,
    }
    policy_rows = []
    for policy in sorted(policies, key=lambda item: item["signature_hash"]):
        signature = policy["signature"]
        nodes = []
        for index, operation in enumerate(signature["operations"]):
            operands = operation["operands"]
            nodes.append(
                "{" + ", ".join(map(str, (
                    operation_codes[operation["operation_id"]], operands[0],
                    operands[1], signature["input_count"] + index))) + "}")
        ranges = [
            "{" + f"{point['min_elements']}ULL, "
            f"{point['max_elements']}ULL, {policy['runtime_codes'][index]}" + "}"
            for index, point in enumerate(policy["points"])
        ]
        policy_rows.append("\n".join((
            "    {",
            f'        "{policy["signature_hash"]}",',
            f"        {signature['input_count']}, "
            f"{len(signature['operations'])}, {signature['scalar_mask']},",
            f"        {signature['input_count'] + len(signature['operations']) - 1}, "
            f"{len(ranges)},",
            "        {" + ", ".join(nodes) + "},",
            "        {" + ", ".join(ranges) + "},",
            f"        {policy['symbol'] or 'NULL'},",
            "    },",
        )))
    source += "\n" + "\n".join((
        "const NumpyCrossJitGeneratedPolicy",
        "numpy_crossjit_production_policies[",
        "        NPY_CROSSJIT_PRODUCTION_POLICY_COUNT] = {",
        *policy_rows,
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
    parser.add_argument("--decision-db", type=Path)
    parser.add_argument("--decision-environment", type=Path)
    args = parser.parse_args()

    generator_root = args.generator_root.resolve()
    sys.path.insert(0, str(generator_root))
    import crossjit_generator
    from crossjit_generator.backend.c import CBackend, generate_bundle
    from crossjit_generator.domains.elementwise import ElementwiseOptimizer
    from crossjit_generator.signature import (
        GenerationMode,
        GenerationRequest,
        GenerationSource,
        ObservedSignatures,
        optimization_signature_from_ir,
        resolve_generation_requests,
    )
    from crossjit_generator.decision import (
        ArtifactDecision, DecisionDB, DecisionEnvironment)

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

    selected_hashes = None
    decision_db_hash = None
    decision_environment_data = None
    production_records = []
    if args.decision_db is not None or args.decision_environment is not None:
        if args.decision_db is None or args.decision_environment is None:
            raise ValueError(
                "DecisionDB and decision environment must be supplied together")
        decision_db_text = args.decision_db.read_text()
        decision_db = DecisionDB.from_data(json.loads(decision_db_text))
        decision_environment_data = json.loads(
            args.decision_environment.read_text())
        selected_hashes = set()
        for record in decision_db.records():
            environment = DecisionEnvironment(
                provider=record.signature.provider,
                provider_abi_version=record.signature.provider_abi_version,
                recipe_hashes=tuple(
                    operation.recipe_hash
                    for operation in record.signature.operations),
                generator_version=crossjit_generator.__version__,
                optimizer_version=ElementwiseOptimizer.VERSION,
                backend_version=CBackend.VERSION,
                target_triple=decision_environment_data["target_triple"],
                target_features=tuple(
                    decision_environment_data["target_features"]),
                compiler_id=decision_environment_data["compiler_id"],
                compiler_version=decision_environment_data["compiler_version"],
                compiler_flags=tuple(
                    decision_environment_data["compiler_flags"]),
                artifact_format_version=decision_environment_data[
                    "artifact_format_version"],
            )
            lookup = decision_db.lookup(record.signature, environment)
            if lookup is not None and lookup.record is not None:
                production_records.append(lookup.record)
        for request in requests:
            environment = DecisionEnvironment(
                provider=request.signature.provider,
                provider_abi_version=request.signature.provider_abi_version,
                recipe_hashes=tuple(
                    operation.recipe_hash
                    for operation in request.signature.operations),
                generator_version=crossjit_generator.__version__,
                optimizer_version=ElementwiseOptimizer.VERSION,
                backend_version=CBackend.VERSION,
                target_triple=decision_environment_data["target_triple"],
                target_features=tuple(
                    decision_environment_data["target_features"]),
                compiler_id=decision_environment_data["compiler_id"],
                compiler_version=decision_environment_data["compiler_version"],
                compiler_flags=tuple(
                    decision_environment_data["compiler_flags"]),
                artifact_format_version=decision_environment_data[
                    "artifact_format_version"],
            )
            lookup = decision_db.lookup(request.signature, environment)
            if (lookup is not None and lookup.record is not None and
                    lookup.state in (ArtifactDecision.AUTO_SELECTED,
                                     ArtifactDecision.FORCED)):
                selected_hashes.add(request.signature.stable_hash())
        selected_hashes = frozenset(selected_hashes)
        decision_db_hash = hashlib.sha256(
            decision_db_text.encode("utf-8")).hexdigest()
    resolution = resolve_generation_requests(
        requests, auto_selected_hashes=selected_hashes)
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
        "decision_db_sha256": decision_db_hash,
        "decision_environment": decision_environment_data,
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
    symbols = {
        kernel["optimization_signature_hash"]: kernel["symbol"]
        for kernel in metadata["kernels"]
    }
    runtime_codes = {"ORIGINAL": 0, "GENERATED": 1, "GENERIC_FALLBACK": 2}
    policies = []
    for record in production_records:
        policies.append({
            "artifact_decision": record.artifact_decision.value,
            "points": [point.to_data()
                       for point in record.profitability.points],
            "runtime_codes": [runtime_codes[point.decision.value]
                              for point in record.profitability.points],
            "signature": record.signature.to_data(),
            "signature_hash": record.signature.stable_hash(),
            "symbol": symbols.get(record.signature.stable_hash()),
        })
    metadata["production_policies"] = policies
    header_text, source_text = _add_numpy_registry(
        bundle.header, bundle.source, metadata["kernels"], policies)
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
