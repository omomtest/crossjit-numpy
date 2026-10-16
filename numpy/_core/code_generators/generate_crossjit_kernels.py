#!/usr/bin/env python3
"""Generate NumPy CrossJIT kernels during the ordinary Meson build."""

from __future__ import annotations

# Frozen GEN-AUTO / GEN-PROFILED signature epoch; independent of runtime ABI4.
FROZEN_GENERATED_SIGNATURE_ABI = 3

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
        "typedef void (*NumpyCrossJitGeneratedOuterInnerKernelEntry)(",
        "    const void *const *inputs, const ptrdiff_t *outer_strides,",
        "    double *restrict output, size_t outer_count, size_t inner_extent);",
        "typedef void (*NumpyCrossJitGeneratedAffineKernelEntry)(",
        "    const void *const *inputs, const ptrdiff_t *inner_strides,",
        "    const ptrdiff_t *outer_strides, double *restrict output,",
        "    size_t outer_count, size_t inner_extent);",
        "",
        "typedef enum {",
        *enum_rows,
        f"    NPY_CROSSJIT_GENERATED_COUNT = {len(ordered)}",
        "} NumpyCrossJitGeneratedSignature;",
        "",
        "typedef struct {",
        "    const char *signature_id;",
        "    unsigned char input_count;",
        "    unsigned char entry_abi;",
        "    union {",
        "        NumpyCrossJitGeneratedKernelEntry flat;",
        "        NumpyCrossJitGeneratedOuterInnerKernelEntry outer_inner;",
        "        NumpyCrossJitGeneratedAffineKernelEntry affine;",
        "    } entry;",
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
        "    NumpyCrossJitGeneratedPolicyNode nodes[12];",
        "    NumpyCrossJitGeneratedPolicyRange ranges[13];",
        "    NumpyCrossJitGeneratedKernelEntry entry;",
        "} NumpyCrossJitGeneratedPolicy;",
        f"#define NPY_CROSSJIT_PRODUCTION_POLICY_COUNT {len(policies)}",
        "extern const NumpyCrossJitGeneratedPolicy",
        "    numpy_crossjit_production_policies[",
        "        NPY_CROSSJIT_PRODUCTION_POLICY_COUNT];",
        "",
        "typedef struct {",
        "    const char *execution_specialization_hash;",
        "    unsigned char input_count, node_count, output_slot;",
        "    unsigned char descriptor_version, iteration_rank_class;",
        "    unsigned char outer_shape_class, range_count;",
        "    unsigned char measured_cell_count;",
        "    size_t measured_outer_counts[64], measured_inner_extents[64];",
        "    unsigned long long measured_scalar_classes[64];",
        "    unsigned char access_kinds[16], broadcast_relations[16];",
        "    NumpyCrossJitGeneratedPolicyNode nodes[12];",
        "    NumpyCrossJitGeneratedPolicyRange ranges[16];",
        "    NumpyCrossJitGeneratedOuterInnerKernelEntry entry;",
        "    NumpyCrossJitGeneratedAffineKernelEntry affine_entry;",
        "    NumpyCrossJitGeneratedKernelEntry flat_entry;",
        "} NumpyCrossJitGeneratedIterationKernel;",
        f"#define NPY_CROSSJIT_PROFILED_ITERATION_KERNEL_COUNT "
        f"{sum(kernel.get('entry_abi') in ('flat_v1', 'outer_inner_v1', 'affine_v2') and 'iteration_descriptor' in kernel for kernel in ordered)}",
        "extern const NumpyCrossJitGeneratedIterationKernel",
        "    numpy_crossjit_profiled_iteration_registry[",
        "        NPY_CROSSJIT_PROFILED_ITERATION_KERNEL_COUNT];",
        "",
    ))
    marker = "#endif /* GENERATED_CROSSJIT_KERNELS_H */"
    if marker not in header:
        raise ValueError("unexpected generated header guard")
    header = header.replace(marker, declarations + marker)
    registry_rows = []
    for kernel in ordered:
        outer_inner = kernel.get("entry_abi") == "outer_inner_v1"
        affine = kernel.get("entry_abi") == "affine_v2"
        member = "affine" if affine else "outer_inner" if outer_inner else "flat"
        registry_rows.append(
            f'    [{_registry_name(kernel["signature_id"])}] = '
            f'{{"{kernel["signature_id"]}", {kernel["input_count"]}, '
            f'{3 if affine else 2 if outer_inner else 1}, {{.{member} = {kernel["symbol"]}}}}},')
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
    iteration_rows = []
    for kernel in ordered:
        if (kernel.get("entry_abi") not in ("flat_v1", "outer_inner_v1", "affine_v2")
                or "iteration_descriptor" not in kernel):
            continue
        signature = kernel["optimization_signature"]
        nodes = []
        for index, operation in enumerate(signature["operations"]):
            left, right = operation["operands"]
            nodes.append("{" + ", ".join(map(str, (
                operation_codes[operation["operation_id"]], left, right,
                signature["input_count"] + index))) + "}")
        accesses = {
            "CONTIGUOUS": 1, "SCALAR": 2, "ZERO_STRIDE": 3,
            "FIXED_STRIDE": 4,
        }
        access_values = [accesses[item["access_kind"]]
                         for item in kernel["iteration_descriptor"]["operands"]]
        broadcast_codes = {
            "NONE": 0, "SCALAR": 1, "BROADCAST_INNER": 2,
            "BROADCAST_OUTER": 3, "BROADCAST_MULTIPLE": 4,
        }
        broadcast_values = [
            broadcast_codes[item["broadcast_relation"]]
            for item in kernel["iteration_descriptor"]["operands"]
        ]
        execution_policy = kernel.get("execution_policy", {})
        measured_cells = kernel.get("production_iteration_cells", [])
        policy_points = execution_policy.get("points", [])
        policy_ranges = [
            "{" + f"{point['min_elements']}ULL, "
            f"{point['max_elements']}ULL, "
            f"{execution_policy['runtime_codes'][index]}" + "}"
            for index, point in enumerate(policy_points)
        ]
        iteration_rows.append("\n".join((
            "    {",
            f'        "{kernel["execution_specialization_hash"]}",',
            f"        {signature['input_count']}, "
            f"{len(signature['operations'])}, "
            f"{signature['input_count'] + len(signature['operations']) - 1},",
            f"        {kernel['iteration_descriptor']['schema_version']}, "
            f"{2 if kernel['iteration_descriptor']['iteration_rank_class'] == 'OUTER_INNER' else 1}, "
            f"{1 if kernel['iteration_descriptor']['outer_shape_class'] == 'AFFINE_ROWS' else 0}, "
            f"{len(policy_ranges)},",
            f"        {len(measured_cells)},",
            "        {" + ", ".join(str(cell['outer_count']) for cell in measured_cells) + "},",
            "        {" + ", ".join(str(cell['inner_extent']) for cell in measured_cells) + "},",
            "        {" + ", ".join(str(cell.get('scalar_class_mask', 0)) + 'ULL' for cell in measured_cells) + "},",
            "        {" + ", ".join(map(str, access_values)) + "},",
            "        {" + ", ".join(map(str, broadcast_values)) + "},",
            "        {" + ", ".join(nodes) + "},",
            "        {" + ", ".join(policy_ranges) + "},",
            f"        {kernel['symbol'] if kernel['entry_abi'] == 'outer_inner_v1' else 'NULL'},",
            f"        {kernel['symbol'] if kernel['entry_abi'] == 'affine_v2' else 'NULL'},",
            f"        {kernel['symbol'] if kernel['entry_abi'] == 'flat_v1' else 'NULL'},",
            "    },",
        )))
    source += "\n" + "\n".join((
        "const NumpyCrossJitGeneratedIterationKernel",
        "numpy_crossjit_profiled_iteration_registry[",
        "        NPY_CROSSJIT_PROFILED_ITERATION_KERNEL_COUNT] = {",
        *iteration_rows,
        "};",
        "",
    ))
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
    parser.add_argument("--profiled-coverage", type=Path)
    parser.add_argument("--enable-affine", action="store_true")
    parser.add_argument("--profiled-layout-class", action="append", default=[])
    parser.add_argument(
        "--profiled-inner-validation-seed", action="append", default=[])
    parser.add_argument(
        "--profiled-zero-stride-validation-seed",
        action="append", default=[])
    parser.add_argument(
        "--execution-decision-db", type=Path, action="append", default=[])
    parser.add_argument("--production-selection", type=Path)
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
        OptimizationSignature,
        optimization_signature_from_ir,
        resolve_generation_requests,
    )
    from crossjit_generator.decision import (
        ArtifactDecision, DecisionDB, DecisionEnvironment,
        ExecutionDecisionDB)

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
            ir, provider_abi_version=FROZEN_GENERATED_SIGNATURE_ABI)
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
            if (lookup is not None and lookup.record is not None and
                    lookup.state in (
                        ArtifactDecision.AUTO_SELECTED,
                        ArtifactDecision.AUTO_REJECTED,
                        ArtifactDecision.FORCED)):
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
    enabled = list(
        (request, payloads[request.request_id])
        for request in resolution.generated
    )
    profiled_info = {}
    if args.profiled_coverage is not None:
        coverage = json.loads(args.profiled_coverage.read_text())
        for record in coverage["signatures"]:
            if (args.profiled_layout_class and
                    record["layout_class"] not in args.profiled_layout_class):
                continue
            try:
                ir, descriptor = lowering.lower_profile_record(record)
                if descriptor.schema_version == 2 and not args.enable_affine:
                    continue
                candidate = optimizer.legal_candidates(
                    ir, iteration_descriptor=descriptor)[0]
            except (ValueError, KeyError):
                continue
            request_id = "profiled_" + record["profile_variant_hash"][:24]
            request = GenerationRequest(
                OptimizationSignature.from_data(record["signature"]),
                GenerationMode.FORCE, GenerationSource.OBSERVED, request_id)
            enabled.append((request, candidate))
            profiled_info[candidate.stable_hash()] = {
                "profile_variant_hash": record["profile_variant_hash"],
                "profiled_layout_class": record["layout_class"],
                "profiled_source": "observed",
            }
    manifest_by_id = {
        item["id"]: item for item in manifest["signatures"]
    }
    for seed_id in args.profiled_inner_validation_seed:
        if seed_id not in manifest_by_id:
            raise ValueError(f"unknown inner-contiguous validation seed: {seed_id}")
        seed = manifest_by_id[seed_id]
        ir, descriptor = lowering.lower_inner_validation_seed(seed["region"])
        candidate = optimizer.legal_candidates(
            ir, iteration_descriptor=descriptor)[0]
        signature = optimization_signature_from_ir(
            ir, provider_abi_version=FROZEN_GENERATED_SIGNATURE_ABI)
        request_id = f"profiled_validation_{seed_id}"
        request = GenerationRequest(
            signature, GenerationMode.FORCE,
            GenerationSource.DEVELOPER_SEED, request_id)
        enabled.append((request, candidate))
        profiled_info[candidate.stable_hash()] = {
            "profiled_layout_class": "INNER_CONTIGUOUS",
            "profiled_source": "validation_seed",
        }
    for seed_id in args.profiled_zero_stride_validation_seed:
        if seed_id not in manifest_by_id:
            raise ValueError(f"unknown zero-stride validation seed: {seed_id}")
        seed = manifest_by_id[seed_id]
        ir, descriptor = lowering.lower_zero_stride_validation_seed(
            seed["region"])
        candidate = optimizer.legal_candidates(
            ir, iteration_descriptor=descriptor)[0]
        signature = optimization_signature_from_ir(
            ir, provider_abi_version=FROZEN_GENERATED_SIGNATURE_ABI)
        request_id = f"profiled_zero_stride_validation_{seed_id}"
        request = GenerationRequest(
            signature, GenerationMode.FORCE,
            GenerationSource.DEVELOPER_SEED, request_id)
        enabled.append((request, candidate))
        profiled_info[candidate.stable_hash()] = {
            "profiled_layout_class": "ZERO_STRIDE_BROADCAST",
            "profiled_source": "validation_seed",
        }
    production_selection = None
    production_selection_hash = None
    if args.production_selection is not None:
        selection_text = args.production_selection.read_text()
        production_selection = json.loads(selection_text)
        legacy_selection = (
            production_selection.get("schema_version") == 1 and
            production_selection.get("policy_name") == "GEN-PROFILED-PF9" and
            production_selection.get("runtime_profitability_retuned") is False)
        measured_selection = (
            production_selection.get("schema_version") == 2 and
            production_selection.get("policy_name") in (
                "CROSSJIT-V3-AFFINE-V2", "CROSSJIT-V3-SINGLE-NODE") and
            production_selection.get("runtime_profitability_retuned") is True and
            bool(production_selection.get("measurement_inputs")))
        if not (legacy_selection or measured_selection):
            raise ValueError("unsupported production selection manifest")
        artifact_records = [
            record for record in production_selection["records"]
            if measured_selection or record["artifact_present_in_pf6"]
        ]
        expected_candidate_hashes = {
            record["candidate_hash"] for record in artifact_records
        }
        actual_candidate_hashes = {
            candidate.stable_hash() for _, candidate in enabled
        }
        if expected_candidate_hashes != actual_candidate_hashes:
            raise ValueError(
                "production selection does not match candidate universe")
        if measured_selection and production_selection[
                "candidate_universe_hash"] != hashlib.sha256(
                    "\n".join(sorted(actual_candidate_hashes)).encode()).hexdigest():
            raise ValueError("production selection universe hash mismatch")
        selected_candidate_hashes = {
            record["candidate_hash"] for record in artifact_records
            if record["production_artifact_decision"] == "SELECT"
        }
        if any(record["production_artifact_decision"] not in (
                "SELECT", "REJECT") for record in artifact_records):
            raise ValueError("invalid production artifact decision")
        enabled = [
            (request, candidate) for request, candidate in enabled
            if candidate.stable_hash() in selected_candidate_hashes
        ]
        production_selection_hash = hashlib.sha256(
            selection_text.encode("utf-8")).hexdigest()
    # A supplemental profile may repeat a previously observed descriptor.
    # Deduplicate only equal candidates with the same structural signature;
    # neither a digest collision nor a different execution key permits reuse.
    unique_enabled = {}
    for request, candidate in enabled:
        key = candidate.stable_hash()
        previous = unique_enabled.get(key)
        if previous is not None:
            prior_request, prior_candidate = previous
            if (candidate != prior_candidate or
                    request.signature != prior_request.signature):
                raise ValueError("conflicting generated candidate identity")
            continue
        unique_enabled[key] = (request, candidate)
    enabled = tuple(unique_enabled.values())
    candidates = tuple(candidate for _, candidate in enabled)
    signature_info = {
        candidate.stable_hash(): request
        for request, candidate in enabled
    }
    candidate_info = {
        candidate.stable_hash(): candidate
        for _, candidate in enabled
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
            "profiled_execution_specialization_count": sum(
                candidate.stable_hash() in profiled_info
                for candidate in candidates),
        },
        "decision_db_sha256": decision_db_hash,
        "decision_environment": decision_environment_data,
        "unsupported_observed_signatures": sorted(
            unsupported_observed, key=lambda item: item["signature_hash"]),
    })
    if production_selection is not None:
        metadata["production_selection"] = {
            "candidate_universe_hash": production_selection[
                "candidate_universe_hash"],
            "manifest_sha256": production_selection_hash,
            "policy_name": production_selection["policy_name"],
            "rejected_artifact_count": production_selection["summary"][
                "production_rejected_artifact_count"],
            "runtime_profitability_retuned": production_selection[
                "runtime_profitability_retuned"],
            "selected_artifact_count": production_selection["summary"][
                "production_selected_artifact_count"],
        }
    for kernel in metadata["kernels"]:
        if production_selection is not None:
            record = next(item for item in artifact_records
                          if item['candidate_hash'] == kernel['candidate_hash'])
            cells = record.get('measured_iteration_cells', [])
            if len(cells) > 64 or any(
                    cell['outer_count'] < 1 or cell['inner_extent'] < 1
                    or not 0 <= cell.get('scalar_class_mask', 0) < 2**64
                    for cell in cells):
                raise ValueError('invalid measured iteration policy cells')
            kernel['production_iteration_cells'] = cells
            if record.get('value_class_policy_version') == 1:
                if any('scalar_class_mask' not in cell for cell in cells):
                    raise ValueError('value-class policy cannot inherit a class-blind cell')
                operands = kernel['iteration_descriptor']['operands']
                for cell in cells:
                    mask = cell['scalar_class_mask']
                    if mask >> (4 * len(operands)) or any(
                            ((mask >> (4 * slot)) & 15) not in (
                                (1, 2, 3, 4, 5) if operand['access_kind'] == 'SCALAR'
                                else (0,))
                            for slot, operand in enumerate(operands)):
                        raise ValueError('value-class policy does not match live-in access classes')
                kernel['value_class_policy_version'] = 1
        request = signature_info[kernel["candidate_hash"]]
        candidate = candidate_info[kernel["candidate_hash"]]
        execution_specialization = candidate.execution_specialization(
            request.signature.stable_hash())
        kernel["generation_mode"] = request.mode.value
        kernel["generation_source"] = request.source.value
        kernel["optimization_signature"] = request.signature.to_data()
        kernel["optimization_signature_hash"] = (
            request.signature.stable_hash())
        kernel["execution_specialization"] = (
            execution_specialization.to_data())
        kernel["execution_specialization_hash"] = (
            execution_specialization.stable_hash())
        if kernel.get('value_class_policy_version') == 1:
            kernel['value_class_profitability_hash'] = hashlib.sha256(
                json.dumps({'schema_version': 1,
                            'execution_specialization_hash': execution_specialization.stable_hash(),
                            'backend': kernel['entry_abi'], 'effect': 'fresh-output',
                            'cells': kernel['production_iteration_cells']},
                           sort_keys=True).encode()).hexdigest()
        kernel["profitability_specialization_hash"] = (
            execution_specialization.stable_hash())
        kernel["signature_id"] = request.request_id
        kernel.update(profiled_info.get(kernel["candidate_hash"], {}))
    runtime_codes = {"ORIGINAL": 0, "GENERATED": 1, "GENERIC_FALLBACK": 2}
    execution_db_hashes = []
    if args.execution_decision_db:
        if decision_environment_data is None:
            raise ValueError(
                "execution DecisionDB requires the decision environment")
        kernels_by_specialization = {
            kernel["execution_specialization_hash"]: kernel
            for kernel in metadata["kernels"]
        }
        attached_specializations = set()
        for execution_db_path in args.execution_decision_db:
            execution_db_text = execution_db_path.read_text()
            execution_db = ExecutionDecisionDB.from_data(
                json.loads(execution_db_text))
            execution_db_hashes.append(hashlib.sha256(
                execution_db_text.encode("utf-8")).hexdigest())
            for record in execution_db.records():
                specialization_hash = record.specialization.stable_hash()
                if specialization_hash in attached_specializations:
                    raise ValueError(
                        "duplicate execution specialization across DecisionDBs")
                attached_specializations.add(specialization_hash)
                kernel = kernels_by_specialization.get(specialization_hash)
                environment = DecisionEnvironment(
                    provider=record.signature.provider,
                    provider_abi_version=record.signature.provider_abi_version,
                    recipe_hashes=tuple(
                        operation.recipe_hash
                        for operation in record.signature.operations),
                    generator_version=crossjit_generator.__version__,
                    optimizer_version=ElementwiseOptimizer.VERSION,
                    backend_version=(CBackend.AFFINE_VERSION
                                     if kernel is not None and kernel.get("entry_abi") == "affine_v2"
                                     else CBackend.VERSION),
                    target_triple=decision_environment_data["target_triple"],
                    target_features=tuple(
                        decision_environment_data["target_features"]),
                    compiler_id=decision_environment_data["compiler_id"],
                    compiler_version=decision_environment_data[
                        "compiler_version"],
                    compiler_flags=tuple(
                        decision_environment_data["compiler_flags"]),
                    artifact_format_version=decision_environment_data[
                        "artifact_format_version"],
                )
                lookup = execution_db.lookup(
                    record.specialization, environment)
                kernel = kernels_by_specialization.get(specialization_hash)
                if (kernel is None or lookup is None or
                        lookup.record is None or lookup.state not in (
                            ArtifactDecision.AUTO_SELECTED,
                            ArtifactDecision.AUTO_REJECTED,
                            ArtifactDecision.FORCED)):
                    continue
                kernel["execution_policy"] = {
                    "artifact_decision": lookup.state.value,
                    "points": [point.to_data()
                               for point in lookup.record.profitability.points],
                    "runtime_codes": [
                        runtime_codes[point.decision.value]
                        for point in lookup.record.profitability.points],
                }
    symbols = {
        kernel["optimization_signature_hash"]: kernel["symbol"]
        for kernel in metadata["kernels"]
        if kernel.get("entry_abi") == "flat_v1"
    }
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
    metadata["execution_decision_db_sha256"] = execution_db_hashes
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
