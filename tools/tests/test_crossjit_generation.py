import json
from pathlib import Path
import subprocess
import sys


def _command(root, manifest, outputs, *extra):
    return [
        sys.executable,
        str(root / "numpy/_core/code_generators/generate_crossjit_kernels.py"),
        "--generator-root", str(root.parent / "crossjit-generator"),
        "--manifest", str(manifest),
        "--output-c", str(outputs[0]),
        "--output-h", str(outputs[1]),
        "--output-json", str(outputs[2]),
        *map(str, extra),
    ]


def test_build_time_generation_is_reproducible(tmp_path):
    root = Path(__file__).parents[2]
    manifest = root / "numpy/_core/code_generators/crossjit_kernel_manifest.json"
    outputs = (
        tmp_path / "generated_crossjit_kernels.c",
        tmp_path / "generated_crossjit_kernels.h",
        tmp_path / "generated_crossjit_kernels.json",
    )
    command = _command(root, manifest, outputs)
    subprocess.run(command, check=True)
    first = tuple(path.read_bytes() for path in outputs)
    subprocess.run(command, check=True)
    second = tuple(path.read_bytes() for path in outputs)
    assert first == second
    metadata = json.loads(outputs[2].read_text())
    assert metadata["provider"] == "numpy"
    assert len(metadata["kernels"]) == 6
    assert {kernel["generation_mode"] for kernel in metadata["kernels"]} == {
        "AUTO"}
    assert {kernel["generation_source"] for kernel in metadata["kernels"]} == {
        "validation"}
    assert metadata["generation_universe"] == {
        "actual_generated_candidate_count": 6,
        "disabled_signature_count": 0,
        "forced_signature_count": 0,
        "observed_signature_count": 0,
        "unsupported_observed_signature_count": 0,
        "validation_seed_count": 6,
    }
    assert all(len(kernel["optimization_signature_hash"]) == 64
               for kernel in metadata["kernels"])
    assert {kernel["signature_id"] for kernel in metadata["kernels"]} == {
        "linear_2_add_multiply",
        "linear_3_add_multiply_subtract",
        "linear_4_add_multiply_subtract_divide",
        "linear_5_add_multiply_subtract_divide_add",
        "linear_6_add_multiply_subtract_divide_add_multiply",
        "branched_3_multiply_multiply_add",
    }
    source = outputs[0].read_text()
    assert source.count("for (") == 6
    assert "switch" not in source
    assert "Python.h" not in source
    header = outputs[1].read_text()
    assert "numpy_crossjit_generated_registry" in source
    assert "NPY_CROSSJIT_GENERATED_COUNT = 6" in header
    for signature_id in {
        kernel["signature_id"] for kernel in metadata["kernels"]
    }:
        enum_name = "NPY_CROSSJIT_GENERATED_" + "".join(
            character.upper() if character.isalnum() else "_"
            for character in signature_id
        )
        assert enum_name in header


def test_observed_force_disable_union_is_bounded(tmp_path):
    root = Path(__file__).parents[2]
    original = json.loads((
        root / "numpy/_core/code_generators/crossjit_kernel_manifest.json"
    ).read_text())
    manifest = tmp_path / "manifest.json"
    forced = dict(original["signatures"][0])
    forced["mode"] = "FORCE"
    forced["source"] = "developer_seed"
    forced["id"] = "forced_duplicate"
    disabled = dict(original["signatures"][1])
    disabled["mode"] = "DISABLE"
    disabled["source"] = "developer_seed"
    disabled["id"] = "disabled_duplicate"
    original["signatures"].extend((forced, disabled))
    manifest.write_text(json.dumps(original))

    generator_root = root.parent / "crossjit-generator"
    sys.path.insert(0, str(generator_root))
    from crossjit_generator.signature import (
        ELEMENT_BUCKET_UPPER_BOUNDS,
        ObservedSignatureProfile,
        ObservedSignatures,
        OptimizationSignature,
    )

    # Same topology as a validation seed, but a scalar relation makes this a
    # distinct observed structural signature without any workload identity.
    signature_data = json.loads(subprocess.check_output(
        _command(root, manifest, (
            tmp_path / "first.c", tmp_path / "first.h", tmp_path / "first.json"))
    ) or (tmp_path / "first.json").read_bytes())
    signature_data = signature_data["kernels"][0]["optimization_signature"]
    signature_data["scalar_mask"] = 1
    observed_signature = OptimizationSignature.from_data(signature_data)
    buckets = [0] * len(ELEMENT_BUCKET_UPPER_BOUNDS)
    buckets[4] = 3
    observations = ObservedSignatures((ObservedSignatureProfile(
        observed_signature, 1, 1, 3, 0, 3, 256, 256, tuple(buckets)),))
    observations_path = tmp_path / "observations.json"
    observations_path.write_text(observations.serialize())

    outputs = (tmp_path / "out.c", tmp_path / "out.h", tmp_path / "out.json")
    subprocess.run(_command(
        root, manifest, outputs, "--observations", observations_path), check=True)
    metadata = json.loads(outputs[2].read_text())
    universe = metadata["generation_universe"]
    assert universe == {
        "actual_generated_candidate_count": 6,
        "disabled_signature_count": 1,
        "forced_signature_count": 1,
        "observed_signature_count": 1,
        "unsupported_observed_signature_count": 0,
        "validation_seed_count": 6,
    }
    assert {kernel["generation_mode"] for kernel in metadata["kernels"]} == {
        "AUTO", "FORCE"}
    assert len(metadata["kernels"]) == 6


def test_decisiondb_selection_and_stale_environment_fail_closed(tmp_path):
    root = Path(__file__).parents[2]
    manifest = root / "numpy/_core/code_generators/crossjit_kernel_manifest.json"
    decision_db = (root /
        "benchmarks/benchmarks/crossjit/results/g12/g12_decision_db.json")
    environment = (root /
        "numpy/_core/code_generators/crossjit_decision_environment.json")
    outputs = (tmp_path / "selected.c", tmp_path / "selected.h",
               tmp_path / "selected.json")
    subprocess.run(_command(
        root, manifest, outputs,
        "--decision-db", decision_db,
        "--decision-environment", environment), check=True)
    selected = json.loads(outputs[2].read_text())
    assert len(selected["kernels"]) == 6
    assert len(selected["production_policies"]) == 8
    assert selected["decision_db_sha256"] == (
        "292c9e1e24134c6c46eb4783bc01224ddf75e0a0303ce9dc9540dd4ec416b1b4")

    disabled_data = json.loads(decision_db.read_text())
    disabled_record = disabled_data["records"][0]
    disabled_record["artifact_decision"] = "DISABLED"
    disabled_db = tmp_path / "disabled_decision_db.json"
    disabled_db.write_text(json.dumps(disabled_data))
    disabled_outputs = (tmp_path / "disabled.c", tmp_path / "disabled.h",
                        tmp_path / "disabled.json")
    subprocess.run(_command(
        root, manifest, disabled_outputs,
        "--decision-db", disabled_db,
        "--decision-environment", environment), check=True)
    disabled = json.loads(disabled_outputs[2].read_text())
    assert len(disabled["production_policies"]) == 7
    assert disabled_record["signature_hash"] not in {
        policy["signature_hash"]
        for policy in disabled["production_policies"]
    }

    stale_data = json.loads(environment.read_text())
    stale_data["compiler_version"] = "stale-compiler"
    stale_environment = tmp_path / "stale_environment.json"
    stale_environment.write_text(json.dumps(stale_data))
    stale_outputs = (tmp_path / "stale.c", tmp_path / "stale.h",
                     tmp_path / "stale.json")
    subprocess.run(_command(
        root, manifest, stale_outputs,
        "--decision-db", decision_db,
        "--decision-environment", stale_environment), check=True)
    stale = json.loads(stale_outputs[2].read_text())
    assert stale["kernels"] == []
    assert stale["production_policies"] == []
