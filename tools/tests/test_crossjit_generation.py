import json
from pathlib import Path
import subprocess
import sys


def test_build_time_generation_is_reproducible(tmp_path):
    root = Path(__file__).parents[2]
    generator_root = root.parent / "crossjit-generator"
    script = root / "numpy/_core/code_generators/generate_crossjit_kernels.py"
    manifest = root / "numpy/_core/code_generators/crossjit_kernel_manifest.json"
    outputs = (
        tmp_path / "generated_crossjit_kernels.c",
        tmp_path / "generated_crossjit_kernels.h",
        tmp_path / "generated_crossjit_kernels.json",
    )
    command = [
        sys.executable, str(script),
        "--generator-root", str(generator_root),
        "--manifest", str(manifest),
        "--output-c", str(outputs[0]),
        "--output-h", str(outputs[1]),
        "--output-json", str(outputs[2]),
    ]
    subprocess.run(command, check=True)
    first = tuple(path.read_bytes() for path in outputs)
    subprocess.run(command, check=True)
    second = tuple(path.read_bytes() for path in outputs)
    assert first == second
    metadata = json.loads(outputs[2].read_text())
    assert metadata["provider"] == "numpy"
    assert len(metadata["kernels"]) == 6
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
