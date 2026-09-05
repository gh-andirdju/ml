from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "poc"))

from cuda_portability_validator import (  # noqa: E402
    build_report,
    classify,
    Finding,
    inspect_cuda_binary,
    markdown_report,
    normalize_target,
    parse_architectures,
    ptx_covers_target,
)


class CudaPortabilityValidatorTests(unittest.TestCase):
    def test_clean_python_project_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("import torch\nprint(torch.zeros(1))\n")
            report = build_report(root, ("sm_90", "sm_120"))
            self.assertEqual(report["overall_status"], "PASS")
            self.assertFalse(report["requires_project_custom_compilation"])

    def test_cuda_source_requires_custom_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kernel.cu").write_text("__global__ void kernel() {}\n")
            report = build_report(root, ("sm_90", "sm_120"))
            self.assertEqual(report["overall_status"], "CUSTOM_BUILD_REQUIRED")
            self.assertEqual(report["findings"][0]["kind"], "custom_cuda_source")

    @unittest.skipUnless(hasattr(os, "symlink"), "symbolic links are unavailable")
    def test_source_symlink_outside_root_is_not_followed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "project"
            root.mkdir()
            outside = parent / "outside.cu"
            outside.write_text("__global__ void kernel() {}\n")
            (root / "linked.cu").symlink_to(outside)
            report = build_report(root, ("sm_90", "sm_120"))
            self.assertEqual(report["overall_status"], "PASS")

    def test_pytorch_cuda_extension_is_detected_with_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "setup.py").write_text(
                "from torch.utils.cpp_extension import CUDAExtension\n"
                "extension = CUDAExtension('example', ['kernel.cu'])\n"
            )
            report = build_report(root, ("sm_90", "sm_120"))
            self.assertEqual(report["overall_status"], "CUSTOM_BUILD_REQUIRED")
            self.assertTrue(
                any(
                    finding["kind"] == "custom_extension_build"
                    and finding["line"] == 1
                    for finding in report["findings"]
                )
            )

    def test_triton_kernel_requires_review_but_torch_compile_is_managed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "compiled.py").write_text(
                "import torch\n"
                "import triton\n"
                "@triton.jit\n"
                "def kernel():\n"
                "    return\n"
                "model = torch.compile(lambda value: value)\n"
            )
            report = build_report(root, ("sm_90", "sm_120"))
            self.assertEqual(report["overall_status"], "CUSTOM_BUILD_REQUIRED")
            kinds = {finding["kind"] for finding in report["findings"]}
            self.assertIn("custom_triton_kernel", kinds)
            self.assertIn("managed_runtime_jit", kinds)
            self.assertTrue(report["runtime_jit_detected"])

    def test_torch_compile_alone_does_not_require_custom_extension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "compiled.py").write_text(
                "import torch\nmodel = torch.compile(lambda value: value)\n"
            )
            report = build_report(root, ("sm_90", "sm_120"))
            self.assertEqual(report["overall_status"], "PASS")
            self.assertFalse(report["requires_project_custom_compilation"])
            self.assertTrue(report["runtime_jit_detected"])

    def test_numba_cuda_kernel_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kernel.py").write_text(
                "from numba import cuda\n"
                "@cuda.jit\n"
                "def kernel(values):\n"
                "    return\n"
            )
            report = build_report(root, ("sm_90", "sm_120"))
            self.assertEqual(report["overall_status"], "CUSTOM_BUILD_REQUIRED")
            self.assertTrue(
                any(
                    finding["kind"] == "custom_runtime_kernel"
                    for finding in report["findings"]
                )
            )

    def test_dynamic_library_load_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "loader.py").write_text(
                "import torch\ntorch.ops.load_library('extension.so')\n"
            )
            report = build_report(root, ("sm_90", "sm_120"))
            self.assertEqual(report["overall_status"], "UNKNOWN")

    def test_known_native_dependency_missing_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.txt").write_text(
                "definitely-missing-test-package==1\n"
            )
            report = build_report(
                root,
                ("sm_90", "sm_120"),
                extra_packages=("definitely-missing-test-package",),
            )
            self.assertEqual(report["overall_status"], "UNKNOWN")
            self.assertEqual(report["summary"]["native_cuda_dependencies"], 1)

    def test_native_import_is_inspected_even_without_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("import torch_sparse\n")
            report = build_report(root, ("sm_90", "sm_120"))
            self.assertEqual(report["overall_status"], "UNKNOWN")
            self.assertEqual(report["summary"]["native_cuda_dependencies"], 1)
            self.assertTrue(
                any(
                    finding["kind"] == "native_cuda_dependency_import"
                    for finding in report["findings"]
                )
            )

    def test_pyproject_dependencies_are_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text(
                "[project]\n"
                "name = 'example'\n"
                "version = '1.0.0'\n"
                "dependencies = ['torch-sparse==1.0']\n"
            )
            report = build_report(root, ("sm_90", "sm_120"))
            self.assertEqual(report["overall_status"], "UNKNOWN")
            self.assertEqual(report["dependencies"][0]["name"], "torch-sparse")

    def test_nested_requirement_file_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirements = root / "config" / "requirements-cuda.txt"
            requirements.parent.mkdir()
            requirements.write_text("torch-sparse==1.0\n")
            report = build_report(root, ("sm_90", "sm_120"))
            self.assertEqual(report["overall_status"], "UNKNOWN")
            self.assertEqual(report["dependencies"][0]["name"], "torch-sparse")

    def test_cuobjdump_native_and_forward_ptx_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "extension.so"
            binary.write_bytes(b"test")
            fake_cuobjdump = root / "cuobjdump"
            fake_cuobjdump.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = \"--list-elf\" ]; then\n"
                "  echo 'ELF file sm_90'\n"
                "else\n"
                "  echo '.target sm_90'\n"
                "fi\n"
            )
            fake_cuobjdump.chmod(0o755)
            inspection = inspect_cuda_binary(
                binary,
                ("sm_90", "sm_120"),
                str(fake_cuobjdump),
            )
            self.assertEqual(inspection.target_coverage["sm_90"], "native")
            self.assertEqual(inspection.target_coverage["sm_120"], "ptx")
            self.assertTrue(inspection.has_ptx)

    def test_cuobjdump_command_error_cannot_report_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "extension.so"
            binary.write_bytes(b"test")
            fake_cuobjdump = root / "cuobjdump"
            fake_cuobjdump.write_text(
                "#!/bin/sh\n"
                "echo 'failed while inspecting sm_90' >&2\n"
                "exit 2\n"
            )
            fake_cuobjdump.chmod(0o755)
            inspection = inspect_cuda_binary(binary, ("sm_90",), str(fake_cuobjdump))
            self.assertEqual(inspection.target_coverage["sm_90"], "unknown")
            self.assertIsNotNone(inspection.error)

    def test_status_classification_covers_ptx_and_rebuild(self) -> None:
        self.assertEqual(classify([], used_ptx=True, rebuild_required=False), "PASS_WITH_PTX")
        self.assertEqual(classify([], used_ptx=False, rebuild_required=True), "REBUILD_REQUIRED")
        custom = Finding(
            kind="custom_cuda_source",
            path="kernel.cu",
            line=None,
            symbol=".cu",
            detail="custom",
            action="build",
        )
        self.assertEqual(
            classify([custom], used_ptx=False, rebuild_required=True),
            "CUSTOM_BUILD_REQUIRED",
        )

    def test_architecture_normalization_and_ptx_forward_coverage(self) -> None:
        self.assertEqual(normalize_target("9.0"), "sm_90")
        self.assertEqual(normalize_target("compute_120"), "sm_120")
        self.assertEqual(
            parse_architectures("elf code for sm_90 and sm_120"),
            ("sm_90", "sm_120"),
        )
        self.assertTrue(ptx_covers_target(("sm_90",), "sm_120"))
        self.assertFalse(ptx_covers_target(("sm_120",), "sm_90"))
        self.assertTrue(ptx_covers_target(("sm_90a",), "sm_90"))
        self.assertFalse(ptx_covers_target(("sm_90a",), "sm_120"))

    def test_markdown_and_json_cli_outputs_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("value = 1\n")
            json_output = root / "artifacts" / "report.json"
            markdown_output = root / "artifacts" / "report.md"
            process = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "poc" / "cuda_portability_validator.py"),
                    "--root",
                    str(root),
                    "--json-output",
                    str(json_output),
                    "--markdown-output",
                    str(markdown_output),
                    "--format",
                    "json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(json_output.read_text())
            self.assertEqual(report["overall_status"], "PASS")
            self.assertIn("# CUDA portability validation", markdown_output.read_text())
            self.assertEqual(json.loads(process.stdout)["overall_status"], "PASS")

    def test_cli_returns_nonzero_when_custom_build_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kernel.cu").write_text("__global__ void kernel() {}\n")
            process = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "poc" / "cuda_portability_validator.py"),
                    "--root",
                    str(root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 2)
            self.assertIn("CUSTOM_BUILD_REQUIRED", process.stdout)

    def test_markdown_escapes_table_separators(self) -> None:
        report = {
            "overall_status": "UNKNOWN",
            "targets": ["sm_90"],
            "requires_project_custom_compilation": False,
            "runtime_jit_detected": False,
            "inspect_binaries": False,
            "findings": [
                {
                    "kind": "test",
                    "path": "a|b.py",
                    "line": 2,
                    "detail": "left|right",
                    "action": "review",
                }
            ],
            "dependencies": [],
            "limitations": [],
        }
        rendered = markdown_report(report)
        self.assertIn("a\\|b.py:2", rendered)
        self.assertIn("left\\|right", rendered)


if __name__ == "__main__":
    unittest.main()
