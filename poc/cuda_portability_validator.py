#!/usr/bin/env python3
"""Detect CUDA code that may require target-specific extension compilation."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


TOOL_VERSION = "1.0.0"
DEFAULT_TARGETS = ("sm_90", "sm_120")
IGNORED_DIRECTORIES = frozenset(
    {
        ".artifacts",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "node_modules",
        "pdf",
        "venv",
    }
)
CUDA_SOURCE_SUFFIXES = frozenset({".cu", ".cuh"})
PYTHON_SUFFIXES = frozenset({".py", ".pyw"})
BUILD_FILENAMES = frozenset(
    {
        "CMakeLists.txt",
        "Makefile",
        "meson.build",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
    }
)
BUILD_SUFFIXES = frozenset(
    {".bash", ".cmake", ".dockerfile", ".mk", ".sh", ".yaml", ".yml", ".zsh"}
)
BINARY_SUFFIXES = frozenset({".dll", ".dylib", ".pyd", ".so"})

# These packages publish or commonly require separately compiled PyTorch/PyG kernels.
KNOWN_NATIVE_CUDA_PACKAGES = frozenset(
    {
        "dgl",
        "flash-attn",
        "pyg-lib",
        "torch-cluster",
        "torch-scatter",
        "torch-sparse",
        "torch-spline-conv",
        "xformers",
    }
)

# These are expected to manage their own official architecture binaries. They are reported
# but are not classified as project-owned custom extensions.
FRAMEWORK_MANAGED_PACKAGES = frozenset(
    {"torch", "torch-geometric", "torchaudio", "torchvision"}
)

CUSTOM_BUILD_KINDS = frozenset(
    {
        "custom_cuda_source",
        "custom_extension_build",
        "custom_runtime_kernel",
        "custom_triton_kernel",
        "nvcc_build_command",
    }
)
UNKNOWN_KINDS = frozenset(
    {"dynamic_native_load", "python_parse_error", "unverified_native_dependency"}
)
CUSTOM_RUNTIME_KERNEL_SYMBOLS = frozenset(
    {
        "cupy.RawKernel",
        "cupy.RawModule",
        "numba.cuda.jit",
        "torch.cuda.jiterator._create_jit_fn",
    }
)


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    line: int | None
    symbol: str
    detail: str
    action: str


@dataclass(frozen=True)
class BinaryInspection:
    path: str
    native_architectures: tuple[str, ...]
    ptx_architectures: tuple[str, ...]
    has_ptx: bool
    target_coverage: dict[str, str]
    error: str | None = None


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def normalize_target(value: str) -> str:
    target = value.strip().lower().replace("compute_", "sm_")
    if re.fullmatch(r"\d+\.\d+", target):
        major, minor = target.split(".", maxsplit=1)
        target = f"sm_{major}{minor}"
    elif re.fullmatch(r"\d+[a-z]?", target):
        target = f"sm_{target}"
    if not re.fullmatch(r"sm_\d+[a-z]?", target):
        raise argparse.ArgumentTypeError(
            f"invalid CUDA target {value!r}; use forms such as sm_90, 9.0, or 120"
        )
    return target


def target_number(target: str) -> tuple[int, str]:
    match = re.fullmatch(r"(?:sm|compute)_(\d+)([a-z]?)", target)
    if match is None:
        raise ValueError(f"invalid normalized CUDA target: {target}")
    return int(match.group(1)), match.group(2)


def path_is_ignored(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    return any(part in IGNORED_DIRECTORIES for part in relative.parts)


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def iter_scannable_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path_is_ignored(path, root):
            continue
        if (
            path.suffix in CUDA_SOURCE_SUFFIXES
            or path.suffix in PYTHON_SUFFIXES
            or path.name in BUILD_FILENAMES
            or path.name.startswith("Dockerfile")
            or path.suffix in BUILD_SUFFIXES
        ):
            candidates.append(path)
    return sorted(candidates)


def dotted_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value, aliases)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def python_findings(path: Path, root: Path) -> list[Finding]:
    display_path = relative_path(path, root)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=display_path)
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        line = getattr(error, "lineno", None)
        return [
            Finding(
                kind="python_parse_error",
                path=display_path,
                line=line,
                symbol="ast.parse",
                detail=f"Python source could not be inspected: {error}",
                action=(
                    "Review the file manually or run the validator with a compatible "
                    "Python version."
                ),
            )
        ]

    aliases: dict[str, str] = {}
    findings: list[Finding] = []
    seen: set[tuple[str, int | None, str]] = set()

    def add(
        node: ast.AST,
        kind: str,
        symbol: str,
        detail: str,
        action: str,
    ) -> None:
        line = getattr(node, "lineno", None)
        key = (kind, line, symbol)
        if key in seen:
            return
        seen.add(key)
        findings.append(
            Finding(kind, display_path, line, symbol, detail, action)
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif node.module:
                modules.append(node.module)
            if any(module.startswith("torch.utils.cpp_extension") for module in modules):
                add(
                    node,
                    "custom_extension_build",
                    "torch.utils.cpp_extension",
                    "Imports PyTorch's local C++/CUDA extension build interface.",
                    "Build and test the extension for every required CUDA target.",
                )
            for module in modules:
                package = normalize_package_name(module.split(".", maxsplit=1)[0])
                if package in KNOWN_NATIVE_CUDA_PACKAGES:
                    add(
                        node,
                        "native_cuda_dependency_import",
                        package,
                        "Imports a package that commonly ships separately compiled CUDA kernels.",
                        "Inspect its installed CUDA wheel for every required target.",
                    )

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                decorator_node = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                name = dotted_name(decorator_node, aliases)
                if name in {"triton.jit", "triton.autotune", "triton.heuristics"}:
                    add(
                        decorator,
                        "custom_triton_kernel",
                        name,
                        "Defines a custom Triton kernel that is compiled at runtime.",
                        "Exercise this kernel on both Hopper and Blackwell targets.",
                    )
                elif name in CUSTOM_RUNTIME_KERNEL_SYMBOLS:
                    add(
                        decorator,
                        "custom_runtime_kernel",
                        name,
                        "Defines or compiles custom GPU device code at runtime.",
                        (
                            "Exercise the generated kernel on both Hopper and "
                            "Blackwell targets."
                        ),
                    )

        if not isinstance(node, ast.Call):
            continue
        name = dotted_name(node.func, aliases)
        if name is None:
            continue
        if name in {
            "torch.utils.cpp_extension.CUDAExtension",
            "torch.utils.cpp_extension.CppExtension",
            "torch.utils.cpp_extension.load",
            "torch.utils.cpp_extension.load_inline",
        } or name.rsplit(".", maxsplit=1)[-1] in {
            alias
            for alias, original in aliases.items()
            if original.startswith("torch.utils.cpp_extension.")
        }:
            add(
                node,
                "custom_extension_build",
                name,
                "Builds or loads a project-specific PyTorch native extension.",
                (
                    "Compile native code for sm_90 and sm_120, with a PTX fallback "
                    "where appropriate."
                ),
            )
        elif name in {"torch.ops.load_library", "ctypes.CDLL", "ctypes.PyDLL"}:
            add(
                node,
                "dynamic_native_load",
                name,
                (
                    "Loads a native library dynamically; its CUDA architecture "
                    "coverage is not statically known."
                ),
                (
                    "Inspect the loaded library with cuobjdump or document that it "
                    "contains no CUDA device code."
                ),
            )
        elif name in CUSTOM_RUNTIME_KERNEL_SYMBOLS:
            add(
                node,
                "custom_runtime_kernel",
                name,
                "Defines or compiles custom GPU device code at runtime.",
                "Exercise the generated kernel on both Hopper and Blackwell targets.",
            )
        elif name == "torch.compile":
            add(
                node,
                "managed_runtime_jit",
                name,
                (
                    "Uses framework-managed runtime compilation, not a project-owned "
                    "CUDA extension."
                ),
                "Smoke-test the compiled path on each target GPU.",
            )
        elif name.startswith("subprocess.") and node.args:
            first = node.args[0]
            command: str | None = None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                command = first.value
            elif isinstance(first, (ast.List, ast.Tuple)) and first.elts:
                item = first.elts[0]
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    command = item.value
            if command and Path(command).name == "nvcc":
                add(
                    node,
                    "nvcc_build_command",
                    name,
                    "Invokes nvcc directly from Python.",
                    "Compile and verify outputs for every required CUDA target.",
                )

    return findings


BUILD_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"\bnvcc\b"),
        "nvcc_build_command",
        "Invokes the CUDA compiler directly.",
    ),
    (
        re.compile(r"\b(?:enable_language|project)\s*\([^\n)]*\bCUDA\b", re.I),
        "custom_extension_build",
        "Enables CUDA as a native build language.",
    ),
    (
        re.compile(r"\bfind_package\s*\(\s*(?:CUDA|CUDAToolkit)\b", re.I),
        "custom_extension_build",
        "Locates a CUDA toolkit during a native build.",
    ),
    (
        re.compile(r"\b(?:CUDAExtension|CppExtension|load_inline)\b"),
        "custom_extension_build",
        "References a PyTorch native extension build helper.",
    ),
)


def build_file_findings(path: Path, root: Path) -> list[Finding]:
    display_path = relative_path(path, root)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    findings: list[Finding] = []
    for line_number, line in enumerate(lines, start=1):
        for pattern, kind, detail in BUILD_PATTERNS:
            if pattern.search(line):
                findings.append(
                    Finding(
                        kind=kind,
                        path=display_path,
                        line=line_number,
                        symbol=pattern.pattern,
                        detail=detail,
                        action=(
                            "Compile and test native outputs for every required CUDA "
                            "target."
                        ),
                    )
                )
    return findings


def scan_project(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    this_file = Path(__file__).resolve()
    for path in iter_scannable_files(root):
        if path.resolve() == this_file:
            continue
        if path.suffix in CUDA_SOURCE_SUFFIXES:
            findings.append(
                Finding(
                    kind="custom_cuda_source",
                    path=relative_path(path, root),
                    line=None,
                    symbol=path.suffix,
                    detail="Project contains CUDA device source.",
                    action=(
                        "Compile it for sm_90 and sm_120 and retain a compatible PTX "
                        "fallback."
                    ),
                )
            )
        elif path.suffix in PYTHON_SUFFIXES:
            findings.extend(python_findings(path, root))
        elif (
            path.name in BUILD_FILENAMES
            or path.name.startswith("Dockerfile")
            or path.suffix in BUILD_SUFFIXES
        ):
            findings.extend(build_file_findings(path, root))
    return findings


def requirement_names(root: Path) -> dict[str, list[str]]:
    declarations: dict[str, list[str]] = {}
    requirement_files = sorted(
        path
        for path in root.rglob("requirements*.txt")
        if path.is_file() and not path_is_ignored(path, root)
    )
    for path in requirement_files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.split("#", maxsplit=1)[0].strip()
            if not line or line.startswith(("-", "http://", "https://", "git+")):
                continue
            match = re.match(r"([A-Za-z0-9_.-]+)", line)
            if match is None:
                continue
            name = normalize_package_name(match.group(1))
            declarations.setdefault(name, []).append(
                f"{relative_path(path, root)}:{line_number}"
            )

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            import tomllib

            configuration = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            configuration = {}
        project = configuration.get("project", {})
        dependency_groups: list[tuple[str, Any]] = [
            ("project.dependencies", project.get("dependencies", [])),
            (
                "build-system.requires",
                configuration.get("build-system", {}).get("requires", []),
            ),
        ]
        dependency_groups.extend(
            (f"project.optional-dependencies.{group}", dependencies)
            for group, dependencies in project.get(
                "optional-dependencies", {}
            ).items()
        )
        poetry_dependencies = (
            configuration.get("tool", {}).get("poetry", {}).get("dependencies", {})
        )
        dependency_groups.append(("tool.poetry.dependencies", poetry_dependencies))
        for location, entries in dependency_groups:
            candidates = entries.keys() if isinstance(entries, dict) else entries
            for entry in candidates:
                match = re.match(r"([A-Za-z0-9_.-]+)", str(entry))
                if match is None or match.group(1).lower() == "python":
                    continue
                name = normalize_package_name(match.group(1))
                declarations.setdefault(name, []).append(
                    f"{relative_path(pyproject, root)}:{location}"
                )
    return declarations


def distribution_binaries(distribution: importlib.metadata.Distribution) -> list[Path]:
    binaries: list[Path] = []
    for entry in distribution.files or ():
        candidate = Path(distribution.locate_file(entry))
        if candidate.suffix.lower() in BINARY_SUFFIXES and candidate.is_file():
            binaries.append(candidate.resolve())
    return sorted(set(binaries))


def parse_architectures(text: str) -> tuple[str, ...]:
    architectures = {
        f"sm_{number}{suffix}"
        for number, suffix in re.findall(r"(?:sm|compute)_([0-9]+)([a-z]?)", text.lower())
    }
    return tuple(sorted(architectures, key=target_number))


def ptx_covers_target(ptx_architectures: Sequence[str], target: str) -> bool:
    target_value, target_suffix = target_number(target)
    for architecture in ptx_architectures:
        ptx_value, ptx_suffix = target_number(architecture)
        if ptx_suffix:
            if ptx_value == target_value and (
                not target_suffix or ptx_suffix == target_suffix
            ):
                return True
        elif ptx_value <= target_value:
            return True
    return False


def inspect_cuda_binary(
    path: Path, targets: Sequence[str], cuobjdump: str
) -> BinaryInspection:
    try:
        elf = subprocess.run(
            [cuobjdump, "--list-elf", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        ptx = subprocess.run(
            [cuobjdump, "--dump-ptx", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return BinaryInspection(
            path=str(path),
            native_architectures=(),
            ptx_architectures=(),
            has_ptx=False,
            target_coverage={target: "unknown" for target in targets},
            error=str(error),
        )

    native_output = "\n".join((elf.stdout, elf.stderr))
    ptx_output = "\n".join((ptx.stdout, ptx.stderr))
    native_architectures = parse_architectures(native_output)
    ptx_architectures = parse_architectures(ptx_output)
    has_ptx = ptx.returncode == 0 and bool(ptx.stdout.strip())
    errors = []
    if elf.returncode not in {0, 1}:
        errors.append(f"cuobjdump --list-elf exited {elf.returncode}")
    if ptx.returncode not in {0, 1}:
        errors.append(f"cuobjdump --dump-ptx exited {ptx.returncode}")
    coverage: dict[str, str] = {}
    for target in targets:
        target_value, target_suffix = target_number(target)
        native_match = any(
            architecture == target
            or (
                target_number(architecture)[0] == target_value
                and not target_suffix
            )
            for architecture in native_architectures
        )
        if errors:
            coverage[target] = "unknown"
        elif native_match:
            coverage[target] = "native"
        elif has_ptx and ptx_covers_target(ptx_architectures, target):
            coverage[target] = "ptx"
        elif has_ptx and not ptx_architectures:
            coverage[target] = "unknown"
        else:
            coverage[target] = "missing"
    return BinaryInspection(
        path=str(path),
        native_architectures=native_architectures,
        ptx_architectures=ptx_architectures,
        has_ptx=has_ptx,
        target_coverage=coverage,
        error="; ".join(errors) or None,
    )


def inspect_dependencies(
    root: Path,
    targets: Sequence[str],
    extra_packages: Sequence[str],
    inspect_binaries: bool,
) -> tuple[list[dict[str, Any]], list[Finding], bool, bool]:
    declarations = requirement_names(root)
    normalized_extra_packages = {
        normalize_package_name(package) for package in extra_packages
    }
    for package in normalized_extra_packages:
        declarations.setdefault(package, []).append("--package or detected import")

    dependencies: list[dict[str, Any]] = []
    findings: list[Finding] = []
    used_ptx = False
    rebuild_required = False
    cuobjdump = shutil.which("cuobjdump") if inspect_binaries else None

    for name, sources in sorted(declarations.items()):
        classification = (
            "framework_managed"
            if name in FRAMEWORK_MANAGED_PACKAGES
            else "native_cuda_extension"
            if name in KNOWN_NATIVE_CUDA_PACKAGES or name in normalized_extra_packages
            else "python_or_unclassified"
        )
        try:
            distribution = importlib.metadata.distribution(name)
            version: str | None = distribution.version
        except importlib.metadata.PackageNotFoundError:
            distribution = None
            version = None

        dependency: dict[str, Any] = {
            "name": name,
            "classification": classification,
            "declared_at": sources,
            "installed_version": version,
            "binary_inspections": [],
        }

        if classification != "native_cuda_extension":
            dependencies.append(dependency)
            continue

        if distribution is None:
            findings.append(
                Finding(
                    kind="unverified_native_dependency",
                    path=sources[0].split(":", maxsplit=1)[0],
                    line=None,
                    symbol=name,
                    detail=(
                        "Native CUDA dependency is declared but is not installed in "
                        "this environment."
                    ),
                    action="Install the target-environment wheel and rerun binary inspection.",
                )
            )
            dependency["coverage_status"] = "unknown_not_installed"
            dependencies.append(dependency)
            continue

        binaries = distribution_binaries(distribution)
        dependency["binary_files"] = [str(path) for path in binaries]
        if not inspect_binaries:
            findings.append(
                Finding(
                    kind="unverified_native_dependency",
                    path=sources[0].split(":", maxsplit=1)[0],
                    line=None,
                    symbol=name,
                    detail=(
                        "Native CUDA dependency is installed but binary inspection "
                        "was not requested."
                    ),
                    action="Rerun with --inspect-binaries on a system with the CUDA toolkit.",
                )
            )
            dependency["coverage_status"] = "unknown_not_inspected"
            dependencies.append(dependency)
            continue

        if cuobjdump is None:
            findings.append(
                Finding(
                    kind="unverified_native_dependency",
                    path=sources[0].split(":", maxsplit=1)[0],
                    line=None,
                    symbol=name,
                    detail="cuobjdump is unavailable, so CUDA binary targets cannot be inspected.",
                    action=(
                        "Install the CUDA toolkit or run this check in the target "
                        "CUDA environment."
                    ),
                )
            )
            dependency["coverage_status"] = "unknown_no_cuobjdump"
            dependencies.append(dependency)
            continue

        inspections = [inspect_cuda_binary(path, targets, cuobjdump) for path in binaries]
        cuda_inspections = [
            inspection
            for inspection in inspections
            if inspection.native_architectures or inspection.has_ptx
        ]
        dependency["binary_inspections"] = [asdict(item) for item in inspections]
        if not cuda_inspections:
            findings.append(
                Finding(
                    kind="unverified_native_dependency",
                    path=sources[0].split(":", maxsplit=1)[0],
                    line=None,
                    symbol=name,
                    detail="No inspectable CUDA device image was found in the package binaries.",
                    action=(
                        "Confirm that the installed package is a CUDA build for the "
                        "target environment."
                    ),
                )
            )
            dependency["coverage_status"] = "unknown_no_device_image"
            dependencies.append(dependency)
            continue

        coverage_values = [
            coverage
            for inspection in cuda_inspections
            for coverage in inspection.target_coverage.values()
        ]
        if "missing" in coverage_values:
            rebuild_required = True
            missing_targets = sorted(
                {
                    target
                    for inspection in cuda_inspections
                    for target, coverage in inspection.target_coverage.items()
                    if coverage == "missing"
                },
                key=target_number,
            )
            findings.append(
                Finding(
                    kind="native_dependency_missing_target",
                    path=sources[0].split(":", maxsplit=1)[0],
                    line=None,
                    symbol=name,
                    detail=(
                        "Installed CUDA binaries lack required coverage for: "
                        + ", ".join(missing_targets)
                        + "."
                    ),
                    action=(
                        "Install a compatible wheel or rebuild the package with native "
                        "targets and a suitable PTX fallback."
                    ),
                )
            )
            dependency["coverage_status"] = "missing_target"
        elif "unknown" in coverage_values:
            findings.append(
                Finding(
                    kind="unverified_native_dependency",
                    path=sources[0].split(":", maxsplit=1)[0],
                    line=None,
                    symbol=name,
                    detail="CUDA binary inspection did not establish target coverage.",
                    action="Review cuobjdump errors and inspect the target-environment wheel.",
                )
            )
            dependency["coverage_status"] = "unknown_inspection_error"
        elif "ptx" in coverage_values:
            used_ptx = True
            dependency["coverage_status"] = "covered_with_ptx"
        else:
            dependency["coverage_status"] = "native"
        dependencies.append(dependency)

    return dependencies, findings, used_ptx, rebuild_required


def classify(
    findings: Sequence[Finding], used_ptx: bool, rebuild_required: bool
) -> str:
    kinds = {finding.kind for finding in findings}
    if kinds & CUSTOM_BUILD_KINDS:
        return "CUSTOM_BUILD_REQUIRED"
    if rebuild_required:
        return "REBUILD_REQUIRED"
    if kinds & UNKNOWN_KINDS:
        return "UNKNOWN"
    if used_ptx:
        return "PASS_WITH_PTX"
    return "PASS"


def build_report(
    root: Path,
    targets: Sequence[str],
    extra_packages: Sequence[str] = (),
    inspect_binaries: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    findings = scan_project(root)
    detected_native_packages = tuple(
        finding.symbol
        for finding in findings
        if finding.kind == "native_cuda_dependency_import"
    )
    dependencies, dependency_findings, used_ptx, rebuild_required = inspect_dependencies(
        root,
        targets,
        (*extra_packages, *detected_native_packages),
        inspect_binaries,
    )
    findings.extend(dependency_findings)
    findings.sort(key=lambda item: (item.path, item.line or 0, item.kind, item.symbol))
    status = classify(findings, used_ptx, rebuild_required)
    return {
        "schema_version": 1,
        "tool": "cuda-portability-validator",
        "tool_version": TOOL_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "root": str(root),
        "targets": list(targets),
        "inspect_binaries": inspect_binaries,
        "overall_status": status,
        "requires_project_custom_compilation": bool(
            {finding.kind for finding in findings} & CUSTOM_BUILD_KINDS
        ),
        "runtime_jit_detected": any(
            finding.kind
            in {
                "custom_runtime_kernel",
                "custom_triton_kernel",
                "managed_runtime_jit",
            }
            for finding in findings
        ),
        "summary": {
            "findings": len(findings),
            "declared_dependencies": len(dependencies),
            "native_cuda_dependencies": sum(
                item["classification"] == "native_cuda_extension"
                for item in dependencies
            ),
        },
        "findings": [asdict(finding) for finding in findings],
        "dependencies": dependencies,
        "limitations": [
            (
                "Static analysis cannot prove that every conditional or downloaded "
                "code path was exercised."
            ),
            (
                "PTX provides forward compatibility; it does not make sm_120 code "
                "run backward on sm_90."
            ),
            (
                "PTX JIT also requires a driver new enough for the PTX ISA emitted "
                "by the toolchain."
            ),
            (
                "Final correctness and performance still require execution on the "
                "target GPU."
            ),
        ],
    }


def markdown_report(report: dict[str, Any]) -> str:
    status = report["overall_status"]
    lines = [
        "# CUDA portability validation",
        "",
        f"- Status: **{status}**",
        f"- Targets: {', '.join(f'`{target}`' for target in report['targets'])}",
        "- Project custom compilation required: "
        f"**{'yes' if report['requires_project_custom_compilation'] else 'no'}**",
        "- Runtime JIT detected: "
        f"**{'yes' if report['runtime_jit_detected'] else 'no'}**",
        "- Binary inspection: "
        f"**{'enabled' if report['inspect_binaries'] else 'disabled'}**",
        "",
        "## Findings",
        "",
    ]
    findings = report["findings"]
    if not findings:
        lines.append(
            "No project-owned CUDA/C++ extension or custom GPU kernel path was detected."
        )
    else:
        lines.extend(
            [
                "| Kind | Location | Detail | Action |",
                "| --- | --- | --- | --- |",
            ]
        )
        for finding in findings:
            location = finding["path"]
            if finding["line"] is not None:
                location += f":{finding['line']}"
            values = [
                finding["kind"],
                location,
                finding["detail"],
                finding["action"],
            ]
            escaped = [
                str(value).replace("|", "\\|").replace("\n", " ")
                for value in values
            ]
            lines.append("| " + " | ".join(escaped) + " |")

    lines.extend(["", "## Declared dependencies", ""])
    dependencies = report["dependencies"]
    if not dependencies:
        lines.append("No requirement files were found.")
    else:
        lines.extend(
            [
                "| Package | Classification | Installed version | Coverage |",
                "| --- | --- | --- | --- |",
            ]
        )
        for dependency in dependencies:
            lines.append(
                "| "
                + " | ".join(
                    (
                        dependency["name"],
                        dependency["classification"],
                        dependency["installed_version"] or "not installed",
                        dependency.get("coverage_status", "managed or not applicable"),
                    )
                )
                + " |"
            )

    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect project-owned CUDA/C++ extensions and inspect optional native "
            "dependencies for CUDA architecture coverage."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root")
    parser.add_argument(
        "--target",
        action="append",
        type=normalize_target,
        dest="targets",
        help=(
            "required CUDA target; repeat for multiple targets "
            "(default: sm_90 and sm_120)"
        ),
    )
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="additional installed native package to inspect; repeat as needed",
    )
    parser.add_argument(
        "--inspect-binaries",
        action="store_true",
        help="use cuobjdump to inspect declared native CUDA package binaries",
    )
    parser.add_argument("--json-output", type=Path, help="write the JSON report")
    parser.add_argument("--markdown-output", type=Path, help="write the Markdown report")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="stdout report format",
    )
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    root = arguments.root.resolve()
    if not root.is_dir():
        print(f"cuda-portability-validator: root is not a directory: {root}", file=sys.stderr)
        return 64
    targets = tuple(dict.fromkeys(arguments.targets or DEFAULT_TARGETS))
    report = build_report(
        root=root,
        targets=targets,
        extra_packages=arguments.package,
        inspect_binaries=arguments.inspect_binaries,
    )
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown_text = markdown_report(report)
    if arguments.json_output:
        write_text(arguments.json_output, json_text)
    if arguments.markdown_output:
        write_text(arguments.markdown_output, markdown_text)
    print(json_text if arguments.format == "json" else markdown_text, end="")
    return {
        "PASS": 0,
        "PASS_WITH_PTX": 0,
        "REBUILD_REQUIRED": 2,
        "CUSTOM_BUILD_REQUIRED": 2,
        "UNKNOWN": 3,
    }[report["overall_status"]]


if __name__ == "__main__":
    raise SystemExit(cli())
