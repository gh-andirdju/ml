#!/usr/bin/env python3
"""Compare validated Kaggle CPU and CUDA prediction artifacts."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kaggle_specs import (
    COMPARISON_SPECS_BY_POC_ID,
    FLICKR_2048_KAGGLE_CUDA_SPEC,
    FLICKR_2048_MINIMUM_CUDA_PEAK_BYTES,
    FLICKR_4096_KAGGLE_CUDA_SPEC,
    FLICKR_4096_MINIMUM_CUDA_PEAK_BYTES,
    FLICKR_8192_KAGGLE_CUDA_SPEC,
    FLICKR_8192_MINIMUM_CUDA_PEAK_BYTES,
    FLICKR_WIDE_KAGGLE_CUDA_SPEC,
    FLICKR_WIDE_MINIMUM_CUDA_PEAK_BYTES,
    KAGGLE_RUNS_BY_POC_ID,
)
from proof_common import ProofError, require
from result_artifact import MAX_ARTIFACT_BYTES, load_and_validate_artifact, utc_now


GPU_MEMORY_TARGETS = {
    FLICKR_WIDE_KAGGLE_CUDA_SPEC.poc_id: FLICKR_WIDE_MINIMUM_CUDA_PEAK_BYTES,
    FLICKR_2048_KAGGLE_CUDA_SPEC.poc_id: FLICKR_2048_MINIMUM_CUDA_PEAK_BYTES,
    FLICKR_4096_KAGGLE_CUDA_SPEC.poc_id: FLICKR_4096_MINIMUM_CUDA_PEAK_BYTES,
    FLICKR_8192_KAGGLE_CUDA_SPEC.poc_id: FLICKR_8192_MINIMUM_CUDA_PEAK_BYTES,
}


def load_known_artifact(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"Artifact not found: {path}")
    require(
        0 < path.stat().st_size <= MAX_ARTIFACT_BYTES,
        "Artifact size is outside the accepted range",
    )
    try:
        preview = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise ProofError(f"Artifact is not valid JSON: {error}") from error
    require(isinstance(preview, dict), "Artifact root must be an object")
    spec = COMPARISON_SPECS_BY_POC_ID.get(preview.get("poc_id"))
    require(spec is not None, "Artifact POC ID is not supported for comparison")
    return load_and_validate_artifact(path, spec)


def kaggle_run_evidence(
    poc_id: str, *, verify_remote_status: bool
) -> dict[str, Any]:
    run = KAGGLE_RUNS_BY_POC_ID.get(poc_id)
    require(run is not None, "Kaggle run identity is not registered")
    project_root = Path(__file__).resolve().parents[1]
    metadata_path = project_root / run.metadata_directory / "kernel-metadata.json"
    require(metadata_path.is_file(), f"Kaggle metadata not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf8"))
    except json.JSONDecodeError as error:
        raise ProofError(f"Kaggle metadata is not valid JSON: {error}") from error
    require(metadata.get("id") == run.kernel_id, "Kaggle kernel ID differs")
    require(metadata.get("is_private") == "true", "Kaggle proof must be private")
    require(metadata.get("enable_tpu") == "false", "Kaggle proof must not use TPU")
    expected_gpu = "true" if run.enable_gpu else "false"
    require(metadata.get("enable_gpu") == expected_gpu, "Kaggle GPU setting differs")
    expected_shape = "NvidiaTeslaT4" if run.enable_gpu else ""
    require(metadata.get("machine_shape") == expected_shape, "Kaggle machine shape differs")

    reference = f"{run.kernel_id}/{run.version}"
    status = "NOT_CHECKED"
    checked_at = None
    if verify_remote_status:
        completed = subprocess.run(
            ["kaggle", "kernels", "status", reference],
            check=True,
            capture_output=True,
            text=True,
        )
        require(
            f'{reference} has status "KernelWorkerStatus.COMPLETE"'
            in completed.stdout,
            "Kaggle kernel version is not COMPLETE",
        )
        status = "COMPLETE"
        checked_at = utc_now()
    return {
        "kernel_id": run.kernel_id,
        "kernel_version": run.version,
        "version_reference": reference,
        "url": f"https://www.kaggle.com/code/{run.kernel_id}",
        "is_private": True,
        "enable_gpu": run.enable_gpu,
        "enable_tpu": False,
        "machine_shape": expected_shape or None,
        "status": status,
        "status_checked_at": checked_at,
    }


def load_cpu_resource_evidence(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"CPU resource evidence not found: {path}")
    try:
        evidence = json.loads(path.read_text(encoding="utf8"))
    except json.JSONDecodeError as error:
        raise ProofError(f"CPU resource evidence is not valid JSON: {error}") from error
    require(isinstance(evidence, dict), "CPU resource evidence must be an object")
    require(evidence.get("exit_status") == 0, "Measured CPU runner did not pass")
    cpu_percent = evidence.get("average_process_cpu_percent")
    maximum_rss_kib = evidence.get("maximum_resident_set_kib")
    maximum_rss_bytes = evidence.get("maximum_resident_set_bytes")
    require(
        isinstance(cpu_percent, (int, float))
        and not isinstance(cpu_percent, bool)
        and float(cpu_percent) > 0,
        "CPU utilization evidence is invalid",
    )
    require(
        isinstance(maximum_rss_kib, int)
        and not isinstance(maximum_rss_kib, bool)
        and maximum_rss_kib > 0,
        "Maximum RSS evidence is invalid",
    )
    require(
        maximum_rss_bytes == maximum_rss_kib * 1024,
        "Maximum RSS byte conversion is invalid",
    )
    for field in ("user_cpu_seconds", "system_cpu_seconds", "wall_clock_seconds"):
        value = evidence.get(field)
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) >= 0,
            f"CPU resource field is invalid: {field}",
        )
    return evidence


def validate_gpu_memory(
    execution: dict[str, Any], minimum_allocated_bytes: int
) -> None:
    allocated = execution.get("cuda_peak_memory_bytes")
    reserved = execution.get("cuda_peak_reserved_memory_bytes")
    total = execution.get("cuda_device_total_memory_bytes")
    for name, value in (
        ("allocated", allocated),
        ("reserved", reserved),
        ("total", total),
    ):
        require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"CUDA {name} memory evidence is invalid",
        )
    require(
        allocated >= minimum_allocated_bytes,
        "CUDA artifact did not meet its allocation target",
    )
    require(allocated <= reserved <= total, "CUDA memory evidence is inconsistent")
    for name, observed, expected in (
        (
            "allocated",
            execution.get("cuda_peak_allocated_fraction"),
            allocated / total,
        ),
        ("reserved", execution.get("cuda_peak_reserved_fraction"), reserved / total),
    ):
        require(
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and math.isfinite(float(observed))
            and math.isclose(float(observed), expected, abs_tol=1e-6),
            f"CUDA {name} memory fraction is invalid",
        )


def compare(cpu: dict[str, Any], gpu: dict[str, Any], minimum_agreement: float) -> dict[str, Any]:
    require(cpu["execution"]["device"] == "cpu", "First artifact is not CPU-only")
    require(str(gpu["execution"]["device"]).startswith("cuda:"), "Second artifact is not CUDA")
    require(cpu["dataset"] == gpu["dataset"], "Dataset metadata differs")
    require(cpu["model"] == gpu["model"], "Model parameters differ")
    require(0 <= minimum_agreement <= 1, "Minimum agreement must be in [0, 1]")
    memory_target = GPU_MEMORY_TARGETS.get(gpu["poc_id"])
    if memory_target is not None:
        validate_gpu_memory(gpu["execution"], memory_target)

    class_matches = 0
    total_score_difference = 0.0
    maximum_score_difference = 0.0
    score_count = 0
    for cpu_prediction, gpu_prediction in zip(
        cpu["predictions"], gpu["predictions"], strict=True
    ):
        class_matches += int(
            cpu_prediction["predicted_class"] == gpu_prediction["predicted_class"]
        )
        for cpu_score, gpu_score in zip(
            cpu_prediction["scores"], gpu_prediction["scores"], strict=True
        ):
            difference = abs(float(cpu_score) - float(gpu_score))
            require(math.isfinite(difference), "Score difference is not finite")
            total_score_difference += difference
            maximum_score_difference = max(maximum_score_difference, difference)
            score_count += 1

    prediction_count = len(cpu["predictions"])
    agreement = class_matches / prediction_count
    require(agreement >= minimum_agreement, "CPU and GPU prediction agreement is too low")
    cpu_execution = cpu["execution"]
    gpu_execution = gpu["execution"]
    metric_names = (
        "accuracy",
        "initial_loss",
        "final_loss",
        "initial_training_loss",
        "final_training_loss",
        "best_stopping_loss",
        "best_validation_loss",
        "validation_accuracy",
        "test_accuracy",
        "best_epoch",
        "epochs_completed",
        "training_seconds",
        "model_parameters",
    )
    cpu_training_seconds = cpu_execution.get("training_seconds")
    gpu_training_seconds = gpu_execution.get("training_seconds")
    timing = None
    if cpu_training_seconds is not None or gpu_training_seconds is not None:
        require(
            isinstance(cpu_training_seconds, (int, float))
            and float(cpu_training_seconds) > 0,
            "CPU training time is invalid",
        )
        require(
            isinstance(gpu_training_seconds, (int, float))
            and float(gpu_training_seconds) > 0,
            "GPU training time is invalid",
        )
        timing = {
            "cpu_training_seconds": cpu_training_seconds,
            "gpu_training_seconds": gpu_training_seconds,
            "cpu_over_gpu_speedup": round(
                float(cpu_training_seconds) / float(gpu_training_seconds), 3
            ),
        }
    return {
        "status": "PASS",
        "generated_at": utc_now(),
        "dataset": cpu["dataset"]["name"],
        "dataset_identity": cpu["dataset"],
        "model": cpu["model"],
        "nodes": cpu["dataset"]["nodes"],
        "classes": cpu["dataset"]["classes"],
        "predictions_compared": prediction_count,
        "class_matches": class_matches,
        "class_agreement": round(agreement, 6),
        "minimum_agreement": minimum_agreement,
        "mean_absolute_score_difference": round(
            total_score_difference / score_count, 9
        ),
        "maximum_absolute_score_difference": round(maximum_score_difference, 9),
        "model_parameters_match": True,
        "framework_releases_match": {
            "python": cpu_execution.get("python") == gpu_execution.get("python"),
            "torch": str(cpu_execution.get("torch", "")).split("+")[0]
            == str(gpu_execution.get("torch", "")).split("+")[0],
            "torch_geometric": cpu_execution.get("torch_geometric")
            == gpu_execution.get("torch_geometric"),
        },
        "frameworks": {
            "cpu": {
                key: cpu_execution.get(key)
                for key in ("python", "torch", "torch_geometric", "torch_cuda")
            },
            "gpu": {
                key: gpu_execution.get(key)
                for key in ("python", "torch", "torch_geometric", "torch_cuda")
            },
        },
        "timing": timing,
        "cpu": {
            "poc_id": cpu["poc_id"],
            "device": cpu_execution["device"],
            "cpu_model": cpu_execution["cpu_model"],
            "cpu_count": cpu_execution.get("cpu_count"),
            "cuda_available": cpu_execution.get("cuda_available"),
            "source_revision": cpu_execution["source_revision"],
            "execution_strategy": {
                key: cpu_execution[key]
                for key in (
                    "aggregation",
                    "edge_chunk_size",
                    "activation_checkpointing",
                    "output_checkpointing",
                    "best_state_on_cpu",
                    "destination_node_chunk_size",
                    "precision",
                    "saved_tensors_on_cpu",
                )
                if key in cpu_execution
            },
            "metrics": {
                key: cpu_execution[key]
                for key in metric_names
                if key in cpu_execution
            },
        },
        "gpu": {
            "poc_id": gpu["poc_id"],
            "device": gpu_execution["device"],
            "cuda_device_name": gpu_execution["cuda_device_name"],
            "cuda_capability": gpu_execution.get("cuda_capability"),
            "cuda_peak_memory_bytes": gpu_execution.get("cuda_peak_memory_bytes"),
            "cuda_peak_reserved_memory_bytes": gpu_execution.get(
                "cuda_peak_reserved_memory_bytes"
            ),
            "cuda_device_total_memory_bytes": gpu_execution.get(
                "cuda_device_total_memory_bytes"
            ),
            "cuda_peak_allocated_fraction": gpu_execution.get(
                "cuda_peak_allocated_fraction"
            ),
            "cuda_peak_reserved_fraction": gpu_execution.get(
                "cuda_peak_reserved_fraction"
            ),
            "source_revision": gpu_execution["source_revision"],
            "execution_strategy": {
                key: gpu_execution[key]
                for key in (
                    "aggregation",
                    "edge_chunk_size",
                    "activation_checkpointing",
                    "output_checkpointing",
                    "best_state_on_cpu",
                    "destination_node_chunk_size",
                    "precision",
                    "saved_tensors_on_cpu",
                )
                if key in gpu_execution
            },
            "metrics": {
                key: gpu_execution[key]
                for key in metric_names
                if key in gpu_execution
            },
        },
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_artifact", type=Path)
    parser.add_argument("gpu_artifact", type=Path)
    parser.add_argument("--minimum-agreement", type=float, default=0.95)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--verify-kaggle-status",
        action="store_true",
        help="Require the registered immutable Kaggle kernel versions to be COMPLETE",
    )
    parser.add_argument(
        "--cpu-resource-usage",
        type=Path,
        help="Optional wait4 resource JSON from the CPU-only runner",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    cpu, cpu_digest = load_known_artifact(arguments.cpu_artifact)
    gpu, gpu_digest = load_known_artifact(arguments.gpu_artifact)
    result = compare(cpu, gpu, arguments.minimum_agreement)
    memory_pair = gpu["poc_id"] in GPU_MEMORY_TARGETS
    if memory_pair:
        require(
            arguments.cpu_resource_usage is not None,
            "Memory benchmark requires CPU resource evidence",
        )
    result["cpu"]["artifact_sha256"] = cpu_digest
    result["gpu"]["artifact_sha256"] = gpu_digest
    result["kaggle_runs"] = {
        "cpu": kaggle_run_evidence(
            cpu["poc_id"], verify_remote_status=arguments.verify_kaggle_status
        ),
        "gpu": kaggle_run_evidence(
            gpu["poc_id"], verify_remote_status=arguments.verify_kaggle_status
        ),
    }
    cpu_run = result["kaggle_runs"]["cpu"]
    gpu_run = result["kaggle_runs"]["gpu"]
    result["proof"] = {
        "artifact_checksums_and_schema_valid": True,
        "source_revisions_valid": True,
        "dataset_identity_match": True,
        "model_configuration_match": True,
        "cpu_only": (
            result["cpu"]["device"] == "cpu"
            and result["cpu"]["cuda_available"] is False
            and cpu_run["enable_gpu"] is False
            and cpu_run["machine_shape"] is None
        ),
        "gpu_cuda": (
            str(result["gpu"]["device"]).startswith("cuda:")
            and bool(result["gpu"]["cuda_device_name"])
            and gpu_run["enable_gpu"] is True
            and gpu_run["machine_shape"] == "NvidiaTeslaT4"
        ),
        "immutable_kaggle_versions_complete": (
            cpu_run["status"] == "COMPLETE" and gpu_run["status"] == "COMPLETE"
        ),
    }
    if arguments.cpu_resource_usage is not None:
        result["cpu"]["resource_usage"] = load_cpu_resource_evidence(
            arguments.cpu_resource_usage
        )
    if memory_pair:
        result["proof"]["gpu_memory_target_met"] = True
        result["proof"]["cpu_resource_measurement_present"] = True
    local_proof_fields = {
        key: value
        for key, value in result["proof"].items()
        if key != "immutable_kaggle_versions_complete"
    }
    require(all(local_proof_fields.values()), "Kaggle CPU/GPU proof is incomplete")
    if arguments.verify_kaggle_status:
        require(
            result["proof"]["immutable_kaggle_versions_complete"],
            "Kaggle version status proof is incomplete",
        )
    result["proof_status"] = (
        "PASS"
        if all(result["proof"].values())
        else "REMOTE_STATUS_NOT_CHECKED"
    )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is not None:
        require(
            arguments.force or not arguments.output.exists(),
            f"Output already exists: {arguments.output}",
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf8")
    print(payload, end="")
    return 0


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (
        ProofError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        print(f"KAGGLE COMPARISON FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
