#!/usr/bin/env python3
"""Compare validated Kaggle CPU and CUDA prediction artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kaggle_specs import COMPARISON_SPECS_BY_POC_ID
from proof_common import ProofError, require
from result_artifact import MAX_ARTIFACT_BYTES, load_and_validate_artifact, utc_now


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


def compare(cpu: dict[str, Any], gpu: dict[str, Any], minimum_agreement: float) -> dict[str, Any]:
    require(cpu["execution"]["device"] == "cpu", "First artifact is not CPU-only")
    require(str(gpu["execution"]["device"]).startswith("cuda:"), "Second artifact is not CUDA")
    require(cpu["dataset"] == gpu["dataset"], "Dataset metadata differs")
    require(cpu["model"] == gpu["model"], "Model parameters differ")
    require(0 <= minimum_agreement <= 1, "Minimum agreement must be in [0, 1]")

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
        "validation_accuracy",
        "test_accuracy",
        "best_epoch",
    )
    return {
        "status": "PASS",
        "generated_at": utc_now(),
        "dataset": cpu["dataset"]["name"],
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
        "framework_versions_match": {
            key: cpu_execution.get(key) == gpu_execution.get(key)
            for key in ("python", "torch", "torch_geometric", "torch_cuda")
        },
        "cpu": {
            "poc_id": cpu["poc_id"],
            "device": cpu_execution["device"],
            "cpu_model": cpu_execution["cpu_model"],
            "source_revision": cpu_execution["source_revision"],
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
            "source_revision": gpu_execution["source_revision"],
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    cpu, cpu_digest = load_known_artifact(arguments.cpu_artifact)
    gpu, gpu_digest = load_known_artifact(arguments.gpu_artifact)
    result = compare(cpu, gpu, arguments.minimum_agreement)
    result["cpu"]["artifact_sha256"] = cpu_digest
    result["gpu"]["artifact_sha256"] = gpu_digest
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is not None:
        require(arguments.force or not arguments.output.exists(), f"Output already exists: {arguments.output}")
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf8")
    print(payload, end="")
    return 0


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (ProofError, OSError, RuntimeError, ValueError) as error:
        print(f"KAGGLE COMPARISON FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
