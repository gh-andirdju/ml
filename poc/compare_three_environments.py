#!/usr/bin/env python3
"""Compare one model-identical GNN workload across MPS, CPU, and CUDA."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from compare_kaggle_results import (
    compare,
    kaggle_run_evidence,
    load_cpu_resource_evidence,
    load_known_artifact,
)
from kaggle_specs import (
    FLICKR_2048_KAGGLE_CPU_SPEC,
    FLICKR_2048_KAGGLE_CUDA_SPEC,
    FLICKR_2048_MPS_SPEC,
    FLICKR_4096_KAGGLE_CPU_SPEC,
    FLICKR_4096_KAGGLE_CUDA_SPEC,
    FLICKR_4096_MPS_SPEC,
    FLICKR_KAGGLE_CPU_SPEC,
    FLICKR_KAGGLE_CUDA_SPEC,
    FLICKR_MPS_SPEC,
    FLICKR_WIDE_KAGGLE_CPU_SPEC,
    FLICKR_WIDE_KAGGLE_CUDA_SPEC,
    FLICKR_WIDE_MPS_SPEC,
    KARATE_KAGGLE_CPU_SPEC,
    KARATE_KAGGLE_SPEC,
    KARATE_MPS_SPEC,
    WIKICS_KAGGLE_CPU_SPEC,
    WIKICS_KAGGLE_SPEC,
    WIKICS_MPS_SPEC,
)
from proof_common import ProofError, require
from result_artifact import utc_now


TRIPLETS = {
    KARATE_MPS_SPEC.poc_id: (
        "karate",
        KARATE_KAGGLE_CPU_SPEC.poc_id,
        KARATE_KAGGLE_SPEC.poc_id,
    ),
    WIKICS_MPS_SPEC.poc_id: (
        "wikics",
        WIKICS_KAGGLE_CPU_SPEC.poc_id,
        WIKICS_KAGGLE_SPEC.poc_id,
    ),
    FLICKR_MPS_SPEC.poc_id: (
        "flickr",
        FLICKR_KAGGLE_CPU_SPEC.poc_id,
        FLICKR_KAGGLE_CUDA_SPEC.poc_id,
    ),
    FLICKR_WIDE_MPS_SPEC.poc_id: (
        "flickr-wide",
        FLICKR_WIDE_KAGGLE_CPU_SPEC.poc_id,
        FLICKR_WIDE_KAGGLE_CUDA_SPEC.poc_id,
    ),
    FLICKR_2048_MPS_SPEC.poc_id: (
        "flickr-2048",
        FLICKR_2048_KAGGLE_CPU_SPEC.poc_id,
        FLICKR_2048_KAGGLE_CUDA_SPEC.poc_id,
    ),
    FLICKR_4096_MPS_SPEC.poc_id: (
        "flickr-4096",
        FLICKR_4096_KAGGLE_CPU_SPEC.poc_id,
        FLICKR_4096_KAGGLE_CUDA_SPEC.poc_id,
    ),
}


def prediction_comparison(
    left: dict[str, Any], right: dict[str, Any], minimum_agreement: float
) -> dict[str, Any]:
    require(0 <= minimum_agreement <= 1, "Minimum agreement must be in [0, 1]")
    require(left["dataset"] == right["dataset"], "Dataset metadata differs")
    require(left["model"] == right["model"], "Model parameters differ")
    require(
        len(left["predictions"]) == len(right["predictions"]),
        "Prediction counts differ",
    )
    class_matches = 0
    score_count = 0
    total_difference = 0.0
    maximum_difference = 0.0
    for left_prediction, right_prediction in zip(
        left["predictions"], right["predictions"], strict=True
    ):
        require(
            left_prediction["node_id"] == right_prediction["node_id"],
            "Prediction node IDs differ",
        )
        class_matches += int(
            left_prediction["predicted_class"]
            == right_prediction["predicted_class"]
        )
        for left_score, right_score in zip(
            left_prediction["scores"], right_prediction["scores"], strict=True
        ):
            difference = abs(float(left_score) - float(right_score))
            require(math.isfinite(difference), "Score difference is not finite")
            total_difference += difference
            maximum_difference = max(maximum_difference, difference)
            score_count += 1
    prediction_count = len(left["predictions"])
    agreement = class_matches / prediction_count
    require(agreement >= minimum_agreement, "Prediction agreement is too low")
    return {
        "predictions_compared": prediction_count,
        "class_matches": class_matches,
        "class_agreement": round(agreement, 6),
        "minimum_agreement": minimum_agreement,
        "mean_absolute_score_difference": round(total_difference / score_count, 9),
        "maximum_absolute_score_difference": round(maximum_difference, 9),
    }


def build_comparison(
    mps: dict[str, Any],
    cpu: dict[str, Any],
    gpu: dict[str, Any],
    minimum_agreement: float,
) -> dict[str, Any]:
    triplet = TRIPLETS.get(mps["poc_id"])
    require(triplet is not None, "MPS artifact is not registered for comparison")
    workload, expected_cpu_id, expected_gpu_id = triplet
    require(cpu["poc_id"] == expected_cpu_id, "CPU artifact is from another workload")
    require(gpu["poc_id"] == expected_gpu_id, "GPU artifact is from another workload")
    require(str(mps["execution"]["device"]).startswith("mps"), "MPS device is invalid")
    require(mps["execution"]["mps_available"] is True, "MPS was unavailable")
    require(
        mps["execution"]["mps_fallback_enabled"] is False,
        "MPS CPU fallback was enabled",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", mps["execution"]["source_revision"])
        is not None,
        "MPS source revision is invalid",
    )
    cpu_gpu = compare(cpu, gpu, minimum_agreement)
    pairwise = {
        "mps_vs_kaggle_cpu": prediction_comparison(mps, cpu, minimum_agreement),
        "mps_vs_kaggle_gpu": prediction_comparison(mps, gpu, minimum_agreement),
        "kaggle_cpu_vs_gpu": {
            key: cpu_gpu[key]
            for key in (
                "predictions_compared",
                "class_matches",
                "class_agreement",
                "minimum_agreement",
                "mean_absolute_score_difference",
                "maximum_absolute_score_difference",
            )
        },
    }
    environments = {
        "mps": {
            "poc_id": mps["poc_id"],
            "device": mps["execution"]["device"],
            "hardware": mps["execution"]["hardware"],
            "mps_available": mps["execution"]["mps_available"],
            "mps_fallback_enabled": mps["execution"]["mps_fallback_enabled"],
            "source_revision": mps["execution"]["source_revision"],
            "execution_strategy": {
                key: mps["execution"][key]
                for key in (
                    "aggregation",
                    "edge_chunk_size",
                    "activation_checkpointing",
                )
                if key in mps["execution"]
            },
            "maximum_process_rss_bytes": mps["execution"].get(
                "maximum_process_rss_bytes"
            ),
            "metrics": {
                key: mps["execution"][key]
                for key in (
                    "accuracy",
                    "validation_accuracy",
                    "test_accuracy",
                    "training_seconds",
                    "epochs_completed",
                    "best_epoch",
                    "model_parameters",
                )
                if key in mps["execution"]
            },
        },
        "kaggle_cpu": cpu_gpu["cpu"],
        "kaggle_gpu": cpu_gpu["gpu"],
    }
    timing = None
    times = {
        name: environment["metrics"].get("training_seconds")
        for name, environment in environments.items()
    }
    if all(isinstance(value, (int, float)) and value > 0 for value in times.values()):
        timing = {
            **times,
            "cpu_over_mps_speedup": round(times["kaggle_cpu"] / times["mps"], 3),
            "mps_over_gpu_speedup": round(times["mps"] / times["kaggle_gpu"], 3),
            "cpu_over_gpu_speedup": round(
                times["kaggle_cpu"] / times["kaggle_gpu"], 3
            ),
        }
    return {
        "status": "PASS",
        "generated_at": utc_now(),
        "workload": workload,
        "dataset": mps["dataset"],
        "model": mps["model"],
        "frameworks": {
            name: {
                key: artifact["execution"].get(key)
                for key in ("python", "torch", "torch_geometric", "torch_cuda")
            }
            for name, artifact in (
                ("mps", mps),
                ("kaggle_cpu", cpu),
                ("kaggle_gpu", gpu),
            )
        },
        "environments": environments,
        "pairwise": pairwise,
        "timing": timing,
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mps_artifact", type=Path)
    parser.add_argument("cpu_artifact", type=Path)
    parser.add_argument("gpu_artifact", type=Path)
    parser.add_argument("--minimum-agreement", type=float, default=0.95)
    parser.add_argument("--cpu-resource-usage", type=Path)
    parser.add_argument("--verify-kaggle-status", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    mps, mps_digest = load_known_artifact(arguments.mps_artifact)
    cpu, cpu_digest = load_known_artifact(arguments.cpu_artifact)
    gpu, gpu_digest = load_known_artifact(arguments.gpu_artifact)
    result = build_comparison(mps, cpu, gpu, arguments.minimum_agreement)
    result["artifacts"] = {
        "mps_sha256": mps_digest,
        "kaggle_cpu_sha256": cpu_digest,
        "kaggle_gpu_sha256": gpu_digest,
    }
    result["kaggle_runs"] = {
        "cpu": kaggle_run_evidence(
            cpu["poc_id"], verify_remote_status=arguments.verify_kaggle_status
        ),
        "gpu": kaggle_run_evidence(
            gpu["poc_id"], verify_remote_status=arguments.verify_kaggle_status
        ),
    }
    memory_intensive = mps["poc_id"] in {
        FLICKR_WIDE_MPS_SPEC.poc_id,
        FLICKR_2048_MPS_SPEC.poc_id,
        FLICKR_4096_MPS_SPEC.poc_id,
    }
    if memory_intensive:
        require(
            arguments.cpu_resource_usage is not None,
            "Memory-intensive workload requires CPU resource evidence",
        )
    if arguments.cpu_resource_usage is not None:
        result["environments"]["kaggle_cpu"]["resource_usage"] = (
            load_cpu_resource_evidence(arguments.cpu_resource_usage)
        )
    remote_complete = all(
        run["status"] == "COMPLETE" for run in result["kaggle_runs"].values()
    )
    result["proof"] = {
        "all_artifact_checksums_and_schemas_valid": True,
        "exact_workload_triplet": True,
        "dataset_and_model_match": True,
        "mps_without_cpu_fallback": True,
        "all_pairwise_agreements_pass": True,
        "kaggle_cpu_only": result["kaggle_runs"]["cpu"]["enable_gpu"] is False,
        "kaggle_single_t4": (
            result["kaggle_runs"]["gpu"]["machine_shape"] == "NvidiaTeslaT4"
        ),
        "immutable_kaggle_versions_complete": remote_complete,
    }
    if memory_intensive:
        result["proof"]["cpu_resource_evidence_present"] = True
        result["proof"]["gpu_memory_target_met"] = True
    local_fields = {
        key: value
        for key, value in result["proof"].items()
        if key != "immutable_kaggle_versions_complete"
    }
    require(all(local_fields.values()), "Three-environment proof is incomplete")
    if arguments.verify_kaggle_status:
        require(remote_complete, "Kaggle versions are not complete")
    result["proof_status"] = (
        "PASS" if all(result["proof"].values()) else "REMOTE_STATUS_NOT_CHECKED"
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
        KeyError,
        ProofError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        print(f"THREE-ENVIRONMENT COMPARISON FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
