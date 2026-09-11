#!/usr/bin/env python3
"""Validate the matched plain-PyG and cuGraph-PyG full-batch T4 artifacts."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from compare_kaggle_results import kaggle_run_evidence
from kaggle_specs import (
    FLICKR_FULLBATCH_CUGRAPH_PYG_KAGGLE_SPEC,
    FLICKR_FULLBATCH_PYG_KAGGLE_SPEC,
)
from proof_common import ProofError, require
from result_artifact import load_and_validate_artifact, utc_now


MINIMUM_AGREEMENT = 0.95
FRAMEWORK_FIELDS = (
    "python",
    "torch",
    "torch_cuda",
    "torch_geometric",
    "cugraph_pyg",
    "pylibcugraph",
    "pylibwholegraph",
)


def positive_number(execution: dict[str, Any], name: str) -> float:
    value = execution.get(name)
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0,
        f"Invalid {name}",
    )
    return float(value)


def validate_execution(
    artifact: dict[str, Any], *, expected_backend: str, cugraph: bool
) -> dict[str, Any]:
    execution = artifact["execution"]
    require(execution.get("backend") == expected_backend, "Backend identity differs")
    require(execution.get("cuda_device_count") == 1, "Run was not single-GPU")
    require("T4" in execution.get("cuda_device_name", ""), "Run did not use a T4")
    require(execution.get("cuda_capability") == [7, 5], "Capability is not 7.5")
    require(execution.get("exact_full_graph_verified") is True, "Graph is not exact")
    require(execution.get("model_parameters") == 391_175, "Model size differs")
    positive_number(execution, "training_seconds")
    positive_number(execution, "backend_preparation_seconds")
    positive_number(execution, "backend_inclusive_seconds")
    for field in (
        "cuda_peak_memory_bytes",
        "cuda_peak_reserved_memory_bytes",
        "cuda_device_total_memory_bytes",
    ):
        require(
            isinstance(execution.get(field), int)
            and not isinstance(execution[field], bool)
            and execution[field] > 0,
            f"Invalid {field}",
        )
    if cugraph:
        require(
            execution.get("cugraph_comms_initialized") is True,
            "cuGraph communicator was not initialized",
        )
        require(execution.get("full_neighbor_fanout") == [-1], "Fanout is not full")
        require(execution.get("materialized_batches") == 1, "Not exactly one batch")
        require(
            execution.get("local_seeds_per_call") == 89_250,
            "cuGraph seed call does not cover the one-batch workload",
        )
        require(execution.get("distributed_backend") == "nccl", "NCCL missing")
        require(execution.get("distributed_world_size") == 1, "World size differs")
        for field in ("loader_module", "graph_store_module", "feature_store_module"):
            require(
                str(execution.get(field, "")).startswith("cugraph_pyg."),
                f"{field} is not provided by cuGraph-PyG",
            )
    else:
        require(
            execution.get("cugraph_comms_initialized") is False,
            "Plain-PyG run initialized cuGraph",
        )
        require(execution.get("materialized_batches") == 0, "Plain run used a loader")
    return execution


def compare(
    plain: dict[str, Any], cugraph: dict[str, Any], plain_digest: str, cugraph_digest: str
) -> dict[str, Any]:
    require(plain["dataset"] == cugraph["dataset"], "Dataset metadata differs")
    require(plain["model"] == cugraph["model"], "Model configuration differs")
    plain_execution = validate_execution(
        plain, expected_backend="pyg-direct-full-batch", cugraph=False
    )
    cugraph_execution = validate_execution(
        cugraph, expected_backend="cugraph-pyg-full-neighbor", cugraph=True
    )
    require(
        plain_execution["source_revision"] == cugraph_execution["source_revision"],
        "Source revisions differ",
    )
    require(
        plain["model"].get("execution_mode") == "full-batch",
        "Model execution mode is not full-batch",
    )
    require(
        all(
            plain_execution.get(field) == cugraph_execution.get(field)
            for field in FRAMEWORK_FIELDS
        ),
        "Framework versions differ",
    )
    matches = 0
    absolute_score_difference = 0.0
    maximum_score_difference = 0.0
    score_count = 0
    for plain_prediction, cugraph_prediction in zip(
        plain["predictions"], cugraph["predictions"], strict=True
    ):
        matches += int(
            plain_prediction["predicted_class"]
            == cugraph_prediction["predicted_class"]
        )
        for plain_score, cugraph_score in zip(
            plain_prediction["scores"], cugraph_prediction["scores"], strict=True
        ):
            difference = abs(float(plain_score) - float(cugraph_score))
            absolute_score_difference += difference
            maximum_score_difference = max(maximum_score_difference, difference)
            score_count += 1
    prediction_count = len(plain["predictions"])
    agreement = matches / prediction_count
    require(agreement >= MINIMUM_AGREEMENT, "Prediction agreement is below 95%")
    plain_training = positive_number(plain_execution, "training_seconds")
    cugraph_training = positive_number(cugraph_execution, "training_seconds")
    plain_preparation = positive_number(
        plain_execution, "backend_preparation_seconds"
    )
    cugraph_preparation = positive_number(
        cugraph_execution, "backend_preparation_seconds"
    )
    plain_total = positive_number(plain_execution, "backend_inclusive_seconds")
    cugraph_total = positive_number(cugraph_execution, "backend_inclusive_seconds")
    require(
        math.isclose(plain_total, plain_training + plain_preparation, abs_tol=2e-6),
        "Plain backend-inclusive timing is inconsistent",
    )
    require(
        math.isclose(
            cugraph_total, cugraph_training + cugraph_preparation, abs_tol=2e-6
        ),
        "cuGraph backend-inclusive timing is inconsistent",
    )
    return {
        "status": "PASS",
        "proof_status": "REMOTE_STATUS_NOT_CHECKED",
        "generated_at": utc_now(),
        "workload": "flickr-fullbatch-pyg-vs-cugraph-pyg-t4",
        "dataset": plain["dataset"],
        "model": plain["model"],
        "frameworks": {field: plain_execution[field] for field in FRAMEWORK_FIELDS},
        "prediction_comparison": {
            "nodes": prediction_count,
            "class_matches": matches,
            "class_agreement": round(agreement, 6),
            "minimum_agreement": MINIMUM_AGREEMENT,
            "mean_absolute_score_difference": round(
                absolute_score_difference / score_count, 9
            ),
            "maximum_absolute_score_difference": round(
                maximum_score_difference, 9
            ),
        },
        "timing": {
            "plain_training_seconds": plain_training,
            "cugraph_training_seconds": cugraph_training,
            "cugraph_over_plain_training_ratio": round(
                cugraph_training / plain_training, 3
            ),
            "plain_backend_preparation_seconds": plain_preparation,
            "cugraph_backend_preparation_seconds": cugraph_preparation,
            "plain_backend_inclusive_seconds": plain_total,
            "cugraph_backend_inclusive_seconds": cugraph_total,
            "cugraph_over_plain_backend_inclusive_ratio": round(
                cugraph_total / plain_total, 3
            ),
        },
        "plain_pyg": {
            "poc_id": plain["poc_id"],
            "artifact_sha256": plain_digest,
            "execution": plain_execution,
        },
        "cugraph_pyg": {
            "poc_id": cugraph["poc_id"],
            "artifact_sha256": cugraph_digest,
            "execution": cugraph_execution,
        },
        "proof": {
            "artifact_checksums_and_schema_valid": True,
            "same_dataset_model_source_and_frameworks": True,
            "single_t4_cuda_both": True,
            "exact_full_graph_both": True,
            "cugraph_loader_store_and_nccl_verified": True,
            "prediction_agreement_met": True,
            "immutable_kaggle_versions_complete": False,
        },
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plain_artifact", type=Path)
    parser.add_argument("cugraph_artifact", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/kaggle-flickr-fullbatch-pyg-vs-cugraph-pyg-t4.json"),
    )
    parser.add_argument("--verify-kaggle-status", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    plain, plain_digest = load_and_validate_artifact(
        arguments.plain_artifact, FLICKR_FULLBATCH_PYG_KAGGLE_SPEC
    )
    cugraph, cugraph_digest = load_and_validate_artifact(
        arguments.cugraph_artifact, FLICKR_FULLBATCH_CUGRAPH_PYG_KAGGLE_SPEC
    )
    result = compare(plain, cugraph, plain_digest, cugraph_digest)
    result["kaggle_runs"] = {
        "plain_pyg": kaggle_run_evidence(
            plain["poc_id"], verify_remote_status=arguments.verify_kaggle_status
        ),
        "cugraph_pyg": kaggle_run_evidence(
            cugraph["poc_id"], verify_remote_status=arguments.verify_kaggle_status
        ),
    }
    complete = all(
        run["status"] == "COMPLETE" for run in result["kaggle_runs"].values()
    )
    result["proof"]["immutable_kaggle_versions_complete"] = complete
    if arguments.verify_kaggle_status:
        require(complete, "Kaggle versions are not both COMPLETE")
    result["proof_status"] = "PASS" if all(result["proof"].values()) else result[
        "proof_status"
    ]
    require(
        arguments.force or not arguments.output.exists(),
        f"Output already exists: {arguments.output}",
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
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
        print(f"FULL-BATCH BACKEND COMPARISON FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
