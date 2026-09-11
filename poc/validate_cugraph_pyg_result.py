#!/usr/bin/env python3
"""Validate and compact the Kaggle cuGraph-PyG integration evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from compare_kaggle_results import kaggle_run_evidence
from kaggle_specs import CUGRAPH_PYG_FLICKR_KAGGLE_SPEC
from proof_common import require
from result_artifact import load_and_validate_artifact, utc_now


EXPECTED_CUGRAPH_PYG_VERSION = "26.8.0"


def validate_execution(artifact: dict[str, Any]) -> dict[str, Any]:
    execution = artifact["execution"]
    require("T4" in execution["cuda_device_name"], "The run did not use a T4")
    require(execution.get("cuda_capability") == [7, 5], "Unexpected CUDA capability")
    require(execution.get("cuda_device_count") == 1, "The run was not single-GPU")
    require(
        execution.get("cugraph_pyg") == EXPECTED_CUGRAPH_PYG_VERSION,
        "Unexpected cuGraph-PyG version",
    )
    require(
        str(execution.get("loader_module", "")).startswith("cugraph_pyg.loader"),
        "The recorded loader was not cuGraph-PyG",
    )
    require(
        str(execution.get("graph_store_module", "")).startswith("cugraph_pyg.data"),
        "The recorded graph store was not cuGraph-PyG",
    )
    require(
        str(execution.get("feature_store_module", "")).startswith("cugraph_pyg.data"),
        "The recorded feature store was not cuGraph-PyG",
    )
    require(execution.get("cugraph_comms_initialized") is True, "cuGraph comms missing")
    require(execution.get("distributed_backend") == "nccl", "NCCL was not active")
    require(execution.get("distributed_world_size") == 1, "World size must be one")
    for field in (
        "batches_processed",
        "seed_nodes_processed",
        "sampled_nodes_processed",
        "sampled_edges_processed",
    ):
        require(
            isinstance(execution.get(field), int) and execution[field] > 0,
            f"Invalid {field} evidence",
        )
    for field in (
        "training_seconds",
        "sampling_seconds",
        "optimization_seconds",
        "validation_seconds",
    ):
        value = execution.get(field)
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value > 0,
            f"Invalid {field} evidence",
        )
    return execution


def compact_result(
    artifact: dict[str, Any], digest: str, *, verify_remote_status: bool
) -> dict[str, Any]:
    execution = validate_execution(artifact)
    run = kaggle_run_evidence(
        CUGRAPH_PYG_FLICKR_KAGGLE_SPEC.poc_id,
        verify_remote_status=verify_remote_status,
    )
    immutable_complete = run["status"] == "COMPLETE"
    proof = {
        "artifact_checksum_and_schema_valid": True,
        "single_t4_cuda": True,
        "pytorch_geometric_model": artifact["model"].get("framework")
        == "PyTorch Geometric",
        "cugraph_pyg_loader_and_stores": True,
        "cugraph_comms_and_nccl_initialized": True,
        "flickr_accuracy_threshold_met": execution["test_accuracy"]
        >= CUGRAPH_PYG_FLICKR_KAGGLE_SPEC.minimum_accuracy,
        "immutable_kaggle_version_complete": immutable_complete,
    }
    require(all(proof.values()), "The cuGraph-PyG integration proof is incomplete")
    return {
        "status": "PASS",
        "proof_status": "PASS",
        "generated_at": utc_now(),
        "workload": "flickr-cugraph-pyg",
        "poc_id": artifact["poc_id"],
        "artifact_sha256": digest,
        "dataset": artifact["dataset"],
        "model": artifact["model"],
        "frameworks": {
            key: execution[key]
            for key in (
                "python",
                "torch",
                "torch_cuda",
                "torch_geometric",
                "cugraph_pyg",
                "pylibcugraph",
                "pylibwholegraph",
            )
        },
        "execution": {
            key: execution[key]
            for key in (
                "device",
                "cuda_device_name",
                "cuda_capability",
                "cuda_device_count",
                "cuda_peak_memory_bytes",
                "cuda_peak_reserved_memory_bytes",
                "cuda_device_total_memory_bytes",
                "loader_class",
                "loader_module",
                "graph_store_class",
                "graph_store_module",
                "feature_store_class",
                "feature_store_module",
                "distributed_backend",
                "distributed_world_size",
                "epochs_completed",
                "best_epoch",
                "model_parameters",
                "batches_processed",
                "seed_nodes_processed",
                "sampled_nodes_processed",
                "sampled_edges_processed",
                "sampling_seconds",
                "optimization_seconds",
                "validation_seconds",
                "training_seconds",
                "validation_accuracy",
                "test_accuracy",
                "source_revision",
            )
        },
        "kaggle_run": run,
        "proof": proof,
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/kaggle-flickr-cugraph-pyg-t4.json"),
    )
    parser.add_argument("--verify-kaggle-status", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    artifact, digest = load_and_validate_artifact(
        arguments.artifact, CUGRAPH_PYG_FLICKR_KAGGLE_SPEC
    )
    result = compact_result(
        artifact, digest, verify_remote_status=arguments.verify_kaggle_status
    )
    require(arguments.force or not arguments.output.exists(), "Output already exists")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
