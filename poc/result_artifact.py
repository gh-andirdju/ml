"""Versioned, checksum-protected prediction artifacts for offline Neo4j import."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from torch import Tensor

from proof_common import ProofError, require


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "gnn-node-predictions"
MAX_ARTIFACT_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class ArtifactSpec:
    poc_id: str
    target_poc_id: str
    dataset_name: str
    nodes: int
    classes: int
    minimum_accuracy: float
    identity: dict[str, Any]
    source_revision: str | None = None


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def artifact_from_logits(
    *,
    spec: ArtifactSpec,
    logits: Tensor,
    model: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    require(tuple(logits.shape) == (spec.nodes, spec.classes), "Artifact logits shape mismatch")
    probabilities = logits.softmax(dim=1)
    predictions = [
        {
            "node_id": node_id,
            "predicted_class": int(probabilities[node_id].argmax()),
            "scores": [float(score) for score in probabilities[node_id].tolist()],
        }
        for node_id in range(spec.nodes)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "poc_id": spec.poc_id,
        "target_poc_id": spec.target_poc_id,
        "generated_at": utc_now(),
        "dataset": {
            "name": spec.dataset_name,
            "nodes": spec.nodes,
            "classes": spec.classes,
            **spec.identity,
        },
        "model": model,
        "execution": execution,
        "predictions": predictions,
    }


def write_artifact(path: Path, artifact: dict[str, Any], *, force: bool = False) -> tuple[Path, str]:
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if not force:
        require(not path.exists(), f"Artifact already exists: {path}")
        require(not checksum_path.exists(), f"Checksum already exists: {checksum_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    checksum_path.write_text(f"{digest}  {path.name}\n", encoding="utf8")
    return checksum_path, digest


def _load_checksum(path: Path, artifact_name: str) -> str:
    require(path.is_file(), f"Checksum file not found: {path}")
    parts = path.read_text(encoding="utf8").strip().split()
    require(len(parts) == 2, "Checksum file must contain a digest and filename")
    digest, filename = parts
    require(filename == artifact_name, "Checksum filename does not match artifact")
    require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Bad SHA-256")
    return digest


def load_and_validate_artifact(
    path: Path,
    spec: ArtifactSpec,
    checksum_path: Path | None = None,
) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"Artifact not found: {path}")
    size = path.stat().st_size
    require(0 < size <= MAX_ARTIFACT_BYTES, "Artifact size is outside the accepted range")
    payload = path.read_bytes()
    expected_digest = _load_checksum(
        checksum_path or path.with_suffix(path.suffix + ".sha256"), path.name
    )
    actual_digest = hashlib.sha256(payload).hexdigest()
    require(actual_digest == expected_digest, "Artifact SHA-256 mismatch")
    try:
        artifact = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ProofError(f"Artifact is not valid JSON: {error}") from error
    require(isinstance(artifact, dict), "Artifact root must be an object")
    require(artifact.get("schema_version") == SCHEMA_VERSION, "Unsupported artifact schema")
    require(artifact.get("artifact_type") == ARTIFACT_TYPE, "Unexpected artifact type")
    require(artifact.get("poc_id") == spec.poc_id, "Unexpected artifact POC ID")
    require(artifact.get("target_poc_id") == spec.target_poc_id, "Unexpected target POC ID")

    dataset = artifact.get("dataset")
    require(isinstance(dataset, dict), "Dataset metadata is missing")
    require(dataset.get("name") == spec.dataset_name, "Unexpected dataset name")
    require(dataset.get("nodes") == spec.nodes, "Unexpected dataset node count")
    require(dataset.get("classes") == spec.classes, "Unexpected dataset class count")
    for key, expected in spec.identity.items():
        require(dataset.get(key) == expected, f"Unexpected dataset identity field: {key}")

    execution = artifact.get("execution")
    require(isinstance(execution, dict), "Execution metadata is missing")
    require(execution.get("status") == "PASS", "GPU execution did not pass")
    require(str(execution.get("device", "")).startswith("cuda:"), "Artifact did not run on CUDA")
    require(bool(str(execution.get("cuda_device_name", "")).strip()), "CUDA device name is missing")
    if spec.source_revision is not None:
        require(
            execution.get("source_revision") == spec.source_revision,
            "Artifact source revision does not match the importer",
        )
    accuracy = execution.get("accuracy")
    require(
        isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool),
        "Accuracy is missing",
    )
    require(math.isfinite(float(accuracy)), "Accuracy is not finite")
    require(float(accuracy) >= spec.minimum_accuracy, "Accuracy is below the proof threshold")

    predictions = artifact.get("predictions")
    require(isinstance(predictions, list), "Predictions must be a list")
    require(len(predictions) == spec.nodes, "Prediction count does not match dataset")
    for expected_id, prediction in enumerate(predictions):
        require(isinstance(prediction, dict), f"Prediction {expected_id} is not an object")
        require(prediction.get("node_id") == expected_id, "Prediction node IDs are incomplete")
        predicted_class = prediction.get("predicted_class")
        require(
            isinstance(predicted_class, int) and not isinstance(predicted_class, bool),
            f"Prediction class is invalid for node {expected_id}",
        )
        require(0 <= predicted_class < spec.classes, f"Prediction class is out of range for node {expected_id}")
        scores = prediction.get("scores")
        require(isinstance(scores, list) and len(scores) == spec.classes, f"Scores are invalid for node {expected_id}")
        require(
            all(
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
                and 0 <= float(score) <= 1
                for score in scores
            ),
            f"Scores are outside [0, 1] for node {expected_id}",
        )
        require(abs(sum(float(score) for score in scores) - 1.0) <= 1e-4, f"Scores do not sum to one for node {expected_id}")
        require(max(range(spec.classes), key=lambda index: scores[index]) == predicted_class, f"Prediction is not the score argmax for node {expected_id}")
    return artifact, actual_digest
