from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "poc"))

from poc_runtime import ProofError  # noqa: E402
from result_artifact import (  # noqa: E402
    ArtifactSpec,
    artifact_from_logits,
    load_and_validate_artifact,
    write_artifact,
)


SPEC = ArtifactSpec(
    poc_id="test-cuda-v1",
    target_poc_id="test-local-v1",
    dataset_name="Tiny",
    nodes=2,
    classes=2,
    minimum_accuracy=0.5,
    identity={"revision": "fixed"},
)


def valid_artifact() -> dict:
    return artifact_from_logits(
        spec=SPEC,
        logits=torch.tensor([[3.0, 1.0], [1.0, 3.0]]),
        model={"type": "test"},
        execution={
            "status": "PASS",
            "device": "cuda:0",
            "cuda_device_name": "Test GPU",
            "accuracy": 1.0,
        },
    )


class ArtifactTests(unittest.TestCase):
    def test_round_trip_validates_checksum_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            _, expected_digest = write_artifact(path, valid_artifact())
            artifact, digest = load_and_validate_artifact(path, SPEC)
            self.assertEqual(digest, expected_digest)
            self.assertEqual(len(artifact["predictions"]), 2)

    def test_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_artifact(path, valid_artifact())
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ProofError, "SHA-256 mismatch"):
                load_and_validate_artifact(path, SPEC)

    def test_non_cuda_execution_is_rejected_after_valid_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            artifact = valid_artifact()
            artifact["execution"]["device"] = "cpu"
            write_artifact(path, artifact)
            with self.assertRaisesRegex(ProofError, "did not run on CUDA"):
                load_and_validate_artifact(path, SPEC)

    def test_wrong_source_revision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            artifact = valid_artifact()
            artifact["execution"]["source_revision"] = "a" * 40
            write_artifact(path, artifact)
            strict_spec = replace(SPEC, source_revision="b" * 40)
            with self.assertRaisesRegex(ProofError, "source revision"):
                load_and_validate_artifact(path, strict_spec)

    def test_inconsistent_prediction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            artifact = valid_artifact()
            artifact["predictions"][0]["predicted_class"] = 1
            write_artifact(path, artifact)
            with self.assertRaisesRegex(ProofError, "not the score argmax"):
                load_and_validate_artifact(path, SPEC)

    def test_invalid_generation_time_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            artifact = valid_artifact()
            artifact["generated_at"] = "not-a-time"
            write_artifact(path, artifact)
            with self.assertRaisesRegex(ProofError, "generation time"):
                load_and_validate_artifact(path, SPEC)

    def test_non_finite_values_cannot_be_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            artifact = valid_artifact()
            artifact["execution"]["accuracy"] = float("nan")
            with self.assertRaises(ValueError):
                write_artifact(path, artifact)

    def test_checksum_filename_is_bound_to_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            checksum_path, _ = write_artifact(path, valid_artifact())
            digest = checksum_path.read_text(encoding="utf8").split()[0]
            checksum_path.write_text(f"{digest}  different.json\n", encoding="utf8")
            with self.assertRaisesRegex(ProofError, "filename does not match"):
                load_and_validate_artifact(path, SPEC)


if __name__ == "__main__":
    unittest.main()
