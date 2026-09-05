from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "poc"))

from compare_three_environments import (  # noqa: E402
    build_comparison,
    prediction_comparison,
)
from kaggle_specs import (  # noqa: E402
    KARATE_KAGGLE_CPU_SPEC,
    KARATE_KAGGLE_SPEC,
    KARATE_MPS_SPEC,
)
from proof_common import ProofError  # noqa: E402


def artifact() -> dict:
    return {
        "dataset": {"name": "Tiny", "nodes": 2, "classes": 2},
        "model": {"type": "GCN"},
        "predictions": [
            {"node_id": 0, "predicted_class": 0, "scores": [0.8, 0.2]},
            {"node_id": 1, "predicted_class": 1, "scores": [0.1, 0.9]},
        ],
    }


def environment_artifact(poc_id: str, device: str) -> dict:
    value = artifact()
    value["poc_id"] = poc_id
    value["execution"] = {
        "status": "PASS",
        "device": device,
        "accuracy": 1.0,
        "source_revision": "a" * 40,
        "python": "3.12.0",
        "torch": "2.10.0",
        "torch_geometric": "2.8.0",
        "torch_cuda": "12.8" if device.startswith("cuda") else None,
    }
    if device.startswith("mps"):
        value["execution"].update(
            {
                "hardware": "arm64",
                "mps_available": True,
                "mps_fallback_enabled": False,
            }
        )
    elif device == "cpu":
        value["execution"].update(
            {"cpu_model": "Test CPU", "cpu_count": 4, "cuda_available": False}
        )
    else:
        value["execution"].update(
            {"cuda_device_name": "Tesla T4", "cuda_capability": [7, 5]}
        )
    return value


class ThreeEnvironmentComparisonTests(unittest.TestCase):
    def test_identical_predictions_pass(self) -> None:
        result = prediction_comparison(artifact(), artifact(), 0.95)
        self.assertEqual(result["class_agreement"], 1.0)
        self.assertEqual(result["maximum_absolute_score_difference"], 0.0)

    def test_low_agreement_fails(self) -> None:
        changed = artifact()
        changed["predictions"][0] = {
            "node_id": 0,
            "predicted_class": 1,
            "scores": [0.2, 0.8],
        }
        with self.assertRaisesRegex(ProofError, "agreement is too low"):
            prediction_comparison(artifact(), changed, 0.95)

    def test_model_mismatch_fails(self) -> None:
        changed = deepcopy(artifact())
        changed["model"]["type"] = "GraphSAGE"
        with self.assertRaisesRegex(ProofError, "Model parameters differ"):
            prediction_comparison(artifact(), changed, 0.95)

    def test_registered_three_environment_triplet_passes(self) -> None:
        result = build_comparison(
            environment_artifact(KARATE_MPS_SPEC.poc_id, "mps:0"),
            environment_artifact(KARATE_KAGGLE_CPU_SPEC.poc_id, "cpu"),
            environment_artifact(KARATE_KAGGLE_SPEC.poc_id, "cuda:0"),
            0.95,
        )
        self.assertEqual(result["workload"], "karate")
        self.assertEqual(result["pairwise"]["mps_vs_kaggle_gpu"]["class_agreement"], 1.0)

    def test_execution_workspace_can_differ_without_changing_model(self) -> None:
        mps = environment_artifact(KARATE_MPS_SPEC.poc_id, "mps:0")
        cpu = environment_artifact(KARATE_KAGGLE_CPU_SPEC.poc_id, "cpu")
        gpu = environment_artifact(KARATE_KAGGLE_SPEC.poc_id, "cuda:0")
        for value, chunk_size in ((mps, 32), (cpu, 128), (gpu, 128)):
            value["execution"].update(
                {
                    "aggregation": "exact chunked mean",
                    "edge_chunk_size": chunk_size,
                    "activation_checkpointing": True,
                }
            )
        result = build_comparison(mps, cpu, gpu, 0.95)
        self.assertEqual(
            result["environments"]["mps"]["execution_strategy"]["edge_chunk_size"],
            32,
        )
        self.assertEqual(
            result["environments"]["kaggle_gpu"]["execution_strategy"][
                "edge_chunk_size"
            ],
            128,
        )

    def test_cross_workload_triplet_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProofError, "CPU artifact is from another"):
            build_comparison(
                environment_artifact(KARATE_MPS_SPEC.poc_id, "mps:0"),
                environment_artifact("wrong-cpu", "cpu"),
                environment_artifact(KARATE_KAGGLE_SPEC.poc_id, "cuda:0"),
                0.95,
            )


if __name__ == "__main__":
    unittest.main()
