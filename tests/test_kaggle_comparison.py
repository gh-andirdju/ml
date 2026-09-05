from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "poc"))

from compare_kaggle_results import compare  # noqa: E402
from poc_runtime import ProofError  # noqa: E402


def artifact(device: str) -> dict:
    execution = {
        "status": "PASS",
        "device": device,
        "accuracy": 1.0,
        "source_revision": "a" * 40,
        "python": "3.12.0",
        "torch": "2.10.0",
        "torch_geometric": "2.8.0",
        "torch_cuda": "12.8" if device.startswith("cuda:") else None,
    }
    if device == "cpu":
        execution.update({"cpu_model": "Test CPU", "cuda_available": False})
    else:
        execution["cuda_device_name"] = "Test GPU"
    return {
        "poc_id": "cpu" if device == "cpu" else "gpu",
        "dataset": {"name": "Tiny", "nodes": 2, "classes": 2},
        "model": {"type": "GCN", "seed": 42},
        "execution": execution,
        "predictions": [
            {"node_id": 0, "predicted_class": 0, "scores": [0.8, 0.2]},
            {"node_id": 1, "predicted_class": 1, "scores": [0.1, 0.9]},
        ],
    }


class KaggleComparisonTests(unittest.TestCase):
    def test_matching_predictions_pass(self) -> None:
        result = compare(artifact("cpu"), artifact("cuda:0"), 0.95)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["class_agreement"], 1.0)
        self.assertEqual(result["mean_absolute_score_difference"], 0.0)

    def test_low_prediction_agreement_fails(self) -> None:
        gpu = artifact("cuda:0")
        gpu["predictions"][0] = {
            "node_id": 0,
            "predicted_class": 1,
            "scores": [0.2, 0.8],
        }
        with self.assertRaisesRegex(ProofError, "agreement is too low"):
            compare(artifact("cpu"), gpu, 0.95)

    def test_model_difference_fails(self) -> None:
        gpu = artifact("cuda:0")
        gpu["model"] = deepcopy(gpu["model"])
        gpu["model"]["seed"] = 7
        with self.assertRaisesRegex(ProofError, "Model parameters differ"):
            compare(artifact("cpu"), gpu, 0.95)


if __name__ == "__main__":
    unittest.main()
