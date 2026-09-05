from __future__ import annotations

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "poc"))

from compare_kaggle_results import (  # noqa: E402
    compare,
    kaggle_run_evidence,
    load_cpu_resource_evidence,
)
from kaggle_specs import KARATE_KAGGLE_CPU_SPEC, KARATE_KAGGLE_SPEC  # noqa: E402
from kaggle_specs import (  # noqa: E402
    FLICKR_2048_KAGGLE_CUDA_SPEC,
    FLICKR_WIDE_KAGGLE_CUDA_SPEC,
)
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

    def test_timed_comparison_reports_speedup(self) -> None:
        cpu = artifact("cpu")
        gpu = artifact("cuda:0")
        cpu["execution"]["training_seconds"] = 20.0
        gpu["execution"]["training_seconds"] = 4.0
        gpu["execution"]["cuda_peak_reserved_memory_bytes"] = 6_000_000_000
        gpu["execution"]["cuda_device_total_memory_bytes"] = 16_000_000_000
        result = compare(cpu, gpu, 0.95)
        self.assertEqual(result["timing"]["cpu_over_gpu_speedup"], 5.0)
        self.assertEqual(
            result["gpu"]["cuda_peak_reserved_memory_bytes"], 6_000_000_000
        )

    def test_one_sided_timing_fails(self) -> None:
        cpu = artifact("cpu")
        cpu["execution"]["training_seconds"] = 20.0
        with self.assertRaisesRegex(ProofError, "GPU training time is invalid"):
            compare(cpu, artifact("cuda:0"), 0.95)

    def test_wide_gpu_memory_evidence_is_required(self) -> None:
        cpu = artifact("cpu")
        gpu = artifact("cuda:0")
        cpu["poc_id"] = "kaggle-flickr-wide-cpu-v1"
        gpu["poc_id"] = FLICKR_WIDE_KAGGLE_CUDA_SPEC.poc_id
        with self.assertRaisesRegex(ProofError, "allocated memory evidence"):
            compare(cpu, gpu, 0.95)

    def test_wide_gpu_memory_target_is_enforced(self) -> None:
        cpu = artifact("cpu")
        gpu = artifact("cuda:0")
        cpu["poc_id"] = "kaggle-flickr-wide-cpu-v1"
        gpu["poc_id"] = FLICKR_WIDE_KAGGLE_CUDA_SPEC.poc_id
        gpu["execution"].update(
            {
                "cuda_peak_memory_bytes": 3 * 1024**3,
                "cuda_peak_reserved_memory_bytes": 4 * 1024**3,
                "cuda_device_total_memory_bytes": 16 * 1024**3,
            }
        )
        with self.assertRaisesRegex(ProofError, "allocation target"):
            compare(cpu, gpu, 0.95)

    def test_wide_gpu_memory_fraction_is_validated(self) -> None:
        cpu = artifact("cpu")
        gpu = artifact("cuda:0")
        cpu["poc_id"] = "kaggle-flickr-wide-cpu-v1"
        gpu["poc_id"] = FLICKR_WIDE_KAGGLE_CUDA_SPEC.poc_id
        gpu["execution"].update(
            {
                "cuda_peak_memory_bytes": 6 * 1024**3,
                "cuda_peak_reserved_memory_bytes": 8 * 1024**3,
                "cuda_device_total_memory_bytes": 16 * 1024**3,
                "cuda_peak_allocated_fraction": 0.1,
                "cuda_peak_reserved_fraction": 0.5,
            }
        )
        with self.assertRaisesRegex(ProofError, "allocated memory fraction"):
            compare(cpu, gpu, 0.95)

    def test_2048_gpu_memory_target_is_enforced(self) -> None:
        cpu = artifact("cpu")
        gpu = artifact("cuda:0")
        cpu["poc_id"] = "kaggle-flickr-2048-cpu-v1"
        gpu["poc_id"] = FLICKR_2048_KAGGLE_CUDA_SPEC.poc_id
        gpu["execution"].update(
            {
                "cuda_peak_memory_bytes": 7 * 1024**3,
                "cuda_peak_reserved_memory_bytes": 8 * 1024**3,
                "cuda_device_total_memory_bytes": 16 * 1024**3,
            }
        )
        with self.assertRaisesRegex(ProofError, "allocation target"):
            compare(cpu, gpu, 0.95)

    def test_registered_metadata_proves_cpu_and_t4_configuration(self) -> None:
        cpu = kaggle_run_evidence(
            KARATE_KAGGLE_CPU_SPEC.poc_id, verify_remote_status=False
        )
        gpu = kaggle_run_evidence(
            KARATE_KAGGLE_SPEC.poc_id, verify_remote_status=False
        )
        self.assertFalse(cpu["enable_gpu"])
        self.assertIsNone(cpu["machine_shape"])
        self.assertTrue(gpu["enable_gpu"])
        self.assertEqual(gpu["machine_shape"], "NvidiaTeslaT4")

    def test_cpu_resource_evidence_is_parsed(self) -> None:
        content = """{
  "average_process_cpu_percent": 376.0,
  "exit_status": 0,
  "maximum_resident_set_bytes": 8318418944,
  "maximum_resident_set_kib": 8123456,
  "measurement": "Linux wait4 resource usage",
  "system_cpu_seconds": 12.5,
  "user_cpu_seconds": 710.0,
  "wall_clock_seconds": 192.0
}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resource.txt"
            path.write_text(content, encoding="utf8")
            evidence = load_cpu_resource_evidence(path)
        self.assertEqual(evidence["average_process_cpu_percent"], 376)
        self.assertEqual(evidence["maximum_resident_set_kib"], 8_123_456)
        self.assertEqual(evidence["maximum_resident_set_bytes"], 8_318_418_944)


if __name__ == "__main__":
    unittest.main()
