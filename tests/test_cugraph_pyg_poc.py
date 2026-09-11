from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "poc"))

from cugraph_pyg_flickr_runner import parse_arguments, validate_arguments  # noqa: E402
from kaggle_specs import CUGRAPH_PYG_KAGGLE_SOURCE_REVISION  # noqa: E402
from poc_runtime import ProofError  # noqa: E402
from validate_cugraph_pyg_result import validate_execution  # noqa: E402


def valid_artifact() -> dict:
    return {
        "execution": {
            "cuda_device_name": "Tesla T4",
            "cuda_capability": [7, 5],
            "cuda_device_count": 1,
            "torch_geometric": "2.7.0",
            "cugraph_pyg": "26.2.1",
            "pylibcugraph": "26.2.1",
            "pylibwholegraph": "26.2.0",
            "loader_module": "cugraph_pyg.loader.neighbor_loader",
            "graph_store_module": "cugraph_pyg.data.graph_store",
            "feature_store_module": "cugraph_pyg.data.feature_store",
            "cugraph_comms_initialized": True,
            "distributed_backend": "nccl",
            "distributed_world_size": 1,
            "batches_processed": 10,
            "seed_nodes_processed": 100,
            "sampled_nodes_processed": 500,
            "sampled_edges_processed": 1000,
            "training_seconds": 3.0,
            "sampling_seconds": 1.0,
            "optimization_seconds": 1.5,
            "validation_seconds": 0.5,
        }
    }


class CuGraphPyGPocTests(unittest.TestCase):
    def test_committed_result_is_a_complete_version_three_proof(self) -> None:
        result = json.loads(
            (
                PROJECT_ROOT / "results" / "kaggle-flickr-cugraph-pyg-t4.json"
            ).read_text(encoding="utf8")
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["proof_status"], "PASS")
        self.assertTrue(all(result["proof"].values()))
        self.assertEqual(result["kaggle_run"]["kernel_version"], 3)
        self.assertEqual(result["kaggle_run"]["status"], "COMPLETE")
        self.assertEqual(result["frameworks"]["cugraph_pyg"], "26.2.1")
        self.assertEqual(result["execution"]["cuda_device_count"], 1)
        self.assertGreaterEqual(result["execution"]["test_accuracy"], 0.30)

    def test_kaggle_environment_is_pinned_to_one_t4(self) -> None:
        metadata = json.loads(
            (
                PROJECT_ROOT
                / "kaggle"
                / "cugraph-pyg-cuda"
                / "kernel-metadata.json"
            ).read_text(encoding="utf8")
        )
        requirements = (
            PROJECT_ROOT / "requirements-kaggle-cugraph.txt"
        ).read_text(encoding="utf8")
        wrapper = (
            PROJECT_ROOT / "kaggle" / "cugraph-pyg-cuda" / "kernel.py"
        ).read_text(encoding="utf8")
        self.assertTrue(metadata["enable_gpu"] == "true")
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertIn("torch-geometric==2.7.0", requirements)
        self.assertIn("cugraph-pyg-cu12==26.2.1", requirements)
        self.assertEqual(len(CUGRAPH_PYG_KAGGLE_SOURCE_REVISION), 40)
        self.assertIn(CUGRAPH_PYG_KAGGLE_SOURCE_REVISION, wrapper)
        self.assertIn('"CUDA_VISIBLE_DEVICES": "0"', wrapper)

    def test_defaults_define_three_layer_sampled_training(self) -> None:
        arguments = parse_arguments([])
        validate_arguments(arguments)
        self.assertEqual(arguments.fanout, [15, 10, 5])
        self.assertEqual(arguments.hidden_channels, 256)
        self.assertEqual(arguments.batch_size, 1024)

    def test_nonpositive_fanout_is_rejected(self) -> None:
        arguments = parse_arguments(["--fanout", "15", "0", "5"])
        with self.assertRaisesRegex(ProofError, "fanout"):
            validate_arguments(arguments)

    def test_execution_requires_real_cugraph_pyg_components(self) -> None:
        execution = validate_execution(valid_artifact())
        self.assertEqual(execution["distributed_backend"], "nccl")

    def test_plain_pyg_loader_cannot_claim_cugraph_proof(self) -> None:
        artifact = valid_artifact()
        artifact["execution"]["loader_module"] = "torch_geometric.loader"
        with self.assertRaisesRegex(ProofError, "loader"):
            validate_execution(artifact)


if __name__ == "__main__":
    unittest.main()
