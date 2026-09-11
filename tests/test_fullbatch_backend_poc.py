from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "poc"))

from compare_fullbatch_backends import compare  # noqa: E402
from fullbatch_backend_runner import parse_arguments, validate_arguments  # noqa: E402
from kaggle_specs import (  # noqa: E402
    FLICKR_FULLBATCH_CUGRAPH_PYG_KAGGLE_SPEC,
    FLICKR_FULLBATCH_PYG_KAGGLE_SPEC,
    FULLBATCH_BACKEND_KAGGLE_SOURCE_REVISION,
)
from poc_runtime import ProofError  # noqa: E402


def prediction(node_id: int, predicted_class: int) -> dict[str, object]:
    scores = [0.0] * 7
    scores[predicted_class] = 1.0
    return {
        "node_id": node_id,
        "predicted_class": predicted_class,
        "scores": scores,
    }


def artifact(backend: str, *, cugraph: bool) -> dict[str, object]:
    execution = {
        "backend": backend,
        "device": "cuda:0",
        "cuda_device_count": 1,
        "cuda_device_name": "Tesla T4",
        "cuda_capability": [7, 5],
        "exact_full_graph_verified": True,
        "model_parameters": 391_175,
        "training_seconds": 6.0,
        "backend_preparation_seconds": 1.0,
        "backend_inclusive_seconds": 7.0,
        "cuda_peak_memory_bytes": 2_000_000_000,
        "cuda_peak_reserved_memory_bytes": 2_500_000_000,
        "cuda_device_total_memory_bytes": 16_000_000_000,
        "source_revision": FULLBATCH_BACKEND_KAGGLE_SOURCE_REVISION,
        "python": "3.12.13",
        "torch": "2.10.0+cu128",
        "torch_cuda": "12.8",
        "torch_geometric": "2.7.0",
        "cugraph_pyg": "26.2.1",
        "pylibcugraph": "26.2.0",
        "pylibwholegraph": "26.2.1",
        "cugraph_comms_initialized": cugraph,
        "materialized_batches": 1 if cugraph else 0,
        "full_neighbor_fanout": [-1] if cugraph else None,
        "test_accuracy": 0.42,
    }
    if cugraph:
        execution.update(
            {
                "distributed_backend": "nccl",
                "distributed_world_size": 1,
                "loader_module": "cugraph_pyg.loader.neighbor_loader",
                "graph_store_module": "cugraph_pyg.data.graph_store",
                "feature_store_module": "cugraph_pyg.data.feature_store",
            }
        )
    return {
        "poc_id": (
            FLICKR_FULLBATCH_CUGRAPH_PYG_KAGGLE_SPEC.poc_id
            if cugraph
            else FLICKR_FULLBATCH_PYG_KAGGLE_SPEC.poc_id
        ),
        "dataset": {"name": "Flickr", "nodes": 2, "classes": 7},
        "model": {"type": "three-layer GraphSAGE", "execution_mode": "full-batch"},
        "execution": execution,
        "predictions": [prediction(0, 1), prediction(1, 2)],
    }


class FullBatchBackendPocTests(unittest.TestCase):
    def test_default_contract_matches_original_flickr_benchmark(self) -> None:
        for backend in ("pyg", "cugraph-pyg"):
            arguments = parse_arguments(backend, [])
            validate_arguments(arguments)
            self.assertEqual(arguments.epochs, 30)
            self.assertEqual(arguments.patience, 8)
            self.assertEqual(arguments.hidden_channels, 256)
            self.assertEqual(arguments.seed, 42)

    def test_both_kernels_share_one_t4_and_one_source_revision(self) -> None:
        self.assertEqual(len(FULLBATCH_BACKEND_KAGGLE_SOURCE_REVISION), 40)
        for directory in (
            "flickr-fullbatch-pyg-cuda",
            "flickr-fullbatch-cugraph-pyg-cuda",
        ):
            root = PROJECT_ROOT / "kaggle" / directory
            metadata = json.loads(
                (root / "kernel-metadata.json").read_text(encoding="utf8")
            )
            wrapper = (root / "kernel.py").read_text(encoding="utf8")
            self.assertEqual(metadata["enable_gpu"], "true")
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
            self.assertIn('"CUDA_VISIBLE_DEVICES": "0"', wrapper)
            self.assertIn(FULLBATCH_BACKEND_KAGGLE_SOURCE_REVISION, wrapper)

    def test_comparison_accepts_identical_predictions(self) -> None:
        result = compare(
            artifact("pyg-direct-full-batch", cugraph=False),
            artifact("cugraph-pyg-full-neighbor", cugraph=True),
            "a" * 64,
            "b" * 64,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["prediction_comparison"]["class_agreement"], 1.0
        )

    def test_comparison_rejects_plain_backend_claiming_cugraph(self) -> None:
        plain = artifact("cugraph-pyg-full-neighbor", cugraph=False)
        with self.assertRaisesRegex(ProofError, "Backend identity"):
            compare(
                plain,
                artifact("cugraph-pyg-full-neighbor", cugraph=True),
                "a" * 64,
                "b" * 64,
            )

    def test_comparison_rejects_a_model_difference(self) -> None:
        plain = artifact("pyg-direct-full-batch", cugraph=False)
        cugraph = artifact("cugraph-pyg-full-neighbor", cugraph=True)
        cugraph["model"]["hidden_channels"] = 1_024
        with self.assertRaisesRegex(ProofError, "Model configuration"):
            compare(plain, cugraph, "a" * 64, "b" * 64)


if __name__ == "__main__":
    unittest.main()
