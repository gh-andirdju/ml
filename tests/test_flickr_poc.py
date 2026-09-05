from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "poc"))

import flickr_core as poc  # noqa: E402
from kaggle_flickr_runner import parse_arguments, validate_arguments  # noqa: E402
from proof_common import ProofError  # noqa: E402


class FlickrModelTests(unittest.TestCase):
    def test_three_layer_model_has_expected_output_shape(self) -> None:
        graph = Data(
            x=torch.ones((3, poc.EXPECTED_FEATURES)),
            edge_index=torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]]),
        )
        model = poc.FlickrGraphSAGE(hidden_channels=8, dropout=0.5)
        self.assertEqual(tuple(model(graph).shape), (3, poc.EXPECTED_CLASSES))
        self.assertEqual(len(model.convolutions), 3)


class FlickrArgumentTests(unittest.TestCase):
    def test_ready_profiles_have_identical_model_defaults(self) -> None:
        cpu = parse_arguments("cpu", [])
        cuda = parse_arguments("cuda", [])
        for name in (
            "epochs",
            "patience",
            "hidden_channels",
            "dropout",
            "seed",
            "learning_rate",
            "weight_decay",
        ):
            self.assertEqual(getattr(cpu, name), getattr(cuda, name))

    def test_invalid_hidden_channels_are_rejected(self) -> None:
        arguments = parse_arguments("cpu", ["--hidden-channels", "0"])
        with self.assertRaisesRegex(ProofError, "Hidden channels"):
            validate_arguments(arguments)


if __name__ == "__main__":
    unittest.main()
