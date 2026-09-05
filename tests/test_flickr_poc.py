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

    def test_wide_profiles_have_identical_larger_defaults(self) -> None:
        cpu = parse_arguments("cpu", [], variant="wide")
        cuda = parse_arguments("cuda", [], variant="wide")
        self.assertEqual(cpu.hidden_channels, 1_024)
        self.assertEqual(cpu.epochs, 20)
        self.assertEqual(cpu.patience, 6)
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
        self.assertIn("flickr-wide-cpu", str(cpu.output))
        self.assertIn("flickr-wide-cuda", str(cuda.output))

    def test_2048_profiles_have_identical_exact_width(self) -> None:
        cpu = parse_arguments("cpu", [], variant="2048")
        cuda = parse_arguments("cuda", [], variant="2048")
        self.assertEqual(cpu.hidden_channels, 2_048)
        self.assertEqual(cpu.epochs, 20)
        self.assertEqual(cpu.patience, 6)
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
        self.assertIn("flickr-2048-cpu", str(cpu.output))
        self.assertIn("flickr-2048-cuda", str(cuda.output))


if __name__ == "__main__":
    unittest.main()
