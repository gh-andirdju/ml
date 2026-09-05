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

    def test_chunked_mean_aggregation_matches_standard_graphsage(self) -> None:
        features = torch.tensor(
            [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], requires_grad=True
        )
        edge_index = torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]])
        standard = poc.SAGEConv(2, 3)
        chunked = poc.ChunkedSAGEConv(2, 3, edge_chunk_size=2)
        chunked.load_state_dict(standard.state_dict())
        expected = standard(features, edge_index)
        actual = chunked(features, edge_index)
        torch.testing.assert_close(actual, expected)

        expected.sum().backward(retain_graph=True)
        expected_gradient = features.grad.detach().clone()
        features.grad = None
        actual.sum().backward()
        torch.testing.assert_close(features.grad, expected_gradient)

    def test_activation_checkpointing_preserves_training_gradients(self) -> None:
        graph = Data(
            x=torch.ones((3, poc.EXPECTED_FEATURES)),
            edge_index=torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]]),
        )
        standard = poc.FlickrGraphSAGE(
            hidden_channels=8,
            dropout=0,
            edge_chunk_size=2,
            activation_checkpointing=False,
        )
        checkpointed = poc.FlickrGraphSAGE(
            hidden_channels=8,
            dropout=0,
            edge_chunk_size=2,
            activation_checkpointing=True,
        )
        checkpointed.load_state_dict(standard.state_dict())
        expected = standard(graph)
        actual = checkpointed(graph)
        torch.testing.assert_close(actual, expected)
        expected.sum().backward()
        actual.sum().backward()
        for standard_parameter, checkpointed_parameter in zip(
            standard.parameters(), checkpointed.parameters(), strict=True
        ):
            torch.testing.assert_close(
                checkpointed_parameter.grad, standard_parameter.grad
            )


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

    def test_4096_profiles_have_identical_bounded_memory_defaults(self) -> None:
        cpu = parse_arguments("cpu", [], variant="4096")
        cuda = parse_arguments("cuda", [], variant="4096")
        self.assertEqual(cpu.hidden_channels, 4_096)
        self.assertEqual(cpu.epochs, 10)
        self.assertEqual(cpu.patience, 4)
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
        self.assertIn("flickr-4096-cpu", str(cpu.output))
        self.assertIn("flickr-4096-cuda", str(cuda.output))


if __name__ == "__main__":
    unittest.main()
