from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "poc"))

import wikics_gnn_neo4j as poc  # noqa: E402


class DatasetTests(unittest.TestCase):
    def test_dataset_url_is_commit_pinned(self) -> None:
        self.assertIn(poc.DATASET_COMMIT, poc.PinnedWikiCS.url)
        self.assertNotIn("/master/", poc.PinnedWikiCS.url)

    def test_file_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            path.write_bytes(b"WikiCS")
            self.assertEqual(
                poc.file_sha256(path),
                "53bea9e21d43087d20a410ebde59b2e6cbc76a4279a1d89806d39b3e652b1517",
            )

    def test_node_batches_preserve_masks_and_features(self) -> None:
        graph = Data(
            x=torch.arange(900, dtype=torch.float32).reshape(3, 300),
            y=torch.tensor([0, 1, 2]),
            train_mask=torch.tensor([True, False, False]),
            val_mask=torch.tensor([False, True, False]),
            stopping_mask=torch.tensor([False, False, True]),
            test_mask=torch.tensor([True, True, True]),
        )
        batches = list(poc.node_batches(graph, 2))
        self.assertEqual([len(batch) for batch in batches], [2, 1])
        self.assertEqual(len(batches[0][0]["features"]), 300)
        self.assertTrue(batches[0][0]["is_training"])
        self.assertTrue(batches[0][1]["is_validation"])

    def test_canonical_relationships_keep_one_direction_and_self_loops(self) -> None:
        graph = Data(
            edge_index=torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.long),
            num_nodes=2,
        )
        with patch.object(poc, "EXPECTED_RELATIONSHIPS", 2):
            self.assertEqual(poc.canonical_relationships(graph), [(0, 1), (1, 1)])


class ModelTests(unittest.TestCase):
    def test_two_layer_model_has_expected_output_shape(self) -> None:
        graph = Data(
            x=torch.ones((3, poc.EXPECTED_FEATURES)),
            edge_index=torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]]),
        )
        model = poc.WikiGCN(hidden_channels=8, dropout=0.25)
        self.assertEqual(tuple(model(graph).shape), (3, poc.EXPECTED_CLASSES))


class DatabaseBatchTests(unittest.TestCase):
    def test_prepare_graph_deletes_until_no_nodes_remain(self) -> None:
        session = Mock()
        constraint_result = Mock()
        first_delete = Mock()
        final_delete = Mock()
        first_delete.single.return_value = {"deleted": 250}
        final_delete.single.return_value = {"deleted": 0}
        session.run.side_effect = [constraint_result, first_delete, final_delete]

        poc.prepare_graph(session, 250)

        constraint_result.consume.assert_called_once_with()
        self.assertEqual(session.run.call_count, 3)


class ArgumentTests(unittest.TestCase):
    def test_invalid_split_is_rejected(self) -> None:
        arguments = poc.parse_arguments(["--split", "20"])
        with self.assertRaisesRegex(poc.ProofError, "Split must be from 0 to 19"):
            poc.validate_arguments(arguments)


if __name__ == "__main__":
    unittest.main()
