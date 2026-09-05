from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "poc"))

import karate_gnn_neo4j as poc  # noqa: E402


class EnvironmentFileTests(unittest.TestCase):
    def test_loads_values_without_overriding_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.env"
            path.write_text(
                "# comment\n TEST_NEW =new-value\nTEST_EXISTING=file-value\n",
                encoding="utf8",
            )
            with patch.dict(
                os.environ, {"TEST_EXISTING": "process-value"}, clear=True
            ):
                poc.load_environment_file(path)
                self.assertEqual(os.environ["TEST_NEW"], "new-value")
                self.assertEqual(os.environ["TEST_EXISTING"], "process-value")

    def test_rejects_malformed_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.env"
            path.write_text("NOT AN ENVIRONMENT ENTRY\n", encoding="utf8")
            with self.assertRaisesRegex(poc.ProofError, r"test\.env:1"):
                poc.load_environment_file(path)


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = poc.source_graph()

    def test_payload_has_exact_expected_shape(self) -> None:
        nodes, edges = poc.graph_payload(self.graph)
        self.assertEqual(len(nodes), poc.EXPECTED_NODES)
        self.assertEqual(len(edges), poc.EXPECTED_RELATIONSHIPS)
        self.assertEqual(len(nodes[0]["features"]), poc.EXPECTED_FEATURES)
        unique_edges = {(edge["source"], edge["target"]) for edge in edges}
        self.assertEqual(len(unique_edges), len(edges))

    def test_cpu_device_is_explicit_and_available(self) -> None:
        self.assertEqual(poc.resolve_device(" CPU "), poc.torch.device("cpu"))

    def test_unsupported_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(poc.ProofError, "Unsupported device type"):
            poc.resolve_device("meta")


class ConnectionTests(unittest.TestCase):
    def test_connection_timeout_is_forwarded_to_driver(self) -> None:
        driver = Mock()
        with patch.object(poc.GraphDatabase, "driver", return_value=driver) as create:
            result = poc.connect_with_retry("bolt://example", "neo4j", "secret", 3)

        self.assertIs(result, driver)
        driver.verify_connectivity.assert_called_once_with()
        timeout = create.call_args.kwargs["connection_timeout"]
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, 3)

    def test_failed_driver_is_closed_at_deadline(self) -> None:
        driver = Mock()
        driver.verify_connectivity.side_effect = OSError("unreachable")
        with (
            patch.object(poc.GraphDatabase, "driver", return_value=driver),
            patch.object(poc.time, "monotonic", side_effect=[0, 0, 2, 2]),
            self.assertRaisesRegex(poc.ProofError, "within 2s"),
        ):
            poc.connect_with_retry("bolt://example", "neo4j", "secret", 2)

        driver.close.assert_called_once_with()


class ArgumentTests(unittest.TestCase):
    def test_profile_arguments_can_be_overridden(self) -> None:
        arguments = poc.parse_arguments(["--device", "mps", "--device", "cpu"])
        self.assertEqual(arguments.device, "cpu")


if __name__ == "__main__":
    unittest.main()
