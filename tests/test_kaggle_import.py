from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "poc"))

import import_kaggle_result as importer  # noqa: E402


class ImportVerificationTests(unittest.TestCase):
    def test_verify_only_argument_is_available(self) -> None:
        arguments = importer.parse_arguments(["result.json", "--verify-only"])
        self.assertTrue(arguments.verify_only)

    def test_import_count_requires_complete_prediction_metadata(self) -> None:
        session = Mock()
        result = Mock()
        result.single.return_value = {"count": 34}
        session.run.return_value = result
        metadata = {
            "device_name": "Tesla T4",
            "accuracy": 0.75,
            "generated_at": "2026-09-05T00:00:00Z",
        }

        count = importer.count_imported(
            session,
            importer.KARATE_KAGGLE_SPEC,
            "a" * 64,
            metadata,
        )

        self.assertEqual(count, 34)
        query = session.run.call_args.args[0]
        self.assertIn("cuda_predicted_community IS NOT NULL", query)
        self.assertIn("node.cuda_device_name = $device_name", query)
        self.assertIn("node.cuda_accuracy = $accuracy", query)
        self.assertIn("node.cuda_generated_at = $generated_at", query)


if __name__ == "__main__":
    unittest.main()
