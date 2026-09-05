from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RESULTS = {
    "karate-mps-cpu-cuda.json": "karate",
    "wikics-mps-cpu-cuda.json": "wikics",
    "flickr-mps-cpu-cuda.json": "flickr",
    "flickr-wide-mps-cpu-cuda.json": "flickr-wide",
    "flickr-2048-mps-cpu-cuda.json": "flickr-2048",
}
EXPECTED_PAIRS = {
    "mps_vs_kaggle_cpu",
    "mps_vs_kaggle_gpu",
    "kaggle_cpu_vs_gpu",
}


class RecordedThreeEnvironmentResultsTests(unittest.TestCase):
    def test_every_workload_has_a_complete_passing_triplet(self) -> None:
        for filename, workload in EXPECTED_RESULTS.items():
            with self.subTest(workload=workload):
                path = PROJECT_ROOT / "results" / filename
                result = json.loads(path.read_text(encoding="utf8"))
                self.assertEqual(result["workload"], workload)
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(result["proof_status"], "PASS")
                self.assertTrue(all(result["proof"].values()))
                self.assertEqual(set(result["pairwise"]), EXPECTED_PAIRS)
                for comparison in result["pairwise"].values():
                    self.assertGreaterEqual(comparison["class_agreement"], 0.95)
                environments = result["environments"]
                self.assertEqual(
                    set(environments), {"mps", "kaggle_cpu", "kaggle_gpu"}
                )
                self.assertTrue(environments["mps"]["mps_available"])
                self.assertFalse(environments["mps"]["mps_fallback_enabled"])
                self.assertEqual(environments["kaggle_cpu"]["device"], "cpu")
                self.assertTrue(
                    environments["kaggle_gpu"]["device"].startswith("cuda:")
                )


if __name__ == "__main__":
    unittest.main()
