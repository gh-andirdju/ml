from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class KaggleBoundaryTests(unittest.TestCase):
    def test_gpu_exporters_do_not_import_neo4j(self) -> None:
        script = f"""
import builtins
import sys
sys.path.insert(0, {str(PROJECT_ROOT / 'poc')!r})
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'neo4j' or name.startswith('neo4j.'):
        raise AssertionError('Kaggle exporter imported Neo4j')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import run_kaggle_karate_cuda
import run_kaggle_wikics_cuda
"""
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
