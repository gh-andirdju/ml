from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class KaggleBoundaryTests(unittest.TestCase):
    def test_kaggle_exporters_do_not_import_neo4j(self) -> None:
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
import run_kaggle_karate_cpu
import run_kaggle_wikics_cpu
import run_kaggle_flickr_cpu
import run_kaggle_flickr_cuda
import run_kaggle_flickr_wide_cpu
import run_kaggle_flickr_wide_cuda
import run_kaggle_flickr_2048_cpu
import run_kaggle_flickr_2048_cuda
import run_kaggle_flickr_4096_cpu
import run_kaggle_flickr_4096_cuda
import run_kaggle_flickr_8192_cpu
import run_kaggle_flickr_8192_cuda
"""
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_mps_comparison_exporters_do_not_import_neo4j(self) -> None:
        script = f"""
import builtins
import sys
sys.path.insert(0, {str(PROJECT_ROOT / 'poc')!r})
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'neo4j' or name.startswith('neo4j.'):
        raise AssertionError('MPS comparison exporter imported Neo4j')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import run_mps_karate_artifact
import run_mps_wikics_artifact
import run_mps_flickr
import run_mps_flickr_wide
import run_mps_flickr_2048
import run_mps_flickr_4096
import run_mps_flickr_8192
"""
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
