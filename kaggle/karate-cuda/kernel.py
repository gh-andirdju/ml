"""Private Kaggle kernel wrapper for POC 3."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


SOURCE_REVISION = "c9fe91f0de80ad82d5c76ce4908ee3e8473b6165"
SOURCE_ARCHIVE = f"https://github.com/gh-andirdju/ml/archive/{SOURCE_REVISION}.zip"
TEMPORARY = Path("/kaggle/temp/ml-poc-3")
TEMPORARY.mkdir(parents=True, exist_ok=True)
archive = TEMPORARY / "ml-source.zip"
urllib.request.urlretrieve(SOURCE_ARCHIVE, archive)
with zipfile.ZipFile(archive) as source_zip:
    source_zip.extractall(TEMPORARY / "ml-source")
project = next((TEMPORARY / "ml-source").glob("ml-*"))
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", str(project / "requirements-kaggle.txt")],
    check=True,
)
subprocess.run(
    [sys.executable, str(project / "poc" / "run_kaggle_karate_cuda.py")],
    cwd=project,
    env={**os.environ, "ML_SOURCE_REVISION": SOURCE_REVISION},
    check=True,
)
