"""Private Kaggle kernel wrapper for POC 4."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


SOURCE_REVISION = "69610294508b61a2148199ccb74c7b5cca050602"
SOURCE_ARCHIVE = f"https://github.com/gh-andirdju/ml/archive/{SOURCE_REVISION}.zip"
TEMPORARY = Path("/kaggle/temp/ml-poc-4")
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
    [
        sys.executable,
        str(project / "poc" / "run_kaggle_wikics_cuda.py"),
        "--data-root",
        "/kaggle/temp/ml-poc-4/wikics-data",
    ],
    cwd=project,
    env={**os.environ, "ML_SOURCE_REVISION": SOURCE_REVISION},
    check=True,
)
