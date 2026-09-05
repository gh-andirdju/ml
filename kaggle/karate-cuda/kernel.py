"""Private Kaggle kernel wrapper for POC 3."""

from __future__ import annotations

import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


SOURCE_ARCHIVE = "https://github.com/gh-andirdju/ml/archive/refs/heads/main.zip"
WORKING = Path("/kaggle/working")
archive = WORKING / "ml-source.zip"
urllib.request.urlretrieve(SOURCE_ARCHIVE, archive)
with zipfile.ZipFile(archive) as source_zip:
    source_zip.extractall(WORKING / "ml-source")
project = next((WORKING / "ml-source").glob("ml-*"))
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", str(project / "requirements-kaggle.txt")],
    check=True,
)
subprocess.run(
    [sys.executable, str(project / "poc" / "run_kaggle_karate_cuda.py")],
    cwd=project,
    check=True,
)
