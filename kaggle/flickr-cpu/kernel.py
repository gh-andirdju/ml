"""Private Kaggle CPU-only kernel wrapper for POC 7."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


SOURCE_REVISION = "95ccbccd57dc4ec9f0c7c9f143dc941e615dc520"
SOURCE_ARCHIVE = f"https://github.com/gh-andirdju/ml/archive/{SOURCE_REVISION}.zip"
temporary = Path("/kaggle/temp/ml-poc-7")
temporary.mkdir(parents=True, exist_ok=True)
archive = temporary / "ml-source.zip"
urllib.request.urlretrieve(SOURCE_ARCHIVE, archive)
with zipfile.ZipFile(archive) as source_zip:
    source_zip.extractall(temporary / "ml-source")
project = next((temporary / "ml-source").glob("ml-*"))
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", str(project / "requirements-kaggle.txt")],
    check=True,
)
subprocess.run(
    [
        "/usr/bin/time",
        "--verbose",
        "--output",
        "/kaggle/working/flickr-cpu-resource-usage.txt",
        sys.executable,
        str(project / "poc" / "run_kaggle_flickr_cpu.py"),
    ],
    cwd=project,
    env={**os.environ, "ML_SOURCE_REVISION": SOURCE_REVISION},
    check=True,
)
