"""Private Kaggle single-T4 CUDA kernel wrapper for POC 16."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


SOURCE_REVISION = "3b967fcbe7a735c4bf5ce7f54780d4d2e7e0519b"
SOURCE_ARCHIVE = f"https://github.com/gh-andirdju/ml/archive/{SOURCE_REVISION}.zip"
temporary = Path("/kaggle/temp/ml-poc-16")
temporary.mkdir(parents=True, exist_ok=True)
archive = temporary / "ml-source.zip"
urllib.request.urlretrieve(SOURCE_ARCHIVE, archive)
with zipfile.ZipFile(archive) as source_zip:
    source_zip.extractall(temporary / "ml-source")
project = next((temporary / "ml-source").glob("ml-*"))
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
        "-r",
        str(project / "requirements-kaggle.txt"),
    ],
    check=True,
)
subprocess.run(
    [sys.executable, str(project / "poc" / "run_kaggle_flickr_8192_cuda.py")],
    cwd=project,
    env={
        **os.environ,
        "ML_SOURCE_REVISION": SOURCE_REVISION,
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    },
    check=True,
)
