"""Private Kaggle T4 wrapper for POC 18 matched plain-PyG full-batch Flickr."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


SOURCE_REVISION = "2d1b50bd0fba997c3effb07ad4426ef562e9e616"
SOURCE_ARCHIVE = f"https://github.com/gh-andirdju/ml/archive/{SOURCE_REVISION}.zip"
temporary = Path("/kaggle/temp/ml-poc-18")
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
        "--extra-index-url",
        "https://pypi.nvidia.com",
        "-r",
        str(project / "requirements-kaggle-cugraph.txt"),
    ],
    check=True,
)
subprocess.run(
    [sys.executable, str(project / "poc" / "run_kaggle_fullbatch_pyg_cuda.py")],
    cwd=project,
    env={
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "ML_SOURCE_REVISION": SOURCE_REVISION,
    },
    check=True,
)
