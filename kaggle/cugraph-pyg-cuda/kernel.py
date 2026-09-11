"""Private Kaggle single-T4 wrapper for the cuGraph-PyG Flickr POC."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


SOURCE_REVISION = "75a50bf0a89b00d6d53ed6dd760cdc9de0158ae8"
SOURCE_ARCHIVE = f"https://github.com/gh-andirdju/ml/archive/{SOURCE_REVISION}.zip"
temporary = Path("/kaggle/temp/ml-poc-17")
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
    [sys.executable, str(project / "poc" / "run_kaggle_cugraph_pyg_cuda.py")],
    cwd=project,
    env={
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "ML_SOURCE_REVISION": SOURCE_REVISION,
    },
    check=True,
)
