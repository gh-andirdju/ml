"""Private Kaggle single-T4 wrapper for the cuGraph-PyG Flickr POC."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


SOURCE_REVISION = "62849be3b1256035d092f1ee98394749df1fd1bc"
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
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
subprocess.run(
    [sys.executable, str(project / "poc" / "run_kaggle_cugraph_pyg_cuda.py")],
    cwd=project,
    env={**os.environ, "ML_SOURCE_REVISION": SOURCE_REVISION},
    check=True,
)
