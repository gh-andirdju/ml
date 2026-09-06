"""Private Kaggle CPU-only kernel wrapper for POC 15."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


SOURCE_REVISION = "PIN_AFTER_EXECUTABLE_COMMIT"
SOURCE_ARCHIVE = f"https://github.com/gh-andirdju/ml/archive/{SOURCE_REVISION}.zip"
temporary = Path("/kaggle/temp/ml-poc-15")
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
started_at = time.monotonic()
process = subprocess.Popen(
    [sys.executable, str(project / "poc" / "run_kaggle_flickr_8192_cpu.py")],
    cwd=project,
    env={**os.environ, "ML_SOURCE_REVISION": SOURCE_REVISION},
)
_, wait_status, usage = os.wait4(process.pid, 0)
elapsed_seconds = time.monotonic() - started_at
process.returncode = os.waitstatus_to_exitcode(wait_status)
if process.returncode != 0:
    raise subprocess.CalledProcessError(process.returncode, process.args)
resource_evidence = {
    "measurement": "Linux wait4 resource usage for the complete Python runner",
    "average_process_cpu_percent": round(
        (usage.ru_utime + usage.ru_stime) / elapsed_seconds * 100, 3
    ),
    "maximum_resident_set_kib": usage.ru_maxrss,
    "maximum_resident_set_bytes": usage.ru_maxrss * 1024,
    "user_cpu_seconds": round(usage.ru_utime, 6),
    "system_cpu_seconds": round(usage.ru_stime, 6),
    "wall_clock_seconds": round(elapsed_seconds, 6),
    "exit_status": process.returncode,
}
Path("/kaggle/working/flickr-8192-cpu-resource-usage.json").write_text(
    json.dumps(resource_evidence, indent=2, sort_keys=True) + "\n",
    encoding="utf8",
)
