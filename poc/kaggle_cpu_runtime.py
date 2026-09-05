"""Shared proof helpers for Kaggle CPU-only runners."""

from __future__ import annotations

import os
import platform
import re
from pathlib import Path

import torch

from proof_common import require


def require_cpu_only() -> torch.device:
    require(not torch.cuda.is_available(), "CUDA is available; this POC must be CPU-only")
    return torch.device("cpu")


def source_revision() -> str:
    revision = os.environ.get("ML_SOURCE_REVISION", "")
    require(
        re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
        "ML_SOURCE_REVISION must be a full Git commit",
    )
    return revision


def cpu_model() -> str:
    model = platform.processor().strip()
    cpuinfo = Path("/proc/cpuinfo")
    if not model and cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "Hardware"}:
                model = value.strip()
                if model:
                    break
    return model or platform.machine()
