"""Create comparison-only MPS artifacts for every GNN workload."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
import torch_geometric

from flickr_core import source_graph as flickr_graph
from flickr_core import train_on_device as train_flickr
from kaggle_specs import (
    FLICKR_MPS_SPEC,
    FLICKR_WIDE_MPS_SPEC,
    KARATE_MPS_SPEC,
    WIKICS_MPS_SPEC,
)
from karate_core import source_graph as karate_graph
from karate_core import train_on_device as train_karate
from proof_common import ProofError, require
from result_artifact import ArtifactSpec, artifact_from_logits, write_artifact
from wikics_core import source_graph as wikics_graph
from wikics_core import train_on_device as train_wikics


Workload = Literal["karate", "wikics", "flickr", "flickr-wide"]
WORKLOADS = {"karate", "wikics", "flickr", "flickr-wide"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def repository_revision() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    require(not status.stdout.strip(), "Tracked worktree changes prevent a reproducible run")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "Invalid Git revision")
    return revision


def mps_device() -> torch.device:
    require(torch.backends.mps.is_available(), "MPS is unavailable")
    require(
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1",
        "CPU fallback is enabled; this would not prove MPS execution",
    )
    device = torch.device("mps")
    torch.empty(1, device=device)
    return device


def maximum_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def _karate(device: torch.device) -> tuple[torch.Tensor, dict, dict, ArtifactSpec]:
    logits, initial, final, accuracy, actual_device = train_karate(
        karate_graph(), 100, device, 42, 0.1, 5e-4
    )
    return (
        logits,
        {
            "device": actual_device,
            "accuracy": round(accuracy, 6),
            "epochs": 100,
            "seed": 42,
            "initial_loss": round(initial, 6),
            "final_loss": round(final, 6),
        },
        {
            "type": "one-layer GCN",
            "learning_rate": 0.1,
            "weight_decay": 5e-4,
        },
        KARATE_MPS_SPEC,
    )


def _wikics(device: torch.device) -> tuple[torch.Tensor, dict, dict, ArtifactSpec]:
    logits, metrics = train_wikics(
        wikics_graph(PROJECT_ROOT / ".artifacts/datasets/wikics", 0),
        100,
        20,
        64,
        0.25,
        device,
        42,
        0.02,
        5e-4,
    )
    metrics["accuracy"] = metrics["test_accuracy"]
    return (
        logits,
        metrics,
        {
            "type": "two-layer GCN",
            "epochs_requested": 100,
            "patience": 20,
            "hidden_channels": 64,
            "dropout": 0.25,
            "seed": 42,
            "learning_rate": 0.02,
            "weight_decay": 5e-4,
        },
        WIKICS_MPS_SPEC,
    )


def _flickr(
    device: torch.device, *, wide: bool
) -> tuple[torch.Tensor, dict, dict, ArtifactSpec]:
    epochs = 20 if wide else 30
    patience = 6 if wide else 8
    hidden_channels = 1_024 if wide else 256
    logits, metrics = train_flickr(
        flickr_graph(PROJECT_ROOT / ".artifacts/datasets/flickr"),
        epochs,
        patience,
        hidden_channels,
        0.5,
        device,
        42,
        0.01,
        5e-4,
    )
    metrics["accuracy"] = metrics["test_accuracy"]
    model = {
        "type": "three-layer GraphSAGE",
        "epochs_requested": epochs,
        "patience": patience,
        "hidden_channels": hidden_channels,
        "dropout": 0.5,
        "seed": 42,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
    }
    if wide:
        model["benchmark_variant"] = "wide"
    return logits, metrics, model, FLICKR_WIDE_MPS_SPEC if wide else FLICKR_MPS_SPEC


def run(workload: Workload) -> tuple[torch.Tensor, dict, dict, ArtifactSpec]:
    device = mps_device()
    if workload == "karate":
        return _karate(device)
    if workload == "wikics":
        return _wikics(device)
    return _flickr(device, wide=workload == "flickr-wide")


def main(workload: Workload, argv: Sequence[str] | None = None) -> int:
    require(workload in WORKLOADS, f"Unsupported MPS workload: {workload}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / f".artifacts/{workload}-mps/{workload}-mps-result.json",
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)
    revision = repository_revision()
    logits, metrics, model, spec = run(workload)
    execution = {
        "status": "PASS",
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback_enabled": False,
        "hardware": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "torch_cuda": torch.version.cuda,
        "source_revision": revision,
        "maximum_process_rss_bytes": maximum_rss_bytes(),
        **metrics,
    }
    artifact = artifact_from_logits(
        spec=spec,
        logits=logits,
        model=model,
        execution=execution,
    )
    checksum_path, digest = write_artifact(
        arguments.output, artifact, force=arguments.force
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "poc_id": spec.poc_id,
                "artifact": str(arguments.output),
                "checksum": str(checksum_path),
                "sha256": digest,
                "device": execution["device"],
                "accuracy": execution["accuracy"],
                "training_seconds": execution.get("training_seconds"),
                "maximum_process_rss_bytes": execution["maximum_process_rss_bytes"],
                "predictions_exported": len(artifact["predictions"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cli(workload: Workload, argv: Sequence[str] | None = None) -> int:
    try:
        return main(workload, argv)
    except (ProofError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"MPS {workload.upper()} ARTIFACT FAILED: {error}", file=sys.stderr)
        return 1
