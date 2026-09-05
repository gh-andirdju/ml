#!/usr/bin/env python3
"""POC 4: train pinned WikiCS on CUDA and export predictions for local Neo4j."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch_geometric

from kaggle_specs import WIKICS_KAGGLE_SPEC
from poc_runtime import ProofError, require
from result_artifact import artifact_from_logits, write_artifact
from wikics_core import train_on_device, source_graph


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/wikics-cuda-result.json"))
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/working/data/wikics"))
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    require(arguments.split == 0, "This versioned artifact supports WikiCS split 0 only")
    require(arguments.epochs > 0, "Epoch count must be positive")
    require(arguments.patience > 0, "Patience must be positive")
    require(arguments.hidden_channels > 0, "Hidden channels must be positive")
    require(0 <= arguments.dropout < 1, "Dropout must be in [0, 1)")
    require(arguments.learning_rate > 0, "Learning rate must be positive")
    require(arguments.weight_decay >= 0, "Weight decay cannot be negative")
    require(torch.cuda.is_available(), "CUDA is unavailable; this POC is GPU-only")
    source_revision = os.environ.get("ML_SOURCE_REVISION", "")
    require(re.fullmatch(r"[0-9a-f]{40}", source_revision) is not None, "ML_SOURCE_REVISION must be a full Git commit")
    device = torch.device("cuda:0")
    graph = source_graph(arguments.data_root, arguments.split)
    logits, metrics = train_on_device(
        graph,
        arguments.epochs,
        arguments.patience,
        arguments.hidden_channels,
        arguments.dropout,
        device,
        arguments.seed,
        arguments.learning_rate,
        arguments.weight_decay,
    )
    execution = {
        "status": "PASS",
        "device": metrics["device"],
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "source_revision": source_revision,
        "accuracy": metrics["test_accuracy"],
        **metrics,
    }
    artifact = artifact_from_logits(
        spec=WIKICS_KAGGLE_SPEC,
        logits=logits,
        model={
            "type": "two-layer GCN",
            "epochs_requested": arguments.epochs,
            "patience": arguments.patience,
            "hidden_channels": arguments.hidden_channels,
            "dropout": arguments.dropout,
            "seed": arguments.seed,
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
        },
        execution=execution,
    )
    checksum_path, digest = write_artifact(arguments.output, artifact, force=arguments.force)
    print(json.dumps({
        "status": "PASS",
        "poc_id": WIKICS_KAGGLE_SPEC.poc_id,
        "artifact": str(arguments.output),
        "checksum": str(checksum_path),
        "sha256": digest,
        "device": execution["device"],
        "cuda_device_name": execution["cuda_device_name"],
        "validation_accuracy": metrics["validation_accuracy"],
        "test_accuracy": metrics["test_accuracy"],
        "predictions_exported": len(artifact["predictions"]),
    }, indent=2, sort_keys=True))
    return 0


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (ProofError, OSError, RuntimeError, ValueError) as error:
        print(f"KAGGLE WIKICS POC FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
