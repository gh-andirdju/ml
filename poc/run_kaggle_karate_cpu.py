#!/usr/bin/env python3
"""POC 5: train Karate Club on Kaggle CPU and export predictions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch_geometric

from kaggle_cpu_runtime import cpu_model, require_cpu_only, source_revision
from kaggle_specs import KARATE_KAGGLE_CPU_SPEC
from karate_core import source_graph, train_on_device
from proof_common import ProofError, require
from result_artifact import artifact_from_logits, write_artifact


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/karate-cpu-result.json"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    require(arguments.epochs > 0, "Epoch count must be positive")
    require(arguments.learning_rate > 0, "Learning rate must be positive")
    require(arguments.weight_decay >= 0, "Weight decay cannot be negative")
    revision = source_revision()
    device = require_cpu_only()
    logits, initial_loss, final_loss, accuracy, actual_device = train_on_device(
        source_graph(),
        arguments.epochs,
        device,
        arguments.seed,
        arguments.learning_rate,
        arguments.weight_decay,
    )
    execution = {
        "status": "PASS",
        "device": actual_device,
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "source_revision": revision,
        "epochs": arguments.epochs,
        "seed": arguments.seed,
        "initial_loss": round(initial_loss, 6),
        "final_loss": round(final_loss, 6),
        "accuracy": round(accuracy, 6),
    }
    artifact = artifact_from_logits(
        spec=KARATE_KAGGLE_CPU_SPEC,
        logits=logits,
        model={
            "type": "one-layer GCN",
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
        },
        execution=execution,
    )
    checksum_path, digest = write_artifact(
        arguments.output, artifact, force=arguments.force
    )
    print(json.dumps({
        "status": "PASS",
        "poc_id": KARATE_KAGGLE_CPU_SPEC.poc_id,
        "artifact": str(arguments.output),
        "checksum": str(checksum_path),
        "sha256": digest,
        "device": actual_device,
        "cpu_model": execution["cpu_model"],
        "cuda_available": execution["cuda_available"],
        "accuracy": execution["accuracy"],
        "predictions_exported": len(artifact["predictions"]),
    }, indent=2, sort_keys=True))
    return 0


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (ProofError, OSError, RuntimeError, ValueError) as error:
        print(f"KAGGLE KARATE CPU POC FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
