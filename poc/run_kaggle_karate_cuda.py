#!/usr/bin/env python3
"""POC 3: train Karate Club on CUDA and export predictions for local Neo4j."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch_geometric

from kaggle_specs import KARATE_KAGGLE_SPEC
from karate_core import train_on_device, source_graph
from poc_runtime import ProofError, require
from result_artifact import artifact_from_logits, write_artifact


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/karate-cuda-result.json"))
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
    require(torch.cuda.is_available(), "CUDA is unavailable; this POC is GPU-only")
    device = torch.device("cuda:0")
    graph = source_graph()
    logits, initial_loss, final_loss, accuracy, actual_device = train_on_device(
        graph,
        arguments.epochs,
        device,
        arguments.seed,
        arguments.learning_rate,
        arguments.weight_decay,
    )
    execution = {
        "status": "PASS",
        "device": actual_device,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "epochs": arguments.epochs,
        "seed": arguments.seed,
        "initial_loss": round(initial_loss, 6),
        "final_loss": round(final_loss, 6),
        "accuracy": round(accuracy, 6),
    }
    artifact = artifact_from_logits(
        spec=KARATE_KAGGLE_SPEC,
        logits=logits,
        model={
            "type": "one-layer GCN",
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
        },
        execution=execution,
    )
    checksum_path, digest = write_artifact(arguments.output, artifact, force=arguments.force)
    print(json.dumps({
        "status": "PASS",
        "poc_id": KARATE_KAGGLE_SPEC.poc_id,
        "artifact": str(arguments.output),
        "checksum": str(checksum_path),
        "sha256": digest,
        "device": actual_device,
        "cuda_device_name": execution["cuda_device_name"],
        "accuracy": execution["accuracy"],
        "predictions_exported": len(artifact["predictions"]),
    }, indent=2, sort_keys=True))
    return 0


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (ProofError, OSError, RuntimeError, ValueError) as error:
        print(f"KAGGLE KARATE POC FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
