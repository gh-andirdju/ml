"""Shared implementation for the ready Kaggle Flickr CPU and CUDA runners."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
import torch_geometric

from flickr_core import source_graph, train_on_device
from kaggle_cpu_runtime import cpu_model, require_cpu_only, source_revision
from kaggle_specs import (
    FLICKR_KAGGLE_CPU_SPEC,
    FLICKR_KAGGLE_CUDA_SPEC,
    FLICKR_WIDE_KAGGLE_CPU_SPEC,
    FLICKR_WIDE_KAGGLE_CUDA_SPEC,
)
from proof_common import ProofError, require
from result_artifact import artifact_from_logits, write_artifact


DeviceProfile = Literal["cpu", "cuda"]
BenchmarkVariant = Literal["baseline", "wide"]
WIDE_HIDDEN_CHANNELS = 1_024
WIDE_EPOCHS = 20
WIDE_PATIENCE = 6
MINIMUM_WIDE_CUDA_PEAK_BYTES = 4 * 1024**3


def parse_arguments(
    profile: DeviceProfile,
    argv: Sequence[str] | None = None,
    *,
    variant: BenchmarkVariant = "baseline",
) -> argparse.Namespace:
    wide = variant == "wide"
    name = "flickr-wide" if wide else "flickr"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"/kaggle/working/{name}-{profile}-result.json"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(f"/kaggle/temp/ml-{name}-{profile}/data"),
    )
    parser.add_argument("--epochs", type=int, default=WIDE_EPOCHS if wide else 30)
    parser.add_argument("--patience", type=int, default=WIDE_PATIENCE if wide else 8)
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=WIDE_HIDDEN_CHANNELS if wide else 256,
    )
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def validate_arguments(arguments: argparse.Namespace) -> None:
    require(arguments.epochs > 0, "Epoch count must be positive")
    require(arguments.patience > 0, "Patience must be positive")
    require(arguments.hidden_channels > 0, "Hidden channels must be positive")
    require(0 <= arguments.dropout < 1, "Dropout must be in [0, 1)")
    require(arguments.learning_rate > 0, "Learning rate must be positive")
    require(arguments.weight_decay >= 0, "Weight decay cannot be negative")


def selected_device(profile: DeviceProfile) -> torch.device:
    if profile == "cpu":
        return require_cpu_only()
    require(torch.cuda.is_available(), "CUDA is unavailable; this POC is GPU-only")
    return torch.device("cuda:0")


def main(
    profile: DeviceProfile,
    argv: Sequence[str] | None = None,
    *,
    variant: BenchmarkVariant = "baseline",
) -> int:
    arguments = parse_arguments(profile, argv, variant=variant)
    validate_arguments(arguments)
    revision = source_revision()
    device = selected_device(profile)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    logits, metrics = train_on_device(
        source_graph(arguments.data_root),
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
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "torch_cuda": torch.version.cuda,
        "source_revision": revision,
        "accuracy": metrics["test_accuracy"],
        **metrics,
    }
    if device.type == "cpu":
        execution.update(
            {
                "cpu_model": cpu_model(),
                "cpu_count": os.cpu_count(),
                "cuda_available": torch.cuda.is_available(),
            }
        )
        spec = (
            FLICKR_WIDE_KAGGLE_CPU_SPEC
            if variant == "wide"
            else FLICKR_KAGGLE_CPU_SPEC
        )
    else:
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        if variant == "wide":
            require(
                peak_allocated >= MINIMUM_WIDE_CUDA_PEAK_BYTES,
                "Wide CUDA workload did not reach its 4 GiB memory target",
            )
        execution.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_capability": list(torch.cuda.get_device_capability(device)),
                "cuda_peak_memory_bytes": peak_allocated,
                "cuda_peak_reserved_memory_bytes": peak_reserved,
                "cuda_device_total_memory_bytes": total_memory,
                "cuda_peak_allocated_fraction": round(
                    peak_allocated / total_memory, 6
                ),
                "cuda_peak_reserved_fraction": round(peak_reserved / total_memory, 6),
            }
        )
        spec = (
            FLICKR_WIDE_KAGGLE_CUDA_SPEC
            if variant == "wide"
            else FLICKR_KAGGLE_CUDA_SPEC
        )

    model = {
        "type": "three-layer GraphSAGE",
        "benchmark_variant": variant,
        "epochs_requested": arguments.epochs,
        "patience": arguments.patience,
        "hidden_channels": arguments.hidden_channels,
        "dropout": arguments.dropout,
        "seed": arguments.seed,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
    }
    artifact = artifact_from_logits(
        spec=spec, logits=logits, model=model, execution=execution
    )
    checksum_path, digest = write_artifact(
        arguments.output, artifact, force=arguments.force
    )
    summary = {
        "status": "PASS",
        "poc_id": spec.poc_id,
        "artifact": str(arguments.output),
        "checksum": str(checksum_path),
        "sha256": digest,
        "device": execution["device"],
        "validation_accuracy": metrics["validation_accuracy"],
        "test_accuracy": metrics["test_accuracy"],
        "training_seconds": metrics["training_seconds"],
        "epochs_completed": metrics["epochs_completed"],
        "predictions_exported": len(artifact["predictions"]),
    }
    if profile == "cpu":
        summary["cpu_model"] = execution["cpu_model"]
    else:
        summary["cuda_device_name"] = execution["cuda_device_name"]
        summary["cuda_peak_memory_bytes"] = execution["cuda_peak_memory_bytes"]
        summary["cuda_peak_reserved_memory_bytes"] = execution[
            "cuda_peak_reserved_memory_bytes"
        ]
        summary["cuda_device_total_memory_bytes"] = execution[
            "cuda_device_total_memory_bytes"
        ]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cli(
    profile: DeviceProfile,
    argv: Sequence[str] | None = None,
    *,
    variant: BenchmarkVariant = "baseline",
) -> int:
    try:
        return main(profile, argv, variant=variant)
    except (ProofError, OSError, RuntimeError, ValueError) as error:
        print(f"KAGGLE FLICKR {profile.upper()} POC FAILED: {error}", file=sys.stderr)
        return 1
