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
    FLICKR_2048_KAGGLE_CPU_SPEC,
    FLICKR_2048_KAGGLE_CUDA_SPEC,
    FLICKR_2048_MINIMUM_CUDA_PEAK_BYTES,
    FLICKR_4096_KAGGLE_CPU_SPEC,
    FLICKR_4096_KAGGLE_CUDA_SPEC,
    FLICKR_4096_MINIMUM_CUDA_PEAK_BYTES,
    FLICKR_KAGGLE_CPU_SPEC,
    FLICKR_KAGGLE_CUDA_SPEC,
    FLICKR_WIDE_KAGGLE_CPU_SPEC,
    FLICKR_WIDE_KAGGLE_CUDA_SPEC,
    FLICKR_WIDE_MINIMUM_CUDA_PEAK_BYTES,
)
from proof_common import ProofError, require
from result_artifact import artifact_from_logits, write_artifact


DeviceProfile = Literal["cpu", "cuda"]
BenchmarkVariant = Literal["baseline", "wide", "2048", "4096"]
VARIANT_HIDDEN_CHANNELS = {
    "baseline": 256,
    "wide": 1_024,
    "2048": 2_048,
    "4096": 4_096,
}
VARIANT_EPOCHS = {"baseline": 30, "wide": 20, "2048": 20, "4096": 20}
VARIANT_PATIENCE = {"baseline": 8, "wide": 6, "2048": 6, "4096": 6}
VARIANT_EDGE_CHUNK_SIZES = {"2048": 262_144, "4096": 131_072}
CHECKPOINTED_VARIANTS = {"4096"}


def parse_arguments(
    profile: DeviceProfile,
    argv: Sequence[str] | None = None,
    *,
    variant: BenchmarkVariant = "baseline",
) -> argparse.Namespace:
    name = "flickr" if variant == "baseline" else f"flickr-{variant}"
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=VARIANT_EPOCHS[variant],
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=VARIANT_PATIENCE[variant],
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=VARIANT_HIDDEN_CHANNELS[variant],
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
        VARIANT_EDGE_CHUNK_SIZES.get(variant),
        variant in CHECKPOINTED_VARIANTS,
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
        cpu_specs = {
            "baseline": FLICKR_KAGGLE_CPU_SPEC,
            "wide": FLICKR_WIDE_KAGGLE_CPU_SPEC,
            "2048": FLICKR_2048_KAGGLE_CPU_SPEC,
            "4096": FLICKR_4096_KAGGLE_CPU_SPEC,
        }
        spec = cpu_specs[variant]
    else:
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        minimum_peak = {
            "wide": FLICKR_WIDE_MINIMUM_CUDA_PEAK_BYTES,
            "2048": FLICKR_2048_MINIMUM_CUDA_PEAK_BYTES,
            "4096": FLICKR_4096_MINIMUM_CUDA_PEAK_BYTES,
        }.get(variant)
        if minimum_peak is not None:
            require(
                peak_allocated >= minimum_peak,
                f"{variant} CUDA peak {peak_allocated} bytes is below "
                f"the {minimum_peak}-byte memory target",
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
        cuda_specs = {
            "baseline": FLICKR_KAGGLE_CUDA_SPEC,
            "wide": FLICKR_WIDE_KAGGLE_CUDA_SPEC,
            "2048": FLICKR_2048_KAGGLE_CUDA_SPEC,
            "4096": FLICKR_4096_KAGGLE_CUDA_SPEC,
        }
        spec = cuda_specs[variant]

    model = {
        "type": "three-layer GraphSAGE",
        "epochs_requested": arguments.epochs,
        "patience": arguments.patience,
        "hidden_channels": arguments.hidden_channels,
        "dropout": arguments.dropout,
        "seed": arguments.seed,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
    }
    if variant != "baseline":
        model["benchmark_variant"] = variant
    if variant in {"2048", "4096"}:
        model["edge_chunk_size"] = VARIANT_EDGE_CHUNK_SIZES[variant]
        model["aggregation"] = "exact chunked mean"
    if variant == "4096":
        model["activation_checkpointing"] = True
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
