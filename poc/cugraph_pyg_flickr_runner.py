#!/usr/bin/env python3
"""Single-T4 Flickr GraphSAGE proof using cuGraph-PyG sampling."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import platform
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
import torch_geometric
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from flickr_core import (
    EXPECTED_CLASSES,
    EXPECTED_EDGES,
    EXPECTED_FEATURES,
    EXPECTED_NODES,
    MINIMUM_GENERALIZATION_ACCURACY,
    masked_accuracy,
    source_graph,
)
from kaggle_cpu_runtime import source_revision
from kaggle_specs import CUGRAPH_PYG_FLICKR_KAGGLE_SPEC
from proof_common import require
from result_artifact import artifact_from_logits, write_artifact


DEFAULT_FANOUT = (15, 10, 5)


class SampledGraphSAGE(torch.nn.Module):
    """Three-layer PyG GraphSAGE model that consumes sampled PyG batches."""

    def __init__(self, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.convolutions = torch.nn.ModuleList(
            [
                SAGEConv(EXPECTED_FEATURES, hidden_channels),
                SAGEConv(hidden_channels, hidden_channels),
                SAGEConv(hidden_channels, EXPECTED_CLASSES),
            ]
        )

    def forward(self, features: Tensor, edge_index: Tensor) -> Tensor:
        for convolution in self.convolutions[:-1]:
            features = convolution(features, edge_index).relu()
            features = functional.dropout(
                features, p=self.dropout, training=self.training
            )
        return self.convolutions[-1](features, edge_index)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Flickr GraphSAGE with cuGraph-PyG on one Kaggle T4"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/kaggle/temp/flickr-cugraph-data")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/cugraph-pyg-flickr-result.json"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fanout", type=int, nargs=3, default=list(DEFAULT_FANOUT)
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def validate_arguments(arguments: argparse.Namespace) -> None:
    require(arguments.epochs > 0, "Epoch count must be positive")
    require(arguments.patience > 0, "Patience must be positive")
    require(arguments.batch_size > 0, "Batch size must be positive")
    require(arguments.hidden_channels > 0, "Hidden channels must be positive")
    require(0 <= arguments.dropout < 1, "Dropout must be in [0, 1)")
    require(arguments.learning_rate > 0, "Learning rate must be positive")
    require(arguments.weight_decay >= 0, "Weight decay cannot be negative")
    require(
        len(arguments.fanout) == 3 and all(value > 0 for value in arguments.fanout),
        "Exactly three positive fanout values are required",
    )


def require_single_t4() -> torch.device:
    require(torch.cuda.is_available(), "CUDA is unavailable; this POC is GPU-only")
    require(torch.cuda.device_count() == 1, "This POC requires exactly one CUDA GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    require("T4" in name, f"Expected a Tesla T4, found {name}")
    require(capability == (7, 5), f"Expected CUDA capability 7.5, found {capability}")
    return device


def initialize_cugraph() -> tuple[object, object]:
    """Initialize the single-worker NCCL and pylibcugraph communicators."""
    from pylibcugraph.comms import (
        cugraph_comms_create_unique_id,
        cugraph_comms_init,
        cugraph_comms_shutdown,
    )

    require(not distributed.is_initialized(), "PyTorch distributed is already active")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    distributed.init_process_group("nccl", rank=0, world_size=1)
    try:
        cugraph_comms_init(
            rank=0,
            world_size=1,
            uid=cugraph_comms_create_unique_id(),
            device=0,
        )
    except Exception:
        distributed.destroy_process_group()
        raise
    return cugraph_comms_shutdown, distributed.destroy_process_group


def create_loader(
    graph: Data, arguments: argparse.Namespace
) -> tuple[object, dict[str, str]]:
    """Create a cuGraph-backed PyG loader without a fallback sampler."""
    from cugraph_pyg.data import FeatureStore, GraphStore
    from cugraph_pyg.loader import NeighborLoader

    graph_store = GraphStore()
    graph_store.put_edge_index(
        graph.edge_index,
        ("node", "connects", "node"),
        "coo",
        False,
        (graph.num_nodes, graph.num_nodes),
    )
    feature_store = FeatureStore()
    feature_store["node", "x", None] = graph.x
    feature_store["node", "y", None] = graph.y
    training_nodes = graph.train_mask.nonzero(as_tuple=False).view(-1)
    loader = NeighborLoader(
        (feature_store, graph_store),
        list(arguments.fanout),
        input_nodes=training_nodes,
        batch_size=arguments.batch_size,
        shuffle=True,
        drop_last=False,
    )
    identity = {
        "loader_class": type(loader).__name__,
        "loader_module": type(loader).__module__,
        "graph_store_class": type(graph_store).__name__,
        "graph_store_module": type(graph_store).__module__,
        "feature_store_class": type(feature_store).__name__,
        "feature_store_module": type(feature_store).__module__,
    }
    require(
        identity["loader_module"].startswith("cugraph_pyg.loader"),
        "The active neighbor loader is not provided by cuGraph-PyG",
    )
    require(
        identity["graph_store_module"].startswith("cugraph_pyg.data"),
        "The active graph store is not provided by cuGraph-PyG",
    )
    return loader, identity


@torch.no_grad()
def evaluate(
    model: SampledGraphSAGE, graph: Data
) -> tuple[Tensor, float, float, float]:
    model.eval()
    logits = model(graph.x, graph.edge_index)
    validation_loss = float(
        functional.cross_entropy(logits[graph.val_mask], graph.y[graph.val_mask])
    )
    validation_accuracy = masked_accuracy(logits, graph.y, graph.val_mask)
    test_accuracy = masked_accuracy(logits, graph.y, graph.test_mask)
    return logits, validation_loss, validation_accuracy, test_accuracy


def train(
    graph: Data,
    loader: object,
    arguments: argparse.Namespace,
    device: torch.device,
) -> tuple[Tensor, dict[str, float | int]]:
    torch.manual_seed(arguments.seed)
    model = SampledGraphSAGE(arguments.hidden_channels, arguments.dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    device_graph = graph.to(device)
    best_state: dict[str, Tensor] | None = None
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    sampling_seconds = 0.0
    optimization_seconds = 0.0
    validation_seconds = 0.0
    total_batches = 0
    total_seed_nodes = 0
    total_sampled_nodes = 0
    total_sampled_edges = 0
    initial_training_loss: float | None = None
    final_training_loss = math.inf
    started_at = time.perf_counter()

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_seeds = 0
        iterator = iter(loader)
        while True:
            torch.cuda.synchronize()
            sample_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            torch.cuda.synchronize()
            sampling_seconds += time.perf_counter() - sample_started
            require(
                type(batch).__module__.startswith("torch_geometric.data"),
                "cuGraph-PyG did not return a PyG data batch",
            )
            batch = batch.to(device)
            seed_count = int(batch.batch_size)
            require(seed_count > 0, "cuGraph-PyG returned an empty seed batch")

            torch.cuda.synchronize()
            optimization_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index)
            loss = functional.cross_entropy(
                logits[:seed_count], batch.y[:seed_count]
            )
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            optimization_seconds += time.perf_counter() - optimization_started

            loss_value = float(loss.detach())
            if initial_training_loss is None:
                initial_training_loss = loss_value
            epoch_loss += loss_value * seed_count
            epoch_seeds += seed_count
            total_batches += 1
            total_seed_nodes += seed_count
            total_sampled_nodes += int(batch.num_nodes)
            total_sampled_edges += int(batch.num_edges)

        require(epoch_seeds > 0, "cuGraph-PyG produced no training seeds")
        final_training_loss = epoch_loss / epoch_seeds
        torch.cuda.synchronize()
        validation_started = time.perf_counter()
        _, validation_loss, validation_accuracy, _ = evaluate(model, device_graph)
        torch.cuda.synchronize()
        validation_seconds += time.perf_counter() - validation_started
        print(
            f"epoch={epoch} training_loss={final_training_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"validation_accuracy={validation_accuracy:.6f}",
            flush=True,
        )
        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= arguments.patience:
                break

    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started_at
    require(best_state is not None and best_epoch > 0, "No best model was retained")
    model.load_state_dict(best_state)
    final_logits, _, validation_accuracy, test_accuracy = evaluate(model, device_graph)
    torch.cuda.synchronize()
    require(
        tuple(final_logits.shape) == (EXPECTED_NODES, EXPECTED_CLASSES),
        "Final logits shape is invalid",
    )
    require(
        validation_accuracy >= MINIMUM_GENERALIZATION_ACCURACY,
        "Validation accuracy is below the proof threshold",
    )
    require(
        test_accuracy >= MINIMUM_GENERALIZATION_ACCURACY,
        "Test accuracy is below the proof threshold",
    )
    require(
        initial_training_loss is not None
        and all(
            math.isfinite(value)
            for value in (
                initial_training_loss,
                final_training_loss,
                best_validation_loss,
                sampling_seconds,
                optimization_seconds,
                validation_seconds,
                training_seconds,
            )
        ),
        "A training metric is not finite",
    )
    metrics: dict[str, float | int] = {
        "epochs_completed": epoch,
        "best_epoch": best_epoch,
        "initial_training_loss": round(initial_training_loss, 6),
        "final_training_loss": round(final_training_loss, 6),
        "best_validation_loss": round(best_validation_loss, 6),
        "validation_accuracy": round(validation_accuracy, 6),
        "test_accuracy": round(test_accuracy, 6),
        "training_seconds": round(training_seconds, 6),
        "sampling_seconds": round(sampling_seconds, 6),
        "optimization_seconds": round(optimization_seconds, 6),
        "validation_seconds": round(validation_seconds, 6),
        "batches_processed": total_batches,
        "seed_nodes_processed": total_seed_nodes,
        "sampled_nodes_processed": total_sampled_nodes,
        "sampled_edges_processed": total_sampled_edges,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    return final_logits.cpu(), metrics


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    validate_arguments(arguments)
    revision = source_revision()
    device = require_single_t4()
    torch.cuda.reset_peak_memory_stats()
    shutdown_cugraph, shutdown_torch = initialize_cugraph()
    try:
        graph = source_graph(arguments.data_root)
        require(graph.num_edges == EXPECTED_EDGES, "Flickr edge count changed")
        loader, backend_identity = create_loader(graph, arguments)
        logits, metrics = train(graph, loader, arguments, device)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        execution = {
            "status": "PASS",
            "device": str(device),
            "accuracy": metrics["test_accuracy"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "torch_cuda": torch.version.cuda,
            "cugraph_pyg": package_version("cugraph-pyg-cu12"),
            "pylibcugraph": package_version("pylibcugraph-cu12"),
            "pylibwholegraph": package_version("pylibwholegraph-cu12"),
            "source_revision": revision,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_peak_memory_bytes": peak_allocated,
            "cuda_peak_reserved_memory_bytes": peak_reserved,
            "cuda_device_total_memory_bytes": total_memory,
            "cuda_peak_allocated_fraction": round(peak_allocated / total_memory, 6),
            "cuda_peak_reserved_fraction": round(peak_reserved / total_memory, 6),
            "cugraph_comms_initialized": True,
            "distributed_backend": distributed.get_backend(),
            "distributed_world_size": distributed.get_world_size(),
            **backend_identity,
            **metrics,
        }
        model = {
            "type": "three-layer sampled GraphSAGE",
            "framework": "PyTorch Geometric",
            "sampling_backend": "cuGraph-PyG",
            "epochs_requested": arguments.epochs,
            "patience": arguments.patience,
            "hidden_channels": arguments.hidden_channels,
            "dropout": arguments.dropout,
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
            "seed": arguments.seed,
            "batch_size": arguments.batch_size,
            "fanout": list(arguments.fanout),
        }
        artifact = artifact_from_logits(
            spec=CUGRAPH_PYG_FLICKR_KAGGLE_SPEC,
            logits=logits,
            model=model,
            execution=execution,
        )
        checksum_path, digest = write_artifact(
            arguments.output, artifact, force=arguments.force
        )
        print(
            {
                "status": "PASS",
                "poc_id": CUGRAPH_PYG_FLICKR_KAGGLE_SPEC.poc_id,
                "artifact": str(arguments.output),
                "checksum": str(checksum_path),
                "sha256": digest,
                "device": str(device),
                "test_accuracy": metrics["test_accuracy"],
                "batches_processed": metrics["batches_processed"],
                "sampling_seconds": metrics["sampling_seconds"],
                "training_seconds": metrics["training_seconds"],
            },
            flush=True,
        )
    finally:
        try:
            shutdown_cugraph()
        finally:
            shutdown_torch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
