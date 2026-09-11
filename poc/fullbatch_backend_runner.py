#!/usr/bin/env python3
"""Matched single-T4 Flickr full-batch runner for plain PyG and cuGraph-PyG."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as distributed
import torch_geometric
from torch import Tensor
from torch_geometric.data import Data

from cugraph_pyg_flickr_runner import initialize_cugraph
from flickr_core import (
    EXPECTED_EDGES,
    EXPECTED_NODES,
    source_graph,
    train_on_device,
)
from kaggle_cpu_runtime import source_revision
from kaggle_specs import (
    FLICKR_FULLBATCH_CUGRAPH_PYG_KAGGLE_SPEC,
    FLICKR_FULLBATCH_PYG_KAGGLE_SPEC,
)
from proof_common import ProofError, require
from result_artifact import artifact_from_logits, write_artifact


Backend = Literal["pyg", "cugraph-pyg"]
EXPECTED_CUGRAPH_RELEASE = "26.2"
EXPECTED_CUGRAPH_PYG_VERSION = "26.2.1"
EXPECTED_PYG_VERSION = "2.7.0"


def parse_arguments(
    backend: Backend, argv: Sequence[str] | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    output_name = (
        "flickr-fullbatch-pyg-result.json"
        if backend == "pyg"
        else "flickr-fullbatch-cugraph-pyg-result.json"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(f"/kaggle/temp/flickr-fullbatch-{backend}/data"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/kaggle/working") / output_name
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=256)
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


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def framework_versions() -> dict[str, str]:
    versions = {
        "torch_geometric": torch_geometric.__version__,
        "cugraph_pyg": package_version("cugraph-pyg-cu12"),
        "pylibcugraph": package_version("pylibcugraph-cu12"),
        "pylibwholegraph": package_version("pylibwholegraph-cu12"),
    }
    require(
        versions["torch_geometric"] == EXPECTED_PYG_VERSION,
        "The installed PyG version does not match the benchmark pin",
    )
    require(
        versions["cugraph_pyg"] == EXPECTED_CUGRAPH_PYG_VERSION,
        "The installed cuGraph-PyG version does not match the benchmark pin",
    )
    for package in ("pylibcugraph", "pylibwholegraph"):
        require(
            versions[package].startswith(EXPECTED_CUGRAPH_RELEASE + "."),
            f"{package} is outside the aligned RAPIDS release family",
        )
    return versions


def edge_keys(edge_index: Tensor) -> Tensor:
    edge_index = edge_index.to(device="cpu", dtype=torch.int64)
    return torch.sort(edge_index[0] * EXPECTED_NODES + edge_index[1]).values


def verify_exact_graph(candidate: Data, reference: Data) -> None:
    require(candidate.num_nodes == EXPECTED_NODES, "Full-batch node count changed")
    require(candidate.num_edges == EXPECTED_EDGES, "Full-batch edge count changed")
    require(
        tuple(candidate.x.shape) == tuple(reference.x.shape),
        "Full-batch feature shape changed",
    )
    require(torch.equal(candidate.x, reference.x), "Full-batch features changed")
    require(torch.equal(candidate.y, reference.y), "Full-batch labels changed")
    require(
        torch.equal(edge_keys(candidate.edge_index), edge_keys(reference.edge_index)),
        "Full-batch edge multiset changed",
    )
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        require(
            torch.equal(getattr(candidate, mask_name), getattr(reference, mask_name)),
            f"Full-batch {mask_name} changed",
        )


def materialize_plain(graph: Data) -> tuple[Data, dict[str, object]]:
    prepared = graph.clone()
    verify_exact_graph(prepared, graph)
    return prepared, {
        "backend": "pyg-direct-full-batch",
        "exact_full_graph_verified": True,
        "full_neighbor_fanout": None,
        "materialized_batches": 0,
        "cugraph_comms_initialized": False,
    }


def materialize_cugraph(graph: Data) -> tuple[Data, dict[str, object]]:
    from cugraph_pyg.data import FeatureStore, GraphStore
    from cugraph_pyg.loader import NeighborLoader

    shutdown_cugraph, shutdown_torch = initialize_cugraph()
    try:
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
        all_nodes = torch.arange(graph.num_nodes, dtype=torch.long)
        loader = NeighborLoader(
            (feature_store, graph_store),
            [-1],
            input_nodes=all_nodes,
            batch_size=graph.num_nodes,
            shuffle=False,
            drop_last=False,
        )
        iterator = iter(loader)
        batch = next(iterator)
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ProofError("cuGraph-PyG produced more than one full batch")
        require(int(batch.batch_size) == EXPECTED_NODES, "Seed batch is incomplete")
        require(hasattr(batch, "n_id"), "cuGraph-PyG batch has no global node IDs")
        node_ids = batch.n_id.detach().to(device="cpu", dtype=torch.long)
        require(
            node_ids.numel() == EXPECTED_NODES,
            "cuGraph-PyG did not materialize every node",
        )
        require(
            torch.equal(torch.sort(node_ids).values, all_nodes),
            "cuGraph-PyG node IDs are incomplete",
        )
        batch_edges = batch.edge_index.detach().to(device="cpu", dtype=torch.long)
        global_edges = node_ids[batch_edges]
        features = torch.empty_like(graph.x)
        labels = torch.empty_like(graph.y)
        features[node_ids] = batch.x.detach().to("cpu")
        labels[node_ids] = batch.y.detach().to("cpu")
        prepared = Data(
            x=features,
            y=labels,
            edge_index=global_edges,
            train_mask=graph.train_mask.clone(),
            val_mask=graph.val_mask.clone(),
            test_mask=graph.test_mask.clone(),
        )
        verify_exact_graph(prepared, graph)
        identity = {
            "backend": "cugraph-pyg-full-neighbor",
            "exact_full_graph_verified": True,
            "full_neighbor_fanout": [-1],
            "materialized_batches": 1,
            "cugraph_comms_initialized": True,
            "distributed_backend": distributed.get_backend(),
            "distributed_world_size": distributed.get_world_size(),
            "loader_class": type(loader).__name__,
            "loader_module": type(loader).__module__,
            "graph_store_class": type(graph_store).__name__,
            "graph_store_module": type(graph_store).__module__,
            "feature_store_class": type(feature_store).__name__,
            "feature_store_module": type(feature_store).__module__,
        }
        return prepared, identity
    finally:
        try:
            shutdown_cugraph()
        finally:
            shutdown_torch()


def main(
    backend: Backend, argv: Sequence[str] | None = None
) -> int:
    arguments = parse_arguments(backend, argv)
    validate_arguments(arguments)
    revision = source_revision()
    device = require_single_t4()
    versions = framework_versions()
    graph = source_graph(arguments.data_root)

    torch.cuda.synchronize()
    preparation_started = time.perf_counter()
    if backend == "pyg":
        prepared_graph, backend_evidence = materialize_plain(graph)
        spec = FLICKR_FULLBATCH_PYG_KAGGLE_SPEC
    else:
        prepared_graph, backend_evidence = materialize_cugraph(graph)
        spec = FLICKR_FULLBATCH_CUGRAPH_PYG_KAGGLE_SPEC
    torch.cuda.synchronize()
    preparation_seconds = time.perf_counter() - preparation_started
    require(
        math.isfinite(preparation_seconds) and preparation_seconds > 0,
        "Backend preparation time is invalid",
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    logits, metrics = train_on_device(
        prepared_graph,
        arguments.epochs,
        arguments.patience,
        arguments.hidden_channels,
        arguments.dropout,
        device,
        arguments.seed,
        arguments.learning_rate,
        arguments.weight_decay,
        optimizer_name="adam",
    )
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    total_seconds = preparation_seconds + float(metrics["training_seconds"])
    execution = {
        "status": "PASS",
        "device": str(device),
        "accuracy": metrics["test_accuracy"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_geometric": versions["torch_geometric"],
        "cugraph_pyg": versions["cugraph_pyg"],
        "pylibcugraph": versions["pylibcugraph"],
        "pylibwholegraph": versions["pylibwholegraph"],
        "source_revision": revision,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_peak_memory_bytes": peak_allocated,
        "cuda_peak_reserved_memory_bytes": peak_reserved,
        "cuda_device_total_memory_bytes": total_memory,
        "cuda_peak_allocated_fraction": round(peak_allocated / total_memory, 6),
        "cuda_peak_reserved_fraction": round(peak_reserved / total_memory, 6),
        "backend_preparation_seconds": round(preparation_seconds, 6),
        "backend_inclusive_seconds": round(total_seconds, 6),
        **backend_evidence,
        **metrics,
    }
    model = {
        "type": "three-layer GraphSAGE",
        "execution_mode": "full-batch",
        "epochs_requested": arguments.epochs,
        "patience": arguments.patience,
        "hidden_channels": arguments.hidden_channels,
        "dropout": arguments.dropout,
        "seed": arguments.seed,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "optimizer": "adam",
    }
    artifact = artifact_from_logits(
        spec=spec, logits=logits, model=model, execution=execution
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
                "backend": execution["backend"],
                "training_seconds": execution["training_seconds"],
                "backend_preparation_seconds": execution[
                    "backend_preparation_seconds"
                ],
                "test_accuracy": execution["test_accuracy"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def cli(backend: Backend, argv: Sequence[str] | None = None) -> int:
    try:
        return main(backend, argv)
    except (ProofError, OSError, RuntimeError, ValueError) as error:
        print(
            f"KAGGLE FULL-BATCH {backend.upper()} POC FAILED: {error}",
            file=sys.stderr,
        )
        return 1
