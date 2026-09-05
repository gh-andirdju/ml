"""Device-neutral WikiCS dataset, model, and training logic."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import WikiCS
from torch_geometric.nn import GCNConv

from proof_common import require


DATASET_COMMIT = "f5207315d649377f936edb66d7d93f5342f01d81"
DATASET_SHA256 = "9bf8cb3ef8eeae81b25e6ccbe0ea195600c205d7edf63ce04f2ec8d9c7dcb3d8"
EXPECTED_NODES = 11_701
EXPECTED_DIRECTED_EDGES = 431_726
EXPECTED_RELATIONSHIPS = 216_123
EXPECTED_FEATURES = 300
EXPECTED_CLASSES = 10
EXPECTED_SPLITS = 20
EXPECTED_TRAINING_NODES = 580
EXPECTED_VALIDATION_NODES = 1_769
EXPECTED_STOPPING_NODES = 3_505
EXPECTED_TEST_NODES = 5_847
MINIMUM_GENERALIZATION_ACCURACY = 0.50


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class PinnedWikiCS(WikiCS):
    """WikiCS loader pinned to one immutable upstream Git commit."""

    url = (
        "https://raw.githubusercontent.com/pmernyei/wiki-cs-dataset/"
        f"{DATASET_COMMIT}/dataset"
    )

    def process(self) -> None:
        raw_path = Path(self.raw_paths[0])
        require(
            file_sha256(raw_path) == DATASET_SHA256,
            f"WikiCS data checksum mismatch: {raw_path}",
        )
        super().process()


class WikiGCN(torch.nn.Module):
    """A full-batch, two-layer GCN for semi-supervised node classification."""

    def __init__(self, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.input_convolution = GCNConv(EXPECTED_FEATURES, hidden_channels, cached=True)
        self.output_convolution = GCNConv(hidden_channels, EXPECTED_CLASSES, cached=True)

    def forward(self, graph: Data) -> Tensor:
        features = self.input_convolution(graph.x, graph.edge_index).relu()
        features = functional.dropout(
            features, p=self.dropout, training=self.training
        )
        return self.output_convolution(features, graph.edge_index)


def source_graph(data_root: Path, split: int) -> Data:
    dataset = PinnedWikiCS(str(data_root), is_undirected=True, force_reload=True)
    raw_path = Path(dataset.raw_paths[0])
    require(
        file_sha256(raw_path) == DATASET_SHA256,
        f"WikiCS data checksum mismatch: {raw_path}",
    )
    graph = dataset[0]
    require(graph.num_nodes == EXPECTED_NODES, "Unexpected WikiCS node count")
    require(graph.num_edges == EXPECTED_DIRECTED_EDGES, "Unexpected WikiCS edge count")
    require(graph.num_node_features == EXPECTED_FEATURES, "Unexpected feature count")
    require(int(graph.y.unique().numel()) == EXPECTED_CLASSES, "Unexpected class count")
    require(tuple(graph.train_mask.shape) == (EXPECTED_NODES, EXPECTED_SPLITS), "Bad train masks")
    require(
        tuple(graph.val_mask.shape) == (EXPECTED_NODES, EXPECTED_SPLITS),
        "Bad validation masks",
    )
    require(
        tuple(graph.stopping_mask.shape) == (EXPECTED_NODES, EXPECTED_SPLITS),
        "Bad stopping masks",
    )
    require(
        int(graph.train_mask[:, split].sum()) == EXPECTED_TRAINING_NODES,
        "Bad train split",
    )
    require(
        int(graph.val_mask[:, split].sum()) == EXPECTED_VALIDATION_NODES,
        "Bad validation split",
    )
    require(
        int(graph.stopping_mask[:, split].sum()) == EXPECTED_STOPPING_NODES,
        "Bad stopping split",
    )
    require(int(graph.test_mask.sum()) == EXPECTED_TEST_NODES, "Bad test split")
    require(graph.is_undirected(), "WikiCS graph is not undirected")
    return Data(
        x=graph.x,
        edge_index=graph.edge_index,
        y=graph.y,
        train_mask=graph.train_mask[:, split],
        val_mask=graph.val_mask[:, split],
        stopping_mask=graph.stopping_mask[:, split],
        test_mask=graph.test_mask,
    )


def masked_accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> float:
    return float((logits[mask].argmax(dim=1) == labels[mask]).float().mean().item())


def train_on_device(
    graph: Data,
    epochs: int,
    patience: int,
    hidden_channels: int,
    dropout: float,
    device: torch.device,
    seed: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[Tensor, dict[str, float | int | str]]:
    if device.type == "mps":
        require(
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1",
            "CPU fallback is enabled; this would not prove MPS execution",
        )

    torch.manual_seed(seed)
    device_graph = graph.to(device)
    model = WikiGCN(hidden_channels, dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    model.eval()
    with torch.no_grad():
        initial_logits = model(device_graph)
        initial_training_loss = functional.cross_entropy(
            initial_logits[device_graph.train_mask],
            device_graph.y[device_graph.train_mask],
        ).item()

    best_stopping_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(device_graph)
        training_loss = functional.cross_entropy(
            logits[device_graph.train_mask], device_graph.y[device_graph.train_mask]
        )
        training_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            stopping_logits = model(device_graph)
            stopping_loss = functional.cross_entropy(
                stopping_logits[device_graph.stopping_mask],
                device_graph.y[device_graph.stopping_mask],
            ).item()
        if stopping_loss < best_stopping_loss - 1e-6:
            best_stopping_loss = stopping_loss
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    require(best_state is not None and best_epoch > 0, "Early stopping found no model")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(device_graph)
        final_training_loss = functional.cross_entropy(
            final_logits[device_graph.train_mask],
            device_graph.y[device_graph.train_mask],
        ).item()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    actual_device = final_logits.device
    require(next(model.parameters()).device == actual_device, "Model device mismatch")
    require(device_graph.x.device == actual_device, "Feature device mismatch")
    require(actual_device.type == device.type, "Requested device type was not used")
    if device.index is not None:
        require(actual_device.index == device.index, "Requested device index was not used")
    require(
        tuple(final_logits.shape) == (EXPECTED_NODES, EXPECTED_CLASSES),
        "Output shape mismatch",
    )
    require(
        math.isfinite(initial_training_loss)
        and math.isfinite(final_training_loss)
        and math.isfinite(best_stopping_loss),
        "Loss is not finite",
    )
    require(final_training_loss < initial_training_loss, "Training loss did not decrease")

    validation_accuracy = masked_accuracy(
        final_logits, device_graph.y, device_graph.val_mask
    )
    test_accuracy = masked_accuracy(final_logits, device_graph.y, device_graph.test_mask)
    require(
        validation_accuracy >= MINIMUM_GENERALIZATION_ACCURACY,
        "Validation accuracy is below the proof threshold",
    )
    require(
        test_accuracy >= MINIMUM_GENERALIZATION_ACCURACY,
        "Test accuracy is below the proof threshold",
    )

    metrics: dict[str, float | int | str] = {
        "device": str(actual_device),
        "epochs_completed": epoch,
        "best_epoch": best_epoch,
        "initial_training_loss": round(initial_training_loss, 6),
        "final_training_loss": round(final_training_loss, 6),
        "best_stopping_loss": round(best_stopping_loss, 6),
        "validation_accuracy": round(validation_accuracy, 6),
        "test_accuracy": round(test_accuracy, 6),
    }
    return final_logits.cpu(), metrics
