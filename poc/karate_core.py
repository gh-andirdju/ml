"""Device-neutral Karate Club dataset, model, and training logic."""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv

from proof_common import require


EXPECTED_NODES = 34
EXPECTED_DIRECTED_EDGES = 156
EXPECTED_RELATIONSHIPS = 78
EXPECTED_FEATURES = 34
EXPECTED_CLASSES = 4
EXPECTED_TRAINING_NODES = 4


class SmallestGCN(torch.nn.Module):
    """A single graph-convolution layer for four-class node prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = GCNConv(EXPECTED_FEATURES, EXPECTED_CLASSES)

    def forward(self, graph: Data) -> Tensor:
        return self.convolution(graph.x, graph.edge_index)


def source_graph() -> Data:
    graph = KarateClub()[0]
    require(graph.num_nodes == EXPECTED_NODES, "Unexpected source node count")
    require(graph.num_edges == EXPECTED_DIRECTED_EDGES, "Unexpected source edge count")
    require(tuple(graph.x.shape) == (EXPECTED_NODES, EXPECTED_FEATURES), "Unexpected features")
    require(int(graph.y.unique().numel()) == EXPECTED_CLASSES, "Unexpected class count")
    require(int(graph.train_mask.sum()) == EXPECTED_TRAINING_NODES, "Unexpected training mask")
    return graph


def train_on_device(
    graph: Data,
    epochs: int,
    device: torch.device,
    seed: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[Tensor, float, float, float, str]:
    if device.type == "mps":
        require(
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1",
            "CPU fallback is enabled; this would not prove MPS execution",
        )

    torch.manual_seed(seed)
    device_graph = graph.to(device)
    model = SmallestGCN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    model.train()
    with torch.no_grad():
        initial_logits = model(device_graph)
        initial_loss = functional.cross_entropy(
            initial_logits[device_graph.train_mask],
            device_graph.y[device_graph.train_mask],
        ).item()

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(device_graph)
        loss = functional.cross_entropy(
            logits[device_graph.train_mask],
            device_graph.y[device_graph.train_mask],
        )
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_logits = model(device_graph)
        final_loss = functional.cross_entropy(
            final_logits[device_graph.train_mask],
            device_graph.y[device_graph.train_mask],
        ).item()
        predictions = final_logits.argmax(dim=1)
        accuracy = float((predictions == device_graph.y).float().mean().item())

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
    require(math.isfinite(initial_loss) and math.isfinite(final_loss), "Loss is not finite")
    require(final_loss < initial_loss, "Training loss did not decrease")
    return final_logits.cpu(), initial_loss, final_loss, accuracy, str(actual_device)
