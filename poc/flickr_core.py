"""Device-neutral Flickr dataset, GraphSAGE model, and timed training."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import Flickr
from torch_geometric.nn import SAGEConv

from proof_common import require


EXPECTED_NODES = 89_250
EXPECTED_EDGES = 899_756
EXPECTED_FEATURES = 500
EXPECTED_CLASSES = 7
EXPECTED_TRAINING_NODES = 44_625
EXPECTED_VALIDATION_NODES = 22_312
EXPECTED_TEST_NODES = 22_313
MINIMUM_GENERALIZATION_ACCURACY = 0.30


class _ChunkedMeanAggregation(torch.autograd.Function):
    """Differentiate exact neighborhood means without retaining edge messages."""

    @staticmethod
    def forward(
        context,
        features: Tensor,
        edge_index: Tensor,
        edge_chunk_size: int,
    ) -> Tensor:
        source, destination = edge_index
        aggregated = features.new_zeros((features.size(0), features.size(1)))
        for start in range(0, edge_index.size(1), edge_chunk_size):
            stop = min(start + edge_chunk_size, edge_index.size(1))
            aggregated.index_add_(
                0,
                destination[start:stop],
                features.index_select(0, source[start:stop]),
            )
        degree = features.new_zeros(features.size(0))
        degree.index_add_(
            0,
            destination,
            torch.ones(
                destination.numel(),
                dtype=features.dtype,
                device=features.device,
            ),
        )
        degree.clamp_min_(1)
        context.save_for_backward(edge_index, degree)
        context.edge_chunk_size = edge_chunk_size
        context.feature_count = features.size(1)
        return aggregated / degree.unsqueeze(1)

    @staticmethod
    def backward(context, gradient: Tensor) -> tuple[Tensor, None, None]:
        edge_index, degree = context.saved_tensors
        source, destination = edge_index
        feature_gradient = gradient.new_zeros(
            (degree.numel(), context.feature_count)
        )
        for start in range(0, edge_index.size(1), context.edge_chunk_size):
            stop = min(start + context.edge_chunk_size, edge_index.size(1))
            destinations = destination[start:stop]
            contribution = gradient.index_select(0, destinations)
            contribution = contribution / degree.index_select(
                0, destinations
            ).unsqueeze(1)
            feature_gradient.index_add_(0, source[start:stop], contribution)
        return feature_gradient, None, None


class ChunkedSAGEConv(SAGEConv):
    """Exact GraphSAGE mean aggregation with a bounded edge workspace."""

    def __init__(
        self, in_channels: int, out_channels: int, *, edge_chunk_size: int
    ) -> None:
        require(edge_chunk_size > 0, "Edge chunk size must be positive")
        super().__init__(in_channels, out_channels)
        self.edge_chunk_size = edge_chunk_size

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        size: tuple[int, int] | None = None,
    ) -> Tensor:
        require(size is None, "Chunked GraphSAGE supports homogeneous graphs only")
        require(
            isinstance(edge_index, Tensor) and edge_index.layout == torch.strided,
            "Chunked GraphSAGE requires a dense COO edge-index tensor",
        )
        require(
            edge_index.ndim == 2 and edge_index.size(0) == 2,
            "Edge index must have shape [2, edges]",
        )
        aggregated = _ChunkedMeanAggregation.apply(
            x,
            edge_index,
            self.edge_chunk_size,
        )
        return self.lin_l(aggregated) + self.lin_r(x)


class FlickrGraphSAGE(torch.nn.Module):
    """A three-layer, full-batch GraphSAGE node classifier."""

    def __init__(
        self,
        hidden_channels: int,
        dropout: float,
        edge_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        convolution = (
            SAGEConv
            if edge_chunk_size is None
            else lambda inputs, outputs: ChunkedSAGEConv(
                inputs, outputs, edge_chunk_size=edge_chunk_size
            )
        )
        self.convolutions = torch.nn.ModuleList(
            [
                convolution(EXPECTED_FEATURES, hidden_channels),
                convolution(hidden_channels, hidden_channels),
                convolution(hidden_channels, EXPECTED_CLASSES),
            ]
        )

    def forward(self, graph: Data) -> Tensor:
        features = graph.x
        for convolution in self.convolutions[:-1]:
            features = convolution(features, graph.edge_index).relu()
            features = functional.dropout(
                features, p=self.dropout, training=self.training
            )
        return self.convolutions[-1](features, graph.edge_index)


def source_graph(data_root: Path) -> Data:
    graph = Flickr(str(data_root))[0]
    require(graph.num_nodes == EXPECTED_NODES, "Unexpected Flickr node count")
    require(graph.num_edges == EXPECTED_EDGES, "Unexpected Flickr edge count")
    require(graph.num_node_features == EXPECTED_FEATURES, "Unexpected feature count")
    require(int(graph.y.unique().numel()) == EXPECTED_CLASSES, "Unexpected class count")
    require(int(graph.train_mask.sum()) == EXPECTED_TRAINING_NODES, "Bad train split")
    require(int(graph.val_mask.sum()) == EXPECTED_VALIDATION_NODES, "Bad validation split")
    require(int(graph.test_mask.sum()) == EXPECTED_TEST_NODES, "Bad test split")
    require(graph.is_undirected(), "Flickr graph is not undirected")
    return graph


def masked_accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> float:
    return float((logits[mask].argmax(dim=1) == labels[mask]).float().mean().item())


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


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
    edge_chunk_size: int | None = None,
) -> tuple[Tensor, dict[str, float | int | str]]:
    require(
        device.type in {"cpu", "cuda", "mps"},
        "Flickr benchmark supports CPU, CUDA, or MPS",
    )
    if device.type == "mps":
        require(
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1",
            "CPU fallback is enabled; this would not prove MPS execution",
        )
    torch.manual_seed(seed)
    device_graph = graph.to(device)
    model = FlickrGraphSAGE(hidden_channels, dropout, edge_chunk_size).to(device)
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

    best_validation_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    synchronize(device)
    started_at = time.perf_counter()

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
            validation_logits = model(device_graph)
            validation_loss = functional.cross_entropy(
                validation_logits[device_graph.val_mask],
                device_graph.y[device_graph.val_mask],
            ).item()
        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: parameter.detach().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    synchronize(device)
    training_seconds = time.perf_counter() - started_at
    require(best_state is not None and best_epoch > 0, "Early stopping found no model")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(device_graph)
        final_training_loss = functional.cross_entropy(
            final_logits[device_graph.train_mask],
            device_graph.y[device_graph.train_mask],
        ).item()
    synchronize(device)

    actual_device = final_logits.device
    require(next(model.parameters()).device == actual_device, "Model device mismatch")
    require(device_graph.x.device == actual_device, "Feature device mismatch")
    require(actual_device.type == device.type, "Requested device type was not used")
    require(
        tuple(final_logits.shape) == (EXPECTED_NODES, EXPECTED_CLASSES),
        "Output shape mismatch",
    )
    require(
        math.isfinite(initial_training_loss)
        and math.isfinite(final_training_loss)
        and math.isfinite(best_validation_loss)
        and math.isfinite(training_seconds),
        "Training metric is not finite",
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
        "best_validation_loss": round(best_validation_loss, 6),
        "validation_accuracy": round(validation_accuracy, 6),
        "test_accuracy": round(test_accuracy, 6),
        "training_seconds": round(training_seconds, 6),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    return final_logits.cpu(), metrics
