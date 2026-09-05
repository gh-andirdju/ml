#!/usr/bin/env python3
"""Prove a pinned WikiCS -> Neo4j -> two-layer GCN -> Neo4j round trip."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import neo4j
import torch
import torch.nn.functional as functional
import torch_geometric
from neo4j.exceptions import Neo4jError
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import WikiCS
from torch_geometric.nn import GCNConv
from torch_geometric.utils import coalesce

from poc_runtime import (
    ProofError,
    connect_with_retry,
    neo4j_configuration,
    require,
    resolve_device,
    verify_neo4j_community,
)


POC_ID = "wikics-gcn-neo4j-v1"
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


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


def node_batches(
    graph: Data, batch_size: int
) -> Iterator[list[dict[str, object]]]:
    for start in range(0, graph.num_nodes, batch_size):
        stop = min(start + batch_size, graph.num_nodes)
        yield [
            {
                "page_id": page_id,
                "features": [float(value) for value in graph.x[page_id].tolist()],
                "category": int(graph.y[page_id]),
                "is_training": bool(graph.train_mask[page_id]),
                "is_validation": bool(graph.val_mask[page_id]),
                "is_stopping": bool(graph.stopping_mask[page_id]),
                "is_test": bool(graph.test_mask[page_id]),
            }
            for page_id in range(start, stop)
        ]


def canonical_relationships(graph: Data) -> list[tuple[int, int]]:
    relationships = sorted(
        {
            (min(int(source), int(target)), max(int(source), int(target)))
            for source, target in graph.edge_index.t().tolist()
        }
    )
    require(
        len(relationships) == EXPECTED_RELATIONSHIPS,
        "Unexpected canonical relationship count",
    )
    return relationships


def relationship_batches(
    relationships: Sequence[tuple[int, int]], batch_size: int
) -> Iterator[list[dict[str, int]]]:
    for start in range(0, len(relationships), batch_size):
        yield [
            {"source": source, "target": target}
            for source, target in relationships[start : start + batch_size]
        ]


def prepare_graph(session, delete_batch_size: int) -> None:
    session.run(
        """
        CREATE CONSTRAINT wikics_page_identity IF NOT EXISTS
        FOR (page:WikiPage) REQUIRE (page.poc_id, page.page_id) IS UNIQUE
        """
    ).consume()
    while True:
        record = session.run(
            """
            MATCH (page:WikiPage {poc_id: $poc_id})
            WITH page LIMIT $batch_size
            DETACH DELETE page
            RETURN count(*) AS deleted
            """,
            poc_id=POC_ID,
            batch_size=delete_batch_size,
        ).single(strict=True)
        if int(record["deleted"]) == 0:
            break


def create_nodes(transaction, nodes: list[dict[str, object]]) -> None:
    transaction.run(
        """
        UNWIND $nodes AS row
        MERGE (page:WikiPage {poc_id: $poc_id, page_id: row.page_id})
        SET page.dataset_commit = $dataset_commit,
            page.features = row.features,
            page.category = row.category,
            page.is_training = row.is_training,
            page.is_validation = row.is_validation,
            page.is_stopping = row.is_stopping,
            page.is_test = row.is_test
        """,
        nodes=nodes,
        poc_id=POC_ID,
        dataset_commit=DATASET_COMMIT,
    ).consume()


def create_relationships(transaction, relationships: list[dict[str, int]]) -> None:
    transaction.run(
        """
        UNWIND $relationships AS row
        MATCH (source:WikiPage {poc_id: $poc_id, page_id: row.source})
        MATCH (target:WikiPage {poc_id: $poc_id, page_id: row.target})
        MERGE (source)-[:CONNECTED_TO {poc_id: $poc_id}]->(target)
        """,
        relationships=relationships,
        poc_id=POC_ID,
    ).consume()


def write_graph(
    session, graph: Data, node_batch_size: int, relationship_batch_size: int
) -> None:
    prepare_graph(session, node_batch_size)
    for nodes in node_batches(graph, node_batch_size):
        session.execute_write(create_nodes, nodes)
    relationships = canonical_relationships(graph)
    for batch in relationship_batches(relationships, relationship_batch_size):
        session.execute_write(create_relationships, batch)


def graph_counts(session) -> tuple[int, int]:
    nodes = session.run(
        "MATCH (page:WikiPage {poc_id: $poc_id}) RETURN count(page) AS count",
        poc_id=POC_ID,
    ).single(strict=True)
    relationships = session.run(
        """
        MATCH (:WikiPage {poc_id: $poc_id})
              -[relationship:CONNECTED_TO {poc_id: $poc_id}]->
              (:WikiPage {poc_id: $poc_id})
        RETURN count(relationship) AS count
        """,
        poc_id=POC_ID,
    ).single(strict=True)
    return int(nodes["count"]), int(relationships["count"])


def read_graph(session) -> Data:
    node_records = list(
        session.run(
            """
            MATCH (page:WikiPage {poc_id: $poc_id})
            RETURN page.page_id AS page_id,
                   page.features AS features,
                   page.category AS category,
                   page.is_training AS is_training,
                   page.is_validation AS is_validation,
                   page.is_stopping AS is_stopping,
                   page.is_test AS is_test,
                   page.dataset_commit AS dataset_commit
            ORDER BY page_id
            """,
            poc_id=POC_ID,
        )
    )
    require(len(node_records) == EXPECTED_NODES, "Neo4j WikiCS node count mismatch")
    require(
        [record["page_id"] for record in node_records] == list(range(EXPECTED_NODES)),
        "Neo4j WikiCS page IDs are incomplete",
    )
    require(
        all(record["dataset_commit"] == DATASET_COMMIT for record in node_records),
        "Neo4j WikiCS dataset provenance mismatch",
    )

    directed_edges: list[tuple[int, int]] = []
    relationship_count = 0
    result = session.run(
        """
        MATCH (source:WikiPage {poc_id: $poc_id})
              -[:CONNECTED_TO {poc_id: $poc_id}]->
              (target:WikiPage {poc_id: $poc_id})
        RETURN source.page_id AS source, target.page_id AS target
        ORDER BY source, target
        """,
        poc_id=POC_ID,
    )
    for record in result:
        source = int(record["source"])
        target = int(record["target"])
        require(source <= target, "Neo4j WikiCS relationship is not canonical")
        directed_edges.append((source, target))
        if source != target:
            directed_edges.append((target, source))
        relationship_count += 1
    require(
        relationship_count == EXPECTED_RELATIONSHIPS,
        "Neo4j WikiCS relationship count mismatch",
    )
    directed_edges.sort()

    return Data(
        x=torch.tensor([record["features"] for record in node_records], dtype=torch.float32),
        edge_index=torch.tensor(directed_edges, dtype=torch.long).t().contiguous(),
        y=torch.tensor([record["category"] for record in node_records], dtype=torch.long),
        train_mask=torch.tensor(
            [record["is_training"] for record in node_records], dtype=torch.bool
        ),
        val_mask=torch.tensor(
            [record["is_validation"] for record in node_records], dtype=torch.bool
        ),
        stopping_mask=torch.tensor(
            [record["is_stopping"] for record in node_records], dtype=torch.bool
        ),
        test_mask=torch.tensor(
            [record["is_test"] for record in node_records], dtype=torch.bool
        ),
    )


def verify_round_trip(source: Data, rebuilt: Data) -> None:
    require(tuple(rebuilt.x.shape) == (EXPECTED_NODES, EXPECTED_FEATURES), "Feature mismatch")
    require(
        tuple(rebuilt.edge_index.shape) == (2, EXPECTED_DIRECTED_EDGES),
        "Edge shape mismatch",
    )
    require(torch.equal(source.x, rebuilt.x), "Features changed during Neo4j round trip")
    require(torch.equal(source.y, rebuilt.y), "Labels changed during Neo4j round trip")
    for name in ("train_mask", "val_mask", "stopping_mask", "test_mask"):
        require(torch.equal(source[name], rebuilt[name]), f"{name} changed during round trip")
    require(
        torch.equal(coalesce(source.edge_index), coalesce(rebuilt.edge_index)),
        "Edges changed during Neo4j round trip",
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

    if device.type != "cpu":
        torch.accelerator.synchronize(device)
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


def prediction_batches(logits: Tensor, batch_size: int) -> Iterator[list[dict[str, object]]]:
    probabilities = logits.softmax(dim=1)
    for start in range(0, EXPECTED_NODES, batch_size):
        stop = min(start + batch_size, EXPECTED_NODES)
        yield [
            {
                "page_id": page_id,
                "predicted_category": int(probabilities[page_id].argmax()),
                "scores": [float(value) for value in probabilities[page_id].tolist()],
            }
            for page_id in range(start, stop)
        ]


def write_prediction_batch(transaction, updates: list[dict[str, object]]) -> None:
    transaction.run(
        """
        UNWIND $updates AS row
        MATCH (page:WikiPage {poc_id: $poc_id, page_id: row.page_id})
        SET page.predicted_category = row.predicted_category,
            page.scores = row.scores
        """,
        updates=updates,
        poc_id=POC_ID,
    ).consume()


def write_predictions(session, logits: Tensor, batch_size: int) -> int:
    for updates in prediction_batches(logits, batch_size):
        session.execute_write(write_prediction_batch, updates)
    return count_predictions(session)


def count_predictions(session) -> int:
    record = session.run(
        """
        MATCH (page:WikiPage {poc_id: $poc_id})
        WHERE page.predicted_category IS NOT NULL AND size(page.scores) = $classes
        RETURN count(page) AS count
        """,
        poc_id=POC_ID,
        classes=EXPECTED_CLASSES,
    ).single(strict=True)
    return int(record["count"])


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".secrets/neo4j-poc.env"))
    parser.add_argument("--data-root", type=Path, default=Path(".data/wikics"))
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--expected-server-version", default="2026.07.1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--connection-timeout", type=float, default=60)
    parser.add_argument("--node-batch-size", type=int, default=250)
    parser.add_argument("--relationship-batch-size", type=int, default=5_000)
    parser.add_argument("--prediction-batch-size", type=int, default=500)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args(argv)


def validate_arguments(arguments: argparse.Namespace) -> None:
    require(0 <= arguments.split < EXPECTED_SPLITS, "Split must be from 0 to 19")
    require(arguments.connection_timeout > 0, "Connection timeout must be positive")
    require(bool(arguments.database.strip()), "Database name cannot be empty")
    require(bool(arguments.expected_server_version.strip()), "Expected version cannot be empty")
    if not arguments.verify_existing:
        require(arguments.epochs > 0, "Epoch count must be positive")
        require(arguments.patience > 0, "Patience must be positive")
        require(arguments.hidden_channels > 0, "Hidden channels must be positive")
        require(0 <= arguments.dropout < 1, "Dropout must be in [0, 1)")
        require(arguments.learning_rate > 0, "Learning rate must be positive")
        require(arguments.weight_decay >= 0, "Weight decay cannot be negative")
        require(arguments.node_batch_size > 0, "Node batch size must be positive")
        require(
            arguments.relationship_batch_size > 0,
            "Relationship batch size must be positive",
        )
        require(arguments.prediction_batch_size > 0, "Prediction batch size must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    validate_arguments(arguments)
    connection = neo4j_configuration(arguments.env_file)

    print("WikiCS: loading and verifying the pinned dataset", flush=True)
    source = source_graph(arguments.data_root, arguments.split)
    device = None if arguments.verify_existing else resolve_device(arguments.device)
    driver = connect_with_retry(
        connection.uri,
        connection.user,
        connection.password,
        arguments.connection_timeout,
    )
    try:
        with driver.session(database=arguments.database) as session:
            server_version, edition = verify_neo4j_community(
                session, arguments.expected_server_version
            )
            if arguments.verify_existing:
                print("WikiCS: reading and verifying the existing Neo4j graph", flush=True)
                rebuilt = read_graph(session)
                verify_round_trip(source, rebuilt)
                predictions_written = count_predictions(session)
                require(predictions_written == EXPECTED_NODES, "Predictions are incomplete")
                evidence = {
                    "status": "PASS",
                    "mode": "verify-existing",
                    "poc_id": POC_ID,
                    "dataset_commit": DATASET_COMMIT,
                    "dataset_sha256": DATASET_SHA256,
                    "neo4j_server": server_version,
                    "neo4j_edition": edition,
                    "nodes": EXPECTED_NODES,
                    "neo4j_relationships": EXPECTED_RELATIONSHIPS,
                    "pyg_directed_edges": EXPECTED_DIRECTED_EDGES,
                    "predictions_present": predictions_written,
                }
                print(json.dumps(evidence, indent=2, sort_keys=True))
                return 0

            print("WikiCS: replacing its isolated Neo4j subgraph in batches", flush=True)
            write_graph(
                session,
                source,
                arguments.node_batch_size,
                arguments.relationship_batch_size,
            )
            require(
                graph_counts(session) == (EXPECTED_NODES, EXPECTED_RELATIONSHIPS),
                "Neo4j WikiCS structural counts do not match",
            )
            print("WikiCS: rebuilding PyG tensors from Neo4j", flush=True)
            rebuilt = read_graph(session)
            verify_round_trip(source, rebuilt)
            require(device is not None, "Training device was not resolved")
            print(f"WikiCS: training the two-layer GCN on {device}", flush=True)
            logits, metrics = train_on_device(
                rebuilt,
                arguments.epochs,
                arguments.patience,
                arguments.hidden_channels,
                arguments.dropout,
                device,
                arguments.seed,
                arguments.learning_rate,
                arguments.weight_decay,
            )
            predictions_written = write_predictions(
                session, logits, arguments.prediction_batch_size
            )
            require(predictions_written == EXPECTED_NODES, "Prediction write-back is incomplete")
    finally:
        driver.close()

    evidence = {
        "status": "PASS",
        "poc_id": POC_ID,
        "dataset": "WikiCS",
        "dataset_commit": DATASET_COMMIT,
        "dataset_sha256": DATASET_SHA256,
        "device_requested": arguments.device,
        **metrics,
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "neo4j_driver": neo4j.__version__,
        "neo4j_server": server_version,
        "neo4j_edition": edition,
        "nodes": EXPECTED_NODES,
        "neo4j_relationships": EXPECTED_RELATIONSHIPS,
        "pyg_directed_edges": EXPECTED_DIRECTED_EDGES,
        "features": [EXPECTED_NODES, EXPECTED_FEATURES],
        "output": [EXPECTED_NODES, EXPECTED_CLASSES],
        "split": arguments.split,
        "training_nodes": EXPECTED_TRAINING_NODES,
        "validation_nodes": EXPECTED_VALIDATION_NODES,
        "stopping_nodes": EXPECTED_STOPPING_NODES,
        "test_nodes": EXPECTED_TEST_NODES,
        "epochs_requested": arguments.epochs,
        "patience": arguments.patience,
        "hidden_channels": arguments.hidden_channels,
        "dropout": arguments.dropout,
        "seed": arguments.seed,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "predictions_written": predictions_written,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (ProofError, Neo4jError, OSError, RuntimeError, ValueError) as error:
        print(f"WIKICS POC FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
