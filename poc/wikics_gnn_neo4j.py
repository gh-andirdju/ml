#!/usr/bin/env python3
"""Prove a pinned WikiCS -> Neo4j -> two-layer GCN -> Neo4j round trip."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import neo4j
import torch
import torch_geometric
from neo4j.exceptions import Neo4jError
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

from poc_runtime import (
    ProofError,
    connect_with_retry,
    neo4j_configuration,
    require,
    resolve_device,
    verify_neo4j_community,
)
from wikics_core import (
    DATASET_COMMIT,
    DATASET_SHA256,
    EXPECTED_CLASSES,
    EXPECTED_DIRECTED_EDGES,
    EXPECTED_FEATURES,
    EXPECTED_NODES,
    EXPECTED_RELATIONSHIPS,
    EXPECTED_SPLITS,
    EXPECTED_STOPPING_NODES,
    EXPECTED_TEST_NODES,
    EXPECTED_TRAINING_NODES,
    EXPECTED_VALIDATION_NODES,
    PinnedWikiCS,
    WikiGCN,
    file_sha256,
    masked_accuracy,
    source_graph,
    train_on_device,
)


POC_ID = "wikics-gcn-neo4j-v1"


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
