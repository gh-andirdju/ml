#!/usr/bin/env python3
"""Prove a Karate Club -> Neo4j -> PyG -> selected device -> Neo4j round trip."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import neo4j
import torch
import torch_geometric
from neo4j.exceptions import Neo4jError
from torch import Tensor
from torch_geometric.data import Data

from karate_core import (
    EXPECTED_CLASSES,
    EXPECTED_DIRECTED_EDGES,
    EXPECTED_FEATURES,
    EXPECTED_NODES,
    EXPECTED_RELATIONSHIPS,
    EXPECTED_TRAINING_NODES,
    SmallestGCN,
    source_graph,
    train_on_device,
)

from poc_runtime import (
    ProofError,
    connect_with_retry,
    neo4j_configuration,
    require,
    resolve_device,
    verify_neo4j_community,
)


POC_ID = "karate-gnn-neo4j-v1"
LEGACY_POC_IDS = ["karate-mps-neo4j-v1"]


def graph_payload(graph: Data) -> tuple[list[dict[str, object]], list[dict[str, int]]]:
    nodes = [
        {
            "member_id": member_id,
            "features": [float(value) for value in graph.x[member_id].tolist()],
            "community": int(graph.y[member_id]),
            "is_training": bool(graph.train_mask[member_id]),
        }
        for member_id in range(graph.num_nodes)
    ]
    undirected = {
        (min(int(source), int(target)), max(int(source), int(target)))
        for source, target in graph.edge_index.t().tolist()
        if source != target
    }
    edges = [{"source": source, "target": target} for source, target in sorted(undirected)]
    require(len(edges) == EXPECTED_RELATIONSHIPS, "Unexpected undirected relationship count")
    return nodes, edges


def replace_graph(transaction, nodes, edges) -> None:
    transaction.run(
        "MATCH (node:KarateMember) WHERE node.poc_id IN $poc_ids DETACH DELETE node",
        poc_ids=[POC_ID, *LEGACY_POC_IDS],
    ).consume()
    transaction.run(
        """
        UNWIND $nodes AS row
        CREATE (:KarateMember {
            poc_id: $poc_id,
            member_id: row.member_id,
            features: row.features,
            community: row.community,
            is_training: row.is_training
        })
        """,
        nodes=nodes,
        poc_id=POC_ID,
    ).consume()
    transaction.run(
        """
        UNWIND $edges AS row
        MATCH (source:KarateMember {poc_id: $poc_id, member_id: row.source})
        MATCH (target:KarateMember {poc_id: $poc_id, member_id: row.target})
        CREATE (source)-[:KNOWS {poc_id: $poc_id}]->(target)
        """,
        edges=edges,
        poc_id=POC_ID,
    ).consume()


def read_graph(session) -> Data:
    node_records = list(
        session.run(
            """
            MATCH (node:KarateMember {poc_id: $poc_id})
            RETURN node.member_id AS member_id,
                   node.features AS features,
                   node.community AS community,
                   node.is_training AS is_training
            ORDER BY member_id
            """,
            poc_id=POC_ID,
        )
    )
    relationship_records = list(
        session.run(
            """
            MATCH (source:KarateMember {poc_id: $poc_id})
                  -[:KNOWS {poc_id: $poc_id}]->
                  (target:KarateMember {poc_id: $poc_id})
            RETURN source.member_id AS source, target.member_id AS target
            ORDER BY source, target
            """,
            poc_id=POC_ID,
        )
    )

    require(len(node_records) == EXPECTED_NODES, "Neo4j node count mismatch")
    require(
        len(relationship_records) == EXPECTED_RELATIONSHIPS,
        "Neo4j relationship count mismatch",
    )
    require(
        [record["member_id"] for record in node_records] == list(range(EXPECTED_NODES)),
        "Neo4j member IDs are incomplete",
    )

    directed_edges: list[tuple[int, int]] = []
    for record in relationship_records:
        source = int(record["source"])
        target = int(record["target"])
        directed_edges.extend(((source, target), (target, source)))
    directed_edges.sort()

    return Data(
        x=torch.tensor([record["features"] for record in node_records], dtype=torch.float32),
        edge_index=torch.tensor(directed_edges, dtype=torch.long).t().contiguous(),
        y=torch.tensor([record["community"] for record in node_records], dtype=torch.long),
        train_mask=torch.tensor(
            [record["is_training"] for record in node_records], dtype=torch.bool
        ),
    )


def verify_round_trip(source: Data, rebuilt: Data) -> None:
    require(
        tuple(rebuilt.x.shape) == (EXPECTED_NODES, EXPECTED_FEATURES),
        "Feature shape mismatch",
    )
    require(tuple(rebuilt.edge_index.shape) == (2, EXPECTED_DIRECTED_EDGES), "Edge shape mismatch")
    require(torch.equal(source.x, rebuilt.x), "Features changed during Neo4j round trip")
    require(torch.equal(source.y, rebuilt.y), "Labels changed during Neo4j round trip")
    require(torch.equal(source.train_mask, rebuilt.train_mask), "Training mask changed")
    require(
        set(map(tuple, source.edge_index.t().tolist()))
        == set(map(tuple, rebuilt.edge_index.t().tolist())),
        "Edges changed during Neo4j round trip",
    )


def write_predictions(session, logits: Tensor) -> int:
    probabilities = logits.softmax(dim=1)
    updates = [
        {
            "member_id": member_id,
            "predicted_community": int(probabilities[member_id].argmax()),
            "scores": [float(value) for value in probabilities[member_id].tolist()],
        }
        for member_id in range(EXPECTED_NODES)
    ]
    session.run(
        """
        UNWIND $updates AS row
        MATCH (node:KarateMember {poc_id: $poc_id, member_id: row.member_id})
        SET node.predicted_community = row.predicted_community,
            node.scores = row.scores
        """,
        updates=updates,
        poc_id=POC_ID,
    ).consume()
    return count_predictions(session)


def count_predictions(session) -> int:
    """Count complete prediction payloads without changing database state."""
    record = session.run(
        """
        MATCH (node:KarateMember {poc_id: $poc_id})
        WHERE node.predicted_community IS NOT NULL AND size(node.scores) = $classes
        RETURN count(node) AS count
        """,
        poc_id=POC_ID,
        classes=EXPECTED_CLASSES,
    ).single(strict=True)
    return int(record["count"])


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".secrets/neo4j-poc.env"))
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--expected-server-version", default="2026.07.1")
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--connection-timeout", type=float, default=60)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Read and verify the existing graph without replacing or training it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    require(arguments.connection_timeout > 0, "Connection timeout must be positive")
    require(bool(arguments.database.strip()), "Database name cannot be empty")
    require(
        bool(arguments.expected_server_version.strip()),
        "Expected server version cannot be empty",
    )
    if not arguments.verify_existing:
        require(arguments.epochs > 0, "Epoch count must be positive")
        require(arguments.learning_rate > 0, "Learning rate must be positive")
        require(arguments.weight_decay >= 0, "Weight decay cannot be negative")
    connection = neo4j_configuration(arguments.env_file)

    source = source_graph()
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
                rebuilt = read_graph(session)
                verify_round_trip(source, rebuilt)
                predictions_written = count_predictions(session)
                require(
                    predictions_written == EXPECTED_NODES,
                    "Persisted predictions are incomplete",
                )
                evidence = {
                    "status": "PASS",
                    "mode": "verify-existing",
                    "poc_id": POC_ID,
                    "neo4j_server": server_version,
                    "neo4j_edition": edition,
                    "nodes": EXPECTED_NODES,
                    "neo4j_relationships": EXPECTED_RELATIONSHIPS,
                    "pyg_directed_edges": EXPECTED_DIRECTED_EDGES,
                    "features": [EXPECTED_NODES, EXPECTED_FEATURES],
                    "training_nodes": EXPECTED_TRAINING_NODES,
                    "predictions_present": predictions_written,
                }
                print(json.dumps(evidence, indent=2, sort_keys=True))
                return 0

            nodes, edges = graph_payload(source)
            session.execute_write(replace_graph, nodes, edges)
            rebuilt = read_graph(session)
            verify_round_trip(source, rebuilt)
            require(device is not None, "Training device was not resolved")
            logits, initial_loss, final_loss, accuracy, actual_device = train_on_device(
                rebuilt,
                arguments.epochs,
                device,
                arguments.seed,
                arguments.learning_rate,
                arguments.weight_decay,
            )
            predictions_written = write_predictions(session, logits)
            require(
                predictions_written == EXPECTED_NODES,
                "Not all predictions were written to Neo4j",
            )
    finally:
        driver.close()

    evidence = {
        "status": "PASS",
        "poc_id": POC_ID,
        "device_requested": arguments.device,
        "device": actual_device,
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
        "training_nodes": EXPECTED_TRAINING_NODES,
        "epochs": arguments.epochs,
        "seed": arguments.seed,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "initial_loss": round(initial_loss, 6),
        "final_loss": round(final_loss, 6),
        "all_node_accuracy": round(accuracy, 6),
        "predictions_written": predictions_written,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


def cli(argv: Sequence[str] | None = None) -> int:
    """Run the shared CLI with consistent user-facing error handling."""
    try:
        return main(argv)
    except (ProofError, Neo4jError, OSError, RuntimeError, ValueError) as error:
        print(f"POC FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
