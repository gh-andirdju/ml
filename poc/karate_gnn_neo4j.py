#!/usr/bin/env python3
"""Prove a Karate Club -> Neo4j -> PyG -> selected device -> Neo4j round trip."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import neo4j
import torch
import torch.nn.functional as functional
import torch_geometric
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv


POC_ID = "karate-gnn-neo4j-v1"
LEGACY_POC_IDS = ["karate-mps-neo4j-v1"]
EXPECTED_NODES = 34
EXPECTED_DIRECTED_EDGES = 156
EXPECTED_RELATIONSHIPS = 78
EXPECTED_FEATURES = 34
EXPECTED_CLASSES = 4
EXPECTED_TRAINING_NODES = 4


class ProofError(RuntimeError):
    """Raised when an end-to-end proof condition is not satisfied."""


class SmallestGCN(torch.nn.Module):
    """A single graph-convolution layer for four-class node prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = GCNConv(EXPECTED_FEATURES, EXPECTED_CLASSES)

    def forward(self, graph: Data) -> Tensor:
        return self.convolution(graph.x, graph.edge_index)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProofError(message)


def load_environment_file(path: Path) -> None:
    if not path.is_file():
        raise ProofError(f"Secret environment file not found: {path}")
    for raw_line in path.read_text(encoding="utf8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value)


def source_graph() -> Data:
    graph = KarateClub()[0]
    require(graph.num_nodes == EXPECTED_NODES, "Unexpected source node count")
    require(graph.num_edges == EXPECTED_DIRECTED_EDGES, "Unexpected source edge count")
    require(tuple(graph.x.shape) == (EXPECTED_NODES, EXPECTED_FEATURES), "Unexpected features")
    require(int(graph.y.unique().numel()) == EXPECTED_CLASSES, "Unexpected class count")
    require(int(graph.train_mask.sum()) == EXPECTED_TRAINING_NODES, "Unexpected training mask")
    return graph


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
    require(tuple(rebuilt.x.shape) == (EXPECTED_NODES, EXPECTED_FEATURES), "Feature shape mismatch")
    require(tuple(rebuilt.edge_index.shape) == (2, EXPECTED_DIRECTED_EDGES), "Edge shape mismatch")
    require(torch.equal(source.x, rebuilt.x), "Features changed during Neo4j round trip")
    require(torch.equal(source.y, rebuilt.y), "Labels changed during Neo4j round trip")
    require(torch.equal(source.train_mask, rebuilt.train_mask), "Training mask changed")
    require(
        set(map(tuple, source.edge_index.t().tolist()))
        == set(map(tuple, rebuilt.edge_index.t().tolist())),
        "Edges changed during Neo4j round trip",
    )


def resolve_device(requested: str) -> torch.device:
    """Resolve auto, CPU, MPS, or CUDA without leaking backend logic downstream."""
    normalized = requested.strip().lower()
    if normalized == "auto":
        accelerator = torch.accelerator.current_accelerator(check_available=True)
        return accelerator if accelerator is not None else torch.device("cpu")

    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as error:
        raise ProofError(f"Invalid device {requested!r}: {error}") from error

    require(
        device.type in {"cpu", "cuda", "mps"},
        f"Unsupported device type {device.type!r}; use auto, cpu, cuda, cuda:N, or mps",
    )
    if device.type != "cpu":
        try:
            torch.empty(1, device=device)
        except Exception as error:
            raise ProofError(f"Requested device {requested!r} is unavailable: {error}") from error
    return device


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

    if device.type != "cpu":
        torch.accelerator.synchronize(device)
    actual_device = final_logits.device
    require(next(model.parameters()).device == actual_device, "Model device mismatch")
    require(device_graph.x.device == actual_device, "Feature device mismatch")
    require(actual_device.type == device.type, "Requested device type was not used")
    if device.index is not None:
        require(actual_device.index == device.index, "Requested device index was not used")
    require(tuple(final_logits.shape) == (EXPECTED_NODES, EXPECTED_CLASSES), "Output shape mismatch")
    require(math.isfinite(initial_loss) and math.isfinite(final_loss), "Loss is not finite")
    require(final_loss < initial_loss, "Training loss did not decrease")
    return final_logits.cpu(), initial_loss, final_loss, accuracy, str(actual_device)


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


def connect_with_retry(uri: str, user: str, password: str, timeout: float):
    driver = GraphDatabase.driver(uri, auth=(user, password), telemetry_disabled=True)
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            driver.verify_connectivity()
            return driver
        except Exception as error:  # Connection errors vary across driver versions.
            last_error = error
            time.sleep(1)
    driver.close()
    raise ProofError(f"Neo4j did not become ready within {timeout:g}s: {last_error}")


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
    require(arguments.epochs > 0, "Epoch count must be positive")
    require(arguments.learning_rate > 0, "Learning rate must be positive")
    require(arguments.weight_decay >= 0, "Weight decay cannot be negative")
    load_environment_file(arguments.env_file)
    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        auth_user, separator, auth_password = os.environ.get("NEO4J_AUTH", "").partition("/")
        require(separator == "/" and bool(auth_password), "NEO4J_AUTH is invalid")
        require(auth_user == user, "NEO4J_AUTH user does not match NEO4J_USER")
        password = auth_password
    require(bool(password), "NEO4J_PASSWORD is missing")

    source = source_graph()
    device = None if arguments.verify_existing else resolve_device(arguments.device)
    driver = connect_with_retry(uri, user, password, arguments.connection_timeout)
    try:
        with driver.session(database=arguments.database) as session:
            component = session.run(
                """
                CALL dbms.components() YIELD name, versions, edition
                RETURN name, versions[0] AS version, edition
                LIMIT 1
                """
            ).single(strict=True)
            server_version = str(component["version"])
            edition = str(component["edition"]).lower()
            require(edition == "community", f"Expected Community Edition, found {edition}")
            require(
                server_version == arguments.expected_server_version,
                f"Expected Neo4j {arguments.expected_server_version}, found {server_version}",
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
