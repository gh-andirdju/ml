#!/usr/bin/env python3
"""Validate a Kaggle CUDA artifact and import it into the matching local Neo4j graph."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import neo4j
from neo4j.exceptions import Neo4jError

from kaggle_specs import KARATE_KAGGLE_SPEC, SPECS_BY_POC_ID, WIKICS_KAGGLE_SPEC
from poc_runtime import (
    connect_with_retry,
    neo4j_configuration,
    verify_neo4j_community,
)
from proof_common import ProofError, require
from result_artifact import MAX_ARTIFACT_BYTES, ArtifactSpec, load_and_validate_artifact


def detect_spec(path: Path) -> ArtifactSpec:
    require(path.is_file(), f"Artifact not found: {path}")
    require(0 < path.stat().st_size <= MAX_ARTIFACT_BYTES, "Artifact size is outside the accepted range")
    try:
        preview = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise ProofError(f"Artifact is not valid JSON: {error}") from error
    require(isinstance(preview, dict), "Artifact root must be an object")
    spec = SPECS_BY_POC_ID.get(preview.get("poc_id"))
    require(spec is not None, "Artifact POC ID is not supported")
    return spec


def batches(items: list[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def verify_target_graph(session, spec: ArtifactSpec) -> None:
    if spec == KARATE_KAGGLE_SPEC:
        record = session.run(
            """
            MATCH (node:KarateMember {poc_id: $poc_id})
            RETURN count(node) AS nodes,
                   count(DISTINCT node.member_id) AS identifiers,
                   min(node.member_id) AS minimum_id,
                   max(node.member_id) AS maximum_id
            """,
            poc_id=spec.target_poc_id,
        ).single(strict=True)
    else:
        record = session.run(
            """
            MATCH (node:WikiPage {poc_id: $poc_id})
            RETURN count(node) AS nodes,
                   count(DISTINCT node.page_id) AS identifiers,
                   min(node.page_id) AS minimum_id,
                   max(node.page_id) AS maximum_id,
                   collect(DISTINCT node.dataset_commit) AS commits
            """,
            poc_id=spec.target_poc_id,
        ).single(strict=True)
        require(
            record["commits"] == [spec.identity["dataset_commit"]],
            "Local WikiCS dataset provenance does not match the artifact",
        )
    require(int(record["nodes"]) == spec.nodes, "Local target graph has the wrong node count")
    require(int(record["identifiers"]) == spec.nodes, "Local target graph has duplicate node IDs")
    require(record["minimum_id"] == 0 and record["maximum_id"] == spec.nodes - 1, "Local target graph node IDs are incomplete")


def write_karate_batch(transaction, updates: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    transaction.run(
        """
        UNWIND $updates AS row
        MATCH (node:KarateMember {poc_id: $target_poc_id, member_id: row.node_id})
        SET node.cuda_predicted_community = row.predicted_class,
            node.cuda_scores = row.scores,
            node.cuda_artifact_sha256 = $sha256,
            node.cuda_device_name = $device_name,
            node.cuda_accuracy = $accuracy,
            node.cuda_generated_at = $generated_at
        """,
        updates=updates,
        **metadata,
    ).consume()


def write_wikics_batch(transaction, updates: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    transaction.run(
        """
        UNWIND $updates AS row
        MATCH (node:WikiPage {poc_id: $target_poc_id, page_id: row.node_id})
        SET node.cuda_predicted_category = row.predicted_class,
            node.cuda_scores = row.scores,
            node.cuda_artifact_sha256 = $sha256,
            node.cuda_device_name = $device_name,
            node.cuda_accuracy = $accuracy,
            node.cuda_generated_at = $generated_at
        """,
        updates=updates,
        **metadata,
    ).consume()


def count_imported(session, spec: ArtifactSpec, digest: str) -> int:
    label = "KarateMember" if spec == KARATE_KAGGLE_SPEC else "WikiPage"
    query = f"""
        MATCH (node:{label} {{poc_id: $poc_id, cuda_artifact_sha256: $sha256}})
        WHERE size(node.cuda_scores) = $classes
        RETURN count(node) AS count
    """
    return int(session.run(
        query,
        poc_id=spec.target_poc_id,
        sha256=digest,
        classes=spec.classes,
    ).single(strict=True)["count"])


def count_local_predictions(session, spec: ArtifactSpec) -> int:
    if spec == KARATE_KAGGLE_SPEC:
        query = """
            MATCH (node:KarateMember {poc_id: $poc_id})
            WHERE node.predicted_community IS NOT NULL AND size(node.scores) = $classes
            RETURN count(node) AS count
        """
    else:
        query = """
            MATCH (node:WikiPage {poc_id: $poc_id})
            WHERE node.predicted_category IS NOT NULL AND size(node.scores) = $classes
            RETURN count(node) AS count
        """
    return int(session.run(
        query, poc_id=spec.target_poc_id, classes=spec.classes
    ).single(strict=True)["count"])


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--checksum-file", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".secrets/neo4j-poc.env"))
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--expected-server-version", default="2026.07.1")
    parser.add_argument("--connection-timeout", type=float, default=60)
    parser.add_argument("--batch-size", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    require(arguments.batch_size > 0, "Batch size must be positive")
    require(arguments.connection_timeout > 0, "Connection timeout must be positive")
    spec = detect_spec(arguments.artifact)
    artifact, digest = load_and_validate_artifact(
        arguments.artifact, spec, arguments.checksum_file
    )
    execution = artifact["execution"]
    metadata = {
        "target_poc_id": spec.target_poc_id,
        "sha256": digest,
        "device_name": execution["cuda_device_name"],
        "accuracy": execution["accuracy"],
        "generated_at": artifact["generated_at"],
    }
    connection = neo4j_configuration(arguments.env_file)
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
            verify_target_graph(session, spec)
            local_predictions = count_local_predictions(session, spec)
            require(local_predictions == spec.nodes, "Laptop predictions are incomplete")
            writer = write_karate_batch if spec == KARATE_KAGGLE_SPEC else write_wikics_batch
            for update_batch in batches(artifact["predictions"], arguments.batch_size):
                session.execute_write(writer, update_batch, metadata)
            imported = count_imported(session, spec, digest)
            require(imported == spec.nodes, "CUDA prediction import is incomplete")
            require(
                count_local_predictions(session, spec) == local_predictions,
                "Laptop predictions were not preserved",
            )
    finally:
        driver.close()
    print(json.dumps({
        "status": "PASS",
        "poc_id": spec.poc_id,
        "target_poc_id": spec.target_poc_id,
        "artifact_sha256": digest,
        "device": execution["device"],
        "cuda_device_name": execution["cuda_device_name"],
        "accuracy": execution["accuracy"],
        "predictions_imported": imported,
        "laptop_predictions_preserved": local_predictions,
        "neo4j_server": server_version,
        "neo4j_edition": edition,
    }, indent=2, sort_keys=True))
    return 0


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (ProofError, Neo4jError, OSError, RuntimeError, ValueError) as error:
        print(f"KAGGLE IMPORT FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
