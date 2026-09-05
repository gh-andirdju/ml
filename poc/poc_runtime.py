"""Shared runtime utilities for the local GNN and Neo4j proofs."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from neo4j import Driver, GraphDatabase


class ProofError(RuntimeError):
    """Raised when an end-to-end proof condition is not satisfied."""


@dataclass(frozen=True)
class Neo4jConfiguration:
    uri: str
    user: str
    password: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProofError(message)


def load_environment_file(path: Path) -> None:
    """Load a small env file while preserving explicit process overrides."""
    if not path.is_file():
        raise ProofError(f"Secret environment file not found: {path}")
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        require("=" in line, f"Invalid environment entry at {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        require(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is not None,
            f"Invalid environment key at {path}:{line_number}",
        )
        os.environ.setdefault(key, value)


def neo4j_configuration(env_file: Path) -> Neo4jConfiguration:
    load_environment_file(env_file)
    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        auth_user, separator, auth_password = os.environ.get("NEO4J_AUTH", "").partition(
            "/"
        )
        require(separator == "/" and bool(auth_password), "NEO4J_AUTH is invalid")
        require(auth_user == user, "NEO4J_AUTH user does not match NEO4J_USER")
        password = auth_password
    require(bool(password), "NEO4J_PASSWORD is missing")
    return Neo4jConfiguration(uri=uri, user=user, password=password)


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


def connect_with_retry(
    uri: str, user: str, password: str, timeout: float
) -> Driver:
    """Connect without allowing one driver attempt to exceed the overall timeout."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while (remaining := deadline - time.monotonic()) > 0:
        driver = None
        try:
            driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                telemetry_disabled=True,
                connection_timeout=remaining,
            )
            driver.verify_connectivity()
            return driver
        except Exception as error:  # Connection errors vary across driver versions.
            last_error = error
            if driver is not None:
                driver.close()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(1, remaining))
    raise ProofError(f"Neo4j did not become ready within {timeout:g}s: {last_error}")


def verify_neo4j_community(session, expected_version: str) -> tuple[str, str]:
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
        server_version == expected_version,
        f"Expected Neo4j {expected_version}, found {server_version}",
    )
    return server_version, edition
