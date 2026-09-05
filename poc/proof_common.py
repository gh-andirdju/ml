"""Dependency-free proof assertions shared by local and remote runners."""

from __future__ import annotations


class ProofError(RuntimeError):
    """Raised when an end-to-end proof condition is not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProofError(message)
