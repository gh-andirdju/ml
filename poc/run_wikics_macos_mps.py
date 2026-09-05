#!/usr/bin/env python3
"""Run the larger WikiCS POC with the macOS MPS profile."""

from __future__ import annotations

from profile_entry import run_profile


PROFILE_ARGUMENTS = [
    "--device",
    "mps",
    "--epochs",
    "100",
    "--patience",
    "20",
    "--hidden-channels",
    "64",
    "--split",
    "0",
]


if __name__ == "__main__":
    raise SystemExit(run_profile(PROFILE_ARGUMENTS, "wikics_gnn_neo4j"))
