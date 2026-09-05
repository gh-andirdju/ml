#!/usr/bin/env python3
"""Run the Karate Club POC with the verified macOS MPS profile."""

from __future__ import annotations

from profile_entry import run_profile


PROFILE_ARGUMENTS = [
    "--device",
    "mps",
    "--epochs",
    "100",
    "--seed",
    "42",
    "--learning-rate",
    "0.1",
    "--weight-decay",
    "0.0005",
]


if __name__ == "__main__":
    raise SystemExit(run_profile(PROFILE_ARGUMENTS))
