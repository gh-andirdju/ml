#!/usr/bin/env python3
"""Export the 256-channel Flickr comparison artifact on macOS MPS."""

from mps_artifact_runner import cli


if __name__ == "__main__":
    raise SystemExit(cli("flickr"))
