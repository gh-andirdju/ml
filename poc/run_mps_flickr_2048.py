#!/usr/bin/env python3
"""Export the 2,048-channel Flickr comparison artifact on macOS MPS."""

from mps_artifact_runner import cli


if __name__ == "__main__":
    raise SystemExit(cli("flickr-2048"))
