#!/usr/bin/env python3
"""POC 11 ready runner: Flickr-2048 GraphSAGE on Kaggle CPU only."""

from kaggle_flickr_runner import cli


if __name__ == "__main__":
    raise SystemExit(cli("cpu", variant="2048"))
