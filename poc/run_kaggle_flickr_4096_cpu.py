#!/usr/bin/env python3
"""POC 13 ready runner: Flickr-4096 GraphSAGE on Kaggle CPU only."""

from kaggle_flickr_runner import cli


if __name__ == "__main__":
    raise SystemExit(cli("cpu", variant="4096"))
