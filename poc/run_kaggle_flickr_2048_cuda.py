#!/usr/bin/env python3
"""POC 12 ready runner: Flickr-2048 GraphSAGE on Kaggle T4 CUDA."""

from kaggle_flickr_runner import cli


if __name__ == "__main__":
    raise SystemExit(cli("cuda", variant="2048"))
