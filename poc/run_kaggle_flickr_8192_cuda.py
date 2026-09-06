#!/usr/bin/env python3
"""POC 16 ready runner: Flickr-8192 GraphSAGE on Kaggle T4 CUDA."""

from kaggle_flickr_runner import cli


if __name__ == "__main__":
    raise SystemExit(cli("cuda", variant="8192"))
