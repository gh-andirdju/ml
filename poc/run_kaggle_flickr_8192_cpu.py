#!/usr/bin/env python3
"""POC 15 ready runner: Flickr-8192 GraphSAGE on Kaggle CPU only."""

from kaggle_flickr_runner import cli


if __name__ == "__main__":
    raise SystemExit(cli("cpu", variant="8192"))
