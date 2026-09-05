#!/usr/bin/env python3
"""POC 9 ready runner: wide Flickr GraphSAGE on Kaggle CPU only."""

from kaggle_flickr_runner import cli


if __name__ == "__main__":
    raise SystemExit(cli("cpu", variant="wide"))
