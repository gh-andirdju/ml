#!/usr/bin/env python3
"""POC 8 ready runner: Flickr GraphSAGE on Kaggle T4 CUDA."""

from kaggle_flickr_runner import cli


if __name__ == "__main__":
    raise SystemExit(cli("cuda"))
