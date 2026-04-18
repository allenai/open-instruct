#!/usr/bin/env python
"""Download an HF Hub repo to a local directory."""

import argparse

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--local_dir", required=True)
    args = parser.parse_args()
    snapshot_download(repo_id=args.repo_id, local_dir=args.local_dir)


if __name__ == "__main__":
    main()
