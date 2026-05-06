#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "datasets>=4.0.0",
#   "numpy>=2",
#   "scipy>=1.14.0",
# ]
# ///

"""Thin CLI wrapper for ``open_instruct.rlvr_difficulty``."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_instruct import rlvr_difficulty


if __name__ == "__main__":
    rlvr_difficulty.main()
