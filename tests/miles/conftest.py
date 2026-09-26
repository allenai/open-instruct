"""Keep dedicated runtime tests out of ordinary CPU-only test collection."""

from importlib import util
from pathlib import Path

import pytest

collect_ignore = []
if util.find_spec("miles") is None or util.find_spec("sglang") is None:
    collect_ignore = [path.name for path in Path(__file__).parent.glob("test_*.py")]


def pytest_addoption(parser):
    parser.addoption(
        "--require-miles-runtime", action="store_true", help="Fail if pinned runtime tests cannot collect"
    )


def pytest_configure(config):
    if config.getoption("--require-miles-runtime") and collect_ignore:
        raise pytest.UsageError(
            "MILES runtime dependencies are missing; use the image built with runtime/miles/Dockerfile"
        )
