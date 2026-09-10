"""Keep dedicated runtime tests out of ordinary CPU-only test collection."""

from importlib import util

collect_ignore = []
if util.find_spec("miles") is None or util.find_spec("sglang") is None:
    collect_ignore = ["test_runtime.py", "test_async.py"]
