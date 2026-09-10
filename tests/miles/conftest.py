"""Keep dedicated runtime tests out of ordinary CPU-only test collection."""

from importlib import util
from pathlib import Path

collect_ignore = []
if util.find_spec("miles") is None or util.find_spec("sglang") is None:
    collect_ignore = [path.name for path in Path(__file__).parent.glob("test_*.py")]
