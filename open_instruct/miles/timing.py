"""Awaited driver stage timings; async generation wait excludes producer overlap."""

import json
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def stage(args, name, rollout_id=None):
    started = time.perf_counter()
    wall = time.time()
    passed = False
    try:
        yield
        passed = True
    finally:
        if args.save:
            root = Path(args.save)
            root.mkdir(parents=True, exist_ok=True)
            with (root / "driver_timing.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            stage=name,
                            rollout_id=rollout_id,
                            started_unix=wall,
                            seconds=time.perf_counter() - started,
                            passed=passed,
                        )
                    )
                    + "\n"
                )
