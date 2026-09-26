"""Retire one real SGLang server after a frozen publication; expect bounded failure.

This patches only the test driver's publisher class, never production dispatch or
weight arithmetic. Run in its own allocation/output directory after a valid save.
"""

import asyncio
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from unittest import mock

from miles.utils.workers.ray_worker_manager import RayWorkerManager

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.execution import driver, workflow
from open_instruct.miles.publication.rolling_publication import RollingPublication


def validate_failure(events, injected, failed, elapsed):
    if not injected or not failed or elapsed > 600:
        raise AssertionError("Expected an injected engine loss and terminal driver error within 600 seconds")
    after = [event for event in events if event["time"] >= injected["time"]]
    quarantined = [
        event for event in after if event["event"] == "engine_unavailable" and event["engine"] == injected["engine"]
    ]
    if not quarantined:
        raise AssertionError("Retired engine was not quarantined by the publication controller")
    if any(
        event["event"] == "engine_reopened"
        and event["engine"] == injected["engine"]
        and event["time"] >= quarantined[0]["time"]
        for event in after
    ):
        raise AssertionError("Retired engine reopened admission")


def main():
    spec = RunSpec.load(sys.argv[1])
    root = Path(spec.output["root"])
    output = Path("/output")
    output.mkdir(parents=True, exist_ok=True)
    injected = {}

    class RetireEngine(RollingPublication):
        async def _initialize(self, info):
            await super()._initialize(info)
            target_url = info.rollout_engines[-1].server_url
            self._failure_cell = next(
                cell_id
                for server in self.inference.servers.values()
                for cell_id, cell in server.server_cells.items()
                if cell.api_client.server_url == target_url
            )

        async def publish(self):
            result = await super().publish()
            if not injected:
                target = str(len(self.deliveries) - 1)
                injected.update(engine=target, version=self.version, time=time.time())
                (output / "injection.json").write_text(json.dumps(injected, indent=2) + "\n")
                # Stop the real server through its owned lifecycle handle. Do not
                # kill unrelated PIDs or change the healthy peer's admission.
                await asyncio.wait_for(RayWorkerManager.get_handle().stop_cells.remote([self._failure_cell]), 60)
            return result

    failed = False
    try:
        with mock.patch.object(driver, "RollingPublication", RetireEngine):
            workflow.execute(spec)
    except Exception:
        failed = True
        (output / "expected-driver-error.txt").write_text(traceback.format_exc())
    elapsed = time.time() - injected["time"] if injected else 0
    path = root / "checkpoints/engine_drain.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    validate_failure(events, injected, failed, elapsed)
    for path in (root / "checkpoints").glob("*.jsonl"):
        shutil.copyfile(path, output / path.name)
    report = {"passed": True, "injection": injected, "terminal_seconds": elapsed, "run_root": str(root)}
    (output / "failure-probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
