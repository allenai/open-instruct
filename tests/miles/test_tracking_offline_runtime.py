"""Exercise driver tracking startup with real CPU Ray and offline W&B processes."""

import asyncio
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import ray
from miles.utils.tracking_utils import tracking

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.execution import driver


@ray.remote
def log_from_worker(args):
    assert not os.environ.get("WANDB_SERVICE"), "Worker inherited the driver's local W&B service"
    assert args.wandb_run_id, "Worker did not receive the primary run ID"
    try:
        tracking.init_tracking(args, primary=False)
        tracking.log(args, {"rollout/step": 0, "rollout/probe": 1}, step_key="rollout/step")
    finally:
        tracking.finish_tracking()
    return args.wandb_run_id


class ProbeComplete(Exception):
    """Stop before creating any model or GPU worker."""


def probe(output):
    args = SimpleNamespace(
        olmo_core=CoreConfig(compiler_cache=False),
        fully_async=False,
        use_wandb=True,
        wandb_mode="offline",
        wandb_key=None,
        wandb_host=None,
        wandb_random_suffix=False,
        wandb_group="offline-probe",
        rank=0,
        wandb_team=None,
        wandb_project="offline-probe",
        wandb_run_id=None,
        wandb_dir=str(output),
        env_report=None,
        sglang_enable_metrics=False,
    )

    def launch(args, **kwargs):
        assert ray.get(log_from_worker.remote(args)) == args.wandb_run_id
        raise ProbeComplete

    try:
        with (
            patch.object(driver.throughput, "report", return_value={"warnings": []}),
            patch.object(driver.startup_cache, "prepare"),
            patch.object(driver.startup_cache, "finish", new_callable=AsyncMock),
            patch.object(driver.wiring, "launch_worker_manager", side_effect=launch),
            patch.object(
                driver.ray,
                "init",
                partial(ray.init, num_cpus=2, num_gpus=0, include_dashboard=False, object_store_memory=100 * 1024**2),
            ),
        ):
            try:
                asyncio.run(driver.train(args))
            except ProbeComplete:
                pass
            else:
                raise AssertionError("Tracking probe did not run")
    finally:
        ray.shutdown()
    assert len(list(output.rglob("run-*.wandb"))) == 2


def test_driver_offline_tracking_in_real_worker(tmp_path):
    # Isolate Ray/W&B process state from the mocked lifecycle tests in this suite.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        env={k: v for k, v in os.environ.items() if k != "WANDB_SERVICE"} | {"CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    probe(Path(sys.argv[1]))
