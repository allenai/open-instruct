"""Contracts at the upstream worker, publication, and token-provenance boundaries."""

import asyncio
import json
import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.types import WeightVersionSpan, WeightVersionsPerCall
from miles.utils.workers.worker_spec import CommandWorkerSpec, SchedulingSpec, WorkerLaunchContext

from open_instruct.miles import policy_versions, startup_cache
from open_instruct.miles.engine_drain import WeightSnapshot
from open_instruct.miles.rolling_publication import RollingPublication


def test_serving_child_receives_model_package_and_its_own_cache_slot():
    spec = CommandWorkerSpec(
        name="inference-engine-0-0",
        port_infos=[],
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0.2),
        env_var=lambda ctx: {"EXISTING": "kept"},
        launch_command=lambda ctx: "/usr/bin/python -m sglang.launch_server --model-path '/model with spaces'",
    )
    args = SimpleNamespace(olmo_core_startup_cache={"shared": "/shared", "restore": True})
    [configured] = startup_cache.configure_specs(args, [spec])
    context = WorkerLaunchContext(cell_index=2, worker_in_cell_index=1, gpu_ids=[3])
    env = configured.env_var(context)
    assert env["SGLANG_EXTERNAL_MODEL_PACKAGE"] == "olmo_sglang.models"
    assert env["EXISTING"] == "kept"
    assert json.loads(env[startup_cache.ENV])["slot"] == "inference-engine-0-0-2-1"
    assert shlex.split(configured.launch_command(context)) == [
        "/usr/bin/python",
        "-m",
        "open_instruct.miles.serving",
        "--model-path",
        "/model with spaces",
    ]
    assert "worker_process_setup_hook" not in env
    assert spec.env_var(context) == {"EXISTING": "kept"}


def test_native_provenance_round_trips_without_collapsing_mixed_versions():
    calls = [
        WeightVersionsPerCall([WeightVersionSpan("2", 5, 7), WeightVersionSpan("3", 7, 9)]),
        WeightVersionsPerCall([WeightVersionSpan("4", 12, 14)]),
    ]
    serialized = json.loads(json.dumps([call.to_dicts() for call in calls]))
    assert policy_versions.spans(calls) == policy_versions.spans(serialized)
    assert policy_versions.versions(serialized) == [2, 3, 4]
    serialized[1][0]["abs_start"] = 8
    with pytest.raises(ValueError, match="overlapping"):
        policy_versions.versions(serialized)


@pytest.mark.parametrize("failed", [False, True])
def test_independent_delivery_holds_controller_window_until_acknowledged(failed):
    async def exercise():
        inference = InferenceController(SimpleNamespace(colocate=False))
        inference.prepare_rollout = AsyncMock()
        entered, release = asyncio.Event(), asyncio.Event()

        async def control(operation, **kwargs):
            if operation == "barrier":
                entered.set()
                await release.wait()
                if failed:
                    raise RuntimeError("receiver failed")
            return {}

        publisher = RollingPublication(
            None,
            SimpleNamespace(execute_workers=AsyncMock(return_value=[WeightSnapshot(1, (), 0, 0)])),
            SimpleNamespace(core_engine_drain=SimpleNamespace(remote=control)),
            inference,
        )
        publisher.version = 1
        await publisher.publish()
        await entered.wait()
        assert inference.context_lock.locked
        assert not inference._health_checker_activeness.get().active
        release.set()
        if failed:
            with pytest.raises(RuntimeError, match="receiver failed"):
                await publisher._window_task
            inference.prepare_rollout.assert_not_awaited()
        else:
            await publisher._window_task
            inference.prepare_rollout.assert_awaited_once()
        assert not inference.context_lock.locked

    asyncio.run(exercise())
