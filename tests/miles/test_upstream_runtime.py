"""Contracts at the upstream worker, publication, and token-provenance boundaries."""

import asyncio
import json
import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.types import WeightVersionSpan, WeightVersionsPerCall
from miles.utils.workers.worker_spec import CommandWorkerSpec, SchedulingSpec, WorkerLaunchContext

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.infrastructure import startup_cache
from open_instruct.miles.publication import policy_versions
from open_instruct.miles.publication.engine_drain import WeightSnapshot
from open_instruct.miles.publication.rolling_publication import RollingPublication
from open_instruct.miles.training import actor


def test_serving_child_receives_model_package_and_its_own_cache_slot():
    spec = CommandWorkerSpec(
        name="inference-engine-0-0",
        port_infos=[],
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0.2),
        env_var=lambda ctx: {"EXISTING": "kept"},
        launch_command=lambda ctx: "/usr/bin/python -m sglang.launch_server --model-path '/model with spaces'",
    )
    args = SimpleNamespace(olmo_core_startup_cache={"shared": "/shared", "restore": True})
    _, configured = startup_cache.configure_specs(args, [spec.model_copy(update={"name": "trainer-actor"}), spec])
    context = WorkerLaunchContext(cell_index=2, worker_in_cell_index=1, gpu_ids=[3])
    env = configured.env_var(context)
    assert env["SGLANG_EXTERNAL_MODEL_PACKAGE"] == "olmo_sglang.models"
    assert env["EXISTING"] == "kept"
    assert json.loads(env[startup_cache.ENV])["slot"] == "inference-engine-0-0-2-1"
    assert shlex.split(configured.launch_command(context)) == [
        "/usr/bin/python",
        "-m",
        "open_instruct.miles.rollout.serving",
        "--model-path",
        "/model with spaces",
    ]
    assert "worker_process_setup_hook" not in env
    assert spec.env_var(context) == {"EXISTING": "kept"}


@pytest.mark.parametrize(
    "names", [[], ["renamed-trainer", "inference-engine-0"], ["trainer-actor", "renamed-serving"]]
)
def test_startup_hooks_reject_missing_worker_pools(names):
    with pytest.raises(ValueError, match="worker specifications"):
        startup_cache.configure_specs(None, [SimpleNamespace(name=name) for name in names])


@pytest.mark.parametrize("mode", ["barrier", "refresh", "engine_drain"])
def test_core_publication_never_aborts_live_decoding(monkeypatch, mode):
    worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
    worker.args = SimpleNamespace(
        olmo_core=CoreConfig(publication_mode=mode), colocate=False, update_weight_buffer_size=1024, save=None
    )
    worker.clock = SimpleNamespace(completed_steps=3, published_step=2, published=Mock())
    worker.weight_updater = Mock()
    worker.train_module = worker.hf_config = None
    engine = SimpleNamespace(
        **{
            name: AsyncMock()
            for name in (
                "pause_generation",
                "begin_weight_update",
                "flush_cache",
                "end_weight_update",
                "update_weight_version",
                "continue_generation",
            )
        }
    )
    monkeypatch.setattr(actor.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(actor.dist, "barrier", lambda: None)
    monkeypatch.setattr(actor.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor.models, "iter_export_state", lambda *a, **kw: iter(()))
    assert (
        worker.update_weights(
            SimpleNamespace(
                rollout_engines=[engine],
                engine_gpu_counts=[1],
                engine_gpu_offsets=[0],
                snapshot_cell_id_to_hashes={"engine": "hash"},
            )
        )
        == 3
    )
    engine.update_weight_version.assert_awaited_once_with("3", abort_all_requests=False)
    engine.continue_generation.assert_awaited_once()


def test_empty_generation_calls_do_not_invent_versions_or_break_metrics():
    empty = WeightVersionsPerCall.from_meta_info({"weight_version": "7", "completion_tokens": 0}, output_end=5)
    assert policy_versions.versions([empty], allow_empty=True) == []
    with pytest.raises(ValueError, match="behavior policy"):
        policy_versions.versions([empty])
    calls = [empty, WeightVersionsPerCall([WeightVersionSpan("8", 5, 7)])]
    assert policy_versions.versions(calls) == [8]


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
