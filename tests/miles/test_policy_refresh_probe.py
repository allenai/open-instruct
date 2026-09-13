"""Probe boundaries: route capture precedes release; publication stays isolated."""

import importlib.util
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from scripts.miles import launch_policy_refresh_probe as launch


def test_capture_precedes_state_release_and_scope_restores(tmp_path, monkeypatch):
    schedule = ModuleType("sglang.srt.managers.schedule_batch")
    events = []

    def release(**kwargs):
        events.append("release")
        kwargs["req"].req_pool_idx = None

    schedule.release_req = release
    for name in ("sglang", "sglang.srt", "sglang.srt.managers"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["sglang.srt.managers"].schedule_batch = schedule
    monkeypatch.setitem(sys.modules, schedule.__name__, schedule)
    spec = importlib.util.spec_from_file_location(
        "refresh_hooks", launch.Path(launch.__file__).with_name("policy_refresh_hooks.py")
    )
    hooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks)
    hooks.ROOT = tmp_path
    req = SimpleNamespace(
        rid="case-0",
        req_pool_idx=3,
        output_ids=[4, 5],
        origin_input_ids=[1, 2, 3],
        retraction_count=0,
        routed_experts=None,
        output_token_logprobs_val=[-0.2, -0.3],
        output_token_logprobs_idx=[4, 5],
    )

    def capture(req):
        assert req.req_pool_idx == 3
        events.append("capture")

    class Scheduler:
        batch_result_processor = SimpleNamespace(_maybe_collect_routed_experts=capture)

        def pause_generation(self, request):
            schedule.release_req(req=request)
            raise RuntimeError("simulated pause failure")

    hooks.install(Scheduler)
    with pytest.raises(RuntimeError, match="simulated"):
        Scheduler().pause_generation(req)
    assert events == ["capture", "release"]
    assert hooks.CURRENT is None
    retained = json.loads((tmp_path / "case-0-0.json").read_text())
    assert retained["output_ids"] == [4, 5]
    assert retained["behavior_logprobs"] == [-0.2, -0.3]
    # Ordinary memory-pressure retraction outside publication does not capture.
    schedule.release_req(req=req)
    assert events == ["capture", "release", "release"]


@pytest.mark.parametrize("mode", ["tiny", "sft"])
def test_launch_is_pinned_isolated_gpu_holmes(mode, monkeypatch):
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *a, **k: "abc123\n")
    task = launch.make_spec("immutable-image-id", mode)["tasks"][0]
    assert task["resources"]["gpuCount"] == 2
    assert task["constraints"]["cluster"] == ["ai2/holmes"]
    assert task["context"] == {"priority": "urgent", "minRuntime": "1h", "autoResume": False}
    assert task["image"] == {"beaker": "immutable-image-id"}
    assert ("datasets" in task) == (mode == "sft")
    assert "--model" in task["arguments"][0] if mode == "sft" else "--model" not in task["arguments"][0]


def test_unknown_probe_mode_rejected():
    with pytest.raises(ValueError, match="tiny or sft"):
        launch.make_spec("image", "unknown")
