"""Ownership, failure and reward contracts, with real local HTTP and subprocesses."""

import asyncio
import importlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from open_instruct.miles import cluster, general_judge, judge_registry, judging, launch, rewards, topology
from open_instruct.miles.run_spec import RunSpec


@pytest.fixture
def document(tmp_path):
    return {
        "schema_version": 1,
        "name": "judge-test",
        "model": {"source": "/weka/oe-training-default/hf"},
        "output": {"root": "/weka/oe-training-default/trial"},
        "data": {"tasks": [{"task": "multiplication", "train_count": 8}]},
        "launch": {"gpus_per_replica": 8, "auto_resume": False},
        "trainer": {"gpus": 8, "expert_parallel_size": 8},
        "inference": {"gpus": 7, "placement_mode": "disaggregated"},
        "judges": {
            "general": {
                "mode": "managed",
                "model": "Qwen/Qwen3-32B",
                "revision": "a" * 40,
                "prepared_dir": "/weka/oe-training-default/judge",
            }
        },
        "rubrics": {"quality": {"profile": "open-instruct/general-quality"}},
        "judging": {"bindings": {"general-quality": {"judge": "general", "rubric": "quality"}}},
    }


def test_production_topology_and_growth_are_additive(document):
    run = RunSpec.from_dict(document)
    assert RunSpec.from_dict(run.to_dict()).to_dict() == run.to_dict()
    plan = topology.plan(run)
    assert (plan["replicas"], plan["allocated_gpus"], plan["policy_gpus"], plan["judge_gpus"]) == (2, 16, 15, 1)
    # More serving GPUs never displace trainer GPUs or the fixed judge.
    document["inference"]["gpus"] = 23
    larger = topology.plan(RunSpec.from_dict(document))
    assert larger["replicas"] == 4 and larger["allocated_gpus"] == 32
    assert sum(n["rollout_gpus"] for n in larger["nodes"]) == 23
    assert sum(n["trainer_gpus"] for n in larger["nodes"]) == 8
    # Eight plus eight plus a judge is 17 requested GPUs, transparently rounded
    # to three homogeneous replicas, rather than silently dropping an engine.
    document["inference"]["gpus"] = 8
    larger = topology.plan(RunSpec.from_dict(document))
    assert larger["replicas"] == 3 and larger["unused_gpus"] == 7


def test_numeric_node_order_and_disjoint_devices(document):
    plan = topology.plan(RunSpec.from_dict(document))
    nodes = topology.assign(plan, ["10.0.0.12", "10.0.0.2"])
    assert nodes["10.0.0.2"]["trainer_gpus"] == 8
    visible = [f"GPU-{i}" for i in range(8)]
    policy, judges = topology.devices(nodes["10.0.0.12"], visible)
    assert policy == visible[:7] and judges == {"general": [visible[7]]}
    with pytest.raises(ValueError, match="distinct"):
        topology.assign(plan, ["10.0.0.2"] * 2)
    with pytest.raises(ValueError, match="visibility"):
        topology.devices(nodes["10.0.0.12"], visible[:7])


def test_task_replication_and_secret_free_ownership(document):
    task = launch.specification("image", RunSpec.from_dict(document))["tasks"][0]
    assert task["replicas"] == 2 and task["resources"]["gpuCount"] == 8
    assert all(task[key] for key in ("leaderSelection", "hostNetworking", "propagateFailure", "propagatePreemption"))
    assert "open_instruct.miles.cluster" in task["arguments"][0]
    assert not any("BEAKER_TOKEN" in v["name"] for v in task["envVars"])


def test_wrong_ray_node_gpu_layout_fails():
    nodes = [
        {"Alive": True, "NodeManagerAddress": "10.0.0.1", "Resources": {"GPU": 8}},
        {"Alive": True, "NodeManagerAddress": "10.0.0.2", "Resources": {"GPU": 8}},
    ]
    with pytest.raises(RuntimeError, match="ownership"):
        cluster.require_layout(nodes, {"10.0.0.1": 8, "10.0.0.2": 7})
    nodes[1]["Resources"]["GPU"] = 7
    cluster.require_layout(nodes, {"10.0.0.1": 8, "10.0.0.2": 7})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["judges"]["general"].update(revision="main"),
        lambda d: d["judges"]["general"].update(endpoint="http://localhost/v1"),
        lambda d: d["judges"]["general"].update(gpus=2),
        lambda d: d["judging"]["bindings"]["general-quality"].update(judge="typo"),
        lambda d: d["rubrics"]["quality"].update(max_response_tokens=40960),
        lambda d: d["rubrics"]["quality"].update(temperature=float("nan")),
    ],
)
def test_invalid_judge_contracts_fail_before_gpu_submission(document, mutate):
    mutate(document)
    with pytest.raises(ValueError):
        RunSpec.from_dict(document)


@pytest.fixture
def server():
    state = {
        "count": 10,
        "finish_reason": "stop",
        "content": '{"REASONING":"ok","SCORE":8}',
        "requests": [],
        "status": 200,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["requests"].append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            result = (
                {"count": state["count"]}
                if self.path == "/tokenize"
                else {"choices": [{"finish_reason": state["finish_reason"], "message": {"content": state["content"]}}]}
            )
            self.send_response(state["status"])
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever)
    thread.start()
    config = general_judge.GeneralJudgeConfig(
        api_url=f"http://127.0.0.1:{http.server_port}/v1/chat/completions",
        api_key="EMPTY",
        model="test",
        max_tokens=20,
        max_context_length=100,
        temperature=0,
        timeout=2,
        seed=1,
        max_concurrent_calls=2,
        check_context=True,
    )
    try:
        yield config, state
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_http_context_checks_full_input_without_truncating(server):
    config, state = server
    assert general_judge.parse_judge_response(general_judge._request(config, "full evidence"))[1] == 0.8
    assert state["requests"][0][0] == "/tokenize"
    assert state["requests"][1][1]["messages"] == [{"role": "user", "content": "full evidence"}]
    state["count"] = 81
    state["requests"].clear()
    with pytest.raises(RuntimeError, match="overflow"):
        general_judge._request(config, "full evidence")
    assert len(state["requests"]) == 1


@pytest.mark.parametrize("finish", ["length", "error", None])
def test_incomplete_grades_raise_with_raw_evidence(server, finish):
    config, state = server
    state["finish_reason"] = finish
    with pytest.raises(general_judge.JudgeResponseError) as caught:
        general_judge._request(config, "input")
    assert caught.value.diagnostics["raw_reply"] == state["content"]


def test_transport_failure_is_not_reward_zero(server):
    config, state = server
    state["status"] = 503
    with pytest.raises(RuntimeError, match="request failed"):
        general_judge._request(config, "input")


def test_named_reward_uses_full_query_and_keeps_diagnostics(document, monkeypatch, tmp_path, server):
    config, state = server
    value = judging.registry(judging.parse(document))
    value["judges"]["general"]["endpoint"] = config.api_url.removesuffix("/chat/completions")
    monkeypatch.setenv(judging.REGISTRY_ENV, json.dumps(value))
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "general-quality": {
                    "factory": "open_instruct.miles.judge_registry.NamedJudgeVerifier",
                    "config": {"name": "general-quality"},
                }
            }
        )
    )
    sample = SimpleNamespace(
        tokens=[1],
        response_length=1,
        response="Answer",
        prompt="rendered",
        metadata={
            "query": "last user message",
            "judge_query": "entire conversation",
            "verifiers": [{"name": "general-quality", "target": "", "weight": 1}],
        },
    )
    args = SimpleNamespace(olmo_core=SimpleNamespace(reward_config=str(path)), seed=17)
    assert asyncio.run(rewards.registered_reward(args, sample)) == 0.8
    request = state["requests"][-1][1]["messages"][0]["content"]
    assert "entire conversation" in request and "last user message" not in request
    assert sample.metadata["verifier_diagnostics"]["general-quality"]["raw_reply"]


def test_supervisor_reaps_child_on_failure(document, tmp_path):
    run = RunSpec.from_dict(document)
    owner = cluster.Supervisor(run, tmp_path, 0, 1)
    owner.thread.start()
    process = owner.start("child", [sys.executable, "-c", "import time; time.sleep(60)"], dict(os.environ))
    try:
        cluster.write(tmp_path / "failed-1.json", {"error": "peer service died"})
        with pytest.raises(RuntimeError, match="Peer failure"):
            owner.check()
    finally:
        owner.close()
    assert process.poll() is not None


def test_missing_reference_and_unused_bindings_are_rejected(document, tmp_path):
    value = judging.registry(judging.parse(document))
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"metadata": {"verifiers": []}}) + "\n")
    with pytest.raises(ValueError, match="absent"):
        judge_registry.validate_data({"prompt_data": str(path)}, value)


@pytest.mark.parametrize("stage", ["prepare", "inspect", "audit"])
def test_cpu_judge_stages_always_use_saturn(document, stage):
    module = importlib.import_module("scripts.miles.launch_judge_preparation")
    task = module.specification("image", RunSpec.from_dict(document), stage)["tasks"][0]
    assert task["constraints"]["cluster"] == ["ai2/saturn"]
    assert "gpuCount" not in task["resources"] and "replicas" not in task
    assert "preflight_attention" not in task["arguments"][0]
    subprocess.run(["bash", "-n"], input=task["arguments"][0], text=True, check=True)


def test_health_probe_tolerates_transient_failures_but_bounds_outage(document, tmp_path, monkeypatch):
    owner = cluster.Supervisor(RunSpec.from_dict(document), tmp_path, 0, 1)
    owner.health["general"] = "http://127.0.0.1/health"

    def fail(*args, **kwargs):
        raise TimeoutError("busy")

    monkeypatch.setattr(cluster.urllib.request, "urlopen", fail)
    owner.probe_health()
    owner.probe_health()
    assert owner.health_failures["general"] == 2

    class Healthy:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(cluster.urllib.request, "urlopen", lambda *args, **kwargs: Healthy())
    owner.probe_health()
    assert owner.health_failures["general"] == 0
    monkeypatch.setattr(cluster.urllib.request, "urlopen", fail)
    owner.probe_health()
    owner.probe_health()
    with pytest.raises(RuntimeError, match="three consecutive"):
        owner.probe_health()
