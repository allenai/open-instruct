"""Inference benchmarks must allocate no trainer and account only measured work."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scripts.miles import (
    analyze_inference_benchmark,
    compare_inference_replay,
    inference_counter_metrics,
    prepare_inference_continuations,
    prepare_inference_workload,
)

from open_instruct.miles.configuration import throughput
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.execution import launch
from open_instruct.miles.rollout import benchmark_load, benchmark_metrics


@pytest.mark.parametrize("failure", [None, "foreground", "background"])
def test_replay_load_propagates_failure_and_joins_every_task(failure):
    async def exercise():
        started = asyncio.Event()
        stopped = []

        async def background(index, stop):
            try:
                started.set()
                if failure == "background":
                    raise ValueError("background failed")
                await stop.wait()
            finally:
                stopped.append(index)

        async def foreground():
            await started.wait()
            if failure == "foreground":
                raise ValueError("foreground failed")
            if failure == "background":
                await asyncio.Event().wait()
            return 7

        async def shutdown(workers):
            # The producer stop event must be set before waiting on outstanding
            # HTTP responses; no next wave may inherit old background work.
            await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), 1)

        if failure:
            with pytest.raises(ValueError, match=f"{failure} failed"):
                await benchmark_load.run_wave([foreground()], background, 2, shutdown)
        else:
            assert await benchmark_load.run_wave([foreground()], background, 2, shutdown) == [7]
        assert sorted(stopped) == [0, 1]

    asyncio.run(exercise())


def test_replay_failure_joins_other_probes_before_abort_and_preserves_primary_error():
    async def exercise():
        started = asyncio.Event()
        joined = asyncio.Event()

        async def failing():
            await started.wait()
            raise ValueError("primary probe failure")

        async def pending():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                joined.set()

        async def background(index, stop):
            await stop.wait()

        async def shutdown(workers):
            assert joined.is_set()
            raise RuntimeError("secondary abort failure")

        with pytest.raises(ValueError, match="primary probe failure") as captured:
            await benchmark_load.run_wave([failing(), pending()], background, 1, shutdown)
        assert "secondary abort failure" in captured.value.__notes__[0]

    asyncio.run(exercise())


def document(tmp_path):
    return {
        "schema_version": 1,
        "name": "benchmark",
        "model": {"source": "/weka/models/policy"},
        "data": {"tasks": [{"task": "gsm8k", "train_count": 32}]},
        "output": {"root": str(tmp_path), "export_hf": False},
        "launch": {"auto_resume": False, "weka_mounts": [{"weka": "models", "mount_path": "/weka/models"}]},
        "trainer": {"gpus": 8, "expert_parallel_size": 8},
        "training": {"save_checkpoints": False, "filter_zero_std_groups": False},
        "inference": {"gpus": 1, "placement_mode": "disaggregated"},
        "async": {"fully_async": True},
        "benchmark": {"measure_seconds": 60},
    }


@pytest.mark.parametrize("gpus", [1, 2, 8])
def test_benchmark_allocates_only_inference_and_preserves_trainer_settings(tmp_path, gpus):
    payload = document(tmp_path)
    payload["inference"]["gpus"] = gpus
    spec = RunSpec.from_dict(payload)
    plan = spec.plan()
    assert plan["allocation"]["allocated_gpus"] == gpus
    assert plan["allocation"]["nodes"][0]["trainer_gpus"] == 0
    assert plan["runtime"]["shape"]["trainer_gpus"] == 0
    assert spec.sections["trainer"] == payload["trainer"]
    restored = RunSpec.from_dict(spec.to_dict())
    assert restored.benchmark == spec.benchmark
    task = launch.specification("image", spec)["tasks"][0]
    assert task["resources"]["gpuCount"] == gpus
    assert "preflight_attention" not in task["arguments"][0]


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("training", "filter_zero_std_groups", True, "reward_mode=none"),
        ("training", "save_checkpoints", True, "save_checkpoints=false"),
        ("launch", "auto_resume", True, "auto_resume=false"),
        ("inference", "gpus", 16, "at most eight"),
        ("benchmark", "measure_seconds", 0, "measure_seconds"),
        ("benchmark", "measure_seconds", float("nan"), "measure_seconds"),
        ("benchmark", "reward_mode", "fake", "reward_mode"),
    ],
)
def test_benchmark_rejects_misleading_or_unbounded_runs(tmp_path, section, key, value, match):
    payload = document(tmp_path)
    payload[section][key] = value
    with pytest.raises(ValueError, match=match):
        RunSpec.from_dict(payload)


def test_measured_throughput_excludes_warmup_and_includes_consumer_overhead():
    batches = [
        dict(phase="warmup", started=0, finished=100, groups=100, samples=400, response_tokens=9999, route_bytes=1),
        dict(phase="measure", started=100, finished=105, groups=2, samples=8, response_tokens=200, route_bytes=100),
        dict(phase="measure", started=110, finished=115, groups=2, samples=8, response_tokens=100, route_bytes=100),
    ]
    summary = benchmark_metrics.summarize(batches, 2)
    assert summary["seconds"] == 15
    assert summary["response_tokens_per_second"] == 20
    assert summary["response_tokens_per_gpu_second"] == 10
    assert summary["samples"] == 16


def test_response_checks_reject_missing_payload_and_wrong_policy():
    sample = SimpleNamespace(
        tokens=[1, 2, 3],
        response_length=2,
        rollout_log_probs=[-0.1, -1.0],
        status=SimpleNamespace(value="completed"),
        all_weight_version_spans=[SimpleNamespace(version="0", abs_start=1, abs_end=3)],
        rollout_routed_experts=SimpleNamespace(shape=(2, 16, 16), nbytes=2048),
        index=0,
        group_index=0,
    )
    assert benchmark_metrics.sample_record(sample, require_routes=True)["route_bytes"] == 2048
    for field, value in [
        ("rollout_log_probs", [-0.1]),
        ("rollout_log_probs", [float("nan"), -1]),
        ("all_weight_version_spans", [SimpleNamespace(version="1")]),
        ("all_weight_version_spans", [SimpleNamespace(version="0", abs_start=2, abs_end=3)]),
        ("all_weight_version_spans", [SimpleNamespace(version="0", abs_start=1, abs_end=2)]),
        ("rollout_routed_experts", None),
    ]:
        broken = copy.copy(sample)
        setattr(broken, field, value)
        with pytest.raises(ValueError):
            benchmark_metrics.sample_record(broken, require_routes=True)


def test_panel_resolves_overrides_and_isolates_case_outputs(tmp_path):
    payload = document(tmp_path)
    payload["benchmark"]["variants"] = [
        {"name": "control", "overrides": []},
        {"name": "larger", "overrides": ["inference.sglang_max_running_requests=96"]},
    ]
    spec = RunSpec.from_dict(payload)
    control, larger = spec.benchmark_variants()
    assert control.output["root"] == str(tmp_path / "variants/control")
    assert larger.conversion["hf_output"].startswith(str(tmp_path / "variants/larger"))
    assert control.compile().miles["sglang_max_running_requests"] == 64
    assert larger.compile().miles["sglang_max_running_requests"] == 96
    assert control.benchmark["variants"] == larger.benchmark["variants"] == []
    assert control.plan()["allocation"] == larger.plan()["allocation"] == spec.plan()["allocation"]


@pytest.mark.parametrize(
    "override",
    [
        'model.source="other"',
        "inference.gpus=8",
        "trainer.gpus=1",
        'miles.sglang_dtype="float16"',
        "miles.sglang_tp_size=2",
        "miles.sglang_enable_return_routed_experts=false",
    ],
)
def test_panel_cannot_change_model_workload_or_gpu_allocation(tmp_path, override):
    payload = document(tmp_path)
    payload["benchmark"]["variants"] = [{"name": "bad", "overrides": [override]}]
    with pytest.raises(ValueError, match="serving/queue controls only"):
        RunSpec.from_dict(payload)


def test_miles_router_warns_about_ineffective_policy_and_connection_gate():
    report = throughput.report(
        {
            "use_miles_router": True,
            "sglang_router_policy": "round_robin",
            "rollout_num_gpus": 2,
            "sglang_server_concurrency": 128,
            "miles_router_max_connections": 64,
        },
        SimpleNamespace(),
    )
    assert {warning["code"] for warning in report["warnings"]} == {
        "miles_router_ignores_native_policy",
        "miles_router_connection_limit",
    }


def test_inference_analysis_validates_accounting_and_keeps_missing_occupancy_unknown(tmp_path):
    batches = [
        {
            "phase": "measure",
            "started": 1,
            "finished": 11,
            "finished_unix": 100,
            "groups": 1,
            "samples": 1,
            "response_tokens": 50,
            "route_bytes": 200,
        }
    ]
    files = {
        "workflow.json": {"status": "complete"},
        "benchmark-summary.json": benchmark_metrics.summarize(batches, 2),
        "resolved-plan.json": {"miles": {"rollout_num_gpus": 2}},
        "benchmark-boundaries.json": {},
        "benchmark-shutdown.json": {},
        "benchmark-server-start.json": [],
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    streams = {
        "benchmark-batches.jsonl": batches,
        "benchmark-responses.jsonl": [
            {"phase": "measure", "prompt_tokens": 5, "response_tokens": 50, "route_bytes": 200, "status": "truncated"}
        ],
        "benchmark-resources.jsonl": [],
    }
    for name, values in streams.items():
        (tmp_path / name).write_text("".join(json.dumps(value) + "\n" for value in values))
    report = analyze_inference_benchmark.analyze(tmp_path)
    assert report["summary"]["response_tokens_per_gpu_second"] == 2.5
    assert report["truncated_fraction"] == 1
    assert report["occupancy"]["pipeline"]["http_active_requests"]["mean"] is None
    files["benchmark-summary.json"]["response_tokens"] = 100
    (tmp_path / "benchmark-summary.json").write_text(json.dumps(files["benchmark-summary.json"]))
    with pytest.raises(ValueError, match="summary disagrees"):
        analyze_inference_benchmark.analyze(tmp_path)


def test_engine_counters_preserve_missing_reset_and_scrape_uncertainty():
    key = 'sglang:realtime_tokens_total{mode="decode",model="a b"}'
    document = {
        "measure_start": [{"url": "engine", "started": 1, "finished": 2, "text": f"{key} 100\n"}],
        "measure_end": [{"url": "engine", "started": 11, "finished": 12, "text": f"{key} 300\n"}],
    }
    report = inference_counter_metrics.analyze(document)["engines"][0]
    assert report["scrape_window_seconds_bounds"] == [9, 11]
    assert report["series"][0]["per_second"] == 20
    document["measure_end"][0]["text"] = f"{key} 20\n"
    assert "decreased" in inference_counter_metrics.analyze(document)["engines"][0]["series"][0]["missing"]
    document["measure_start"][0]["text"] = ""
    assert "absent" in inference_counter_metrics.analyze(document)["engines"][0]["series"][0]["missing"]
    document["measure_end"][0]["error"] = "timeout"
    assert "failed" in inference_counter_metrics.analyze(document)["engines"][0]["missing"]
    with pytest.raises(ValueError, match="nonfinite"):
        inference_counter_metrics.parse(f"{key} NaN")
    with pytest.raises(ValueError, match="duplicated"):
        inference_counter_metrics.parse(f"{key} 1\n{key} 2")


def test_replay_requires_explicit_diagnostic_server_setting(tmp_path):
    payload = document(tmp_path)
    payload["benchmark"]["replay_reference"] = str(tmp_path / "reference")
    with pytest.raises(ValueError, match="custom_logit_processor"):
        RunSpec.from_dict(payload)
    payload.setdefault("miles", {})["sglang_enable_custom_logit_processor"] = True
    assert RunSpec.from_dict(payload).benchmark["replay_repeats"] == 2
    assert RunSpec.from_dict(payload).benchmark["replay_copies"] == 1
    payload["benchmark"]["replay_copies"] = 0
    with pytest.raises(ValueError, match="replay_copies"):
        RunSpec.from_dict(payload)


def test_core_reference_requires_a_single_gpu_and_reference(tmp_path):
    payload = document(tmp_path)
    payload["benchmark"]["score_reference"] = True
    with pytest.raises(ValueError, match="Core reference"):
        RunSpec.from_dict(payload)
    payload["benchmark"]["replay_reference"] = str(tmp_path / "reference")
    assert RunSpec.from_dict(payload).benchmark["score_reference"]
    payload["benchmark"]["variants"] = [{"name": "unused", "overrides": []}]
    with pytest.raises(ValueError, match="no variants"):
        RunSpec.from_dict(payload)


def test_frozen_reference_validates_model_completion_and_lengths(tmp_path):
    (tmp_path / "run-spec.json").write_text('{"model": {"source": "fixed"}}')
    (tmp_path / "workflow.json").write_text('{"status": "complete"}')
    record = {"index": 0, "prompt_tokens": 1, "response_tokens": 2, "tokens": [1, 2, 3]}
    (tmp_path / "benchmark-captures.jsonl").write_text(json.dumps(record) + "\n")
    _, rows = benchmark_metrics.reference_records(tmp_path, {"source": "fixed"}, 3)
    assert rows == [record]
    with pytest.raises(ValueError, match="same checkpoint"):
        benchmark_metrics.reference_records(tmp_path, {"source": "changed"}, 3)
    with pytest.raises(ValueError, match="exceeds"):
        benchmark_metrics.reference_records(tmp_path, {"source": "fixed"}, 2)
    (tmp_path / "workflow.json").write_text('{"status": "running"}')
    with pytest.raises(ValueError, match="completed"):
        benchmark_metrics.reference_records(tmp_path, {"source": "fixed"}, 3)


def test_core_comparison_measures_paired_error_changes(tmp_path):
    (tmp_path / "workflow.json").write_text('{"status": "complete"}')
    (tmp_path / "benchmark-summary.json").write_text(
        json.dumps({"reference_backend": "core", "reference_sha256": "fixed", "trajectories": 1})
    )
    row = {"index": 0, "group_index": 0, "prompt_tokens": 1, "response_tokens": 2, "logprobs": [-1.0, -1.0]}
    (tmp_path / "benchmark-core-reference.jsonl").write_text(json.dumps(row) + "\n")
    left = [{**row, "logprobs": [-1.1, -1.1]}]
    right = [{**row, "logprobs": [-1.3, -1.3]}]
    report = compare_inference_replay.core_comparison(tmp_path, "fixed", left, right)
    assert report["paired_change"]["mean_absolute_error_change"]["value"] == pytest.approx(0.2)
    assert report["paired_change"]["ratio_outlier_fraction_change"]["value"] == 1
    with pytest.raises(ValueError, match="same captured reference"):
        compare_inference_replay.core_comparison(tmp_path, "changed", left, right)


def test_core_position_bands_expose_late_error_without_adding_short_prompt_clusters(tmp_path):
    (tmp_path / "workflow.json").write_text('{"status": "complete"}')
    (tmp_path / "benchmark-summary.json").write_text(
        json.dumps({"reference_backend": "core", "reference_sha256": "fixed", "trajectories": 2})
    )
    rows = [
        {"index": i, "group_index": i, "prompt_tokens": 1, "response_tokens": n, "logprobs": [-1.0] * n}
        for i, n in enumerate((1025, 2))
    ]
    (tmp_path / "benchmark-core-reference.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    changed = copy.deepcopy(rows)
    changed[0]["logprobs"][-1] = -1.5
    report = compare_inference_replay.core_comparison(tmp_path, "fixed", rows, changed)
    assert report["paired_change"]["mean_absolute_error_change"]["value"] < 0.001
    early, late = report["by_response_position"]
    assert early["paired_change"]["mean_absolute_error_change"]["value"] == 0
    assert late["start"] == 1024
    assert late["right_error"]["prompt_clusters"] == 1
    assert late["right_error"]["tokens_including_repeats"] == 1
    assert late["paired_change"]["mean_absolute_error_change"]["value"] == 0.5


def test_replay_wave_reports_do_not_hide_opposite_signed_changes(tmp_path):
    paths = [tmp_path / side for side in ("left", "right")]
    row = {"index": 0, "group_index": 0, "prompt_tokens": 1, "response_tokens": 2, "logprobs": [-1.0, -1.0]}
    for side, path in enumerate(paths):
        path.mkdir()
        (path / "workflow.json").write_text('{"status": "complete"}')
        (path / "benchmark-summary.json").write_text(
            json.dumps(
                {
                    "diagnostic_only": True,
                    "reference_sha256": "fixed",
                    "trajectories": 1,
                    "repeats": 2,
                    "response_tokens_per_repeat": 2,
                }
            )
        )
        for repeat in range(2):
            changed = {**row, "logprobs": [-1.0 + side * (0.1 if repeat == 0 else -0.1)] * 2}
            (path / f"benchmark-replay-{repeat}.jsonl").write_text(json.dumps(changed) + "\n")
    report = compare_inference_replay.compare(*paths)
    assert report["metrics"]["mean_signed_logprob_delta"]["value"] == pytest.approx(0)
    assert [x["metrics"]["mean_signed_logprob_delta"]["value"] for x in report["per_wave"]] == pytest.approx(
        [0.1, -0.1]
    )


def test_replay_copies_keep_distinct_ids_and_shared_prompt_clusters():
    original = [{"index": 0, "group_index": 0}, {"index": 3, "group_index": 0}, {"index": 5, "group_index": 1}]
    expanded = benchmark_metrics.replicate_trajectories(original, 4)
    assert len({row["index"] for row in expanded}) == 12
    assert {row["group_index"] for row in expanded} == {0, 1}
    assert [row["source_index"] for row in expanded] == [0, 3, 5] * 4
    assert original == [{"index": 0, "group_index": 0}, {"index": 3, "group_index": 0}, {"index": 5, "group_index": 1}]


def test_route_comparison_counts_response_conditioning_positions(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    a = np.zeros((4, 2, 2), dtype=np.int32)
    b = a.copy()
    b[0, 0, 0] = 3  # Earlier prompt route is outside response-conditioning positions.
    b[1, 1, 0] = 4
    records = []
    for root, matrix in ((left, a), (right, b)):
        root.mkdir()
        np.savez_compressed(root / "routes.npz", routes=matrix)
        records.append(
            [
                {
                    "group_index": 0,
                    "prompt_tokens": 2,
                    "response_tokens": 3,
                    "route_path": "routes.npz",
                    "route_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                }
            ]
        )
    report = compare_inference_replay.route_agreement(left, right, *records)
    assert report["coverage_fraction"] == 1
    assert report["expert_slot_changed_fraction"]["value"] == pytest.approx(1 / 12)
    assert report["conditioning_token_changed_fraction"]["value"] == pytest.approx(1 / 3)
    assert report["per_layer_expert_slot_changed_fraction"] == [0, pytest.approx(1 / 6)]
    records[1][0]["route_sha256"] = "invalid"
    with pytest.raises(ValueError, match="hash"):
        compare_inference_replay.route_agreement(left, right, *records)


def test_route_comparison_excludes_dense_padding_and_separates_expert_membership(tmp_path):
    paths = [tmp_path / name for name in ("left", "right")]
    x = np.zeros((3, 2, 2), dtype=np.int32)
    x[:, 1] = [1, 2]
    y = x.copy()
    y[:, 1] = [2, 1]  # Same experts, opposite top-k order.
    y[-1, 1] = [2, 3]  # One genuinely replaced expert in one response position.
    records = []
    for path, matrix in zip(paths, (x, y), strict=True):
        path.mkdir()
        (path / "benchmark-model-config.json").write_text(
            json.dumps({"model_type": "olmo3moe", "num_hidden_layers": 2, "dense_layers_indices": [0]})
        )
        np.savez_compressed(path / "routes.npz", routes=matrix)
        records.append(
            [
                {
                    "group_index": 0,
                    "prompt_tokens": 1,
                    "response_tokens": 3,
                    "route_path": "routes.npz",
                    "route_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                }
            ]
        )
    report = compare_inference_replay.route_agreement(*paths, *records)
    assert report["routed_layer_indices"] == [1]
    assert report["per_layer_expert_slot_changed_fraction"] == [None, 1]
    assert report["expert_slot_changed_fraction"]["value"] == 1
    assert report["expert_membership_changed_fraction"]["value"] == pytest.approx(1 / 6)
    assert report["conditioning_token_expert_set_changed_fraction"]["value"] == pytest.approx(1 / 3)
    assert not report["expert_id_contract_checked"]


@pytest.mark.parametrize("fault", [None, "negative", "out_of_range", "duplicate", "width"])
def test_route_comparison_rejects_identically_corrupt_payloads(tmp_path, fault):
    matrix = np.zeros((3, 2, 2), dtype=np.int32)
    matrix[:, 1] = [1, 2]
    # Corrupt an earlier prompt position, outside response-conditioning metrics.
    if fault in {"negative", "out_of_range", "duplicate"}:
        matrix[0, 1, 0] = {"negative": -1, "out_of_range": 4, "duplicate": 2}[fault]
    if fault == "width":
        matrix = np.pad(matrix, ((0, 0), (0, 0), (0, 1)))
    paths, records = [], []
    for name in ("left", "right"):
        path = tmp_path / name
        path.mkdir()
        paths.append(path)
        (path / "benchmark-model-config.json").write_text(
            json.dumps(
                dict(
                    model_type="olmo3moe",
                    num_hidden_layers=2,
                    dense_layers_indices=[0],
                    n_routed_experts=4,
                    num_experts_per_tok=2,
                )
            )
        )
        np.savez_compressed(path / "routes.npz", routes=matrix)
        records.append(
            [
                dict(
                    group_index=0,
                    prompt_tokens=2,
                    response_tokens=2,
                    route_path="routes.npz",
                    route_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
                )
            ]
        )
    if fault:
        with pytest.raises(ValueError, match="out-of-range|duplicate expert|top-k dimension"):
            compare_inference_replay.route_agreement(*paths, *records)
    else:
        report = compare_inference_replay.route_agreement(*paths, *records)
        assert report["expert_id_contract_checked"]
        assert report["expert_slot_changed_fraction"]["value"] == 0


def test_workload_preparation_filters_without_truncating_and_keeps_sources(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer_config.json").write_text("{}")
    tokenizer = SimpleNamespace(chat_template="template", encode=lambda text, **kwargs: list(text))
    monkeypatch.setattr(prepare_inference_workload.run_data, "_tokenizer", lambda path: tokenizer)
    candidates = [
        {
            "input": text,
            "metadata": {"prepared_sample_id": str(index), "verifiers": [{"name": "gsm8k", "target": "1"}]},
        }
        for index, text in enumerate(["too long", "one", "two", "last"])
    ]
    monkeypatch.setattr(prepare_inference_workload.run_data, "_tasks", lambda *args: ({"train": candidates}, {}, None))
    result = prepare_inference_workload.prepare(model, tmp_path / "corpus", counts={"gsm8k": 2}, max_prompt_length=3)
    rows = [json.loads(line) for line in Path(result["prompt_data"]).read_text().splitlines()]
    assert {r["input"] for r in rows} == {"one", "two"}
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert all(Path(path).is_file() for path in manifest["inputs"])


def test_continuation_preparation_preserves_exact_prefixes_and_prompt_clusters(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer_config.json").write_text("{}")

    def decode(ids, **kwargs):
        # Simulate a token boundary that does not survive text roundtripping.
        return "".join(map(chr, ids[:-1] if len(ids) == 5 else ids))

    tokenizer = SimpleNamespace(
        chat_template="template", encode=lambda text, **kwargs: list(map(ord, text)), decode=decode
    )
    monkeypatch.setattr(prepare_inference_workload.run_data, "_tokenizer", lambda path: tokenizer)
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "run-spec.json").write_text(
        json.dumps({"model": {"source": str(model)}, "inference": {"max_context_length": 16}})
    )
    (reference / "workflow.json").write_text(json.dumps({"status": "complete"}))
    records = [
        {
            "index": index,
            "group_index": 0,
            "tokens": list(map(ord, text)),
            "prompt_tokens": 1,
            "response_tokens": len(text) - 1,
        }
        for index, text in enumerate(["abcdefgh", "abcdefgx", "abc"])
    ]
    (reference / "benchmark-captures.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    result = prepare_inference_continuations.prepare(
        model, tmp_path / "corpus", [reference], prompt_tokens=5, boundary_slack=1, remaining_tokens=2
    )
    rows = [json.loads(line) for line in Path(result["prompt_data"]).read_text().split("\n") if line]
    assert len(rows) == 1
    assert rows[0]["input"] == "abcd"
    # The compiled MILES data source always selects label_key="label", even
    # when this benchmark has no reward target.
    assert rows[0]["label"] is None
    assert rows[0]["metadata"]["prompt_token_ids"] == list(map(ord, "abcd"))
    assert len(rows[0]["metadata"]["continuation_origins"]) == 2
    selection = json.loads((tmp_path / "corpus/selection.json").read_text())
    assert selection["original_prompt_clusters"] == 1
    assert selection["excluded"][0]["sample_index"] == 2
    assert selection["actual_prompt_lengths"] == [4]
    with pytest.raises(ValueError, match="no reward targets"):
        prepare_inference_continuations.BenchmarkOnlyVerifier()
    with pytest.raises(ValueError, match="No captured continuations"):
        prepare_inference_continuations.prepare(
            model, tmp_path / "no-boundary", [reference], prompt_tokens=5, boundary_slack=0, remaining_tokens=2
        )
    assert not (tmp_path / "no-boundary").exists()


def test_replay_comparison_pairs_prompt_clusters_and_rejects_different_trajectories():
    left = [
        {
            "index": i,
            "group_index": i // 2,
            "prompt_tokens": 1,
            "response_tokens": 2,
            "logprobs": [-1.0, -2.0],
            "route_sha256": "same",
        }
        for i in range(8)
    ]
    exact = compare_inference_replay.paired_metrics(left, left)
    assert exact["prompt_clusters"] == 4
    assert exact["metrics"]["mean_absolute_logprob_delta"]["value"] == 0
    right = copy.deepcopy(left)
    for row in right:
        row["logprobs"] = [-0.5, -1.5]
    shifted = compare_inference_replay.paired_metrics(left, right)
    assert shifted["metrics"]["mean_signed_logprob_delta"]["prompt_cluster_95pct_interval"] == [0.5, 0.5]
    assert shifted["metrics"]["ratio_outside_0_8_to_1_2_fraction"]["value"] == 1
    right[0]["index"] = 9
    with pytest.raises(ValueError, match="identities"):
        compare_inference_replay.paired_metrics(left, right)
