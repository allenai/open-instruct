"""Numerical disagreement remains evidence; incomplete or mismatched inputs fail."""

import json

import pytest
import torch
from scripts.miles import compare_update_zero as compare
from scripts.miles import update_zero_capture as capture


def trace():
    logits = torch.tensor([[4.0, 3.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]])
    routes = capture.route_record(logits, torch.tensor([[0, 1], [0, 1]]), torch.tensor([[0.75, 0.25], [0.75, 0.25]]))
    return {
        "phase": "hf",
        "case_id": "341",
        "capture_id": "hf-341-capture",
        "input_ids": [1, 2],
        "positions": [0, 1],
        "activations": {
            "model.embed_tokens": torch.ones(2, 3),
            "model.layers.0.input": torch.ones(2, 3),
            "model.layers.0": torch.ones(2, 3),
        },
        "routes": {"model.layers.1.mlp.topk": routes},
        "sources": {"source": {"sha256": "abc", "file": "model.py"}},
        "controls": {"torch": "2", "pid": 123},
        "parameter_dtypes": {"router": {"dtype": "torch.float32", "shape": [2, 3]}},
        "next_token_logits": torch.tensor([[0.0, 1.0, 2.0]]),
    }


def test_exact_values_and_unordered_route_sets_are_separate():
    left, right = trace(), trace()
    right["routes"]["model.layers.1.mlp.topk"]["topk_ids"] = torch.tensor([[1, 0], [0, 1]])
    result = compare.compare_captures(left, right)
    assert result["all_activation_values_exact"] and result["all_route_sets_exact"]
    route = result["routes"]["model.layers.1.mlp.topk"]
    assert route["assignment_agreement"] == 1
    assert route["exact_order_fraction"] == 0.5
    assert route["changed_order_positions"] == [0]


def test_first_activation_difference_and_near_margin_route_flip():
    left, right = trace(), trace()
    right["activations"]["model.layers.0"][1, 0] += 0.25
    route = right["routes"]["model.layers.1.mlp.topk"]
    route["logits"][1] = torch.tensor([3.0, 1.0, 2.0, 0.0])
    route["topk_ids"][1] = torch.tensor([0, 2])
    result = compare.compare_captures(left, right)
    assert result["earliest_nonexact_activation"] == "model.layers.0"
    assert result["earliest_changed_route_set"] == "model.layers.1.mlp.topk"
    metrics = result["routes"]["model.layers.1.mlp.topk"]
    assert metrics["changed_set_positions"] == [1]
    assert metrics["stability_bound_violation_positions"] == []
    assert result["activations"]["model.layers.0"]["rows"]["max_abs"] == [0, 0.25]


def test_impossible_route_flip_is_flagged_without_erasing_observation():
    left, right = trace(), trace()
    right["routes"]["model.layers.1.mlp.topk"]["topk_ids"][0] = torch.tensor([0, 3])
    result = compare.compare_captures(left, right)
    assert result["routes"]["model.layers.1.mlp.topk"]["stability_bound_violation_positions"] == [0]


@pytest.mark.parametrize(
    "field,value", [("input_ids", [2, 1]), ("positions", [1, 0]), ("activations", {}), ("routes", {})]
)
def test_different_prefix_or_capture_inventory_rejected(field, value):
    left, right = trace(), trace()
    right[field] = value
    with pytest.raises(ValueError):
        compare.compare_captures(left, right)


def test_dtype_and_source_changes_explicit_even_equal_values():
    left, right = trace(), trace()
    right["activations"]["model.embed_tokens"] = right["activations"]["model.embed_tokens"].bfloat16()
    right["sources"]["source"]["sha256"] = "changed"
    right["controls"]["pid"] = 999
    result = compare.compare_captures(left, right)
    assert result["all_activation_values_exact"]
    assert not result["activations"]["model.embed_tokens"]["dtype_equal"]
    assert result["changed_source_hashes"]
    assert not result["changed_execution_controls"]


def write_trace(root, item):
    directory = root / "trace/worker-123"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{item['capture_id']}.pt"
    torch.save(item, path)
    metadata = {**{k: item[k] for k in ("positions", "sources", "controls")}, "sha256": compare.sha256(path)}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path


def test_load_capture_checks_original_tensor_file_hash(tmp_path):
    item = trace()
    path = write_trace(tmp_path, item)
    result = compare.load_capture(tmp_path, item["capture_id"], [1, 2])
    assert result["input_ids"] == [1, 2]
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash differs"):
        compare.load_capture(tmp_path, item["capture_id"], [1, 2])


def test_separate_retry_roots_full_campaign(tmp_path):
    cases = [{"case_id": str(case), "input_ids": [1, 2]} for case in (341, 975, 1039, 605)]
    roots = {backend: tmp_path / backend / "different-attempt" for backend in compare.BACKENDS}
    for root in roots.values():
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps({"inputs": {"cases": cases}}))
        (root / "probe-complete.json").write_text(
            json.dumps({"completed": True, "optimizer_calls": 0, "initial_publications": 1})
        )
        (root / "cleanup.json").write_text(json.dumps({"completed": True}))
        for name in (
            "resolved-arguments.json",
            "serving-resolved.json",
            "serving-topology.json",
            "initial-weight-comparison.json",
        ):
            (root / name).write_text("{}")
        for phase in compare.PHASES:
            for case in cases:
                for suffix in ("capture", "repeat") if case is cases[0] else ("capture",):
                    item = trace()
                    item.update(phase=phase, case_id=case["case_id"], capture_id=f"{phase}-{case['case_id']}-{suffix}")
                    write_trace(root, item)
                for suffix in ("control", "capture"):
                    (root / f"{phase}-{case['case_id']}-{suffix}-response.json").write_text(
                        json.dumps(
                            {
                                "text": "same",
                                "output_ids": [1],
                                "meta_info": {"output_token_logprobs": [[0.0, 1, None]]},
                            }
                        )
                    )
    report = compare.compare_campaign(tmp_path / "unused", core_root=roots["core"], megatron_root=roots["megatron"])
    assert report["valid"] and len(report["comparisons"]) == 20
    assert len(report["observer_controls"]) == 16
    assert all(row["all_activation_values_exact"] for row in report["comparisons"])
    assert all(row["same_output_ids"] for row in report["observer_controls"])

    (roots["core"] / "cleanup.json").write_text(json.dumps({"completed": False, "errors": ["TimeoutError()"]}))
    with pytest.raises(ValueError, match="cleanup failed"):
        compare.compare_campaign(tmp_path / "unused", core_root=roots["core"], megatron_root=roots["megatron"])
    evidence = compare.compare_campaign(
        tmp_path / "unused", core_root=roots["core"], megatron_root=roots["megatron"], evidence_only=True
    )
    assert evidence["valid"] and evidence["captured_phases_complete"] and not evidence["full_protocol_complete"]
    assert not evidence["cleanup_results"]["core"]["completed"]
    for root in roots.values():
        (root / "probe-complete.json").unlink()
        (root / "cleanup.json").unlink()
    with pytest.raises(FileNotFoundError):
        compare.compare_campaign(tmp_path / "unused", core_root=roots["core"], megatron_root=roots["megatron"])
    limited = compare.compare_campaign(
        tmp_path / "unused", core_root=roots["core"], megatron_root=roots["megatron"], hf_only=True
    )
    assert limited["valid"] and not limited["full_protocol_complete"]
    assert len(limited["comparisons"]) == 6 and len(limited["observer_controls"]) == 8
    assert all(row["comparison"] != "before_after_publication" for row in limited["comparisons"])
    for root in roots.values():
        (root / "manifest.json").write_text(
            json.dumps({"backend": "core", "image": "original", "inputs": {"cases": cases}})
        )
    twins = compare.compare_campaign(tmp_path, core_root=roots["core"], megatron_root=roots["megatron"], hf_only=True)
    assert set(twins["manifests"]) == {"left", "right"}
    assert twins["participants"]["right"]["backend"] == "core"
    assert twins["participants"]["right"]["image"] == "original"
    assert all("backend" not in row for row in twins["observer_controls"])
    assert {row["comparison"] for row in twins["comparisons"]} == {"cross_run", "repeated_prefix"}


def test_autotune_choice_comparison_is_explicit_about_missing_and_empty():
    left, right = trace(), trace()
    assert compare.compare_captures(left, right)["autotune_observations"] == {"available_both": False}
    left["autotune_configs"], right["autotune_configs"] = {}, {}
    observed = compare.compare_captures(left, right)["autotune_observations"]
    assert observed["available_both"] and observed["populated_cache_counts"] == {"left": 0, "right": 0}
    left["autotune_configs"] = {"kernel": {"shape": {"num_warps": 4}}}
    right["autotune_configs"] = {"kernel": {"shape": {"num_warps": 8}}}
    assert "kernel" in compare.compare_captures(left, right)["autotune_observations"]["changed_configs"]
