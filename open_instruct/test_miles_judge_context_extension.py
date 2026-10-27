"""Context extensions must be explicit and leave immutable weights untouched."""

import hashlib
import json

import pytest

from open_instruct.miles.rewards import judge_server, judging


def service(tmp_path):
    template = tmp_path / "template.jinja"
    template.write_text("<think>\n\n</think>")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "max_position_embeddings": 40960, "rope_scaling": None})
    )
    value = {
        "model": "Qwen/Qwen3-32B",
        "revision": "a" * 40,
        "prepared_dir": str(tmp_path),
        "max_context_length": 40960,
        "tensor_parallel_size": 1,
        "max_concurrent_calls": 4,
    }
    (tmp_path / "prepared.json").write_text(
        json.dumps(
            {
                "verdict": "passed",
                "model": value["model"],
                "revision": value["revision"],
                "template": str(template),
                "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
                "rendered_canary": template.read_text(),
                "snapshot": str(tmp_path),
            }
        )
    )
    return value


def test_long_context_requires_explicit_scaling_and_preserves_checkpoint(tmp_path):
    value = service(tmp_path)
    original = (tmp_path / "config.json").read_bytes()
    assert "--json-model-override-args" not in judge_server.command(value, 1234)
    value["max_context_length"] = 131072
    with pytest.raises(ValueError, match="context_extension"):
        judge_server.command(value, 1234)
    value["context_extension"] = "qwen3-yarn-128k"
    command = judge_server.command(value, 1234)
    scaling = json.loads(command[command.index("--json-model-override-args") + 1])
    assert scaling["max_position_embeddings"] == 131072
    assert scaling["rope_scaling"] == {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
    assert scaling["rope_parameters"] == {**scaling["rope_scaling"], "rope_theta": 1000000.0}
    assert (tmp_path / "config.json").read_bytes() == original
    value["max_context_length"] = 131073
    with pytest.raises(ValueError, match="at most 131072"):
        judge_server.command(value, 1234)


def test_extension_is_validated_by_run_schema(tmp_path):
    value = service(tmp_path)
    value.update(mode="managed", max_context_length=131072, context_extension="qwen3-yarn-128k")
    document = {
        "judges": {"general": value},
        "rubrics": {"quality": {"profile": "open-instruct/general-quality"}},
        "judging": {"bindings": {"general-quality": {"judge": "general", "rubric": "quality"}}},
    }
    assert judging.parse(document)["judges"]["general"]["context_extension"] == "qwen3-yarn-128k"
    # Omitting the new opt-in must preserve historical resolved run fingerprints.
    assert "cuda_graphs" not in judging.parse(document)["judges"]["general"]
    value["context_extension"] = "typo"
    with pytest.raises(ValueError, match="context_extension"):
        judging.parse(document)


def test_decode_graph_opt_in_preserves_preparation_guards(tmp_path):
    value = service(tmp_path)
    eager = judge_server.command(value, 1234)
    assert "--disable-cuda-graph" in eager
    assert "--cuda-graph-max-bs-decode" not in eager
    assert judge_server.command({**value, "cuda_graphs": False}, 1234) == eager
    value.update(cuda_graphs=True, max_concurrent_calls=64)
    graph = judge_server.command(value, 1234)
    assert "--disable-cuda-graph" not in graph
    assert graph[graph.index("--cuda-graph-max-bs-decode") + 1] == "64"
    assert graph[graph.index("--max-running-requests") + 1] == "64"
    (tmp_path / "template.jinja").write_text("changed template")
    with pytest.raises(ValueError, match="template hash mismatch"):
        judge_server.command(value, 1234)


@pytest.mark.parametrize("graphs", [False, True, "true", 1])
def test_graph_setting_is_boolean_and_managed_only(tmp_path, graphs):
    value = service(tmp_path)
    value.update(mode="managed", cuda_graphs=graphs)
    document = {
        "judges": {"general": value},
        "rubrics": {"quality": {"profile": "open-instruct/general-quality"}},
        "judging": {"bindings": {"general-quality": {"judge": "general", "rubric": "quality"}}},
    }
    if isinstance(graphs, bool):
        assert judging.parse(document)["judges"]["general"]["cuda_graphs"] is graphs
    else:
        with pytest.raises(ValueError, match="cuda_graphs"):
            judging.parse(document)
    document["judges"]["general"] = {
        "mode": "external",
        "model": "judge",
        "endpoint": "http://judge:8000",
        "cuda_graphs": graphs,
    }
    with pytest.raises(ValueError, match=r"external judge cannot.*judges.general: cuda_graphs"):
        judging.parse(document)
