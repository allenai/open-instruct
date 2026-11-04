"""Screening must keep training population, reward semantics and missingness."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct import test_miles_run_data, test_miles_run_spec
from open_instruct.miles import screening
from open_instruct.miles.configuration.config import CoreConfig, RunConfig
from open_instruct.miles.datasets import run_data
from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import rewards

environment = test_miles_run_data.environment


def test_screen_samples_training_artifacts_without_rewriting_targets(environment, tmp_path):
    data = {
        "sources": [
            {
                "name": "arithmetic",
                "fraction": 1,
                "tasks": [{"task": "multiplication", "train_count": 12, "eval_count": 2}],
            }
        ]
    }
    spec = test_miles_run_spec.spec(tmp_path, data=data)
    hf, output, _ = environment
    prepared = run_data.prepare_run_data(spec, hf, output)
    original = run_data._rows(Path(prepared["prompt_data"]).read_bytes())
    selected, populations = screening.select_prompts(prepared, 8, 17)
    assert populations == [{"name": "arithmetic", "rows": 12, "fraction": 1}]
    assert len(selected) == len({p["id"] for p in selected}) == 8
    assert screening.select_prompts(prepared, 8, 17)[0] == selected
    every_row, _ = screening.select_prompts(prepared, None, 17)
    assert [p["row_index"] for p in every_row] == list(range(12))
    assert len({p["id"] for p in every_row}) == 12
    assert screening.select_prompts(prepared, None, 99)[0] == every_row
    for prompt in selected:
        expected = copy.deepcopy(original[prompt["row_index"]])
        expected["metadata"]["source_name"] = "arithmetic"
        assert prompt["row"] == expected
    assert run_data.prepare_run_data(spec, hf, output) == prepared
    Path(prepared["prompt_data"]).write_text("changed")
    with pytest.raises(InputError, match="Prepared artifact changed"):
        run_data.prepare_run_data(spec, hf, output)


@pytest.mark.parametrize("truncated,final_only,expected", [(False, False, 1), (True, False, 0), (False, True, 0)])
def test_scores_are_the_actual_training_bridge(tmp_path, truncated, final_only, expected):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps({"number": {"factory": "open_instruct.miles.rewards.candidate_verifiers.CandidateVerifier"}})
    )
    row = {
        "input": "A question",
        "metadata": {
            "source_name": "numbers",
            "verifiers": [{"name": "number", "target": {"kind": "numeric", "value": "42"}}],
        },
    }
    completion = {"response": "42", "token_ids": [42], "truncated": truncated}
    config = RunConfig(CoreConfig(reward_config=str(path), reward_final_answer_only=final_only), {})
    actual = asyncio.run(screening.score_completion(config, {"row": row}, completion))
    sample = SimpleNamespace(
        prompt=row["input"],
        response="42",
        tokens=[42],
        response_length=1,
        metadata=copy.deepcopy(row["metadata"]),
        status="truncated" if truncated else "completed",
    )
    direct = asyncio.run(rewards.registered_reward(SimpleNamespace(olmo_core=config.core), sample))
    assert actual["score"] == direct == expected
    assert actual["metadata"] == sample.metadata
    assert "reward_components" not in row["metadata"]


def test_exception_is_unknown_not_zero(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps({"n": {"factory": "open_instruct.miles.rewards.candidate_verifiers.CandidateVerifier"}})
    )
    prompt = {"row": {"input": "q", "metadata": {"verifiers": [{"name": "n", "target": {"kind": "bad"}}]}}}
    result = asyncio.run(
        screening.score_completion(
            RunConfig(CoreConfig(reward_config=str(path)), {}),
            prompt,
            {"response": "42", "token_ids": [], "truncated": False},
        )
    )
    assert result["score"] is None and result["status"] == "scoring_error"


def test_pass_and_variation_do_not_hide_unknown_or_nonbinary_groups():
    records = []
    for identity, values in [("mixed", [0, 1, 0, 0]), ("unknown", [None, 0, 0, 0]), ("partial", [0, 0.5, 0.5, 0])]:
        records.extend({"source": "s", "id": identity, "score": v} for v in values)
    result = screening.summarize(records)["s"]
    assert result["complete_groups"] == 2 and result["unresolved_groups"] == 1
    assert result["metrics"]["1"]["pass"] == 0.25
    assert result["metrics"]["4"]["pass"] == 1
    assert result["metrics"]["4"]["pass_groups"] == 1
    assert result["metrics"]["4"]["varying_reward_groups"] == 2
    assert result["metrics"]["4"]["varying_reward"] == 1
    assert result["metrics"]["16"]["pass"] is None


def test_generation_settings_preserve_training_stops_and_sampling():
    config = RunConfig(
        CoreConfig(),
        {
            "rollout_max_response_len": 1024,
            "rollout_temperature": 0.7,
            "rollout_top_p": 0.95,
            "rollout_top_k": 50,
            "rollout_stop_token_ids": [7],
            "rollout_stop": ["END"],
        },
    )
    params = screening.generation_settings(config)
    assert params["temperature"] == 0.7 and params["top_k"] == 50 and params["top_p"] == 0.95
    assert params["max_new_tokens"] == 1024 and params["stop_token_ids"] == [7] and params["stop"] == ["END"]
    assert params["no_stop_trim"] and not params["spaces_between_special_tokens"]


@pytest.mark.parametrize("count,expected_prompts", [(2, 2), (None, 3)])
def test_screen_generation_resume_and_contract_change(environment, tmp_path, monkeypatch, count, expected_prompts):
    hf, _, tokenizer = environment
    spec = test_miles_run_spec.spec(tmp_path, data={"tasks": [{"task": "multiplication", "train_count": 3}]})
    monkeypatch.setattr(screening.specs, "load", lambda _: spec)
    calls = []

    class Response:
        def __init__(self, value):
            self.value = value

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        async def json(self):
            return self.value

    class Session(Response):
        def __init__(self, **kwargs):
            pass

        def get(self, url):
            return Response({"model_path": spec.model["source"]})

        def post(self, url, json):
            calls.append(json)
            assert "input_ids" in json and "text" not in json
            return Response(
                {
                    "text": "unfinished",
                    "meta_info": {
                        "output_token_logprobs": [[-1, 7, None]],
                        "completion_tokens": 1,
                        "finish_reason": {"type": "length"},
                    },
                }
            )

    monkeypatch.setattr(screening.aiohttp, "ClientSession", Session)
    options = SimpleNamespace(
        config=tmp_path / "run.toml",
        tokenizer=hf,
        output=tmp_path / "screen",
        prompts_per_source=count,
        draws=4,
        seed=17,
        request_timeout=30,
        server="http://fixture",
        concurrency=2,
    )
    first = asyncio.run(screening.run(options))
    assert len(calls) == expected_prompts * 4
    assert first["sources"]["train"]["complete_groups"] == expected_prompts
    assert first["sources"]["train"]["metrics"]["4"]["pass"] == 0
    assert asyncio.run(screening.run(options)) == first
    assert len(calls) == expected_prompts * 4
    options.draws = 8
    with pytest.raises(InputError, match="Screen contract changed"):
        asyncio.run(screening.run(options))


def test_cli_all_is_exclusive_and_default_remains_sampled(tmp_path, monkeypatch):
    counts = []

    async def capture(options):
        counts.append(options.prompts_per_source)

    monkeypatch.setattr(screening, "run", capture)
    args = ["run.toml", "--tokenizer", "hf", "--server", "http://fixture", "--output", str(tmp_path)]
    screening.main(args)
    screening.main([*args, "--all"])
    screening.main([*args, "--prompts-per-source", "7"])
    assert counts == [96, None, 7]
    with pytest.raises(SystemExit) as error:
        screening.main([*args, "--all", "--prompts-per-source", "7"])
    assert error.value.code == 2
