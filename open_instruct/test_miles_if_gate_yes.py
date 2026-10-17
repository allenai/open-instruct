"""CPU contracts for H087's opt-in reward and the exact H080 client."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration import run_spec
from open_instruct.miles.datasets import run_data
from open_instruct.miles.rewards import (
    general_judge,
    h080,
    if_gate_yes,
    judge_metrics,
    judge_registry,
    judging,
    rewards,
)


def registry_document():
    return {
        "judges": {"general": copy.deepcopy(h080.SERVICE)},
        "rubrics": {"gate": {"profile": h080.PROFILE, "max_response_tokens": 16, "temperature": 0.0}},
        "judging": {"bindings": {"ifeval": {"judge": "general", "rubric": "gate"}}},
    }


@pytest.fixture
def gate(monkeypatch, tmp_path):
    value = judging.registry(judging.parse(registry_document()))
    value["judges"]["general"]["endpoint"] = "http://judge.invalid/v1"
    monkeypatch.setenv(judging.REGISTRY_ENV, json.dumps(value))
    path = tmp_path / "verifiers.json"
    path.write_text(json.dumps(judge_registry.verifier_overrides(judging.parse(registry_document()))))
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(reward_config=str(path), reward_final_answer_only=True, reward_zero_truncated=True),
        seed=17,
    )
    sample = SimpleNamespace(
        prompt="policy template is not judge evidence",
        response="reasoning</think>\n  final  \n",
        status="completed",
        tokens=[1, 2],
        response_length=2,
        metadata={
            "query": "User request",
            "verifiers": [
                {
                    "name": "ifeval",
                    "target": {"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["final"]}]},
                }
            ],
        },
    )
    return args, sample


def test_h080_rendering_fixtures_and_prompt_hash():
    fixtures = json.loads((Path(__file__).parents[1] / "tests/miles/fixtures/h080-rendering.json").read_text())
    assert hashlib.sha256(h080.PROMPT_PATH.read_bytes()).hexdigest() == h080.PROMPT_SHA256
    for case in fixtures:
        assert h080.render_conversation(case["messages"]) == case["conversation"]
        prompt = h080.render_prompt(case["conversation"], case["response"])
        assert prompt.encode() == case["prompt"].encode()
        assert hashlib.sha256(prompt.encode()).hexdigest() == case["prompt_sha256"]


def test_rubric_drift_fails_closed(monkeypatch, tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_bytes(h080.PROMPT_PATH.read_bytes() + b" ")
    monkeypatch.setattr(h080, "PROMPT_PATH", path)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        h080.render_prompt("user:\nx", "y")


def test_preparation_requires_all_policy_turns_without_rewriting_rows():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)

    messages = [
        {"role": "user", "content": "First request"},
        {"role": "assistant", "content": "Earlier response"},
        {"role": "user", "content": "Final request"},
    ]
    row = {"input": json.dumps(messages), "metadata": {"query": "Final request", "verifiers": [{"name": "ifeval"}]}}
    with pytest.raises(ValueError, match="complete policy input"):
        run_data._verify_gate_conversation(row, Tokenizer(), "template", {"ifeval"})
    row["metadata"]["judge_conversation"] = messages
    before = copy.deepcopy(row)
    run_data._verify_gate_conversation(row, Tokenizer(), "template", {"ifeval"})
    assert row == before


@pytest.mark.parametrize(
    "reply,label",
    [
        ("Verdict: yes", "yes"),
        (" \nVeRdIcT: PARTIAL.\r\n", "partial"),
        ("Verdict: no.", "no"),
        ("yes", None),
        ("Verdict:  yes", None),
        ("Verdict: yes..", None),
        ("Verdict: yes\nexplanation", None),
        ("<think>x</think>Verdict: yes", None),
        ('"Verdict: yes"', None),
        ("Verdict: yes\nVerdict: no", None),
        ("Verdict:\tyes", None),
    ],
)
def test_h080_full_match_parser(reply, label):
    assert h080.parse_verdict(reply) == label


@pytest.mark.parametrize(
    "messages", [[], {}, [{"role": "assistant", "content": "x"}], [{"role": "user", "content": ["not text"]}]]
)
def test_never_flatten_or_drop_conversation(messages):
    with pytest.raises(ValueError):
        h080.render_conversation(messages)


@pytest.mark.parametrize(
    "keywords,label,constraint,constraint_pass,expected",
    [
        (["final"], "yes", 1.0, 1.0, 1.0),
        (["final", "pear"], "yes", 0.5, 0.0, 0.0),
        (["final"], "partial", 1.0, 1.0, 0.0),
        (["final"], "no", 1.0, 1.0, 0.0),
        (["pear"], "yes", 0.0, 0.0, 0.0),
        (["pear"], "partial", 0.0, 0.0, 0.0),
        (["pear"], "no", 0.0, 0.0, 0.0),
    ],
)
def test_binary_loose_product_judges_even_zero_pass(
    gate, monkeypatch, keywords, label, constraint, constraint_pass, expected
):
    args, sample = gate
    sample.metadata["verifiers"][0]["target"] = {
        "instruction_id": ["keywords:existence"] * len(keywords),
        "kwargs": [{"keywords": [word]} for word in keywords],
    }
    seen = []
    metrics = judge_metrics.JudgeMetrics()

    async def request(config, prompt):
        assert config.seed == 1 and args.seed == 17
        assert config.max_tokens == 16 and config.max_concurrent_calls == 32
        assert config.temperature == 0 and config.check_context and not config.scale_timeout
        seen.append(prompt)
        return f"Verdict: {label}"

    monkeypatch.setattr(general_judge, "async_request", request)
    monkeypatch.setattr(judge_metrics, "tracker", lambda: metrics)

    async def score():
        assert await rewards.registered_reward(args, sample) == expected
        state = await metrics.snapshot()
        assert (state["arrived"], state["completed"], state["skipped"]) == (1, 1, 0)

    asyncio.run(score())
    assert seen == [h080.render_prompt("user:\nUser request", "\n  final  \n")]
    diagnostics = sample.metadata["verifier_diagnostics"]["ifeval"]
    assert diagnostics["constraint_score"] == constraint
    assert diagnostics["constraint_pass"] == constraint_pass
    assert diagnostics["label"] == label
    assert diagnostics["status"] == "ok"


@pytest.mark.parametrize(
    "status,response,reason",
    [
        ("truncated", "x</think>final", "truncated"),
        ("completed", "no closing tag", "missing_final_answer"),
        ("completed", "reason</think> \n", "missing_final_answer"),
    ],
)
def test_h070_pre_gates_do_not_call_either_scorer(gate, monkeypatch, status, response, reason):
    args, sample = gate
    sample.status, sample.response = status, response

    async def forbidden(*args, **kwargs):
        pytest.fail("zeroed sample must not reach either scorer")

    monkeypatch.setattr(if_gate_yes.task_verifiers.ManifestIFVerifier, "async_call", forbidden)
    monkeypatch.setattr(general_judge, "async_request", forbidden)
    assert asyncio.run(rewards.registered_reward(args, sample)) == 0
    assert sample.metadata["reward_gate"] == reason
    assert sample.metadata["verifier_diagnostics"]["ifeval"]["status"] == "skipped"


@pytest.mark.parametrize("policy", ["zero", "raise"])
def test_retry_exhaustion_follows_failure_policy(monkeypatch, policy):
    replies = iter(["invalid", "Verdict: maybe", "explanation"])

    async def request(*args):
        return next(replies)

    monkeypatch.setattr(general_judge, "async_request", request)
    metadata = {}
    operation = if_gate_yes.score_prompt(
        if_gate_yes.client_config(h080.SERVICE, "http://judge/v1"),
        "user:\nx",
        "y",
        policy=policy,
        metadata=metadata,
        name="ifeval",
    )
    if policy == "zero":
        assert asyncio.run(operation) == 0
    else:
        with pytest.raises(general_judge.JudgeResponseError):
            asyncio.run(operation)
    record = metadata["verifier_diagnostics"]["ifeval"]
    assert record["label"] is None and len(record["attempts"]) == 3
    assert record["status"] == "judge_error"


@pytest.mark.parametrize("policy", [None, "raise"])
@pytest.mark.parametrize("failure", ["parse", "transport"])
@pytest.mark.parametrize("constraint_pass", [0, 1])
def test_training_retry_exhaustion_halts_even_zero_pass(gate, monkeypatch, policy, failure, constraint_pass):
    args, sample = gate
    if policy is None:
        monkeypatch.delenv("OI_MILES_JUDGE_FAILURE_POLICY", raising=False)
    else:
        monkeypatch.setenv("OI_MILES_JUDGE_FAILURE_POLICY", policy)
    if not constraint_pass:
        sample.metadata["verifiers"][0]["target"]["kwargs"] = [{"keywords": ["pear"]}]
    seen = []

    async def request(*args):
        seen.append(1)
        if failure == "transport":
            raise general_judge.JudgeResponseError("transport failure")
        return "invalid"

    monkeypatch.setattr(general_judge, "async_request", request)
    with pytest.raises(general_judge.JudgeResponseError, match="exhausted three attempts"):
        asyncio.run(rewards.registered_reward(args, sample))
    assert len(seen) == 3
    record = sample.metadata["verifier_diagnostics"]["ifeval"]
    assert record["failure_policy"] == "raise"
    assert record["status"] == "judge_error" and record["label"] is None
    assert record["constraint_pass"] == constraint_pass
    assert "fallback_reward" not in record


def test_retry_can_recover_without_becoming_partial_credit(monkeypatch):
    replies = iter(["invalid", "Verdict: partial"])

    async def request(*args):
        return next(replies)

    monkeypatch.setattr(general_judge, "async_request", request)
    record = asyncio.run(
        if_gate_yes.VerdictClient(if_gate_yes.client_config(h080.SERVICE, "http://judge/v1")).grade("user:\nx", "y")
    )
    assert record["label"] == "partial" and len(record["attempts"]) == 2


def test_context_failures_remain_fatal_and_measured(monkeypatch):
    async def run():
        metrics = judge_metrics.JudgeMetrics()

        async def fail(*args):
            raise RuntimeError("context overflow; no truncation")

        monkeypatch.setattr(general_judge, "async_request", fail)
        client = if_gate_yes.VerdictClient(if_gate_yes.client_config(h080.SERVICE, "http://judge/v1"), metrics)
        with pytest.raises(RuntimeError, match="overflow"):
            await client.grade("user:\nx", "y")
        state = await metrics.snapshot()
        assert (state["arrived"], state["completed"], state["failed"], state["backlog"], state["attempts"]) == (
            1,
            1,
            1,
            0,
            3,
        )

    asyncio.run(run())


def test_pinned_client_has_literal_600_second_timeout(monkeypatch):
    seen = []

    def post(url, **kwargs):
        seen.append((url, kwargs))
        payload = (
            {"count": 20}
            if url.endswith("/tokenize")
            else {"choices": [{"finish_reason": "stop", "message": {"content": "Verdict: yes"}}]}
        )
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    monkeypatch.setenv("MILES_INFRA_TIMEOUT_MULTIPLIER", "5")
    monkeypatch.setattr(general_judge, "_get_session", lambda: SimpleNamespace(post=post))
    cfg = if_gate_yes.client_config(h080.SERVICE, "http://judge/v1")
    assert general_judge._request(cfg, "full prompt") == "Verdict: yes"
    assert [kwargs["timeout"] for _, kwargs in seen] == [600, 600]
    assert seen[1][1]["json"] == {
        "model": h080.SERVICE["model"],
        "messages": [{"role": "user", "content": "full prompt"}],
        "temperature": 0.0,
        "max_tokens": 16,
        "seed": 1,
    }


def test_live_metrics_measure_queued_and_in_flight_and_skips():
    async def run():
        metrics = judge_metrics.JudgeMetrics()
        metrics.arrive()
        metrics.admit()
        metrics.arrive()
        metrics.skip()
        state = await metrics.snapshot()
        assert (state["arrived"], state["queued"], state["in_flight"], state["backlog"], state["skipped"]) == (
            2,
            1,
            1,
            2,
            1,
        )
        metrics.finish(True, 1, 2, 1)
        metrics.admit()
        metrics.finish(False, 3, 4, 2)
        state = await metrics.snapshot()
        assert (state["arrived"], state["completed"], state["failed"], state["attempts"], state["backlog"]) == (
            2,
            2,
            1,
            4,
            0,
        )
        assert state["latency_seconds_sum"] >= state["service_seconds_sum"] >= 0

    asyncio.run(run())


@pytest.mark.parametrize("remote", [False, True])
def test_repeated_cancellation_during_admission_does_not_leak_slots(monkeypatch, remote):
    async def run():
        counters = judge_metrics.JudgeMetrics()

        class RemoteMethod:
            def __init__(self, method):
                self.method = method

            def remote(self, *args):
                self.method(*args)
                # Model a Ray call whose response never arrives. Admission and
                # cleanup must not wait for any actor response.
                return asyncio.get_running_loop().create_future()

        actor = SimpleNamespace(
            **{name: RemoteMethod(getattr(counters, name)) for name in ("arrive", "admit", "finish")}
        )
        metrics = judge_metrics._Remote(actor) if remote else counters
        release = asyncio.Event()
        active = peak = requests = 0

        async def request(*args):
            nonlocal active, peak, requests
            active += 1
            requests += 1
            peak = max(peak, active)
            try:
                await release.wait()
                return "Verdict: yes"
            finally:
                active -= 1

        monkeypatch.setattr(general_judge, "async_request", request)
        config = if_gate_yes.client_config(h080.SERVICE, "http://judge/v1")

        def grade():
            # New clients must share the same process-local admission limit.
            return asyncio.create_task(if_gate_yes.VerdictClient(config, metrics).grade("user:\nx", "y"))

        holders = [grade() for _ in range(32)]
        await asyncio.sleep(0)
        assert active == 32
        for _ in range(4):
            waiting = [grade() for _ in range(64)]
            await asyncio.sleep(0)
            state = await counters.snapshot()
            assert (state["queued"], state["in_flight"]) == (64, 32)
            for task in waiting:
                task.cancel()
            outcomes = await asyncio.gather(*waiting, return_exceptions=True)
            assert all(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes)
            state = await counters.snapshot()
            assert (state["queued"], state["in_flight"]) == (0, 32)
            assert requests == 32
        # Cancel waiters after release wakes them but before they can resume;
        # asyncio.Semaphore must return those grants to the available pool.
        woken = [grade() for _ in range(32)]
        await asyncio.sleep(0)
        for task in holders:
            task.cancel()
        await asyncio.sleep(0)
        for task in woken:
            task.cancel()
        await asyncio.gather(*holders, *woken, return_exceptions=True)
        assert requests == 32
        assert (await counters.snapshot())["backlog"] == 0
        # Fill all 32 slots again and queue another full batch; all must finish.
        continuing = [grade() for _ in range(64)]
        await asyncio.sleep(0)
        assert active == 32
        assert (await counters.snapshot())["queued"] == 32
        release.set()
        records = await asyncio.wait_for(asyncio.gather(*continuing), timeout=2)
        assert all(record["label"] == "yes" for record in records)
        state = await counters.snapshot()
        assert state["arrived"] == state["completed"] == 32 + 4 * 64 + 32 + 64
        assert state["failed"] == 32 + 4 * 64 + 32
        assert state["backlog"] == state["queued"] == state["in_flight"] == 0
        assert requests == 96 and peak == 32

    asyncio.run(run())


def test_failed_metrics_delivery_cannot_block_grading(monkeypatch):
    def unavailable(*args):
        raise RuntimeError("metrics actor unavailable")

    method = SimpleNamespace(remote=unavailable)
    metrics = judge_metrics._Remote(SimpleNamespace(arrive=method, admit=method, finish=method, skip=method))

    async def request(*args):
        return "Verdict: yes"

    monkeypatch.setattr(general_judge, "async_request", request)
    metrics.skip()
    record = asyncio.run(
        if_gate_yes.VerdictClient(if_gate_yes.client_config(h080.SERVICE, "http://judge/v1"), metrics).grade(
            "user:\nx", "y"
        )
    )
    assert record["label"] == "yes"


def test_consumed_metrics_never_count_skips_as_grades():
    records = [
        {"status": "skipped"},
        {"status": "ok", "label": "partial", "elapsed_seconds": 2},
        {"status": "judge_error", "elapsed_seconds": 4},
    ]
    samples = [
        SimpleNamespace(metadata={"verifier_diagnostics": {"ifeval": {"kind": "general_judge", **r}}}) for r in records
    ]
    metrics = judge_metrics.consumed_metrics(samples)
    assert metrics["rollout/general_judge/samples"] == 2
    assert metrics["rollout/general_judge/skipped"] == 1
    assert metrics["rollout/general_judge/errors"] == 1
    assert metrics["rollout/general_judge/partial"] == 1
    assert metrics["rollout/general_judge/latency_seconds_mean"] == 3


def test_existing_h070_constraint_call_is_the_unchanged_parent():
    # One of two keywords passes => H070's fractional value, not all-or-none loose P.
    target = {
        "instruction_id": ["keywords:existence", "keywords:existence"],
        "kwargs": [{"keywords": ["apple"]}, {"keywords": ["pear"]}],
    }
    assert asyncio.run(if_gate_yes.task_verifiers.ManifestIFVerifier().async_call([], "apple", target)).score == 0.5
    assert if_gate_yes.prompt_level_loose_pass("apple", target) == 0.0


@pytest.mark.parametrize(
    "response",
    [
        '**{"ok": true}**',
        'Preface\n{"ok": true}',
        '{"ok": true}\nAfterword',
        'Preface\n{"ok": true}\nAfterword',
        'Preface\n**{"ok": true}**\nAfterword',
    ],
)
def test_loose_pass_checks_variants_instead_of_thresholding_h070_credit(response):
    target = {"instruction_id": ["detectable_format:json_format"], "kwargs": [{}]}
    assert asyncio.run(if_gate_yes.task_verifiers.ManifestIFVerifier().async_call([], response, target)).score == 0.0
    assert if_gate_yes.prompt_level_loose_pass(response, target) == 1.0


def test_each_constraint_can_pass_a_different_loose_variant():
    target = {"instruction_id": ["startend:quotation", "keywords:existence"], "kwargs": [{}, {"keywords": ["footer"]}]}
    response = '"apple"\nfooter'
    assert asyncio.run(if_gate_yes.task_verifiers.ManifestIFVerifier().async_call([], response, target)).score == 0.5
    assert if_gate_yes.prompt_level_loose_pass(response, target) == 1.0


def test_loose_pass_can_earn_reward_when_h070_credit_is_zero(gate, monkeypatch):
    args, sample = gate
    sample.response = 'reasoning</think>Preface\n**{"ok": true}**\nAfterword'
    sample.metadata["verifiers"][0]["target"] = {"instruction_id": ["detectable_format:json_format"], "kwargs": [{}]}

    async def request(config, prompt):
        assert prompt == h080.render_prompt("user:\nUser request", sample.response.partition("</think>")[2])
        return "Verdict: yes"

    monkeypatch.setattr(general_judge, "async_request", request)
    assert asyncio.run(rewards.registered_reward(args, sample)) == 1.0
    record = sample.metadata["verifier_diagnostics"]["ifeval"]
    assert record["constraint_score"] == 0.0 and record["constraint_pass"] == 1.0


def test_named_override_and_data_validation_keep_constraint_factory(gate, tmp_path):
    overrides = judge_registry.verifier_overrides(judging.parse(registry_document()))
    assert overrides["ifeval"]["factory"].endswith("IfGateYesVerifier")
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"metadata": {"query": "raw user", "verifiers": [{"name": "ifeval", "target": "bundle"}]}}) + "\n"
    )
    judge_registry.validate_data({"prompt_data": str(path)}, judge_registry.registry())


def test_recipe_schema_accepts_pinned_gate_and_rejects_drift():
    document = {
        "schema_version": 1,
        "name": "gate",
        "model": {"source": "/weka/hf"},
        "output": {"root": "/weka/trial"},
        "data": {"prompt_data": "/weka/train", "reward_config": "/weka/verifiers"},
        **registry_document(),
    }
    run_spec.RunSpec.from_dict(document)
    for section, key, value in [
        ("judges", "revision", "a" * 40),
        ("rubrics", "seed", 17),
        ("rubrics", "max_response_tokens", 32),
    ]:
        changed = copy.deepcopy(document)
        changed[section]["general" if section == "judges" else "gate"][key] = value
        with pytest.raises(ValueError):
            run_spec.RunSpec.from_dict(changed)
