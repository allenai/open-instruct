"""Dolci-Think-RL preparation keeps Open Instruct's per-domain verifier targets for Megatron GRPO."""

import json

import pytest
from scripts.miles import prepare_dolci_rl_prompts as dolci

from open_instruct import ground_truth_utils
from open_instruct.code_utils import code_utils
from open_instruct.miles.distillation import opd_runtime
from open_instruct.miles.megatron_grpo import grpo_config
from open_instruct.miles.rewards import code_rewards, if_rewards, math_rewards, rewards

SOURCE = {"source": "allenai/Dolci-Think-RL-7B", "revision": "0" * 40}
COMPRESSED = code_utils.encode_tests([{"input": "1\n", "output": "2"}])
IF_TARGET = repr(
    [{"instruction_id": ["change_case:english_lowercase", "punctuation:no_comma"], "kwargs": [None, None]}]
)


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and enable_thinking is False
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"


def source_row(name, truth, prompt="user: Solve it.", **extra):
    return {
        "id": f"{name}-1",
        "prompt": prompt,
        "ground_truth": [truth],
        "dataset": [name],
        "dataset_source": "hamishivi/source",
        "original_dataset": "hamishivi/original",
        **extra,
    }


@pytest.mark.parametrize(
    "name,truth,domain,target",
    [
        ("math", "99", "math", "99"),
        ("ifeval", IF_TARGET, "ifeval", IF_TARGET),
        ("code", json.dumps(["assert f() == 1"]), "code", ["assert f() == 1"]),
        ("code_stdio", json.dumps([{"input": "1\n", "output": "2"}]), "code", [{"input": "1\n", "output": "2"}]),
        # The execution service decodes this form itself, so it passes through unchanged.
        ("code_stdio", COMPRESSED, "code", COMPRESSED),
    ],
)
def test_render_row_keeps_open_instruct_verifier_and_target(name, truth, domain, target):
    row, outcome = dolci.render_row(source_row(name, truth), Tokenizer(), SOURCE)
    assert outcome == domain
    assert row["input"] == "<user>Solve it.</user><assistant>"
    assert row["metadata"]["query"] == "Solve it."
    assert row["metadata"]["domain"] == domain
    assert row["metadata"]["verifiers"] == [{"name": name, "target": target, "weight": 1.0}]


@pytest.mark.parametrize(
    "row,reason",
    [
        (source_row("general-quality_ref", "x"), "other_domain"),
        (source_row("math", "1", prompt="user: Hi\nassistant: Hello\nuser: 1+0?"), "not_single_user_turn"),
        (source_row("math", "1", prompt="system: Be brief."), "not_single_user_turn"),
        (source_row("code", "not json"), "malformed_target"),
        (source_row("code", "[]"), "malformed_target"),
        (source_row("code_stdio", "eJxLTEoGAAJNASc="), "malformed_target"),
        (source_row("ifeval", repr([{"instruction_id": ["a"], "kwargs": []}])), "malformed_target"),
        ({**source_row("math", "1"), "dataset": ["math", "ifeval"]}, "not_single_verifier"),
        ({**source_row("math", "1"), "ground_truth": ["1", "2"]}, "not_single_target"),
    ],
)
def test_render_row_drops_rows_outside_the_study(row, reason):
    assert dolci.render_row(row, Tokenizer(), SOURCE) == (None, reason)


def test_split_holds_out_a_fixed_set_and_removes_it_from_training():
    rows = [{"input": str(i % 30)} for i in range(40)]
    held_out, train = dolci.split_domain(rows, 5, seed=3)
    assert (held_out, train) == dolci.split_domain(rows, 5, seed=3)
    assert len(held_out) == 5
    assert not {row["input"] for row in held_out} & {row["input"] for row in train}


def test_mix_keeps_the_ratio_without_repeating_a_prompt():
    train = {
        "math": [{"input": f"m{i}"} for i in range(30)],
        "ifeval": [{"input": f"i{i}"} for i in range(30)],
        "code": [{"input": f"c{i}"} for i in range(21)],
    }
    rows, counts = dolci.mix_rows(train, {"math": 1, "ifeval": 1, "code": 1}, seed=0)
    assert counts == {"math": 21, "ifeval": 21, "code": 21}
    assert len({row["input"] for row in rows}) == len(rows) == 63
    _, counts = dolci.mix_rows(train, {"math": 2, "code": 1}, seed=0)
    assert counts == {"math": 30, "code": 15}
    assert dolci.parse_mix("pair=math:2,code") == ("pair", {"math": 2.0, "code": 1.0})
    with pytest.raises(SystemExit):
        dolci.parse_mix("pair=math,math")


def test_registry_builds_the_verifiers_the_rows_name():
    registry = dolci.verifier_registry(dolci.DEFAULT_CODE_API_URL)
    assert set(registry) == {name for names in dolci.DOMAINS.values() for name in names}
    built = {name: rewards._resolve_verifier(spec) for name, spec in registry.items()}
    assert isinstance(built["ifeval"], ground_truth_utils.IFEvalVerifier)
    assert isinstance(built["code"], code_rewards.CodeVerifier)
    assert not built["code"].config.stdio and built["code_stdio"].config.stdio
    assert built["code"].config.pass_rate_reward_threshold == 0.99
    args = code_rewards.SimpleNamespace(code_api_url=built["code_stdio"].config.api_url)
    assert code_rewards.code_verifier_config(args, stdio=True).api_url.endswith("/test_program_stdio")


def test_ifeval_target_scores_with_open_instruct_partial_credit():
    verifier = ground_truth_utils.IFEvalVerifier()
    assert verifier([], "all lowercase, with a comma", IF_TARGET).score == 0.5
    assert verifier([], "all lowercase and no commas", IF_TARGET).score == 1.0


@pytest.mark.parametrize("architecture", ["qwen3.5-4B", "qwen3.5-9B"])
def test_larger_qwen35_learners_use_the_qwen35_checkpoint_type(architecture):
    # Their Megatron profiles come from the pinned MILES scripts/models, not this repository.
    assert grpo_config.ARCHITECTURES[architecture] == "qwen3_5"
    assert not (opd_runtime.PROFILES / f"{architecture}.py").exists()


def test_reward_fixes_add_answer_format_and_hints():
    row, _ = dolci.render_row(source_row("math", "99"), Tokenizer(), SOURCE, reward_fixes=True)
    assert row["metadata"]["query"].endswith(math_rewards.MATH_BOXED_INSTRUCTION)
    assert row["metadata"]["prompt_fixes"] == ["boxed_instruction"]
    boxed = source_row("math", "99", prompt="user: Solve it. Put the answer in \\boxed{}.")
    assert dolci.render_row(boxed, Tokenizer(), SOURCE, reward_fixes=True)[0]["metadata"]["prompt_fixes"] == []
    code, _ = dolci.render_row(
        source_row("code", json.dumps(["assert add(1, 2) == 3"])), Tokenizer(), SOURCE, reward_fixes=True
    )
    assert "`add`" in code["metadata"]["query"] and code["metadata"]["prompt_fixes"] == ["function_name_hint"]
    stdio, _ = dolci.render_row(
        source_row("code_stdio", json.dumps([{"input": "1\n", "output": "2"}])), Tokenizer(), SOURCE, reward_fixes=True
    )
    assert stdio["metadata"]["prompt_fixes"] == ["python_hint"]
    assert "prompt_fixes" not in dolci.render_row(source_row("math", "99"), Tokenizer(), SOURCE)[0]["metadata"]


@pytest.mark.parametrize(
    "row,reason",
    [
        (source_row("math", "-1"), "math_placeholder_target"),
        (source_row("code_stdio", json.dumps([{"input": ["1"], "output": ["2"]}])), "code_non_string_stdio_test"),
    ],
)
def test_reward_fixes_drop_ungradable_rows(row, reason):
    assert dolci.render_row(row, Tokenizer(), SOURCE) != (None, reason)
    assert dolci.render_row(row, Tokenizer(), SOURCE, reward_fixes=True) == (None, reason)


def test_reward_fixes_registry_masks_code_failures_and_scores_if_strictly():
    registry = dolci.verifier_registry(dolci.DEFAULT_CODE_API_URL, reward_fixes=True)
    built = {name: rewards._resolve_verifier(spec) for name, spec in registry.items()}
    assert isinstance(built["ifeval"], if_rewards.StrictIFVerifier)
    assert built["code"].config.failure_policy == built["code_stdio"].config.failure_policy == "mask"


def test_split_by_normalized_query_removes_near_duplicate_training_rows():
    def row(i, text):
        return {"input": f"<{i}>", "metadata": {"query": text}}

    rows = [row(i, f"Question {i % 10}.") for i in range(10)] + [row(10 + i, f"question  {i}") for i in range(10)]
    held_out, train = dolci.split_domain(rows, 5, seed=0, key=dolci.normalized_query)
    held = {dolci.normalized_query(r) for r in held_out}
    assert held and not held & {dolci.normalized_query(r) for r in train}
    # Each held-out question's differently punctuated twin leaves training too.
    assert len(train) == 20 - 2 * len(held)


def test_reward_fixes_drop_unsatisfiable_and_trivial_if_rows():
    lower_and_capital = repr(
        [{"instruction_id": ["change_case:english_lowercase", "change_case:english_capital"], "kwargs": [None, None]}]
    )
    assert dolci.render_row(source_row("ifeval", lower_and_capital), Tokenizer(), SOURCE, reward_fixes=True) == (
        None,
        "ifeval_contradiction",
    )
    trivial = repr([{"instruction_id": ["punctuation:no_comma"], "kwargs": [None]}])
    assert dolci.render_row(source_row("ifeval", trivial), Tokenizer(), SOURCE, reward_fixes=True) == (
        None,
        "ifeval_trivial_answer_passes",
    )
    row, _ = dolci.render_row(source_row("ifeval", IF_TARGET), Tokenizer(), SOURCE, reward_fixes=True)
    assert row is None or row["metadata"]["domain"] == "ifeval"
