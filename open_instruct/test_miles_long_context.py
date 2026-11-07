"""Long-context contracts reject answer spraying and preserve source prompts."""

import asyncio
import copy
import json
import math

import pytest
from scripts.data.rl_readiness import prepare_long_context
from scripts.data.rl_readiness.qualify_bundle import check_control

from open_instruct.miles.datasets import long_context as adapters
from open_instruct.miles.datasets import source_preparation as sources
from open_instruct.miles.rewards import long_context as rewards
from open_instruct.test_miles_source_preparation import Tokenizer, prepare_one


@pytest.mark.parametrize(
    "metric,gold,response,expected",
    [
        ("exact", ["42"], "42", 1),
        ("exact", ["42"], "142", 0),
        ("exact", ["42"], "The answer is 42", 0),
        ("docqa", ["Ada Lovelace"], "ADA  LOVELACE.", 1),
        ("docqa", ["Ada Lovelace"], "Ada Lovelace or Charles Babbage", 0),
        ("decimal", ["14.285714"], "14.2857", 1),
        ("decimal", ["14.285714"], "114.285714", 0),
        ("decimal", ["14.285714"], "NaN", 0),
        ("decimal", ["14.285714"], "1e999999", 0),
        ("line_accuracy", ["a\na\nb"], "a\na\nb", 1),
        ("line_accuracy", ["a\na\nb"], "a\nb", 1 / 3),
        ("line_accuracy", ["a\nb"], "a\nb\nextra", 2 / 3),
        ("line_accuracy", ["a\nb"], "b\na", 0),
        ("lines_f1", ["doc1", "doc2"], "doc1", 2 / 3),
        ("lines_f1", ["doc1", "doc2"], "doc1\ndoc1", 2 / 3),
        ("lines_f1", ["doc1", "doc2"], "doc1\ndoc2\nwrong", 0.8),
        ("ndcg", ["first", "second", "third"], "first\nsecond\nthird", 1),
        ("ndcg", ["first", "second", "third"], "first\nfirst", 3 / (3 + 2 / math.log2(3) + 0.5)),
    ],
)
def test_reward_contract(metric, gold, response, expected):
    target = adapters.target(metric, gold)
    assert rewards.score("reasoning</think>" + response, target) == pytest.approx(expected)
    assert rewards.score("", target) == 0
    assert rewards.score("<think>" + response, target) == 0


@pytest.mark.parametrize("response", ["C", "(C)", "The correct answer is (C).", "Reasoning</think>C"])
def test_choice(response):
    assert rewards.score(response, adapters.target("choice", ["C"], "choice")) == 1


@pytest.mark.parametrize("response", ["AC", "A B C D", "The correct answer is (C). The correct answer is (D)."])
def test_choice_not_substring(response):
    assert rewards.score(response, adapters.target("choice", ["C"], "choice")) == 0


def test_last_answer_region():
    target = adapters.target("lines_f1", ["gold"], "answer_marker")
    assert rewards.score("[Answer]\ngold\n[Answer]\nwrong", target) == 0
    assert rewards.score("[答案]\ngold", target) == 1
    assert rewards.score("Therefore, the answer is 2.", adapters.target("decimal", ["2"], "therefore")) == 1


@pytest.mark.parametrize("answers", [["a", "a"], ["A", "a"]])
def test_duplicate_rank_reference_rejected(answers):
    with pytest.raises(ValueError, match="duplicate"):
        adapters.target("ndcg", answers)


@pytest.mark.parametrize(
    "task,body,gold",
    [
        ("unnumbered_streaming_sum", "Input:\n\n1+2-4\n2+3", "1+2-4=-1\n2+3=5"),
        ("streaming_sum", "Input:\n\n[1]2+3\n[2]3-4", "[1]2+3=5\n[2]3-4=-1"),
        ("shuffled_streaming_sum", "Input:\n\n[2]3-4\n[1]2+3", "[1]2+3=5\n[2]3-4=-1"),
        ("unnumbered_uuid_sort", "Input:\n\nffffffff,00000000", "00000000,ffffffff"),
        ("streaming_uuid_sort", "Input:\n\n[1]ffffffff,00000000", "[1]00000000,ffffffff"),
        ("shuffled_streaming_uuid_sort", "Input:\n\n[1]ffffffff,00000000", "[1]00000000,ffffffff"),
        ("unnumbered_var_expand", "a=cat\nb=dog\n\nExpressions:\n\na+b", "cat dog"),
        ("streaming_var_expand", "a=cat\nb=dog\n\nExpressions:\n\n[1]b+a", "[1]dog cat"),
        ("shuffled_streaming_var_expand", "a=cat\nb=dog\n\nExpressions:\n\n[1]b+a", "[1]dog cat"),
        ("csv_kv_lookup", "a1=cat\nv1=dog\n\nInput:\n\n,c1\nr1,a1+v1", ",c1\nr1,cat dog"),
        (
            "csv_permutation_homogeneous",
            "Input:\n\n,c1,c2\nr1,A,B\n\nReordering Instructions:\nNew row order: r1\nNew col order: c2,c1",
            ",c2,c1\nr1,B,A",
        ),
        (
            "csv_permutation_heterogeneous",
            "Input:\n\n,c1,c2\nr1,A,2\n\nReordering Instructions:\nNew row order: r1\nNew col order: c2,c1",
            ",c2,c1\nr1,2,A",
        ),
    ],
)
def test_independent_reference(task, body, gold):
    raw = {
        "messages": [
            {"role": "user", "content": "Here is the real task:\n" + body},
            {"role": "assistant", "content": gold},
        ],
        "metadata": {"task_type": task},
    }
    original = copy.deepcopy(raw)
    ms, target, verifier, stratum = sources.adapt(raw, "lc_transduction_v1")
    assert raw == original
    assert ms == raw["messages"][:1]
    assert verifier == "long_context" and stratum == task
    assert rewards.score(gold, target) == 1
    raw["messages"][1]["content"] = "incorrect publisher reference"
    with pytest.raises(ValueError, match="reference mismatch"):
        sources.adapt(raw, "lc_transduction_v1")


def docqa():
    return {
        "prompt": [{"role": "user", "content": "An intact long document. Question?"}],
        "ability": "doc-mc",
        "data_source": "long_toc_choices_0_20000",
        "reward_model": {"ground_truth": "The correct answer is (C)."},
    }


def test_preparation_pool_and_runtime_controls(tmp_path):
    raw = docqa()
    entry, reasons = prepare_one(tmp_path, "lc_docqa_v1", [{"id": "real-row", "raw_record": raw}], Tokenizer())
    assert entry["counts"]["prepared_rows"] == 1
    row = json.loads((tmp_path / "out/source/prompts.jsonl").read_text())
    assert row["metadata"]["pool_id"] == "long_context"
    assert row["metadata"]["messages"] == raw["prompt"]
    target = row["label"]
    results = asyncio.run(
        check_control(row, prepare_long_context.control_cases(target), tmp_path / "out/source/verifiers.json")
    )
    assert [r["actual"] for r in results] == [1, 0, 0, 0, 0]


@pytest.mark.parametrize("code", sorted(adapters.DEFERRED_TASKS | {"T4.1", "T4.2"}))
def test_unsafe_native_routes_explicitly_deferred(code):
    raw = {
        "prompt": "Preserve",
        "data_source": "longbench_pro",
        "extra_info": {"reward_mode": code + " description"},
        "reward_model": {"ground_truth": {"doc_ids": ["a", "b", "c"]}},
    }
    with pytest.raises(ValueError, match="deferred_"):
        sources.adapt(raw, "lc_golong_v1")


@pytest.mark.parametrize("gold", ["0", "-1", "14.285714", "1e100"])
def test_numeric_controls_exceed_rounding_tolerance(gold):
    t = adapters.target("decimal", [gold], "therefore")
    cases = prepare_long_context.control_cases(t)
    assert cases[-1]["expected"] == 0
    assert rewards.score(cases[0]["response"], t) == 1


def test_golong_text_accuracy_label():
    raw = {
        "prompt": "Question",
        "data_source": "longbench_pro",
        "extra_info": {"reward_mode": "T3.1 Multi-Doc Integration QA"},
        "reward_model": {"ground_truth": {"doc_ids": ["unanswerable"]}},
    }
    _, t, _, _ = sources.adapt(raw, "lc_golong_v1")
    assert rewards.score("[Answer]\nUNANSWERABLE", t) == 1
    assert rewards.score("[Answer]\nunanswerable\nor Paris", t) == 0
