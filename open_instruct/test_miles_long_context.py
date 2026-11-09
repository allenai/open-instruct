"""Long-context contracts reject answer spraying and preserve source prompts."""

import asyncio
import copy
import json
import math
from types import SimpleNamespace

import pytest

from open_instruct.miles.datasets import long_context as adapters
from open_instruct.miles.datasets import pools
from open_instruct.miles.datasets import source_preparation as sources
from open_instruct.miles.rewards import long_context as rewards
from open_instruct.miles.rewards import rewards as runtime_rewards
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


@pytest.mark.parametrize(
    "response",
    [
        "C",
        "(C)",
        "(C),",
        "C)",
        "The correct answer is C) The option text.",
        ": C",
        "The correct answer is (C).",
        "The correct answer is: C",
        "Reasoning</think>C",
        "The correct answer is (C) The option's text.",
        "The correct answer is C\nExplanation follows.",
        "The correct answer is (C) 答案应是 (C).",
        "The correct answer is (C). (A) involves a different mechanism.",
        "The correct answer is (C) Vitamin (D) deficiency",
        "The correct answer is (C) (see paragraph (B))",
        "The correct answer is (C).\n(D) is incorrect.",
        "The correct answer is (C).\r\n(D) is incorrect.",
    ],
)
def test_choice(response):
    assert rewards.score(response, adapters.target("choice", ["C"], "choice")) == 1


@pytest.mark.parametrize(
    "response",
    [
        "**C**",
        "**(C)**",
        "The correct answer is **(C) The option text.**",
        "The correct answer is **C** The option text.",
        "**The correct answer is (C)**",
        "The correct answer is (**C**).",
        r"\boxed{C}",
        r"\boxed{(C)}",
        r"$\boxed{C}$",
        r"The correct answer is \(\boxed{C}\).",
        r"The correct answer is \boxed{C}。",
        r"The correct answer is \( (C) \).",
        "(c)",
        "The correct answer is (c).",
        "The correct answer is (C)。",
        "The correct answer is (C)仅对该企业采取措施。",
        "The correct answer is (C)51.8%.",
        "The correct answer is **（c）**",
    ],
)
def test_choice_presentation_preserves_letter(response):
    assert rewards.extract(response, "choice") == "C"
    assert rewards.score(response, adapters.target("choice", ["C"], "choice")) == 1
    assert rewards.score(response, adapters.target("choice", ["D"], "choice")) == 0


@pytest.mark.parametrize(
    "response",
    [
        "**CD**",
        r"\boxed{CD}",
        r"\boxed{C or D}",
        r"\boxed{C} \boxed{D}",
        "**C** or **D**",
        "The correct answer is **I think (C)**",
        "The correct answer is a possibility",
        "**(C) Option text without an answer prefix**",
        "The correct answer is **C**. The correct answer is (d).",
        r"The correct answer is \boxed{C}. The correct answer is **D**.",
    ],
)
def test_choice_formatting_does_not_extract_prose_or_conflicting_declarations(response):
    assert rewards.extract(response, "choice") == ""


@pytest.mark.parametrize("response", ["AC", "A B C D", "The correct answer is (C). The correct answer is (D)."])
def test_choice_not_substring(response):
    assert rewards.score(response, adapters.target("choice", ["C"], "choice")) == 0


@pytest.mark.parametrize("response", ["The correct answer is I think (B)", "The correct answer is A possibility"])
def test_choice_does_not_extract_prose_initial(response):
    assert rewards.extract(response, "choice") == ""


@pytest.mark.parametrize("response", ["(C", "(C) or (D)", ": C D", "The correct answer is C or D"])
def test_choice_rejects_malformed_or_ambiguous_answer(response):
    assert rewards.extract(response, "choice") == ""


@pytest.mark.parametrize(
    "response",
    [
        "The correct answer is (C) and (D)",
        "The correct answer is (C) or (D)",
        "The correct answer is (C), (D)",
        "The correct answer is C) and (D)",
        "The correct answer is C. Or (D)",
        "The correct answer is (C) and (C)",
    ],
)
def test_choice_suffix_ambiguity_is_a_known_limitation(response):
    target = adapters.target("choice", ["C"], "choice")
    assert rewards.extract(response, "choice") == "C"
    assert rewards.score(response, target) == 1


@pytest.mark.parametrize("metric", ["lines_f1", "ndcg"])
@pytest.mark.parametrize("response", ["doc1\n\ndoc2", "\n doc1\r\n \t\r\ndoc2 \n", "doc1\rdoc2"])
def test_list_metrics_ignore_blank_lines(metric, response):
    target = adapters.target(metric, ["doc1", "doc2"], "answer_marker")
    assert rewards.score("[Answer]\n" + response, target) == 1
    assert rewards.score("[Answer]\n \t\n", target) == 0


def test_ranked_duplicates_keep_first_occurrence():
    target = adapters.target("ndcg", ["first", "second", "third"], "answer_marker")
    response = "junk\nfirst\nsecond\nthird"
    expected = (3 / math.log2(3) + 2 / math.log2(4)) / (3 + 2 / math.log2(3) + 1 / math.log2(4))
    assert rewards.score(response, target) == pytest.approx(expected)
    for repeated in ["junk", "first", "second", "third", " JUNK "]:
        assert rewards.score(response + "\n" + repeated, target) == pytest.approx(expected)
    assert expected < 1


def test_golong_em_preserves_native_strict_contract():
    raw = {
        "prompt": "Return only the answer.",
        "data_source": "EM",
        "extra_info": {"reward_mode": "EM"},
        "reward_model": {"ground_truth": {"golden_label": "Paris"}},
    }
    _, target, _, _ = sources.adapt(raw, "lc_golong_v1")
    assert rewards.score("Paris", target) == 1
    for response in ["paris", "Paris.", "[Answer]\nParis", "Paris\nLondon"]:
        assert rewards.score(response, target) == 0


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
        ("unnumbered_var_expand", "a=cat\nb=dust storm\n\nExpressions:\n\na+b", "cat dust storm"),
        ("streaming_var_expand", "a=cat\nb=dog\n\nExpressions:\n\n[1]b+a", "[1]dog cat"),
        ("shuffled_streaming_var_expand", "a=cat\nb=dog\n\nExpressions:\n\n[1]b+a", "[1]dog cat"),
        ("csv_kv_lookup", "a1=cat\nv1=dust storm\n\nInput:\n\n,c1\nr1,a1+v1", ",c1\nr1,cat dust storm"),
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


@pytest.mark.parametrize("order", ["New row order: r1", "New col order: c1"])
def test_missing_permutation_order(order):
    raw = {
        "messages": [{"content": "Here is the real task:\nInput:\n\n,c1\nr1,A\n\nReordering Instructions:\n" + order}],
        "metadata": {"task_type": "csv_permutation_homogeneous"},
    }
    with pytest.raises(ValueError, match="Missing source permutation order"):
        adapters.transduction_reference(raw)


@pytest.mark.parametrize(
    "prompt,reason",
    [
        ("Input:\n\n1+2", "Missing source task delimiter"),
        ("Here is the real task:\n1+2", "Missing source input delimiter"),
    ],
)
def test_transduction_missing_delimiter_is_omitted_not_fatal(tmp_path, prompt, reason):
    valid = {
        "messages": [
            {"role": "user", "content": "Here is the real task:\nInput:\n\n1+2"},
            {"role": "assistant", "content": "1+2=3"},
        ],
        "metadata": {"task_type": "unnumbered_streaming_sum"},
    }
    malformed = copy.deepcopy(valid)
    malformed["messages"][0]["content"] = prompt
    entry, reasons = prepare_one(
        tmp_path,
        "lc_transduction_v1",
        [{"id": "malformed", "raw_record": malformed}, {"id": "valid", "raw_record": valid}],
    )
    assert entry["counts"]["input_rows"] == 2
    assert entry["counts"]["omitted_rows"] == 1
    assert entry["counts"]["prepared_rows"] == 1
    assert reasons == [f"ValueError: {reason}"]
    disposition = json.loads((tmp_path / "out/source/dispositions.jsonl").read_text())
    assert disposition["id"] == "malformed"
    row = json.loads((tmp_path / "out/source/prompts.jsonl").read_text())
    assert row["metadata"]["sample_id"] == "valid"


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
    assert "pool_id" not in row["metadata"]
    assert row["metadata"]["adapter"] == "lc_docqa_v1"
    assert row["metadata"]["subtask"] == raw["data_source"]
    assert row["metadata"]["messages"] == raw["prompt"]
    args = SimpleNamespace(
        reward_key=None,
        olmo_core=SimpleNamespace(
            reward_config=str(tmp_path / "out/source/verifiers.json"),
            reward_zero_truncated=True,
            reward_final_answer_only=False,
        ),
    )

    async def score():
        results = []
        for response, status in [
            ("The correct answer is (C).", "completed"),
            ("unrelated", "completed"),
            ("The correct answer is (B).", "completed"),
            ("", "completed"),
            ("The correct answer is (C).", "truncated"),
        ]:
            sample = SimpleNamespace(
                prompt=row["input"],
                metadata=copy.deepcopy(row["metadata"]),
                response=response,
                tokens=[],
                response_length=0,
                status=status,
            )
            results.append(await runtime_rewards.registered_reward(args, sample, failure_policy="raise"))
        return results

    assert asyncio.run(score()) == [1, 0, 0, 0, 0]
    row["pool_id"] = "document_tasks"
    pools.annotate(row, {"documents": {"label": "document_tasks"}})
    assert row["metadata"]["sampling_pool"] == "documents"


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


@pytest.mark.parametrize("adapter", ["lc_docqa_v1", "lc_golong_v1"])
def test_routed_adapter_provenance_without_pool_assignment(tmp_path, adapter):
    raw = (
        docqa()
        if adapter == "lc_docqa_v1"
        else {
            "prompt": "Compute 1+1",
            "data_source": "math_longcot_math_verify",
            "extra_info": {"reward_mode": "math"},
            "reward_model": {"ground_truth": {"golden_label": "2"}},
        }
    )
    raw["route"] = "chosen"
    raw_path = tmp_path / "input.jsonl"
    raw_path.write_text(json.dumps({"id": "row", "raw_record": raw}) + "\n")
    hygiene = tmp_path / "hygiene.json"
    hygiene.write_text('{"completed":true}')
    spec = {
        "adapter": "routed_v1",
        "options": {"field": "route", "routes": {"chosen": {"adapter": adapter}}},
        "inputs": [
            {"path": str(raw_path), "sha256": sources.file_hash(raw_path), "revision": "fixed", "split": "train"}
        ],
        "hygiene_receipt": {"path": str(hygiene), "sha256": sources.file_hash(hygiene)},
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema_version": 1, "sources": {"source": spec}}))
    result = sources.prepare(plan, tmp_path / "out", Tokenizer())
    assert result["sources"]["source"]["counts"]["prepared_rows"] == 1
    row = json.loads((tmp_path / "out/source/prompts.jsonl").read_text())
    assert row["metadata"]["adapter"] == adapter
    assert "pool_id" not in row["metadata"]
    row["pool_id"] = "document_tasks"
    pools.annotate(row, {"documents": {"label": "document_tasks"}})
    assert row["metadata"]["sampling_pool"] == "documents"
