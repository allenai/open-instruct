import json

import pytest

from open_instruct import ifeval_constraints


def bundle(instruction_id="count:lowercase_counting", **kwargs):
    return {"instruction_id": [instruction_id], "kwargs": [kwargs]}


@pytest.mark.parametrize(
    "wrap",
    [
        lambda x: x,
        lambda x: [x],
        json.dumps,
        repr,
        lambda x: repr([x]),
        lambda x: [json.dumps(x)],
        lambda x: repr([json.dumps(x)]),
        lambda x: json.dumps([json.dumps([x])]),
    ],
)
def test_decode_supported_wrappers(wrap):
    value = bundle(N=5)
    assert ifeval_constraints.parse_constraint_bundle(wrap(value)) == value


def test_parse_none_kwargs_and_copy():
    value = {"instruction_id": ["unknown:valid"], "kwargs": [None], "extra": "preserved"}
    assert ifeval_constraints.parse_constraint_bundle(value) == {**value, "kwargs": [{}]}
    assert value["kwargs"] == [None]


@pytest.mark.parametrize(
    "target, message",
    [
        ("{", "encode a constraint dictionary"),
        (None, "must be a dictionary"),
        (3, "must be a dictionary"),
        ([], "exactly one"),
        ([{}, {}], "exactly one"),
        ({}, "instruction_id"),
        ({"instruction_id": [], "kwargs": []}, "instruction_id"),
        ({"instruction_id": ["x"], "kwargs": []}, "kwargs"),
        ({"instruction_id": ["x", "y"], "kwargs": [{}]}, "lengths differ"),
        ({"instruction_id": [""], "kwargs": [{}]}, r"instruction_id\[0\]"),
        ({"instruction_id": ["   "], "kwargs": [{}]}, r"instruction_id\[0\]"),
        ({"instruction_id": [7], "kwargs": [{}]}, r"instruction_id\[0\]"),
        ({"instruction_id": ["x"], "kwargs": ["not kwargs"]}, r"kwargs\[0\]"),
    ],
)
def test_malformed_bundle_context(target, message):
    with pytest.raises(ValueError, match=message):
        ifeval_constraints.parse_constraint_bundle(target)
    assert ifeval_constraints.if_training_issues(target) == ["malformed_target"]


def test_wrapper_depth_is_bounded():
    value = bundle(N=4)
    for _ in range(40):
        value = [value]
    with pytest.raises(ValueError, match="nested encoding"):
        ifeval_constraints.parse_constraint_bundle(value)


@pytest.mark.parametrize("n", [None, False, True, 0, -1, 2.0, "2", [], {}])
@pytest.mark.parametrize(
    "instruction_id, prefix", [("count:lowercase_counting", "lowercase"), ("letters:letter_counting", "letter")]
)
def test_invalid_n(n, instruction_id, prefix):
    assert ifeval_constraints.if_training_issues(bundle(instruction_id, N=n, relation="at least")) == [
        f"{prefix}_counting_invalid_n"
    ]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_tiny_lowercase_n(n):
    assert ifeval_constraints.if_training_issues(bundle(N=n)) == ["lowercase_counting_tiny_n"]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_tiny_letter_less_than(n):
    assert ifeval_constraints.if_training_issues(bundle("letters:letter_counting", N=n, relation="less than")) == [
        "letter_counting_tiny_less_than"
    ]
    assert ifeval_constraints.if_training_issues(bundle("letters:letter_counting", N=n, relation="at least")) == []


@pytest.mark.parametrize("relation", [None, "", "exactly", "Less than", 3, [], {}])
def test_invalid_relation(relation):
    assert ifeval_constraints.if_training_issues(bundle("letters:letter_counting", N=8, relation=relation)) == [
        "letter_counting_invalid_relation"
    ]


def test_missing_count_args_and_deduplication():
    assert ifeval_constraints.if_training_issues(bundle()) == ["lowercase_counting_invalid_n"]
    assert ifeval_constraints.if_training_issues(bundle("letters:letter_counting")) == [
        "letter_counting_invalid_n",
        "letter_counting_invalid_relation",
    ]
    target = {"instruction_id": ["count:lowercase_counting"] * 2, "kwargs": [{"N": 2}, {"N": 3}]}
    assert ifeval_constraints.if_training_issues(target) == ["lowercase_counting_tiny_n"]


@pytest.mark.parametrize("n", [4, 5, 100])
def test_larger_counts_are_kept(n):
    assert ifeval_constraints.if_training_issues(bundle(N=n)) == []
    assert ifeval_constraints.if_training_issues(bundle("letters:letter_counting", N=n, relation="less than")) == []


@pytest.mark.parametrize("field", ["num_paragraphs", "nth_paragraph"])
@pytest.mark.parametrize("value", [None, False, True, -1, 0, "2", 2.0])
def test_invalid_paragraph_indices(field, value):
    args = {"num_paragraphs": 2, "nth_paragraph": 1, "first_word": "bonjour", field: value}
    assert ifeval_constraints.if_training_issues(bundle("length_constraints:nth_paragraph_first_word", **args)) == [
        f"nth_paragraph_first_word_invalid_{field}"
    ]


@pytest.mark.parametrize(
    "first_word, reason", [(None, "missing"), (3, "invalid"), ("", "invalid"), ("   ", "invalid")]
)
def test_invalid_paragraph_first_word(first_word, reason):
    assert ifeval_constraints.if_training_issues(
        bundle("length_constraints:nth_paragraph_first_word", num_paragraphs=2, nth_paragraph=1, first_word=first_word)
    ) == [f"nth_paragraph_first_word_{reason}_first_word"]


def test_out_of_bounds_paragraph_and_valid_final_paragraph():
    target = bundle(
        "length_constraints:nth_paragraph_first_word", num_paragraphs=2, nth_paragraph=3, first_word="世界"
    )
    assert ifeval_constraints.if_training_issues(target) == ["nth_paragraph_first_word_nth_exceeds_num_paragraphs"]
    target["kwargs"][0]["nth_paragraph"] = 2
    assert ifeval_constraints.if_training_issues(target) == []


def test_explicit_multilingual_and_composition_preserved():
    target = {
        "instruction_id": ["language:response_language", "count:counting_composition", "keywords:existence"],
        "kwargs": [{"language": "zh"}, {"n_sent": 2, "n_words": 4}, {"keywords": ["世界", "bonjour"]}],
    }
    assert ifeval_constraints.if_training_issues(target) == []
