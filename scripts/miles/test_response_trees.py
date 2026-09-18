import math

import pytest
from scripts.miles import response_trees as trees


def sample(i, text, score):
    return {"sample": i, "response": text, "score": score}


def terminal_ids(tree):
    return tree["terminal"] + [i for child in tree["children"] for i in terminal_ids(child)]


def test_identical_prefix_preserves_every_sample_without_artificial_information():
    samples = [sample(i, f"same prefix continuation{i}", i % 2) for i in range(8)]
    tree = trees.build_tree(samples, depth=2)
    assert tree["label"] == "same prefix"
    assert tree["n"] == 8 and tree["passed"] == 4
    assert tree["terminal"] == list(range(8))
    assert tree["h"] == tree["ig"] == 0


def test_ending_at_an_internal_node_is_a_real_branch():
    tree = trees.build_tree([sample(0, "shared", 0), sample(1, "shared continues", 1)])
    assert tree["label"] == "shared"
    assert tree["terminal"] == [0]
    assert tree["h"] == tree["ig"] == 1
    assert sorted(terminal_ids(tree)) == [0, 1]


def test_diverse_paths_with_constant_verdict_have_no_verdict_information():
    tree = trees.build_tree([sample(i, f"word{i}", 1) for i in range(8)])
    assert tree["h"] == 3
    assert tree["ig"] == 0


def test_empirical_verdict_association_matches_weighted_conditional_entropy():
    samples = [sample(0, "a one", 1), sample(1, "a two", 0), sample(2, "b", 1), sample(3, "b", 1)]
    tree = trees.build_tree(samples)
    expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)) - 0.5
    assert tree["ig"] == pytest.approx(expected)
    assert trees.build_tree(list(reversed(samples))) == tree
    assert sorted(terminal_ids(tree)) == list(range(4))


def test_missing_extracted_answers_are_not_lost():
    samples = [sample(0, "thinking only", 0), sample(1, "full response", 1)]
    samples[1]["answer"] = "answer"
    tree = trees.build_tree(samples, field="answer")
    assert tree["terminal"] == [0]
    assert tree["h"] == tree["ig"] == 1
    assert sorted(terminal_ids(tree)) == [0, 1]


def test_pair_rejects_missing_duplicate_or_changed_prompt_contracts():
    before = [{**sample(i, "text", 1), "prompt_hash": "hash", "cap": 8192} for i in range(8)]
    trees.validate_pair(before, before, 8)
    for after in (before[:-1], [before[0]] * 8, [{**s, "prompt_hash": "different"} for s in before]):
        with pytest.raises(ValueError):
            trees.validate_pair(before, after, 8)


def test_legacy_adapter_preserves_exact_grader_verdict():
    rows = [{"id": "gsm8k:test:1279", "sampled": [{"text": "180.00", "correct": False, "tokens": 4}]}]
    normalized = trees.samples_for(rows, "gsm8k:test:1279")
    assert normalized[0]["response"] == "180.00"
    assert normalized[0]["score"] == 0
    assert normalized[0]["sample"] == 0


def test_all_empty_final_answers_remain_an_available_view():
    rows = [{"id": "x", **sample(0, "unfinished", 0), "answer": "", "tokens": 8192}]
    normalized = trees.samples_for(rows, "x")
    assert "answer" in normalized[0]["available_views"]
    assert "code" not in normalized[0]["available_views"]


def test_nonbinary_scores_are_rejected_instead_of_thresholded():
    with pytest.raises(ValueError, match="Nonbinary"):
        trees.samples_for([{"id": "x", **sample(0, "text", 0.5)}], "x")
