"""Label parsing, scoring and the paired statistics of the GSM8K test evaluator."""

from scripts.miles import gsm8k_test_eval as ev


def test_label_parsing_strips_reasoning_and_commas():
    assert ev.parse_label("Some steps.\n#### 1,234") == "1234"
    assert ev.parse_label("#### -7") == "-7"


def test_mcnemar_exact_is_symmetric_and_bounded():
    assert ev.mcnemar_exact(0, 0) == 1.0
    assert ev.mcnemar_exact(5, 5) == 1.0
    assert abs(ev.mcnemar_exact(0, 8) - 2 / 256) < 1e-12
    assert ev.mcnemar_exact(3, 9) == ev.mcnemar_exact(9, 3)


def test_paired_counts_and_net_change():
    first = [{"id": i, "greedy": {"correct": c}} for i, c in enumerate([1, 1, 0, 0, 1])]
    second = [{"id": i, "greedy": {"correct": c}} for i, c in enumerate([1, 0, 1, 0, 1])]
    result = ev.paired(first, second)
    assert result["both_correct"] == 2 and result["neither"] == 1
    assert result["only_first"] == 1 and result["only_second"] == 1 and result["net_change"] == 0
