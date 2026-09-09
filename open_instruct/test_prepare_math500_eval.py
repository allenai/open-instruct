from scripts.data.rlvr.prepare_math500_eval import format_math500_row


def test_format_math500_row_builds_rlvr_math_example():
    row = {
        "problem": "Compute $1+1$.",
        "solution": "The result is $\\boxed{2}$.",
        "answer": "2",
        "subject": "Algebra",
        "level": 1,
        "unique_id": "test/algebra/1.json",
    }

    assert format_math500_row(row, dataset_label="math_500") == {
        "messages": [{"role": "user", "content": "Compute $1+1$."}],
        "ground_truth": "2",
        "dataset": "math_500",
        "source_dataset": "HuggingFaceH4/MATH-500",
        "source_id": "test/algebra/1.json",
        "subject": "Algebra",
        "level": 1,
    }
