"""Convert agentica-org/DeepCoder-Preview-Dataset into open-instruct RLVR format.

The source dataset has four configs: codeforces, lcbv5, primeintellect, taco.
- codeforces only ships a "test" split -> used entirely as held-out eval.
- lcbv5 ships "train" and "test" -> lcbv5 test is reserved as held-out eval.
- primeintellect and taco only ship "train" -> used entirely for training.

Each config mixes stdin/stdout-style problems with function-signature
("functional") problems. open-instruct's code_stdio verifier only executes
programs against stdin/stdout test pairs, so functional problems (identified
via lcbv5's testtype, primeintellect's per-test type, and taco's fn_name) are
dropped.

Pushes one HF dataset per (source, split) to mnoukhov/deepcoder_<source>[_test].
"""

import json

from datasets import Dataset, load_dataset

import open_instruct.utils as open_instruct_utils

HUB_PREFIX = "mnoukhov/deepcoder"

INSTRUCTION = (
    "\n\nWrite Python code to solve this problem. Your program should read the input from stdin "
    "and write the output to stdout. Enclose your complete solution in a single ```python code block."
)


def to_example(problem: str, tests: list[dict]) -> dict:
    return {
        "messages": [{"role": "user", "content": problem.strip() + INSTRUCTION}],
        "ground_truth": json.dumps(tests),
        "dataset": "code_stdio",
    }


def convert_lcbv5(split: str) -> Dataset:
    dataset = load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset",
        "lcbv5",
        split=split,
        num_proc=open_instruct_utils.max_num_processes(),
    )
    examples = []
    for sample in dataset:
        tests = json.loads(sample["tests"]) if isinstance(sample["tests"], str) else sample["tests"]
        if any(test.get("testtype") != "stdin" for test in tests):
            continue
        examples.append(to_example(sample["problem"], tests))
    return Dataset.from_list(examples)


def convert_primeintellect() -> Dataset:
    dataset = load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset",
        "primeintellect",
        split="train",
        num_proc=open_instruct_utils.max_num_processes(),
    )
    examples = []
    for sample in dataset:
        tests = json.loads(sample["tests"]) if isinstance(sample["tests"], str) else sample["tests"]
        if any(test.get("type") != "stdin_stdout" for test in tests):
            continue
        examples.append(to_example(sample["problem"], tests))
    return Dataset.from_list(examples)


def convert_taco() -> Dataset:
    dataset = load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset",
        "taco",
        split="train",
        num_proc=open_instruct_utils.max_num_processes(),
    )
    examples = []
    for sample in dataset:
        tests = json.loads(sample["tests"]) if isinstance(sample["tests"], str) else sample["tests"]
        if not isinstance(tests, dict) or tests.get("fn_name"):
            continue
        inputs, outputs = tests.get("inputs"), tests.get("outputs")
        if not inputs or not outputs or len(inputs) != len(outputs):
            continue
        pairs = [{"input": i, "output": o} for i, o in zip(inputs, outputs)]
        examples.append(to_example(sample["problem"], pairs))
    return Dataset.from_list(examples)


def convert_codeforces() -> Dataset:
    dataset = load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset",
        "codeforces",
        split="test",
        num_proc=open_instruct_utils.max_num_processes(),
    )
    examples = []
    for sample in dataset:
        tests = json.loads(sample["tests"]) if isinstance(sample["tests"], str) else sample["tests"]
        examples.append(to_example(sample["problem"], tests))
    return Dataset.from_list(examples)


if __name__ == "__main__":
    lcbv5_train = convert_lcbv5("train")
    print(f"lcbv5 train (stdin-only): {len(lcbv5_train)}")
    lcbv5_train.push_to_hub(f"{HUB_PREFIX}_lcbv5")

    lcbv5_test = convert_lcbv5("test")
    print(f"lcbv5 test (stdin-only, eval): {len(lcbv5_test)}")
    lcbv5_test.push_to_hub(f"{HUB_PREFIX}_lcbv5_test")

    primeintellect_train = convert_primeintellect()
    print(f"primeintellect train (stdin-only): {len(primeintellect_train)}")
    primeintellect_train.push_to_hub(f"{HUB_PREFIX}_primeintellect")

    taco_train = convert_taco()
    print(f"taco train (stdin-only): {len(taco_train)}")
    taco_train.push_to_hub(f"{HUB_PREFIX}_taco")

    codeforces_test = convert_codeforces()
    print(f"codeforces test (eval): {len(codeforces_test)}")
    codeforces_test.push_to_hub(f"{HUB_PREFIX}_codeforces_test")
