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

Held-out eval splits (lcbv5 test, codeforces) are tagged with source-specific "dataset" values
("code_stdio_lcbv5", "code_stdio_codeforces") instead of the generic "code_stdio" used by train
splits, so eval metrics report separately per source (see `resolve_reward_function` in
ground_truth_utils.py, which routes any "code_stdio*"-prefixed name to the "code_stdio" verifier).
"""

import json

from datasets import Dataset, load_dataset

import open_instruct.utils as open_instruct_utils

HUB_PREFIX = "mnoukhov/deepcoder"

INSTRUCTION = (
    "\n\nWrite Python code to solve this problem. Your program should read the input from stdin "
    "and write the output to stdout. Enclose your complete solution in a single ```python code block."
)

# DeepCoder's own recipe samples "the 15 most challenging tests" per problem for reward
# calculation (https://www.together.ai/blog/deepcoder). That cap alone isn't enough here, though:
# a handful of LiveCodeBench-v5 stress-test problems ship individual tests running several MB each
# (up to 160MB combined for one problem's 30 tests), and CodeStdioVerifier.async_call
# (open_instruct/ground_truth_utils.py) sends every test for an example in a single HTTP POST to
# the code-execution API -- an AWS API Gateway endpoint with a hard 10MB request limit (Lambda's
# sync-invoke limit is 6MB). A handful of oversized tests would fail at request time regardless of
# storage concerns, and unfiltered totals also overflow PyArrow's 2GB string-offset limit while
# caching the dataset. Greedily keep the largest ("most challenging") tests within a total byte
# budget with plenty of margin below the 6MB wall; problems with no test under the per-test cap end
# up with zero usable tests and are dropped (can't be graded by this infra either way).
MAX_TESTS_PER_PROBLEM = 15
MAX_TEST_BYTES = 100_000  # drop any single test case bigger than this
MAX_TOTAL_TEST_BYTES = 500_000  # total serialized size budget for the kept tests of one problem


def _test_size(test: dict) -> int:
    return len(test.get("input", "")) + len(test.get("output", ""))


def cap_tests(tests: list[dict]) -> list[dict]:
    candidates = sorted((t for t in tests if _test_size(t) <= MAX_TEST_BYTES), key=_test_size, reverse=True)
    kept = []
    total = 0
    for test in candidates:
        size = _test_size(test)
        if len(kept) >= MAX_TESTS_PER_PROBLEM or total + size > MAX_TOTAL_TEST_BYTES:
            break
        kept.append(test)
        total += size
    return kept


def to_example(problem: str, tests: list[dict], dataset_tag: str = "code_stdio") -> dict | None:
    tests = cap_tests(tests)
    if not tests:
        return None
    return {
        "messages": [{"role": "user", "content": problem.strip() + INSTRUCTION}],
        "ground_truth": json.dumps(tests),
        "dataset": dataset_tag,
    }


def convert_lcbv5(split: str) -> Dataset:
    # Eval-only split gets its own tag so wandb can report it separately from other eval sets;
    # `resolve_reward_function` (ground_truth_utils.py) still routes any "code_stdio*" prefix to
    # the "code_stdio" verifier, so reward computation is unaffected.
    dataset_tag = "code_stdio_lcbv5" if split == "test" else "code_stdio"
    dataset = load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset",
        "lcbv5",
        split=split,
        num_proc=open_instruct_utils.max_num_processes(),
    )
    examples = []
    dropped_no_usable_tests = 0
    for sample in dataset:
        tests = json.loads(sample["tests"]) if isinstance(sample["tests"], str) else sample["tests"]
        if any(test.get("testtype") != "stdin" for test in tests):
            continue
        example = to_example(sample["problem"], tests, dataset_tag=dataset_tag)
        if example is None:
            dropped_no_usable_tests += 1
            continue
        examples.append(example)
    if dropped_no_usable_tests:
        print(f"  lcbv5[{split}]: dropped {dropped_no_usable_tests} problems with no test under the size cap")
    return Dataset.from_list(examples)


def convert_primeintellect() -> Dataset:
    dataset = load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset",
        "primeintellect",
        split="train",
        num_proc=open_instruct_utils.max_num_processes(),
    )
    examples = []
    dropped_no_usable_tests = 0
    for sample in dataset:
        tests = json.loads(sample["tests"]) if isinstance(sample["tests"], str) else sample["tests"]
        if any(test.get("type") != "stdin_stdout" for test in tests):
            continue
        example = to_example(sample["problem"], tests)
        if example is None:
            dropped_no_usable_tests += 1
            continue
        examples.append(example)
    if dropped_no_usable_tests:
        print(f"  primeintellect: dropped {dropped_no_usable_tests} problems with no test under the size cap")
    return Dataset.from_list(examples)


def convert_taco() -> Dataset:
    dataset = load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset",
        "taco",
        split="train",
        num_proc=open_instruct_utils.max_num_processes(),
    )
    examples = []
    dropped_no_usable_tests = 0
    for sample in dataset:
        tests = json.loads(sample["tests"]) if isinstance(sample["tests"], str) else sample["tests"]
        if not isinstance(tests, dict) or tests.get("fn_name"):
            continue
        inputs, outputs = tests.get("inputs"), tests.get("outputs")
        if not inputs or not outputs or len(inputs) != len(outputs):
            continue
        pairs = [{"input": i, "output": o} for i, o in zip(inputs, outputs)]
        example = to_example(sample["problem"], pairs)
        if example is None:
            dropped_no_usable_tests += 1
            continue
        examples.append(example)
    if dropped_no_usable_tests:
        print(f"  taco: dropped {dropped_no_usable_tests} problems with no test under the size cap")
    return Dataset.from_list(examples)


def convert_codeforces() -> Dataset:
    dataset = load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset",
        "codeforces",
        split="test",
        num_proc=open_instruct_utils.max_num_processes(),
    )
    examples = []
    dropped_no_usable_tests = 0
    for sample in dataset:
        tests = json.loads(sample["tests"]) if isinstance(sample["tests"], str) else sample["tests"]
        example = to_example(sample["problem"], tests, dataset_tag="code_stdio_codeforces")
        if example is None:
            dropped_no_usable_tests += 1
            continue
        examples.append(example)
    if dropped_no_usable_tests:
        print(f"  codeforces: dropped {dropped_no_usable_tests} problems with no test under the size cap")
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
