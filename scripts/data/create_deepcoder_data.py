"""Convert agentica-org/DeepCoder-Preview-Dataset into open-instruct RLVR format.

The source dataset has four configs: codeforces, lcbv5, primeintellect, taco.
- codeforces only ships a "test" split -> used entirely as held-out eval.
- lcbv5 ships "train" and "test" -> lcbv5 test is reserved as held-out eval.
- primeintellect and taco only ship "train" -> used entirely for training.

Each config mixes stdin/stdout-style problems with function-signature ("functional") problems.
lcbv5 keeps both: stdin problems are graded by the code_stdio verifier and functional ones by the
code_functional verifier, which calls the solution's function directly (see
`get_successful_tests_functional` in open_instruct/code_utils/code_utils.py). This keeps the lcbv5
test split at the full 279 problems that DeepCoder/LiveCodeBench report on. primeintellect and taco
encode their functional problems differently (pre-parsed argument lists rather than LiveCodeBench's
JSON-per-line) and are still filtered to stdin-only.

Pushes one HF dataset per (source, split) to mnoukhov/deepcoder_<source>[_test]. The lcbv5 splits
are pushed under a "_full" suffix: the earlier stdin-only lcbv5 repos (mnoukhov/deepcoder_lcbv5,
mnoukhov/deepcoder_lcbv5_test) are left untouched so runs launched against them stay reproducible
and comparable.

Held-out eval splits (lcbv5 test, codeforces) are tagged with source-specific "dataset" values
("code_stdio_lcbv5", "code_functional_lcbv5", "code_stdio_codeforces") instead of the generic
"code_stdio"/"code_functional" used by train splits, so eval metrics report separately per source
(see `resolve_reward_function` in ground_truth_utils.py, which routes any "code_stdio*"- or
"code_functional*"-prefixed name to the matching verifier).
"""

import json

from datasets import Dataset, load_dataset

import open_instruct.utils as open_instruct_utils

HUB_PREFIX = "mnoukhov/deepcoder"

INSTRUCTION = (
    "\n\nWrite Python code to solve this problem. Your program should read the input from stdin "
    "and write the output to stdout. Enclose your complete solution in a single ```python code block."
)

# Functional problems are graded by calling the solution directly, so the model has to match the
# expected signature. LiveCodeBench's own prompt hands the model the starter code for this reason.
FUNCTIONAL_INSTRUCTION = (
    "\n\nWrite Python code to solve this problem. You will use the following starter code to write "
    "the solution, and enclose your complete solution in a single ```python code block."
    "\n\n```python\n{starter_code}\n```"
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


def to_example(
    problem: str,
    tests: list[dict],
    dataset_tag: str = "code_stdio",
    fn_name: str | None = None,
    starter_code: str = "",
) -> dict | None:
    tests = cap_tests(tests)
    if not tests:
        return None
    if fn_name is None:
        instruction = INSTRUCTION
    else:
        # `get_successful_tests_functional` reads fn_name off the tests, so stamp it on each one to
        # keep the whole problem inside the single `ground_truth` field.
        tests = [{**test, "fn_name": fn_name} for test in tests]
        instruction = FUNCTIONAL_INSTRUCTION.format(starter_code=starter_code.strip())
    return {
        "messages": [{"role": "user", "content": problem.strip() + instruction}],
        "ground_truth": json.dumps(tests),
        "dataset": dataset_tag,
    }


def convert_lcbv5(split: str) -> Dataset:
    # Eval-only split gets its own tag so wandb can report it separately from other eval sets;
    # `resolve_reward_function` (ground_truth_utils.py) still routes any "code_stdio*" /
    # "code_functional*" prefix to the matching verifier, so reward computation is unaffected.
    tag_suffix = "_lcbv5" if split == "test" else ""
    dataset = load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset",
        "lcbv5",
        split=split,
        num_proc=open_instruct_utils.max_num_processes(),
    )
    examples = []
    dropped_no_usable_tests = 0
    dropped_no_fn_name = 0
    dropped_mixed_testtype = 0
    for sample in dataset:
        tests = json.loads(sample["tests"]) if isinstance(sample["tests"], str) else sample["tests"]
        testtypes = {test.get("testtype") for test in tests}
        if testtypes == {"stdin"}:
            example = to_example(sample["problem"], tests, dataset_tag=f"code_stdio{tag_suffix}")
        elif testtypes == {"functional"}:
            metadata = sample["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            fn_name = (metadata or {}).get("func_name")
            if not fn_name:
                # Nothing to call, so the problem can't be graded.
                dropped_no_fn_name += 1
                continue
            example = to_example(
                sample["problem"],
                tests,
                dataset_tag=f"code_functional{tag_suffix}",
                fn_name=fn_name,
                starter_code=sample["starter_code"],
            )
        else:
            # A single example carries one `dataset` tag and therefore one verifier, so a problem
            # mixing both formats can't be graded. None exist in lcbv5 today.
            dropped_mixed_testtype += 1
            continue
        if example is None:
            dropped_no_usable_tests += 1
            continue
        examples.append(example)
    if dropped_no_usable_tests:
        print(f"  lcbv5[{split}]: dropped {dropped_no_usable_tests} problems with no test under the size cap")
    if dropped_no_fn_name:
        print(f"  lcbv5[{split}]: dropped {dropped_no_fn_name} functional problems with no func_name")
    if dropped_mixed_testtype:
        print(f"  lcbv5[{split}]: dropped {dropped_mixed_testtype} problems mixing stdin and functional tests")
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
    print(f"lcbv5 train (stdin + functional): {len(lcbv5_train)}")
    lcbv5_train.push_to_hub(f"{HUB_PREFIX}_lcbv5_full")

    lcbv5_test = convert_lcbv5("test")
    print(f"lcbv5 test (stdin + functional, eval): {len(lcbv5_test)}")
    lcbv5_test.push_to_hub(f"{HUB_PREFIX}_lcbv5_test_full")

    primeintellect_train = convert_primeintellect()
    print(f"primeintellect train (stdin-only): {len(primeintellect_train)}")
    primeintellect_train.push_to_hub(f"{HUB_PREFIX}_primeintellect")

    taco_train = convert_taco()
    print(f"taco train (stdin-only): {len(taco_train)}")
    taco_train.push_to_hub(f"{HUB_PREFIX}_taco")

    codeforces_test = convert_codeforces()
    print(f"codeforces test (eval): {len(codeforces_test)}")
    codeforces_test.push_to_hub(f"{HUB_PREFIX}_codeforces_test")
