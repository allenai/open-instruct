"""Convert agentica-org/DeepCoder-Preview-Dataset into open-instruct RLVR format.

The source dataset has four configs: codeforces, lcbv5, primeintellect, taco.
- codeforces only ships a "test" split -> used entirely as held-out eval.
- lcbv5 ships "train" and "test" -> lcbv5 test is reserved as held-out eval.
- primeintellect and taco only ship "train" -> used entirely for training.

Each config mixes stdin/stdout-style problems with function-signature ("functional") problems, all
now kept: stdin problems are graded by the code_stdio verifier and functional ones by the
code_functional verifier, which calls the solution's function directly (see
`get_successful_tests_functional` in open_instruct/code_utils/code_utils.py). lcbv5's functional
tests already encode one JSON argument per line and a bare JSON-encoded return value, matching
LiveCodeBench's own format, and ship a `starter_code` field so the prompt can hand the model the
exact signature (as LiveCodeBench does). primeintellect and taco instead ship pre-parsed argument
lists and wrap the expected return in a single-element list (not always consistently, so
`grade_call_based` accepts the value either wrapped or bare -- see testing_util.py); they also
don't ship starter code, so a signature stub is synthesized for them.

The prompt is DeepCoder's, not ours: `fetch_live_code_bench_system_prompt` below is a verbatim port
of the function rllm's examples/deepcoder/prepare_deepcoder_data.py uses, which is in turn
LiveCodeBench's own generation prompt. Reproducing it exactly matters because the reference 1.5B
run we compare against was trained on it -- in particular its "do not directly test on the sample
inputs" clause, which suppresses the self-check prints that otherwise fail exact-stdout grading.

Pushes one HF dataset per (source, split) to mnoukhov/deepcoder_<source>[_test]<HUB_SUFFIX>. Each
prompt change gets a new suffix rather than overwriting, so runs launched against earlier repos
stay reproducible and comparable: the stdin-only repos have no suffix, "_full" added the functional
problems back in under a hand-written prompt, and "_lcb" is "_full" restated in DeepCoder's prompt.

Held-out eval splits (lcbv5 test, codeforces) are tagged with source-specific "dataset" values
("code_stdio_lcbv5", "code_functional_lcbv5", "code_stdio_codeforces") instead of the generic
"code_stdio"/"code_functional" used by train splits, so eval metrics report separately per source
(see `resolve_reward_function` in ground_truth_utils.py, which routes any "code_stdio*"- or
"code_functional*"-prefixed name to the matching verifier).
"""

import json

from datasets import Dataset, load_dataset

import open_instruct.utils as open_instruct_utils
from open_instruct.code_utils.code_utils import encode_tests

HUB_PREFIX = "mnoukhov/deepcoder"

# Suffix for the repos this script pushes to. The prompt is part of the dataset, so changing it
# means a new set of repos rather than an overwrite: runs launched against the previous suffix stay
# reproducible and comparable. "_lcb" == the LiveCodeBench/DeepCoder prompt below; the earlier
# "_full" repos carry the hand-written paraphrase this replaced.
HUB_SUFFIX = "_lcb"

# Verbatim from rllm (DeepCoder's trainer), which is itself a port of LiveCodeBench's generation
# prompt -- rllm/system_prompts.py and `fetch_live_code_bench_system_prompt` in rllm/data/utils.py,
# used by examples/deepcoder/prepare_deepcoder_data.py. Reproduced exactly, whitespace included, so
# our rollouts see the same prompt DeepCoder's reported 1.5B run saw.
# https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
LCB_SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question (problem specification) and "
    "will generate a correct Python program that matches the specification and passes all tests."
)

LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem and enclose your "
    "code within delimiters."
)

LCB_FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly "
    "test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when "
    "the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def fetch_live_code_bench_system_prompt(prompt: str, starter_code: str | None = None) -> str:
    """Port of rllm's `fetch_live_code_bench_system_prompt`, kept line-for-line identical.

    Note `prompt` is deliberately *not* stripped: rllm concatenates the problem straight onto
    "### Format:", and DeepCoder-Preview-Dataset problems are inconsistent about a trailing newline,
    so some prompts run the two together. That is what DeepCoder trains on, so we keep it.
    """
    prompt = LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + prompt
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


# Which tests a problem is graded on, and how they are serialized into `ground_truth`.
#
# rllm (DeepCoder's trainer) does NOT use one policy here, it uses two, chosen by data_source
# (rllm/rewards/code_reward.py at aba20b1429, the commit behind their 1.5B 32K run):
#   - taco / apps / code_contests / primeintellect -> check_correctness(max_tests=15), keeping the
#     15 longest-input tests. This is the "15 most challenging tests" from
#     https://www.together.ai/blog/deepcoder.
#   - livecodebench / codeforces -> lcb_check_correctness_v2, which caps nothing and runs EVERY test.
# So the published LiveCodeBench numbers are all-tests numbers, and a 15-test cap on the lcbv5 eval
# set makes our score incomparable (a solution only has to pass the 15 largest tests, not all ~42).
#
# Keeping every test is not unconditionally affordable: CodeVerifier.async_call
# (open_instruct/ground_truth_utils.py) posts a problem's whole test set to the code API on every
# rollout, and the lcbv5 test split holds 3.7GB of tests, one problem alone accounting for 189MB.
# So keep all tests whenever the compressed payload fits FULL_TEST_BUDGET and fall back to the
# 15-test cap only for the heavy tail. At 10MB that is 238/279 eval problems (90% of all tests)
# graded exactly as LiveCodeBench does, for a 249MB dataset and a 10MB worst-case request.
#
# Test sets kept in full are stored with `encode_tests` (zlib+base64) rather than plain JSON;
# `decode_tests` on the API side accepts either. Plain JSON at this size would also overflow
# PyArrow's 2GB string-offset limit while caching the dataset.
MAX_TESTS_PER_PROBLEM = 15
MAX_TEST_BYTES = 100_000  # drop any single test case bigger than this
MAX_TOTAL_TEST_BYTES = 500_000  # total serialized size budget for the kept tests of one problem
FULL_TEST_BUDGET = 10_000_000  # keep every test when its compressed payload fits in this


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


def select_tests(tests: list[dict], full_test_budget: int) -> tuple[list[dict], str]:
    """Pick the tests to grade on and serialize them for the `ground_truth` field.

    Returns (tests, serialized). With `full_test_budget=0` this is the old capped behaviour.
    """
    if full_test_budget:
        encoded = encode_tests(tests)
        if len(encoded) <= full_test_budget:
            return tests, encoded
    capped = cap_tests(tests)
    return capped, json.dumps(capped)


def to_example(
    problem: str,
    tests: list[dict],
    dataset_tag: str = "code_stdio",
    fn_name: str | None = None,
    starter_code: str = "",
    num_args: int | None = None,
    full_test_budget: int = 0,
) -> dict | None:
    if fn_name is not None:
        # `get_successful_tests_functional` reads fn_name off the tests, so stamp it on each one to
        # keep the whole problem inside the single `ground_truth` field.
        tests = [{**test, "fn_name": fn_name} for test in tests]
        if not starter_code:
            # primeintellect/taco functional problems ship no starter code, a case rllm never hits
            # (its DeepCoder prep is lcbv5-only, where every functional problem has it). Synthesize
            # the signature into a stub so these still go through LiveCodeBench's starter-code
            # branch rather than needing a prompt shape DeepCoder never trained on. They're
            # overwhelmingly bare functions, not `class Solution` methods -- verified against the
            # primeintellect sources.
            arg_placeholders = ", ".join(f"arg{i}" for i in range(num_args)) if num_args else ""
            starter_code = f"def {fn_name}({arg_placeholders}):\n    "
    kept, ground_truth = select_tests(tests, full_test_budget)
    if not kept:
        return None
    return {
        "messages": [{"role": "user", "content": fetch_live_code_bench_system_prompt(problem, starter_code or None)}],
        "ground_truth": ground_truth,
        "dataset": dataset_tag,
    }


def to_functional_test(args: list, output: object) -> dict:
    """Re-encode a primeintellect/taco call-based test into the same string encoding lcbv5's
    functional tests use (see `grade_call_based` in testing_util.py): one JSON-encoded argument per
    line, and the JSON-encoded expected return value. The return value is left wrapped if the source
    wrapped it (not all of them consistently do) -- `grade_call_based` accepts it either way.
    """
    return {"input": "\n".join(json.dumps(arg) for arg in args), "output": json.dumps(output)}


def convert_lcbv5(split: str) -> Dataset:
    # Eval-only split gets its own tag so wandb can report it separately from other eval sets;
    # `resolve_reward_function` (ground_truth_utils.py) still routes any "code_stdio*" /
    # "code_functional*" prefix to the matching verifier, so reward computation is unaffected.
    tag_suffix = "_lcbv5" if split == "test" else ""
    # Uncapped tests are reserved for the eval split, where fidelity to LiveCodeBench's number is
    # the whole point. The train split stays capped: uncapped it reaches 103 tests on one problem,
    # and at 6s/test a looping rollout would occupy a grading worker for 10 minutes.
    full_test_budget = FULL_TEST_BUDGET if split == "test" else 0
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
            example = to_example(
                sample["problem"], tests, dataset_tag=f"code_stdio{tag_suffix}", full_test_budget=full_test_budget
            )
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
                full_test_budget=full_test_budget,
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
    dropped_no_fn_name = 0
    dropped_mixed_type = 0
    for sample in dataset:
        tests = json.loads(sample["tests"]) if isinstance(sample["tests"], str) else sample["tests"]
        types = {test.get("type") for test in tests}
        if types == {"stdin_stdout"}:
            example = to_example(sample["problem"], tests)
        elif types == {"function_call"}:
            fn_name = next((test.get("fn_name") for test in tests if test.get("fn_name")), None)
            if not fn_name:
                dropped_no_fn_name += 1
                continue
            functional_tests = [to_functional_test(test["input"], test["output"]) for test in tests]
            example = to_example(
                sample["problem"],
                functional_tests,
                dataset_tag="code_functional",
                fn_name=fn_name,
                num_args=len(tests[0]["input"]),
            )
        else:
            # A single example carries one `dataset` tag and therefore one verifier. None exist in
            # primeintellect today.
            dropped_mixed_type += 1
            continue
        if example is None:
            dropped_no_usable_tests += 1
            continue
        examples.append(example)
    if dropped_no_usable_tests:
        print(f"  primeintellect: dropped {dropped_no_usable_tests} problems with no test under the size cap")
    if dropped_no_fn_name:
        print(f"  primeintellect: dropped {dropped_no_fn_name} functional problems with no fn_name")
    if dropped_mixed_type:
        print(f"  primeintellect: dropped {dropped_mixed_type} problems mixing stdin and functional tests")
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
        if not isinstance(tests, dict):
            continue
        inputs, outputs = tests.get("inputs"), tests.get("outputs")
        if not inputs or not outputs or len(inputs) != len(outputs):
            continue
        fn_name = tests.get("fn_name")
        if fn_name:
            functional_tests = [to_functional_test(i, o) for i, o in zip(inputs, outputs)]
            example = to_example(
                sample["problem"],
                functional_tests,
                dataset_tag="code_functional",
                fn_name=fn_name,
                num_args=len(inputs[0]),
            )
        else:
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
    lcbv5_train.push_to_hub(f"{HUB_PREFIX}_lcbv5{HUB_SUFFIX}")

    lcbv5_test = convert_lcbv5("test")
    print(f"lcbv5 test (stdin + functional, eval): {len(lcbv5_test)}")
    lcbv5_test.push_to_hub(f"{HUB_PREFIX}_lcbv5_test{HUB_SUFFIX}")

    primeintellect_train = convert_primeintellect()
    print(f"primeintellect train (stdin + functional): {len(primeintellect_train)}")
    primeintellect_train.push_to_hub(f"{HUB_PREFIX}_primeintellect{HUB_SUFFIX}")

    taco_train = convert_taco()
    print(f"taco train (stdin + functional): {len(taco_train)}")
    taco_train.push_to_hub(f"{HUB_PREFIX}_taco{HUB_SUFFIX}")

    codeforces_test = convert_codeforces()
    print(f"codeforces test (eval): {len(codeforces_test)}")
    codeforces_test.push_to_hub(f"{HUB_PREFIX}_codeforces_test{HUB_SUFFIX}")
