"""Convert evalplus/humanevalplus into open-instruct RLVR format.

Unlike DeepCoder's stdin/stdout problems, HumanEval(+) problems are function-signature
("functional") style: the model must define a function matching a given signature, which is then
called directly by a test harness. These are graded by open-instruct's base `code` verifier
(POST /test_program, exec'd assert-style tests), not `code_stdio`. Held out entirely as an
eval-only set -- there is no train split.

Tagged with the source-specific "dataset" value "code_humanevalplus" (not the generic "code") so
eval metrics report separately from other eval sets sharing the "code" verifier. See
`resolve_reward_function` in ground_truth_utils.py, which routes any "code*"-prefixed name
(other than "code_stdio*") to the "code" verifier.

Pushes mnoukhov/humanevalplus_test to the Hub.
"""

import json

from datasets import Dataset, load_dataset

HUB_PREFIX = "mnoukhov/humanevalplus"

INSTRUCTION = (
    "\n\nComplete the Python function above. Your response must include the complete function "
    "definition, matching the given signature exactly. Enclose your complete solution in a "
    "single ```python code block."
)


def to_example(sample: dict) -> dict:
    check_program = f"{sample['test']}\ncheck({sample['entry_point']})\n"
    return {
        "messages": [{"role": "user", "content": sample["prompt"].strip() + INSTRUCTION}],
        "ground_truth": json.dumps([check_program]),
        "dataset": "code_humanevalplus",
    }


def convert_humanevalplus() -> Dataset:
    dataset = load_dataset("evalplus/humanevalplus", split="test")
    examples = [to_example(sample) for sample in dataset]
    return Dataset.from_list(examples)


if __name__ == "__main__":
    humanevalplus_test = convert_humanevalplus()
    print(f"humanevalplus test (eval): {len(humanevalplus_test)}")
    humanevalplus_test.push_to_hub(f"{HUB_PREFIX}_test")
