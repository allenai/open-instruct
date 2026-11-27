# Dolci-Think-RL-7B stdio target overlay

This source-preparation step addresses the text wrappers in [#2200](https://github.com/allenai/open-instruct/issues/2200). From the repository root, with PyArrow and requests installed, run

```sh
PYTHONPATH=. python scripts/data/rlvr_code/dolci_2200/prepare.py /path/to/snapshot/data overlay.jsonl
```

Use the nine local shards from [Dolci revision 0fb6466](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B/tree/0fb6466d31ef3a9dd16985ef635e6429e05a6491/data). During preparation, join the overlay on custom_id, verify target_sha256 against the UTF-8 bytes of ground_truth[0], and replace the ground_truth column for matched rows. The overlay contains 136 replacement targets. It preserves the order and content of tests except for joining 146 expected-output lists with a newline. The published shards are read-only inputs.

## Evidence and quarantine

The manifest lists every rejected target. The 136 rows with wrapped_outputs greater than zero have matching ordered upstream suites and a stdio contract. Of their 146 wrapped outputs, 51 contain one line, 93 contain multiple lines, and two contain zero lines. The [PRIME stdio comparator](https://github.com/PRIME-RL/PRIME/blob/18ad596f08d487bb546d80d738d99ec697bd2e75/training/verl/utils/reward_score/evaluation_utils/code_util/testing_util.py#L353) joins output lines, including empty lists. Four single-line Klear wrappers also match the original problem examples.

The 504 rows with a function_name are the quarantine list. The names come from structured metadata joined by source ID and checked against each ordered suite. The source revisions are [PrimeIntellect SYNTHETIC-2](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-Base-Code/tree/42191f9f32be3a724e16ae6a5b4f0ee34fceb710) and [Kwai-Klear CodeSub](https://huggingface.co/datasets/Kwai-Klear/KlearReasoner-CodeSub-15K/tree/92b67015a1e84cb596dadcdc09462006c83c8a9a). Keep these 504 IDs out of stdio grading. Count them explicitly when filtering a training mix.

Before adding a function route, choose and version its return-value comparison rule. Preserve the original arguments and stored expectations until then. A first fixture should cover prime_rl_code_10165 (array_diff), whose expected value for the first test is stored as [[2]]. Test candidate returns [2], [[2]], and an incorrect [1] independently. A second fixture should check the structured name get_exponent for prime_rl_code_10122. These fixtures can use literal values and require no solution execution. The [reference comparator](https://github.com/PRIME-RL/PRIME/blob/18ad596f08d487bb546d80d738d99ec697bd2e75/training/verl/utils/reward_score/evaluation_utils/code_util/testing_util.py#L279) can accept both direct and first-element matches.

Static validation on 102,014 rows accounted for all 958,494 code tests. The overlay raises accepted stdio rows from 8,624 to 8,760. All 12,121 assert rows retain their original targets. Runtime grading equivalence remains unmeasured.
