# ToolRL offline function-calling data

This integration prepares the external [ToolRL](https://github.com/qiancheng0/ToolRL)
4K RL dataset and grades generated function calls locally. It does not execute
tools or require search credentials. The supported training entry point is MILES;
the legacy Core/vLLM and DeepSpeed/vLLM verifier registry is not wired to this
grader. No training run or BFCL improvement is implied by the CPU integration.

## Source and format

The converter pins upstream commit `8cee13ec0ca72f0461da372a93a6fd8140dbb840`
and verifies the SHA256 of `dataset/rlla_4k_raw/rlla_rl.json`. Upstream describes
the mixture as 2K ToolACE, 1K Hammer (Masked), and 1K xLAM examples. The raw file
does not identify those component sources per row; we retain its original row
index instead of inferring component labels.

| Input field | Prepared use |
|---|---|
| `instruction` | System message, including tool descriptions and response-format instructions |
| `input` | User message containing the recorded dialogue, including any previous tool observations |
| `output` | Reference target in `ground_truth` and verifier metadata; never appended to the prompt |

The existing dialogue is retained verbatim as text. A training attempt generates
the next response only; it does not interact with a live multi-turn environment.
The source uses JSON calls inside `<tool_call>` and reasoning inside `<think>`.
It is not converted to Olmo's native `<function=...>` XML call syntax. Retaining
this source format is useful for a baseline, but format adaptation and GPU
qualification are separate work before the matched atomic-token experiment.

| Reference calls per row | Rows |
|---|---:|
| None | 482 |
| One | 1,823 |
| Two or more, structurally valid | 1,694 |
| Rejected malformed reference | 1 |
| Source total | 4,000 |

## Prepare and select

From the repository root:

```bash
python -m scripts.data.rlvr.toolrl --output-dir runs/toolrl-data
```

Use `--input /path/to/rlla_rl.json` for a previously downloaded copy; its hash
must still match the pinned source. The converter refuses to overwrite an
existing output directory and writes `train.jsonl`, `eval.jsonl`, and
`rl-manifest.json`. It performs no upload or training launch.

The default requests 80 held-out rows with seed 31415. Identical dialogue inputs
stay together, even if their tool menus differ. Whole groups may exceed the
requested count slightly. Of 4,000 source rows, 3,999 are retained, with duplicates
retained within their assigned partition. Source row 572 (zero-based) contains
four calls with empty function names and is excluded, not repaired. Rejected
reference indices, hashes, and reasons are recorded. The manifest records counts, hashes,
source IDs, and the split algorithm. This is a new split, not the paper's random
row split. It is not a semantic deduplication or BFCL contamination audit.

Select the resulting manifest in a copy of a supported MILES configuration:

```toml
[data]
rl_manifest = "/absolute/path/to/toolrl-data/rl-manifest.json"

[core]
reward_final_answer_only = false
reward_zero_truncated = true

[miles]
custom_reward_post_process_path = "open_instruct.miles.rewards.truncation.exclude_truncated"
```

Remove any existing `data.tasks` or `data.prompt_data` selection: MILES permits
one data mode at a time. On Beaker, place the files on mounted storage and use
that mount's absolute path. The truncation postprocessor excludes unfinished
responses from learning: assigning them zero alone would make truncation more
rewarding than an incorrect call with negative reward. Follow the [MILES launch guide](../miles/launching.md)
for model, topology, runtime, and plan/validation checks. The `toolrl` entries are
**verifier bindings**, not new `[[data.tasks]]` dataset aliases. Manifest adoption
uses the standard tokenizer, prompt-length validation, and immutable preparation
checks; it does not need a published Hugging Face dataset.

If the model's generation prompt already ends in `<think>`, prepare with:

```bash
python -m scripts.data.rlvr.toolrl \
  --output-dir runs/toolrl-data-think-prefix --think-prefix-in-prompt
```

This selects `toolrl_think_prefix`, which restores that exact prefix for grading.
Both variants select the same rows for an equal split seed. MILES rejects a
mismatch between the selected verifier and the rendered prompt suffix. Prefixes
such as `<think>\n` need a separate explicit adapter; they are not silently
normalized. Keep `reward_final_answer_only=false` so the grader receives the
reasoning delimiters and call format. This flag does not reward the reasoning's
content or prescribe its opening words.

Local validation with the saved C11 tokenizer rendered all 3,999 retained rows;
the longest prompt was 3,821 tokens and 15 exceeded 2,048. A 4,096-token prompt
budget accommodates this source under that tokenizer. Recheck lengths for the
actual checkpoint and any format adaptation rather than silently truncating.

## Reward behavior

The adapter implements upstream's fixed default reward settings. It does not
vendor the veRL trainer or enable upstream's optional length and scheduling
variants.

| Component | Meaning | Range |
|---|---|---|
| Format | Expected reasoning and call/response delimiters, including newlines | 0 or 1 |
| Call correctness | Multiset overlap of tool names, followed by greedy matching of parameter names and values | -3 to 3 |
| Total | Sum of those two components | -3 to 4 |

This reward is not a probability or a BFCL accuracy score. A correct call target
with the expected format earns 4. Missing, extra, or wrong calls can receive
partial credit. For the 482 no-call targets, call correctness is always zero:
only the response **format** is graded, not the factual answer. Track those
examples separately when interpreting learning curves.

The port deliberately hardens malformed-output handling: duplicate JSON keys,
nonfinite constants, invalid call shapes, and repeated call blocks are rejected;
calls quoted only inside reasoning are not actions. It requires a single
reasoning block for format credit. These are differences from upstream on
malformed responses, not a claim of byte-for-byte paper reproduction. The
parameter-value comparison and greedy matching semantics otherwise follow the
upstream defaults, including Python's numeric equality semantics.

The code attribution is retained in `open_instruct/toolrl_utils.py`; upstream
code is Apache-2.0. The constituent datasets retain their own terms. This
integration does not publish or relicense the source data or model checkpoints.
BFCL remains an external evaluation: an overlap audit is still required before
using this mixture for a BFCL recovery claim.
