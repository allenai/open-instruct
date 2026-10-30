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
| `instruction` | Extracted function definitions in `tools`; upstream response-format instructions are discarded |
| `input` | Structured user, assistant (reasoning/calls/text), and tool-result turns |
| `output` | Canonical function names and typed arguments in the verifier target; never appended to the prompt |

The selected checkpoint's chat template supplies the system message, tool menu,
call syntax, conversation markers, and generation prefix. For Olmo 3.5 this is
`<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`
(with newlines). Parallel calls have separate `<tool_call>` wrappers. Ordinary
answers are plain text after reasoning, without `<response>` tags. Historical
observations become tool turns, rendered as environment/tool-response messages.
Training generates one next assistant turn against an offline reference; it does
not execute tools or continue a live environment.

The manifest declares `tools_key` and `tool_format = "olmo-toolrl-v1"`. MILES
passes structured tools to `apply_chat_template` and checks that the selected
template renders the native function/parameter syntax. It does not replace the
checkpoint's template. Old JSON-format manifests/targets must be regenerated;
the verifier rejects their unversioned string targets.

The verifier removes at most one trailing `<|im_end|>` or `<|endoftext|>`
boundary token before grading. These may remain in the serving response when
special-token skipping is disabled. Tokens inside arguments are preserved;
extra text or repeated boundary tokens after a call still fail strict parsing.

Tool definitions use JSON Schema. The adapter translates Python-like source types
(`int`, `List[float]`, etc.), treats parameters marked optional or given defaults
as optional, and represents source callable expressions as strings without
executing them. Argument values are decoded using the declared top-level type;
numeric strings become numbers, JSON containers become containers, and string
arguments remain strings (outer whitespace is stripped). Nested container values
are preserved; this is not a full JSON Schema validator.

| Native preparation outcome | Rows |
|---|---:|
| No-call targets | 456 |
| One reference call | 1,615 |
| Two or more reference calls | 1,515 |
| Retained total | 3,586 |
| Excluded during reference/history/schema conversion | 414 |
| Retained references with normalized argument values | 371 |

Unlike the initial JSON-format draft, preparation checks that reference and
historical calls name available functions and declared parameters, and that their
argument values can be decoded. Some upstream rows contain truncated JSON inside
argument strings, expressions in numeric fields, or unavailable function names.
These are excluded with their original row index, reference hash, and reason;
they are not guessed or repaired. The manifest also lists normalized-reference
indices. This filtering changes the training population, so it is not a pure
serialization ablation or reproduction of the paper's dataset split.

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
requested count slightly. The default native preparation produces 3,506 training
rows and 80 evaluation rows. Duplicates stay in their assigned partition. The
manifest records artifact hashes, source IDs, conversion outcomes, and the split
algorithm. This is a new split, not the paper's random row split. It is not a
semantic deduplication or BFCL contamination audit.

Select the resulting manifest in a copy of a supported MILES configuration:

```toml
[data]
rl_manifest = "/absolute/path/to/toolrl-data/rl-manifest.json"

[core]
reward_final_answer_only = false
reward_zero_truncated = false

[miles]
custom_reward_post_process_path = "open_instruct.miles.rewards.truncation.exclude_truncated"
```

Remove any existing `data.tasks` or `data.prompt_data` selection: MILES permits
one data mode at a time. On Beaker, place the files on mounted storage and use
that mount's absolute path. The truncation postprocessor excludes unfinished
responses from learning: assigning them zero alone would make truncation more
rewarding than an incorrect call with negative reward. Keep zero-truncation
disabled because MILES rejects combining it with reward exclusion. If every
response in an optimizer batch is excluded across all ranks, Core records a
`no_active_tokens` skip and consumes the rollout without advancing optimizer,
learning-rate, or policy-version state. A locally empty rank still participates
when another rank has active tokens. Follow the [MILES launch guide](../miles/launching.md)
for model, topology, runtime, and plan/validation checks. The `toolrl` entries are
**verifier bindings**, not new `[[data.tasks]]` dataset aliases. Manifest adoption
uses the standard tokenizer, prompt-length validation, and immutable preparation
checks; it does not need a published Hugging Face dataset.

Start a mechanics configuration with `cp configs/miles/examples/dev.toml
runs/toolrl.toml`, apply the data/reward settings above, and set the checkpoint,
output directory, prompt/response budgets, and launch resources for that model.
Then inspect the configuration without submitting a job:

```bash
python -m open_instruct.miles plan runs/toolrl.toml
python -m open_instruct.miles validate runs/toolrl.toml
```

After those checks and runtime qualification, `python -m open_instruct.miles run
runs/toolrl.toml` launches through the committed-image wrapper. The selected
image must contain this integration; reusing an older image does not upload
local Python changes. These commands document the run path, not evidence of a
completed GPU run.

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

Local validation with the saved C11 tokenizer rendered and graded all 3,586
retained references successfully, including parallel calls and no-call answers.
The longest prompt was 3,770 tokens; a 4,096-token prompt budget accommodates this
source under that tokenizer. Recheck lengths for the actual checkpoint rather
than silently truncating. These are CPU checks, not GPU rollout qualification.

## Reward behavior

The adapter retains upstream's fixed default semantic call scoring, with native
Olmo formatting and canonical typed targets. It does not
vendor the veRL trainer or enable upstream's optional length and scheduling
variants.

| Component | Meaning | Range |
|---|---|---|
| Format | One reasoning block followed by native calls or a plain answer; flexible whitespace | 0 or 1 |
| Call correctness | Multiset overlap of tool names, followed by greedy matching of parameter names and values | -3 to 3 |
| Total | Sum of those two components | -3 to 4 |

This reward is not a probability or a BFCL accuracy score. A correct call target
with the expected format earns 4. Missing, extra, or wrong calls can receive
partial credit. For the 456 no-call targets, call correctness is always zero:
only the response **format** is graded, not the factual answer. Track those
examples separately when interpreting learning curves.

The offline native parser requires complete call blocks and rejects duplicate
parameters, malformed tails, nonfinite JSON constants, and top-level type
mismatches. Calls quoted only inside reasoning are not actions. Separate complete
parallel-call blocks are accepted. Whitespace between reasoning and calls or
between calls is not a reward signal. The two verifier bindings still differ
only in whether `<think>` is already in the prompt; neither forces particular
reasoning words.

This parser is deliberately stricter than permissive serving parsers, and has no
vLLM or SGLang import dependency. Unknown predicted function/parameter names are
scored as semantic errors with upstream partial credit. Parameter equality and
greedy matching retain upstream semantics (including Python numeric equality),
but compare normalized targets rather than the upstream serialization. No-call
answers receive format credit only; no factual correctness is claimed.

The code attribution is retained in `open_instruct/toolrl_utils.py`; upstream
code is Apache-2.0. The constituent datasets retain their own terms. This
integration does not publish or relicense the source data or model checkpoints.
BFCL remains an external evaluation: an overlap audit is still required before
using this mixture for a BFCL recovery claim.
