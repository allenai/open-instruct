# Preparing dataset sources

Prepared source bundles connect datasets to MILES training without
changing their original prompts or silently narrowing their populations. The
preparation command consumes local JSONL exports from completed hygiene work;
it does not download datasets or rerun decontamination. Each input and the prior
hygiene receipt must have a SHA256. Use the latest selected dataset revision,
including Litmus v0.2 rather than both Litmus releases.

```json
{
  "schema_version": 1,
  "sources": {
    "structured-outputs-v2": {
      "adapter": "structured_v1",
      "inputs": [{"path": "clean/structured.jsonl", "sha256": "REPLACE", "revision": "PINNED_DATASET_COMMIT", "split": "train"}],
      "hygiene_receipt": {"path": "decon-receipt.json", "sha256": "REPLACE"},
      "options": {"schema_policy": "nemo_strict_v1"}
    }
  }
}
```

```bash
python -m open_instruct.miles.datasets.source_preparation plan.json \
  --tokenizer /path/to/checkpoint --output /path/to/new-bundle
```

Inputs may be raw rows or hygiene wrappers containing `id`, `raw_record` and
`provenance`. Preserve the prior hygiene decision in the input selection and
receipt; pointing at a receipt does not independently certify a new input's
cleanliness. The adapter does not select on policy reward, reserve screened
prompts, cap pool size, or rewrite messages with a new answer suffix. It records
all unsupported rows in `dispositions.jsonl`, including rows with malformed
messages, invalid schemas or regular expressions. An unknown adapter name is a
plan error and stops preparation. Long prompts remain in the pool:
set an adequate training context limit, or explicitly
name and review a shorter-context variant. Repeated prompts are counted, not
silently removed; resolve conflicting targets before qualification.

Responses API `instructions` are placed first in a single system message,
followed by the existing leading system/developer messages in their original
order, separated by blank lines. This follows
[NeMo Gym's Responses-to-chat conversion](https://github.com/NVIDIA-NeMo/Gym/blob/main/nemo_gym/models/chat_completions_converter.py)
and lets our chat templates render developer instructions through their supported
system role. Instruction text and subsequent conversation turns are preserved;
the selected tokenizer's chat template supplies the role delimiters and assistant
generation prefix. Developer messages after the leading context remain unsupported
and are recorded as omissions rather than moved across conversation turns.

| Adapter | Target and scope |
|---|---|
| `nemo_v1` | Structural dispatch for formatting, structured formats, native IF, calendars and MCQA; unknown routes are reported |
| `structured_v1` | `schema_type`/`schema_str`; explicit schema policy, all five formats |
| `instruction_v1` | Native instruction IDs and kwargs, including bundles in verifier metadata or args |
| `dolci_if_v1` | A single `ifeval` route in `dataset`, with one encoded constraint bundle in `ground_truth`; optional recorded keyword correction |
| `calendar_v1` | Full final calendar feasibility from `exp_cal_state`, without a 20-event cutoff |
| `format_v1` | Existing format verifier payload; formatting only |
| `litmus_v02_v1` | Row-carried regex, answer type and matching policy; all scalar properties/methods are representable |
| `reasoning_gym_v1` | Saved entries for the ten qualified families, including ARC/reARC; no regeneration or difficulty reset |
| `mcqa_v1` | Displayed options and expected answer, existing final-choice parser; no science-only filtering |
| `math_v1` | Explicit symbolic-math source contract; proof requests are reported as requiring judged support |
| `deterministic_v1` | Existing numeric, exact-text/JSON, countdown and knights contracts in `target` |

New registries use
`open_instruct.miles.rewards.deterministic_verifiers.DeterministicVerifier`.

Litmus preserves row-specific answer formats and tolerances. The versioned
`litmus_scalar_v1` contract deliberately requires complete finite numeric or
explicit boolean captures; it does not recover an incidental number from malformed
text or treat arbitrary nonzero numbers as true. A row needs an explicit
one-capture `output_regex` and a nonempty scalar `expected_answer`. The matching rules are `isclose`, rounded-integer
`exact`, absolute/relative windows, boolean equality and normalized string
equality. This expands *verifier wiring*, not independent scientific validation:
new property/method strata still need reference checks and positive/near-miss
controls before qualification. Tool-dependent chemistry remains unsupported by
single-turn training even when its final scalar can be graded.

Litmus regex matching has a 100 ms budget. As with the other local validation
adapters, a timeout always returns zero with `status: timeout`, the error and
elapsed time in diagnostics; there is no policy setting. Invalid targets and
unexpected failures still raise. Qualification rejects timed-out controls, even
though training can continue with zero reward for a timed-out response.

The catalog initially says `prepared`, not `qualified`. It cannot be resolved by
the named-source resolver until reviewed controls and the training loader path
have passed. Integration readiness, representative policy evidence and downstream
training benefit remain separate fields of evidence. Historical measurements
from changed populations do not qualify these bundles.

See [dataset reward verifiers](dataset-verifiers.md) for the structured-output,
instruction and calendar contracts and their runtime dependencies.

## Explicit Dolci keyword correction

Historical Dolci IF prompts say “Include keyword X” but use `keywords:word_once`,
whose historical grader requires exactly one regex match. Preserve that identifier
for compatibility. To keep those prompts unchanged while grading literal inclusion,
select the correction explicitly in the source plan:

```json
{
  "adapter": "dolci_if_v1",
  "options": {"keyword_inclusion": "at_least_once_v1"}
}
```

Include the usual pinned `inputs` and `hygiene_receipt` fields alongside this fragment.
This adapter consumes Dolci IF exports with `dataset: ["ifeval"]`, one `ground_truth`
entry (the existing JSON/Python singleton wrappers are accepted), and text or chat
messages in the normal preparation input fields. Token-only exports must first be
decoded upstream; this adapter does not infer chat boundaries from rendered text.
It keeps the normal IF fraction-of-constraints grading through `ManifestIFVerifier`.

The default policy is `preserve`. With `at_least_once_v1`, each `keywords:word_once`
constraint is changed to `keywords:at_least_once` only when the final user message
contains the audited sentence `Include keyword X in your response.` with the matching
keyword. A mismatch is recorded as an omitted row, not guessed or silently repaired.
Other constraints and prompt text are preserved. Existing admission screens still apply.
The new checker uses literal, case-insensitive substring matching and permits repetition;
it does not reinterpret other frequency rules or uniqueness constraints.

Each prepared row records `original_target_sha256` and `target_transformations`,
including the versioned correction name, constraint index, and old/new identifiers.
The catalog records the selected options and transformed row/constraint counts;
the original input hash and saved plan remain available. Write a new bundle rather
than replacing historical artifacts. The option is rejected for other adapters:
NeMo sources retain their original keyword identifiers and grading. There is no
runtime switch based on dataset origin.

The shared `word_once` prompt generator now explicitly says “exactly once” for
future generated examples. This does not rewrite existing dataset prompts.
