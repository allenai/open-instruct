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
| `dolci_if_v1` | Published single-user Dolci IF rows or structured chat exports, with one encoded `ground_truth` bundle; optional recorded keyword correction |
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
for compatibility. To keep those prompts unchanged while grading inclusion,
select the correction explicitly in the source plan:

```json
{
  "adapter": "dolci_if_v1",
  "options": {"keyword_inclusion": "at_least_once_v1"}
}
```

Include the usual pinned `inputs` and `hygiene_receipt` fields alongside this fragment.
This adapter consumes Dolci IF rows with one `ground_truth` entry
(the existing JSON/Python singleton wrappers are accepted). `dataset` may be absent
for Dolci-RLZero-IF-7B; if supplied, it must be `["ifeval"]`. The explicit adapter
selection identifies the IF-only source; it does not infer a route for mixed data.

Published single-user `prompt` strings such as `user: Solve this problem…` are
converted directly to `[{"role": "user", "content": "Solve this problem…"}]`.
Only the initial, exact `user: ` serialization prefix is removed; all remaining
whitespace and text are preserved. No separate conversation exporter is needed.
Use the preparation command above on local JSONL rows (or hygiene wrappers),
selecting the IF-only rows when exporting a mixed Dolci dataset to JSONL. Existing
source hashes, revision, split, and hygiene receipt requirements still apply.

Flattened strings with additional newline-delimited role labels are ambiguous:
they could represent additional turns or quoted dialogue. These receive an
`ambiguous_dolci_conversation` omission in `dispositions.jsonl`. Other leading
role serializations receive `unsupported_dolci_role_serialization`. Recover these
from an authoritative conversation representation and supply structured messages,
e.g. `input: [{"role": "user", "content": "..."}]`; do not guess turn boundaries.
Structured message content is never scanned for role labels or stripped, so real
text starting with `user:` remains intact. Ordinary strings without a role prefix
remain ordinary user content. Token-only exports still need upstream decoding.
This conversion is specific to `dolci_if_v1`; other adapters are unchanged. The
prepared bundle uses `ManifestIFVerifier` and normal fraction-of-constraints grading.

The default policy is `preserve`. With `at_least_once_v1`, each `keywords:word_once`
constraint becomes `keywords:frequency` with arguments
`{"keyword": X, "frequency": 1, "relation": "at least"}`. The final user message
must contain the audited sentence `Include keyword X in your response.`. The
keyword must contain only ASCII letters with no surrounding whitespace, matching
the audited Dolci rows: the existing frequency checker treats keywords as regexes,
so expanding this repair to arbitrary punctuation would require a separate decision.
Unrecognized wording or keyword shapes are reported as omitted rows. Other
constraints and prompt text are preserved; existing admission screens still apply.
An already-present frequency constraint remains a separate constraint in the bundle.

Each prepared row records `original_target_sha256` and `target_transformations`,
including the correction version, constraint index, old/new identifiers, and
`from_kwargs`/`to_kwargs`. The catalog records the selected options and transformed
row/constraint counts; the original input hash and saved plan remain available.
Write a new bundle rather than replacing historical artifacts. The option is
rejected for other adapters: NeMo sources retain their original keyword identifiers
and grading. Existing `run_data` and older trainer paths also keep exactly-once
grading unless they explicitly consume corrected prepared targets. There is no
runtime switch based on dataset origin and no new checker identifier.

The shared `word_once` prompt generator now explicitly says “exactly once” for
future generated examples. This does not rewrite existing dataset prompts.
