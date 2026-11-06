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
| `structured_v1` | `schema_type`/`schema_str`; explicit schema policy, all five formats and non-executing final tool calls |
| `instruction_v1` | Native instruction IDs and kwargs, including bundles in verifier metadata or args |
| `dolci_if_v1` | Published single-user Dolci IF rows or structured chat exports, with one encoded `ground_truth` bundle; optional recorded keyword correction |
| `calendar_v1` | Full final calendar feasibility from `exp_cal_state`, without a 20-event cutoff |
| `format_v1` | Existing format verifier payload; formatting only |
| `litmus_v02_v1` | Row-carried regex, answer type and matching policy; all scalar properties/methods are representable |
| `reasoning_gym_v1` | Saved entries for 104 pinned task families, including state-based graders; no regeneration or difficulty reset |
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

Structured tool-call rows are final-answer tasks, not tool-execution environments.
The adapter preserves every function declaration and renders it through the Olmo
Pythonic tool template. The verifier requires exactly one `<function_calls>` call,
the declared function name and payload key, and a schema-valid payload. It never
executes the call. CSV/object and TOML/array root requirements are recorded as
unrepresentable: the native parsers cannot produce those shapes.

ReasoningGym uses `reasoning-gym==0.1.25`. The 33 families inheriting its substring
scorer use exact answers; custom scorers retain native partial credit. Empty and
ambiguous answers score zero. Legitimate empty AB/equation solutions can use the
explicit `""` answer. State-based tasks need their puzzle metadata, not a fabricated
reference. Literal and symbolic answers are checked against a non-executable
grammar before native parsing. Malformed factorization, grid, action-plan and
nonfinite probability answers score zero with diagnostics. Puzzle24 additionally
requires the actual puzzle numbers and an exact total of 24. A five-second response
budget and outer worker deadline return zero on timeout; qualification rejects
such controls. Invalid targets and unexpected worker errors remain fatal.

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

## Qualification and publication

Source preparation and qualification run before training. Keep the qualified
bundle together: `catalog.json`, `qualification.json`, and each source's
`prompts.jsonl`, `verifiers.json`, hygiene receipt, and dispositions. The prompt
JSONL contains rendered input text, verifier targets, and source metadata; the
verifier registry selects trusted reward implementations and their configuration.

Qualification receipts bind each source to the SHA-256 hashes of its prompt data
and reward configuration. Catalog resolution verifies both those bindings and the
files themselves. Updating a catalog's file hashes cannot reuse an old receipt:
changed artifacts must be requalified. Receipts without these bindings also need
requalification. Moving an unchanged bundle preserves qualification because the
bindings use content hashes rather than absolute paths. These checks apply to
machine-readable qualification receipts; trusted catalogs with manual qualification
annotations remain a separate, manually reviewed path.

The catalog command emits source paths and mixture weights for the training
config. It does not embed the receipt into that config. At run preparation, MILES
writes its own source JSONL files, verifier registry, `mixture.json`, and
`manifest.json` containing input/output hashes and preparation settings. Resume
checks those hashes. Retain the qualified bundle alongside the run evidence:
the run manifest establishes what data was used, while the qualification receipt
records which controls passed for those source artifacts. Direct-path training
configs do not automatically enforce catalog qualification.

Provide reviewed controls keyed to actual prepared sample IDs. Each stratum needs
at least one case set containing `gold` (expected 1), `wrong` and `near_miss`
(expected 0 for binary graders; an explicit score below 1 for native partial-credit graders). A near miss should exercise the likely shortcut: a wrong number containing
the gold digits, a calendar constraint violation, a malformed schema field, or a
wrong MCQA option. The command additionally checks empty and truncated responses.
Do not manufacture a positive witness by assuming the model's response is right.

```json
{"source":"my-source","sample_id":"original-stable-id","cases":[{"kind":"gold","response":"2","expected":1},{"kind":"wrong","response":"3","expected":0},{"kind":"near_miss","response":"12","expected":0}]}
```

```bash
python -m scripts.data.rl_readiness.qualify_bundle new-bundle/catalog.json \
  --controls reviewed-controls.jsonl --tokenizer /path/to/checkpoint \
  --max-prompt-length 8192 --output qualified-bundle
python -m open_instruct.miles.datasets.source_catalog qualified-bundle/catalog.json \
  --source my-source=1.0 --output data.toml
```

Qualification validates the complete pool through `run_data.prepare_data` and
scores controls through `registered_reward`. It rejects conflicting targets for
the same prompt. Different code test suites may share a prompt only when every
row has a reviewed common gold solution that passes its complete original suite. Missing strata, failed controls, unresolved judge services, or
loader failures leave the source unqualified. Passing control rows never replace
the full source. Quarantine genuinely malformed source rows without repairing their
prompts or targets. Pass `--quarantine reviewed-quarantine.jsonl`, with one record
`{"source":"my-source","id":"original-stable-id","reason":"Invalid original schema"}`
per reviewed omission. The identities must match **every** omitted row exactly;
unsupported valid rows cannot silently disappear. The qualifier preserves the
hashed dispositions and quarantine report, and records the retained population.
This qualifies the explicitly retained variant, not the original malformed rows.
The report includes source/controls
hashes, loader manifest hash, context limit, judge registry fingerprint and all
missing strata. It does not claim that a GPU optimizer update was exercised.

Use the readiness Docker overlay on an application image built from this stack,
so training receives the pinned optional verifier versions. Its build checks the
qualification and source-preparation entry points. After CPU checks, use the
existing local toy-model coverage workflow for a GPU mechanics exercise. Controls
validate reward behavior; training continues to use the complete qualified pool.

## Complete-test code sources

`acecode_v1` uses the original `context_messages` when present and preserves every
`test_cases` assertion. `livecodebench_v1` reads
`verifier_metadata.unit_tests` (`inputs`, `outputs`, optional `fn_name`). It retains
all aligned pairs and distinguishes stdio from module/Solution function calls.
The readiness requirements pin the maintained execution client to commit
`4cab7e85e8748db501163e6cbbc5ab7c041692af`.

Each source plan must supply trusted service configuration, outside raw rows:

```json
"code_service": {
  "api_url": "https://1n7avfskve.execute-api.us-west-2.amazonaws.com",
  "expected_source_sha256": "f8ee353acb3e49580439ae1dbdb1cbb66550e9f4fbc5f2dcb64feb5c64cc5905",
  "max_execution_time": 1.0,
  "total_timeout": 600.0
}
```

Inject `CODE_EXECUTION_TOKEN` at runtime from the authorized Beaker secret;
credentials never belong in plans, rows or qualification reports. Reward is one
only when **every** test passes. All tests execute, including after a failure;
suites above 2,000 tests are partitioned without truncation. Each partition uses
the configured total client deadline, and each test uses a fresh interpreter.
Syntax/runtime errors and execution timeouts are zero; infrastructure, incomplete
result and service-source mismatches raise. Reports retain request identifiers.
Start qualification sequentially; training must bound aggregate in-flight service
requests to eight across all callers, not eight per trainer rank. The live control
panel is not a sustained-load certification.

The Nemotron-backed `livecodebench_v1` adapter automatically selects the pinned
`nemo_v1` output comparison through `prepare_nemotron_tests`; there is no training
comparison option. Whole-line yes/no verdicts ignore case, floating stdio tokens
use NeMo's 1e-6 tolerance while integer tokens stay exact, and function results
use the native scalar/list comparison rules. AceCode retains its original assertions.
The client rejects a service that does not advertise the required comparison;
unknown outcomes never become zero rewards. Existing prepared Nemotron bundles
must be rebuilt and qualified so they contain the new comparison metadata and
function assertions. Update their trusted service source pin as well.
See the pinned [NeMo comparison contract](https://github.com/allenai/code-execution-service/blob/4cab7e85e8748db501163e6cbbc5ab7c041692af/docs/nemotron-comparison-contract.md)
for exact rules and the 10,053-case native-checker differential validation.
This closes the comparison implementation gap; whole-source dataset qualification
and training-load validation remain separate gates.

## Reference judges and routed blends

`reference_judge_v1` preserves a nonempty reference from `expected_answer` (or
explicit `options.reference_field`) and the full conversation in `judge_query`.
It uses the named judge marker `reference_judge`. Configure that
name in the run's `[judging.bindings]` using the same approved service and rubric
for training. Qualification requires a resolved registry and live
positive/negative controls. This draft has not launched or calibrated a new
judge service. A generic reference judge is a new explicit reward contract, not
an assertion of parity with a dataset's original judge prompt/model or a hybrid
math-plus-judge scorer.

For blends, `routed_v1` selects adapters through an explicit field and mapping:

```json
{"adapter":"routed_v1","options":{"field":"agent_ref.name","routes":{"math_with_judge_simple_agent":{"adapter":"reference_judge_v1"},"instruction_following_simple_agent":{"adapter":"instruction_v1"}}}}
```

Unknown routes are reported as omissions; no fallback turns them into math or
incorrect answers. Routes cannot recursively dispatch. Select `math_v1` only
when intentionally declaring a symbolic-answer contract; use
`options.reference_field="final_answer"` for DeepMath. The source's native
messages remain intact. The runtime registry and prepared target encode the
selected reward semantics.

## Coverage and remaining work

This table covers the 21 standalone-trial sources. It is not a claim that the
entire 100+ acquisition catalog has been integrated or requalified.

| Sources | Shared integration | Remaining qualification or environment work |
|---|---|---|
| Litmus v0.2 (C010) | Native answer formats and scalar rules across properties | Independent scientific reference checks and controls for all new strata; tool-dependent variants need their environment |
| Llama-Nemotron IF, Nemotron IF (C016/C021) | Maintained native checker adapters, full original messages | Language classifier calibration; malformed/unsupported arguments remain visible |
| Citation and Free-Form Formatting (C018/C019) | Existing bounded formatting verifier | Per-format witnesses, long-context loader checks; rewards measure format only |
| Calendar v2 (C023) | Full final-state schedule, including empty calendars | Controls across calendar strata and original-history context; no interactive dialogue-quality claim |
| Structured Outputs v2 (C029) | Five formats and bounded schema regex evaluation | Reviewed witnesses across schemas/formats, coercion/strictness contract review |
| Knowledge MCQA (C031) | Displayed choice validation and final-choice reward | Native-template extraction checks, chance floor when interpreting MCQA rewards |
| ReasoningGym (C036) | Saved entries/difficulty for all 104 exported families, including state-based tasks | Pinned native scoring with safe expression guards, partial-credit controls and bounded workers; no procedural population replacement |
| Cascade Math, Nemotron Math v2, OpenMathReasoning, Math Stack Overflow, AceReason Math, DeepMath (C037–C040/C091/C094) | Explicit symbolic or reference-judge routing | Proof/judge parity and service calibration; canceled Nemotron Math run is not restarted |
| Nano, Cascade-2, Lightning, Super, Ultra blends (C043–C047) | Explicit per-agent routes with full omissions report | Native judge/hybrid contracts; full code-source qualification and tool/SWE/Lean environment integration remain open |
| AceCode (C076) | `acecode_v1` preserves context and every assertion; authenticated, source-pinned execution service | Full retained-pool qualification and aggregate service-load checks |

The web-search candidate remains excluded at Robert's request. SWE, Lean,
browser/tool interaction and distillation-only records must not be presented as
single-turn end-to-end training support. They need an actual environment or
objective implementation. Licensing and scientific validity remain properties of
the source and its evidence; successful parsing is not either kind of approval.
