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
| `toolrl_v1` | Canonical ToolRL messages, tools and reference calls; original partial-credit reward and format-only answer rows |
| `single_step_tool_v1` | NeMo offline next-action routes with native conversational, function-calling or SWE-pivot comparison rules |

New registries use
`open_instruct.miles.rewards.deterministic_verifiers.DeterministicVerifier`.

## Offline tool-call datasets

ToolRL and NeMo's single-step tool datasets teach the next action from an existing
conversation. Their prompts include tool definitions, previous calls, observations
and reasoning. Training generates one next response and grades it against the
source target; it does not execute the tools or require a live environment.
These are separate reward contracts, even though they share native-format parsing.

### ToolRL

The source converter reuses the prior ToolRL integration's pinned 4,000-row source,
schema translation, reference-argument conversion and dialogue-group train/holdout
split. Its immutable source is
[ToolRL revision 8cee13ec0ca7](https://github.com/qiancheng0/ToolRL/tree/8cee13ec0ca72f0461da372a93a6fd8140dbb840).
To reproduce canonical rows offline:

```bash
python -m scripts.data.rlvr.toolrl --input /path/to/rlla_rl.json \
  --output-dir /path/to/toolrl-canonical
```

The converter checks the source SHA256 and writes `train.jsonl`, `eval.jsonl`
and `source-manifest.json`, including every rejected source coordinate. These
are intermediate records. Use `toolrl_v1` in the source plan above, with the
hashed `train.jsonl` input and the existing hygiene receipt, then run
`source_preparation`. Keep `eval.jsonl` out of training. The adapter validates
reference and historical calls against their schemas and reports malformed rows
as omissions; it does not silently truncate long histories or choose rows by reward.
Native XML parameters allow unquoted strings when `string` is one of a union's
types, such as `["string", "null"]`. JSON values still use their declared types;
duplicate JSON keys and nonfinite values remain invalid.
Canonical source strings in unions retain their original type, including numeric-looking
IDs and the literal string `null`. In XML calls, quote a string as JSON when its
bare form would also match another allowed type (for example, `"5"` for a
string/integer union); Pythonic calls carry literal types explicitly.

`options.tool_format` is `olmo_pythonic_v1` by default; use `olmo_xml_v1` for
the XML template from the earlier ToolRL experiments. Preparation checks the
template and records whether its generation prefix already supplies `<think>`.
Historical reasoning is embedded in assistant content for templates that do not
consume `reasoning_content`. Literal tool/parameter names may contain spaces,
digits or parentheses: they remain unchanged and are parsed without executing
Python. Keep `core.reward_final_answer_only: false`, since ToolRL grades the
reasoning/answer boundary as well as the calls.

| Target | Preserved reward |
|---|---|
| One or more reference calls | Format credit 0 or 1, plus native multiset-name overlap and greedy argument matching from −3 to 3; total −3 to 4 |
| No reference calls | Plain response format only, from 0 to 1; no factual correctness check |

The reward factory is `open_instruct.miles.rewards.toolrl.ToolRLVerifier`.
Qualification recognizes this native scale; it does not rescale training rewards
to fit a generic binary control. Training's configured truncated-response gate
still returns zero. Account for the reward scale when designing mixtures.

The 2026-11-06 local first-pass check reproduced the prior 3,506 training rows
and 80 held-out rows exactly; the converter reports 414 malformed or unsupported
raw records separately. Six previously identified schema-invalid training
rows were quarantined. All remaining **3,500** rows passed prompt preparation,
reference-action rendering/reward checks and the whole-pool training loader:
3,056 call targets and 444 format-only targets. Maximum prompt length was 3,570
tokens; the nine rows previously deferred at 2,048 tokens are retained. Gold,
wrong, near-miss, empty and truncated controls passed for both reward categories.
This validates the integration, not an independent semantic relabeling or a new
learning run. The existing BFCL overlap report contains only short embedded
matches (one to four words); the receipt preserves those review flags and does
not assert independent benchmark cleanliness.

The corrected v2 release recovers one additional string-union row, reducing
conversion omissions to 413 and increasing the qualified training pool to
**3,501** rows. The original 80-row holdout and all previously retained rows are
unchanged. Full-pool qualification and 15 runtime controls passed; see the
[first-pass handoff](first-pass-data.md) for the corrected release and image.

### NeMo single-step tool routes

`single_step_tool_v1` selects the source's `agent_ref.name`: ordinary
`single_step_tool_use_with_argument_comparison_agent`,
`toolcall_schema_single_step_tool_use_with_argument_comparison_agent`, or
`swe_pivot_single_step_tool_use_with_argument_comparison_agent`. It uses the same
explicit `tool_format` options as ToolRL. Unknown routes require an explicit
reviewed profile, rather than guessing from the prompt.

Comparison code and profile settings are pinned to
[NeMo Gym 966145fc02ae](https://github.com/NVIDIA-NeMo/Gym/tree/966145fc02ae58b887fd49379675604cc38fdd43/resources_servers/single_step_tool_use_with_argument_comparison).
The native string word-count thresholds are 0.1 for conversational/function-calling
and 0 for SWE-pivot; numeric tolerance is 1e-6. Zero word-count tolerance still
ignores word order. Calls take precedence over accompanying prose; with native
parallel-call rewarding disabled, surplus calls are not penalized when the expected
calls match. A message target rewards any nonempty assistant prose, not semantic
equivalence to the reference message. Preserve these upstream limitations for
initial mixing; investigate stricter behavior separately rather than silently
changing reward semantics. Response parsing/resource failures score zero with
diagnostics, while malformed targets remain fatal verifier errors.

These routes do not implement SWE patch execution, Workplace Assistant state
changes, browser interaction or Lean compilation. Those environment-dependent
routes remain separate integration work.

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
unexpected failures still raise from the verifier. Qualification rejects timed-out controls, even
though training can continue with zero reward for a timed-out response.

Managed training uses `rollout_reward`: if an individual verifier raises or a
judge reports a failed grade, its reward is missing rather than zero. The pinned
MILES collector discards the entire prompt group (including healthy sibling
responses) and draws replacements. `rollout/verification/omitted_records` in W&B
counts these missing-reward prompt groups per collection interval, including groups
that never enter the training batch. It counts sampling attempts, not distinct
source IDs; legitimate wrong answers and local validation timeouts do not count.
The native `rollout/dynamic_filter/drop_group_has_missing_reward` counter remains
available too. Diagnostics retain the verifier name/error on the failed sample,
with bounded failed-input captures under `<save>/verifier_failures`.

Offline qualification and evaluation remain strict; registry construction and
routing/configuration errors also remain fatal. **Open decision:** how many or
what fraction of omissions should stop a run. No tolerance setting or abort
threshold is introduced here. A sustained verifier outage can therefore keep
collection waiting for replacements; W&B receives the count when collection
logs, while individual failures are logged immediately.

Structured tool-call rows are final-answer tasks, not tool-execution environments.
The adapter preserves every function declaration and renders it through the Olmo
Pythonic tool template. The verifier requires exactly one `<function_calls>` call,
the declared function name and payload key, and a schema-valid payload. It never
executes the call. CSV/object and TOML/array root requirements are recorded as
unrepresentable: the native parsers cannot produce those shapes.
Structured preparation resolves each local schema reference in its own `$id`
scope, without fetching external resources. Missing pointers or anchors are
quarantined as invalid targets before training, even in optional branches.
Valid nested resource identifiers remain supported.

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
wrong MCQA option. The command additionally checks empty and truncated responses. Timeout,
`judge_error` and `service_error` diagnostics fail qualification even when a
negative control otherwise receives the expected zero.
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
the same prompt. Different code suites, citation-marker sets and equivalent math labels may share
a prompt only when every row has a reviewed common gold response that passes its
original target through the runtime bridge. A shared witness establishes joint
satisfiability, not identical rewards for every response; compatible duplicate
code/citation targets can still contribute label noise. ReasoningGym export UUIDs are
provenance only; all scoring fields remain part of duplicate-target comparison. Missing strata, failed controls, unresolved judge services, or
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

For a lightweight first IF pass, explicitly select
`--instruction-family-coverage` and give each control a zero-based
`constraint_index`. The qualifier projects that real row's selected constraint
for the control, and requires coverage of every checker family found in the
complete population. Training retains the original complete constraint bundles.
The receipt calls this **family coverage**, not a solved witness for every
combination of constraints. The default remains per-stratum coverage; projected
controls without the flag are rejected. The flag applies to the invocation, so
qualify IF and non-IF source catalogs separately before assembling a mixed catalog.

Large prepared files are streamed during qualification, training preparation,
serialization and resume hashing. Code-test payloads remain complete without
requiring several whole-file copies in memory. Identity and prompt-hash indexes
still scale with the number of rows.

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
`103ca279b35e8e81d118947f82b72c22676c6540`.

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
result and service-source mismatches raise. A returned partition must explicitly
report `ok` or `completed`; any other or missing status becomes a top-level
service failure. Training omits the ungraded group, while offline scoring and
qualification fail. Reports retain partition diagnostics and request identifiers,
and complete-code records participate in `rollout/code_verifier/*` metrics.

The service owner's 2026-11-06 scaling panel supersedes the original eight-request
guidance. Reserved Lambda concurrency is now 900, with no provisioned warm
baseline. The panel tested 12, 16, 32, 64 and 128 concurrent HTTP requests:
38,056 requests and 542,262 test outcomes completed with no retries,
infrastructure failures, throttles or scoring differences from the sequential
baseline. Timeout recovery passed, and CloudWatch confirmed scaling with zero
throttles. At 128 concurrent requests, the selected workload sustained 266.6
requests/second (about 60 complete solutions/second).

The configured ceiling of 900 is not a tested operating point: load qualification
reached 128 concurrent HTTP requests, with roughly one minute of submissions per
phase followed by draining. Capacity is shared across runs; keep pending work
bounded and account for overload. End-to-end multi-run admission/backpressure
remains to be validated. Service version 9 and the client/source pins above are
unchanged. Evidence: `docs/scaling-load-validation-20261106.md` and its companion
JSON in the `allenai/code-execution-service` repository (service-owner report).

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
and end-to-end training-load validation remain separate gates; the bounded
service-load panel above is complete.

### Alternative valid code outputs

Some questions permit arbitrary output order, any tied optimum or any valid
construction, while their supplied tests accept only one reference output. For
example, an AceCode TwoSum question allows either index order but its assertion
accepts `[0, 1]` and rejects `[1, 0]`; a Nemotron string-construction task similarly
rejects a different valid string. These are limitations of the supplied grading
contracts: [NeMo compares reference outputs](https://github.com/NVIDIA-NeMo/Gym/blob/bd6f5f0d8f3a56cbf3f7e1d2d4a1d2220b7ff6fe/resources_servers/code_gen/lcb_integration/testing_util.py),
and [AceCode executes the supplied assertions](https://github.com/TIGER-AI-Lab/AceCoder/blob/e37ef2aea2e3710be6b06e9a91dd92c98321df96/src/acecoder/evalplus_eval.py#L160-L168).

**Deferred investigation:** consider recovering original special judges or adding
task-specific property checkers, with valid-alternative and invalid-output
controls. Unordered comparison must preserve multiplicity where required;
construction and optimality tasks need their own checks. This would intentionally
diverge from upstream grading and requires a separate review of reward changes.
For now, preserve the upstream tests and comparisons; this concern does not
introduce new filtering, repairs or a qualification blocker. Such false negatives
are ordinary zero rewards, not missing grades, so they do not increment
`rollout/verification/omitted_records`.

## Native language constraints

Preserve the native checkers' behavior when language detection raises
`LangDetectException`: the language constraint passes. Both our existing IFEvalG
checker and [NeMo's pinned upstream checker](https://github.com/abukharin-nv/verifiable-instructions/blob/f46a5ac87b1400a4f8973039844b6be9b56e3faf/verifiable_instructions/instructions.py#L137-L186)
use this fallback. An unclassifiable answer is not necessarily a language
violation: `42` can correctly answer an English question without containing
detectable language. This is an intentional acceptance of ambiguity; do not
replace it with a zero reward or omit the source row solely for that reason.

The uppercase/lowercase English checkers retain their separate case tests;
digits alone fail those tests before language detection runs. A successfully
detected language that differs from the required language still fails.
The NeMo adapter admits all three language-related checkers using their existing
native semantics, without a separate classifier-calibration gate.

The upstream detector is randomized: identical responses can receive different
language labels and therefore different rewards. Short or ambiguous text is
particularly susceptible, but length alone does not guarantee stability (for
example, a long multilingual response can remain ambiguous). See the
[detector's documentation](https://github.com/Mimino666/langdetect#basic-usage).
Preserve upstream behavior without introducing a fixed seed or replacing the
classifier in this integration. TODO: evaluate a better language classifier on
representative training responses, including short answers, mixed languages and
case constraints; measure accuracy, repeatability and reward changes before
adopting it. This follow-up is outside the current integration's scope and is
not a qualification blocker. Admission does not establish classifier accuracy.

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

The [first-pass data handoff](first-pass-data.md) records the released source
variants, counts, qualification scope, artifact location and an example mixture.
Use that snapshot for initial mixing experiments. Source support does not imply
that every route in a heterogeneous upstream blend has its required judge or
execution environment.

Keep these follow-ups separate from the released maintained-verifier pools:

| Area | Remaining work |
|---|---|
| Judged/hybrid math, proof, GenRM, rubrics, safety and abstention | Preserve each upstream judge prompt, parsing and aggregation contract; deploy and calibrate the matching service. A generic reference judge is not parity. |
| Workplace, SWE execution, Lean, browser and other interactive tasks | Actual environment integration and recovery checks; offline next-action tool grading does not implement these environments. |
| Alternative valid code constructions, arithmetic grammar and language detection | Preserve the documented upstream behavior for the first pass; investigate deliberate changes separately. |
| Runtime verifier outages | Omitted-group W&B metric is implemented; the systemic-failure stopping threshold remains an explicit open decision. |
| Semantic data quality | Qualification covers loader/reward integration and reviewed controls, not every source reference's scientific or mathematical correctness. Investigate observed training failures without retroactively selecting rows by policy success. |
