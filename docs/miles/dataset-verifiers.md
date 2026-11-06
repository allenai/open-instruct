# Dataset reward verifiers

Datasets need runtime rewards that implement their declared answer
contracts. The structured-output verifier lets MILES train on JSON, YAML, TOML,
XML and CSV using its existing registered reward path. This guide explains the
supported contracts, runtime dependencies and limits needed to prepare and
validate these training sources.

## Structured output sources

Install `scripts/data/rl_readiness/requirements.txt` in the training
image before using optional dataset verifiers. Registry factory
`open_instruct.miles.rewards.structured_outputs.StructuredOutputVerifier` accepts
a target with `format` (`json`, `yaml`, `toml`, `xml`, `csv`), `schema`, an explicit
`schema_policy` (`as_declared` or `nemo_strict_v1`), and optional `coerce_types`.
These rewards measure structural compliance, not factual correctness. Preserve
original messages; no JSON-only suffix is needed for non-JSON sources.

`nemo_strict_v1` and XML/CSV type coercion use the pure helpers extracted from
NVIDIA NeMo Gym's Apache-2.0 `resources_servers/structured_outputs/app.py`
(SHA256 `5e9d85edaa6743abcdf0fd80859321bb9ce386e461846399069e80bb25cbbc70`).
Schema strengthening walks dictionary-valued schema positions, leaving property
maps and literal defaults intact. It preserves native NeMo list boundaries:
combinator and tuple-item list branches are not independently strictified.
Closing separate `allOf` branches can make a satisfiable schema impossible.
Validation and regex/resource checks still inspect all schema branches. Nested `$id` resources may resolve
fragment-local references; the registry never retrieves remote resources.
CSV boolean subschemas are left unchanged for schema validation rather than type
coercion. YAML mapping keys must be strings to match the JSON Schema data model.
Unlike the legacy JSON-only slice, the contract does not silently strengthen a
schema: the policy is part of every target and therefore the artifact hash.
Duplicate mapping keys, YAML aliases, nonfinite values, external schema references
and XML entities are rejected. Local references and regex schema keywords are supported.
A worker process and a configurable response deadline (`timeout_seconds`, default
1 second) bound parsing and schema regex matching together. Timeouts always
return zero with `status: timeout`, the error and elapsed time in diagnostics,
including the outer worker timeout. There is no timeout policy setting. The
response budget includes target validation; slow targets can exhaust it. Invalid schema/configuration exceptions propagate, as do unexpected worker failures.
The old `JSONSchemaVerifier` remains unchanged.

After successful parsing and target checks, instance validation can still exceed
Python's recursion or numeric limits (for example, deeply nested recursive arrays
or a huge integer with a fractional `multipleOf`). `RecursionError` and
`OverflowError` from instance validation receive zero with `status: validation_error`,
`error_type`, `error`, `failure_policy: zero`, and `elapsed_seconds` diagnostics.
This fixed policy records a validator
limit, not a completed schema mismatch. Target validation, validator construction,
and unresolved-reference errors remain fatal.

XML preserves its root element: `<root><a>1</a><b>2</b></root>` needs a schema
whose top-level `properties.root` contains the `a` and `b` properties. Coercion
applies only to XML/CSV; `coerce_types` has no effect on JSON/YAML/TOML. Answers
must contain the format itself, without Markdown code fences.

YAML and TOML values must be JSON-compatible after parsing. Quote dates and times
when the schema expects strings: `date: "2024-05-01"` in YAML and
`date = "2024-05-01"` in TOML. Unquoted native dates/times become Python temporal
objects and receive a `parse_error` score of zero; they are not implicitly
converted to strings. PyYAML's YAML 1.1 resolution also treats unquoted
`yes`/`no`/`on`/`off` as booleans. Quote those words for string fields; leave them
unquoted only when boolean values are intended. Qualification controls should
use the same representation expected in training responses.

With `as_declared`, use `required` and `minItems` where the task demands nonempty
objects or arrays: an empty TOML document or header-only CSV can otherwise satisfy
the schema. `nemo_strict_v1` requires all declared object properties and does not
add array length requirements; qualification controls must validate each target's
intended behavior. The qualification script uses the default one-second budget.

## Native instruction and calendar sources

`open_instruct.miles.rewards.nvidia_constraints.InstructionVerifier` uses the
maintained `IFEvalG` instruction registry. Targets have `family="nemo"` and a
`target` object with parallel `instruction_id_list` and `kwargs` arrays. The
optional `grading_mode` is `all` (default) or `fraction`. Unknown modes, invalid
argument types, and native argument repairs are errors. Numeric counts/indices
must be integers, text arguments must be strings, and keyword collections must
be lists of strings. Target validation runs even when the response is empty. The
adapter rejects zero parameters that still trigger native random defaults. Copy
indices now preserve explicit zero and include the end index; admission checks
that `0 <= n_start <= n_end < len(prompt_to_repeat)`. Grading must not depend on a
worker's RNG seed.
Known constructor normalization is accepted: lowercasing the paragraph's required
first word or a frequency checker's letter, and trimming the specific text
arguments that the native checker itself trims. For example, `first_word: "The"`
and `letter: "A"` retain their native rewards. This is not blanket normalization:
keyword-list entries retain their whitespace, and a padded letter such as `" A "`
is rejected because the native checker randomly replaces it before trimming.
Other argument changes, including numeric repairs, remain fatal. Regex validation
uses the normalized pattern that the native checker will actually evaluate.
Native regex arguments are compiled during admission and invalid patterns are
fatal target errors. Regex matching semantics are retained, rather than treating
keywords as literal text. The adapter applies the existing IF training admission
policy, including rejection
of impossible paragraph positions and tiny total-letter limits. It admits the
corrected lowercase-repetition and Unicode letter-counting checkers. Language
classification and English case-plus-language families remain deferred pending
calibration; this is not a claim that every native family has passed controls.

`CalendarVerifier` in the same module validates the full final JSON calendar
against `exp_cal_state`: event IDs, durations, non-overlap, windows and explicit
before/after/at/between constraints. Empty target calendars are supported. This
version scores schedule feasibility, not event-name paraphrase or dialogue
quality. Both adapters use the isolated verifier worker path. Their extraction
uses the final answer after reasoning, like the other dataset verifiers.
Calendar JSON parsing failures, including excessive nesting, score zero with
`parse_error`, `error_type`, and `failure_policy: zero` diagnostics. Calendar target
errors remain fatal and are checked before response parsing.
Times accept 24-hour notation or case-insensitive AM/PM (for example, `09:00`,
`9am`, and `9:00 AM`). Calendar answers must be raw JSON; Markdown code fences
are rejected. Schedule feasibility is evaluated within this response contract.

Both adapters return zero when the isolated worker exceeds its 45-second
deadline, with `status: timeout`, `failure_policy: zero`, `error`, and
`elapsed_seconds` in diagnostics. There is no timeout policy setting.
Other worker/configuration errors remain fatal. Diagnostics consistently identify
`calendar_v1` or `nemo_if_v1`, with `status: completed` for ordinary scores.

Before training, qualify every prepared row through
[`scripts/data/rl_readiness/qualify.py`](https://github.com/allenai/open-instruct/blob/main/scripts/data/rl_readiness/qualify.py).
It invokes the actual registered reward on each row's positive, negative, empty,
and truncated controls. Target errors defer the row, and timeout fallbacks cannot
pass qualification. Use only the resulting qualified catalog; this guarantee
applies to those rows, not to the entire raw dataset. The generic run-data IF
screen uses another target schema and does not validate these native adapters;
manually supplied rows that bypass qualification may fail when first scored.

Reproducible integration controls live in
[`open_instruct/test_miles_nvidia_constraints.py`](https://github.com/allenai/open-instruct/blob/main/open_instruct/test_miles_nvidia_constraints.py).
Run `uv run pytest open_instruct/test_miles_nvidia_constraints.py -q`. They cover
native scoring, deterministic admission, regex errors, timeout policies, calendar
formatting, diagnostics, and real subprocess execution. These integration checks
do not qualify an entire raw dataset.


The `keywords:word_once` identifier retains its historical exactly-once regex
matching. The optional Dolci preparation correction uses the existing
`keywords:frequency` checker with `frequency=1` and `relation="at least"`; it
applies only to audited alphabetic keyword shapes. NeMo preserves its supplied
identifiers and regex-validation rules. Standard raw-dataset training paths do
not opt into the Dolci correction automatically.

Shared copy and bigram-wrapping fixes apply to both standard IF and NeMo. Wrapping
requires complete wrapped word pairs, optionally followed by one wrapped final
word for odd word counts (`<<hello world>> <<again>>` or just `<<hello>>`). Unwrapped
trailing text and earlier singleton groups are rejected. NeMo's `all`/`fraction`
aggregation modes are unchanged. See the
[explicit Dolci preparation correction](source-preparation.md#explicit-dolci-keyword-correction).
