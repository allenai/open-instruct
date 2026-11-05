# Dataset reward verifiers

Candidate datasets need runtime rewards that implement their declared answer
contracts. The structured-output verifier lets MILES train on JSON, YAML, TOML,
XML and CSV using its existing registered reward path. This guide explains the
supported contracts, runtime dependencies and limits needed to prepare and
validate these training sources.

## Structured output sources

Install `scripts/data/rl_readiness/requirements.txt` in the training
image before using optional candidate verifiers. Registry factory
`open_instruct.miles.rewards.structured_outputs.StructuredOutputVerifier` accepts
a target with `format` (`json`, `yaml`, `toml`, `xml`, `csv`), `schema`, an explicit
`schema_policy` (`as_declared` or `nemo_strict_v1`), and optional `coerce_types`.
These rewards measure structural compliance, not factual correctness. Preserve
original messages; no JSON-only suffix is needed for non-JSON sources.

`nemo_strict_v1` and XML/CSV type coercion use the pure helpers extracted from
NVIDIA NeMo Gym's Apache-2.0 `resources_servers/structured_outputs/app.py`
(SHA256 `5e9d85edaa6743abcdf0fd80859321bb9ce386e461846399069e80bb25cbbc70`).
CSV boolean subschemas are left unchanged for schema validation rather than type
coercion. YAML mapping keys must be strings to match the JSON Schema data model.
Unlike the legacy JSON-only slice, the contract does not silently strengthen a
schema: the policy is part of every target and therefore the artifact hash.
Duplicate mapping keys, YAML aliases, nonfinite values, external schema references
and XML entities are rejected. Local references and regex schema keywords are supported.
A worker process and a configurable response deadline (`timeout_seconds`, default
1 second) bound parsing and schema regex matching together. Timeout diagnostics
follow `OI_MILES_FORMAT_TIMEOUT_POLICY`, falling back to the math policy (`zero`
by default; `raise` preserves a scoring failure), including the outer worker
timeout. The response budget includes target validation; slow targets can exhaust
it. Invalid schema/configuration exceptions propagate, as do unexpected worker failures.
The old `JSONSchemaVerifier` remains unchanged.

After successful parsing and target checks, instance validation can still exceed
Python's recursion or numeric limits (for example, deeply nested recursive arrays
or a huge integer with a fractional `multipleOf`). `RecursionError` and
`OverflowError` from instance validation receive zero with `status: validation_error`,
`error_type`, `error`, `failure_policy: zero`, and `elapsed_seconds` diagnostics.
This fixed policy is independent of the timeout policy and records a validator
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
the schema. `nemo_strict_v1` retains the native helper's traversal and does not
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
adapter applies the existing IF training admission policy, including rejection
of impossible paragraph positions and tiny total-letter limits. It admits the
corrected lowercase-repetition and Unicode letter-counting checkers. Language
classification and English case-plus-language families remain deferred pending
calibration; this is not a claim that every native family has passed controls.

`CalendarVerifier` in the same module validates the full final JSON calendar
against `exp_cal_state`: event IDs, durations, non-overlap, windows and explicit
before/after/at/between constraints. Empty target calendars are supported. This
version scores schedule feasibility, not event-name paraphrase or dialogue
quality. Both adapters use the isolated verifier worker path. Their extraction
uses the final answer after reasoning, like the other candidate verifiers.
Calendar JSON parsing failures, including excessive nesting, score zero with
`parse_error`, `error_type`, and `failure_policy: zero` diagnostics. Calendar target
errors remain fatal and are checked before response parsing.

Existing control replay: 112 positive/negative pairs across 36 native instruction
families passed; one impossible paragraph constraint was rejected. The dedicated
unit controls include gold-containing/constraint-violating answers, Unicode
letters, per-word repetition, fractional versus all-constraints scoring, calendar
near misses and real subprocess execution. These are integration checks, not qualification of an entire raw dataset.
