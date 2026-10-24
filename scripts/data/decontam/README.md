# Report-only decon adapter

`report.py` wraps an independently built, pinned decon 0.3 executable. It does not
implement a matcher, remove data, or make release, quality, or legal decisions.
Every receipt says `report_only` and `needs_review`, including zero-candidate runs.
It is separate from the older Elasticsearch-based overlap scripts in `decontamination/`.

Provide uncompressed JSONL files containing `messages`. The adapter joins nonempty
system and user contents in conversation order with two newlines, normalizes CR/LF
to LF and Unicode to NFC, and trims each included message. Assistant, tool and
environment (Olmo chat templates' tool-output role) contents are excluded; any
other role aborts the run. Each row needs a nonempty user message; malformed and blank
rows abort instead of shifting row numbers. Candidate rows use **zero-based physical
source row numbers**, source ordinals, original-line and normalized-text SHA256s,
and reference eval keys/indices. Each row is one (training row, eval item) pair;
`positions` counts decon matches at different offsets of that pair. Original source
IDs and paths are not exported.

```sh
python scripts/data/decontam/report.py \
  --source /data/input.jsonl \
  --decon-binary /tools/decon --decon-binary-sha256 FULL_SHA256 \
  --decon-repo-sha FULL_40_CHARACTER_COMMIT \
  --config /data/detect.yaml --config-sha256 FULL_SHA256 \
  --evals-dir /data/references --reference-manifest /data/reference-pins.json \
  --work-dir /private-output/new-run
```

The reference manifest is a JSON object mapping **every** relative reference file
name to its byte SHA256. Only `.jsonl` and `.jsonl.gz` references are supported;
each row must have a nonempty `question`, an `eval_key`, and an unsigned 64-bit integer
`eval_instance_index`, and a globally unique integer `doc_id` in `0..4294967294`
(decon reserves headroom for an internal ID).
Literal JSON `-0` integers are rejected to avoid decoder-dependent identity changes.
Eval key/index pairs must also be unique across the reference set. Independent
exports often restart `doc_id`; resolve those collisions before running this adapter.
Empty references, missing or extra files, symlinks, and changed hashes fail.
Reference `question`, `answer` and `passage` strings get the same CR/LF and NFC
normalization as prompts in the staged copies decon reads. The receipt hashes the
pinned and the staged reference bytes, and counts rows NFC changed on each side
(`non_nfc_rows`).
The supplied repository commit is a provenance assertion by the caller. The adapter
does not check the binary hash against it, so establish their relationship in the
external build record. `purify` stays off because this decon revision resets it from
the CLI whatever the config says; a different binary might not.
The receipt also hashes the adapter source itself (`adapter_sha256`) and records
the fixed CLI overrides, binding the normalization implementation and matcher
settings alongside the binary, config, reference and source file hashes.

The adapter reads `ngram_size` and `punctuation_chars` from the config, using decon's
defaults when absent, so it can mirror decon's training-row skip. Its reader accepts
only decon's flat layout: one `key: value` per line, numbers in decimal, and quoted
strings with JSON-style escapes. Anything else fails with `unsupported_config_syntax`.
It passes `--ngram-size` explicitly so the value it uses is the value decon uses.

The work directory must be new and outside Git. It is created with mode `0700`.
Pinned binary and config snapshots, normalized reference and training inputs, row
mapping, raw matcher reports and logs stay there. **These raw artifacts contain
text and are private.** Only `receipt.json` and `candidates.jsonl` are designed for
review without text or input paths; review their eval identifiers before sharing.
Stdout contains only the aggregate receipt, never raw matcher output or exceptions.

The command always supplies fresh training/reports directories, an explicit
reference directory, `--content-key text`, `--index-answers false
--index-passages false`, and `--ngram-size`. It never supplies `--purify` or changes
the source files. The wrapper requires exactly one complete decon summary table.

Decon skips some rows by design, and the receipt counts them in `coverage_limits`:
- **References.** Rows below `eval_min_token_length` tokens or
  `eval_min_unique_word_count` unique words, and `(eval_key, fingerprint)` duplicates,
  come from decon's index summary. Examined rows must equal the manifest's row count,
  and indexed plus skipped rows must equal examined.
- **Training prompts.** Decon skips prompts under `2 * ngram_size` tokens without
  reporting a count. The adapter attributes the gap between staged and processed rows
  to this rule only when the gap is at most the number of rows with fewer than
  `2 * ngram_size` words after decon's text cleaning. A cleaned word is always at
  least one token, so a longer row is never skipped by this rule. Both numbers are
  recorded.

Skipped rows were not checked for overlap: review short prompts and short or
duplicate references separately. Any other count mismatch makes coverage incomplete,
even when decon itself returns zero. So do an error, tokenization-failure or panic
line in either log, a nonzero matcher exit, and candidate positions that do not sum
to the matcher's match count. Thresholds, supported language and reference selection
need separate audits. There is no claim of semantic, translated, response-only, or
unprovided-reference coverage.

Exit codes:
- `0`: every row was checked or skipped for a counted reason, and there are no
  candidates.
- `1`: the same, with candidates.
- `2`: invalid input, execution failure, or incomplete or unknown coverage.

All three outcomes still require review. Release
freezing, source/teacher terms review, other contamination audits, and candidate
disposition remain external to this adapter.

Run isolated tests without repository pytest fixtures:

```sh
python -m unittest discover -s scripts/data/decontam -p 'test_*.py'
```

The default tests use a fake executable and a planted literal to verify adapter
extraction, row mapping, privacy, skip accounting and failure behavior. An optional
integration test runs a real executable on synthetic data only. The training side has
an exact planted question, an unrelated control and a short prompt. The reference
side has long questions, a fingerprint duplicate and a short question:

```sh
DECON_REAL_BINARY=/tools/decon \
DECON_REAL_CONFIG=/tools/decon-source/config/default.yaml \
DECON_REAL_COMMIT=FULL_40_CHARACTER_COMMIT \
python -m unittest discover -s scripts/data/decontam -p 'test_*.py'
```

All three variables are required; without them the real-matcher test is skipped.
The `Decon adapter` CI workflow builds the pinned public decon revision with its
lockfile using Rust `1.99.0` and Python `3.12` on an Ubuntu CPU runner and
enables this test. It prints only the
sanitized aggregate receipt, retaining binary/config/adapter/reference hashes and
coverage counts in the CI log. Raw prompts, matcher reports and logs are not
published. A new or pending CI job does not qualify the executable: require that
job to pass before claiming the smoke succeeded. This narrow smoke verifies the
adapter contract on synthetic data; it does not establish broad matcher recall,
full dataset coverage, or release readiness.
