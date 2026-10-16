# Report-only decon adapter

`report.py` wraps an independently built, pinned decon 0.3 executable. It does not
implement a matcher, remove data, or make release, quality, or legal decisions.
Every receipt says `report_only` and `needs_review`, including zero-candidate runs.

Provide uncompressed JSONL files containing `messages`. The adapter joins nonempty
system and user contents in conversation order with two newlines, normalizes CR/LF
to LF and Unicode to NFC, and trims each included message. Assistant and tool
contents are excluded. Each row needs a nonempty user message; malformed and blank
rows abort instead of shifting row numbers. Candidate rows use **zero-based physical
source row numbers**, source ordinals, original-line and normalized-text SHA256s,
and reference eval keys/indices. Original source IDs and paths are not exported.

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
The supplied repository commit is a provenance assertion by the caller; establish
its relationship to the binary in the external build record.
The receipt also hashes the adapter source itself (`adapter_sha256`) and records
the four fixed CLI overrides, binding the normalization implementation and matcher
settings alongside the binary, config, reference and source file hashes.

The work directory must be new and outside Git. It is created with mode `0700`.
Pinned binary, config and reference snapshots, normalized training inputs, row
mapping, raw matcher reports and logs stay there. **These raw artifacts contain
text and are private.** Only `receipt.json` and `candidates.jsonl` are designed for
review without text or input paths; review their eval identifiers before sharing.
Stdout contains only the aggregate receipt, never raw matcher output or exceptions.

The command always supplies fresh training/reports directories, an explicit
reference directory, `--content-key text`, and `--index-answers false
--index-passages false`. It never supplies `--purify` or changes the source files.
The wrapper requires exactly one complete decon summary table. Training count must
match every staged row; examined and indexed reference counts must both match the
entire manifest. Candidate report count must equal the matcher match count. An
error or tokenization-failure diagnostic line in either log, a nonzero matcher exit,
or any count mismatch makes coverage incomplete even when decon itself returns zero.

Decon can skip short training prompts (fewer than twice its n-gram size in tokens)
and filter or deduplicate short/reference questions. Such skips cause exit `2`;
they are not evidence of absence of overlap. Thresholds, supported language,
reference selection, and short-prompt review need separate audits. There is no
claim of semantic, translated, response-only, or unprovided-reference coverage.

Exit codes: `0` means complete reported counts and no candidates; `1` means complete
reported counts with candidates; `2` means invalid input, execution failure, or
incomplete/unknown coverage. All three outcomes still require review. Release
freezing, source/teacher terms review, other contamination audits, and candidate
disposition remain external to this adapter.

Run isolated tests without repository pytest fixtures:

```sh
python -m unittest discover -s scripts/data/decontam -p 'test_*.py'
```

Tests use a fake executable and a planted literal to verify adapter extraction,
row mapping, privacy and failure behavior. They do not validate decon's matching
algorithm or replace a pinned real-executable smoke run.
