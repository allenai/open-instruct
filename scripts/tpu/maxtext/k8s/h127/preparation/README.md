# H127 preparation and reference checks

These scripts qualify the C11–14T original-versus-uniform-presentation SFT pair.
They do not submit training jobs. The monitor can deactivate only its named
JobSet on a verified numerical or pod failure. The scientific overlays remain blocked until
their numerical, data, checkpoint, and export checks have receipts.

Use the image digest pinned by the adjacent overlays. Host orchestration stages
the scripts at `/work` and mounts the prepared corpus read-only at `/corpus`.

- `prepare_full_pair.py`: verify the input checksums and unchanged non-user
  messages, then globally shuffle the paired corpora with identical document
  order. Produce 512 Parquet shards per arm and a complete manifest.
- `run_full_replays.py`, `replay_batches.py`: reproduce steps 1–20 on each of
  eight data hosts, for both full corpora. Use four workers, two CPU cores each.
  The registered production placement maps logical data host `h` to process
  `4*h`, with two real rows and two padding rows per host. The complete batch
  hash covers every batch key. The first-batch NPZ retains the two real rows.
  These are expected hashes; production still has to match them.
- `full_reference_ce.py`: run on one B300 with the source checkpoint's WEKA
  mount. Stage the replay under `/probe/full-replay`, and the registered
  `math-opener-panel.jsonl.gz` and `newline-token-family.json` under `/probe`.
  It checks the source HF embedding/head against native BF16 weights, applies
  the same two atomic-row initializations as the converted base, and scores
  each arm's exact first batch. It then measures the shared step-zero opener
  distribution on 1,000 fixed prompts, including 16 identical-forward repeats.
  No optimizer updates or sampled generation occur. Results go to `/output`.
- `check_mesh_shapes.py`: offline topology and data-host mapping inspection for
  the candidate and alternative 128-chip layouts. Shape acceptance is not
  runtime or numerical qualification.
- `aot_memory.py`: stage the production kustomization as
  `/work/control-config.yaml`. This estimates accelerator memory using the
  pinned image's **pretraining** compiler. It does not establish Tunix SFT
  memory use or parity; live preflight is still required.
- `export_bounded.py`: use the established exporter pinned at open-instruct
  `268c124abd07c1f1addc83a47729b79ab53aed5c`, with layout views instead of duplicate
  FP32 copies and bounded Orbax restore concurrency. `Dockerfile.export` adds
  CPU PyTorch 2.11.0 to the pinned checkpoint environment. The wrapper accepts
  captured HF shape metadata, keeps the padded 100352-row vocabulary, and copies
  the registered atomic tokenizer. `--check-base` requires every exported BF16
  tensor hash to equal the independent seeded-HF reference. Generation defaults
  use T=0.6/top-p=0.95 with top-k disabled and end-of-text/im-end stop IDs; the
  thinking-close token does not stop answer generation.
  `test_export_bounded.py` compares byte-identical safetensors exports with the
  unchanged exporter on base and wrapped Tunix fixtures, including scanned
  layers and unused RNG state. This fixture check is not the full-base check.

The replay extension reproduced all 20 previously validated smoke-batch hashes
and loss-token counts for host 0. The full-reference model loading and atomic-row
initialization are unchanged from the completed smoke HF reference. Full-data
reference values must be measured afresh, rather than copied from that smoke.

Training uses the full 2,253,684-record corpus per arm. The 32,768-record smoke
subset is for infrastructure only. Changed prompt lengths can change packed
boundaries even though paired document order and assistant messages match.

Resume qualification uses `control-resume50/` only after the initial 100-step
smoke succeeds. Copy its **complete step-50 training checkpoint**, including
optimizer state, to the new resume run's `checkpoints/50/` prefix. Verify source
and destination object sets, sizes and CRC32C checksums before submitting; never
point the resume test at the original smoke's output prefix. The same recipe,
image and data continue from update 50 through 70. `resume_audit.py` wraps only
checkpoint restoration: each host checks all installed local model and optimizer
leaves bitwise against the restored values, and requires optimizer step 50.
`test_resume_audit.py` exercises a real small Flax/Optax/Tunix checkpoint and
rejects a weights-only checkpoint. This test is not a TPU resume receipt.

`monitor_tpu.py CONFIG.json OUTPUT` never submits or restarts a job. Supply `job`,
`steps`, `expected_batches` (host, step, sha256), and for fresh runs `reference_ce`
and `reference_tokens`. Resume configuration additionally uses `first_step=51`
and `resume_step=50`; expected hashes are the uninterrupted smoke's batches
51–70 on all eight data hosts. It preserves retrieved log lines across rotations,
checks every loss/drop step, rejects conflicting duplicate records, and requires
32 restoration audits for a resumed run. A watcher timeout leaves the workload
untouched for inspection. `test_monitor_tpu.py` verifies rejection of incomplete
or incorrect evidence. Resume loss differences from the uninterrupted smoke
remain diagnostics, not an arbitrary statistical pass/fail gate.

Scientific measurement uses `measure_openers.py` on the verified HF export at
step 512 or 2048. It accepts only the frozen primary panel or its historical
four-paraphrase diagnostic expansion, verifies the export weight hash, and uses
the same single-prompt BF16 forward / FP32 softmax calculation as the completed
step-zero reference. It makes no token-row changes. Each panel repeats 16
original prompts and records raw category probabilities, prompt hashes, model
metadata hashes and checkpoint identity. The four diagnostic paraphrases never
replace the primary original suffix. This code is prepared; runtime validation
on a trained export remains pending.

`compare_openers.py CONTROL.jsonl TREATMENT.jsonl OUTPUT.json` joins by qualified
question ID and variant, requires all 1,000 matched prompts at the same step,
and reports treatment-minus-control paired bootstrap intervals (10,000 samples,
seed 61023). The primary read is the original-suffix newline-family delta at
step 2048. Other steps, categories and suffixes remain diagnostics. The tests
reject missing/duplicate questions, mismatched prompts and different checkpoints,
and check the comparison sign and invariance to record order. No training-seed
floor or confirmatory claim is supplied by these question-bootstrap intervals.

Secondary MATH quality uses `generate_math.py` at step 2048: the original
1,000 prompts, one sampled trajectory each, T=0.6/top-p=0.95/top-k disabled,
32,768 generated tokens maximum, and end-of-text/im-end stop IDs. Both tags
remain atomic; no first-word forcing, newline mask, or suffix change. Use the
pinned Olmo SGLang image/runtime in the evaluation protocol; first qualify a
small pilot on the trained export. Shards are disjoint slices of the frozen
panel, and caches are bound to weights, metadata, runtime, code and sampling.
Per-question seeds are requested, without claiming identical random draws.

`score_math.py` uses the pinned historical MATH harness
`allenai/olmo-eval@6f4cbee81ffbd79b210b92d6adb7e37744613020`. It requires a complete
1,000-question union, the exact frozen prompt IDs, matching checkpoint/arm and
sampling, and checks all generation-file hashes. It reports historical accuracy
and explicit-final accuracy alongside cap hits, closing, extraction failures and
lengths. Six archived historical cases passed in the pinned CPU scorer image.
No generated quality results exist yet; serving qualification remains pending.
