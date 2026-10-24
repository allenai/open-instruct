# H127 preparation and reference checks

These scripts qualify the C11–14T original-versus-uniform-presentation SFT pair.
They do not submit training jobs. The scientific overlays remain blocked until
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
