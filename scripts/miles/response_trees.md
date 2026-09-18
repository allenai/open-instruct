# Comparing response prefixes across checkpoints

`response_trees.py` extends the September 12, 2026 Avery word-trie experiment
(recovered in `.artifacts/two-node-async-20260912/`). It reads existing scored
outputs, preserves their verdicts and full text, and writes an interactive HTML
fragment plus an inspectable JSON file. No model inference or external requests
are performed. The renderer lives in `assets/response_trees.html`.

```bash
python scripts/miles/response_trees.py --manifest /path/to/manifest.json --output /path/to/output
uv run --no-sync pytest -q scripts/miles/test_response_trees.py
```

A minimal manifest (paths resolve relative to the manifest):

```json
{
  "title": "Checkpoint comparison",
  "samples": 8,
  "labels": {"before": "Step 0", "after": "Step 200"},
  "contract_label": "8 samples per checkpoint; temperature 1; 8192-token cap",
  "cases": [{
    "id": "Mbpp/477",
    "title": "Lowercase function",
    "tag": "Function-name compliance",
    "prompt": "The exact task prompt",
    "rule": "Describe the actual verifier, including what it does not check.",
    "default_view": "code",
    "before": "before-scored.jsonl",
    "after": "after-scored.jsonl"
  }]
}
```

For multiple prompts, use `"prompts": "selected.jsonl"`, with one `id`/`query`
record per line, instead of per-case `prompt`. Optional `contracts.before` and
`contracts.after` paths load the learning-probe generation contracts: the builder
checks tokenizer, chat template, serving flags, dataset hash, sampling seed and
parameters, number of samples and response cap. It records input SHA256 hashes
and full contracts in the output JSON. Without contracts, decoding comparability
is the caller's responsibility; the builder still checks per-sample prompt hashes,
prompt token counts and caps when available.

Input scored records use `id`, zero-based `sample`, binary `score`, `tokens`,
`response`, and optionally `answer`, `code`, `finish`, `cap`, `prompt_hash`,
`prompt_tokens`. The original GSM8K format (`id` plus `sampled` records containing
`text`, `correct`, `tokens`) is also accepted. Legacy records do not supply a final
answer extractor, so only the generated-text view is offered. Samples must have
IDs 0 through `samples - 1` in each arm. Displayed sample numbers are one-based.

The gallery switches between 12-, 28-, and 60-word prefixes of generated text,
extracted final answers, or graded code. Code indentation is normalized **only in
the word trie**; the full-text inspector preserves it. Each leaf button opens the
unaltered response and its graded extraction. Identical displayed prefixes retain
all their samples, even if later text or verdicts differ. An empty extraction
appears at the root. Prefix cutoff, actual text ending, hitting the generation cap,
and having no extracted final answer are distinct conditions.

## What the fork statistics mean

- **H** is empirical branching entropy in bits, computed from the number of
  sampled responses taking each next-word branch. Responses ending at a fork form
  an additional branch. A one-child chain is compressed without changing the
  underlying split. Fill intensity runs from 0 to 3 bits for eight samples.
- **IG** is empirical mutual information between that local branch and the
  recorded binary verdict, conditional on reaching the node. The ring width
  increases with IG. Support `n` is shown at every fork.
- All-pass and all-fail nodes have IG zero, even when their text is diverse.
  Separating eight samples into singleton branches can yield large IG by chance.
  This is neither a causal decision point nor a measure of the model's full
  next-token distribution or policy entropy. It is not a measurement of how many
  bits the optimizer learned. No counterfactual interventions are performed.
- The checkpoints were sampled separately under the same generation seed and
  decoding contract. Matching sample numbers are not matched reasoning trajectories. Per-question `6/8` is the observed count of successful
  attempts, **not** a benchmark pass@8 percentage; the problem-level pass@8 event
  is simply whether at least one attempt succeeds.
- Use explicit verifier labels. An instruction-format pass need not be coherent
  or factually correct. Exact-string math grading can reject a numerically correct
  answer, and code tests can reject a wrong function name. Never silently change
  the evaluator to make the gallery look better.

The recovered renderer displayed only the first sample at a terminal prefix and
used displayed leaf counts as totals. The recovered builder omitted terminal
mass at mixed stop/continue forks. Both issues are handled here and covered by
focused tests. The recovered originals remain unchanged.
