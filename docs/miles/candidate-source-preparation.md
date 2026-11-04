# Preparing candidate sources

Use one prepared artifact for both MILES training and `miles screen`. The
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
python -m open_instruct.miles.datasets.candidate_sources plan.json \
  --tokenizer /path/to/checkpoint --output /path/to/new-bundle
```

Inputs may be raw rows or hygiene wrappers containing `id`, `raw_record` and
`provenance`. Preserve the prior hygiene decision in the input selection and
receipt; pointing at a receipt does not independently certify a new input's
cleanliness. The adapter does not select on policy reward, reserve screened
prompts, cap pool size, or rewrite messages with a new answer suffix. It records
all unsupported rows in `dispositions.jsonl`. Long prompts remain in the pool:
set the same adequate context limit for screening and training, or explicitly
name and review a shorter-context variant. Repeated prompts are counted, not
silently removed; resolve conflicting targets before qualification.

| Adapter | Target and scope |
|---|---|
| `nemo_v1` | Structural dispatch for formatting, structured formats, native IF, calendars and MCQA; unknown routes are reported |
| `structured_v1` | `schema_type`/`schema_str`; explicit schema policy, all five formats |
| `instruction_v1` | Native instruction IDs and kwargs, including bundles in verifier metadata or args |
| `calendar_v1` | Full final calendar feasibility from `exp_cal_state`, without a 20-event cutoff |
| `format_v1` | Existing format verifier payload; formatting only |
| `litmus_v02_v1` | Row-carried regex, answer type and matching policy; all scalar properties/methods are representable |
| `reasoning_gym_v1` | Saved entries for the ten qualified families, including ARC/reARC; no regeneration or difficulty reset |
| `mcqa_v1` | Displayed options and expected answer, existing final-choice parser; no science-only filtering |
| `math_v1` | Explicit symbolic-math source contract; proof requests are reported as requiring judged support |
| `candidate_v1` | Existing numeric, exact-text/JSON, countdown and knights contracts in `target` |

Litmus preserves row-specific answer formats and tolerances. The versioned
`litmus_scalar_v1` contract deliberately requires complete finite numeric or
explicit boolean captures; it does not recover an incidental number from malformed
text or treat arbitrary nonzero numbers as true. A row needs an explicit
one-capture `output_regex`. The matching rules are `isclose`, rounded-integer
`exact`, absolute/relative windows, boolean equality and normalized string
equality. This expands *verifier wiring*, not independent scientific validation:
new property/method strata still need reference checks and positive/near-miss
controls before qualification. Tool-dependent chemistry remains unsupported by
the single-turn screen even when its final scalar can be graded.

The catalog initially says `prepared`, not `qualified`. It cannot be resolved by
the named-source resolver until reviewed controls and the training loader path
have passed. Integration readiness, representative policy evidence and downstream
training benefit remain separate fields of evidence. Historical screens from
changed populations must not be relabeled as representative of these bundles.
