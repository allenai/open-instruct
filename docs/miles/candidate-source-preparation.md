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

## Qualification and publication

Provide reviewed controls keyed to actual prepared sample IDs. Each stratum needs
at least one case set containing `gold` (expected 1), `wrong` (0) and `near_miss`
(0). A near miss should exercise the likely shortcut: a wrong number containing
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
the same prompt. Missing strata, failed controls, unresolved judge services, or
loader failures leave the source unqualified. Passing control rows never replace
the full source. Review all dispositions before using `--accept-omissions` to
qualify an explicitly accepted subset. This switch accepts the stated omissions;
it does not certify full dataset coverage. The report includes source/controls
hashes, loader manifest hash, context limit, judge registry fingerprint and all
missing strata. It does not claim that a GPU optimizer update was exercised.

Use the readiness Docker overlay on an application image built from this stack,
so training and screening receive the same optional verifier versions. Its build
checks the qualification, source-preparation and screening entry points. Use the
existing local toy-model coverage workflow for a GPU exercise, then run a matched
representative policy screen from the actual training configuration. Preserve
original source proportions; do not use the control set as a policy screen.

## Reference judges and routed blends

`reference_judge_v1` preserves a nonempty reference from `expected_answer` (or
explicit `options.reference_field`) and the full conversation in `judge_query`.
It uses the existing named judge marker `candidate_reference`. Configure that
name in the run's `[judging.bindings]` using the same approved service and rubric
for screening and training. Qualification requires a resolved registry and live
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
| Knowledge MCQA (C031) | Displayed choice validation and final-choice reward | Native-template extraction checks, chance floor in screen interpretation |
| ReasoningGym (C036) | Saved entries/difficulty for ten qualified families, including ARC | Additional families require scorer audits and adversarial controls; no procedural population replacement |
| Cascade Math, Nemotron Math v2, OpenMathReasoning, Math Stack Overflow, AceReason Math, DeepMath (C037–C040/C091/C094) | Explicit symbolic or reference-judge routing | Proof/judge parity and service calibration; canceled Nemotron Math run is not restarted |
| Nano, Cascade-2, Lightning, Super, Ultra blends (C043–C047) | Explicit per-agent routes with full omissions report | Native judge/hybrid contracts; unsupported code/tool/SWE/Lean routes remain deferred |
| AceCode (C076) | Existing source support remains unchanged | Code-verifier debugging stays paused pending its replacement |

The web-search candidate remains excluded at Robert's request. SWE, Lean,
browser/tool interaction and distillation-only records must not be presented as
single-turn end-to-end training support. They need an actual environment or
objective implementation. Licensing and scientific validity remain properties of
the source and its evidence; successful parsing is not either kind of approval.
