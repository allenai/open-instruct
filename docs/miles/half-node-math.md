# Half-node math benchmarks

These two Olmo 3.5 Tiny examples use one trainer and three inference B300 GPUs.
Background evaluation requests an additional GPU per task group and can overlap
training. Both use the text-only midtraining step-9708 export from #2195, requiring
internal Weka access. The application baseline includes merged #2229; both recipes
retain PPO+TIS, not score centering.

| Recipe | Training | Evaluation | Response budget |
|---|---|---|---|
| [GSM8K](../../configs/miles/examples/tiny/half-node-gsm8k.toml) | All 7,473 original GSM8K examples | Full GSM8K Platinum chat, one sample/problem | 4,096 |
| [Dolci](../../configs/miles/examples/tiny/half-node-dolci-math.toml) | Pinned Dolci-RL-Zero-Math-7B, excluding exact AIME overlaps | AIME 2024 and AIME 2025 separately, 32 samples/problem | 16,384 |

Both specify 1,000 updates, 8 prompts × 8 responses, LR 1e-6, seed 17,
maximum policy lag 4, response-averaged loss, routing replay, and zero reward
for truncated responses. Native checkpoints are saved every 50 updates with two
retained. Evaluation runs at update 0, every 100 updates, and finally. Evaluation
HF snapshots have a separate lifecycle from native checkpoint retention.

The GSM8K recipe preserves the historical 2K prompt / 4K response limits and
evaluation temperature 1.0 / top-p 1.0. Dolci proposes 2K / 16K limits and
evaluation temperature 1.0 / top-p 0.95. Its 18K serving context and reduced
request concurrency need GPU memory and throughput qualification before launch.
Neither refreshed recipe has new GPU validation. Historical #2195 results qualify
only their recorded source/image, not this updated main-based implementation.

These non-thinking benchmarks explicitly grade the full response and do not
require `</think>`. That preserves #2195's grading behavior and differs from the
thinking-policy defaults in the other examples. GSM8K keeps its numeric verifier;
Dolci uses the symbolic math verifier. The boxed instruction is not an extra
format reward.

## Prepare data

Run preparation in an environment with `datasets`, `transformers`, and access to
the exact learner checkpoint. This is CPU data preparation, not a training launch.
Use the same tokenizer/checkpoint for preparation and training:

```bash
MODEL=/weka/oe-training-default/jasonr/olmo35-vision-alignment/exports/olmo35-small-hero-14T-dec600-mt-lc-256k-textonly/hf-step9708-local
DATA=/weka/oe-adapt-default/YOUR_USERNAME/open-instruct/data
uv run --no-sync python -m scripts.miles.prepare_boxed_math \
  --dataset gsm8k --tokenizer "$MODEL" --output "$DATA/boxed-gsm8k"
uv run --no-sync python -m scripts.miles.prepare_boxed_math \
  --dataset dolci --tokenizer "$MODEL" --output "$DATA/boxed-dolci"
```

The script pins source revisions and adds this instruction to the user question:

> Please reason step by step, and put your final answer within \boxed{}.

It strips reference assistant solutions, preserves preceding system instructions,
and applies the learner's own chat template once. An already-appended identical
instruction is normalized rather than duplicated. No `qwen_math` trainer wrapper
or Qwen chat template is needed. The outputs are `train.jsonl`, `verifiers.json`,
`chat_template.jinja`, and an audit `manifest.json`. Supply the first two through
`data.prompt_data` and `data.reward_config`; the audit manifest is not an
`rl_manifest`. Template hashes and token IDs allow runtime compatibility checks.
Remote tokenizers require an immutable `--tokenizer-revision`. Existing output
directories and overlong prompts fail rather than being overwritten or truncated.

GSM8K keeps the named-task seed-17 ordering. Dolci checks training questions
against the pinned `allenai/aime-2022-2025` 2024/2025 questions, removes exact
matches after whitespace normalization, and records original row indices. The source audit found 13,314 Dolci rows, all retained, and 30 questions per
AIME year with no whitespace-normalized exact overlap. All 7,473 GSM8K rows
also passed schema/target preparation. This audit used a test renderer; it does
not establish the learner-tokenizer length limit or GPU readiness. This
is an exact-overlap check, not a semantic contamination audit. The evaluator's
dataset revision is controlled by its image/cache separately; verify that its
actual questions match this audit source before reporting benchmark results.

## Evaluation and launch checklist

The pinned evaluator supports `aime_2024:pass_at_32` and
`aime_2025:pass_at_32`, both with the boxed-answer instruction. Its formatter uses
one newline before that instruction; training retains #2195's two-newline
separator. The wording and answer convention match, but prompts are not
byte-identical. Report each year's pass@1 estimate from 32 draws separately;
retain pass@4/8/16/32 as supplementary metrics, not substitutes for pass@1.
Keep full coverage, per-question predictions, sampling settings, and evaluator
revision with results. No evaluation limit is set.

- [x] Replace the proposed `qwen_math` option with standalone data preparation.
- [x] Base the refreshed examples on main after #2229.
- [x] Add separate Dolci training and AIME 2024/2025 configuration.
- [ ] Prepare with the actual Weka learner tokenizer and inspect rendered examples.
- [ ] Verify evaluator question coverage against the pinned overlap-audit source.
- [ ] Review proposed Dolci hyperparameters/resources before launch.
- [ ] Build a compatible committed application image on Beaker and run GPU qualification.
- [ ] Establish persistent monitoring ownership, registry, watchdog, and incident alerts.
- [ ] Run benchmarks; record routine progress/results in the dashboard and W&B.

Copy the chosen template to ignored `runs/`, set account paths, credentials and
resource settings, then run `python -m open_instruct.miles plan` and `validate`.
Use the [launch guide](launching.md) for committed-image execution. Plan/validate
checks do not prove GPU fit or learning quality. Keep all preparation receipts,
resolved settings and historical results separate from new run results.
