# MILES run configurations

Use the [structured examples](examples/README.md) for new runs, with the
[workflow](../../docs/miles/workflow.md), [launch guide](../../docs/miles/launching.md)
and [current support matrix](../../docs/miles/feature-parity.md). Inspect `plan`
before submission. Examples are starting recipes; recorded qualification applies
to its exact model, image, hardware and configuration.

For the measured 1-, 2- and 8-GPU trainer family, use the
[throughput starting profiles](../../docs/miles/throughput-profiles.md). The older
low-level recipes below have different batch, packing and publication choices;
they are not interchangeable throughput baselines.

| Directory / file | Role |
|---|---|
| `examples/` | Researcher TOMLs with model, data, allocation and training settings; supports plan/validate/run/status |
| `opd/` | Explicit Core MoE/dense and native Megatron OPD examples; see the [OPD guide](../../docs/miles/opd.md) for model and tokenizer boundaries |
| `profiles/` | Low-level prepared-input Core/MILES files; plan/validate/train, no configuration-driven allocation/preparation |
| `qualification/` | Frozen bounded exercises, often account-specific paths; read their measurement before reusing |
| `proposals/` | Future recipes that can reference unprepared assets; not ready to submit |
| `dense.toml`, `prompts.jsonl`, `verifiers.json` | Low-level schema/prepared-data examples, not full-model qualification |

Low-level profiles include tiny resident development, EP2 full-SFT synchronous
and bounded async variants. The structured async example requests an EP8 trainer
node plus eight inference GPUs and uses packing, TIS and 8 prompts × 8 responses.
These are different topology/recipe choices, not aliases for an identical run.
For exact values see [generated example summaries](../../docs/miles/configuration.md#example-recipes).

Set client concurrency, engine admission, decode graphs, KV tokens and recurrent
state capacity together. See [capacity](../../docs/miles/topology.md) and
[length controls](../../docs/miles/long-sequences.md). Core stays resident in
colocation; do not put a full SFT MoE checkpoint into a tiny profile or copy its
memory fractions into dense Olmo 3. Dense and MoE KV/optimizer budgets differ.

Full-weight audits run at startup/resume in the training examples; periodic
probes are disabled. Evaluation shares engine admission settings and its latency
depends on actual response lengths. Old admission/evaluation targets are retained
in [measurements](../../docs/miles/measurements/index.md), not promised timings.

Copy personal configurations into Git-ignored `runs/` or outside the checkout.
Start with `examples/grpo-sharing.toml` for the tested three-GPU colleague setup. Replace account/model/output paths and select a compatible immutable
image. Keep historical experiment TOMLs unchanged when adjusting a starter.
