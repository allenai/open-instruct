# Structured MILES run examples

Start with the [MILES guide](../../../docs/miles/index.md) and
[launch instructions](../../../docs/miles/launching.md). Copy a TOML into the
Git-ignored `runs/` directory, replace YOUR_USERNAME and input/output paths, then
run plan and validate. Keep these tracked templates unchanged for other users.

```bash
mkdir -p runs
cp configs/miles/examples/grpo-sharing.toml runs/my-grpo.toml
# Edit name/output.root; retain the supplied read-only checkpoint for the first run.
python -m open_instruct.miles plan runs/my-grpo.toml
python -m open_instruct.miles validate runs/my-grpo.toml
export MILES_EXISTING_IMAGE=01M2F1RKZFZVJYAS0XQGEC3SEJ
python -m open_instruct.miles run runs/my-grpo.toml
```

The MILES GRPO starter uses four B300 GPUs, mixed-policy refresh and offline W&B. It needs Beaker login
and resource access, but no extra HF/W&B secret mappings for the supplied inputs.
Optional secret-name examples are commented at the end of each template.
`runs/` is excluded from Git and Docker build contexts; the launcher still carries
the selected TOML into the job. Relative input paths resolve from the copied TOML's
directory; prefer absolute WEKA paths for remote inputs.

Examples are editable starting points, not
historical measurement records or universal memory-fit guarantees.

| Example | Purpose |
|---|---|
| [grpo-sharing.toml](grpo-sharing.toml) | First colleague run: tested full-SFT checkpoint, EP2 + two engines, two updates, mixed-policy refresh/TIS and packing/replay |
| [grpo-basic.toml](grpo-basic.toml) | Tiny-model, one-GPU resident colocated execution check |
| [grpo-disaggregated.toml](grpo-disaggregated.toml) | EP2 trainer plus one dedicated TP1 engine, synchronous training |
| [grpo-async-disaggregated.toml](grpo-async-disaggregated.toml) | Production-shaped EP8 trainer node plus eight TP1 engines on another node; async/TIS and packing |
| [grpo-multitask.toml](grpo-multitask.toml) | Short GSM8K/math mixed-task preparation and evaluation exercise |

The first-run `grpo-sharing.toml` follows `small.toml`: 32 prompts × 4 responses
and global batch 128. The older `grpo-basic`, `grpo-disaggregated`,
`grpo-async-disaggregated` and `grpo-multitask` recipes keep 8 × 8 and batch 64
with barrier publication. Image selection does not change these recipe settings.
The full async example is 16 GPUs, not the older three-GPU comparison layout.
Use [generated example summaries](../../../docs/miles/configuration.md#example-recipes)
for exact current values. `profiles/` contains low-level prepared-input configs;
`qualification/` contains bounded exercise inputs. Neither is interchangeable with
a full workflow example. Preserve frozen experiment inputs when changing starters.

## Size-based throughput starters

Use `dev.toml` and `tiny.toml` for one-GPU and disaggregated tiny-model mechanics.
`small.toml` uses two full-model trainer GPUs; `large.toml` uses eight. Their
inference fleets and batch sizes follow the [measured throughput guide](../../../docs/miles/throughput-profiles.md).
These full-model examples use mixed-policy refresh and keep FIFO/lag-two semantics;
that remains experimental outside the qualified full-SFT GSM8K/B300 workload.
They are different recipes from the older packed async example above.

The throughput basket disables eval/saves/export and enables detailed replay
qualification. These researcher starters retain practical eval/save/export
settings and disable the extra route audit. Read the measurement scope before
using normal-cycle timings to estimate total job duration.
