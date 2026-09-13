# Structured MILES run examples

Start with the [MILES guide](../../../docs/miles/index.md) and
[launch instructions](../../../docs/miles/launching.md). Copy a TOML outside the
checkout or into an ignored run directory, replace YOUR_USERNAME and input/output
paths, then run plan and validate. Examples are editable starting points, not
historical measurement records or universal memory-fit guarantees.

| Example | Purpose |
|---|---|
| [grpo-basic.toml](grpo-basic.toml) | Tiny-model, one-GPU resident colocated execution check |
| [grpo-disaggregated.toml](grpo-disaggregated.toml) | EP2 trainer plus one dedicated TP1 engine, synchronous training |
| [grpo-async-disaggregated.toml](grpo-async-disaggregated.toml) | Production-shaped EP8 trainer node plus eight TP1 engines on another node; async/TIS and packing |
| [grpo-multitask.toml](grpo-multitask.toml) | Short GSM8K/math mixed-task preparation and evaluation exercise |

These starters restore 8 prompts × 8 responses = 64 samples and global batch 64.
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
