# Throughput starting configurations

Use the trainer size and **desired optimization batch** to choose a starting
example, then provision inference to keep completed-group waiting near zero.
Measure decode CUDA graphs, engine admission and trainer wait together. Faster
training can shift the bottleneck to rollout supply.

Throughput depends on the model architecture, response lengths, GPU type,
trainer backend and any judges or code execution in the reward path. Treat the
guidance here as a starting point and check capacity for each new combination.
[Development defaults](development-defaults.md) summarizes the recommended
starting values.

## Choose an example

Choose among the three [maintained starters](../../configs/miles/examples/README.md).
Their [generated recipe tables](configuration.md#example-recipes) show GPU
allocations and batch geometry from the current TOMLs. For the runtime image, see
the [MILES GRPO guide](grpo.md). Do not transfer a short-context
no-recomputation setting to long packs without checking memory; the maintained
node and three-node examples use recomputation and a measured B300 engine starting admission of 512.
Their [capacity notes](../../configs/miles/examples/README.md#capacity-and-qualification-settings)
describe the supported model geometry, runtime requirements and qualification limits. The
generic loader still defaults to 64; size other models and GPUs as described below.

## Size engine admission from memory

Engine admission is the number of requests each SGLang engine decodes at once.
`sglang_max_running_requests` (the engine's batch limit),
`sglang_cuda_graph_max_bs_decode` (the largest batch replayed from a CUDA graph)
and the two memory pools below size GPU capacity. `sglang_server_concurrency`
instead sizes end-to-end HTTP permits per engine: a request holds its permit
while in transit and while its response is processed, even when the GPU has
finished it. Increasing admission can improve decode throughput, but memory fit
is only a capacity bound. Kernel efficiency, memory bandwidth, CPU scheduling,
prefill and transport can reach diminishing returns before GPU memory fills.

As a worked example, consider a hybrid MoE served on one GPU per engine whose
weights occupy 34.5 GiB, with 4 full-attention layers × 8 KV heads × 128 dims and
16 recurrent (KDA) layers × 16 heads × 128 × 256 state:

| Pool | Size per unit | Derivation |
|---|---|---|
| Weights | 34.5 GiB | Engine log `Load weight end ... mem usage` |
| Full-attention KV cache | 16 KiB per token | 2 (K, V) × 4 layers × 8 KV heads × 128 dims × 2 bytes |
| KDA recurrent state | About 32 MiB per slot | 16 layers × 16 heads × 128 × 256, FP32 |

Read the actual sizes for your model from the engine startup logs (`Load weight
end`, `KV Cache is allocated`). SGLang reports these values in GiB although its
logs print “GB”.

To choose admission `R` for context length `C` (`inference.max_context_length`):

1. Set `sglang_max_running_requests` and `sglang_cuda_graph_max_bs_decode` to `R`.
   For async runs, start `sglang_server_concurrency` at `2 × R`. Structured
   configs derive this when omitted. This adds transport headroom, not GPU slots.
   The named B300 preset uses `R`: at 512 admission, additional HTTP
   headroom mostly increased queueing. Recheck occupancy for other workloads.
2. Set `sglang_max_total_tokens = R × C`. The KV pool then holds every running
   request at full context in the nominal capacity calculation. Verify the
   resolved token capacity and retractions after startup; this does not prevent
   retractions required by mid-sequence weight refresh.
3. Keep `sglang_max_mamba_cache_size` at least `R` with radix caching off. The
   pinned Olmo hybrid backend also supports the qualified cached path; see
   [model limits](moe-model-support.md#current-limits).
   For compatible KDA backends with radix caching enabled, the capacity validator
   requires more than `5 × R` slots. Passing capacity validation does not establish
   that a model's kernel supports prefix-cache snapshots.
4. Check that weights + `R × C × (KV bytes per token)` + slots × (state bytes per
   slot) fits within `sglang_mem_fraction_static` × visible GPU memory, leaving
   room for decode graphs, activation/workspace peaks and runtime overhead. The
   sum is a lower bound on the footprint, not a complete memory-fit guarantee.
   With explicit pool sizes, changing the memory fraction alone does not resize
   those requested pools; inspect the actual allocations after graph capture.
5. When `async.async_max_concurrent_samples` is omitted, the producer sizes itself to
   `max(collection, 2 × engines × R)` samples when HTTP capacity is at least `R`.
   Both producer and HTTP capacity need headroom: raising only one leaves the
   other as a bottleneck. Two waves are a starting point, not a measured guarantee
   of full occupancy; monitor response latency, host memory and stale work.
   The learning examples use this automatic rule after profile resolution.
   Explicit budgets still win. Larger fleets can produce much more work than the
   trainer can consume within the lag limit; inspect policy-age warnings and
   measure useful consumption before using the capacity-based budget in a long run.

For the example model at `C = 34,816` with 1,024 state slots (32 GiB):

| `R` | KV pool | Weights + KV + state |
|---|---|---|
| 16 | 8.5 GiB | 75 GiB |
| 32 | 17 GiB | 84 GiB |
| 64 | 34 GiB | 101 GiB |
| 128 | 68 GiB | 135 GiB |

On a large-memory GPU, memory may stop binding well before other limits such as
router and transport capacity. The generic starting admission remains 64 (see
[development defaults](development-defaults.md)); the named B300 preset changes it
for the measured smaller-state Olmo 3.5 architecture. Their 512 setting does not
follow from the larger-model arithmetic above. On a smaller GPU, run the same arithmetic before
lowering anything: an 80 GiB GPU at static fraction 0.7 leaves about 21 GiB after
the example weights, so reduce `R` or the state-slot count to fit that.

After launch, confirm the setting from the engine metrics:

- `#running-req` should sit near `R` while the trainer waits for batches.
- `token usage` should stay below 1, with few retractions.
- Per-engine generation throughput should rise over a lower-admission baseline.

If requests run at the cap while training waits, compare a higher admission at
the same workload. Low KV usage suggests memory headroom; it does not establish
a throughput gain. Keep the smaller setting when marginal throughput gains are
small or queueing, response age and memory costs rise. The measured B300 model
showed diminishing returns above roughly 512 despite additional memory room.

For a new GPU, use model geometry to narrow the search before launching: compare
two plausible admissions, repeat the lower control, and add one capacity probe
only if useful. Repeat the selected setting at representative prompt and response
lengths. Full-context KV sizing is conservative; a smaller pool can support more
short requests, but long responses may trigger retractions. Label that tradeoff
and measure it rather than treating the larger admission as guaranteed capacity.

Retain the resolved config, runtime revisions, hardware identity, raw engine and
client logs, memory/occupancy samples, response-length distribution, retractions
and warmup/measurement boundaries for every case, including failed attempts.
Report repeated-control drift and cache-hit rates with throughput. Small repeated
prompt sets can make cached prefill unusually cheap; include uncached prefill
when generalizing to new workloads. Test cache choices on a frozen stream from
the intended prompt mixture, retaining normal sibling counts and natural
duplicates while avoiding dataset cycling during measurement. Preserve source
IDs and prompt order so each window's task mix and actual prompt/response lengths
can be audited. Record filtering or truncation explicitly. Fixed-weight cache
results also need validation across production weight refreshes, which can change
cache reuse and prefill work.

## Whole presets and explicit settings

**A named profile applies a measured preset whole. To change any of its settings,
use `generic` and write them explicitly.** There are two behaviors:

- **Omitted profile or `profile = "generic"`:** existing generic defaults fill
  unset values; your explicit settings are retained. If model, hardware and
  topology match a measured preset, the plan prints a boxed tip naming it.
  The tip changes no settings and never requires adopting the preset.
- **A named profile:** apply that preset verbatim. Every explicit preset-owned
  key conflicts, even if its value equals the preset. Native aliases and `--set`
  values count too. Incompatible model, hardware or topology is an error, never
  a fallback to different settings. `auto` is not a supported selector.

The learning examples pin `profile = "olmo35-b300-v1"` on Holmes. To use Jupiter,
change both the cluster and the profile to `olmo35-h100-v1`. A wrong cluster error
names the matching hardware's preset. The planner checks known hardware,
dedicated TP1 placement, context/response caps and cached-KDA prerequisites
without needing model metadata.

The model guard reads the checkpoint's `config.json` and requires the studied
geometry and BF16 weights; the model-family name alone is insufficient. It runs
during planning when the actual checkpoint is local, otherwise after preparation.
A named preset applies the same settings at both stages. Missing or incompatible
metadata after preparation stops training. An OLMo-core conversion template is
not treated as evidence of the source checkpoint's geometry.

| Preset | Cache | Admission / HTTP / graph batch | KV tokens | Recurrent slots | Memory fraction |
|---|---|---:|---:|---:|---:|
| `olmo35-b300-v1` | On, `extra_buffer` | 512 | 17,825,792 | 3,072 | 0.7 |
| `olmo35-h100-v1` | Off | 512 | 6,684,672 | 512 | 0.9 |

Both use prefill chunks of 8,192, full decode graphs and disabled prefill graphs.
The preset owns those fields, cache enablement/strategy, admission, HTTP concurrency,
graph ceiling, KV-token pool, recurrent pool and memory fraction. Settings outside
that set, such as topology, context/response caps, metrics, producer budget,
trainer batch and policy lag, remain separate controls. Existing compatibility
checks still apply. Presets do not interpolate, resize pools, estimate memory fit,
or change the runtime image. Cached hybrid serving requires the pinned FP32
snapshot implementation described in [model limits](moe-model-support.md#current-limits).

### Deliberately departing from a preset

A conflicting setting stops planning with the offending normalized keys and a
complete preset block as TOML. The error offers two paths:

1. Remove the explicit preset-owned settings, including their native aliases,
   and retain the named profile.
2. Select `generic`, paste the displayed block into the existing `[inference]`
   section, and edit the settings yourself. Merge the block; do not create a
   duplicate TOML section.

For example, this is an editable B300 preset with admission deliberately reduced:

```toml
[inference]
profile = "generic"
sglang_disable_radix_cache = false
sglang_mamba_radix_cache_strategy = "extra_buffer"
sglang_max_running_requests = 256
sglang_server_concurrency = 512
sglang_cuda_graph_backend_decode = "full"
sglang_cuda_graph_max_bs_decode = 512
sglang_cuda_graph_backend_prefill = "disabled"
sglang_chunked_prefill_size = 8192
sglang_max_total_tokens = 17825792
sglang_max_mamba_cache_size = 3072
sglang_mem_fraction_static = 0.7
```

Only admission changed here. The other pools and limits remain explicit; nothing
is resized implicitly. This is an experiment, not another measured preset. Generic
mode does not reject a cluster change merely because the block came from B300.
On H100, the availability tip names the matching H100 preset, but you own the
explicit memory configuration. Qualify any copied block on its target device.
The separately measured cache-off B300 and cached-320 H100 alternatives in the
[example notes](../../configs/miles/examples/README.md#capacity-and-qualification-settings)
must also be configured explicitly in generic mode.

### Planning evidence and fleet capacity

Configuration advice is read-only. Actionable suggestions appear in
`runtime.inference_profile.tips` and a single **Configuration tips** box at the
end of stderr. Routine evidence notes remain in `inference_profile.warnings`;
capacity and throughput warnings stay in their own sections and print as plain
lines. `plan`, `validate` and `run` print the same advice; plan JSON stays on stdout.
Preparation checks again and records values and provenance in `resolved-plan.json`
and `inference-profile.json`.

| Code | Meaning |
|---|---|
| `inference_profile_available` | Tip: a measured preset matches a generic run. Names the setting to add; changes nothing. |
| `inference_pool_hardware_mismatch` | Tip: explicit KV and recurrent pools equal a preset for a different GPU. Review before running; this is a heuristic, not a memory-fit check. |
| `inference_metadata_unavailable` | The model metadata for this generic run is unavailable here; recommendations wait for preparation. Settings stay unchanged. |
| `inference_model_guard_deferred` | A named preset is selected; its model guard must pass after preparation before training. |
| `inference_fleet_extrapolation` | A named preset covers per-engine settings, not this fleet's routing, publication latency or useful throughput. |

For a new model or topology, start with generic settings, inspect the plan, and
run bounded measurements rather than interpolating these pools. Unknown shapes
receive no preset recommendation. Users and agents can inspect the structured
codes above without parsing the rendered tips box.

Fleet advisories remain visible for any named preset with more than one engine.
Per-engine measurements do not establish fully async refresh or useful throughput
for a different fleet.
Managed replicas inventory their rollout GPUs at startup. After preparation, all
inventories are checked against an applied preset, excluding trainer/judge GPUs.
Direct single-node training checks the rollout slice; a matched multi-node preset
requires managed per-replica inventories. Checks run outside generation.

A producer budget equal to one wave of engine slots gets producer-backfill advice,
regardless of HTTP headroom or profile selection. HTTP capacity below admission
and an underfeeding producer have separate advisories.

Omitted producer budgets still use two waves of resolved capacity, rounded to
whole groups (or one collection, whichever is larger). Applied presets give
6,144/19,456 samples for node/three-node; generic defaults give 768/2,432. Explicit
budgets win. Batch 256 and lag 6 are unchanged. The policy-age advisory is
intentional: one full wave across six 512-slot engines already represents 12
batches. A cap near six batches necessarily underfeeds that fleet. Neither 24
batch-equivalents nor a static lag limit predicts drain time or actual policy age.
Measure useful consumption, queue age/discards and publication latency in the
fully async workload; explicitly reduce the producer budget or fleet when needed.

For a deliberate departure, retain the resolved plan, checkpoint/runtime versions,
full logs and response results. Begin with a bounded run on the intended shape;
check memory, sustained useful throughput, occupancy and mid-generation refresh.
There is no profile memory estimator or automatic pool adjustment in generic mode.

## Qualification limits

The presets are starting points for the studied BF16 Olmo 3.5 geometry and workload,
not established optima for other models, context distributions, prompt reuse or fleets.
Short correctness and per-engine throughput checks do not establish sustained
fully async fleet throughput, learning equivalence or save/resume reliability;
validate the intended workload and placement before a long run.

## Other serving and training settings

* Enable **full decode CUDA graphs** through the configured request admission;
  keep prefill graphs disabled on the refresh path.
* Requested pool sizes are limits; inspect the engine's resolved capacities and
  memory after graph capture.
* Use dynamic-row Core kernels and persistent Triton caching. Cache namespaces
  include source/configuration identity; new configurations can still start cold.
* Publish every optimizer step using flattened 1-GiB buckets and per-expert export.
* In qualified long runs, use `check_weight_update_equal = false`. The optional
  startup snapshot/reset audit remains enabled in the dev mechanics example.
  Turning it off does not remove required publication acknowledgments.
* Leave `core.scoring_pass_required` at its default `false`: the recipe selects whether a
  standalone pre-update scoring forward is needed. One optimizer step per
  collection with no KL in advantages can reuse the training forward. Multiple
  optimizer steps, KL requirements and runtime guards still require their scores.
  The learning templates use `core.scoring_check_interval = 0`, which validates
  reuse on the first update of each process, including resume, without repeating
  the audit during steady training. Enable periodic checks during qualification.
  This does not disable TIS or replace trainer scores with serving scores.
  Recipes copied from trainer-comparison
  campaigns may force it; check `plan` output under `scoring_pass` before launching.
* Disabling activation recomputation can help short contexts when memory allows;
  restore it for long contexts or smaller memory budgets.
* Keep the completed FIFO to one collection. A large explicit producer budget can
  retain substantial work at shutdown. The automatic producer default, when not
  overridden, is one collection or two waves of requested serving admission,
  whichever is larger. It is a starting heuristic, not an instruction to produce
  as far ahead as possible.
* Enable pipeline observations and serving metrics to see where work waits.
  Starter files leave detailed route replay diagnostics disabled. Ordinary
  shape/probability/version and optimizer checks remain active.

The examples prepare a normal train/eval task split and retain evaluation, native
saves and optional final HF export. Disable eval, saves and export when isolating
normal training cycles, and remember that such timings do not predict total wall
time with those stages.

If HTTP permits are full but engine slots are not, inspect the time outside the
engine before adding GPUs. The October H077 review found approximately 721 active
HTTP requests but 360 running sequences across 12 engines, with 768 permits and
768 GPU slots configured. In 7,285 completed timed responses, mean HTTP duration
was 141 seconds versus 63 seconds inside engine timestamps. The original H1
configuration also used only one wave of permits and showed this delay. This
motivates separate transport headroom; it does not establish which CPU or network
stage caused the delay or certify a speedup from the revised settings.

## Read the right measurements

**Completed-buffer get time** includes waiting for eligible groups and filtering
expired ones. The broader driver's `generation_wait` stage also includes batch
collection and handoff. Neither is hardware GPU utilization. Once buffer-get time
is near zero, adding inference cannot remove the remaining handoff or training
cost. The report shows both separately.

**Discarded tokens** are dropped response tokens divided by delivered plus dropped
response tokens at dequeue. Also inspect the response-attempt fraction and the
length/age bins. With `retry`, expired response attempts are discarded and their
prompts are requeued; the spent generation work is still lost. Shutdown leftovers
and unfinished requests are outside this denominator.

Producer-owned completions waiting for insertion are separate from the completed
FIFO. A 64-sample producer plus a 32-sample FIFO can retain three future batches
when the optimization batch is 32. Rapid version advancement can expire some of
that work even while the trainer occasionally waits for eligible replacements.
Reduce ahead-of-training work when drops are high; keep the configured age rule
visible rather than hiding the tradeoff by raising it.

Sampled engine admission and queue counts show occupancy. NVML GPU activity shows
kernel activity, not SM occupancy or achieved FLOPs. High HTTP occupancy alone
does not mean an efficient serving engine: without decode CUDA graphs, engines can
occupy their request slots while showing much lower device activity and useful
throughput.

## Warmup and a short comparison procedure

1. Keep model, response cap, optimization batch, samples per prompt, lag and
   objective fixed when testing a scheduling change. Record allocated GPUs as
   well as actively used GPUs.
2. Run at least 16 updates. Initially exclude the first six and inspect the entire
   scoring/forward-backward trace. Extend or move the window if late spikes or a
   changing queue distribution remain; update seven is not inherently steady state.
3. Compare useful consumed tokens/second, completed-buffer waiting, other handoff
   time, publication, and discarded attempts/tokens. Use the age/length breakdowns
   to check for selective waste.
4. If training waits while engines have spare capacity, inspect producer, HTTP,
   state-pool and capture limits. If engines are efficiently busy, add inference.
   If training is supplied and drops grow, reduce ahead-of-training work.
5. Confirm all trainer ranks completed the intended optimizer sequence and the
   workflow shut down successfully before recording a result.

Cold first steps can last several minutes. Excluding early updates is a
**timing-based** warmup criterion, not a guarantee that no further kernel
compilation can occur.

## How the limits interact

| Control | What it limits | External constraint and tuning signal |
|---|---|---|
| Trainer GPUs / EP | Model and optimizer distribution, training time | Expert divisibility, model memory, collective bandwidth; compare training time at the same optimization batch. |
| Inference GPUs / engine TP | Independent engines and model fit | GPU memory and interconnect; increasing TP reduces engine count at a fixed GPU budget. |
| Producer sample budget | Owned groups, or unfinished samples with sample backfill | Fleet service time, grading latency and group stragglers; enough headroom to refill engines, then watch age and discarded tokens. |
| HTTP concurrency per engine | Global generation semaphore, scaled by engine count | Too low leaves serving slots empty; too high moves waiting work into the serving system without creating GPU capacity. |
| Running requests per engine | Requested decode batch admission | Effective token pool, recurrent-state pool and GPU memory can cap it further. Record the engine's resolved limit. |
| Token/context and recurrent-state pools | Capacity for active contexts and cached prefixes | Model geometry, dtype, radix strategy, overlap scheduling and memory headroom. KDA state slots are not necessarily one per request. |
| Completed-buffer factor | Whole ready groups waiting for consumption | Trainer service rate and allowed policy lag; a larger queue absorbs bursts but cannot repair a sustained rate mismatch. |
| Collection / optimization batch | Responses collected and samples per optimizer step | Trainer divisibility, memory, desired RL statistics; changing these is an optimization change, not just a throughput tweak. |
| Allowed policy lag | Which completed groups remain eligible | Current trainer version versus oldest sampled token version; raising it accepts more off-policy data rather than making generation faster. |
| Publication interval | How often serving receives current weights | Collective transfer and re-prefill latency versus policy freshness; keep this cost visible. |
| Save / eval cadence | Interruptions outside normal training cycles | Checkpoint I/O, evaluation size and draining outstanding work; compare total run time separately. |

Start with model fit and the desired optimization batch, then size engine
admission from memory. Use the topology-derived producer budget as a starting
point. If training waits and engines have spare capacity, increase admission or
backfill; if engines are already busy, compare more inference GPUs. If completed
work ages out while training stays busy, reduce ahead-of-training work before
loosening the lag limit. Report dropped tokens as well as samples: a small sample
fraction can hide substantial wasted long-response work. Keep the length/age
breakdowns alongside the aggregate so this tradeoff stays visible.

Persistent compilation caching reduces repeat startup cost; it does not increase
engine admission or buffer capacity. Keep it enabled for normal examples, while
recording cold and warm timings separately. Cache-off comparisons should use
separate run identities and private cache locations, not delete shared caches.

## Plan, validate, and launch

Copy an example, replace the model and output paths, and inspect the resolved
allocation and advisories before launching:

```bash
python -m open_instruct.miles plan /path/to/run.toml
python -m open_instruct.miles validate /path/to/run.toml
python -m open_instruct.miles run /path/to/run.toml
```

Follow the [launch guide](launching.md) to build the runtime from this branch's
pinned sources. Use a new output root for each configuration.

`plan` includes `runtime.throughput` and `runtime.async_capacity`. Invalid geometry
and nonpositive limits fail validation; advisories explain undersupplied engines,
insufficient capture/pool coverage, and excessive retained work. Intentional
small runs can retain warnings. See [the queue guide](async-pipeline.md) for exact
ownership, semaphore, retry and lifecycle semantics.

The examples use high-priority Holmes placement in `ai2/open-instruct-dev`
and explicit GPU allocation. Minimum runtimes vary by tier; see
[placement guidance](launching.md#placement-secrets-and-results), including the
separate CPU scheduling policy. Multi-node auto-resume covers the ordinary
restart path; forced preemption, and restarts of runs carrying a managed judge,
are not yet supported paths.
