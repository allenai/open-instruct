# Throughput starting configurations

Use the trainer size and **desired optimization batch** to choose a starting
profile, then provision inference to keep completed-group waiting near zero.
Measure decode CUDA graphs, engine admission and trainer wait together. Faster
training can shift the bottleneck to rollout supply.

Throughput depends on the model architecture, response lengths, GPU type,
trainer backend and any judges or code execution in the reward path. Treat the
guidance here as a starting point and check capacity for each new combination.
[Development defaults](development-defaults.md) summarizes the recommended
starting values.

## Choose a profile

Choose among the four [maintained starters](../../configs/miles/examples/README.md).
Their [generated recipe tables](configuration.md#example-recipes) show GPU
allocations and batch geometry from the current TOMLs. For the runtime image, see
the [MILES GRPO guide](grpo.md). Do not transfer a short-context
no-recomputation setting to long packs without checking memory; the maintained
medium and large use recomputation and engine admission 64, sized as described
below.

## Size engine admission from memory

Engine admission is the number of requests each SGLang engine decodes at once. Four
settings express it and must move together: `sglang_server_concurrency` (HTTP
slots per engine), `sglang_max_running_requests` (the engine's batch limit),
`sglang_cuda_graph_max_bs_decode` (the largest batch replayed from a CUDA graph)
and the two memory pools below. Decode throughput grows with batch size until the
GPU saturates, so an admission set below what memory allows leaves throughput
unused whenever training waits for batches.

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

1. Set `sglang_server_concurrency`, `sglang_max_running_requests` and
   `sglang_cuda_graph_max_bs_decode` to `R`.
2. Set `sglang_max_total_tokens = R × C`. The KV pool then holds every running
   request at full context, so KV capacity never forces a retraction.
3. Keep `sglang_max_mamba_cache_size` above `5 × R` with the KDA radix cache (the
   validator enforces this), or at least `R` with radix caching off.
4. Check that weights + `R × C × (KV bytes per token)` + slots × (state bytes per
   slot) fits within `sglang_mem_fraction_static` × visible GPU memory.
5. Omit `async.async_max_concurrent_samples`. The producer then sizes itself to
   `max(collection, 2 × engines × R)` samples, which keeps every engine refilled
   without a long upstream queue.

For the example model at `C = 34,816` with 1,024 state slots (32 GiB):

| `R` | KV pool | Weights + KV + state |
|---|---|---|
| 16 | 8.5 GiB | 75 GiB |
| 32 | 17 GiB | 84 GiB |
| 64 | 34 GiB | 101 GiB |
| 128 | 68 GiB | 135 GiB |

On a large-memory GPU, memory may stop binding well before other limits such as
router and transport capacity. Keep `R ≤ 64` as the starting point (see
[development defaults](development-defaults.md)) and raise it only while the
router and engines stay healthy. On a smaller GPU, run the same arithmetic before
lowering anything: an 80 GiB GPU at static fraction 0.7 leaves about 21 GiB after
the example weights, so reduce `R` or the state-slot count to fit that.

After launch, confirm the setting from the engine metrics:

- `#running-req` should sit near `R` while the trainer waits for batches.
- `token usage` should stay below 1, with few retractions.
- Per-engine generation throughput should rise over a lower-admission baseline.

If requests run at the cap with KV usage far below one half while training waits,
admission is too low.

## Other serving and training settings

* Enable **full decode CUDA graphs** through the configured request admission;
  keep prefill graphs disabled on the refresh path.
* Requested pool sizes are limits; inspect the engine's resolved capacities and
  memory after graph capture.
* Use dynamic-row Core kernels and persistent Triton caching. Cache namespaces
  include source/configuration identity; new configurations can still start cold.
* Publish every optimizer step using flattened 1-GiB buckets and per-expert export.
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

Copy a profile, replace the model and output paths, and inspect the resolved
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
