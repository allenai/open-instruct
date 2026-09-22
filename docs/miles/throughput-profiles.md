# Throughput starting configurations

Use the trainer size and **desired optimization batch** to choose a starting
profile, then provision inference to keep completed-group waiting near zero.
The September 13 exercise found that full decode CUDA graphs mattered more than
adding serving GPUs. The initial unpacked concurrency-32 follow-up used two
inference GPUs without warm completed-queue drops. The historical EP2 throughput experiment combined 6144-token packing with
no recomputation and guarded scoring skip. Its 24-update live qualification
measured 5,089 useful response tokens/s, 58% awaited collection and 0.79% warm
stale-token drops. The faster trainer shifted the bottleneck toward batch supply.
See [qualification](measurements/full-sft-basket-20260914.md) and the preceding
[packed controls](measurements/packed-capacity-results-20260914.md).

These recommendations apply to the existing **18.5B-total full-SFT KDA/latent MoE**,
GSM8K-style responses capped at 4096 tokens, and Holmes B300 GPUs. They are a
measured starting point, not a universal fit or learning-quality guarantee. The
hero checkpoint, dense/FSDP trainer, judges, code execution and longer contexts
need their own capacity checks. See [measurements and figures](measurements/throughput-20260913.md)
and the [chronological campaign log](measurements/throughput-campaign-20260913.md).

## Runtime image

For the maintained examples, use the current image and qualification boundaries
in the [MILES GRPO guide](grpo.md). Its [September 18 qualification record](measurements/router-controls-20260918.md)
distinguishes completed tiny-model checks from the full-policy checks still in progress.

The historical throughput results below used application source `2c477efd5`
and image `01M2F1RKZFZVJYAS0XQGEC3SEJ`, with the exact qualification overlays
recorded in [the original report](measurements/full-sft-basket-20260914.md).
That runtime passed 113 packaged CPU tests; the EP8 mixed-task attempt stopped
after one update on an HTTP transport error. These measurements are not a new
throughput qualification of the current image or the 32K medium template.

## Choose a profile

Use the four [maintained starters](../../configs/miles/examples/README.md).

| Profile | Trainer / inference / judge GPUs | Role |
|---|---|---|
| dev | 1 shared / — | Tiny-model colocation mechanics |
| small | 1 / 1 / 0 | Disaggregated GSM8K mechanics |
| medium | 8 / 7 / 1 | Mixed-workload 32K training starting point |
| large | 16 / 32 / 1 | Production proposal; reserves 56 GPUs, not qualified |

The measurements below describe historical 4K experiments. Their labels such as
“small” and “large” are historical campaign names, not the current starter sizes.
Do not transfer the 4K no-recomputation setting to 32K packs without measuring
memory. The maintained medium and large use recomputation and engine admission 64,
sized as described below.

## Size engine admission from memory

Engine admission is the number of requests each SGLang engine decodes at once. Four
settings express it and must move together: `sglang_server_concurrency` (HTTP
slots per engine), `sglang_max_running_requests` (the engine's batch limit),
`sglang_cuda_graph_max_bs_decode` (the largest batch replayed from a CUDA graph)
and the two memory pools below. Decode throughput grows with batch size until the
GPU saturates, so an admission set below what memory allows leaves throughput
unused whenever training waits for batches.

For the 18.5B-total KDA/latent MoE served on one GPU per engine:

| Pool | Size per unit | Source |
|---|---|---|
| Weights | 34.5 GiB | `Load weight end ... mem usage=34.52 GB` |
| Full-attention KV cache | 16 KiB per token: 2 (K, V) × 4 attention layers × 8 KV heads × 128 dims × 2 bytes | `KV Cache is allocated ... #tokens: 786432, K size: 6.00 GB, V size: 6.00 GB` |
| KDA recurrent state | About 32 MiB per slot (16 KDA layers × 16 heads × 128 × 256, FP32) | Pool allocation minus KV: 45 GiB for 1,024 slots and 786,432 tokens |

SGLang reports these values in GiB although its logs print “GB”. The sources are
engine logs from the September 20 mixed 32K runs on Holmes B300, where each engine
saw 266.9 GiB and kept 187.3 GiB free after all pools and graphs were allocated.

To choose admission `R` for context length `C` (`inference.max_context_length`):

1. Set `sglang_server_concurrency`, `sglang_max_running_requests` and
   `sglang_cuda_graph_max_bs_decode` to `R`.
2. Set `sglang_max_total_tokens = R × C`. The KV pool then holds every running
   request at full context, so KV capacity never forces a retraction.
3. Keep `sglang_max_mamba_cache_size` above `5 × R` with the KDA radix cache (the
   validator enforces this), or at least `R` with radix caching off.
4. Check that weights + `R × C × 16 KiB` + slots × 32 MiB fits within
   `sglang_mem_fraction_static` × visible GPU memory. At 0.7 on a B300 the budget
   is about 187 GiB.
5. Omit `async.async_max_concurrent_samples`. The producer then sizes itself to
   `max(collection, 2 × engines × R)` samples, which keeps every engine refilled
   without a long upstream queue.

Worked budget at `C = 34,816` with 1,024 state slots (32 GiB):

| `R` | KV pool | Weights + KV + state | Fits in 187 GiB? |
|---|---|---|---|
| 16 | 8.5 GiB | 75 GiB | Yes |
| 32 | 17 GiB | 84 GiB | Yes |
| 64 | 34 GiB | 101 GiB | Yes: maintained medium and large |
| 128 | 68 GiB | 135 GiB | Yes, but see the router limit below |

On B300, memory stops binding well before the other limits. The 4K live refresh
runs at 128 and 256 concurrency failed after 8 and 6 updates with MILES-router `ReadError`/503
transport errors, without evidence of an out-of-memory failure
([packed capacity results](measurements/packed-capacity-results-20260914.md)).
Keep `R ≤ 64` until that failure is understood. On a smaller GPU, run the same
arithmetic before lowering anything: an 80 GiB GPU at static fraction 0.7 leaves
about 21 GiB after weights, so reduce `R` or the state-slot count to fit that.

After launch, confirm the setting from the engine metrics:

- `#running-req` should sit near `R` while the trainer waits for batches.
- `token usage` should stay below 1, with few retractions.
- Per-engine generation throughput should rise over the admission-16 baseline of
  about 1,730 tokens/s.

If requests run at the cap with KV usage far below one half while training waits,
admission is too low. That is what the September 20 run showed at `R = 16`: 15.6
of 16 running, 24% KV usage, and training waiting for batches for 78% of the
workflow. The throughput gain from 16 to 64 at 32K has not yet been measured live. The fixed-policy single-engine benchmark, with
2,048-token responses, rose from 4,749 to 6,171 tokens/s between 32 and 64
concurrent sequences and was still rising at 512.

## Settings in the historical 4K measurements

* Enable **full decode CUDA graphs** through the configured request admission;
  keep prefill graphs disabled for this qualified refresh path. The historical EP2 experiment
  used 32 HTTP/running slots per engine and capture through batch 32.
* Keep radix caching with the `extra_buffer` KDA strategy and static memory
  fraction 0.6. The historical small profile used 786432 token slots and 1024
  recurrent-state slots per engine; the historical large profile used
  131072/128 pools at concurrency 16. These are requested
  limits; inspect the engine's resolved capacities and memory after capture.
* Use dynamic-row Core kernels and persistent Triton caching. Cache namespaces
  include source/configuration identity; new configurations can still start cold.
* Publish every optimizer step using flattened 1-GiB buckets and per-expert
  export. Warm publication was about three to four seconds in these trials.
* The historical EP2 profile enabled 6144-token trainer packing, disabled
  activation recomputation and used guarded scoring skip. The full-SFT EP2 live test passed
  all 24 updates, with an initial bit-exact scoring check. It used about 197 GiB
  per trainer GPU in the matched screen; restore recomputation for smaller memory
  budgets. See [qualification and the larger baseline](measurements/full-sft-basket-20260914.md).
* Keep the completed FIFO to one collection. The historical small profile used a
  512-sample producer budget; it can retain substantial work at shutdown.
  The automatic producer default, when not overridden, is
  one collection or two waves of requested serving admission, whichever is larger.
  It is a starting heuristic, not an instruction to produce as far ahead as possible.
* Enable pipeline observations and serving metrics to see where work waits.
  Detailed route replay diagnostics were enabled in graph qualification; starter
  files leave that extra audit disabled. Ordinary shape/probability/version and
  optimizer checks remain active.

The examples prepare a normal train/eval task split and retain evaluation, native
saves and optional final HF export. The throughput basket used frozen prepared
GSM8K inputs and disabled eval/saves/export to isolate normal cycles. Its timing
numbers therefore do not predict total wall time with those additional stages.
Do not infer resume or export qualification from this basket.

See the [capacity dashboard guide](capacity-dashboard.md) for rate definitions,
trainer tuning controls and the reusable W&B report publisher.

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
is not proof of an efficient serving engine: the graphs-off runs occupied their
request slots while showing much lower device activity and useful throughput.

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
   workflow shut down successfully before recording a passing qualification.

The observed two-trainer and eight-trainer training times generally settled by updates four to six,
with earlier outliers and cold first steps lasting several minutes. This is a
**timing-based** warmup criterion, not proof that no further kernel compilation
can occur. The separate compiler-cache tests cover cache reuse; this topology
basket is not a cache hit-rate experiment.

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
pinned sources. Qualification reused an older immutable image with a recorded,
committed source overlay; that base image alone is not the complete qualified
runtime. The lock and patches include the corresponding MILES changes for a
normal build. Use a new output root for each configuration.

`plan` includes `runtime.throughput` and `runtime.async_capacity`. Invalid geometry
and nonpositive limits fail validation; advisories explain undersupplied engines,
insufficient capture/pool coverage, and excessive retained work. Intentional
small runs can retain warnings. See [the queue guide](async-pipeline.md) for exact
ownership, semaphore, retry and lifecycle semantics.

The examples use urgent Holmes placement in `ai2/open-instruct-dev`, one-hour
minimum runtime and explicit GPU allocation. CPU-only jobs needing WEKA belong
on Saturn. Multi-node auto-resume remains disabled until that restart topology
is separately qualified.
