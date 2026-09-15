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

Use immutable image `01M2F1RKZFZVJYAS0XQGEC3SEJ` for these throughput profiles.
It builds the merged source at `2c477efd5` with the locked Core/MILES/SGLang
sources, including refresh and completed-queue instrumentation. No experimental
source overlay is needed. The earlier sharing image predates this integration;
do not assume a local TOML upgrades code inside an existing image.

The new image passed source reconstruction/build checks; its runtime code passed
113 packaged CPU tests. GPU evidence is the matching EP2 runtime exercised through
the committed qualification overlay. The EP8 mixed-task attempt stopped after one update on an HTTP transport error;
its [record](measurements/full-sft-basket-20260914.md) distinguishes successful
first-step checks from the uncompleted baseline. This image is the current
MILES GRPO runtime; qualification remains specific to the recorded workload.

```bash
MILES_EXISTING_IMAGE=01M2F1RKZFZVJYAS0XQGEC3SEJ \
  python -m open_instruct.miles run /path/to/my-small.toml
```

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
memory. The maintained medium uses recomputation and candidate admission 16.

## Settings behind the recommendation

* Enable **full decode CUDA graphs** through the configured request admission;
  keep prefill graphs disabled for this qualified refresh path. The historical EP2 experiment
  used 32 HTTP/running slots per engine and capture through batch 32.
* Keep radix caching with the `extra_buffer` KDA strategy and static memory
  fraction 0.6. Small uses 786432 token slots and 1024 recurrent-state slots per
  engine; large retains its qualified 131072/128 pools at concurrency 16. These are requested
  limits; inspect the engine's resolved capacities and memory after capture.
* Use dynamic-row Core kernels and persistent Triton caching. Cache namespaces
  include source/configuration identity; new configurations can still start cold.
* Publish every optimizer step using flattened 1-GiB buckets and per-expert
  export. Warm publication was about three to four seconds in these trials.
* The small example enables 6144-token trainer packing, disables activation
  recomputation and uses guarded scoring skip. The full-SFT EP2 live test passed
  all 24 updates, with an initial bit-exact scoring check. It used about 197 GiB
  per trainer GPU in the matched screen; restore recomputation for smaller memory
  budgets. See [qualification and the larger baseline](measurements/full-sft-basket-20260914.md).
* Keep the completed FIFO to one collection. Small explicitly uses the measured
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
