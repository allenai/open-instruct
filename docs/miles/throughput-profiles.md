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

Choose among the three [maintained starters](../../configs/miles/examples/README.md).
Their [generated recipe tables](configuration.md#example-recipes) show GPU
allocations and batch geometry from the current TOMLs. For the runtime image, see
the [MILES GRPO guide](grpo.md). Do not transfer a short-context
no-recomputation setting to long packs without checking memory; the maintained
node and large use recomputation and a measured B300 engine starting admission of 512.
Their [capacity notes](../../configs/miles/examples/README.md#capacity-and-qualification-settings)
record the measured model, runtime differences and qualification limits. The
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
   Measured B300 example overrides use `R`: at 512 admission, additional HTTP
   headroom mostly increased queueing. Recheck occupancy for other workloads.
2. Set `sglang_max_total_tokens = R × C`. The KV pool then holds every running
   request at full context in the nominal capacity calculation. Verify the
   resolved token capacity and retractions after startup; this does not prevent
   retractions required by mid-sequence weight refresh.
3. Keep `sglang_max_mamba_cache_size` at least `R` with radix caching off. The
   pinned Olmo hybrid backend requires this mode; see [model limits](moe-model-support.md#current-limits).
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
   The learning examples preserve their previous global budgets explicitly while
   increasing engine ceilings. Do not multiply a fixed-policy per-engine budget
   into a training fleet without checking policy age and useful consumption.

For the example model at `C = 34,816` with 1,024 state slots (32 GiB):

| `R` | KV pool | Weights + KV + state |
|---|---|---|
| 16 | 8.5 GiB | 75 GiB |
| 32 | 17 GiB | 84 GiB |
| 64 | 34 GiB | 101 GiB |
| 128 | 68 GiB | 135 GiB |

On a large-memory GPU, memory may stop binding well before other limits such as
router and transport capacity. The generic starting admission remains 64 (see
[development defaults](development-defaults.md)); the B300 examples override it
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

## Automatic selection and overrides

The learning examples use `inference.profile = "auto"`. Selection reads only the
checkpoint's `config.json`, matches the measured model geometry and BF16 dtype,
and combines cluster GPU family with dedicated TP1 placement and the measured
34,816-token context / 32,768-token response caps. It does not infer fit from the
Olmo model-family name alone. Other shapes retain generic defaults, with the
reason recorded in `plan.runtime.inference_profile`.

When model metadata is unavailable on the submission host, local planning reports
`pending_model_metadata`. Preparation resolves against the actual checkpoint and
writes the final values to `resolved-plan.json`. Named profiles
`olmo35-b300-v1` and `olmo35-h100-v1` require a match; `generic` disables selection.
Explicit TOML values win over profiles, and `--set` replaces the corresponding
TOML value. Plan output records each resolved value and its source. Existing
conflict checks still reject contradictory aliases for the same native option.

The selected capacities are per engine:

| GPU and cache mode | Admission / HTTP / graph batch | KV tokens | Recurrent slots | Static memory fraction |
|---|---:|---:|---:|---:|
| B300, automatic cache-on | 512 | 17,825,792 | 3,072 | 0.7 |
| B300, caching explicitly disabled | 512 | 17,825,792 | 1,024 | 0.7 |
| H100, automatic cache-off | 512 | 6,684,672 | 512 | 0.9 |
| H100, caching explicitly enabled | 320 | 4,456,448 | 1,920 | 0.9 |

All use prefill chunks of 8,192, full decode graphs and disabled prefill graphs.
Profiles choose cache mode with hardware unless `inference.radix_cache` is
explicit. Generic configurations keep caching disabled. Cache-on profiles select
the `extra_buffer` strategy. This branch's runtime lock includes the required
FP32 snapshot implementation; profile resolution itself never changes an image
or its dependencies. Rebuild with the lock and inspect the recorded runtime.
The throughput measurements and updated-runtime deployment checks have distinct
scopes, described below.

The global producer budget is independent: profiles preserve its generic sizing
basis, and examples retain explicit 768/2,432 budgets. An explicit running-request
override adjusts otherwise-unset graph, KV and recurrent capacities within the
selected profile; an explicit pool override wins. H100 retains the bounded KV
pool rather than promising all requests can reach full context simultaneously.
Startup verifies the expected GPU family and memory once, outside generation.

### Deliberately departing from a profile

`auto` selects a measured per-engine starting point; it is not an autotuner or
an assurance that the complete fleet has been validated. Changing admission
within a matched profile derives related capacities arithmetically. Changing
model geometry, precision, context caps or engine topology does not interpolate
to a new optimum: it falls back to generic defaults. Generic defaults themselves
are not a memory-fit guarantee.

Planning emits these advisory codes to stderr and records them in
`runtime.inference_profile.warnings` and `runtime.throughput.warnings` in the
JSON plan. `validate` and `run` also print them, with submission warnings emitted
before an image build. JSON remains on stdout, so it can be saved independently.

| Code | Meaning and next step |
|---|---|
| `inference_profile_fallback` | Auto could not match the model, hardware, workload or engine topology. Read the specific reason and qualify the generic or explicit settings on that shape. |
| `inference_profile_pending` | Checkpoint metadata could not be inspected yet. Preparation checks again; inspect `resolved-plan.json` before interpreting the profile as matched or bypassed. An explicit generic choice remains usable even when advisory metadata cannot be read. |
| `inference_profile_bypassed` | `generic` was explicitly selected even though a measured profile matches. The warning's `differences` map compares configured values to hardware defaults. Keep the deliberate choice, or select `auto` / the matching named profile. |
| `inference_profile_modified` | Explicit overrides and their derived capacities differ from hardware defaults. The warning reports configured/default values. Some alternate cache configurations have separate measurements below; a difference alone does not imply an invalid configuration. |
| `inference_fleet_extrapolation` | More than one engine is requested for a matching profile. Per-engine measurements do not establish fleet routing, publication latency, policy age or producer sufficiency. Review the separate capacity warnings and validate the intended placement. |

The fleet advisory also applies when `generic` bypasses a matching profile. It
is independent of whether queues have enough capacity: a ten-node configuration
can be internally consistent while still exceeding the validation evidence.
The B300 study includes a two-engine throughput check, but that is not a
multi-engine fully async weight-refresh qualification. The updated-runtime
trainer-consumption check used one B300 engine; H100 replay and deployment checks
have their own scope below. The advisory therefore remains visible even for two
engines. Unsupported topologies receive the fallback advisory rather than a
claim that the profile's fleet evidence transfers to them.

For example, retain a matched profile but experiment with admission:

```bash
python -m open_instruct.miles plan configs/miles/examples/medium.toml \
  --set 'inference.sglang_max_running_requests=256' > plan.json
```

Or deliberately use generic defaults and specify your own settings:

```bash
python -m open_instruct.miles plan configs/miles/examples/medium.toml \
  --set 'inference.profile="generic"' \
  --set 'inference.sglang_max_running_requests=128' > plan.json
```

Omitting `inference.profile` means `auto`. A matching named profile is also a
valid selection and does not receive a bypass warning. Explicit values equal to
the profile defaults do not receive a modification warning. `--set` overrides
TOML values; individual explicit settings override profile defaults. Selecting
`generic` does not disable the capacity advisories or existing validity checks.
These advisories require no approval and do not change your configuration.

For a deliberate departure, save the resolved plan, checkpoint/runtime versions,
full logs and response results. Begin with a bounded run on the intended shape;
check memory, sustained useful throughput, engine occupancy, policy age/discards,
and publication acknowledgments with fully async refresh during generation.
Changing fleet size does not automatically retune producer budgets, trainer
batches or lag limits. Inspect `runtime.async_capacity` as well as the profile
before scaling. Record any new measurements with their exact scope rather than
treating a successful startup as throughput or correctness qualification.

## Realistic-prompt cache comparison on B300

The [cache-on/off/on comparison](https://beaker.org/ex/01M5V6YDN8FKT670Q4Y6Z7WH76)
used the pinned 102,014-row Dolci-Think-RL-7B prompt mixture, four siblings and a
fixed order, with admission/HTTP 512, producer 1,024, prefill chunk 8,192 and the
same 32K response cap. Native decode rates were **23,906 / 19,746 / 23,989 tokens/s**:
cached controls were 21.1–21.5% faster than uncached, with 0.34% control drift.
Delivered complete-group rates were 23,700 / 20,026 / 23,496 tokens/s. These
different accounting boundaries include different unfinished work; native counters
are the primary engine measure. Window populations differ, so the bracket is
not a paired confidence interval or a learning-throughput result.

All three measurement windows completed. The parent hit its two-hour timeout
during the closing arm's drain, so that arm does not establish successful shutdown.
All 902 collected files (871.6 MB) were downloaded and hash-verified, with no
collector omissions. The closing window was audited from its retained boundaries,
counter snapshots, batches and response records without marking its workflow complete.

The examples' SGLang pin requires selected recurrent-state snapshots written
from FP32 accumulators. The updated adapter implements that contract. It passed
[eight GPU precision/layout cases and two dispatch cases](https://beaker.org/ex/01M5VGGMDMTV1KFFH7N94WR7S4).
Full-model deployment was checked separately below. Casting BF16 intermediate
states to FP32 would not restore the missing precision. Older images lacking
this implementation reject the cached path; changing a configuration does not
update their runtime.

## Fully async refresh coverage and retained evidence

The [ordered-fixture validation](https://beaker.org/ex/01M5VF9BVN115C8E3VQHB99YMB)
completed two optimizer updates with the default two-GPU EP2 trainer, one B300
engine, batch 256, lag 6, producer 768 and cached admission 512. Both publications
occurred with generation active: the nearest engine observations showed 512 and
505 running requests within four seconds before publication. Publication took
3.85 and 3.65 seconds. All 512 consumed samples passed finite-score, alignment,
version-ledger and expert-route consistency checks, including 105 responses that
crossed a weight version. No invalid expert IDs were found.

This fixture deliberately repeats short, reward-varying prompt clusters before
the remaining real-reward prompts. It exercises overlap and trainer consumption;
it does not measure representative learning throughput or establish learning
equivalence. All 69 regular source files, including both rollout tensors
(2.65 GB total), were recovered and hash-verified. The experimental runtime differs
from the updated example pins.

The [updated-runtime validation](https://beaker.org/ex/01M5VP6DPS9XJCTPWY2XK5EN07)
completed two optimizer steps per trainer rank and exited successfully. It used
the same default EP2 trainer, batch 256, lag 6 and producer budget 768. The second
batch contained **29 mixed-version responses**, confirmed by both trainer ranks'
scoring records. Engine observations showed 511 running requests before each
publication; the updates took **2.70 and 2.99 seconds**. All four workers restored
their compiler caches. The first cold attempt also completed both updates but
consumed zero mixed-version responses, so it was insufficient to qualify that
path. Neither run changes the trainer settings to force overlap.

The 56-file result/log archive passed hash verification, profile-resolution,
version-ledger, engine-occupancy and per-rank optimizer checks. All 76 regular
source files (2.64 GB), including both rollout tensors, were recovered and
hash-verified. All 512 consumed samples passed finite-score, token/version
alignment and full expert-route checks, including all 29 mixed-version responses;
no invalid expert IDs were found. Automatic B300 cache selection
resolves to exactly the native and Core options exercised by this explicit-cache
fixture. These short checks establish mechanics, not representative training
throughput or learning equivalence. The optional startup GC-freeze race and a
five-second child shutdown timeout are retained in the issue log; a separate
startup-readiness fix is available but is not part of this validated image.

The study retains raw results and a machine-readable fitting table under
`runs/inference-throughput-20261025`. `analysis/fit-observations.jsonl` contains
31 completed cases with full model geometry, workload lengths, native options,
immutable image IDs, exact producing-job allocation and artifact hashes.
`analysis/fit-observations.csv` provides a verified 59-column projection, with
metric definitions and fitting limits in `analysis/fit-observations.README.md`.
`analysis/artifact-catalog.json` indexes the attempts and archives. Failed and
interrupted attempts remain available; a completed measurement followed by an
interrupted drain is labeled separately. This is evidence for one model shape,
not a fitted estimator for arbitrary models. New shapes need their own memory
calculation and a small measured admission bracket.

## H100 measurements for the studied Olmo 3.5 checkpoint

The first H100 panel used the same approximately 23.31-GiB checkpoint as the
B300 measurements, one active TP1 engine, a 34,816-token context and a 32,768-token
response cap. It used the experimental cache-capable runtime, with `extra_buffer`
recurrent caching, full decode graphs, prefill chunks of 8,192 and static memory
fraction 0.9. These are measured-runtime throughput results; the updated-runtime
deployment check below has a separate scope.

| Admission / HTTP permits | Producer samples | KV tokens | Recurrent slots | Native decode tokens/s | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| 128, opening control | 256 | 4,456,448 | 768 | 8,788 | 53.84 GiB |
| 192 | 384 | 6,684,672 | 1,152 | 11,417 | 68.04 GiB |
| 128, closing control | 256 | 4,456,448 | 768 | 8,529 | 53.96 GiB |
| 224, separate job | 448 | 4,456,448 | 1,536 | 12,469 | 64.76 GiB |
| 256 | 512 | 4,456,448 | 1,536 | 13,321 | 64.80 GiB |
| 320, separate job | 640 | 4,456,448 | 1,920 | 14,654 | 70.26 GiB |

The [128/192/128/256 panel](https://beaker.org/ex/01M5V23DSCQE90HWD86944WHE9), [224 probe](https://beaker.org/ex/01M5V8ASGNV8H28GNSBBFQKDPK) and
[320 probe](https://beaker.org/ex/01M5VA8KDF63EJKHV8MNMHWH4J) completed
successfully. All 1,809 archived files passed hash verification. Measured running
requests averaged at least 99.7% of admission, and all completed measured responses
had timing records with zero reported retractions. The repeated 128 native
control declined 2.95%. At 256, native throughput was 6.83% above 224; delivered
complete-group throughput was 14,323 versus 13,101 tokens/s. Native counters include
unfinished responses, so those rates have different boundaries.

At 320, native throughput increased another 10.01% over 256, with delivered
throughput of 15,501 tokens/s. Treat 320 as the best measured candidate on this
workload, not an established peak or production recommendation. The 224 job ran on a different host with fewer CPU/RAM resources;
the other panel reserved two GPU slots because of its host-memory request while
using only one engine. Per-allocated-GPU throughput for that panel is half the
per-active-engine rate. The 224 and 320 jobs requested 128 GiB host RAM and each reserved one
GPU. Record both denominators when comparing cost.

The 256 and 320 KV pools each hold only 128 full-length contexts: higher admission relies on the
actual length distribution and is not a guarantee that every request can reach
the cap simultaneously. Pool sizes also changed between 128 and 192; 320 adds
recurrent slots and a larger graph ceiling. A
[realistic-prompt comparison](https://beaker.org/ex/01M5VC43SYKSF300X9VYEP6XZ5)
of cache enabled at 320 versus cache disabled at 512 completed successfully:

| Candidate | Native decode tokens/s | Delivered tokens/s | Peak GPU memory |
|---|---:|---:|---:|
| Cache enabled, admission/HTTP 320, producer 640 | 14,485 | 14,730 | 70.23 GiB |
| Cache disabled, admission/HTTP 512, producer 1,024 | 16,714 | 16,796 | 59.36 GiB |

The uncached candidate is 15.4% faster by native counters. It uses 6,684,672 KV
tokens, 512 recurrent slots and decode graphs through 512. Both cases had zero
reported retractions with complete delivered-response timing coverage. All 607
archived files (551.5 MB) passed hash verification; full response-source joins
are retained. This supports 512 uncached as the selected starting point. The comparison
changes multiple capacity settings and has no repeated control, so it cannot
isolate caching or establish a universal admission peak.

The [complete H100 numerical replay](https://beaker.org/ex/01M5VKRGF4159AM219QK9EGDNB)
finished both waves under 512-request background load. All 540 files (1.70 GB)
were downloaded and hash-verified, with no collector omissions. The 512
foreground traces cover 64 prompt clusters and 3,953,730 response tokens including
repeats; all scores and full expert-route payloads passed integrity checks.
Background requests were joined before the next wave and shutdown completed.
Relative to the same frozen Core reference, the paired mean absolute error change
from B300 to H100 was +0.0000064, with a prompt-cluster 95% interval of
[-0.000051, +0.000061]. Both per-wave intervals also include zero. This finds no
aggregate error increase; it does not establish bitwise or learning equivalence.
Expert membership differs by 3.16% (95% interval 2.91–3.39%), with no invalid IDs.
Hardware, cache mode, memory pools and HTTP/background concurrency differ together.
Both arms use the measured experimental runtime. The first 45-minute attempt
timed out and remains separately archived; the identical replacement completed
in 52.5 minutes with a 75-minute budget.

The updated-runtime fully async trainer-consumption check used B300. Do not copy
B300 pools to H100 or enable prefix caching on a runtime that lacks the required
snapshot support.

The [candidate-runtime H100 deployment check](https://beaker.org/ex/01M5VMR1YJ24M6HN74QR578CGS)
completed two batches of 512 requests, each generating 512 tokens, with the
automatically selected uncached profile. All 1,024 response payloads passed an
independent finite-score, token-alignment and full expert-route audit; no invalid
expert IDs or retractions were found. Request timestamps show all 512 requests
had finished prefill before the first completion in both batches. Decode graphs
were configured and captured through 512; these logs do not directly establish
per-step graph replay. Shutdown completed, and all 1,037 files (307.3 MB) were
downloaded and hash-verified. This checks short-generation deployment on the
candidate image, not long-context throughput, Core numerical agreement or
mid-sequence refresh. This is the same dependency set selected in the updated
runtime lock.

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
