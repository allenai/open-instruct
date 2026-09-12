# MILES/Core feature parity audit — September 11, 2026

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

The Core path has the essential synchronous and bounded-async RL machinery,
working serving-capacity controls, full-SFT learning runs, native checkpoint
continuation, and basic rollout routing replay. The largest remaining differences
from olmo-miles are the researcher workflow, failure recovery, batching and
parallel topology, and the breadth of exercised combinations. Sharing MILES and
SGLang does not automatically import the custom olmo-miles drivers or lifecycle.

This audit separates **implemented**, **qualified in a specified experiment**,
**missing**, and **replaced by a different Core mechanism**. A parser option is
neither an implementation nor a GPU qualification. Successful short exercises do
not establish learning equivalence or production endurance.

## Scope and revisions

The baseline is `/home/robert/proj/olmo-miles` at
`07887b783ab254577a6656168dc0e0d21aebfe3d`. The integration is the
`robertb/miles-olmo-core` working branch. The consolidated runtime pins are:

| Repository | Development revision | Change during consolidation |
| --- | --- | --- |
| MILES | `df24d2ed5da4598264c6682f9dbe49aee64acc95` | Retained test work incorporated. |
| OLMo-core | `779b183d81d290e91fc289a47cd32fd9c0bd7311` | Historical qualification evidence incorporated. |
| olmo-sglang | `11ae9f6c4c131ded0ab67420e4f30ea67b924e99` | Existing serving implementation. |

The [runtime lock](../../../runtime/miles/runtime.lock.json) records base revisions,
patch hashes and image identity. Earlier experiments retain their original
revisions; consolidation is not a rerun of those experiments.

The audit enumerated:

- All **136 `MilesSmokeConfig` fields**. The [field inventory](knob-inventory.md)
  has exactly one row for each, with no omissions or extra rows.
- All **145 baseline run CLI arguments**, including **14 names outside that
  dataclass**. Five dataclass names are internal/resolved fields rather than
  direct run CLI arguments; the totals therefore differ by nine.
- The **1,434 unique native option destinations** in the Core parser snapshot.
  This is the exposed parser surface, not a claim that every native feature was
  semantically audited or implemented by Core.
- Every public baseline run-spec section, the current starter profiles, and the
  actual Core driver/model/optimizer/checkpoint consumers behind important flags.

No GPU or cloud experiment was launched for this review. The consolidation's
configuration hardening passed 95 targeted tests, followed by 62 tests after
updating the runtime pins. These are separate test invocations, not a count of
157 unique cases. The GPU evidence below comes from retained qualification runs.

## What carries over today

| Capability | Core implementation and evidence | Remaining difference or qualification |
| --- | --- | --- |
| Ordinary full-SFT RL | Native Core model/optimizer plus shared MILES loss, rollout manager and SGLang serving. Fixed-heldout GSM8K campaigns include 100-update and extended runs with retained generations. | Successful learning runs do not establish identical objectives or trajectories across trainers. |
| Resident colocation | Tiny hybrid-model colocated updates and restart passed on an RTX 4090. | Full 18.5B SFT colocation and rollout-offload transitions are unqualified. **Both projects' normal colocated configurations keep the trainer resident**: baseline `config.py` explicitly emits `--no-offload-train`. Core rejects optional trainer/optimizer offload capabilities. |
| Disaggregation | Full-model B300 qualification uses two EP trainer GPUs and one TP1 serving GPU. More serving engines are configurable. | Multiple-engine scaling, multi-node training and EP8 Core runs remain unqualified. Separate GPU pools do not necessarily mean separate physical nodes. |
| Bounded async | Implemented producer, homogeneous-policy groups, optimizer-step lag budget and pending-prompt cursor. A 24-update-per-arm scheduling pair and a 12-update-per-arm admission-64 pair passed audits. | Combined async + replay + restart/failure endurance remains unqualified; some baseline recovery mechanisms are missing, not merely untested. |
| Multiple steps per collection | Core splits complete optimizer batches and reserves sufficient lag for later steps. | No equivalent full-model combined multistep/async/replay/restart qualification. Starter profiles retain one optimizer step per collection. |
| Serving admission and graphs | Admission 64, decode graphs through 64, 524,288 KV-token slots and 128 recurrent slots reached the requested concurrency without OOM/retractions in the 12+12 trial. | Prefix caching remains off in Core starters. Cached replay and full-model colocated graph/offload lifecycle need dedicated checks. |
| Weight publication | Streaming Core-to-HF tensor interchange with flattened 1 GiB broadcast; colocated IPC is available. Full-model publications transfer about 37 GB in roughly 2–4 seconds in the measured trials. Exact weight checks exist. | No per-step disk HF conversion. Pipeline depth two and other transport alternatives are rejected; additional engine and node layouts need qualification. |
| Router replay | R3 connects serving expert IDs to scoring, training and recomputation. Full-SFT EP2 synchronous qualification passed eight updates, 512 responses and 128 prompt groups, with zero returned-ID mismatches over 19 routed layers. | R2 trainer-recorded replay is rejected. Trainer TP/PP/CP remain one. Final unscored-token auxiliary semantics and an independent selected-gate-weight equality diagnostic remain gaps. |
| Native checkpoint continuation | Schema-2 checkpoints commit model/optimizer, scheduler, RNG, data cursor and policy clock, and validate model/topology. Full-model exact restoration plus two subsequent updates/HF exports passed under controlled inputs/caches. | Same topology only. Saves remain synchronous; background writing, retention and tokens-per-expert cadence are missing. |
| Checkpoint performance | Arithmetic planning, compact storage and balanced replicated ownership are now defaults. Qualified full-model save was 118.466 seconds; fresh-process load was 144.307 seconds, for about 222 GB. | Spawned process writers are exposed but not full-model qualified. The reported save improvement is within Core against its earlier writer, not a Megatron/Core checkpoint-speed comparison. |
| Compiler cache | Actual optional Ray worker integration restores/publishes fingerprinted node-local Triton artifacts, including serving child processes. A tiny two-lifetime Core/SGLang exercise passed. | `core.compiler_cache` defaults false. Full-SFT and multi-node cache lifecycles remain unqualified; broader cache families and FLA prewarming are not interchangeable with this implementation. |
| Math, IF and mixtures | Trusted open-instruct verifier registry, isolated bounded math workers and prepared immutable rows. Full-SFT math/IF trials and a three-source mixture passed. | This is not adoption of the whole baseline task/recipe/manifest catalog. Live code/judge services and broader source families need their own integration and qualification. |
| Evaluation | Fixed 128-question GSM8K in-loop sets and retained generations were used in comparison campaigns. Shared-engine evaluation and explicit HF snapshot export are implemented. | Admission 64 carries into shared-engine eval, but the target of a warm 128-question evaluation under 60 seconds was not measured in the admission trial, which had no eval. Dedicated/external eval configurations are not qualified merely by parser support. |
| Observability | MILES W&B/dashboard controls, full rollout dumps, per-rank contract JSONL, publication timing, startup/eval timing, gradient/update diagnostics. Grouped controls exercised offline W&B. | No turnkey baseline run/status lifecycle, bounded generation-summary sampler, or complete recovery/service-cost reporting schema. Raw trainer debug dumps have an additional limitation described below. |

Evidence: [control exercise](control-exercise-20260911.md),
[admission-64 results](admission64-20260911.json),
[full-model replay](core-replay-full-sft-20260911.md),
[checkpoint qualification](checkpoint-perf-20260911.md),
[tiny startup-cache results](startup-tiny-20260911.json),
[full-SFT datasource results](core-datasources-sft-20260910.json), and
[mixture results](mixture-20260910.json).

The admission-64 pair measured warm response throughput of approximately
1,856 tokens/s synchronously versus 2,281 asynchronously, a 22.9% increase in
that bounded comparison. Completion order changed which prompts were consumed;
this is throughput evidence, not a learning-quality comparison. Async generation
wait measures consumer stall rather than total inference work, so overlapping
stage times must not be summed as serial work.

Replay qualification checked 9,728 returned scoring-router calls and 19,456
training/recomputation calls, with finite nonzero router gradients. It proves
that the requested assignments reach the tested forward/backward path, not that
unforced serving/trainer routes agree or that replay improves learning. Generic
Megatron has more parallel scheduling machinery, but the baseline's custom Olmo
replay also restricts TP/PP/CP; these are not all established baseline capabilities
that Core has lost.

## Configurations can still mean different things

| Setting | Baseline behavior | Core behavior and implication |
| --- | --- | --- |
| Async objective | `fully_async` always adds TIS and forbids `use_rollout_logprobs` in the baseline wrapper. | The async starter selects rollout probabilities as the policy anchor and does not enable TIS. Both mechanisms are available, but these starter defaults are different algorithm choices. |
| Async queue capacity | Default factor 2.0. | Async starter uses factor 1.0. The measured scheduling pairs deliberately specify their own common objective and queue settings. |
| `use_rollout_logprobs` | Can avoid a separate trainer scoring pass when other requirements permit. | Core always scores for behavior-policy agreement, then the shared loss uses rollout probabilities as denominator when selected. This is not a Core scoring-performance shortcut. |
| Group geometry | Historical lightly-SFT comparison used eight prompts × eight generations; newer baseline examples also use other explicit shapes. | Full-SFT starter uses 16 prompts × four generations. Both give 64 responses, but different group sizes change advantage statistics. |
| Microbatch | Auto-selects a fitting power of two up to 16; explicit larger microbatches and other batching controls exist. | One unpadded sequence per microbatch; dynamic batching rejected. `data_pad_size_multiplier` does not change Core model padding. |
| Recomputation | Full/selective/off, with module selection. | Block activation checkpointing on/off. Megatron selective module names do not translate into Core options. |
| Auxiliary loss | Baseline has explicit auxiliary wiring and its padded-layout semantics. | Core uses unpadded samples and an explicit model-token denominator. Matching 0.01 balancing and 1e-5 z-loss coefficients does not establish matching gradients. |
| Learning rate | Baseline convenience defaults include constant 1e-6, Adam beta2 0.95, zero weight decay. | Shared scheduler/Adam controls work; explicit starters match these values. Raw native parser defaults should not be assumed to match the baseline convenience wrapper. |
| GRPO | Baseline wrapper disables standard-deviation normalization and uses upper clipping 0.28. | Explicit full-SFT starters match. Native parser defaults are not automatically rewritten to the baseline recipe. These are independent objective choices. |
| Save cadence | Baseline default every 15 collections, async saving enabled. | Full-SFT starter saves at the final collection initially; native saves are synchronous. Choose recovery cadence deliberately for long jobs. |
| Resume evaluation | Baseline suppresses the initial evaluation on a resumed process. | Core repeats the restored initial-state evaluation when `skip_eval_before_train=false`; this can add a duplicate boundary point and overhead. |
| Policy drift threshold | Configurable warn/fail action. | Configured violations fail. The threshold is an active-token **mean absolute** log-probability gap, despite the legacy `max_...` name. |
| Recurrent/prefix cache | Baseline ordinary defaults enable radix cache with a larger recurrent pool; replay recipes may disable it. | Starters disable radix cache and use their measured recurrent budget. Prefix-cache publication invalidation and replay coverage still need qualification. |
| Compiler cache | Baseline public run-spec defaults enable its cache lifecycle. | Core cache implementation exists but remains opt-in. Dynamic row specialization is a separate option, selected by RL starters. |

For source-level detail see baseline `src/olmo_miles/config.py:269`, `:403`,
`:1234` and `:1239`, and Core [actor.py](../../../open_instruct/miles/actor.py),
[scheduler.py](../../../open_instruct/miles/scheduler.py),
[async buffer](../../../open_instruct/miles/async_buffer.py), and the
[starter profiles](../../../configs/miles/README.md).

Raw defaults are not a recommended full-model recipe. `CoreConfig` defaults to
EP1, torch attention, an 8,192-token limit, static row specialization, cache off
and lag zero. Full-SFT starters choose EP2, flash_4, 6,144 tokens, dynamic rows and
lag zero or one. Baseline dataclass defaults are eight trainer GPUs/EP8,
colocation, a 4,096 context, four prompts × two responses and automatic
microbatching; maintained examples and historical runs override them.

Both public surfaces let users express many serving settings. The baseline also
checks response/context relationships and recurrent-cache capacity against
admission/radix mode. Core planning currently checks the individual values more
than these coupled constraints. A future Core check should be architecture-aware:
KDA state-slot requirements should not be imposed on a dense model. Increasing
one concurrency setting alone is not equivalent to increasing the full engine
capacity budget.

## Unsupported native controls corrected during this audit

The native parser snapshot includes options implemented by other trainers. This
review traced their consumers and found several that previously compiled despite
having no corresponding Core operation. They now fail early in both the CPU
facade and direct native argument validation:

| Control | Enforced Core contract | Why |
| --- | --- | --- |
| `no_save_optim` | False. | Native Core save includes optimizer state; weights-only saves were not wired. |
| `reset_optimizer_states` | False. | Core does not implement the requested optimizer reset. |
| `override_lr_scheduler`, `use_checkpoint_lr_scheduler` | False / true respectively. | Core restore follows the checkpoint scheduler; an override request previously had no effect. |
| `compute_advantages_and_returns` | True. | The actor computes advantages unconditionally; disabling it previously overwrote rather than preserved supplied advantages. |
| `skip_actor_forward_only` | False. | Core always scores. The installed MILES parser already restricted its skip implementation to Megatron; CPU planning now agrees. |
| `keep_old_actor` | False. | Core has current/reference modules, not Megatron's extra old-actor backup and switching. |
| `dp_replicate_size` | One. | Core owns its DDP/EP or dense FSDP topology; this native FSDP mesh setting was ignored. |
| `deterministic_mode` | False. | This trainer-specific switch has no Core implementation. The separate `debug_deterministic_collective` remains distinct and is consumed by the shared base actor. |
| `lora_rank`, `lora_train_only` | Nonpositive / false. | Core builds a full-parameter model and optimizer; it does not inject LoRA adapters. |
| `save_hf` | Unset. | Core native save does not also export HF. The dispatcher previously assumed this path already existed, potentially pointing snapshot evaluation at a nonexistent export. Use `eval_hf_dir` for implemented snapshot export, or export separately. |

These are fixed interface defects, not newly implemented capabilities. Normal
working profiles retain their behavior. The guards and tests are in
[config.py](../../../open_instruct/miles/config.py),
[arguments.py](../../../open_instruct/miles/arguments.py), and
[test_options.py](../../../tests/miles/test_options.py).

**Remaining debug-dump limitation:** `save_debug_train_data` does not currently
invoke Core trainer-payload dumping. Full rollout dumps and contract JSONL do
work. Native `dump_details` also auto-populates the trainer-dump field, so blindly
rejecting every non-null value would interfere with the broader dump workflow.
This needs a deliberate implementation or an explicit partial-support contract.
The audit also does not certify all other inherited actor-specific options,
including profiler/memory-history controls, FSDP state-dict settings, alternate
trainer attention selectors, Megatron hooks, custom model-provider paths and
trainer dumpers. Some fail later in native validation; some require consumer
review. The 1,434-option surface remains larger than the reviewed Core contract.

## Public workflow and data parity

The baseline run file is more than training arguments. Its public command plans,
validates locally, submits Beaker, prepares data, converts/validates the model,
trains and optionally exports. `olmo-miles status` reads launch history and
lifecycle progress. Core's `plan / validate / train` facade is useful and real,
but the surrounding workflow is currently assembled by experiment scripts.

Baseline sections `training`, `trainer`, `inference`, `optimizer`, `async`,
`tracking` and `runtime` all feed one common run-option namespace. Only
`trainer.gpus` and `inference.gpus` have special aliases. The Core run parser
accepts exactly `[core]` and `[miles]`; the following baseline sections have no
implicit translation:

| Baseline section and fields | Core replacement or missing workflow |
| --- | --- |
| Identity: `schema_version`, `name` | Run filename, harness metadata and W&B run name; no equivalent lifecycle identity section. |
| `model`: `source`, `format`, `hf_template`, `reference_hf` | HF initialization/reference plus native RL `load`. Pretraining-native conversion is separate tooling, not automatic model-format detection inside a run plan. |
| `conversion`: `output`, `hf_output`, `advertised_max_context_length`, `dtype`, `device`, `trust_remote_code`, `low_memory_save`, `allow_router_weight_cast` | Core interchange and experiment-specific conversion scripts replace Megatron conversion. Generic cached conversion/publication stages are not exposed in the facade. |
| `conversion_validation` / `validation` | Separate retained-input, activation, route and training-contract probes exist. They are not a generic selectable pre/post-run conversion gate. |
| `data`: `seed`, `shuffle`, `tasks`, `recipe`, `rl_manifest`; task entries select task/train/eval counts | Prepared JSONL, verifier configuration and datasource/mixture harnesses. No direct baseline manifest importer or full task/recipe catalog bridge. |
| `output`: `root`, `export_hf`, `hf_dir` | Split native-save, rollout-dump, tracking and eval-export paths. Final standalone HF export remains a missing public lifecycle stage. |
| `compiler_cache`: `enabled`, `shared_root`, `local_root` | Core cache enabled/root/restore/diagnostics controls; worker-local temporary directories are implementation-owned. Cache-family coverage and prewarming differ. |
| `launch`: workspace, budget, cluster, shared memory, priority, minimum runtime, auto-resume, GPUs per replica, WEKA mounts, environment and secrets | Python launch scripts plus `build_image_and_launch.sh --miles`. No generic submit/status/resume command driven by the training TOML. |

The baseline conversion gate exposes input IDs/prompts, dtype/device/seed,
HF MoE backend, logit/router cosine and route-overlap thresholds, capture retention,
operator counterfactual warmup/measurement counts, training/post-training parity,
relative-loss/gradient thresholds, projected LR, generation length, serving memory,
sampling seeds/temperature/top-p and remote-code trust. Core has useful probes for
many of these questions, but probe existence is not equivalent to an integrated
workflow gate.

The baseline catalog covers 41 static pinned sources plus generated arithmetic,
recipes, wrappers, overlap checks and manifest metadata. Core's concrete math,
IF and mixture work is more than a placeholder, but does not expose all of that
catalog through a public preparation contract. Existing open-instruct datasource
or verifier classes outside this facade are not automatically bridged.

Code and judge services require more than importing verifier classes: service
startup, isolation, timeouts, limits, cancellation, cleanup and provenance must
be connected. Baseline coverage itself includes bounded real code-executor tests
and mock judge HTTP tests; mock success must not be called real judge-quality
qualification. Neither project's exposed extension hooks establish a completed
stateful multi-turn/tool recipe. Token masks and extension points are useful
building blocks; a chosen episode/task contract is still required.

The 14 baseline run CLI names outside `MilesSmokeConfig` are:

| Names | Core status |
| --- | --- |
| `disable_radix_cache` | Native `miles.sglang_disable_radix_cache`. |
| `miles_train_script` | Replaced by the Core-owned driver; not an arbitrary script-selection knob. |
| `python_path` | Image/PYTHONPATH/runtime environment provisioning. |
| `fla_prewarm`, `fla_prewarm_sequence_length` | No generic Core prewarming option; persistent compiler cache is different. |
| `skip_cuda_check` | CPU plan already avoids CUDA; installed-runtime validation is a separate operation. |
| `validate_miles_args` | Public `validate` subcommand. |
| `no_start_ray` | Launcher/runtime ownership. |
| `start_code_service`, `code_service_source_root`, `code_service_python`, `code_service_host`, `code_service_port`, `code_service_log` | No equivalent managed per-run code-service lifecycle. |

The five dataclass-only/resolved names are `capture_generation_samples`,
`code_service_mode`, `code_service_source_revision`, `radix_cache` (inverse CLI
alias), and `wandb_dir`. All 136 dataclass fields remain individually documented
in the [inventory](knob-inventory.md).

## Reliability, models and scale: missing versus unqualified

**Missing code or public workflow:** bounded driver-stage timeout/retry budget,
recovery-time engine memory policy, a complete serving-interruption retry and
republish loop, automatic trainer-cell recovery, generic Beaker resumption from
the latest durable boundary, final HF export stage, background checkpoint saves,
retention and tokens-per-expert cadence, bounded generation-summary sampling,
and the broader task/service workflow described above.

Shared MILES health flags and Core's new-engine reconnect hook do not provide the
baseline recovery contract. Core's driver does not implement the full catch,
communicator retirement, republish and prompt-conservation retry sequence. This
is an implementation gap before it is an endurance-testing gap.

**Implemented but narrower or unqualified:** higher DP/EP node counts, more
serving engines, full-model colocation, optional compiler cache on full SFT,
combined async/replay/restart/failure, multiple updates per collection under that
combination, high-concurrency heldout-eval latency, and external/dedicated eval
fleets. Trainer TP/PP/CP and microbatch packing need real implementation changes,
not simply another launcher value. Router payload ownership, sequence identity,
recomputation ordering, normalization, checkpointing and weight export all need
to follow the new topology.

The original dense Olmo 3 trainer is isolated in
[standard_models.py](../../../open_instruct/miles/standard_models.py). It has
architecture/conversion support and local update/resume evidence, but not a full
original Olmo 3 training recipe qualification. Core H100 MoE execution is likewise
unqualified; the baseline's smaller-model H100 lifecycle does not prove this
full-model path portable.

Hero support is a separate unfinished qualification: native/HF conversion checked
23,441 tensors exactly at step 75,500, while production forced-prefix serving
probability checks failed. A controlled common-operator forward reproduced exact
results and helped localize arithmetic differences; that is diagnostic progress,
not full hero RL acceptance. See [conversion](hero-conversion-20260910.json),
[serving](hero-serving-20260910.json), and
[controlled forward](hero-controlled-forward-20260911.json).

## Recommended order of work

1. **Keep the interface honest.** The confirmed unsupported-control guards are
   now fixed. Resolve raw trainer dump semantics and review remaining trainer-owned
   options before treating native parser breadth as a supported API. Keep current
   qualification summaries linked to retained measurements.
2. **Complete the ordinary researcher workflow.** Add prepared-task/manifest
   adoption, consistent output roots, launch/status, explicit durable resume and
   final HF export around the existing facade. Reuse open-instruct data/verifiers
   instead of copying a second catalog wholesale. Preserve the difference between
   conversion validation and runtime argument validation.
3. **Implement recovery, then test combinations.** Add bounded stage waits,
   retry budgets, safe communicator replacement, republishing and pending-group
   accounting before promising unattended operation. Run one combined
   async + replay + checkpoint/restart/fault exercise rather than repeating the
   already-passed basic synchronous replay gate.
4. **Qualify small operational promotions.** Full-SFT compiler-cache lifecycle,
   two-engine scaling and high-concurrency evaluation can be folded into useful
   subsequent runs. Promote defaults based on those results, retaining hardware
   and source isolation. Compare async objectives explicitly.
5. **Broaden sources in bounded groups.** Real code-service and mock-judge plumbing
   first, then representative live judge/IFBench/source-mixture work. Keep tool
   episodes separate until a task actually requires the bridge.
6. **Treat batching and topology as trainer projects.** Microbatch/packing,
   selective recomputation and PP/TP/CP require token-mask, auxiliary-normalization,
   route-identity and same-update tests. Pursue background checkpointing/retention,
   full-model colocation, H100 and complete hero/dense-model recipes with their own
   measured acceptance gates.

Historical documents contain older pending statements and smaller capacity
settings. They remain useful experiment records; the capability claims in this
audit use later completed evidence where available. In particular, compiler-cache
code, full-model async, math/IF/mixture runs, replay and checkpoint continuation
should no longer be described as wholly absent or awaiting their first exercise.

## Subsequent workflow implementation

This audit remains a point-in-time record. Later structured run files, task and
manifest preparation, one-node Beaker launch/status, native auto resume, optional
final export, and the 8 × 8 async-TIS defaults are described in the
[current researcher workflow](../workflow.md). Implementation does not
retroactively qualify those paths or change the measured configurations above.
