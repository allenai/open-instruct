# MILES/Core defaults and feature parity

For the current CLI/API surface, overrides and slide-level run controls, see the
[run-control guide](miles-run-controls.md) and [complete knob inventory](miles-knob-inventory.md).

Current experiment status and the restored matched Megatron comparison are tracked
in [the active-work ledger](miles-active-work.md). The architecture-specific hero
port is documented in [hero support](miles-hero-support.md).

This compares the Core integration with **our customized `~/proj/olmo-miles`
implementation**, including its Megatron adapters, serving optimizations and
operational tooling. It does not compare against unmodified upstream MILES.
The review used olmo-miles `07887b783ab254577a6656168dc0e0d21aebfe3d` on
September 10, 2026. Core starts from Jacob's `moe-v2-core-gdn2` branch;
[runtime.lock.json](../runtime/miles/runtime.lock.json) owns the exact runtime pins.

“Implemented” means there is code for the capability. “Validated” always names
the scope of the evidence. Parser acceptance alone is neither runtime support
nor a qualification. The live evidence ledger is
[validation.json](../runtime/miles/validation.json), with the
[contract checks](miles-core.md#training-contract-checks) defining the numerical
acceptance work. Existing GRPO entrypoints remain available.

## Starting profiles

The [starter index](../configs/miles/README.md) identifies the default tiny
colocated dev/test profile and the longer disaggregated training starter. The
short profiles below remain qualification exercises, not a model-capacity table. They use the actual `[core]`/`[miles]` configuration
schema and preserve open-instruct verifier dispatch. No additional config
inheritance or environment-variable substitution is implied.

| Profile | Topology and purpose | Starting shape | Evidence / limitation |
| --- | --- | --- | --- |
| [tiny-resident](../configs/miles/profiles/tiny-resident.toml) | One GPU shared by a tiny Core model and one SGLang engine | 1 prompt × 4 responses; 2 updates; context 512, response 256; eager decode; Torch attention | This profile passed two updates plus a separate-process third update through the public entrypoint with schema-2 checkpoints on the random tiny KDA/latent fixture. Memory fit remains model-dependent. All-zero rewards validate plumbing, not policy learning. |
| [sft-b300-ep2-sync](../configs/miles/profiles/sft-b300-ep2-sync.toml) | Two Core EP ranks plus one dedicated SGLang GPU | 16 prompts × 4 responses; 2 updates; context 6144, response 4096; admission/decode graphs through batch 64; FA4 | Based on the successful full SFT path. Admission is raised from the historical four to 64, with 524288 KV tokens and 128 recurrent slots; capacity and scheduling passed the 12-update B300 pair; this two-update eval/diagnostics recipe is a separate combination. |
| [sft-b300-ep2-async-candidate](../configs/miles/profiles/sft-b300-ep2-async-candidate.toml) | Same three-GPU allocation; bounded asynchronous generation | 16 prompts × 4 responses; 4 updates; context 2560, response 512; lag ≤1; one collection buffered; eager decode; admission 64 | Candidate for async qualification only. Core's bounded queue and ledger have targeted tests; full-model async endurance, restart and failures need their own run. Replay stays off. |

The additional [train-disaggregated](../configs/miles/profiles/train-disaggregated.toml)
starter uses the same 64-completion/64-admission shape, 100 updates, initial and
every-20-update heldout evaluation, a final native checkpoint and offline W&B.
It keeps the established GSM8K objective explicit. See the starter index for
the standalone async training example, overrides and the memory constraints on engine scaling.

All profiles explicitly select `core.row_specialization="dynamic"` for forward-only
routed-expert scoring; Core defaults remain static. See the
[row-specialization integration](measurements/miles-core-row-specialization-20260911.md)
for isolation, rollback and qualification scope.

All profiles keep trainer offload disabled (the compiler explicitly supplies
`--no-offload-train`), microbatch size one, activation recomputation enabled,
router auxiliary coefficient 0.01 and z-loss coefficient 1e-5. They enable initial
serving equality, a 0.05 active-token mean logprob-difference guard and
per-step diagnostics in the short qualification profiles; the longer training
starter keeps extra diagnostic republications off. With serving checks enabled, each diagnostic publication
adds a snapshot/reset/republish round trip; include its extra transfer time when
measuring performance. The logprob tolerance is inherited from our bounded
checks; it is not a universal acceptable drift, particularly under async lag.
A violation should trigger diagnosis, not automatic relaxation.

The synchronous SFT profile uses the same actor-recomputed logprob baseline
as the successful trial. The async candidate explicitly uses rollout behavior
log probabilities as its policy-ratio denominator; stale data must not silently be
treated as current-policy data. Neither profile turns on replay, TIS, reference
KL, reward shaping or a changed advantage estimator. Those are independent
algorithmic choices requiring their own acceptance, not throughput toggles.

Before running, copy a profile and replace every `/data` path with a fresh,
mounted run directory. Prepare its HF descriptor (weights, config, tokenizer and
exact RL chat template), rendered `train.jsonl`, and trusted `verifiers.json`.
The synchronous SFT profile also expects a disjoint `eval.jsonl`. Inputs use
`input`, `label`, and `metadata`; metadata identifies registered verifiers and
targets. Do not render the template a second time or include reference assistant
answers in prompts. The existing
[SFT preparation script](../scripts/miles/sft_gsm8k.py) demonstrates the exact
checkpoint/template and immutable GSM8K revision used for the successful run.

```bash
# CPU-safe compilation: does not check installed runtime or input files.
python -m open_instruct.miles plan /path/to/run.toml
# Inside the pinned runtime, with the real HF descriptor visible:
python -m open_instruct.miles validate /path/to/run.toml
python -m open_instruct.miles train /path/to/run.toml
```

Run FA4's real forward/backward preflight before the B300 profiles, as the
[SFT launcher](../scripts/miles/launch_sft_trial.py) does. Beaker launches still go
through the repository's committed image-build wrapper. These TOMLs do not
allocate GPUs, mount WEKA, configure Beaker retries or set compiler caches.
CPU-only preparation requiring WEKA belongs on `ai2/saturn`; this does not
change GPU job placement.

The two SFT profiles intentionally omit `save_interval`: `save` provides a
metrics destination, not optimizer-checkpoint durability. The tiny profile
saves every rollout to exercise native checkpoint completion. For a restart,
keep topology/model settings fixed, set `miles.load` to that checkpoint root,
and choose a total `num_rollout` and original LR horizon that cover the intended
continuation. A production run must choose and measure real checkpoint cadence;
these bounded SFT defaults provide no recovery checkpoint.

## Additional datasource trials

[The datasource harness](../scripts/miles/datasource_trials.py) has pinned
[math](../configs/miles/tasks/math.toml) and
[legacy IF](../configs/miles/tasks/ifeval.toml) task specifications. These task
TOMLs are inputs to the harness, not standalone Core run configurations. It
selects the first 24 unique in-budget prompts from at most 256 rows, uses eight
for two updates, and reserves sixteen for before/after evaluation. It records
actual source indices and prepared-data hashes; this bounded length-selected
slice is not the full dataset or a decontaminated benchmark.

Local qualification completed for both sources on a fresh random 13.66M-parameter
KDA + latent-MoE model: two updates and 64 independently audited responses per
source, with exact policy-version and diagnostic-republication checks. All
responses reached the 32-token cap, every training group had zero reward
variance, and the policy objective was zero; these runs exercise source/reward
and auxiliary-update plumbing, not task learning. The IF GPU lifecycle completed
but its first audit hit an async-helper bug; the corrected helper successfully
re-audited the unchanged outputs. Math completed with exit code zero. See the
[combined measurements](measurements/miles-core-datasources-local-20260910.json)
for raw contract metrics, source hashes, verifier fixtures, and limitations.
The full-SFT datasource launcher below remains a separate qualification step.

After the numerical and verifier preflights pass and changes are committed:

```bash
MILES_BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks \
  ./scripts/train/build_image_and_launch.sh --miles \
  scripts/train/debug/miles_core_datasources.sh
# Append --task math or --task ifeval to run just one datasource.
```

The [launcher](../scripts/miles/launch_datasource_trial.py) runs math then IF
sequentially within one three-GPU Holmes allocation. Each starts independently
from the same SFT source through a fresh shared HF descriptor with the verified
RL template; the second task does not continue the first task's updated model.
It runs verifier fixtures before model startup, then prepares, validates and
runs each task with per-step diagnostics and independent response audits. The
pinned image isolates symbolic-math dependencies from the trainer. Default wall
time is bounded to 90 minutes for both tasks, or 45 minutes for one task, with
no automatic restart. Reports are copied into the Beaker result; weights and
responses stay on WEKA. A failed task stops the allocation, preserving completed
reports for diagnosis. Preparing a launch is not evidence that it passed.

## What carries over, and what still needs work

The sources named below are files in the reviewed olmo-miles checkout. Local
Core implementation links point to this repository's adapter. Baseline features
may themselves have limits; their presence is not a blanket performance claim.

| Capability | Customized olmo-miles baseline | Core integration and acceptance still needed |
| --- | --- | --- |
| Native KDA / latent MoE | Custom Megatron/Bridge model representation and conversion | Core already owns the architecture; adapter constructs it from HF and loads native weights. Full SFT EP2 two-update run passed; longer runs remain. [models](../open_instruct/miles/models.py) |
| HF initialization / export | HF↔Megatron conversion, config manifests and parity tooling | HF→native Core import and native→HF export replace Bridge. Exact initial serving-weight equality and tiny round trips passed. Broader configurations and conversion parity remain. |
| RL objective / accumulation | MILES Megatron schedule plus Olmo-specific auxiliary-loss wiring | New arbitrary-objective Core hook, rank/microbatch normalization and optimizer lifecycle. Independent fixed-batch contract suite passed. Native EP1/EP2 moments and recomputation passed twelve replayed-route arms; matched Megatron measurements remain. [EP evidence](measurements/miles-core-native-ep-20260910.json) [actor](../open_instruct/miles/actor.py), [contract](../open_instruct/miles/contract.py) |
| MoE auxiliary losses | Explicit 0.01 balancing / 1e-5 z-loss and backend scaling | Same coefficients; Core per-sequence router objective with an explicit model-token denominator. Policy-only, auxiliary-only and combined gradient tests matter separately. Equal coefficients do not establish equal training semantics. |
| Router replay | Layer mapping, replay layout and serving/trainer alignment fixes; live replay has dedicated baseline evidence | `use_rollout_routing_replay` connects captured routes to Core's override and recomputation context. This differs from Megatron's `use_routing_replay`, which Core rejects. Router-gradient tests exist. The first live SGLang attempt failed before updating because the pinned SGLang router stripped expert-ID requests; configuration now requires `use_miles_router=true` for rollout replay. The retry passed two updates, separate-process restart, a third update and a 12-response route/reward audit. Full-model synchronous EP2 replay subsequently passed eight updates, 512 samples and an independent audit including recomputation; TP/PP/CP remain 1. Final unscored-token auxiliary semantics remain unqualified. [Full-model evidence](measurements/miles-core-replay-full-sft-20260911.md) [Evidence](measurements/miles-core-replay-local-20260910.json) |
| Efficient weight sync | Custom direct exporter, flattened transport, 1 GiB bucket screens and publication instrumentation | Streaming Core exporter replaces Megatron export; existing transport design is reused. EP2 SFT warm publications of 37.0 GB took 3.82/3.83 s. No per-step disk HF conversion. These are one-run measurements, not backend speed parity. [publication](../open_instruct/miles/publication.py) |
| Resident colocation | Trainer stays resident; rollout memory can be offloaded; graph/offload hooks validated in baseline | Tiny resident colocation passed. Full SFT colocation and rollout offload transitions need their own memory/update test. Do not copy baseline EP2 memory fractions onto Core without measurement. |
| Trainer offload | Not required by the tested baseline resident configuration | Rejected by Core, including CPU optimizer offload. There is no hidden support inherited from the MILES flag. |
| Disaggregated placement | Dedicated serving pool, including validated baseline async shapes | Tiny EP1/EP2 and full SFT EP2+one engine passed within one node. Multi-node and scaling curves remain. |
| Bounded async | Policy clocks, group admission, recovery ledger; baseline multistep/endurance evidence | Managed producer and homogeneous-policy queue with lag reservation; publication pauses and durable pending prompt ledger. Targeted tests exist; full-model async endurance is not established. [async buffer](../open_instruct/miles/async_buffer.py), [driver](../open_instruct/miles/driver.py) |
| Multiple updates per collection | Explicit steps-per-rollout clock and restart protocol | Config requires complete optimizer batches and sufficient lag budget; adapter splits the collection. Multi-step live async parity needs a separate test; defaults keep one step per collection. |
| Decode CUDA graphs | Decode-only graph/offload compatibility and matched performance screens | Disaggregated full SFT ran with sizes 1/2/4; prefill disabled. Full-model colocated graph/offload lifecycle remains unqualified. Replay may change graph compatibility; qualify it separately. |
| Prefix / recurrent caches | Tuned KDA state pools and radix behavior; replay recipes deliberately disable prefix caching | Profiles disable radix caching and bound recurrent slots. Prefix caching is not promoted without route/logprob and publication invalidation tests. |
| Compiler cache lifecycle | WEKA-backed, fingerprinted restore/publish of node-local compiler artifacts | No equivalent launch lifecycle yet. Current SFT and datasource launchers use temporary HF cache and incur substantial cold compilation. Raw TOML options do not implement cache persistence. |
| Microbatch throughput / packing | Baseline measured larger fixed microbatches; dynamic packing has its own router-loss limitations | Core accepts only one unpadded sequence per microbatch and rejects dynamic batching. Batching/packing needs independent token masks, auxiliary denominators, routing and same-update tests. |
| Expert / dense parallelism | Megatron parallel runtime plus optional optimized DeepEP/DeepGEMM path | Native Core MoE DDP with expert parallelism and sharded optimizer state; non-MoE models use FSDP. EP2 tiny and SFT paths exercised; trainer TP/PP/CP >1 rejected. Megatron optimized/compatibility backend flags have no direct Core meaning. |
| Native checkpoint / restart | Megatron tensors, optimizer, scheduler, RNG, data cursor and policy-clock sidecar | Native Core state plus scheduler/RNG, durable cursor, completion marker and architecture/topology checks. Tiny separate-process restart passed; full SFT restart and longer failure recovery remain. [checkpoint](../open_instruct/miles/checkpoint.py) |
| Asynchronous checkpoint writes | Background staging/storage and cadence instrumentation; baseline default every 15 rollouts | Core driver completes synchronous checkpoint boundaries. Background writer/retention/capacity management are additional work; do not advertise async-save parity. |
| Serving fault recovery | Customized health probes, replay-compatible recovery and failure evidence | Some serving machinery is inherited, but Core recovery with restored weight versions and in-flight group accounting is not qualified. Automatic trainer-cell recovery is explicitly unsupported. |
| Reward and source mixtures | Prepared RL manifests, source recipes, verifier dispatch and coverage inventory | Trusted registry calls open-instruct verifier classes; multiple weighted components supported. Bounded immutable-revision math and legacy IF each completed a local tiny-model two-update/64-response lifecycle and independent audit. Advantages were zero; this is plumbing qualification, not full-SFT task performance. See [measurements](measurements/miles-core-datasources-local-20260910.json). [rewards](../open_instruct/miles/rewards.py) |
| Code / judge rewards | Baseline has code/judge infrastructure and domain-specific acceptance work | Verifier adapter is a connection point, not a running sandbox or judge service. Symbolic math now uses an isolated subprocess/dependency path; code/judge service setup, limits, cleanup, timeouts and representative live reward checks remain required. |
| Multi-turn tools | Baseline and upstream have distinct rollout/environment integrations | Token masks survive the Core adapter, but complete open-instruct environment/tool bridge remains to migrate and qualify. |
| Evaluation | Native and HF export paths, fixed heldouts and broader evaluation tooling | MILES dispatcher reused; before/after 16-question GSM8K plus independent 64-response audit passed. Full task matrix, external evaluation workflows and learning-quality conclusions remain. |
| Tracking / observability | Comparison dashboards, run-event streams, phase timing and recovery metrics | MILES tracking plus publication and per-rank training-contract JSONL; gradient/update probes on demand. Baseline dashboard schemas, export events, reward service cost and recovery dashboards are not all ported. |
| SFT / DPO handoff | Baseline supports its own SFT/conversion workflows | Open-instruct's existing SFT/DPO paths remain separate. Starting from SFT works; this backend integration is not a DPO or release-recipe reproduction. |

## Which performance defaults should transfer

The baseline's documented findings are useful hypotheses, but Core changes
training memory and execution. Transfer settings with the matching contract:

- **Keep 1 GiB publication buckets and streaming export.** Baseline
  `docs/measurements/weight-sync-buffer-screen.md` established that starting point;
  our full SFT run also used it. Compare steady-window phase timings and exact
  weights before changing bucket size or transport overlap.
- **Capture decode only, at the actual concurrency.** Baseline
  `docs/measurements/decode-cuda-graph-screen.md` measured benefit. Our first
  attempt at generic graph enablement also captured many unwanted prefill sizes;
  the successful Core run explicitly disabled prefill and capped decode at four.
- **Choose response length first, then reserve the prompt budget.** Baseline
  `docs/topology-and-length-guide.md` distinguishes context from response length
  and total from active parameters. The SFT profile reserves 2048+4096 tokens.
  Shortening responses to make a test cheaper changes the task, so report cap
  hits and do not interpret truncated scores as task-quality parity.
- **Size client admission, engine admission, graphs and both cache pools together.**
  The new full-SFT baseline admits all 64 completions, reserves 524288 total KV
  tokens and 128 recurrent slots, and captures decode graphs through 64. Historical
  runs used four requests, 32768 tokens and eight slots. The larger baseline passed the 12-update sync/async B300 trial. Tiny fixtures retain four requests/4096 tokens. The collection and optimizer batch rise from 16 to 64 to supply enough work;
  each additional engine needs more queued samples to sustain the same occupancy;
  dedicated memory fraction stays 0.6, rather than copying 0.85 blindly.
- **Use recomputation initially.** Disable it only in a measured memory and
  same-update comparison. Core's microbatch one stays explicit until larger
  batches preserve the complete loss/replay contract.
- **Persist compiler artifacts with provenance, not a shared writable hot cache.**
  Port the baseline restore-to-node-local/publish-on-success approach. Key it by
  compiled image/runtime, GPU architecture, kernel versions and compiler inputs;
  use a WEKA TTL root such as `tmp-30d`. Dataset/tokenizer caches are a separate
  cache and need their own revision identity. Do not copy Megatron compiler
  artifacts into Core without an exact compatibility key.

The Core SFT timing record is
[measurements/miles-core-sft-20260910.json](measurements/miles-core-sft-20260910.json).
Its graph-enabled generation and earlier eager failure are different runs;
those values cannot establish a matched end-to-end speedup against Megatron.

## Promotion order

The [qualification plan](miles-qualification-plan.md) specifies bounded experiment
shapes, the cross-backend comparison contract and criteria for promoting each
default. The sequence below summarizes those gates.

1. Extend the passing fixed-global-batch gates (native EP, recomputation,
   optimizer moments, isolated auxiliary gradients and tiny restart) to a matched
   batch through the customized Megatron trainer under matched semantics.
2. Qualify each new data/verifier family with immutable revisions and independent
   reward audits. Include nonzero policy signal, per-source denominators,
   truncation and latency; then test a mixed-source collection.
3. Establish full-model checkpoint/restart and several consecutive updates.
   Capture compiler-cache cold/warm cost and real save latency before promoting
   a persistent-cache and checkpoint-cadence default.
4. Complete live route replay alignment, including prompt/response offset,
   every routed layer, replay through recomputation and the unscored final
   generated token. Compare router policy/auxiliary gradients with replay off/on.
5. Qualify bounded async on the already qualified task/model: lag distribution,
   useful versus discarded groups, publication integrity, restart ledger and
   injected serving failures. Then test multiple optimizer steps per collection.
6. Measure full-model resident colocation and graph/offload transitions against
   the same disaggregated workload. Optimize batching/packing only after its
   same-update contract passes. Keep multi-node and larger topologies separate.

Relevant baseline sources in `~/proj/olmo-miles` are
`docs/runtime-integration.md`, `docs/checkpointing.md`,
`docs/disaggregated-rollout.md`, `docs/topology-and-length-guide.md`,
`docs/rl-dataset-status.md`, `docs/measurements/async-multistep-20260907.md`,
`examples/qualification/async-multistep-b300.toml`, and
`src/olmo_miles/runtime/{compiler_cache,cache_lifecycle}.py`.
The baseline's Olmo 3 recipe review is useful for task mixture and stage budgets,
but does not certify this different KDA model or the Core training objective.

All three templates were loaded with `RunConfig` and passed the actual pinned
MILES argument parser using a local tiny KDA/latent HF descriptor in place of the
external input path. This validates configuration compatibility, not the target
model's memory fit, data contents, serving startup or completed training.
