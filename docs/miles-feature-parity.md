# MILES/Core defaults and feature parity

The Core backend supports ordinary synchronous and bounded-async RL, live weight
publication, native checkpoint/resume, and rollout router replay on the tested
EP2 configuration. The main remaining differences from **our customized
`~/proj/olmo-miles`** are preparation/launch workflows, recovery operations,
checkpoint policies, and trainer topology/batching support. Shared MILES or
SGLang flags do not automatically provide the baseline's custom adapters.

This September 11 review uses olmo-miles
`07887b783ab254577a6656168dc0e0d21aebfe3d`. The current implementation is on
`robertb/miles-hero-support`, with Core `robertb/miles-hero-adapter` and MILES
`robertb/olmo-core-backend`. The adapter originated on Jacob's
`moe-v2-core-gdn2` work and was ported to the later hero HF-support branch.
[runtime.lock.json](../runtime/miles/runtime.lock.json) owns the exact source
pins; individual measurement records identify the revisions actually exercised.

Use the [detailed audit](measurements/miles-feature-parity-audit-20260911.md),
[run-control guide](miles-run-controls.md), and
[complete knob inventory](miles-knob-inventory.md) for option mappings and limits.
“Implemented,” “profile default,” and “qualified on a particular workload” are
separate claims. Parser acceptance is not runtime qualification. Existing
open-instruct GRPO, SFT and DPO entrypoints remain separate.

## Starting profiles

The [starter index](../configs/miles/README.md) describes preparation, overrides
and engine scaling. These are standalone `[core]`/`[miles]` TOMLs, without implicit
inheritance or environment-variable expansion.

| Profile | Default shape | Qualification and remaining scope |
| --- | --- | --- |
| [tiny-resident](../configs/miles/profiles/tiny-resident.toml) | One shared GPU; 1 prompt × 4 responses; two updates, native saves, eager decode | Tiny hybrid-MoE updates and separate-process continuation passed. Zero task rewards establish plumbing, not learning. |
| [train-disaggregated](../configs/miles/profiles/train-disaggregated.toml) | Two B300 Core EP ranks + one TP1 engine; 16 prompts × 4 responses; admission 64; 100 updates; initial/every-20 heldout eval; final native save; offline W&B | Full-model capacity and synchronous scheduling passed a 12-update trial. That trial did not exercise the complete 100-update starter's eval/save lifecycle. |
| [train-disaggregated-async](../configs/miles/profiles/train-disaggregated-async.toml) | Same hardware/batch; lag ≤1 optimizer step; buffer factor one; retry; rollout behavior logprobs | Full-model bounded async passed 24-update and admission-64 12-update exercises. Combined replay/restart/failure endurance remains separate. |
| [sft-b300-ep2-sync](../configs/miles/profiles/sft-b300-ep2-sync.toml) | Two updates, initial/final eval, diagnostics; same admission 64 | Short correctness recipe; no optimizer checkpoint requested. |
| [sft-b300-ep2-async-candidate](../configs/miles/profiles/sft-b300-ep2-async-candidate.toml) | Four updates; 512-token responses; eager decode; admission 64 | Short lifecycle variant. Its name does not mean all Core async support remains unqualified. |

Full-SFT profiles refer to the existing 18.5B-total KDA/latent model, not hero.
They reserve 2048 prompt + 4096 response tokens, except the short async variant.
The ordinary Core and olmo-miles colocated configurations **both keep the trainer
resident**: the baseline explicitly emits `--no-offload-train`. Core's difference
is its unqualified full-model colocation footprint and unsupported optional
trainer/optimizer offload, not a different default swapping policy.

Profiles select dynamic forward-only SwiGLU rows, microbatch one, activation
recomputation, router auxiliary coefficient 0.01, z-loss coefficient 1e-5,
streaming export, and 1 GiB publication buckets. Shared Core's row-specialization
default remains static. The wrapper enables the qualified arithmetic checkpoint
planner, compact storage and balanced replicated ownership; compiler-cache
persistence remains opt-in.

The synchronous starter uses actor-recomputed policy logprobs. The Core async
starter uses rollout behavior logprobs directly and leaves TIS off. **The
olmo-miles async recipe instead uses trainer-recomputed scoring plus TIS**
(with optional clipping/alternative correction). Similar placement, lag and
batch sizes therefore do not establish identical objectives. The measured Core
sync/async scheduling pair used rollout logprobs in both arms to control that
comparison. Also, Core's 16 prompts × 4 responses differs from an 8 × 8 recipe
despite both containing 64 samples. Neither replay nor a larger group is an
inference-only performance switch.

Copy a profile, replace `/data` paths, and prepare an HF descriptor with the exact
tokenizer/chat template, rendered `train.jsonl`, disjoint eval data where needed,
and trusted `verifiers.json`. Apply the chat template once and exclude reference
assistant answers from prompts. [SFT preparation](../scripts/miles/sft_gsm8k.py)
records the checkpoint, template and immutable source identities.

```bash
python -m open_instruct.miles plan /path/to/run.toml
python -m open_instruct.miles validate /path/to/run.toml
python -m open_instruct.miles train /path/to/run.toml
```

`plan` compiles configuration without checking installed runtime or data;
`validate` checks the installed MILES parser and backend configuration; neither
proves model memory fit. Beaker submission uses the committed
`./scripts/train/build_image_and_launch.sh --miles` workflow. Allocation,
mounts and automatic resubmission are launch concerns, not implicit TOML actions.
GPU exercises use urgent Holmes in `ai2/open-instruct-dev` with positive minimum
runtime; CPU-only WEKA work belongs on Saturn. Run the real attention
forward/backward preflight for the target hardware.

## Additional datasource trials

The [datasource harness](../scripts/miles/datasource_trials.py) consumes pinned
[math](../configs/miles/tasks/math.toml) and
[legacy IF](../configs/miles/tasks/ifeval.toml) task specifications. These task
TOMLs are preparation inputs, not standalone training configs. It selects 24
unique in-budget prompts from at most 256 source rows: eight for two updates and
sixteen held out for before/after evaluation. Hashes and actual row indices are
retained; this is a bounded slice, not a decontaminated benchmark.

Both [local tiny-model trials](measurements/miles-core-datasources-local-20260910.json)
and [full-SFT EP2 trials](measurements/miles-core-datasources-sft-20260910.json)
passed independent audits. The full-SFT allocation
[01M26GC6F3TRRQEXR9HJQR0XGG](https://beaker.org/ex/01M26GC6F3TRRQEXR9HJQR0XGG)
ran two updates and audited 64 training/eval responses per task, starting each
from the original SFT checkpoint. Math reached the response cap on 63/64 answers;
this qualifies integration, not useful math learning.

A separate [GSM8K/math/IF mixture](measurements/miles-mixture-20260910.json)
passed two EP2 updates with 48 training and 12 eval responses. Each source had
at least one mixed-reward training group. Every math training response reached
8192 tokens, and six heldout questions cannot support a learning conclusion.

After committing, the existing bounded launcher remains available:

```bash
MILES_BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks \
  ./scripts/train/build_image_and_launch.sh --miles \
  scripts/train/debug/miles_core_datasources.sh
```

Append `--task math` or `--task ifeval` to select one task. Broader olmo-miles
catalog/recipe/manifest adoption, generated arithmetic, code-service lifecycle
and judge-service qualification remain work. The baseline's 41 static-source
and generated-arithmetic gates mostly used a tiny development model; judge
transport was tested against a mock service. Neither those gates nor existing
open-instruct verifier classes imply a complete Core task bridge.

## What carries over, and what still needs work

| Capability | Implemented and qualified here | Remaining difference or acceptance |
| --- | --- | --- |
| Training contract | Native Core objective/accumulation, optimizer, masked response loss, auxiliary losses and clocks; fixed-batch, EP1/EP2, recomputation and full-SFT runs. [Contract checks](miles-core.md#training-contract-checks) | Matching coefficients do not equate auxiliary normalization/padding semantics. Replay fixes expert selection, not the entire numerical objective. |
| Router replay | R3 captures serving routes and reuses them for scoring, training and recomputation. Full-model EP2 synchronous eight-update/512-sample audit passed with zero expert-ID mismatches and finite nonzero router gradients. [Evidence](measurements/miles-core-replay-full-sft-20260911.md) | R2 (`use_routing_replay`) unsupported; TP/PP/CP remain one. Final unscored token uses a synthetic route; its auxiliary contribution is not serving-equivalent. Combined async/replay/restart endurance and selected-gate-score diagnostics remain. |
| Weight publication | Native Core → HF-named tensors → flattened NCCL buckets; colocated IPC. Exact weight checks; roughly 37 GB full-model publications measured in seconds. [Implementation](../open_instruct/miles/publication.py) | No per-step disk HF conversion. Alternative RDT/p2p/disk transports and pipeline depth two unsupported; multi-engine/multinode scaling needs measurement. |
| Placement and offload | Tiny resident colocation and full-SFT EP2 disaggregation passed | Full-model colocation/rollout-offload lifecycle unqualified; Core trainer/optimizer offload unsupported. More SGLang engines are configurable, not yet a measured scaling result. |
| Bounded async | Managed producer, homogeneous prompt groups, lag reservation, pending-prompt cursor, publication pause and clean teardown. [24-update and 12-update pairs](measurements/miles-control-exercise-20260911.md) passed | Baseline additionally has 45-update basic endurance and combined replay/recovery/restart evidence. Core needs those combinations, not another claim of missing basic async. |
| Multiple updates per collection | Complete optimizer-batch splitting and sufficient lag enforced | Full-model combined multistep/async/replay/restart not yet qualified; baseline has an EP2 initial/restart gate. |
| Serving admission and graphs | Actual 64 concurrent requests, decode graphs through 64, 524288 KV slots and 128 recurrent slots passed the larger B300 trial without observed OOM/retractions. [Evidence](measurements/miles-admission64-20260911.json) | Prefix caching remains disabled in profiles; cache/replay/publication invalidation and full-model colocated graphs need their own acceptance. |
| Compiler caches | Optional fingerprinted per-worker Triton restore into node-local storage and publication after successful teardown. Real tiny Core/SGLang two-lifetime trial passed. [Evidence](measurements/miles-startup-tiny-20260911.json) | Off by default; full-SFT/multinode and TP>1 serving qualification pending. Broader compiler families are not covered by the worker integration. [Cache guide](miles-compiler-cache.md) |
| Native checkpoint/resume | Schema-2 architecture/topology checks, optimizer/scheduler/RNG/cursor, completion marker. Full-model fast save/read gate passed exact restored state and two subsequent fixed-input updates/HF exports. [Evidence](measurements/miles-checkpoint-fast-audit-20260911.json) | Synchronous saves, same topology. Background writes, retention and token-per-expert cadence absent. Process writer unqualified at full model. Exact continuation used separate persistent per-rank caches; it does not promise identical future sampled rollouts. |
| HF export | Canonical conversion and `actor.export_hf`; evaluation snapshots via `eval_hf_dir` | No baseline-style final export workflow. `save_hf` is rejected rather than implying native saves create HF output. |
| Recovery | Pending prompt tracking and some inherited engine management | Baseline's bounded retry/stage-timeout/communicator-replacement/republish driver is not ported. Upstream health flags alone do not provide that contract. Trainer-cell recovery unsupported. |
| Batching and parallelism | Unpadded microbatch one with accumulation; MoE DDP/EP and separate dense FSDP backend | Dynamic batching, packing and trainer TP/PP/CP >1 unsupported. EP8/multinode unqualified. Our customized Olmo Megatron replay also restricts TP/PP/CP; general Megatron features are not baseline qualification. |
| Rewards and tools | Trusted weighted verifier registry, isolated bounded math workers, full-SFT math/IF/mixture gates; tool-token masks tested | Code/judge service lifecycle and broader sources remain. No complete multi-turn environment bridge; baseline's adopted catalog also does not establish such a bridge. |
| Evaluation | Fixed heldout GSM8K runs with retained questions/generations; shared-engine and snapshot dispatcher | Admission 64 now applies to shared eval, but the warm 128-question <60 s target has not been measured in the admission-only run. External/dedicated evaluation workflows need qualification. |
| Reporting and workflow | MILES W&B/dashboard interfaces, rollout dumps, contract/publication/startup/eval JSONL; grouped offline W&B exercise passed | No general baseline `run/status`, task/manifest preparation, sample-summary or automatic resume workflow; service/recovery dashboards are incomplete. |
| Model/hardware breadth | Prior SFT KDA/latent on B300 EP2; isolated standard dense Olmo3 path with local conversion/update/resume tests; exact hero native/HF conversion | Full hero optimized probability gate remains unresolved despite controlled-operator diagnostic parity. Full hero RL, original Olmo3 recipe and Core H100 MoE qualification remain. Baseline H100 evidence uses a smaller model and a different trainer backend. [Hero support](miles-hero-support.md) |

Unsupported native options are rejected by the
[configuration contract](../open_instruct/miles/config.py), including retained old
actors, LoRA, optimizer-free save/reset alternatives and direct `save_hf`.
Do not infer support from their presence in the upstream option inventory.
The latest Core/MILES consolidation pins do not expand any GPU qualification's
recorded scope.

## Which performance defaults should transfer

- **Serving capacity:** the 64-completion/admission pair measured warm throughput
  of 1856 response tokens/s synchronous and 2281 asynchronous, about 23% higher
  with async. Eight warm observations and completion-order differences limit
  extrapolation. The historical smaller-batch comparison changes both admission
  and optimizer batch, so it is not an isolated concurrency speedup.
- **Dynamic rows:** identical full-model scoring across 154,531 logprobs;
  changing-batch scoring improved 3.72× in the bounded static/dynamic screen.
  Later scheduling runs showed nearly flat score time against tokens. Other
  kernels can still compile. [Measurement](measurements/miles-control-exercise-20260911.md)
- **Checkpoints:** qualified direct save fell from 437.128 to 118.466 seconds;
  fresh-process load took 144.307 seconds. The approximately 222 GB checkpoint
  still makes cadence a real cost. These are checkpoint timers, not an entire
  Ray startup/restart measurement. [Record](measurements/miles-checkpoint-perf-20260911.md)
- **Caches:** the tiny public-entrypoint cold/restored trial reduced driver-entry
  to first update from 173.57 to 67.34 seconds. This establishes worker lifecycle
  and reuse; full-model improvement remains to measure before default promotion.
- **Diagnostics:** snapshot/reset/republish and replay checks add overhead.
  Compare ordinary runs separately from correctness exercises, and report async
  generation wait as consumer stalls rather than total inference work.

## Promotion order

1. Close researcher workflow gaps: prepared catalog/manifest adoption, launch and
   durable resume, and an explicit final HF-export lifecycle. Keep unsupported
   options rejected until wired.
2. Port bounded serving recovery and exercise async + replay + checkpoint/restart
   with fault injection and prompt/version conservation. Then extend multistep
   and endurance qualification.
3. Finish full-model compiler-cache qualification and measure high-concurrency
   evaluation during an ordinary run. Promote defaults only with retained evidence.
4. Broaden datasource/service gates in bounded groups, starting with real code
   execution and mock judge transport before substantive judge evaluation.
5. Treat batching/packing, higher topology, full-model colocation, H100 and hero
   training as separate changes with numerical and lifecycle gates.

The [qualification plan](miles-qualification-plan.md) gives acceptance principles;
completed measurements above supersede its older pending milestones. Relevant
baseline evidence lives in `~/proj/olmo-miles/docs/rl-dataset-status.md`,
`docs/measurements/async-multistep-20260907.md`,
`docs/measurements/v02-discrete-experiments-20260905.md`,
`docs/hardware-profiles.md`, and `docs/checkpointing.md`.
