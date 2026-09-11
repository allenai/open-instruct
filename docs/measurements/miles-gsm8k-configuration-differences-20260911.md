# Frozen100 GSM8K configuration and timing audit

This compares the completed Core100 allocation with the still-running Megatron r3
allocation. It describes their actual recipes, not a claim that the trainers implement
identical mathematics. Final learning curves and completed-run timing belong in the
[campaign ledger](miles-gsm8k-parity-20260910.json); partial Megatron timings below are
explicitly identified. The new500 campaign is a separate experiment.

## Provenance

| Item | Open-instruct / Core100 | olmo-miles / Megatron r3 |
|---|---|---|
| Beaker | `01M26P6XX6SN886DCVZ68WMQK2` | `01M26YNP5E64YGNXR85TA2RP4Q` |
| Application source | `c3a41598e434d50058f929e6c55b82e34aa4ff09` | `0e648108c70c5d5256a9b93a88b4f8d610e44ea0` |
| Trainer source | Core `b7f1e5296704779e7deb06de4c4242be9729d1a9` | olmo-megatron `ba5615df741a24ba4aee678d0e853306b3502282` |
| Image | `01M26N80T0V9PREQTS87J849P8` | `01M26TT1EQTB84RV0MM650W0YE` |
| Initial trainer load | Shared HF safetensors imported into native Core topology | Native Megatron DCP derived from the same SFT checkpoint |
| Serving load | Shared HF checkpoint, then native trainer publication | Same shared HF checkpoint, then native trainer publication |

The checkpoint is the **older Dolci-think SFT65536 router-BF16 checkpoint**, approximately
18.5B total parameters. It is not the later12.5B hero step38000 checkpoint. Both use the
frozen preparation manifest SHA256 `1d8a22500fffdd9a7461c445998815d5e85cf9dd79859d93313798472c0e76e0`.
Initial serving weight checks passed in both allocations. Equal HF initialization does
not make different trainer and inference kernels bitwise equivalent.

Evidence comes from the frozen Core `scripts/miles/gsm8k_parity.py`, Core configuration,
actor and normalization code at the application SHA above; the r3 TOML and resolved
Megatron arguments in `/tmp/miles-gsm8k-parity-megatron-r3.log`; the actual image's
SGLang `ServerArgs` and Megatron checkpoint code; and full retained Core logs. Avoid
reading current working-tree defaults as though they were used by these allocations.

## Shared recipe

| Dimension | Both100-update allocations |
|---|---|
| Data | Same400 ordered training prompts, same128 held-out prompts; prepared source IDs and tokenizer/prompt hashes retained |
| Sampling |4 prompts ×4 completions per update; global batch16; seed17; train temperature1, top-p1, top-k disabled; greedy evaluation |
| Horizon |100 optimizer updates; evaluation before training and after20/40/60/80/100 updates |
| Length | Response cap4096, context6144, prompt cap2048 |
| Placement |3 B300 GPUs on Holmes:2 resident trainer GPUs, EP2;1 resident SGLang GPU, TP1/EP1; synchronous disaggregated loop |
| Policy loss | GRPO group reward centering; standard-deviation normalization disabled; no further advantage normalization; response-mean policy reduction |
| PPO anchor | Trainer recomputes old log probabilities before update (`use_rollout_logprobs=false`); serving log probabilities retained for drift checks |
| PPO clipping | Lower0.2, upper0.28; response loss masks applied by shared MILES loss machinery |
| Optimizer | AdamW, constant LR1e-6, no warmup, betas0.9/0.95, epsilon1e-8, weight decay0, global gradient norm clip1 |
| Additional losses | Router load-balancing coefficient0.01, router z coefficient1e-5; KL0, entropy0; no critic |
| Routing | Replay off; router remains trainable through policy and auxiliary gradients |
| Trainer layout | Microbatch1, bshd, TP/PP/CP1, activation recomputation enabled; BF16 model storage and FP32 master weights/moments/gradient reduction |
| Serving | Triton attention, static memory fraction0.6, max4 running requests/server concurrency4, mamba cache8, decode graphs up to4, prefill graphs disabled, radix cache disabled |
| Publication | Every optimizer update; streamed canonical HF names/tensors;1GiB transfer buffers; initial full serving-weight check |
| Storage | Rollout/debug outputs retained; no durable training checkpoints or final HF export in either100 run |
| Compiler cache | Cold allocations; no persistent cross-allocation cache lifecycle |

The native MoE EP implementations differ. Matching EP degree and FP32 reductions is
not a claim that communication order, expert sharding, fused kernels or reductions
are identical. Core's production normalization accounts for rank averaging and expert
parameters separately; the native EP1/EP2 gates test this contract independently.

## Differences that can affect learning or speed

| Area | Core100 | Megatron r3 | Consequence |
|---|---|---|---|
| Auxiliary normalization | Per-sequence router losses weighted by real input-token counts; denominator is global real model tokens / trainer world size before rank averaging | `seq_aux_loss`; response-average recipe equally weights microbatch/sequence means | Unequal sequence lengths give different auxiliary objectives despite equal coefficients |
| Auxiliary padding | Unpadded individual samples | bshd padded forward; current auxiliary path includes artificial padded positions | Policy mask does not automatically remove padded tokens from router balancing/z loss |
| Gradient-enabled versus scoring activation | Old Core has a no-grad fused SwiGLU path with different BF16 rounding from gradient-enabled eager path | Separate Megatron kernels/precision contract | Old Core PPO scoring anchor need not exactly equal its gradient-enabled forward at unchanged weights; future Core290d2ca corrects the identified rounding path |
| Sampler implementation | Explicit SGLang `sampling_backend=pytorch` | Actual SGLang resolves unspecified backend to `flashinfer` | Same seed/distribution settings do not imply identical sampled completions |
| Serving token pool | Explicit `max_total_tokens=32768` | Unspecified / automatic (`None` in actual ServerArgs) | Admission/memory allocation may differ; do not attribute all generation timing to trainer choice |
| Prefill chunking | Not explicitly fixed in recipe | Actual ServerArgs resolves `chunked_prefill_size=16384` | Verify resolved Core startup value before calling this a matched kernel setting |
| Diagnostics | Contract validation and probability histograms remain on; expensive parameter probes disabled (`diagnostic_interval=0`) | Trainer diagnostics enabled; different checks and timer instrumentation | Observation overhead differs |
| Initial checkpoint I/O | HF import and native optimizer construction | Distributed checkpoint metadata/read/resharding and optimizer construction | Cold startup is not a trainer-step throughput comparison |
| Version convention | Initial0, post-update20 is20 | Initial1, post-update20 is21 | Audit aligns completed optimizer steps, not raw version numbers |
| Reward bridge | Open-instruct registered GSM8K verifier | Prepared-task baseline verifier | Independent retained-response audit is required; config equality alone cannot prove reward equality |

Neither loss gives policy gradients to masked response positions. Router auxiliary losses
operate over model-input positions, including prompts; they are not response-loss-mask
objectives. In Core the denominator therefore counts prompt plus response tokens. This
is intentional current behavior, and different from treating router loss as another
response-mean policy term.

A local fixed-token policy-only diagnostic found exact initial weights/masters and
independently correct clipping/Adam updates in each backend, with about1.1% relative
preclip gradient difference. A separate equal-length/no-padding four-arm diagnostic
(policy, LB, z, combined) verified actual auxiliary scalars and nonzero router gradients.
Those tests remove the unequal-length/padding difference by construction; they do not
validate equality of the online auxiliary objectives or explain the held-out curve.
The auxiliary gate used corrected future Core290d2ca and EP1, not the frozen Core100 image.
Raw canonical gradients and finite-precision superposition residuals are retained in
baseline `docs/measurements/core-auxiliary-contract-20260910/`.

## Performance measurements and boundaries

Use identical-index warm **generation-end to next generation-end cycles** as the main
operational comparison. Both boundaries are native MILES RolloutManager `perf N:` log
records after collection. A cycle includes the preceding batch's training/publication
and the next batch's collection, rather than one precisely isolated optimizer step.
Exclude the first5 warmup indices, gaps/duplicate boundaries, and evaluation-crossing
intervals19→20,39→40,59→60,79→80. Final update99's trailing work has no next collection
boundary. The committed analyzer intersects available indices between arms; it never
fills absent events or compares unequal coverage silently.

| Phase | Defensible evidence / scope | Current observation |
|---|---|---|
| Generation | Native `perf/rollout_time`, matched warm indices; includes collection-side work measured by MILES | Core95 warm events mean36.7885s; final Megatron matched coverage pending |
| Operational cycle | Consecutive native generation-end timestamps, excluding eval crossings | Core90 valid warm cycles mean91.065s, median90.3405s; early Megatron cycles4→5 through18→19 mean73.1585s, median69.782s (15 intervals, partial coverage only) |
| Core ingress/scoring | Same-rollout generation-end timestamp to rank0 score-contract timestamp |95 warm events mean44.32847s, median44.872s; includes dispatch, data conversion, preflight, forward/logprob work and score checks; not a pure model-forward timer |
| Training | Core optimizer contract starts after scoring; Megatron `actor_train` also excludes its separately timed log-prob pass but has different instrumentation/reductions | Core recorded95-event mean6.1697s; not plotted as an equivalent trainer-speed comparison with Megatron |
| Publication | Core explicit publication timer / optimizer-log to publication boundary; baseline `update_weights` timer | Core95-event recorded mean3.74132s; boundary mean3.76832s. Baseline final aggregate pending; export/communication/check scopes differ |
| Orchestration | Requires explicit boundaries or a profiler | Do not obtain this by subtracting independently averaged generation/training phases |
| Evaluation | Initial/final held-out generation and scoring logs, separated from warm training cycles | Same128 prompts, but response lengths/caps and therefore duration change with policy |
| Startup | Beaker scheduled→started→first eval / first update | Megatron r3 cold DCP load consumed approximately22min; startup/checkpoint I/O excluded from warm throughput |
| Save |100 runs disabled native saves | No save throughput measurement exists for this pair |
| Allocated time | Beaker scheduled→exited; report started→exited separately | Final Megatron exit pending; multiply each allocated duration by3 GPUs for GPU-hours |

The Core score interval is genuinely large in observed wall time. An implicit entropy
calculation is not its explanation: the actual MILES logprob helper defaults to
`with_entropy=False`. Both Core scoring and training use the configured synchronous EP
path and chunk-KDA dispatch; source inspection alone does not identify the expensive
operation. The no-grad SwiGLU rounding finding is a correctness issue, not evidence that
it caused this timing gap. A bounded EP2 same-batch CUDA/CPU profiler is the next useful
measurement, keeping ingress, per-forward communication, logprob extraction and postscore
checks separate. No profiling-based attribution has yet been established.

## Longer500 campaign

The new campaign changes horizon to500 updates and repeats the same400 prompts in five
ordered passes with eval every20. An actual pinned MILES loader rehearsal executed500
calls of4 prompts, verified every ID modulo400, all4 completions per group, and no shuffle:
2000 groups/8000 completions, final epoch4/offset400. The loader method SHA256 was
`5c82470b89c44b532184f02178ad0a1aa52cc272ac8a0c521868871c0fd9956a`.

Megatron config `examples/qualification/gsm8k-abhishek-500-20260911.toml` at baseline
commits1b426fd/aa114a1 preserves the optimizer/data recipe, explicitly aligns PyTorch
sampling, token pool32768 and prefill chunk16384 with Core500, uses min runtime12h, and saves synchronously
every100 updates with rolling latest retention. The public baseline launch schema has no
hard deadline;18h requires external monitoring unless a supported task override is used.
Core uses corrected runtime290d2ca and currently retains five native checkpoints. This
retention difference must remain explicit in disk usage and allocation-time comparisons.

Approximate native state size from18,514,193,152 parameters is241.4GiB per checkpoint
(2-byte model +4-byte master + two4-byte Adam moments), before metadata/extra tensors.
Five such checkpoints are about1.18TiB; rolling latest still needs approximately483GiB
while writing the replacement before deleting its predecessor. These are forecasts,
not observed filesystem sizes. Save timers must include durable model/optimizer write
and rollout cursor completion, and report pruning separately when available. Do not
call a model-only write a complete resumable checkpoint.
