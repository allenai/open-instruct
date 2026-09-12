# Frozen100 GSM8K configuration and timing audit

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

Both100-update allocations exited successfully and passed the paired independent
rollout/reward audit (`01M279KT4PQNC2HGZ7JVB1TDBJ`). This describes their actual recipes,
not a claim that the trainers implement identical mathematics. Final learning curves
and completed-run records are in the [campaign ledger](gsm8k-parity-20260910.json).
The new500 campaign is a separate experiment.

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

## Runtime library inventory

Core100 inherits compiled base image `01M24E7MSDGN2QFW1T8Z31BCKS`, Docker ID
`sha256:fe34fb1fef4910eb6610d8458a97c778360f67d0b38fbc0952cec28a321be629`.
A direct local inspection of that exact image and the r3 candidate inventory found:

| Library | Frozen Core100 | Frozen Megatron r3 |
|---|---|---|
| PyTorch / CUDA wheel |2.13.0+cu130 |2.13.0+cu130 |
| Triton |3.7.1 |3.7.1 |
| FlashAttention4 |4.0.0b27 |4.0.0b27 |
| Flash Linear Attention |0.5.2, source `9c8e42e762fce087c27b673af4922795d9edb85e` | Same |
| SGLang |0.5.19.dev49+g3145136, source `3145136dcd1238754e0ea2b2ffd546532119c71c` | Same |
| olmo-sglang adapter |`81a312ee8326e279a4641e03542ef971a5ffb863` | Same |
| MILES |Base `dbbab1566ae438f7202fff653eae938e07b1d4b6` plus OI patch, development `64ba71950c973e458f22a8d10541475940c911b0` | Base `dbbab1566ae438f7202fff653eae938e07b1d4b6` plus olmo-miles runtime adapters |
| Transformer Engine |2.17.0, source `4329ff84bfbdaa778a33cba02a15fb0807c64689` | Same installed package; Megatron actively uses TE |
| Megatron-LM base |Installed `235952df607b3820716e5e67728a5ab470ca33ae`, not Core trainer | Same base, adapter patch stack applied |
| Megatron-Bridge |Installed `db723bae699dae5d29003ec4789c67730a343c32`, not Core trainer | Same source, used by Megatron |
| olmo-megatron |Installed `b84044ffb0d52620ec1599eda19d8f3d1de816b4`, not Core trainer | Repaired BF16-storage/FP32-router-compute `ba5615df741a24ba4aee678d0e853306b3502282` |
| Ray / Transformers |2.58.0 /5.12.1 | Same |

The Core source overlay replaces Core and MILES Python sources; it does not rebuild
these compiled libraries. Package/version equality still does not establish kernel-path
equality: trainer factories, padding and attention wrappers differ. FA4's installed
package version is available here; a distinct upstream commit was not asserted without
a recorded build source. These are frozen100 inventories, not the newer hero/500 runtime.

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
| Auxiliary padding | Unpadded individual samples | bshd pads each sample to its rank rollout-batch maximum length even with pad multiple1; auxiliary includes these artificial positions | Policy mask does not automatically remove padded tokens from router balancing/z loss |
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
| Generation | Native `perf/rollout_time`, matched warm indices; includes collection-side work measured by MILES | 95 matched warm events: Core mean36.78847s; Megatron39.52733s |
| Operational cycle | Consecutive native generation-end timestamps, excluding eval crossings | 90 matched warm cycles: Core mean91.06497s; Megatron68.24182s. Generation is included in these cycles, so do not add the generation row |
| Core ingress/scoring | Same-rollout generation-end timestamp to rank0 score-contract timestamp |95 warm events mean44.32847s, median44.872s; includes dispatch, data conversion, preflight, forward/logprob work and score checks; not a pure model-forward timer |
| Training | Core optimizer contract starts after scoring; Megatron `actor_train` also excludes its separately timed log-prob pass but has different instrumentation/reductions | 95 warm events each: Core6.16966s; Megatron17.38376s. Diagnostic scopes differ; not an isolated backend compute-speed ratio |
| Publication | Matched completed-update indices5–99; explicit publication timers | Core3.72784s; Megatron5.88441s. Core export0.4356s and transport/load3.2223s; Megatron gather1.1027s, conversion0.2518s and engine wait3.4089s. Stage scopes differ; see retained weight-sync evidence |
| Orchestration | Requires explicit boundaries or a profiler | Do not obtain this by subtracting independently averaged generation/training phases |
| Evaluation | Initial/final held-out generation and scoring logs, separated from warm training cycles | Same128 prompts, but response lengths/caps and therefore duration change with policy |
| Startup | Beaker scheduled→started→first eval / first update | Megatron r3 cold DCP load consumed approximately22min; startup/checkpoint I/O excluded from warm throughput |
| Save |100 runs disabled native saves | No save throughput measurement exists for this pair |
| Allocated time | Beaker scheduled→exited; report started→exited separately | Core11922.131943s allocated /11845.799765s started→exit; Megatron10844.651858s /10773.821174s. Each allocation used3 GPUs |

The Core score interval is genuinely large in observed wall time. An implicit entropy
calculation is not its explanation: the actual MILES logprob helper defaults to
`with_entropy=False`. Both Core scoring and training use the configured synchronous EP
path and chunk-KDA dispatch; source inspection alone does not identify the expensive
operation. The no-grad SwiGLU rounding finding is a correctness issue, not evidence that
it caused this timing gap. The completed frozen-image EP2 scorer profile below establishes
large cold compilation overhead and cheap identical-batch warm scoring, while attribution
of the historical changing-batch boundary remains incomplete.

A concrete compilation hypothesis remains: frozen `kernels/swiglu.py` marks
`rows=x.shape[0]` as a Triton constexpr. Different routed row capacities can therefore
produce separate no-grad kernel specializations; gradient-enabled eager SwiGLU does not
use this kernel. There is no autotuner in this wrapper and its valid-count scalar stays
on CUDA (loaded in-kernel), so this wrapper does not synchronize that count to the CPU.
The measured JIT misses and artifact writes below confirm cold specialization; the future
rounding fix retains this shape key.

## Startup and held-out evaluation boundaries

[Retained boundary evidence](gsm8k-startup-eval-boundaries-20260911.json) includes
source-log hashes, exact native timestamps and Beaker statuses. Core scheduled→process
start took76.332s and Megatron70.831s. Process start→initial rank0 publication completion
was652.715s for Core and1918.807s for Megatron; process start→initial evaluation summary
was940.485s and2221.632s. These are observed end-to-end startup boundaries, including
model loading, serving startup and setup. They do not isolate DCP reads or compiler time.

| Completed updates | Core publication→eval summary (s) | Megatron publication→eval summary (s) | Core progress-bar collection elapsed (s) | Megatron progress-bar collection elapsed (s) |
|---|---:|---:|---:|---:|
|0 |287.770 |302.825 |267 |280 |
|20 |269.335 |289.336 |269 |289 |
|40 |280.570 |298.810 |280 |298 |
|60 |260.271 |284.238 |260 |283 |
|80 |230.524 |218.355 |230 |218 |
|100 |254.025 |197.408 |253 |197 |

The first two columns use direct native rank0 publication-end and held-out-summary
markers. They include intervening driver/setup/checking, generation, reward processing
and dump/log work. The last two columns are the printed128/128 progress elapsed,
rounded down to whole seconds. These are different scopes, not components to subtract
into a precise orchestration estimate. The initial evaluation also includes setup/check
work absent from later evaluations. Changing response length and capped outcomes affects
evaluation time; these durations are not fixed-output-token speed measurements.

Megatron separately records95 warm `log_probs` events with mean4.18246s; the Core
44.32847s boundary above includes more than scoring. The completed
[frozen-image EP2 profile](core-score-profile-20260911.md) measured actual `_score`
on retained rollout5 tokens at initial weights:230.458s with cold caches, then1.249/1.235/
1.246s on identical repeats. Cold ranks generated362/327 cubins; warm repeats generated
none and preserved scores exactly. SwiGLU JIT calls consumed41.38/41.40s per rank,
with another93.21/85.85s in FLA JIT calls. These per-rank durations are not additive across
ranks. No optimizer update occurred. This identifies a concrete cold compilation cost,
not the fraction of historical changing-batch overhead caused by it.

## Longer500 campaign

The new campaign changes horizon to500 updates and repeats the same400 prompts in five
ordered passes with eval every20. An actual pinned MILES loader rehearsal executed500
calls of4 prompts, verified every ID modulo400, all4 completions per group, and no shuffle:
2000 groups/8000 completions, final epoch4/offset400. The loader method SHA256 was
`5c82470b89c44b532184f02178ad0a1aa52cc272ac8a0c521868871c0fd9956a`.

Megatron config `examples/qualification/gsm8k-abhishek-500-20260911.toml` at baseline
commits1b426fd/aa114a1 preserves the optimizer/data recipe, explicitly aligns PyTorch
sampling, token pool32768 and prefill chunk16384 with Core500, requests the eight-hour server maximum, and saves synchronously
every100 updates with rolling latest retention. The first12h request was rejected before experiment creation by Beaker's eight-hour cap.
The successful submission uses sourceaa114a1 with explicit `--min-runtime8h`; subsequent
readable-config commitf45f243 records8h directly. The public baseline launch schema has no
hard deadline;18h is monitored.
Core uses corrected runtime290d2ca and currently retains five native checkpoints. This
retention difference must remain explicit in disk usage and allocation-time comparisons.

Approximate native state size from18,514,193,152 parameters is241.4GiB per checkpoint
(2-byte model +4-byte master + two4-byte Adam moments), before metadata/extra tensors.
Five such checkpoints are about1.18TiB; rolling latest still needs approximately483GiB
while writing the replacement before deleting its predecessor. These are forecasts,
not observed filesystem sizes. Save timers must include durable model/optimizer write
and rollout cursor completion, and report pruning separately when available. Do not
call a model-only write a complete resumable checkpoint.

The500 serving adapters also differ: Core uses olmo-sglang11ae9f6 while Megatron
uses81a312. Their source difference adds gated per-head QK-gain/scalable-softmax
support plus validation tools. Both flags are false in this campaign's old SFT
checkpoint, as verified from its actual HF configuration. These are different serving
revisions even when the added architecture branches are inactive; no throughput or
initial-evaluation difference is attributed to that revision change without evidence.

[Matched weight-sync evidence](gsm8k-weight-sync-20260911.json) resolves the
publication stage timings above. [Actual Megatron padding evidence](gsm8k-megatron-padding-20260911.json)
checks all200 retained rank dumps from the100-update r3 run:3,281,711 real input tokens
occupied6,034,840 padded forward slots. Artificial positions total2,753,129, or45.6206%
of capacity (1.83893× real input length). All eight samples on each rank pad to that
rank's rollout-batch maximum, even with microbatch1 and effective pad multiple1. This
counts sequence slots, not measured FLOPs or time; Core and Megatron also generated
different responses, so it is not a direct total-work ratio between the two runs.

The later [successive-batch scorer qualification](core-score-variants-20260911.md)
passed exact cross-arm scores while removing forward SwiGLU capacity specialization.
On retained batches6–9 at initial weights, parent Core290 scorer time averaged63.283s
and isolated candidate15.215s; no optimizer update or active runtime pin change occurred.
This demonstrates recurring compilation cost in a controlled diagnostic, not a measured
end-to-end training improvement or an explanation of the learning-curve difference.

## Prioritized checks of the learning difference

The compilation result is a performance finding, not a demonstrated cause of the
learning difference. The next causal checks should isolate these concrete differences:

1. **Scoring versus gradient-forward arithmetic.** The completed Core100 used the
   pre-fix no-grad SwiGLU rounding path; new Core500 uses290d2ca. Measure old and corrected
   score anchors, selected router IDs and actual gradient-forward logprobs on identical
   retained tokens/weights. A difference here changes the PPO ratio even before an
   optimizer update. Existing tiny tests prove the arithmetic discrepancy and correction;
   they do not establish how much of the100-run learning difference it caused.
2. **Unequal-length, padded auxiliary objectives under EP2.** Core averages unpadded
   token contributions while Megatron's actual batches include rank-maximum padding.
   The existing four-arm auxiliary derivative comparison uses equal lengths, no padding
   and EP1; it intentionally cannot settle this real-workload difference. Reuse identical
   tokens/anchors in both actual EP2 trainers, independently reconstruct auxiliary scalars
   and compare policy/auxiliary/combined gradients by router/expert/dense category before
   clipping. Preserve each implementation's current normalization rather than silently
   making the gate pass by changing the algorithm.
3. **Initial serving divergence before learning.** Some initial held-out completions
   already differ. Compare the exact prepared token IDs, checkpoint tensor inventory,
   serving settings and conditional logits/routes under controlled request scheduling.
   Sampler/revision differences alone do not establish a cause, particularly for greedy
   decoding. Keep this inference diagnostic separate from optimizer and reward checks.

[Audited group-signal counts](gsm8k-zero-policy-advantages-20260911.json) show21
Core and29 Megatron updates with no mixed-reward prompt groups, hence zero current
policy advantages under the configured centered GRPO objective. Counts were independently
reconstructed from all3,200 audited sample outcomes. Auxiliary gradients and accumulated
Adam momentum can still move weights on those steps. In successive20-update windows,
counts are Core4/3/8/0/6 and Megatron3/4/6/6/10. This is useful objective context, not
proof that auxiliary updates explain the curves or that Core uniquely lacks policy signal.

Router replay is disabled in both completed100 and current500 runs:
`use_rollout_routing_replay=False` and `use_routing_replay=False`; Megatron additionally
prints `moe_enable_routing_replay=False`. Actual Core100 and Core500 rollout0 artifacts
each contain16 samples with no populated `rollout_routed_experts`. Core's internal
`_score(..., use_replay=True)` only permits its context wrapper; that wrapper remains a
no-op when the configuration flag is false. Thus a replay-code failure is not an active
mechanism in these runs. Separate replay qualification covers tiny live serving and
native EP1/EP2 gradient/recomputation checks; full-SFT replay remains unqualified, and
the final unscored token's synthetic IDs are a known auxiliary-loss limitation when
replay is enabled. No active-run settings were changed during this review.

Replay being off does **not** establish that serving and training naturally choose the
same experts. Router IDs were not captured in these comparison runs; a future fixed-prefix
routing comparison is needed to measure that potential disagreement. The current CPU
replay subset (`test_model_backend_dispatch.py` plus `test_contract.py -k replay`) passed
6 tests with11 deselected, covering the disabled no-op path and replay contexts, router
gradients and recomputation behavior. These tests qualify the mechanism at their stated
scope; they do not replace full-checkpoint serving/training route comparison.
