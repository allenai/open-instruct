# Olmicro: native SGLang migration and the 450m GSM8K run

Updated November 2, 2026. Status: **native integration implemented; image build
submitted; native end-to-end RL not yet run**. Training submission is blocked by
read-only access to `ai2/scaling-ladders` for `michaeln`. See the
[launch preparation](../../experiment.md#native-sglang-450m-gsm8k-launch-preparation-2026-11-02)
for build identifiers and the one-time launch controller.

We already have a completed 450m Core/vLLM baseline. The next work is to connect
native SGLang serving to the MILES + OLMo-core learner, validate its complete RL
lifecycle, and run 500 GSM8K optimizer updates. We do not need another baseline
training run for this first comparison.

## What is changing

OLMo-core remains the learner. The migration changes rollout serving to native
SGLang and uses MILES to coordinate generation, learner updates and publication.
It does not change the pretrained checkpoint or introduce a different model.

| Component | Completed Core/vLLM baseline | Previous MILES experiment | Planned native MILES experiment |
| --- | --- | --- | --- |
| Learner | OLMo-core | OLMo-core | OLMo-core |
| Coordination | Open-Instruct Core/vLLM path | MILES | MILES |
| Inference | vLLM Hybrid implementation | SGLang server with HF reference-backbone adapter | Native `sglang.srt.models.mainline_ladder` |
| Model execution | vLLM kernels and caches | HF forward per request; adapter-owned recurrent state | Batched GDN kernels, managed recurrent/convolution state, paged full attention |
| Precision | BF16; FP32 SSM cache | BF16 | BF16; FP32 recurrent state and GDN gates |
| Completed 500-update result | Yes | No, personal four-GPU run exhausted retries | Not launched |

The earlier MILES run was not a native SGLang training result. Its serving
adapter limited each engine to four concurrent requests. The new implementation
uses SGLang's own model execution and state allocation; a standalone probe tested
16 slots and 19 queued requests. That establishes serving functionality, not
cross-process RL publication or training correctness.

## Source and reproducibility

The local workspace is `/home/mnoukhov/olmicro/`:

- `open-instruct/`: MILES run configuration and submission.
- `miles-olmo-core/`: learner integration and the published native serving patch.
- `sglang/`: patched native SGLang source.
- The original Core/vLLM checkout is `/home/mnoukhov/open-instruct-hybrid/`.

The shared implementation is in the pushed
[miles-olmo-core branch `codex/hybrid-mainline`](https://github.com/allenai/miles-olmo-core/tree/10a8f07a3284d5c30597dbf8d731049757ca54fd/patches/sglang-olmicro-native).
This contains the patch, reproduction instructions, tests and numerical reports;
it does not depend on coworkers finding a local branch. The patch applies to
SGLang `3145136dcd1238754e0ea2b2ffd546532119c71c`. The local resulting source
commit was `986877f4cd38b090ce2338e685f914cd65c18e8c`.

The implementation preserves release parameter names and handles Olmicro's
normalization, NoPE attention, embedding scaling, gates and beta range `[0,2]`.
Loading weights also refreshes packed QKV, gate/up and convolution tensors.
The supported scope is TP1/PP1, eager execution, no prefix cache, no quantization
and no speculative decoding. CUDA graphs and chunked prefill stay disabled for
the first RL experiment.

## Evidence so far

All five final-midtrain sizes passed native BF16 functional probes on one L40S:
275m, 450m, 810m, 1.4b and 2.7b. Tests covered prompt lengths 1–3071, 32-token
greedy generation, queueing, reordered slot reuse and same-checkpoint reload.
Thirty-five focused regressions passed.

For 450m, maximum bidirectional full-vocabulary KL versus vLLM was **0.000788**,
passing the fixed 0.001 gate. The selected-token logprob gate still failed:
maximum absolute error **0.156**, against a threshold of 0.08. Greedy outputs are
not uniformly identical across backends. Keep these failures visible: serving
functionality is qualified, strict numerical equivalence is not. The 810m model
also failed the full-vocabulary KL gate; that is outside this first 450m run.

| Size | HF bridge, A6000: output tok/s | Native SGLang FA3, L40S: output tok/s |
| --- | ---: | ---: |
| 275m | 62.1–67.1 | 410.3 |
| 450m | 42.2–45.0 | 253.8 |
| 810m | 39.2–45.0 | 261.1 |
| 1.4b | 28.9–32.0 | 201.4 |
| 2.7b | 19.8–22.4 | 134.0 |

These are short synthetic serving batches including prefill/request overhead.
The native batch produced 224 output tokens across seven prompts. Hardware and
batching differ, so the roughly sixfold observed rate difference is not an
isolated backend speedup or a forecast for RL throughput. The HF bridge's
49m17s initial evaluation over 1,209 examples motivates the change; there is
no native full GSM8K evaluation timing yet.

Native raw reports are saved under
`/home/mnoukhov/olmicro/reports/native-sglang-20261030/final-export/native-fa3-final/`.
See the [qualification log](../../experiment.md#native-sglang-olmicro-implementation-and-vllm-compatibility-2026-10-30)
for checkpoint identities, failed gates and GPU provenance.

## The existing baseline we will compare against

[Core/vLLM experiment `01M64J9FRG4PYSB6E7WJ739A7F`](https://beaker.org/ex/01M64J9FRG4PYSB6E7WJ739A7F)
completed **500 optimizer updates** on four Rhea A6000s: two learners and two
independent TP1 inference engines. It saved native checkpoint 500 and the final
HF export. A preemption restored checkpoint 100 and the data cursor successfully.
Its final GSM8K Platinum accuracy was **17.287%**, versus **3.970%** initially.
W&B run ID: `h2r4vd1u`.

The later personal-project
[MILES/HF-bridge experiment `01M67081AXN33GBZJASMPYM1N8`](https://beaker.org/ex/01M67081AXN33GBZJASMPYM1N8)
is a separate partial result. Its attempts reached at most step 91. The last
productive attempt reached step 51 at October 31, 06:47:28 UTC. Preemptions
occurred before checkpoint 100, so attempts restarted from the base weights.
Subsequent starts failed the application retry limit; the experiment is terminal.
It supplies bridge operational evidence, not a completed native comparison.

## Planned runs: 450m only

These runs are proposed and unsubmitted. Record the actual config, image digest,
source revisions, dataset checksums, Beaker links and W&B links in
[`experiment.md`](../../experiment.md#native-sglang-450m-gsm8k-end-to-end-plan-2026-11-01)
when launched. Keep separate output roots for validation and the main run.

| Run | Purpose | Work | Gate |
| --- | --- | --- | --- |
| N1: native lifecycle smoke | Validate the complete MILES/Core/native-SGLang path | Four GPUs, two optimizer updates; checkpoint each update; small held-out evaluation slice | Finite metrics, verified native model, both learners, live publication to both engines, eval completion and committed checkpoints |
| N1R: smoke continuation | Prove recovery before committing to 500 updates | Resume N1's committed checkpoint in a second attempt; perform at least one additional optimizer update | Restore optimizer, scheduler, completed-step count and data cursor; publish restored weights; do not restart from base |
| N2: fresh GSM8K run | Measure learning and end-to-end performance | Four GPUs, 500 optimizer updates from the original 450m midtrain weights; full evaluation at 0/100/200/300/400/500 | Learning/eval report, checkpoints every 100, final export and completed result |

N1 and N1R now share one four-GPU allocation: the first training process stops
after two of three updates, then a separate process must restore checkpoint two
and complete update three. The small smoke evaluation uses eight examples and
publication diagnostics run every update. N1R continues only the smoke.
N2 starts with a fresh optimizer and data cursor,
and does not inherit smoke training. Reuse the completed Core/vLLM baseline;
no additional old-infrastructure training run is planned.

### Shared model, data and objective

| Setting | Planned value |
| --- | --- |
| Initial model | General release 450m final midtrain, `step0038147` |
| HF checkpoint | `/weka/oe-training-default/yashasbls/hybridish-local/hf/v1/450m/midtrain/step0038147` |
| Core configuration | Matching `olmo-core/v1/450m/midtrain/step0038147/config.json` |
| Training data | Existing `gsm8k-base/train.jsonl` under `/weka/oe-adapt-default/michaeln/open-instruct-hybrid/data/` |
| Evaluation data | Existing `gsm8k-base/eval.jsonl`, 1,209 GSM8K Platinum examples; disjoint from training |
| Training budget | 500 optimizer updates; 4 prompts × 4 completions = 16 completions/update, nominally 8,000 consumed completions |
| Lengths | Prompt cap 512, response cap 2,048, packed/context limit 4,096 |
| Objective | Existing MILES GRPO recipe; keep zero-variance groups; token loss reduction; GRPO std normalization disabled |
| Optimizer | Adam betas 0.9/0.999, constant LR `1e-6`, zero warmup/weight decay, gradient clipping 1 |
| PPO clipping | Lower epsilon 0.2; upper epsilon 0.272 |
| Regularization | KL and entropy coefficients zero |
| Training sampling | Temperature 1, top-p 1; seed 1 |
| Off-policy bound | Maximum policy lag 1, TIS clip 2; barrier publication; reject over-age groups |
| Evaluation | One sampled completion per example, temperature 1, top-p 0.95, response cap 2,048 |
| Saves | Main native checkpoints every 100 updates, retain last 3; final HF export |
| Tracking | `mnoukhov/open_instruct_internal`, distinct native run/group identity |

The reported evaluation is single-sample accuracy (pass@1), not pass@k for
k greater than one and not greedy accuracy. Log attempted/completed examples,
truncation, response lengths and answer-verification results. Verify the data
checksums and counts before launch. Extra generated/discarded samples from async
prefetch must be reported separately from the 8,000 consumed completions.

### Hardware and scheduling proposal

Use **one Rhea node with four A6000 GPUs: two OLMo-core learners and two
independent TP1 native SGLang engines**, matching the completed Core/vLLM
baseline. The workspace is **`ai2/scaling-ladders`**, selected by the user, under
budget **`ai2/oe-other`**. Keep the same requested host memory, mounts and GPU
split. This removes the GPU-model confound from the planned training comparison.

Native serving was previously tested on L40S. N1 must first qualify native kernel
support on A6000 (SM86), including full attention: the L40S FA3 configuration is
not yet qualified there. If FA3 is unsupported, select and qualify a native
backend supported on A6000, retaining the numerical reports and gates. Do not
silently fall back to the HF bridge or move the main run to different hardware.
N2 remains gated on the A6000 smoke.

The user requested a **two-hour minimum runtime**. Proposed main scheduling:
normal/preemptible priority, minimum runtime 2h, 24h attempt timeout, auto-resume,
128 GiB requested host memory and 32 GiB shared memory. These are distinct
durations: a minimum runtime is not a two-hour job timeout. The smoke needs a
wall-time cap at least as long as its minimum runtime.

The earlier allocation check found a Rhea allocation for `ai2/scaling-ladders`;
verify current capacity and minimum-runtime eligibility before submission. The
workspace choice is now resolved. This document records the run plan; no new
job has been submitted.

### Required integration work before N1

1. Build the pinned native overlay into a Beaker-built image and record its
   immutable identity. The previous training image alone does not contain the
   qualified native overlay. Build on the cluster, using the repo launch wrapper
   after committing the implementation/config changes.
2. Remove the HF reference model selection from the rendered environment and
   register the built-in native model. Inspect the actual worker imports/model
   class; a successful SGLang HTTP health check is insufficient evidence.
3. Qualify the native full-attention backend on A6000 first. The L40S reference
   used `fa3`, linear-attention backend `triton` and
   `SGLANG_USE_SGL_FA3_KERNEL=1`; do not assume that FA3 path supports SM86. Use BF16 weights/logits and FP32 recurrent
   state. Keep graphs, prefix caching and chunked prefill disabled.
4. Set an initial cap of 16 running requests per engine, within the standalone
   tested range. Preserve the lag-1 hard guard; start with the previous training
   admission cap of eight concurrent samples across both engines. Measure actual
   occupancy before increasing admission. Eval can use the tested engine slots;
   verify its concurrency and memory behavior in N1.
5. Verify live learner publication updates every packed parameter representation
   in both engines, invalidates stale recurrent/cache state and produces the
   expected new weight version. A same-checkpoint disk reload is not this test.
6. Fix and test restart accounting: scheduler preemption previously consumed the
   application's three-retry ledger. Keep a finite failure retry budget, classify
   preemptions explicitly, and ensure scheduler/application limits agree. Do not
   simply raise retry counts and call restoration solved.
7. Recheck native rollout versus learner logprobs on identical inputs, including
   tails and long responses. Preserve the known vLLM selected-token failure in
   the report. Before N2, either resolve it or explicitly record that N2 is an
   exploratory stack comparison with residual numerical mismatch; do not label
   it a strict equivalence result. Stop on nonfinite loss/ratios or unexplained
   policy-version discrepancies.

## Comparison and reporting

Compare N2 with the existing Core/vLLM result at updates 0/100/200/300/400/500.
Report accuracy, reward, answer length, truncation, gradient norm, clipping,
rollout/learner logprob differences and policy lag. Keep numerical and
end-to-end learning comparisons separate.

For speed, report wall-clock seconds per completed optimizer update, median and
tail latency between eval boundaries, and **minutes per 100 updates**. Report
startup, generation, learner compute, publication, evaluation and checkpoint
time separately. Give both active-runtime throughput and total elapsed time
including queueing/preemption; neither W&B `_step` nor learner compute time alone
is the training rate. Include GPU model/count and learner/inference split.

The previous stacks intentionally retained differences: Core/vLLM used exact
string answer checking, its existing warmup/gradient behavior and eval top-p 1;
MILES uses numeric answer checking, zero warmup, grad clip 1 and eval top-p 0.95.
PPO upper clipping is matched at 0.272. These choices make this a practical
whole-stack comparison, not an isolated causal test of SGLang. Preserve them for
this first run, as requested. If needed, evaluate both saved final checkpoints
later with one shared scoring/sampling protocol; that does not require retraining
the old baseline.

During runs, monitor actual optimizer updates, generation/evaluation counts,
committed checkpoints and restoration receipts. Flag a suspected stall after
20 minutes without optimizer, generation or evaluation progress; distinguish
startup, evaluation, queueing, preemption and application failure. A running
Beaker job or repeated producer waits are not evidence of progress.
