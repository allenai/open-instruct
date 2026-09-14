# Qwen3.5 OPD through Open Instruct: prototype and bounded run plan

Status: proposed implementation and experiment plan, September 14, 2026.
No OPD training run has been launched or qualified by this investigation.
Source inspected: `robertb/miles-olmo-core`, Open Instruct `26c9f7926`.

Implement two text-only, full-parameter on-policy distillation experiments:
Qwen3.5-4B and Qwen3.5-2B learners, each supervised by a frozen Qwen3.5-9B
teacher. Use upstream Miles' Megatron trainer and SGLang rollout/scoring,
exposed through Open Instruct's configuration, preparation, launch and results
workflow. Start with 4B, then exercise 2B using the same protocol.

The learner generates, the teacher scores the learner's exact token sequence,
and the learner updates. Training inputs are prompts, not preference pairs.
No separate frozen learner reference is required by the proposed pure-OPD
objective. A checkpoint initialization flag named `ref-load` must not be confused
with allocating a reference model or enabling reference-KL regularization.

## 1. Findings and implementation decision

Upstream Miles supplies sampled-token OPD, teacher scoring and loss integration.
Its external SGLang teacher supports differently sized models with compatible
token IDs. Use that mode for 9B supervising 2B/4B; its in-process Megatron teacher
loads weights into the learner architecture and is not the starting point for
this size mismatch.
[Upstream OPD documentation](https://github.com/radixark/miles/blob/main/docs/advanced/on-policy-distillation.md).

| Current component | Required extension |
|---|---|
| `RunSpec.compile()` creates Core/GRPO settings | Explicit algorithm/backend selection with backend-specific defaults |
| `RunConfig` emits `train_backend=olmo_core` and rejects `use_opd` | Separate Megatron config path; preserve Core validation |
| `workflow.parse_runtime()` requires Core | Parse the selected backend and validate its dependencies |
| `driver.py` implements Core publication/checkpoints | Dispatch Megatron runs to pinned Miles `train.py` |
| `topology.py` allocates trainers, rollout engines and judges | Independently owned teacher GPU/service |
| `run_data.py` prepares task prompts and evaluation data | Reuse prompts/manifests; separate teacher training scores from task evaluation |
| Launch commands run Core attention preflight | Backend-specific preflight and environment |

Removing the Core OPD rejection alone does not implement support. Megatron must
not enter the Core driver, native checkpoint adapter, router replay, mixed-policy
refresh or Core model converter.

## 2. Image and patch strategy

Keep the current patched stack as the first candidate. Diagnose individual
incompatibilities before changing patches. Preserve future Olmo/Megatron use
without adding work to qualify it in this prototype.

The Open Instruct lock authenticates base image `01M24E7MSDGN2QFW1T8Z31BCKS`,
Docker ID `sha256:fe34fb1fef4910eb6610d8458a97c778360f67d0b38fbc0952cec28a321be629`.
It overlays Miles `dbbab1566ae438f7202fff653eae938e07b1d4b6` and checksum-pinned
`runtime/miles/patches/miles.patch`.

A read-only, network-disabled, no-GPU inventory of the locally available
`open-instruct:miles-fast-qualified-2c477efd5` image found:

| Component | Observed value |
|---|---|
| Docker image | `sha256:60aa54bfdeba71c49a863e393d7715c567d70cf293512f76f3a6939a6502759f` |
| PyTorch / CUDA | `2.13.0+cu130` |
| Transformers | `5.12.1` |
| SGLang | `0.5.19.dev49+g3145136` |
| Megatron Core | `0.19.0`, base `235952df607b3820716e5e67728a5ab470ca33ae` |
| NVIDIA Megatron Bridge | `0.7.0`, source `db723bae699dae5d29003ec4789c67730a343c32` |
| Olmo Megatron | `b84044ffb0d52620ec1599eda19d8f3d1de816b4` |
| FLA / Transformer Engine | `0.5.2` / `2.17.0` |
| `mbridge` | Not importable; no distribution metadata |

This establishes package presence, not Qwen execution compatibility. Inherited
image labels can describe older source layers; capture actual module paths and
overlay identities as well as labels in the candidate preflight.

**Concrete dependency addition:** Miles' `tools/convert_hf_to_torch_dist.py`
imports `mbridge.AutoBridge`, distinct from NVIDIA `megatron.bridge`. The pinned
Miles Dockerfile installs `ISEEKYAN/mbridge` at
`89eb10887887bc74853f89a4de258c0702932a1c`; use this as the initial candidate pin.
Bake it into the reusable dependency/source layer, record its revision and
artifact hash, and verify dependencies without allowing an unconstrained install
to replace Torch, Megatron or Transformers. This is a dependency-layer change,
not just copying application code. Do not install packages at job startup.

| Patch group found | Initial treatment |
|---|---|
| Megatron empty/unowned optimizer buckets and absent checkpoint globals | Retain; verify dense optimizer and save/resume |
| Megatron DeepEP-V2 transport/capacity and FP32 MoE routers | Retain; dense Qwen selects no MoE dispatch/replay path |
| Miles padding-mask forwarding | Retain; test Qwen forward signatures and masked padding |
| Miles deferred bridge registration | Retain; test Qwen conversion/export and namespace imports |
| Miles teardown, broadcast locking and publication fixes | Retain; test native Megatron completion |
| Core adapter and Olmo model hooks | Keep available but do not activate for Qwen/Megatron |

If a patch changes dense-Qwen numerics or prevents imports, reproduce against
the pinned upstream file, then narrow/guard that change. If the Megatron base
itself is incompatible, prepare a separately pinned candidate stack and requalify
it; do not replace the shared base or remove every patch as the first experiment.
Keep Qwen `PYTHONPATH` and Ray worker setup free of historical Olmo-only
monkeypatch hooks.

## 3. Public interface and implementation boundaries

Keep `python -m open_instruct.miles {plan,validate,train,run,status}`. Extend the
structured configuration with these proposed fields; they are not accepted by
the current parser:

| Proposed field | First supported meaning |
|---|---|
| `training.algorithm = "opd"` | Pure OPD; existing files default to GRPO |
| `trainer.backend = "megatron"` | Native Miles execution; existing files keep Core |
| `teacher.source`, `teacher.revision` | Frozen HF checkpoint with resolved identity |
| `teacher.mode = "managed"` | Teacher process owned by this allocation |
| `teacher.gpus = 1`, `teacher.tensor_parallel_size = 1` | Explicit service allocation |
| `teacher.max_context_length`, `teacher.max_running_requests` | Bounded scoring capacity |
| `distillation.kl_coef = 1.0` | Teacher signal coefficient |
| `distillation.log_prob_top_k = 0` | Sampled-token path |
| `distillation.task_reward_weight = 0.0` | First implementation accepts zero only |
| `megatron.model_type` | Validated Qwen3.5-4B/2B model argument definition |
| `megatron.tensor_parallel_size = 2` | Two trainer ranks; PP/CP/EP remain one |

Reuse existing optimizer, model/output, data, tracking, inference, launch and
checkpoint cadence fields where meanings agree. Reject ambiguous passthrough
overrides of managed fields. One rollout batch equals one optimizer step in this
pilot; distinguish total target steps from additional steps on resume.

Implementation chunks:

1. `run_spec.py`, `__main__.py`, `config.py`: backend-aware compilation and
   validation, with a separate Megatron compiler/config module. Preserve Core
   defaults and generated-reference coverage. Initially advertise Core+GRPO and
   Megatron+OPD, not every algorithm/backend combination.
2. `workflow.py` plus a Megatron execution module: reuse directory locks,
   preparation manifests and result capture; invoke pinned Miles `train.py` in
   a subprocess with explicit arguments/environment. Use native Megatron
   checkpoint/export behavior, not Core resume manifests.
3. `topology.py`, `cluster.py`, `launch.py`: one-node teacher allocation,
   backend-specific preflight, image identity and receipts. Reuse service process
   primitives without treating the teacher as an LLM judge.
4. OPD adapter: delegate scoring/postprocessing to Miles; add only Open Instruct
   task-evaluation and lifecycle integration.
5. `run_data.py`/preparation: prompt-only OPD inputs, tokenizer identity checks,
   HF-to-Megatron learner conversion and immutable preparation manifests.
6. Add `configs/miles/examples/opd-qwen35-{4b,2b}.toml`, focused tests,
   configuration-reference entries and an OPD guide after implementation.

Pure OPD compiles to `use_opd=true`, `opd_type=sglang`, `opd_kl_coef=1.0`,
`opd_log_prob_top_k=0`, upstream OPD reward/postprocessing hooks and the teacher
`/generate` endpoint. Keep advantage computation enabled: zero task advantages
receive the OPD token signal. Disable filters that discard identical/zero task
reward groups. Do not inherit reference KL, entropy bonuses or task reward from
a GRPO starter.

Pin one rollout implementation and test sampled-token field transport through
it. The top-k student-side strategies in the pinned runtime need its v1 rollout;
defer top-k, full async and policy-refresh experiments.
[Upstream example](https://github.com/radixark/miles/blob/main/examples/on_policy_distillation/run-qwen3-8B-opd.sh).

## 4. Model and data preparation

Resolve immutable HF revisions for all three models, dataset revisions,
tokenizer files and the chat template. Compare token-to-ID mappings,
added/special tokens and representative encodings; equal vocabulary sizes alone
are insufficient. Teacher scoring must consume the learner token IDs directly,
with correct next-token and response-span alignment.

Use text-only GSM8K from existing preparation: 256 seeded training prompts and
64 held-out test prompts, seed 17. Answers remain evaluation-only; correctness
reward is zero during training. Keep prompt IDs identical across learners but
sample fresh learner-specific rollouts. Teacher probabilities depend on the
entire generated prefix and cannot be cached by prompt alone.

Use non-thinking rendering for the initial short mechanics exercise, explicitly
recorded in template and eval configuration. This does not establish long-reasoning
behavior. Subsequent thinking runs need their own length budgets and matching
render/mask rules. Original HF assets remain read-only; prepared assets go to a
fresh run-owned WEKA directory.

Convert each learner with the pinned Miles Qwen spec and `mbridge`; the teacher
stays HF. Determine explicitly how vision-encoder and MTP tensors are preserved
or excluded, and whether the export is text-only. Do not label a text-only export
as a complete multimodal checkpoint. Check `A_log` stays FP32 through load,
training, publication and export.

4B has a named Miles definition. Add a narrowly scoped 2B definition from its
pinned HF `text_config`, including layer count, hidden/FFN sizes, attention heads
and Gated DeltaNet dimensions, using the existing Qwen3.5 spec. Validate conversion
and probabilities; changing the 4B model name is insufficient.
[Dense model guide](https://github.com/radixark/miles/blob/main/docs/models/qwen/qwen3-5.md).

## 5. Teacher ownership and evaluation

First topology: one four-B300 Beaker allocation on `ai2/holmes`, comprising two
Megatron trainer GPUs (TP2), one TP1 student SGLang GPU, and one TP1 teacher GPU.
This is a capacity proposal, not a measured fit. Exclude the teacher GPU from
Ray's visible pool; show physical allocation and role mapping in `plan`. Exercise
the learners sequentially so they do not compete for teacher capacity.

Start the teacher with a finite readiness deadline, verify checkpoint identity,
and complete an input-logprob request before training starts. Score exact input
IDs without generating a teacher continuation. Bound scoring concurrency
(initially four), request timeout (180 seconds) and retries (at most two for
transient failures). Missing/nonfinite/misaligned scores fail the batch; never
replace them with zero. Capture teacher logs and score latency separately.

Own service process handles/groups and clean up on success, error or termination.
Do not copy upstream example-wide `pkill python/ray/sglang` commands. A teacher
crash must terminate the run rather than strand learner workers.

Use the ordinary GSM8K verifier via an explicit eval hook or eval reward override:
OPD's training reward payload is not task correctness. Establish both starting
learner baselines and a 9B baseline with matching template, decoding settings and
limits. Track accuracy, response length, truncation and sample outputs alongside
teacher-signal metrics. Falling OPD loss alone is not evidence of task improvement.

## 6. Bounded experiment and acceptance gates

| Setting | Pilot value |
|---|---|
| Allocation | 4 B300 GPUs, one node, sequential learner runs |
| Precision | BF16 with required FP32 model parameters |
| Schedule | Synchronous; one optimizer step per rollout; no stale-rollout reuse |
| Rollout batch | 16 prompts × 2 responses = 32 sequences/update |
| Global batch | 32 sequences |
| Length | 1,024 generated tokens; 2,048 total; record/filter excessive prompts |
| Sampling | Temperature 1, top-p 1, seed 17 |
| Optimizer | Adam, LR 1e-6 constant, beta1 0.9, beta2 0.98, clip norm 1 |
| Other regularization | Weight decay 0, entropy 0, reference KL disabled |
| Parallelism | Trainer TP2/PP1/CP1/EP1; student/teacher serving TP1 |
| Memory controls | Microbatch 1, activation recomputation, bounded token batches |
| Distillation | Sampled-token reverse-KL signal, coefficient 1, task reward zero |
| Steps | Stop/save after 2 updates; fresh-process resume to 20 total updates |
| Evaluation | Before training and final, 64 held-out prompts |
| Checkpoints | Native saves at step 2 and final; final HF export |
| Tracking | Offline W&B plus JSON evidence in a fresh run root |
| Limits | 3-hour timeout, 1-hour minimum runtime per 4-GPU training job |

These are experimental choices, not upstream-qualified settings. Twenty updates
mean 640 sampled responses per learner. Impose an aggregate 12 allocated GPU-hour
cap per learner across its two segments, carrying the remaining timeout budget
into the resume job. Both learners together have at most 24 allocated GPU-hours
for training segments, excluding separately reported preparation/probe costs.
Stop early when gates pass; do not silently extend the budget for compilation.
Before submission, put a finite bound on any separate GPU conversion/probe job
and include it in the campaign estimate.

Run the following gates in order:

1. **Local CPU checks.** Schema/default/regression tests, compiled arguments,
   teacher exclusion from Ray, mock service readiness/failure/cleanup,
   deterministic data splits, mask/alignment tests and pure-OPD signal tests with
   zero task reward. Keep existing Core example plans passing. Run lint/format
   and required quality checks; no local CUDA installation needed for submission.
2. **Image preflight.** Build committed source. Inventory actual imports and
   revisions. Exercise `megatron.training`, both bridge namespaces, Qwen spec,
   FLA and OPD parser. Compare candidate-image CPU plan with submitting-checkout
   plan. Check the full imported plugin chain, not only top-level module lookup.
3. **GPU probability checks before updates.** Convert/load the learner; compare
   fixed response-token logprobs with HF and student SGLang; compare teacher
   scoring with HF; validate EOS, padding and response masks. Record numeric
   tolerances before judging results, calibrated against repeated BF16 reference
   passes. Verify gradients and sensitive dtypes. Matching decoded answers or two
   mutually consistent converted implementations is insufficient.
4. **Two-update 4B run.** Demonstrate nonzero teacher advantages/gradients and
   changed learner weights. Check the teacher stays fixed and serving receives
   the learner update. Save a complete native checkpoint and exit cleanly. On an
   identical captured batch, disabling OPD with all other losses zero must remove
   the distillation gradient.
5. **Fresh-process resume to update 20.** Restore optimizer, scheduler, counters
   and data cursor; verify restored policy before continuing. Record resume
   behavior without claiming bitwise future sampling reproduction. Evaluate,
   export HF, and reload the final export in a fresh process for generation and
   fixed-token scoring. Report task changes descriptively: 64 evaluation prompts
   and 20 steps do not establish quality gains.
6. **Repeat for 2B.** First pass its architecture/conversion checks, then the same
   bounded two-segment exercise. 4B success does not qualify 2B automatically.

On memory failure, reduce concurrency/token budgets first while retaining
recorded batch semantics. Do not silently switch model, precision, algorithm or
backend. Stop before the learning budget if alignment/publication checks fail.
If 9B does not outperform a learner on this task, report that; mechanics remain
testable, but a quality experiment needs a more informative task/prompt mix.

## 7. Launch, deliverables and completion

Use a focused implementation branch/worktree from the inspected integration.
Deliver four reviewable chunks: (1) runtime dependency/preflight and model
configs, (2) backend dispatch and teacher lifecycle, (3) data/eval/checkpoint
integration and tests, (4) examples and measured run report. Avoid expanding this
into a generic backend framework or implementing Core OPD as a prerequisite.

After implementation, the intended operator flow is:

```bash
# These proposed examples are runnable only after implementation.
cp configs/miles/examples/opd-qwen35-4b.toml runs/my-opd-4b.toml
python -m open_instruct.miles plan runs/my-opd-4b.toml
python -m open_instruct.miles validate runs/my-opd-4b.toml
# Commit implementation before build/launch; choose a fresh output path.
python -m open_instruct.miles run runs/my-opd-4b.toml
python -m open_instruct.miles status runs/my-opd-4b.toml
```

`run` continues through `scripts/train/build_image_and_launch.sh --miles`, using
the candidate built from those commits. `MILES_EXISTING_IMAGE` may only reuse an
image containing the prototype, not the existing Core-only image. GPU jobs can
use Holmes; every CPU-only preparation job needing WEKA uses `ai2/saturn`.
Models/data/checkpoints stay on WEKA; small logs, configuration and receipts go
to Beaker results. Verify existing workspace/budget access before submission.
For queues inspect `beaker job events`; follow the latest job on retries.

The report must include image/source/package identities, patch disposition,
model/tokenizer/data revisions, resolved arguments, role allocation, teacher
scoring validation, task metrics, GPU time, native and HF checkpoint paths,
resume/reload evidence and failures. Link actual experiments in numbered `Runs:`
format. Training smokes are not GPU pytest runs: do not reuse their IDs in
`GPU_TESTS=...`. If a PR changes `open_instruct/`, add its PR-linked changelog entry.

Completion means both learners execute OPD through the public Open Instruct
workflow, with validated teacher signals, observable updates, checkpoint lifecycle
and a bounded evidence report. Broader quality claims, thinking-mode training,
top-k OPD, mixed task rewards, multi-node operation and Core OPD are follow-ups.
