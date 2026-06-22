# Design: On-Policy Distillation in Open Instruct

Status: design proposal

Primary target: `open_instruct/grpo.py` with the OLMo-core backend.

Reference: [verl On-Policy Distillation docs](https://verl.readthedocs.io/en/latest/algo/opd.html).

## Summary

Implement On-Policy Distillation (OPD) as a reusable teacher-scoring and
distillation-loss layer, integrated first with the OLMo-core GRPO stack.

The v1 loss should match the stable verl-style GKD OPD recipe:

```text
loss_mode = forward_kl_topk
use_policy_gradient = false
```

This means:

- The student samples rollouts from its current policy.
- A teacher scores the states visited by the student rollout.
- The learner directly backpropagates a teacher-top-k forward KL loss.
- The same infrastructure can run as either GRPO + OPD or pure OPD.

The first implementation should not target `olmo_core_finetune.py` or `dpo.py`
directly. Those are offline SFT/DPO paths today, while OPD requires online
student rollouts. Once the shared loss and scorer are in place, SFT/DPO can get
separate distillation integrations.

## Background

For a prompt `x`, the student samples a response:

```text
y ~ pi_theta(. | x)
```

At response token `t`, the state is:

```text
s_t = (x, y_<t)
```

OPD trains on these student-induced states. A teacher distribution `nu(. | s_t)`
provides token-level supervision at the exact prefixes the student actually
visited.

This differs from standard KD or SFT:

- Standard KD often trains on teacher-generated trajectories.
- SFT trains on fixed dataset trajectories.
- OPD trains on trajectories sampled from the current student.

This also differs from RLVR:

- RLVR gives sparse outcome rewards.
- OPD gives dense token-level teacher supervision at every response position.

## Loss Variants

verl exposes two OPD families. The implementation should start with the first.

### GKD OPD

GKD OPD directly minimizes a distribution-level KL between the teacher and
student at student-induced states.

The full forward KL is:

```text
KL(nu || pi_theta)
  = sum_{v in V} nu(v | s_t) * [log nu(v | s_t) - log pi_theta(v | s_t)]
```

Full-vocabulary KL is expensive because it requires teacher probabilities for
the entire vocabulary at every response position. Instead, v1 should implement
the teacher-top-k approximation:

```text
L_forward_kl_topk(s_t)
  = sum_{v in TopK(nu(. | s_t))}
      nu(v | s_t) * [log nu(v | s_t) - log pi_theta(v | s_t)]
```

Important details:

- `TopK` is selected by the teacher distribution, not the student distribution.
- Teacher probabilities are not renormalized over the top-k set.
- The missing tail mass is intentionally omitted from the approximation.
- Metrics should log teacher top-k mass so we can tell whether `k` is large
  enough.
- This loss uses direct backprop through `log pi_theta(v | s_t)`.

This is the recommended v1 loss.

### PG OPD

PG OPD uses a sampled-token estimator of reverse KL as a reward:

```text
KL(pi_theta || nu)
  = E_{y_t ~ pi_theta}[
      log pi_theta(y_t | s_t) - log nu(y_t | s_t)
    ]
```

A single-sample k1 estimator gives a token reward:

```text
r_t = stop_grad(log nu(y_t | s_t) - log pi_theta(y_t | s_t))
```

Then a policy-gradient update increases or decreases the sampled token.

This is not the v1 target.

Reasons:

- It only updates the sampled token.
- It discards most of the top-k distributional teacher signal.
- It is higher variance than direct forward-KL backprop.
- It is a different optimization path from the SDFT-style per-token forward KL
  we want to support.

## Chosen V1 Behavior

Add OPD to the OLMo-core GRPO online pipeline with these default semantics:

```text
opd_enabled = false
opd_loss_mode = forward_kl_topk
opd_use_policy_gradient = false
opd_topk = 128
opd_loss_coef = 1.0
opd_use_task_rewards = true
opd_teacher_prompt_mode = student_prompt
```

When `opd_use_task_rewards=true`, train with:

```text
loss = grpo_pg_loss + beta * ref_kl + opd_loss_coef * opd_loss
```

When `opd_use_task_rewards=false`, train with:

```text
loss = opd_loss_coef * opd_loss
```

This "pure OPD" mode still uses the GRPO online rollout infrastructure, but the
task-reward policy loss is zeroed out. In that mode, `beta * ref_kl` should also
be ignored or rejected with a clear validation error, because the user asked for
distillation-only training.

For GRPO + OPD runs, keep the existing reference KL configurable. Most OPD
recipes should run with `--beta 0.0 --load_ref_policy False`, but the code should
not silently override existing GRPO behavior. Instead, warn when OPD is enabled
and `beta > 0`.

## Architecture

### Current GRPO Flow

Current OLMo-core GRPO has the right online structure:

1. `LLMRayActor` vLLM workers sample student responses.
2. `DataPreparationActor` receives rollouts and rewards.
3. `DataPreparationActor` computes GRPO advantages and packs sequences.
4. `StreamingDataLoader` serves `CollatedBatchData` to learners.
5. `GRPOTrainModule.train_batch` recomputes student logprobs and applies GRPO.

OPD should hook between steps 2 and 3:

1. Collect student responses.
2. Build teacher scoring requests.
3. Score `teacher_prompt + student_response` with teacher vLLM.
4. Attach teacher top-k targets.
5. Pack and train.

This preserves one batch contract for GRPO, GRPO + OPD, and pure OPD.

### Teacher Scoring

Add a teacher scorer actor that owns one teacher vLLM engine or replica.

Input per sample:

```text
teacher_prompt_tokens: list[int]
response_tokens: list[int]
response_mask: list[int]
metadata: dict
```

For v1 external OPD:

```text
teacher_prompt_tokens = student_prompt_tokens
```

For later OPSD:

```text
teacher_prompt_tokens = demo_conditioned_prompt_tokens
```

The scorer should request teacher `prompt_logprobs=opd_topk` over:

```text
teacher_prompt_tokens + response_tokens
```

Then it should extract the response-token positions and return:

```text
teacher_topk_token_ids: Tensor[response_len, K]
teacher_topk_logprobs:  Tensor[response_len, K]
```

Alignment rule:

- For response token `y_t`, use the teacher prompt-logprob entry at the position
  of `y_t` inside `teacher_prompt + response`.
- That entry represents the teacher distribution conditioned on
  `teacher_prompt + y_<t`.
- Ignore or sentinel-fill non-trainable response positions.

The scorer should use the same extraction strategy as `sample_logits_vllm.py`,
but it should return in-memory tensors rather than compressed parquet.

### Student Loss

The learner still performs the normal student forward pass.

For OPD, gather student logprobs at the teacher top-k token ids:

```text
student_topk_logprobs = log pi_theta(teacher_topk_token_ids | s_t)
```

Then compute:

```text
teacher_topk_probs = exp(teacher_topk_logprobs)
opd_loss_bt =
    sum_k teacher_topk_probs_btk
      * (teacher_topk_logprobs_btk - student_topk_logprobs_btk)
```

Apply the same response/tool mask used by GRPO:

```text
masked_opd_loss = masked_mean(opd_loss_bt, response_mask)
```

Metrics should also compute:

```text
teacher_topk_mass = sum_k exp(teacher_topk_logprobs)
student_topk_mass = sum_k exp(student_topk_logprobs)
```

These are diagnostic metrics. Low teacher mass means `opd_topk` is too small for
a faithful forward-KL approximation.

## Config And Public Interface

Add a new dataclass in `open_instruct/opd_utils.py`:

```python
@dataclass
class OPDConfig:
    opd_enabled: bool = False
    opd_loss_mode: Literal["forward_kl_topk"] = "forward_kl_topk"
    opd_use_policy_gradient: bool = False
    opd_use_task_rewards: bool = True
    opd_loss_coef: float = 1.0
    opd_topk: int = 128
    opd_loss_max_clamp: float | None = None
    opd_log_prob_min_clamp: float | None = None

    opd_teacher_source: Literal["external", "self_sync"] = "external"
    opd_teacher_model_name_or_path: str | None = None
    opd_teacher_revision: str | None = None
    opd_teacher_num_engines: int = 1
    opd_teacher_tensor_parallel_size: int = 1
    opd_teacher_gpu_memory_utilization: float = 0.8
    opd_teacher_enable_prefix_caching: bool = True
    opd_teacher_max_model_len: int | None = None

    opd_teacher_prompt_mode: Literal["student_prompt", "demo_conditioned"] = "student_prompt"
    opd_teacher_prompt_column: str = "teacher_input_ids_prompt"
```

Validation:

- `opd_enabled=false` leaves all behavior unchanged.
- `opd_use_policy_gradient` must be `False` in v1.
- `opd_loss_mode` must be `forward_kl_topk` in v1.
- `opd_topk > 0`.
- `opd_loss_coef >= 0`.
- External teachers require `opd_teacher_model_name_or_path`.
- Teacher and student must share tokenizer/vocab.
- `opd_use_task_rewards=false` should reject or ignore `beta > 0` explicitly.
- `opd_teacher_prompt_mode=demo_conditioned` requires the teacher prompt column.

Parser wiring:

- Add `OPDConfig` to `open_instruct/grpo.py`.
- Optionally add it to `grpo_fast.py` after the OLMo-core path is stable.
- Do not add it to `olmo_core_finetune.py` or `dpo.py` in v1.

## Data Model Changes

Extend `PackedSequences` and `CollatedBatchData` with optional teacher fields:

```text
teacher_topk_token_ids: list[Tensor] | None   # [B, T, K]
teacher_topk_logprobs:  list[Tensor] | None   # [B, T, K]
```

Packing rules:

- Query tokens get sentinel teacher ids/logprobs.
- Response tokens get teacher top-k targets.
- Padding positions get sentinel values.
- Teacher fields are present only when OPD is enabled.

Sequence parallelism:

- `UlyssesSPSplitter` currently pads/slices `[B, T]` tensors along the last
  dimension.
- Teacher fields are `[B, T, K]`.
- Splitter logic must pad/slice along dimension `T`, not dimension `K`.
- Use pad value `0` for token ids and `-inf` or `INVALID_LOGPROB` for logprobs.

Mock data:

- `StreamingDataLoader.get_mock_batch` should include valid teacher fields when
  OPD is enabled and `None` otherwise.
- Tests should cover both cases.

## File-Level Implementation Plan

### `open_instruct/opd_utils.py`

New module containing:

- `OPDConfig`
- teacher scoring request/result dataclasses
- top-k prompt-logprob extraction helpers
- `compute_forward_kl_topk`
- student top-k gather helper
- OPD metrics helpers
- validation helpers

Use `logger_utils.setup_logger(__name__)`.

### `open_instruct/vllm_utils.py`

Add a teacher scorer actor or reusable base actor for logprob-only scoring.

The actor should:

- launch a teacher vLLM engine
- request `prompt_logprobs=opd_topk`
- set `max_tokens=1`
- set `detokenize=False`
- return top-k prompt logprob tensors

It should not use tool execution or reward verification.

### `open_instruct/data_types.py`

Add optional teacher top-k fields to `CollatedBatchData`.

The `__getitem__`, `__len__`, and `.to(...)` style helpers should keep working
through dataclass field iteration.

### `open_instruct/rl_utils.py`

Update `PackedSequences` and `pack_sequences` to carry teacher top-k fields.

Filtering must keep response tokens, masks, vLLM logprobs, and teacher top-k
targets aligned.

### `open_instruct/data_loader.py`

Update `DataPreparationActor`:

1. Accumulate inference batches as today.
2. Apply truncation masking as today.
3. If OPD is enabled, call teacher scorer before packing.
4. Pass teacher top-k targets into `pack_sequences`.
5. Collate teacher top-k targets for each worker.

Also update:

- `prepare_collated_data_for_workers`
- mock batch construction
- saved rollout metadata only with scalar OPD diagnostics, not full top-k tensors

### `open_instruct/grpo.py`

Add `OPDConfig` parsing and teacher scorer setup.

Pass:

- `opd_config`
- teacher scorer handles

into `DataPreparationActor`.

### `open_instruct/olmo_core_train_modules.py`

Update `GRPOTrainModule.train_batch`:

- normal student forward still computes sampled-token logprobs for GRPO
- if OPD is enabled, gather student logprobs at teacher top-k ids
- compute `forward_kl_topk`
- add it to the loss according to `opd_use_task_rewards`
- record OPD metrics

Do not change `DPOTrainModule` in v1.

### `open_instruct/grpo_utils.py`

Prefer keeping OPD-specific math in `opd_utils.py`, but add or refactor shared
forward helpers if needed so the student forward can return:

- sampled-token logprobs
- optional student top-k logprobs
- optional entropy

Avoid a second student forward for OPD.

## OPSD Extension

On-policy self-distillation should reuse the OPD scorer and loss.

The difference is the teacher prompt:

```text
student prompt = normal problem prompt
teacher prompt = problem prompt + demonstration / privileged context
completion = same student-generated completion
```

The distribution comparison becomes:

```text
pi_theta(. | normal_prompt + y_<t)
vs.
pi_teacher(. | demo_prompt + y_<t)
```

For same-model OPSD, add:

```text
opd_teacher_source = self_sync
opd_teacher_prompt_mode = demo_conditioned
```

`self_sync` means the teacher scorer is initialized from the student model and
receives learner weight updates at the same cadence as the rollout policy vLLM.
This avoids implementing OPSD as a static snapshot unless the user explicitly
asks for that variant.

Dataset transforms should produce:

```text
teacher_input_ids_prompt
```

The v1 OPSD implementation should require this pre-tokenized teacher prompt
field. It should not try to reconstruct arbitrary chat templates inside the
training loop.

## SFT And DPO Scope

### `olmo_core_finetune.py`

This path trains from pre-tokenized numpy SFT data with a vanilla OLMo-core
`TransformerTrainModule`. It is not online today.

Possible future work:

- offline teacher-top-k distillation over fixed SFT examples
- an online SDFT trainer that samples student completions before training
- a custom OLMo-core distillation train module

Only the second option is truly OPD.

### `dpo.py`

This path trains on offline chosen/rejected pairs. `DPOTrainModule` currently
reduces each branch to scalar sequence logprobs for preference loss.

Possible future work:

- add teacher-top-k distillation as an auxiliary regularizer
- add online generation of preference candidates before DPO
- implement a DPO-specific per-token top-k gather path

This should come after OLMo-core GRPO OPD is working.

## Metrics

Record these metrics per training step:

```text
train/opd_loss
train/opd_forward_kl_topk
train/opd_teacher_topk_mass
train/opd_student_topk_mass
train/opd_teacher_entropy_topk
train/opd_sampled_token_in_teacher_topk
time/opd_teacher_scoring
```

For GRPO + OPD, keep current GRPO metrics unchanged.

For pure OPD, GRPO reward metrics may still be logged if rewards are computed,
but they must not contribute to the loss.

## Testing Plan

Unit tests:

```bash
uv run pytest open_instruct/test_opd_utils.py
uv run pytest open_instruct/test_rl_utils.py
uv run pytest open_instruct/test_utils.py
uv run pytest open_instruct/test_olmo_core_train_modules.py
```

Coverage:

- top-k extraction alignment
- unnormalized forward-KL math
- logprob clamps
- masking
- teacher top-k mass metrics
- pack/unpack/collation alignment
- sequence-parallel splitting of `[B, T, K]`
- GRPO loss composition with and without task rewards
- validation failures for invalid config combinations

Integration tests:

- mocked teacher scorer inside `DataPreparationActor`
- OLMo-core GRPO train batch with synthetic teacher targets
- pure OPD mode where policy loss is disabled

GPU validation:

- add a Beaker debug script for small OLMo-core GRPO + OPD
- launch with `scripts/train/build_image_and_launch.sh`
- run `scripts/test/run_gpu_pytest.sh` before PR if GPU coverage is required

Final checks:

```bash
make style
make quality
```

## Acceptance Criteria

The implementation is complete when:

- `opd_enabled=false` is behaviorally unchanged.
- OLMo-core `grpo.py` can run GRPO + OPD with an external teacher.
- OLMo-core `grpo.py` can run pure OPD by disabling task rewards.
- Teacher top-k targets are aligned with student response tokens.
- No extra student forward is required for OPD.
- Sequence parallelism works with teacher top-k tensors.
- OPD metrics make top-k approximation quality visible.
- OPSD can be added by providing a demo-conditioned teacher prompt without
  rewriting the scorer or loss.

## Implementation Checklist

This section is intentionally kept as a running teaching/debugging checklist.

- [x] Stage 1: Current GRPO data flow
  - Problem: OPD needs online student rollouts, not a fixed offline dataset.
  - Current repo fact: OLMo-core GRPO already routes student rollouts through
    `DataPreparationActor` into `GRPOTrainModule.train_batch`.
  - Decision: attach teacher scoring between rollout accumulation and sequence
    packing, then keep learner input as collated tensors.
- [x] Stage 2: DistillKit sparse signal and loss
  - Problem: teacher full-vocab logits are too expensive for online OPD.
  - Decision: add reusable DistillKit sparse teacher signals and direct
    unnormalized teacher-top-k forward KL.
  - Edge case: missing top-k entries use `-inf` logprobs and contribute zero.
- [x] Stage 3: Teacher scorer alignment
  - Problem: vLLM prompt logprobs may omit the first prompt-token slot.
  - Decision: extraction accepts both alignments and slices response-token rows
    from `teacher_prompt + student_response`.
- [x] Stage 4: Batch schema and sequence-parallel splitting
  - Problem: existing tensors are `[B, T]`; OPD top-k tensors are `[B, T, K]`.
  - Decision: optional `teacher_topk_token_ids` and `teacher_topk_logprobs`
    travel through packing/collation and Ulysses slices them on `T`.
- [x] Stage 5: Learner loss integration
  - Problem: OPD should not require a second student forward pass.
  - Decision: gather sampled-token logprobs and teacher-top-k student logprobs
    from the same shifted logits, then add the masked OPD loss.
- [x] Stage 6: Validation
  - Focused local tests passed for DistillKit loss/extraction and teacher-top-k
    packing.
  - `make style` and `make quality` passed locally.
  - Local macOS validation could not collect vLLM-dependent GRPO tests because
    `vllm` is not installed in this environment.
  - GPU/Beaker validation is still required before opening a production PR.
