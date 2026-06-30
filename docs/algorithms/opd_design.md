# Design: On-Policy Distillation in Open Instruct

Status: implemented, experimental v1

Primary target: `open_instruct/grpo.py` with the OLMo-core backend.

Reference: [verl On-Policy Distillation docs](https://verl.readthedocs.io/en/latest/algo/opd.html).

## Summary

Open Instruct now implements On-Policy Distillation (OPD) as a reusable
teacher-scoring and distillation-loss layer, integrated first with the
OLMo-core GRPO stack.

The v1 loss matches the stable verl-style GKD OPD recipe:

```text
loss_mode = forward_kl_topk
use_policy_gradient = false
```

This means:

- The student samples rollouts from its current policy.
- A teacher scores the states visited by the student rollout.
- The learner directly backpropagates a teacher-top-k forward KL loss.
- The same infrastructure can run as either GRPO + OPD or pure OPD.

The v1 implementation does not target `olmo_core_finetune.py` or `dpo.py`
directly. Those are offline SFT/DPO paths today, while OPD requires online
student rollouts. The shared sparse signal/loss pieces can be reused later for
separate SFT/DPO distillation integrations.

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

verl exposes two OPD families. The v1 implementation starts with the first.

### GKD OPD

GKD OPD directly minimizes a distribution-level KL between the teacher and
student at student-induced states.

The full forward KL is:

```text
KL(nu || pi_theta)
  = sum_{v in V} nu(v | s_t) * [log nu(v | s_t) - log pi_theta(v | s_t)]
```

Full-vocabulary KL is expensive because it requires teacher probabilities for
the entire vocabulary at every response position. Instead, v1 implements the
teacher-top-k approximation:

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

OPD is added to the OLMo-core GRPO online pipeline with these default
semantics:

```text
opd_enabled = false
opd_loss_mode = forward_kl_topk
opd_topk = 128
opd_loss_coef = 1.0
opd_use_task_rewards = true
opd_teacher_prompt_mode = student_prompt
```

There is no `opd_use_policy_gradient` flag in v1. The policy-gradient OPD
variant is intentionally unimplemented.

When `opd_use_task_rewards=true`, train with:

```text
loss = grpo_pg_loss + beta * ref_kl + opd_loss_coef * opd_loss
```

When `opd_use_task_rewards=false`, train with:

```text
loss = opd_loss_coef * opd_loss
```

This "pure OPD" mode still uses the GRPO online rollout infrastructure, but the
task-reward policy loss is zeroed out. In the current implementation,
`grpo.py` requires pure OPD runs to set `--beta 0.0 --load_ref_policy false`
because the GRPO task loss and reference KL are not part of the objective.

For GRPO + OPD runs, keep the existing reference KL configurable. Most OPD
recipes should run with `--beta 0.0 --load_ref_policy False`, but the code does
not silently override existing GRPO behavior.

## Architecture

### Current GRPO Flow

Current OLMo-core GRPO has the right online structure:

1. `LLMRayActor` vLLM workers sample student responses.
2. `DataPreparationActor` receives rollouts and rewards.
3. `DataPreparationActor` computes GRPO advantages and packs sequences.
4. `StreamingDataLoader` serves `CollatedBatchData` to learners.
5. `GRPOTrainModule.train_batch` recomputes student logprobs and applies GRPO.

OPD hooks between steps 2 and 3:

1. Collect student responses.
2. Build teacher scoring requests.
3. Score `teacher_prompt + student_response` with teacher vLLM.
4. Attach teacher top-k targets.
5. Pack and train.

This preserves one batch contract for GRPO, GRPO + OPD, and pure OPD.

### Teacher Scoring

The OPD implementation adds a teacher scorer actor that owns one teacher vLLM
engine or replica.

Input per sample:

```text
query_tokens: list[int]
response_tokens: list[int]
```

For v1 external OPD:

```text
teacher_prompt_tokens = student_prompt_tokens
```

For later OPSD:

```text
teacher_prompt_tokens = demo_conditioned_prompt_tokens
```

The scorer requests teacher `prompt_logprobs=opd_topk` over:

```text
teacher_prompt_tokens + response_tokens
```

Then it extracts the response-token positions and returns:

```text
teacher_topk_token_ids: Tensor[response_len, K]
teacher_topk_logprobs:  Tensor[response_len, K]
```

Alignment rule:

- For response token `y_t`, use the teacher prompt-logprob entry at the position
  of `y_t` inside `teacher_prompt + response`.
- That entry represents the teacher distribution conditioned on
  `teacher_prompt + y_<t`.
- Prompt tokens are context only. Query/pad sentinel filling happens later in
  packing/collation; the learner mask decides which positions contribute to the
  loss.

The scorer uses the same broad extraction strategy as `sample_logits_vllm.py`,
but returns in-memory tensors rather than compressed parquet.

### Student Loss

The learner still performs the normal student forward pass.

For OPD, gather student logprobs at the teacher top-k token ids:

```text
student_topk_logprobs = log pi_theta(teacher_topk_token_ids | s_t)
```

Important temperature detail:

- GRPO sampled-token logprobs use the rollout temperature because they are part
  of the policy-gradient/importance-ratio objective for the sampling policy.
- OPD student top-k logprobs use raw `T=1` model logits because the teacher
  scorer returns raw teacher logprobs. This keeps the distillation objective as
  `KL(teacher_raw || student_raw)` rather than accidentally training
  `KL(teacher_raw || student_at_rollout_temperature)`.

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

Current v1 metrics also compute:

```text
teacher_topk_mass = sum_k exp(teacher_topk_logprobs)
sampled_token_in_topk = 1[sampled_token in teacher_topk]
```

These are diagnostic metrics. Low teacher mass means `opd_topk` is too small for
a faithful forward-KL approximation. Low sampled-token-in-top-k means the
teacher top-k support is often missing the tokens the student actually sampled.

Useful future diagnostics include `student_topk_mass` and teacher top-k entropy,
but they are not part of the current v1 metric set.

## Config And Public Interface

The v1 implementation keeps OPD fields in the existing GRPO streaming config
dataclass in `open_instruct/data_loader.py`. This was an intentional divergence
from the original standalone `OPDConfig` idea: OPD v1 is not a separate trainer,
and keeping the fields in the existing config avoids threading a parallel config
object through the GRPO stack.

Current v1 fields:

```python
opd_enabled: bool = False
opd_loss_mode: Literal["forward_kl_topk"] = "forward_kl_topk"
opd_topk: int = 128
opd_loss_coef: float = 1.0
opd_use_task_rewards: bool = True
opd_teacher_model_name_or_path: str | None = None
opd_teacher_model_revision: str | None = None
opd_teacher_num_engines: int = 1
opd_teacher_tensor_parallel_size: int = 1
opd_teacher_gpu_memory_utilization: float = 0.9
opd_teacher_dtype: str = "bfloat16"
opd_teacher_enforce_eager: bool = False
opd_teacher_enable_prefix_caching: bool = False
opd_teacher_prompt_mode: Literal["student_prompt"] = "student_prompt"
```

Notable omitted fields from the original proposal:

- `opd_use_policy_gradient`: PG OPD is future work.
- `opd_loss_max_clamp` and `opd_log_prob_min_clamp`: not needed for current
  stable runs; missing top-k entries are represented with `-inf` logprobs.
- `opd_teacher_source=self_sync`: future OPSD/self-distillation work.
- `opd_teacher_prompt_mode=demo_conditioned` and `opd_teacher_prompt_column`:
  future OPSD dataset/prompt work.
- `opd_teacher_max_model_len`: current v1 derives this from
  `max_prompt_token_length + response_length`.

Current validation:

- `opd_enabled=false` leaves all behavior unchanged.
- `opd_loss_mode` must be `forward_kl_topk` in v1.
- `opd_topk > 0`.
- `opd_loss_coef >= 0`.
- `opd_teacher_num_engines > 0`.
- `opd_teacher_tensor_parallel_size > 0`.
- `opd_teacher_prompt_mode` must be `student_prompt`.
- Pure OPD disables `filter_zero_std_samples` because rewards are neutral.
- At least one reward must be configured unless running pure OPD.
- `grpo.py` requires `opd_teacher_model_name_or_path` when OPD is enabled.
- Pure OPD hard-errors unless `beta == 0.0` and `load_ref_policy=false`.
- Teacher and student tokenizers must be compatible: every teacher vocab token
  must have the same id in the student tokenizer. Student-only extra tokens above
  the teacher vocab range are allowed to support harmless added pad tokens.

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

- `StreamingDataLoader.get_mock_batch` includes valid teacher fields when OPD
  is enabled and `None` otherwise.
- Tests should cover both cases.

## File-Level Implementation

### `open_instruct/opd_utils.py`

OPD-specific online orchestration helpers:

- teacher scoring request/result dataclasses
- `OPDTeacherScorerRayActor`
- teacher scorer Ray actor creation
- vLLM teacher initialization settings
- teacher scoring latency/throughput metrics

Use `logger_utils.setup_logger(__name__)`.

### `open_instruct/distillkit`

Reusable sparse distillation pieces live under DistillKit:

- `losses.py`: `forward_kl_topk_from_logprobs` (teacher-top-k forward KL from sparse logprobs)
- `vllm_logprobs.py`: vLLM prompt-logprob extraction helpers

This keeps distributional distillation math reusable for future offline
distillation, SFT distillation, or OPSD work. GRPO owns only the online rollout
and batch plumbing.

### `open_instruct/vllm_utils.py`

No OPD-specific teacher scorer was added here. This is intentional: the scorer
semantics are OPD-specific, while `vllm_utils.py` remains focused on the shared
policy rollout engine utilities.

### `open_instruct/data_types.py`

Adds optional teacher top-k fields to `CollatedBatchData`.

The `__getitem__`, `__len__`, and `.to(...)` style helpers keep working through
dataclass field iteration.

### `open_instruct/rl_utils.py`

Updates `PackedSequences` and `pack_sequences` to carry teacher top-k fields.

Filtering keeps response tokens, masks, vLLM logprobs, and teacher top-k
targets aligned.

### `open_instruct/data_loader.py`

`DataPreparationActor`:

1. Accumulates inference batches as today.
2. Applies truncation masking as today.
3. Calls teacher scorer before packing if OPD is enabled.
4. Passes teacher top-k targets into `pack_sequences`.
5. Collates teacher top-k targets for each worker.

Also covers:

- `prepare_collated_data_for_workers`
- mock batch construction
- scalar OPD timing/throughput metrics in step metrics

### `open_instruct/grpo.py`

Sets up teacher scorers when `opd_enabled=true`.

Pass:

- teacher scorer handles

into `DataPreparationActor`.

### `open_instruct/olmo_core_train_modules.py`

Updates `GRPOTrainModule.train_batch`:

- normal student forward still computes sampled-token logprobs for GRPO
- if OPD is enabled, gather student logprobs at teacher top-k ids
- compute `forward_kl_topk`
- add it to the loss according to `opd_use_task_rewards`
- record OPD metrics

Do not change `DPOTrainModule` in v1.

### `open_instruct/grpo_utils.py`

The student forward helper returns:

- sampled-token logprobs
- optional student top-k logprobs
- optional entropy

This avoids a second student forward for OPD.

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

This remains future work after the OLMo-core GRPO OPD v1 path.

## Metrics

Current v1 records these OPD metrics:

```text
loss/opd_avg
opd/teacher_topk_mass
opd/sampled_token_in_topk
time/opd_teacher_scoring
time/opd_teacher_scoring_worker_sum
opd/teacher_tokens_per_second
```

For GRPO + OPD, keep current GRPO metrics unchanged.

For pure OPD, GRPO reward metrics may still be logged if rewards are computed,
but they must not contribute to the loss.

Useful future metrics:

```text
opd/student_topk_mass
opd/teacher_entropy_topk
```

## Testing Plan

Unit tests:

```bash
uv run pytest open_instruct/distillkit/test_losses.py
uv run pytest open_instruct/test_rl_utils.py
uv run pytest open_instruct/test_utils.py
uv run pytest open_instruct/test_grpo_fast.py
```

Coverage:

- top-k extraction alignment
- unnormalized forward-KL math
- masking
- teacher top-k mass metrics
- pack/unpack/collation alignment
- sequence-parallel splitting of `[B, T, K]`
- GRPO loss composition with and without task rewards
- validation failures for invalid config combinations

GPU/integration tests:

- `uv run pytest open_instruct/test_opd_gpu.py` exercises live teacher scorer
  setup on GPU.
- launch with `scripts/train/build_image_and_launch.sh`
- run `scripts/test/run_gpu_pytest.sh` before PR if GPU coverage is required
- run a small OLMo-core GRPO + OPD Beaker script before scale-up

Final checks:

```bash
make style
make quality
```

## Acceptance Criteria

The v1 implementation is complete when:

- `opd_enabled=false` is behaviorally unchanged.
- OLMo-core `grpo.py` can run GRPO + OPD with an external teacher.
- OLMo-core `grpo.py` can run pure OPD by disabling task rewards.
- Teacher top-k targets are aligned with student response tokens.
- No extra student forward is required for OPD.
- Sequence parallelism works with teacher top-k tensors.
- OPD metrics make top-k approximation quality visible.
- The scorer/loss structure leaves a path for OPSD to add demo-conditioned
  teacher prompts without rewriting the sparse loss.

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
  - Temperature detail: sampled-token GRPO logprobs use rollout temperature;
    OPD teacher-top-k student logprobs use raw `T=1` model logits to match
    teacher `raw_logprobs`.
- [x] Stage 6: Validation
  - Focused local tests passed for DistillKit loss/extraction and teacher-top-k
    packing.
  - `make style` and `make quality` passed locally.
  - Local macOS validation could not collect vLLM-dependent GRPO tests because
    `vllm` is not installed in this environment.
  - GPU pytest was launched through `scripts/train/build_image_and_launch.sh
    scripts/test/run_gpu_pytest.sh`.
  - A 500-step Beaker rehearsal completed successfully with Qwen3-0.6B student,
    Qwen3-4B teacher, `opd_topk=128`, W&B tracking, and final checkpoint save:
    `01KVRJ1Y7HHQHWAMQ3ECBVAXCK`.

## Remaining Implementation Gap Audit

The original design contains a few ideas that are intentionally future work and
should not block v1:

- PG OPD / reverse-KL reward shaping.
- OPSD with `self_sync` teacher source.
- Demo-conditioned teacher prompts and `teacher_input_ids_prompt`.
- SFT/DPO-specific distillation integrations.
- Separate `opd_teacher_max_model_len` config.
- Loss/logprob clamp knobs.

The current v1 gaps worth considering before a production PR or larger scale
run are:

1. Consider adding `opd/student_topk_mass` as an alignment diagnostic.
2. Consider adding `opd/teacher_entropy_topk` as a teacher uncertainty
   diagnostic.
3. Run a matched GRPO-only control before making quality claims. The current
   Beaker rehearsal validates the system path, not OPD superiority.
