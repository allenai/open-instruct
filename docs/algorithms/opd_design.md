# On-Policy Distillation (OPD) for OLMo-core GRPO

Status: implemented, experimental v1

Entry point: `open_instruct/grpo.py` (OLMo-core backend). `grpo_fast.py` rejects
`opd_enabled=true` at startup.

Reference: [verl On-Policy Distillation docs](https://verl.readthedocs.io/en/latest/algo/opd.html).

## Summary

OPD adds dense, token-level teacher supervision to the online GRPO pipeline:

- The student samples rollouts from its current policy (unchanged GRPO flow).
- A frozen teacher vLLM scores the exact states the student visited, returning
  its top-k token ids and logprobs per response position.
- The learner backpropagates a teacher-top-k forward KL directly, either on top
  of the GRPO task loss (GRPO + OPD) or alone (pure OPD).

Everything is disabled by default (`opd_enabled=false` leaves all behavior
unchanged). The distillation math lives in `open_instruct/distillkit/` so it can
be reused by future offline/SFT distillation work; GRPO owns only the rollout
and batch plumbing.

## Background

For a prompt `x`, the student samples `y ~ pi_theta(. | x)`. At response token
`t`, the state is `s_t = (x, y_<t)`. OPD trains on these student-induced states:
a teacher distribution `nu(. | s_t)` supervises the exact prefixes the student
actually visited.

This differs from standard KD (teacher-generated trajectories), SFT (fixed
dataset trajectories), and RLVR (sparse outcome rewards): OPD gives dense
teacher supervision at every response position of student-sampled trajectories.

## Loss

The v1 loss is the verl-style GKD OPD recipe: direct forward KL on the
teacher's top-k support.

Full forward KL requires teacher probabilities over the whole vocabulary at
every position, which is too expensive online. Instead:

```text
L_forward_kl_topk(s_t)
  = sum_{v in TopK(nu(. | s_t))}
      nu(v | s_t) * [log nu(v | s_t) - log pi_theta(v | s_t)]
```

Key properties:

- `TopK` is selected by the teacher distribution, not the student.
- Teacher probabilities are **not** renormalized over the top-k set; the tail
  mass is intentionally omitted. `opd/teacher_topk_mass` makes the
  approximation quality visible.
- Missing top-k entries carry `-inf` teacher logprobs and contribute exactly 0
  (the student term is masked there to avoid `0 * inf = NaN`).
- Gradients flow directly through `log pi_theta(v | s_t)` — no policy-gradient
  estimator.

The alternative PG OPD family (sampled-token reverse-KL used as a reward) is
intentionally unimplemented: it updates only the sampled token, discards the
distributional top-k signal, and is higher variance. There is no
`opd_use_policy_gradient` flag.

## Loss Composition

```text
opd_use_task_rewards=true  (GRPO + OPD):
    loss = grpo_pg_loss + beta * ref_kl + opd_loss_coef * opd_loss

opd_use_task_rewards=false (pure OPD):
    loss = opd_loss_coef * opd_loss
```

Pure OPD reuses the full online rollout infrastructure but zeroes the
task-reward policy loss. It requires `--beta 0.0 --load_ref_policy false`
(enforced at startup), and disables `filter_zero_std_samples` automatically.
For GRPO + OPD, existing GRPO behavior (including reference KL) is left
configurable and is never silently overridden.

## Architecture

The existing OLMo-core GRPO flow:

1. `LLMRayActor` vLLM workers sample student responses.
2. `DataPreparationActor` receives rollouts and rewards.
3. `DataPreparationActor` computes GRPO advantages and packs sequences.
4. `StreamingDataLoader` serves `CollatedBatchData` to learners.
5. `GRPOTrainModule.train_batch` recomputes student logprobs and applies GRPO.

OPD hooks between steps 2 and 3: `DataPreparationActor` sends
`query_tokens + response_tokens` to teacher scorer actors, receives per-token
top-k targets, and threads them through packing/collation. The batch contract
is identical for GRPO, GRPO + OPD, and pure OPD.

### Teacher Scoring

`OPDTeacherScorerRayActor` (`opd_utils.py`) owns one frozen teacher vLLM
replica on its own placement-group GPU bundle (it never shares the learner or
rollout GPUs, and never receives weight updates). This holds in
`single_gpu_mode` too: that flag co-locates the rollout vLLM with the learner,
but the teacher always needs
`opd_teacher_num_engines * opd_teacher_tensor_parallel_size` additional GPUs,
and startup waits for the combined learner + teacher GPU count before creating
scorers so an under-provisioned cluster reports what it is waiting for instead
of hanging. Scoring requests
`prompt_logprobs=opd_topk` with `logprobs_mode="raw_logprobs"` over
`prompt_tokens + response_tokens` and extracts the response-token rows,
returning:

```text
teacher_topk_token_ids: Tensor[response_len, K]
teacher_topk_logprobs:  Tensor[response_len, K]
```

Alignment: the entry at the position of `y_t` represents the teacher
distribution conditioned on `prompt + y_<t`. Responses are pad-filtered before
scoring — vLLM can emit pad tokens mid-response, `pack_sequences` strips them
from what the learner trains on, and pad ids may not exist in the teacher
vocab — so teacher rows align with the pad-filtered response, and the scorer
hard-errors on any token id outside the teacher vocab. vLLM may omit the first
prompt-token slot; extraction accepts both alignments
(`distillkit/vllm_logprobs.py`). The teacher's `max_model_len` is
`max_prompt_token_length + response_length + 1` (headroom for the single dummy
generated token vLLM requires).

With multiple teacher engines, work is split into contiguous,
token-count-balanced ranges (`_make_token_balanced_ranges`) so one long-response
chunk doesn't serialize the step.

### Student Loss

The learner's single forward produces both GRPO and OPD views — no second
forward pass (`grpo_utils.forward_for_logprobs_and_topk`):

- GRPO sampled-token logprobs use the **rollout temperature** (they feed the
  importance ratio for the sampling policy).
- OPD student top-k logprobs use **raw T=1 logits**, matching the teacher's
  `raw_logprobs`. The objective is `KL(teacher_raw || student_raw)`, not
  `KL(teacher_raw || student_at_rollout_temperature)`.
- Teacher top-k ids out of the student's logit range raise a hard error;
  invalid ids are never clamped.

The OPD loss uses the same response/tool mask and token-count denominator as
the GRPO loss.

## Config

OPD fields live on the existing streaming config in
`open_instruct/data_loader.py` (OPD is not a separate trainer, so no parallel
config object is threaded through the stack):

```python
opd_enabled: bool = False
opd_loss_mode: Literal["forward_kl_topk"] = "forward_kl_topk"
opd_topk: int = 16
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

Deliberately not included in v1: PG OPD, loss/logprob clamp knobs (missing
entries are `-inf` and contribute 0), `opd_teacher_max_model_len` (derived),
and OPSD fields (`opd_teacher_source=self_sync`, demo-conditioned teacher
prompts) — see Future Work.

### Validation

Config-level (`StreamingDataLoaderConfig.__post_init__`):

- `opd_loss_mode` must be `forward_kl_topk`; `opd_topk > 0`;
  `opd_loss_coef >= 0`; engine counts positive;
  `opd_teacher_prompt_mode` must be `student_prompt`.
- `opd_use_task_rewards=false` requires `opd_enabled=true`.
- Pure OPD disables `filter_zero_std_samples` and rejects
  `active_sampling=true` (both depend on reward signal).
- At least one reward must be configured unless running pure OPD.

Startup (`grpo.py` + `opd_validation.py`, before any GPU-heavy setup):

- `opd_teacher_model_name_or_path` is required when OPD is enabled.
- Pure OPD hard-errors unless `beta == 0.0` and `load_ref_policy=false`.
- Tokenizer compatibility: every teacher vocab token must have the same id in
  the student tokenizer (student-only extra tokens above the teacher vocab
  range are allowed, e.g. an added pad token).
- Output vocab: the teacher's HF config must load and expose `vocab_size`
  (vLLM requires an HF-format teacher anyway), and it must not exceed the
  student's output dim (taken from the built OLMo-core config; falls back to
  the student HF config, skipping only for non-HF student checkpoints).

Runtime backstop: the learner forward raises if any teacher top-k id is
outside the student logit range.

## Data Model

`PackedSequences` and `CollatedBatchData` carry two optional fields, present
only when OPD is enabled:

```text
teacher_topk_token_ids: list[Tensor] | None   # [B, T, K]
teacher_topk_logprobs:  list[Tensor] | None   # [B, T, K]
```

Packing gives query/padding positions sentinel values (token id 0, logprob
`-inf`) so they contribute nothing; the learner's response mask decides what
enters the loss. Ulysses sequence parallelism pads/slices the `[B, T, K]`
tensors along `T` (pad values: `0` for ids, `-inf` for logprobs).
`get_mock_batch` produces valid teacher fields when OPD is enabled.

## File Map

- `open_instruct/distillkit/losses.py` — `forward_kl_topk_from_logprobs`
  (teacher-top-k forward KL from sparse logprobs).
- `open_instruct/distillkit/vllm_logprobs.py` — vLLM prompt-logprob top-k
  extraction.
- `open_instruct/opd_utils.py` — teacher scorer Ray actor, actor creation,
  vLLM teacher settings, scoring metrics.
- `open_instruct/opd_validation.py` — pure-OPD, tokenizer, and output-vocab
  validators.
- `open_instruct/data_loader.py` — config fields/validation, teacher scoring
  in `DataPreparationActor`, token-balanced scorer chunking.
- `open_instruct/rl_utils.py` — teacher fields through `pack_sequences`.
- `open_instruct/data_types.py` — optional teacher fields on
  `CollatedBatchData`.
- `open_instruct/grpo_utils.py` — shared forward returning sampled-token
  logprobs + optional student top-k logprobs; OPD loss-stat keys.
- `open_instruct/olmo_core_train_modules.py` — OPD loss composition and
  metrics in `GRPOTrainModule.train_batch`.
- `open_instruct/grpo.py` — teacher scorer setup and startup validation.

## Metrics

Recorded only when OPD is enabled (non-OPD runs log no `opd/*` keys):

```text
loss/opd_avg
opd/teacher_topk_mass          # low → opd_topk too small for a faithful KL
opd/sampled_token_in_topk      # low → teacher top-k misses student samples
time/opd_teacher_scoring
time/opd_teacher_scoring_worker_sum
opd/teacher_tokens_per_second
```

GRPO metrics are unchanged. In pure OPD, reward metrics may still be logged if
rewards are computed, but they never contribute to the loss.

## Testing

- Unit: `distillkit/test_losses.py` (KL math, `-inf`/NaN edge cases),
  `distillkit/test_vllm_logprobs.py` (extraction alignment), `test_rl_utils.py`
  (packing), `test_utils.py` (SP splitting of `[B, T, K]`),
  `test_opd_validation.py`, `test_data_loader.py` / `test_grpo_fast.py`
  (config validation, scorer chunking), `test_olmo_core_train_modules.py`
  (end-to-end learner loss with real backprop, temperature invariance).
- GPU: `test_opd_gpu.py` (live teacher scorer) runs in
  `scripts/test/run_gpu_pytest.sh`.
- Example launch: `scripts/train/debug/opd.sh` (GRPO + OPD on 2 GPUs; the
  header comments show the pure-OPD and larger-teacher variants).

## Limitations and Future Work

- **No efficacy claim yet**: Beaker rehearsals validate the system path, not
  OPD superiority. Run a matched GRPO-only control before making quality
  claims.
- **OPSD (self-distillation)**: reuse the same scorer/loss with a
  demo-conditioned teacher prompt (`teacher_prompt = problem prompt +
  privileged context`, same student completion) and a `self_sync` teacher that
  receives learner weight updates. Requires a pre-tokenized
  `teacher_input_ids_prompt` dataset field; intentionally out of v1.
- **SFT/DPO distillation**: `olmo_core_finetune.py` and `dpo.py` are offline
  paths today; the distillkit loss can back future offline or auxiliary
  distillation there, but OPD itself requires online rollouts.
- **Diagnostics**: `opd/student_topk_mass` and teacher top-k entropy would be
  useful additions.
