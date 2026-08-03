# On-Policy Distillation (OPD)

On-policy distillation trains a student model on its **own rollouts**, using a frozen
teacher's per-token log-probabilities as the learning signal instead of (or in addition
to) an environment reward. The student samples trajectories as usual; the teacher scores
every sampled token; tokens the teacher likes more than the student get pushed up, and
tokens the student is overconfident about get pushed down. This follows the recipe from
[Thinking Machines' On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/),
matching the implementations in [slime](https://github.com/THUDM/slime)
(`apply_opd_kl_to_advantages`) and the tinker cookbook.

Typical uses:

- **Distill an RL-trained teacher into a fresh student** (e.g. a DPPO-trained terminal
  agent back into its base model, or a large teacher into a smaller student) at a
  fraction of RL's cost — every rollout produces a dense, per-token signal, so no
  group-variance filtering or reward sparsity issues.
- **Combine with RL** by adding the distillation term on top of the reward advantage
  (leave `opd_pure` off).

## How it works in `grpo_fast.py`

1. The teacher is loaded **learner-side** with the same machinery as the reference
   policy (`load_ref_policy` + a DeepSpeed eval config: ZeRO-3-sharded when the policy
   trains with stage 3, unsharded otherwise). It is never updated, never checkpointed,
   and never served by vLLM — so RL checkpoints saved as `Qwen3_5ForCausalLM` work
   directly as teachers without CG conversion.
2. At the start of each training step, the teacher runs the same no-grad forward pass
   the reference policy uses (`compute_logprobs`) over the packed rollout tokens,
   sharing the sequence-parallel / FLA CP context plumbing.
3. The sampled per-token reverse KL is folded into the advantages before the minibatch
   loop (`compute_opd_advantages` in `grpo_utils.py`):

   ```
   reverse_kl_t = log π_vllm(y_t) − log π_teacher(y_t)
   A_t ← A_t − opd_kl_coef · reverse_kl_t          # additive (default)
   A_t = −opd_kl_coef · reverse_kl_t               # with --opd_pure
   ```

   The student side of the KL is the **behavior policy** (vLLM rollout logprobs), so the
   term is constant w.r.t. the trainer policy; gradients flow only through the usual
   surrogate loss. Because OPD only rewrites advantages, it composes unchanged with
   every `loss_fn` (DAPO/CISPO/DPPO/TVPO), the TIS/trust-region masks, and both the
   eager and tiled (liger) loss paths.

## Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--opd_teacher_model_name_or_path` | `None` | Enables OPD: HF name or local/weka path of the frozen teacher. Must share the student's tokenizer. |
| `--opd_teacher_model_revision` | `None` | Teacher revision. |
| `--opd_kl_coef` | `1.0` | Coefficient on the reverse-KL advantage term. |
| `--opd_pure` | `False` | Replace the environment advantage entirely instead of adding to it. Rewards are still computed and logged. |

Incompatible with `--use_value_model` (both rewrite advantages).

For **pure** OPD also pass `--filter_zero_std_samples false` and drop
`--active_sampling`: reward-variance filtering exists to discard groups that are useless
for GRPO baselines, but pure distillation learns from every rollout. A warning is logged
otherwise.

## Metrics

- `objective/opd_reverse_kl` — masked mean of the per-token reverse KL. **The main
  convergence signal: it should decrease** as the student approaches the teacher on the
  student's own distribution.
- `objective/opd_teacher_logprob` — masked mean teacher logprob of the sampled tokens
  (rises as rollouts move into regions the teacher likes).

## Example scripts

- `scripts/train/debug/grpo_fast_opd.sh` — local 2-GPU smoke test (Qwen3-0.6B student,
  Qwen3-1.7B teacher, gsm8k, full DPPO+liger production loss path).
- `scripts/general_agent/terminal/rl/qwen35_9b_dppo_opd_4node_64k.sh` — Terminal RL:
  pure-distill the DPPO-trained tmax-15k teacher (step_120) back into base Qwen3.5-9B on
  the same sandboxed terminal tasks.

## Caveats

- The teacher forward adds one no-grad pass per step over the packed batch (same cost
  as enabling the reference policy) plus the teacher's weight memory on the learners
  (sharded under ZeRO-3).
- Teacher and student must share a tokenizer. Under sequence parallelism with Qwen3.5
  hybrid-attention models, the teacher must also share the linear-attention conv-kernel
  configuration (true for same-family checkpoints), since the FLA CP contexts are built
  from the policy's config.
- Tool/observation tokens are excluded automatically: the reverse KL is masked with the
  same `response_mask` the loss uses, while the teacher still conditions on the full
  trajectory context.
