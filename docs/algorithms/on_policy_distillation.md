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

## Empirical results (2026-08, terminal RL / tmax-15k)

Two 4-node experiments, both pure OPD (`opd_kl_coef 1.0`), evaluated on full
TerminalBench-2.1 (89 tasks) and TBlite (100 tasks) at 64k, pass@1:

**Same-lineage distillation works dramatically well.** Distilling
`allenai/tmax-9b` (an RL fine-tune of the student's own base) into base
`Qwen3.5-9B`: `opd_reverse_kl` fell 0.26 → 0.01 in ~40 steps, and the student
reached **teacher parity on both benchmarks in ~40–60 steps (~7–10 h)** —
TB2.1 0.281–0.292 vs teacher 0.276; TBlite 0.540–0.580 vs teacher 0.534 —
capability the equivalent DPPO RL run needed 300+ steps (days) to build. After
KL convergence the curve plateaus at teacher level (steps 60–120 oscillate
within noise): pure OPD matches but does not exceed the teacher. The runs are
rollout-bound: the teacher forward is a small share of step time.

**Cross-policy distillation (independently-trained teacher) shows a transition
dip.** Distilling `allenai/tmax-27b` into the already-RL-trained
`allenai/tmax-4b`: the KL stalls (~0.36 → 0.18) instead of converging, and the
student goes through a disrupted phase (~steps 20–60: truncation spikes to
~33%, benchmark scores drop well below its starting level) before recovering
(TBlite 0.448 → 0.310 → ~0.40 by step 60–80; train metrics healed by step
~110+ with `opd_teacher_logprob` still improving). For this experiment shape,
prefer a smaller `opd_kl_coef` (0.25–0.5) and/or keep the environment reward
(drop `--opd_pure`) as ballast; expect a mid-run behavioral trough either way.

## Evaluating OPD checkpoints

- Checkpoints save as raw `Qwen3_5ForCausalLM` — CG-convert before vLLM-serving
  (`convert_qwen35_causallm_to_cg.py`, donor = the student's base model).
- Beaker retries move checkpoints to a new
  `<exp>__<seed>__<NEW_TIMESTAMP>_checkpoints/` dir while the step counter
  continues — glob for `step_N` across instances rather than hardcoding one.
- **Always check `result.json` → `stats.n_errored_trials`** before trusting an
  eval score. Model-behavior timeouts run 5–25% of trials on these benchmarks;
  a dead docker mirror instead shows up as **mass `docker-compose` startup
  failures (50–90% errored) with silently depressed scores**.

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
