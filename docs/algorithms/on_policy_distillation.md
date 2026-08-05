# On-Policy Distillation (OPD / multi-teacher MOPD)

On-policy distillation trains a student model on its **own rollouts**, using a frozen
teacher's per-token log-probabilities as the learning signal instead of (or in addition
to) an environment reward. The student samples trajectories as usual; the teacher scores
every sampled token; tokens the teacher likes more than the student get pushed up, and
tokens the student is overconfident about get pushed down. This follows the recipe from
[Thinking Machines' On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/),
matching the implementations in [slime](https://github.com/THUDM/slime)
(`apply_opd_kl_to_advantages`) and the tinker cookbook.

**Multiple teachers** are supported: pass several models to
`--opd_teacher_model_name_or_path` and pick a combination strategy with
`--opd_teacher_combine` (see [Multi-teacher distillation](#multi-teacher-distillation-mopd)
below). The `route` strategy reproduces
[MOPD](https://arxiv.org/abs/2606.30406) — merging several per-domain RL
specialists into one student on the student's own rollouts.

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

## Multi-teacher distillation (MOPD)

With more than one teacher, every teacher scores every rollout token and the per-teacher
logprobs are combined into a single `log π_teacher(y_t)` before the KL fold
(`combine_opd_teacher_logprobs` in `grpo_utils.py`). Strategies
(`--opd_teacher_combine`, default `mixture`):

| Strategy | Combined target | Semantics |
| --- | --- | --- |
| `mixture` | `logsumexp_k(log w_k + log π_k)` | Distill toward the (weighted) probability **mixture** of the teachers. Weights via `--opd_teacher_weights` (default uniform). |
| `max` | `max_k log π_k` | Optimistic **union of experts**: each token is judged by whichever teacher likes it most; the student is only penalized where it out-confidences *every* teacher. |
| `min` | `min_k log π_k` | Pessimistic **consensus**: a token must be probable under *all* teachers to escape penalty. |
| `route` | `log π_{k(sample)}` | **MOPD-style deterministic routing**: each rollout is scored only by the teacher that owns its dataset/domain (`--opd_teacher_domains`). |

`route` is the strategy validated by the [MOPD paper](https://arxiv.org/abs/2606.30406)
(per-domain RL specialists → one student, on Qwen3-30B-A3B it inherited nearly all of
each teacher's capability and beat Mix-RL / Cascade-RL / Param-Merge baselines). Domain
claims are matched against each sample's RLVR `dataset` field (the verifier name(s)),
case-insensitively; `*` marks a catch-all teacher:

```bash
--opd_teacher_model_name_or_path path/to/math_teacher path/to/code_teacher \
--opd_teacher_combine route \
--opd_teacher_domains "gsm8k,math" "*"
```

Routing rides on plumbing that ships each step's `{rollout_sample_id: dataset}` map from
the data-prep actor to the learners, so it needs no changes to packing/collation and
works under sequence parallelism. Note that even under `route`, **all** teachers forward
**all** tokens (only the routed teacher's logprobs are kept): skipping non-routed
forwards would desynchronize ZeRO-3 allgathers across ranks whose microbatches route
differently. Budget one no-grad forward per teacher per step.

The MOPD paper's findings transfer here: teachers work best when they are RL fine-tunes
**of the student's own starting checkpoint** (same-origin); a foreign/much larger
teacher can destabilize training (see the cross-policy dip below, and the paper's
Qwen3-235B failure). `--opd_adv_clip 2.0` (their `A_max`) bounds the per-token pull and
is their default stabilizer.

## Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--opd_teacher_model_name_or_path` | `None` | Enables OPD: HF name(s) or local/weka path(s) of the frozen teacher(s), space-separated. All must share the student's tokenizer. |
| `--opd_teacher_model_revision` | `None` | Teacher revision(s); omit entirely or give one per teacher. |
| `--opd_teacher_combine` | `mixture` | Multi-teacher combination: `mixture` \| `max` \| `min` \| `route`. Irrelevant with one teacher. |
| `--opd_teacher_weights` | `None` | Per-teacher mixture weights (`mixture` only), normalized internally. |
| `--opd_teacher_domains` | `None` | Per-teacher comma-separated dataset claims (`route` only); `*` = catch-all. |
| `--opd_adv_clip` | `None` | Clamp the folded per-token KL term to `±clip` (MOPD's `A_max`; logged KL stays unclipped). |
| `--opd_kl_coef` | `1.0` | Coefficient on the reverse-KL advantage term. |
| `--opd_pure` | `False` | Replace the environment advantage entirely instead of adding to it. Rewards are still computed and logged. |

Incompatible with `--use_value_model` (both rewrite advantages).

For **pure** OPD also pass `--filter_zero_std_samples false` and drop
`--active_sampling`: reward-variance filtering exists to discard groups that are useless
for GRPO baselines, but pure distillation learns from every rollout. A warning is logged
otherwise.

## Metrics

- `objective/opd_reverse_kl` — masked mean of the per-token reverse KL against the
  **combined** teacher signal. **The main convergence signal: it should decrease** as
  the student approaches the teacher on the student's own distribution.
- `objective/opd_teacher_logprob` — masked mean combined teacher logprob of the sampled
  tokens (rises as rollouts move into regions the teacher likes).
- `objective/opd_reverse_kl_teacher_{k}` (multi-teacher only) — per-teacher reverse KL
  over all response tokens; shows which teacher the student is tracking.
- `objective/opd_route_frac_teacher_{k}` (`route` only) — fraction of response tokens
  routed to teacher k; sanity-checks the domain claims against the data mix.

## Example scripts

- `scripts/train/debug/grpo_fast_opd.sh` — local 2-GPU smoke test (Qwen3-0.6B student,
  Qwen3-1.7B teacher, gsm8k, full DPPO+liger production loss path).
- `scripts/train/debug/grpo_fast_mopd.sh` — multi-teacher twin (two teachers, defaults
  to `mixture`; append flags to exercise `max`/`min`/`route`/weights/clip).
- `scripts/train/debug/grpo_fast_mopd_sp2.sh` — 4-GPU SP=2 multi-teacher test with
  heterogeneous teacher geometries (one full-sequence-scored, one shard-scored).
- `scripts/train/debug/grpo_fast_mopd_beaker.sh` — Beaker smoke: two teachers routed
  over a real two-domain mix (gsm8k + math); `OPD_COMBINE=mixture` for the mixture path.
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
