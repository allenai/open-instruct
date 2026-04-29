# IcePop: Observed Training Collapse

## TL;DR

The first end-to-end run of IcePop on Qwen3-4B-Base + DAPO-Math collapsed around
step 530 and the inference engines wedged by step 765. A side-by-side run on the
exact same script with TIS clamping instead of IcePop completed all 1000 steps
cleanly. We do not have a definitive root cause, but the evidence points to a
feedback loop between IcePop's hard zeroing of out-of-range tokens and the
remaining gradient signal becoming biased.

This document records what happened so the next person who reaches for IcePop
knows the failure mode exists.

## What IcePop does

IcePop applies the operator `pop(ρ, 1/β, β)` to each token, where
`ρ = exp(old_logprob_train - vllm_logprobs)` is the train/infer mismatch ratio.
Tokens with `ρ ∈ [1/β, β]` keep their per-token loss; tokens outside that band
have their contribution zeroed.

This is implemented in `open_instruct/grpo_utils.py::compute_icepop_mask` and
re-uses the existing `truncated_importance_sampling_ratio_cap` field as β. The
PPO clip on the *training* ratio `r = π_θ / π_train_old` is preserved.

For comparison, TIS (truncated importance sampling) computes the same `ρ` but
*clamps* it to `[1/β, β]` rather than zeroing — out-of-range tokens still
contribute to the gradient with bounded magnitude.

## What we ran

Script: `scripts/train/qwen/qwen3_4b_dapo_math.sh` with `--use_icepop` and
`--truncated_importance_sampling_ratio_cap 2.0` (β = 2.0).

Key hyperparameters:
- Qwen3-4B-Base, DAPO-Math-17k, 1× 8×H100 node
- `learning_rate 1e-6`, constant schedule
- `async_steps 4`, `inflight_updates`, `num_samples_per_prompt_rollout 16`
- `response_length 8192`, `temperature 1.0`

Beaker experiment: `01KQAR63YRM1KDA1W7WNB2Z34V`.

Control (TIS, same script before IcePop was wired in):
`01KPTSPMH4Q2SBQMKGZ6YA145Q`. 1000/1000 steps, exit 0, 14h19m.

## Timeline

| Time (UTC) | Step | `icepop_drop_frac` | `vllm_vs_local_logprob_diff_mean` | `verifiable_correct_rate` | `stop_rate` |
|------------|------|---------------------|-----------------------------------|----------------------------|-------------|
| 19:18      | 1    | 0.0                 | ~0                                | n/a                        | n/a         |
| 20:12      | 117  | 4.23e-06            | ~0.05                             | 0.39                       | n/a         |
| 02:55+1d   | ~530 | 0.17 (first spike)  | ~0.5                              | dropping                   | dropping    |
| 03:55      | 649  | **0.61**            | **8.83**                          | **0.07**                   | **0.49**    |
| 10:13      | 764  | (last train step)   |                                   |                            |             |
| 12:19      | 765  | (training stuck)    |                                   |                            |             |
| 14:01      | 765  | killed manually     |                                   |                            |             |

Reward, stop rate, and correctness all degraded together as `drop_frac` climbed.
By step 649, IcePop was zeroing 61 % of tokens, the model was no longer emitting
EOS on roughly half of generations, and `verifiable_correct_rate` had fallen
from 0.39 (step 117) to 0.07.

## What surprised us

`vllm_vs_local_logprob_diff_mean` went from ~0.1 to **8.83 nats** in the span of
a few logged steps. A mean log-prob disagreement of 8.83 means the average ρ is
exp(8.83) ≈ 7000 — vLLM and the training-side forward pass are computing
wildly different probabilities for the same tokens under what should be the
same weights.

Our best guess at the mechanism: the model started emitting low-entropy /
repetitive output (consistent with `stop_rate` halving, sequence lengths
clustering at the 8192 cap). On low-entropy distributions over long sequences:
- vLLM uses FlashAttention with chunked prefill; the training-side forward pass
  uses a different attention impl
- bf16 reductions accumulate differently across 5000+ tokens
- When the softmax is very peaked, sub-bit numerical noise can flip the argmax

So the diff explosion is a *symptom* of model collapse, not its cause.

## Why we suspect IcePop, not just bad luck

The TIS control (`01KPTSPMH4Q2SBQMKGZ6YA145Q`) finished 1000/1000 steps with the
same script, same model, same dataset, same hyperparameters — only difference
is TIS clamping vs IcePop zeroing. Same seed was not verified, but the run was
on the same branch and same configuration.

That makes the most plausible chain:

1. Step ~530: real train/infer divergence creeps above β = 2.0 on a small
   fraction of tokens (this is normal, TIS handles it by clamping).
2. IcePop zeros those tokens. The remaining in-range gradient is computed only
   from the "easy" tokens where train and infer already agree.
3. Training on this biased subset pushes the policy in a direction that
   increases mismatch on harder tokens (since they were never represented in
   the gradient).
4. More tokens fall out of range → `drop_frac` rises → bias amplifies.
5. Policy collapses to low-entropy / repetitive output.
6. vLLM/training-side numerical disagreement explodes on degenerate sequences.
7. Eventually the inference engines wedge (likely an `inflight_updates` race
   condition exposed by the broken model state).

We did not run a controlled A/B with the same seed, so this is a hypothesis
supported by one IcePop run vs one TIS run, not proof.

## What might fix it

In rough order of cheapness:

- **Larger β** (5 or 10) so the band is wide enough that out-of-range tokens
  are genuinely rare and zeroing them cannot create a biased subset.
- **Warm-start with TIS**, switch to IcePop after the policy stabilizes (e.g.,
  after the first eval at step 100).
- **Hybrid**: clamp like TIS for moderate `ρ`, zero only for extreme `ρ` (say
  outside `[1/10, 10]`).
- **Disable `inflight_updates`** for IcePop runs — even if it's not the root
  cause, it almost certainly amplified the divergence by letting vLLM run on
  weights up to `async_steps` behind.

The cheapest next experiment is just bumping β to 5 or 10 and rerunning.

## Files involved

- `open_instruct/grpo_utils.py` — `compute_icepop_mask`, the `use_icepop`
  config flag, and the loss-side mask application.
- `open_instruct/grpo_fast.py` — call site that selects between
  `compute_tis_weights` and `compute_icepop_mask`, plus the `icepop_drop_frac`
  metric.
- `open_instruct/olmo_core_train_modules.py` — same wiring on the olmo-core
  trainer path.
- `scripts/train/qwen/qwen3_4b_dapo_math.sh` — run script that hit the issue.
