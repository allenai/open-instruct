# Qwen3-4B-Base RL-Zero · GFPO Shortest (8/16)

## Status

- **State**: running (launched 2026-05-29)
- **Eval state**: submitted 2026-06-01 — oe-eval jobs for steps 100–600 via `cse-579-scripts/submit_all_current_evals.sh` (priority normal, ai2/saturn+ceres). Fetch with `cse-579-experiments/fetch_eval_results.sh` once the jobs finish.
- **Last updated**: 2026-05-29

## Purpose

Our reproduction of **GFPO** (Group Filtered Policy Optimization; Shrivastava et
al., 2025, [arXiv:2508.09726](https://arxiv.org/abs/2508.09726)) as the published
**comparison method** for our length-aware reward shaping. No official GFPO code
was released, so this is our own implementation (`open_instruct/gfpo.py`).

GFPO oversamples a larger group of G responses per prompt and trains only on the
top-k ranked by a filter metric — here **shortest response length**. Advantages
are normalized over the retained subset only; the dropped G−k responses get zero
advantage (kept in-batch per the objective). Crucially, the reward is untouched
and the filter is **length-only** — correctness stays entirely in the reward and
is not conditioned on in the filter.

The contrast the project is built on: does filtering-on-length (GFPO) preserve
reasoning where our reward-decaying-on-length collapsed it
([`qwen_4b_base_linear_alpha1.md`](qwen_4b_base_linear_alpha1.md) went to 7-token
outputs, lost all of AIME pass@32)? The paper claims GFPO avoids this by keeping
accuracy in the reward.

## Beaker

### Attempts

| # | URL | Launched (UTC) | Terminated (UTC) | Exit | Notes |
|---|-----|----------------|------------------|------|-------|
| 1 | [01KSTV2P…](https://beaker.org/ex/01KSTV2PE1M5E2X8A4XY5FYY7H) | 2026-05-29 21:42 | 2026-05-29 22:26 | 1 | Died at step 3: `Weight sync timed out after 120.0s`. G=16 generation drain exceeds the timeout under default drain-then-sync. Fixed with `--inflight_updates True`. |
| 2 | [01KSTY5H…](https://beaker.org/ex/01KSTY5HQE69Z01FVTJSC8K2MZ) | 2026-05-29 22:37 | 2026-05-29 22:44 | cancelled | `--inflight_updates True` fix worked, but switched approach: bumped `WEIGHT_SYNC_TIMEOUT_S` 120→600s instead, to keep inflight_updates=False (identical weight-sync semantics to the baseline — fair comparison). |
| 3 | [01KSTYJC…](https://beaker.org/ex/01KSTYJCZN8P74JKY29VK0WE04) | 2026-05-29 22:44 | 2026-05-29 23:24 | 1 | Died at step 3 — but a *genuine* vLLM hang (engines silent ~31 min, hit the full 600s timeout), not the slow-drain overrun the bump fixed. The sibling token-efficiency run reached step 23 on identical config, so this was transient/infra (bad node), not a `shortest`-metric issue. |
| 4 | [01KSXDZA…](https://beaker.org/ex/01KSXDZAHMG54BQYZ6ZM96929Y) | 2026-05-30 21:51 | — | — | running (relaunch after a colleague deleted the in-flight jobs); inflight_updates=False, timeout=600s |

- **Workspace**: ai2/olmo-instruct
- **Cluster**: ai2/jupiter
- **Resources**: 1 node × 8 GPUs

## Configuration

- **Launch script**: `cse-579-scripts/gfpo_rl_qwen.sh` (`GFPO_METRIC=shortest`)
- **Branch / commit**: `jacobm/cse-579` @ `33ffe714`
- **Base model**: `Qwen/Qwen3-4B-Base` (RL directly on base; no SFT, no DPO)
- **Dataset**: `jacobmorrison/cse-579-mixed-rl` (verifiable-rewards-only mix)
- **GFPO**:
  - filter_metric=shortest, retain_k=8
  - group size G = num_samples_per_prompt_rollout = 16 (oversampled)
  - reward untouched; length-only filter; subset advantage normalization
  - non-retained responses kept in-batch with zero advantage (literal Eq. 2)
- **Step matching**: total_episodes scaled to 1,024,000 (= 64 prompts × G=16 ×
  1000 steps) so the optimizer-step count and per-step prompt count match the
  baseline's 1000 steps. Costs ~2× inference AND ~2× train compute vs our 8-sample
  runs (the kept-with-zero-advantage choice keeps all 16 in the train batch).
- **Other hyperparams**: lr=1e-6, response_length=30720, pack_length=32768,
  deepspeed_stage=3, num_learners_per_node=4, vllm_num_engines=4
- **Image**: `nathanl/open_instruct_auto`, branch overlaid via in-container git-clone

## Outputs

- **exp_name**: `gfpo_qwen_4b_base_mixed_shortest_g16k8`
- **Checkpoints**: `/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/gfpo_qwen_4b_base_mixed_shortest_g16k8__<seed>__<ts>_checkpoints/`
  (steps 100 → 1000 every 100 steps; exact suffix assigned at runtime)
- **W&B run**: _(fill in once the run registers)_
- **Eval results path**: `cse-579-experiments/results/gfpo_qwen_4b_base_mixed_shortest_g16k8/`

## Pair / baseline

- **Compare to**: [`qwen_4b_base_baseline.md`](qwen_4b_base_baseline.md) (no
  intervention) and our shaping runs ([`qwen_4b_base_linear_alpha1.md`](qwen_4b_base_linear_alpha1.md)
  and the warmup variants). Sibling: [`qwen_4b_base_gfpo_token_efficiency.md`](qwen_4b_base_gfpo_token_efficiency.md).

## Notes

Watch `val/gfpo_kept_mean_length` vs `val/gfpo_dropped_mean_length` — the gap is
the filter's selection pressure. If accuracy holds while lengths shrink, GFPO
beats our shaping on the same setup. `val/gfpo_frac_kept` should sit at 8/16=0.5.

## Known issues

None yet. The kept-with-zero-advantage choice means step time will be notably
longer than the shaping runs (training forward/backward on all 16 responses).
