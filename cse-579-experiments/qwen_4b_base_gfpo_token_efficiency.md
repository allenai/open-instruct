# Qwen3-4B-Base RL-Zero · GFPO Token-Efficiency (8/16)

## Status

- **State**: running (launched 2026-05-29)
- **Eval state**: submitted 2026-06-01 — oe-eval jobs for steps 100–300 via `cse-579-scripts/submit_all_current_evals.sh` (priority normal, ai2/saturn+ceres). Fetch with `cse-579-experiments/fetch_eval_results.sh` once the jobs finish.
- **Last updated**: 2026-05-29

## Purpose

Our reproduction of **GFPO** (Shrivastava et al., 2025,
[arXiv:2508.09726](https://arxiv.org/abs/2508.09726)) using the paper's second
filter metric: **token efficiency = reward / response length**. GFPO oversamples
G responses per prompt and trains only on the top-k by this metric — keeping the
responses that deliver the most reward per token. Unlike the Shortest variant,
this metric *does* read the reward (to form reward/length), so it favors short
**and correct** responses; the paper reports it gives the largest length-inflation
reduction (up to ~85%).

Sibling of [`qwen_4b_base_gfpo_shortest.md`](qwen_4b_base_gfpo_shortest.md);
together they cover both GFPO filter metrics as the published comparison method
for our length-aware reward shaping.

## Beaker

### Attempts

| # | URL | Launched (UTC) | Terminated (UTC) | Exit | Notes |
|---|-----|----------------|------------------|------|-------|
| 1 | [01KSTV2X…](https://beaker.org/ex/01KSTV2XV9EA2CHX44SJPKK214) | 2026-05-29 21:42 | 2026-05-29 22:02 | 1 | Died at step 2: `Weight sync timed out after 120.0s` (same G=16 drain issue as the Shortest run). Fixed with `--inflight_updates True`. |
| 2 | [01KSTY5M…](https://beaker.org/ex/01KSTY5MGE6X1P5DKNTG8RM69T) | 2026-05-29 22:37 | 2026-05-29 22:44 | cancelled | `--inflight_updates True` fix; switched approach to bumping `WEIGHT_SYNC_TIMEOUT_S` 120→600s instead (keeps inflight_updates=False for a fair comparison — see sibling doc). |
| 3 | [01KSTYJF…](https://beaker.org/ex/01KSTYJFPHY1VYPXAJBW2RYM91) | 2026-05-29 22:44 | 2026-05-30 ~21:50 | deleted | Healthy (reached ~step 23 — validated the 600s timeout fix at G=16), then deleted by a colleague by accident. |
| 4 | [01KSXDZD…](https://beaker.org/ex/01KSXDZD4S6N64J6YV1A1D6NG5) | 2026-05-30 21:51 | 2026-05-30 22:34 | 1 | Transient vLLM weight-sync hang at step 1 (engines stuck, hit the full 600s timeout). Not config/code — the identical-config Shortest twin was healthy at step 473 and both warmup runs completed 1000 steps. Bad node. |
| 5 | [01KSZKGM…](https://beaker.org/ex/01KSZKGM6S41D5TAVD95A0BXDX) | 2026-05-31 | 2026-05-31 18:50 | 1 | Died step 3, weight-sync timeout (600s). Diagnosed as a nondeterministic early-step generation-drain lottery, NOT degeneration: early metrics (seq len, scores, ±8.75 advantages) identical to the healthy Shortest run, and token-eff reached step 169/23 on other attempts. |
| 6 | [01KT07XH…](https://beaker.org/ex/01KT07XH59FKN4XRPEG4DKZ71Q) | 2026-05-31 | — | — | running; inflight_updates=False, timeout=1200s, --max_retries 3 (re-roll the opening-window lottery) |

- **Workspace**: ai2/olmo-instruct
- **Cluster**: ai2/jupiter
- **Resources**: 1 node × 8 GPUs

## Configuration

- **Launch script**: `cse-579-scripts/gfpo_rl_qwen.sh` (`GFPO_METRIC=token_efficiency`)
- **Branch / commit**: `jacobm/cse-579` @ `33ffe714`
- **Base model**: `Qwen/Qwen3-4B-Base` (RL directly on base; no SFT, no DPO)
- **Dataset**: `jacobmorrison/cse-579-mixed-rl` (verifiable-rewards-only mix)
- **GFPO**:
  - filter_metric=token_efficiency (reward/length), retain_k=8
  - group size G = num_samples_per_prompt_rollout = 16 (oversampled)
  - reward untouched; subset advantage normalization
  - non-retained responses kept in-batch with zero advantage (literal Eq. 2)
- **Step matching**: total_episodes scaled to 1,024,000 (= 64 × 16 × 1000) so the
  optimizer-step count matches the baseline's 1000. ~2× inference AND ~2× train
  compute vs our 8-sample runs.
- **Other hyperparams**: lr=1e-6, response_length=30720, pack_length=32768,
  deepspeed_stage=3, num_learners_per_node=4, vllm_num_engines=4
- **Image**: `nathanl/open_instruct_auto`, branch overlaid via in-container git-clone

## Outputs

- **exp_name**: `gfpo_qwen_4b_base_mixed_token_efficiency_g16k8`
- **Checkpoints**: `/weka/oe-adapt-default/allennlp/deletable_checkpoint/ianm/gfpo_qwen_4b_base_mixed_token_efficiency_g16k8__<seed>__<ts>_checkpoints/`
  (steps 100 → 1000 every 100 steps; exact suffix assigned at runtime)
- **W&B run**: _(fill in once the run registers)_
- **Eval results path**: `cse-579-experiments/results/gfpo_qwen_4b_base_mixed_token_efficiency_g16k8/`

## Pair / baseline

- **Compare to**: [`qwen_4b_base_baseline.md`](qwen_4b_base_baseline.md), the
  shaping runs, and the sibling [`qwen_4b_base_gfpo_shortest.md`](qwen_4b_base_gfpo_shortest.md).

## Notes

Because the metric reads the reward, an all-wrong group degenerates to "shortest
among zero-reward responses" (all reward/length ≈ 0, tie-broken by stable sort)
— acceptable, those groups carry little signal anyway. Watch
`val/gfpo_kept_mean_length` vs the Shortest run: token-efficiency should keep
*correct* responses preferentially, so its kept set may be slightly longer but
higher-reward.

## Known issues

None yet. Same step-time caveat as the Shortest run (all 16 responses trained on).
