# hamish/vip branch — summary + GPU testing plan

> Handoff doc for a GPU-capable agent. The branch `hamish/vip` (off `main`) adds a scalar/LM/SAE
> value model, several conditioning variants, a frozen-policy pretraining window, an offline
> value-estimation harness, and a scaffold for a sibling generative-value training script.
>
> All 15 plan todos are implemented and 17 unit tests pass on CPU. Nothing has been committed.

## Repository state

```
Branch:         hamish/vip
Base:           main (d4bdc57d9)
Files changed:  5 modified + 4 new Python modules + 11 train scripts + 8 eval scripts + 1 test file
Commits:        0 (all edits are uncommitted; push/commit are up to the user)
Unit tests:     17 passing in open_instruct/test_value_model.py on CPU
                (16 existing tests in test_rl_utils.py still pass)
```

## What was built

### Core training pipeline (modifies existing code)

| File                              | Change                                                                                                                                                                                                     |
| :-------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `open_instruct/grpo_utils.py`     | +24 flags on `GRPOExperimentConfig`: `use_value_model`, value-init variants, `vf_clip_range`, `gamma`, `gae_lambda`, decoupled/length-adaptive GAE, warmup windows, `use_sae`, `use_lm_value_model`, GT+rollout-context conditioning, with full `__post_init__` validation. |
| `open_instruct/data_types.py`     | `CollatedBatchData` gains `rewards`, `dones`, `ground_truths`, `sibling_rollouts`, `segment_boundaries`. `.to()` now skips non-tensor fields.                                                              |
| `open_instruct/rl_utils.py`       | Adds `calculate_length_adaptive_lambda` + three new GAE variants (`_vapo`, `_sae`, `_sae_vapo`); `PackedSequences` gets matching new fields.                                                               |
| `open_instruct/data_loader.py`    | `DataPreparationActor` now populates per-token rewards (EOS-sparse), SAE boundaries, per-pack ground truths, sibling rollouts. `StreamingDataLoaderConfig` gains `init=False` mirror fields auto-synced by `setup_runtime_variables`. |
| `open_instruct/grpo_fast.py`      | `_init_value_model`, `forward_value`, `_forward_value_with_conditioning`, `_dispatch_gae`; value forward + GAE override of advantages in `step()`; value backward with MSE (PPO2-clipped) or LM-yesno BCE; warmup gating on policy backward + weight sync; value-model save/load + `training_args.json` dump. |

### New modules

- `open_instruct/value_model_utils.py` — conditioning text builders (all 9 templates), generative-value prompt + score parser, LM-yesno BCE, PPO2-clipped value MSE.
- `open_instruct/grpo_fast_genvalue.py` — sibling script with `GenValueExperimentConfig`, segmentation helper, vLLM scoring helper. `main()` is a **scaffold** with a clear `TODO`: the second vLLM pool + parallel weight-sync group still needs wiring to `grpo_fast_resource_plan.py` and `weight_sync_thread`.
- `open_instruct/value_estimation.py` — `make_dataset`, `score_dataset`, `compare_runs` with CLI and subcommands.

### Scripts

- `scripts/train/vip/` — 11 recipes covering plain PPO, RM-init, GT/expected_accuracy/rollout_context/correct_demo, SAE vpretrain + GT, LM-yesno vpretrain, PPO-from-vpretrain-ckpt, generative-value.
- `scripts/eval/value_estimation/` — 8 wrappers for dataset creation, scoring each value-model type + conditioning, and `compare.sh`.

### Tests

- `open_instruct/test_value_model.py` — 17 passing CPU-only tests covering GAE variants, sibling assembly, SAE boundary marking, every conditioning template, value loss (clipped MSE + LM-yesno BCE), gen-value prompt parsing, segmentation helpers.

## Known scaffolds (intentional incomplete pieces)

1. **`grpo_fast_genvalue.main()`** — raises `NotImplementedError`. The segmentation + scoring + score-parsing helpers are complete and unit-tested; what's missing is the Ray placement-group reservation for a second vLLM pool and the parallel `NCCLWeightTransferEngine` group. Attempting to run the gen-value recipe will fail at `main()`.
2. **Value model sharing with SP** — `_init_value_model` passes `self.mpu` to `deepspeed.initialize` for sequence parallelism, but I have not exercised that path. SP>1 runs should start with the existing SP=1 recipes until confirmed.
3. **`_forward_value_with_conditioning` is per-sub-seq** — it unpacks packed batches and runs one forward per sub-sequence. Correct but slow. Batching across sub-sequences of the same length is a perf follow-up once correctness is verified.
4. **Score dataset for scalar value model** loads via plain HF `from_pretrained`. If you saved the value head as a `Linear(h, 1)`, the lm_head replacement must be redone at load time before `load_state_dict`. The current code just trusts the checkpoint has the right shape. For the first GPU test, use `--value_model_path` pointing at a checkpoint dir that contains `value_model.bin` exactly as saved by `grpo_fast.save_checkpoint_state`.

---

## GPU testing plan

Order runs from cheapest to most expensive so failures surface fast. All recipes default to `Qwen/Qwen3-4B-Base` + `hamishivi/DAPO-Math-17k-Processed_filtered` — adjust if you want to use a smaller model.

### Phase 0 — CPU + import checks (no GPU needed, < 2 min)

Run these first to confirm nothing obvious regressed:

```bash
uv run python -m pytest open_instruct/test_value_model.py open_instruct/test_rl_utils.py -v
uv run python -c "from open_instruct.grpo_utils import GRPOExperimentConfig; GRPOExperimentConfig(use_value_model=True, gae_lambda=0.95)"
uv run python -c "from open_instruct.value_estimation import MakeDatasetConfig, ScoreDatasetConfig; print('ok')"
uv run python -m open_instruct.value_estimation --help
```

Expected: 33 passing tests, 2 skipped (vllm-dependent), clean CLI help. Failure here means something is broken before we even start a GPU.

### Phase 1 — Smallest working PPO+value run (1 node, 8 GPUs, ~10 minutes to first step)

Goal: confirm `_init_value_model` loads, value forward runs end-to-end, GAE dispatches, value backward doesn't OOM or crash, checkpointing writes `value_model.bin`.

```bash
bash scripts/train/vip/qwen3_4b_base_ppo.sh
```

**What to watch**:
- Log line `Value model ready (lr=..., mini_batches=...)` appears during init on every rank.
- First `Value forward (no-grad)` timer logs (at ~step 1).
- `loss/value_avg`, `value/clipfrac_avg`, `value/grad_norm` appear in wandb.
- At the first `save_freq` boundary, `value_model/value_model.bin` appears next to the policy under `output_dir`.

**Likely failure modes**:
- OOM during `_init_value_model`: reduce `--num_unique_prompts_rollout` or switch to `deepspeed_stage 3` (already on).
- Value model init hangs on `deepspeed.initialize` — the value model uses the same `mpu` as the policy. If `sequence_parallel_size > 1` you might see NCCL group collision; the default recipe uses SP=1 so this shouldn't bite.
- `data_BT.rewards` is None assertion — the `StreamingDataLoaderConfig.use_value_model` sync isn't firing; confirm `setup_runtime_variables` is called before `DataPreparationActor` starts.

### Phase 2 — GT conditioning (1 node, 8 GPUs)

One run per template; skim the first 50 steps of each.

```bash
# answer_prefix (prefix template — entire pack gets a short prefix)
bash scripts/train/vip/qwen3_4b_base_ppo_gt.sh  # defaults to answer_prefix

# expected_accuracy (postfix — inserted between prompt and response)
bash scripts/train/vip/qwen3_4b_base_ppo_expected_accuracy.sh

# rollout_context (postfix, includes sibling rollouts)
bash scripts/train/vip/qwen3_4b_base_ppo_rollout_context.sh

# correct_demo (postfix, single sibling)
bash scripts/train/vip/qwen3_4b_base_ppo_correct_demo.sh
```

**What to watch**:
- `value/warmup` metric is 0 (no warmup enabled).
- Wandb value-loss curves differ between templates (they should be distinct learning dynamics).
- For `rollout_context`/`correct_demo`: logs should show non-empty `sibling_rollouts` being carried through the data actor. Add `--verbose` if you want the data-prep actor to log shapes.

**Common pitfall**: the `_forward_value_with_conditioning` path unpacks sub-sequences and runs a serial forward per sub-sequence, so this is **slower** than the no-conditioning path. Expect ~2-3x slower step time. If too slow for your patience, drop `--num_samples_per_prompt_rollout` to 8 for the first test.

### Phase 3 — SAE (1 node, 8 GPUs)

```bash
# Standalone SAE (no GT conditioning).
# Modify qwen3_4b_base_ppo.sh and add `--use_sae --sae_threshold 0.2`, OR run:
bash scripts/train/vip/qwen3_4b_base_vpretrain_sae.sh
```

**What to watch**:
- `value/sae_boundary_frac` metric in wandb — should hover around 0.05 to 0.25 on math rollouts for threshold 0.2.
- Training is NOT faster or slower than standard PPO (SAE only changes the advantage math).
- Ensure `--use_value_model` is on (the validator enforces this).

### Phase 4 — LM-yesno value model (1 node, 8 GPUs)

```bash
bash scripts/train/vip/qwen3_4b_base_vpretrain_lm_value.sh
```

**What to watch**:
- Log line `LM-value yes_id=... no_id=...` at init, confirming both tokens are single-token.
- `value/lm_p_yes_mean`, `value/lm_p_yes_correct`, `value/lm_p_yes_incorrect`, `value/lm_accuracy` in wandb.
- `p_yes_correct` should drift above 0.5, `p_yes_incorrect` below 0.5 (the value model is learning to separate correct from incorrect rollouts).

**Common pitfall**: some tokenizers split `Yes`/`No` into multiple tokens. The current code uses the first token id only — if your tokenizer doesn't produce single-token `Yes`/`No`, the value signal will be garbage. Check the init log.

### Phase 5 — Frozen-policy warmup (1 node, 8 GPUs)

```bash
bash scripts/train/vip/qwen3_4b_base_vpretrain_sae.sh  # has --value_warmup_steps 200
```

**What to watch**:
- `value/warmup` metric is 1.0 during steps 1..200, then drops to 0.
- `optim/grad_norm` should be absent or zero-logged during warmup (policy is not stepping).
- `Skipping weight sync at step N due to value/policy warmup` log lines appear during steps 1..200.
- After step 200, normal PPO resumes — confirm by watching weight-sync start firing and vLLM engines receiving updated weights.
- `reset_optimizer_after_value_warmup`: first post-warmup step prints `Reset policy optimizer state at training_step=201` on rank 0.

### Phase 6 — init_value_from_pretrained_checkpoint (1 node, 8 GPUs)

Requires a checkpoint from Phase 5. Point `PRETRAINED_VALUE_DIR` at a directory containing
`value_model.bin` (e.g. `<output_dir>/step_000200/value_model/`):

```bash
bash scripts/train/vip/qwen3_4b_base_ppo_vpretrain_init.sh "$BEAKER_IMAGE" \
    /path/to/vpretrain_sae_run/step_000200/value_model
```

**What to watch**:
- Init log: `Loaded pretrained value model from <path>/value_model.bin (missing=..., unexpected=...)`. Missing keys are OK if the head shape differs but should be small.
- Value loss should start **lower** than a fresh init because the pretrained value model is already calibrated.

### Phase 7 — init_value_from_rm (1 node, 8 GPUs)

Requires a trained reward model. Before running, confirm the RM has a `score` head (it's an `AutoModelForSequenceClassification`). Update the RM path in `qwen3_4b_base_ppo_rm_init.sh` and run:

```bash
bash scripts/train/vip/qwen3_4b_base_ppo_rm_init.sh "$BEAKER_IMAGE" allenai/your-rm-path
```

**What to watch**:
- Init log: `Initialized value model from RM <path> with trained score head`.
- Initial value predictions should be close to RM rewards — track `loss/value_avg` in the first 5 steps; it should be 10-100x smaller than a fresh-init value head.

### Phase 8 — Value estimation harness

One GPU is enough for this if the model is ≤7B. Two stages:

**Stage A: build the dataset** (takes ~20-40 minutes on a single H100 depending on length):

```bash
bash scripts/eval/value_estimation/make_dapo_dataset.sh \
    Qwen/Qwen3-4B-Base \
    /tmp/value_estimation/dapo_math_100pairs.parquet
```

**What to watch**:
- Log: `Kept N prompts with at least one correct + incorrect rollout` with N eventually reaching `target_num_pairs` (100) — if not, `num_prompts_to_sample` is too small.
- Output parquet has 200 rows (100 prompts × 2 rollouts).
- Each row has nonempty `probe_positions` and `mc_values` (probably empty for shorter rollouts; that's fine, just flag if **all** rows are empty which means `probe_interval` is wrong for your response length).

**Stage B: score with each value-model variant**:

```bash
# Pick a checkpoint from Phase 1-7:
VM=/path/to/value_model_checkpoint_dir

bash scripts/eval/value_estimation/score_scalar_value.sh "$VM" /tmp/value_estimation/dapo_math_100pairs.parquet /tmp/value_estimation/scalar.parquet
bash scripts/eval/value_estimation/score_value_with_gt.sh "$VM" /tmp/value_estimation/dapo_math_100pairs.parquet /tmp/value_estimation/gt.parquet
bash scripts/eval/value_estimation/score_value_with_rollout_context.sh "$VM" /tmp/value_estimation/dapo_math_100pairs.parquet /tmp/value_estimation/rc.parquet

bash scripts/eval/value_estimation/compare.sh /tmp/value_estimation/compare \
    /tmp/value_estimation/scalar.parquet \
    /tmp/value_estimation/gt.parquet \
    /tmp/value_estimation/rc.parquet
```

**What to watch**:
- Each `score_dataset` call emits a `.summary.json` with MAE, Pearson, Spearman, calibration bins, correct/incorrect pred means.
- `compare.sh` produces a markdown table comparing runs.
- If `mae > 0.5` consistently, the value model is ~as bad as random — check that the conditioning flags you passed to `score_dataset` match the training-time conditioning (a warning should fire if `training_args.json` is present and disagrees).

### Phase 9 — Generative value model (advanced, may expose scaffolds)

**This will hit the `NotImplementedError` in `grpo_fast_genvalue.main()`** until the second vLLM pool is wired. Don't attempt this until Phases 1-8 are green. When you do:

1. Work from `scripts/train/vip/qwen3_4b_base_genvalue.sh` as the CLI template.
2. The helpers that *do* work end-to-end today: `segment_rollout`, `score_partial_rollout_batch`, `build_generative_value_prompt`, `parse_generative_value_score`. These are all unit-tested and can be stitched into a single-node proof-of-concept.
3. Integration shopping list (what `main()` needs to learn to do):
   - Call `vllm_utils.create_vllm_engines` a second time with separate `prompt_queue` / `results_queue` / `actor_manager` name / `WeightTransferConfig`.
   - Reserve bundles for that second pool in `grpo_fast_resource_plan.py`.
   - Add a `setup_value_model_update_group` mirror of `setup_model_update_group` plus a `broadcast_to_value_vllm` method on the trainer actor.
   - Extend `weight_sync_thread` to sleep/wake both pools together.
   - In `DataPreparationActor._data_preparation_loop`, compute segment boundaries, send scoring requests to the gen-value pool, pack per-token values, and feed them into the policy GAE path.
   - Run REINFORCE on the trainer-side replica of the generative value model using the scored-segments + outcome as reward.

### Phase 10 — Regression (plain GRPO still works)

At the end, confirm nothing broke for runs that don't touch the value model. A 1-step smoke run of any existing `scripts/train/grpo_fast*.sh` (without the `--use_value_model` flag) should behave identically to `main`. Key invariant: when `use_value_model=False`, the `DataPreparationActor` takes the original group-relative advantage code path unchanged.

---

## Quick health check (one-liner to paste to a reviewer)

```
17 unit tests pass on CPU (new: test_value_model.py).
Branch compiles cleanly. grpo_fast.py +718 LOC, 3 new Python modules, 19 shell scripts.
Runs expected to pass GPU-side: Phases 1-8 above.
Known scaffold: Phase 9 (generative value model) requires second-vLLM-pool wiring in
  grpo_fast_genvalue.main(); all its dependency helpers are implemented + tested.
```
