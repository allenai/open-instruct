# Changelog

All notable changes to this project will be documented in this file.


### Added
- Add `--eval_only` / `--eval_only_set_checkpoint` to the OLMo-core GRPO path (`open_instruct/grpo.py`): runs a single local evaluation round on `dataset_mixer_eval_list` with vLLM serving `model_name_or_path` directly (no learner GPUs, weight sync, or trainer), plus additive per-dataset eval metrics (`eval/pass_at_1/<dataset>` etc.) in `maybe_evaluate` for mixed eval sets, and fix `scripts/train/convert_olmo_core_to_hf.py` to read checkpoints with OLMo-core's DCP loader.
- Add tool-schema support to SFT tokenization: the `tools` column is parsed (JSON strings accepted) and passed to `apply_chat_template`, assistant labels are derived from offset mappings, and the tools column is consumed rather than persisted (https://github.com/allenai/open-instruct/pull/1746).
- Drop stale async rollout results whose generating policy is more than `async_steps` behind the trainer (`max_result_age_steps`), replenishing a fresh prompt and logging a `stale_results_dropped` metric (https://github.com/allenai/open-instruct/pull/1738).

### Changed
- Automatically publish stable CUDA 12 and CUDA 13 Beaker image aliases after merge-queue integration tests pass (https://github.com/allenai/open-instruct/pull/1783).
- Wire `keep_last_n_checkpoints` through `build_checkpointer_callback` and `build_base_callbacks` to OLMo-core's new `max_checkpoints` parameter across SFT, DPO, and GRPO training paths; bump OLMo-core to the commit that added `max_checkpoints` (`fa6c501`). Negative values (e.g. `-1`) mean unlimited (https://github.com/allenai/open-instruct/pull/1701).
- Add selectable CUDA 12.8 and CUDA 13.0 Docker builds, including matching torch, vLLM, and flash-attention dependency variants, and add B300 support on the new `ai2/holmes` cluster (https://github.com/allenai/open-instruct/pull/1758).
- Increase default environment pool acquire timeout to 7200s (https://github.com/allenai/open-instruct/pull/1729).
- Make ModelDims.from_hf_config robust to explicit head_dim (https://github.com/allenai/open-instruct/pull/1743).
- Change the default generation `temperature` to 1.0 and make `SamplingConfig.temperature` a required field so `StreamingConfig.temperature` is the single source of truth (https://github.com/allenai/open-instruct/pull/1725).
- Bump OLMo-core to the latest `main` commit (`9aa3280`) (https://github.com/allenai/open-instruct/pull/1723).
- Refactor OLMo-core DPO metrics: reduce token-weighted metrics inline in `train_batch` with a single `all_reduce` over the DP group (matching `GRPOTrainModule`), align wandb keys with `dpo_tune_cache.py` (`train_loss`, `logps/*`, `rewards/*`, `perf/mfu_step`, `perf/tokens_per_second_step`/`_total`), add `train/padding_fraction`, `train/sequences_per_rank`, and `train/global_sequences_per_step` metrics, and make `get_num_sequences` always return an `int` (https://github.com/allenai/open-instruct/pull/1719).
- Add `ModelConfig.loss_implementation` to select olmo-core's LM loss implementation (e.g. `fused_linear` for Liger FLCE), applied in `setup_model` before the model is built (https://github.com/allenai/open-instruct/pull/1714).

### Deprecated

### Removed

### Fixed
- Skip `ray start` in `configs/beaker_configs/ray_node_setup.sh` for single-node jobs (detected via the new `MASON_NUM_NODES` env var injected by `mason.py`), letting `ray.init()` pick random ports; previously two sub-node jobs packed onto one Beaker node collided on the hardcoded Ray head port and crashed at startup.
- Track the CUDA 12 image suffix in the merge-queue Beaker workflow and allow enough time for the larger image build and upload (https://github.com/allenai/open-instruct/pull/1783).
- Exclude nested virtualenvs (e.g. `oe-eval-internal/.venv/`) from the Docker build context, so a uv venv inside a nested clone no longer fails the image build on a dangling host-interpreter symlink (https://github.com/allenai/open-instruct/pull/1786).
