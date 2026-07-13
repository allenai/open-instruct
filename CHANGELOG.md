# Changelog

All notable changes to this project will be documented in this file.


### Added
- Add `keep_last_n_saves` to `grpo.py`/`grpo_fast.py` to bound the number of intermediate HuggingFace-format model saves kept under `{output_dir}_checkpoints` (-1 for unlimited, matching current behavior); on the OLMo-core (`grpo.py`) training path, periodic HF-format saves following `save_freq` are now performed via a new `HFCheckpointCallback`, since previously `save_freq` was a no-op there (only the final model was saved in HF format). Builds on the `keep_last_n_checkpoints`/`max_checkpoints` wiring from https://github.com/allenai/open-instruct/pull/1701 (https://github.com/allenai/open-instruct/pull/1754).
- Drop stale async rollout results whose generating policy is more than `async_steps` behind the trainer (`max_result_age_steps`), replenishing a fresh prompt and logging a `stale_results_dropped` metric (https://github.com/allenai/open-instruct/pull/1738).

### Changed
- Increase default environment pool acquire timeout to 7200s (https://github.com/allenai/open-instruct/pull/1729).
- Change the default generation `temperature` to 1.0 and make `SamplingConfig.temperature` a required field so `StreamingConfig.temperature` is the single source of truth (https://github.com/allenai/open-instruct/pull/1725).
- Bump OLMo-core to the latest `main` commit (`9aa3280`) (https://github.com/allenai/open-instruct/pull/1723).
- Refactor OLMo-core DPO metrics: reduce token-weighted metrics inline in `train_batch` with a single `all_reduce` over the DP group (matching `GRPOTrainModule`), align wandb keys with `dpo_tune_cache.py` (`train_loss`, `logps/*`, `rewards/*`, `perf/mfu_step`, `perf/tokens_per_second_step`/`_total`), add `train/padding_fraction`, `train/sequences_per_rank`, and `train/global_sequences_per_step` metrics, and make `get_num_sequences` always return an `int` (https://github.com/allenai/open-instruct/pull/1719).
- Add `ModelConfig.loss_implementation` to select olmo-core's LM loss implementation (e.g. `fused_linear` for Liger FLCE), applied in `setup_model` before the model is built (https://github.com/allenai/open-instruct/pull/1714).

### Deprecated

### Removed

### Fixed
