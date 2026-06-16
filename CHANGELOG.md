# Changelog

All notable changes to this project will be documented in this file.


### Added
- Add OLMo-core (FSDP) launch scripts `scripts/train/olmo3/32b_think_rl_olmocore.sh` and `scripts/train/olmo3/32b_rlzero_math_olmocore.sh`, which run 32B GRPO via `open_instruct/grpo.py` instead of the DeepSpeed `grpo_fast.py` (https://github.com/allenai/open-instruct/pull/XXXX).

### Changed
- Change the default generation `temperature` to 1.0 and make `SamplingConfig.temperature` a required field so `StreamingConfig.temperature` is the single source of truth (https://github.com/allenai/open-instruct/pull/1725).
- Bump OLMo-core to the latest `main` commit (`9aa3280`) (https://github.com/allenai/open-instruct/pull/1723).
- Refactor OLMo-core DPO metrics: reduce token-weighted metrics inline in `train_batch` with a single `all_reduce` over the DP group (matching `GRPOTrainModule`), align wandb keys with `dpo_tune_cache.py` (`train_loss`, `logps/*`, `rewards/*`, `perf/mfu_step`, `perf/tokens_per_second_step`/`_total`), add `train/padding_fraction`, `train/sequences_per_rank`, and `train/global_sequences_per_step` metrics, and make `get_num_sequences` always return an `int` (https://github.com/allenai/open-instruct/pull/1719).

### Deprecated

### Removed

### Fixed
