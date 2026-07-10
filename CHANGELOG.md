# Changelog

All notable changes to this project will be documented in this file.


### Added
- Add `--eval_only` / `--eval_only_set_checkpoint` to the OLMo-core GRPO path (`open_instruct/grpo.py`): runs a single local evaluation round on `dataset_mixer_eval_list` with vLLM serving `model_name_or_path` directly (no learner GPUs, weight sync, or trainer), plus additive per-dataset eval metrics (`eval/pass_at_1/<dataset>` etc.) in `maybe_evaluate` for mixed eval sets, and fix `scripts/train/convert_olmo_core_to_hf.py` to read checkpoints with OLMo-core's DCP loader.

### Changed
- Change the default generation `temperature` to 1.0 and make `SamplingConfig.temperature` a required field so `StreamingConfig.temperature` is the single source of truth (https://github.com/allenai/open-instruct/pull/1725).
- Bump OLMo-core to the latest `main` commit (`9aa3280`) (https://github.com/allenai/open-instruct/pull/1723).
- Refactor OLMo-core DPO metrics: reduce token-weighted metrics inline in `train_batch` with a single `all_reduce` over the DP group (matching `GRPOTrainModule`), align wandb keys with `dpo_tune_cache.py` (`train_loss`, `logps/*`, `rewards/*`, `perf/mfu_step`, `perf/tokens_per_second_step`/`_total`), add `train/padding_fraction`, `train/sequences_per_rank`, and `train/global_sequences_per_step` metrics, and make `get_num_sequences` always return an `int` (https://github.com/allenai/open-instruct/pull/1719).
- Add `ModelConfig.loss_implementation` to select olmo-core's LM loss implementation (e.g. `fused_linear` for Liger FLCE), applied in `setup_model` before the model is built (https://github.com/allenai/open-instruct/pull/1714).

### Deprecated

### Removed

### Fixed
- Skip `ray start` in `configs/beaker_configs/ray_node_setup.sh` for single-node jobs (detected via the new `MASON_NUM_NODES` env var injected by `mason.py`), letting `ray.init()` pick random ports; previously two sub-node jobs packed onto one Beaker node collided on the hardcoded Ray head port and crashed at startup.
