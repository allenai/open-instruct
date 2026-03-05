# Changelog

All notable changes to this project will be documented in this file.


### Added
- Tensor parallelism (TP) support for OLMo-core DPO training (https://github.com/allenai/open-instruct/pull/1467).

### Changed
- Enable multiple active targets per rollout in RL training by unifying tool and environment dispatch in vLLM with upfront pool activation (no lazy tool acquisition), normalizing row-level `env_config` during preprocessing (`dict`/`list` forms -> canonical `{"env_configs": [...]}`), enforcing canonical-only runtime parsing, validating unknown configured targets early, and reporting per-target metrics while retaining text-environment handling (https://github.com/allenai/open-instruct/pull/1500).
- Bound async data preparation to stay within `async_steps` of training, preventing training data getting too far out of sync with trainer. (https://github.com/allenai/open-instruct/pull/1496).
- Refactor Legacy and DRTulu tool parsers to use OpenAI-format `tool_definitions` instead of Ray `tool_actors`. Removes `import ray` from `parsers.py`, fixes DRTulu parser which was broken after the pool refactor, and fixes `--tool_parser_type` typo in dr_tulu debug script (https://github.com/allenai/open-instruct/pull/1491).
- Replaces lambda collators with a "single_example_collator" (https://github.com/allenai/open-instruct/pull/1472).
- Clarified `activation_memory_budget` guidance in DPO utils with a practical default (`0.5`) and memory/speed tradeoff notes (https://github.com/allenai/open-instruct/pull/1460).
- Let TransformerTrainModule handle FSDP parallelism instead of manual application in DPO (https://github.com/allenai/open-instruct/pull/1458).
- Refactored DPOTrainModule to inherit from TransformerTrainModule (https://github.com/allenai/open-instruct/pull/1456)
- Increased vLLM health check timeout from 30s to 600s (10 minutes) (https://github.com/allenai/open-instruct/pull/1452).
- Updated vllm version to 0.14.1 (https://github.com/allenai/open-instruct/pull/1433).
- Changed default wandb x-axis from `episode` to `training_step` for grpo_fast (https://github.com/allenai/open-instruct/pull/1437).
- Made a bunch of changes to `dpo.py` so it matches `dpo_tune_cache.py` perfectly (https://github.com/allenai/open-instruct/pull/1451).

### Fixed

### Removed
