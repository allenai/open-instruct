# Changelog

All notable changes to this project will be documented in this file.


### Added
- RL environment abstraction: `RLEnvironment` base class with `Tool` as a subclass, unifying tools and environments under a single `step(EnvCall) -> StepResult` interface. Removes `Executable`/`EnvOutput`/`_execute`/`safe_execute` indirection. Moves tools under `open_instruct/environments/tools/`. Includes example environments (`CounterEnv`, `GuessNumberEnv`) (https://github.com/allenai/open-instruct/pull/1478).
- Enable packing with torch.compile for DPO training, fix cu_seq_lens offset bug for padded chosen/rejected sequences, add tokens_per_second_per_gpu metric (https://github.com/allenai/open-instruct/pull/1466).
- Production DPO script for OLMo3-7B hybrid (https://github.com/allenai/open-instruct/pull/1449).
- Gradient accumulation/microbatching support for OLMo-core DPO training (https://github.com/allenai/open-instruct/pull/1447).
- Evolving rubrics support with RubricVerifier and utility functions for GRPO training (https://github.com/allenai/open-instruct/pull/1460).
- New perf metrics in PerfCallback: total_tokens, data_loading_seconds, data_loading_pct, wall_clock_per_step, step_overhead_pct (https://github.com/allenai/open-instruct/pull/1457).
- Warning when eval prompts are queuing up (new eval round starts before the previous one completes) (https://github.com/allenai/open-instruct/pull/1461).
- OLMo 3 tokenizer settings documentation covering chat template decisions for Instruct and Think models (https://github.com/allenai/open-instruct/pull/1455).
- torch.compile support for OLMo-core DPO training (https://github.com/allenai/open-instruct/pull/1445).
- Adds a GRPOTrainModule as part of the Olmo-core migration (https://github.com/allenai/open-instruct/pull/1412/)
- FSDP shard_degree and num_replicas configuration for OLMo-core DPO training (https://github.com/allenai/open-instruct/pull/1446).
- Budget mode gradient checkpointing support for OLMo-core DPO training (https://github.com/allenai/open-instruct/pull/1444).
- PerfCallback for MFU metrics in OLMo-core DPO training (https://github.com/allenai/open-instruct/pull/1442).
- NVIDIA H200 GPU support in `GPU_SPECS` (https://github.com/allenai/open-instruct/pull/1441).
- Documentation and runtime warning for `dataset_mixer_list` format (float=proportion, int=count) (https://github.com/allenai/open-instruct/pull/1434).

### Changed
- Replaces lambda collators with a "single_example_collator" (https://github.com/allenai/open-instruct/pull/1472).
- Clarified `activation_memory_budget` guidance in DPO utils with a practical default (`0.5`) and memory/speed tradeoff notes (https://github.com/allenai/open-instruct/pull/1460).
- Let TransformerTrainModule handle FSDP parallelism instead of manual application in DPO (https://github.com/allenai/open-instruct/pull/1458).
- Refactored DPOTrainModule to inherit from TransformerTrainModule (https://github.com/allenai/open-instruct/pull/1456)
- Increased vLLM health check timeout from 30s to 600s (10 minutes) (https://github.com/allenai/open-instruct/pull/1452).
- Updated vllm version to 0.14.1 (https://github.com/allenai/open-instruct/pull/1433).
- Changed default wandb x-axis from `episode` to `training_step` for grpo_fast (https://github.com/allenai/open-instruct/pull/1437).
- Made a bunch of changes to `dpo.py` so it matches `dpo_tune_cache.py` perfectly (https://github.com/allenai/open-instruct/pull/1451).

### Fixed
- Fixed test `single_example_collator` returning raw int for index, causing `TypeError` in `_iter_batches` (https://github.com/allenai/open-instruct/pull/1477).
- Fixed SFT integration test failing due to missing `--try_launch_beaker_eval_jobs false` flag (https://github.com/allenai/open-instruct/pull/1470).
- Fixed checkpoint cleanup race condition on shared filesystems by using `ignore_errors=True` and restricting cleanup to global rank 0 (https://github.com/allenai/open-instruct/pull/1468).
- Fixed checkpoint resume failing on Beaker retries by removing non-deterministic timestamp from `exp_name` (https://github.com/allenai/open-instruct/pull/1468).
- Fixed MFU calculation to count LM head FLOPs per token (https://github.com/allenai/open-instruct/pull/1457).
- Fixed training hang when `inflight_updates` is disabled by waiting for weight sync to complete before health check (https://github.com/allenai/open-instruct/pull/1454).
- Fixed evaluation responses being lost on timeout in grpo_fast by requeuing partial results (https://github.com/allenai/open-instruct/pull/1439).
- Beaker Experiment Launch now passes (https://github.com/allenai/open-instruct/pull/1424#pullrequestreview-3708034780).

### Removed
