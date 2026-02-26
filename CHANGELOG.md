# Changelog

All notable changes to this project will be documented in this file.


### Fixed
- Fix ZeRO-2 discarding gradients during manual gradient accumulation by using `set_gradient_accumulation_boundary()` (https://github.com/allenai/open-instruct/pull/1498).

### Added
- Add Docker sandbox backend and `GenericSandboxEnv` environment for code execution during RL training. `DockerBackend` with command timeout, configurable memory limits, `put_archive`/`get_archive` file I/O, and `remove=True` auto-cleanup. `GenericSandboxEnv` provides `execute_bash` (stateful bash with env/cwd persistence) and `str_replace_editor` (view/create/str_replace/insert with correct line numbering). Configurable penalty, image, and memory via `GenericSandboxEnvConfig`. Includes 1-GPU debug script (https://github.com/allenai/open-instruct/pull/1490).
- Wire RL environments into vLLM generation loop and preprocessing: unified tool/env system with single `TOOL_REGISTRY`, pooled actors via shared `EnvironmentPool` Ray actor (async acquire/release, auto-sized to rollout concurrency), `RolloutState` tracks all per-rollout state, `PassthroughVerifier` + `RewardAggregator` for per-turn rewards (verifier score folded into last turn before aggregation), `BaseEnvConfig` in `environments/base.py`, `--max_steps` unified, `--pool_size` configurable, auto-discovery of tools from datasets, 1-GPU debug scripts for counter/guess_number envs (https://github.com/allenai/open-instruct/pull/1479).
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

- Fixed weight sync thread hang when `inflight_updates=False`: wait for all vLLM `engine.update_weight` RPCs to complete before unpausing actors, preventing `health_check_fn` from blocking indefinitely (https://github.com/allenai/open-instruct/pull/1480).
- Fixed `nodes_needed` calculation in `grpo_fast` `kv_cache_max_concurrency` warning using `math.ceil()` instead of floor division to avoid undercounting required inference nodes (https://github.com/allenai/open-instruct/pull/1474).
- Fixed `eval_on_step_0` never triggering in `grpo_fast` because it was gated behind the `training_step % local_eval_every == 0` modulo check; also guard `local_eval_every <= 0` to prevent accidental every-step eval or `ZeroDivisionError` (https://github.com/allenai/open-instruct/pull/1485).
- Fixed `TypeError` in `pack_padded_sequences` when `attention_mask` is a float tensor, and vectorized the packing to avoid per-sequence host-device synchronizations (https://github.com/allenai/open-instruct/pull/1486).
- Fixed silent prompt/ground-truth mismatch in RLVR caused by redundant dataset shuffle desyncing the `"index"` column from positional indices, leading to wrong rewards and wrong `exclude_index` exclusions (https://github.com/allenai/open-instruct/pull/1484).
- Fixed test `single_example_collator` returning raw int for index, causing `TypeError` in `_iter_batches` (https://github.com/allenai/open-instruct/pull/1477).
- Fixed SFT integration test failing due to missing `--try_launch_beaker_eval_jobs false` flag (https://github.com/allenai/open-instruct/pull/1470).
- Fixed checkpoint cleanup race condition on shared filesystems by using `ignore_errors=True` and restricting cleanup to global rank 0 (https://github.com/allenai/open-instruct/pull/1468).
- Fixed checkpoint resume failing on Beaker retries by removing non-deterministic timestamp from `exp_name` (https://github.com/allenai/open-instruct/pull/1468).
- Fixed MFU calculation to count LM head FLOPs per token (https://github.com/allenai/open-instruct/pull/1457).
- Fixed training hang when `inflight_updates` is disabled by waiting for weight sync to complete before health check (https://github.com/allenai/open-instruct/pull/1454).
- Fixed evaluation responses being lost on timeout in grpo_fast by requeuing partial results (https://github.com/allenai/open-instruct/pull/1439).
- Beaker Experiment Launch now passes (https://github.com/allenai/open-instruct/pull/1424#pullrequestreview-3708034780).

### Removed
