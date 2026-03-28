# Changelog

All notable changes to this project will be documented in this file.

### Changed
- **smolzero-merge cleanup:** Restore single `generation_time` token stats (remove vLLM-summed timing split), main-style placement group and vLLM hybrid bundling, void `ensure_hf_repo_cached`, drop Qwen `CHAT_TEMPLATES` entries and `TokenizerConfig` template validation, remove `dataset_overwrite_cache` and `system_prompt_remove`. Follow-up PRs: push branches `feat/wandb-experiment-tracking` (W&B group, extended `wandb.init` config, mason `RUN_NAME` default description) and `feat/llm-judge-compass` (Compass verifier + judge utils) from dedicated worktrees.

### Added
- Add documentation for Slack alert integrations in GRPO and DPO training (https://github.com/allenai/open-instruct/pull/1529).
- Add `flash-attn-3` dependency for Flash Attention 3 support on H100/H800 GPUs. DPO training via olmo-core auto-detects FA3 at runtime (https://github.com/allenai/open-instruct/pull/1525).
- Tensor parallelism (TP) support for OLMo-core DPO training (https://github.com/allenai/open-instruct/pull/1467).
- Pulls out weight sync code from GRPO into a more generic function (https://github.com/allenai/open-instruct/pull/1411#pullrequestreview-3694117967)
- Adds callbacks for GRPO training with Olmo-core's trainer (https://github.com/allenai/open-instruct/pull/1397).
- Adds FSDP2 block-by-block weight gathering support for vLLM weight sync.
- OLMo-core GRPO actor with Ray-distributed FSDP2 training (https://github.com/allenai/open-instruct/pull/1398).
### Fixed
- Fix ZeRO-2 discarding gradients during manual gradient accumulation by using `set_gradient_accumulation_boundary()` (https://github.com/allenai/open-instruct/pull/1498).
- Pre-download HF model on main process before Ray actors spawn to avoid hitting HuggingFace rate limits.
- Batch vLLM weight sync broadcasts to reduce Ray RPCs from ~200+ to 1, fixing timeouts with 32k response lengths (https://github.com/allenai/open-instruct/pull/1535).
- Fix `wandb_tracker.run.url` `AttributeError` on non-main processes in multi-node SFT training by guarding accesses with `accelerator.is_main_process` checks (https://github.com/allenai/open-instruct/pull/1539).
- Fix `UnboundLocalError` for `beaker_config` in SFT tracking setup when `push_to_hub` is disabled (https://github.com/allenai/open-instruct/pull/1539).
- Pre-download HF model on main process before Ray actors spawn to avoid hitting HuggingFace rate limits (https://github.com/allenai/open-instruct/pull/1528).
- Fixed GPU test failures: DPO `get_num_tokens` attention mask matching, DPO forward pass logps computation, mock model interface in `test_dpo_utils_gpu.py`, patch target in `test_olmo_core_callbacks_gpu.py`, reference logprobs cache `drop_last`, and flaky streaming dataloader tool test (https://github.com/allenai/open-instruct/pull/1514).
- Extended CONTRIBUTING.md with documentation on running tests, CI workflows, Beaker experiments, GRPO/DPO test scripts, and environment variables.

### Added
- Add hybrid model (Olmo-Hybrid) support: MambaSpec monkey-patch for vLLM dtype serialization, `trust_remote_code` pass-through to vLLM engines, `get_text_config()` for multimodal model support, dependency upgrades (vllm>=0.18.0, transformers>=5.3.0), `return_dict=False` for transformers 5.x compat, and hybrid test/production training scripts (https://github.com/allenai/open-instruct/pull/1425).
- Add Ulysses sequence parallelism support to SFT training via `--sequence_parallel_size`, using HF Accelerate's `ParallelismConfig` with the DeepSpeed Ulysses SP backend. Enables training with much longer context lengths by sharding sequences across GPUs. Includes SP-aware loss aggregation, batch collation (padding to divisible seq len, index column removal), LR scheduler correction, and a two-node integration test script (https://github.com/allenai/open-instruct/pull/1539).
- Added a GRPO implementation that uses OLMo-core with Ray-distributed FSDP2 training (https://github.com/allenai/open-instruct/pull/1389).
- Add the Qwen 3 4B DAPO math 32k training launch script under `scripts/train/qwen/` (https://github.com/allenai/open-instruct/pull/1536).
- Add Muon optimizer support to DPO training via OLMo-core's native MuonConfig (https://github.com/allenai/open-instruct/pull/1533).
- Add documentation for Slack alert integrations in GRPO and DPO training (https://github.com/allenai/open-instruct/pull/1529).
- Add `flash-attn-3` dependency for Flash Attention 3 support on H100/H800 GPUs. DPO training via olmo-core auto-detects FA3 at runtime (https://github.com/allenai/open-instruct/pull/1525).
- Tensor parallelism (TP) support for OLMo-core DPO training (https://github.com/allenai/open-instruct/pull/1467).
- Pulls out weight sync code from GRPO into a more generic function (https://github.com/allenai/open-instruct/pull/1411#pullrequestreview-3694117967)
- Adds callbacks for GRPO training with Olmo-core's trainer (https://github.com/allenai/open-instruct/pull/1397).
- Adds FSDP2 block-by-block weight gathering support for vLLM weight sync.
- OLMo-core GRPO actor with Ray-distributed FSDP2 training (https://github.com/allenai/open-instruct/pull/1398).
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

### Fixed
- Fix GPU test deadlock and make dataset transformation tests fully offline with local fixtures (https://github.com/allenai/open-instruct/pull/1563).
- Remove stale `VLLM_ATTENTION_BACKEND` from `DEFAULT_ENV_VARS`; vLLM 0.18+ auto-detects attention backends (https://github.com/allenai/open-instruct/pull/1564).
- Use `setup_zero_stage3_hooks()` for DeepSpeed 0.18+ compat in `add_hooks` (https://github.com/allenai/open-instruct/pull/1566).
- Remove the runtime `temperature` field from GRPO `ExperimentConfig` and pass streaming temperature explicitly, avoiding W&B config collisions with `StreamingDataLoaderConfig.temperature` (https://github.com/allenai/open-instruct/pull/1561).
- Log `val/tis_ratio` and `val/tis_clipfrac` in `grpo_fast` so truncated importance sampling diagnostics are visible during GRPO training (https://github.com/allenai/open-instruct/pull/1558).
- Fix SP double-shift bug: keep both `labels` and `shift_labels` in batch so `ForCausalLMLoss` uses pre-shifted labels (https://github.com/allenai/open-instruct/pull/1549).
- Fix `total_batch_size` logging to account for sequence parallelism (SP ranks share data, not independent) (https://github.com/allenai/open-instruct/pull/1542).
- Got Olmo-core GRPO running in single-gpu mode and added a grpo.py debug script (https://github.com/allenai/open-instruct/pull/1543).
- Batch vLLM weight sync broadcasts to reduce Ray RPCs from ~200+ to 1, fixing timeouts with 32k response lengths (https://github.com/allenai/open-instruct/pull/1535).
- Fix `wandb_tracker.run.url` `AttributeError` on non-main processes in multi-node SFT training by guarding accesses with `accelerator.is_main_process` checks (https://github.com/allenai/open-instruct/pull/1539).
- Fix `UnboundLocalError` for `beaker_config` in SFT tracking setup when `push_to_hub` is disabled (https://github.com/allenai/open-instruct/pull/1539).
- Pre-download HF model on main process before Ray actors spawn to avoid hitting HuggingFace rate limits (https://github.com/allenai/open-instruct/pull/1528).
- Fixed GPU test failures: DPO `get_num_tokens` attention mask matching, DPO forward pass logps computation, mock model interface in `test_dpo_utils_gpu.py`, patch target in `test_olmo_core_callbacks_gpu.py`, reference logprobs cache `drop_last`, and flaky streaming dataloader tool test (https://github.com/allenai/open-instruct/pull/1514).
- Extended CONTRIBUTING.md with documentation on running tests, CI workflows, Beaker experiments, GRPO/DPO test scripts, and environment variables.
<<<<<<< HEAD
=======

### Changed
- Add a configurable vLLM attention backend option and switch remaining `flash_attention_2` defaults/references to `flash_attention_3` (https://github.com/allenai/open-instruct/pull/1559).
- Switch back to CUDA 12.8.1, pin `flash-attn-3` to a direct x86_64 wheel URL to avoid flat-index drift to aarch64-only releases (https://github.com/allenai/open-instruct/pull/1560).
- Added other configs to wandb logging so all hyperparams are visible, set beaker name with RUN_NAME for grpo_fast.py (https://github.com/allenai/open-instruct/pull/1554).
- Updated vLLM to 0.17.1 and torch to 2.10+.
- Log `optim/grad_norm` in `grpo_fast`, including non-finite DeepSpeed values (`nan`/`inf`) when they occur (https://github.com/allenai/open-instruct/pull/1540).
>>>>>>> 7caec8666ee8b75ec70d59b8f4e52bfe4ef539ac
- Update GRPO/DPO defaults to match Olmo 3 experiments (`async_steps=8`, `advantage_normalization_type=mean_std`, `inflight_updates=True`, `clip_higher=0.28`, `truncated_importance_sampling_ratio_cap=10.0`) and remove redundant flags from training scripts (https://github.com/allenai/open-instruct/pull/1547).
- Removed all Augusta cluster (`ai2/augusta`) references and GCP-cluster-specific code paths since the cluster has been decommissioned.
- Added GRPO fast idle wait-time metrics for trainer waiting on inference and generation waiting on trainer consumption (`time/trainer_idle_waiting_for_inference`, `time/generation_idle_waiting_for_trainer`) (https://github.com/allenai/open-instruct/pull/1516).
- Updated vLLM to 0.16.0 and fixed `ChatCompletionRequest` import path which moved to `vllm.entrypoints.openai.chat_completion.protocol` (https://github.com/allenai/open-instruct/pull/1510).
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
- Fixed GSM8K reward verification for signed final answers by preserving explicit `+` and `-` signs when extracting the last numeric prediction, including boxed negative answers (https://github.com/allenai/open-instruct/pull/1530).
- Exclude `CUDA_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` from the Ray `runtime_env` so Ray can manage per-worker GPU visibility correctly on heterogeneous clusters and avoid invalid GPU assignments (https://github.com/allenai/open-instruct/pull/1519).
- Include tokenizer configuration in per-transform dataset cache fingerprints so rerunning transformations with a different tokenizer does not silently reuse stale cached outputs (https://github.com/allenai/open-instruct/pull/1518).
- Fixed `grpo_fast` local eval rounds enqueueing 0 prompts after the first run by resetting `eval_data_loader` after each eval pass (stateful `DataLoaderBase` requires reset after epoch exhaustion); also switched eval prompt ID prefix from constant `0` to `training_step` to avoid cross-round metadata key collisions in vLLM request tracking (https://github.com/allenai/open-instruct/pull/1493).
- Force `generation_config="vllm"` in vLLM engine kwargs to prevent model HF generation defaults from capping OpenAI request `max_tokens` (https://github.com/allenai/open-instruct/pull/1512).
- Avoided synchronous CUDA transfers when moving batches to device (https://github.com/allenai/open-instruct/pull/1443).

### Removed
- Deletes some commented out code (https://github.com/allenai/open-instruct/pull/1537).
