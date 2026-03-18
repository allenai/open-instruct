# Changelog

All notable changes to this project will be documented in this file.


### Added
- Add Muon optimizer support to DPO training via microsoft/dion (https://github.com/allenai/open-instruct/pull/1531).
- Add documentation for Slack alert integrations in GRPO and DPO training (https://github.com/allenai/open-instruct/pull/1529).
- Add `flash-attn-3` dependency for Flash Attention 3 support on H100/H800 GPUs. DPO training via olmo-core auto-detects FA3 at runtime (https://github.com/allenai/open-instruct/pull/1525).
- Tensor parallelism (TP) support for OLMo-core DPO training (https://github.com/allenai/open-instruct/pull/1467).
- Pulls out weight sync code from GRPO into a more generic function (https://github.com/allenai/open-instruct/pull/1411#pullrequestreview-3694117967)
- Adds callbacks for GRPO training with Olmo-core's trainer (https://github.com/allenai/open-instruct/pull/1397).
- Adds FSDP2 block-by-block weight gathering support for vLLM weight sync.
- OLMo-core GRPO actor with Ray-distributed FSDP2 training (https://github.com/allenai/open-instruct/pull/1398).

### Fixed
- Pre-download HF model on main process before Ray actors spawn to avoid hitting HuggingFace rate limits (https://github.com/allenai/open-instruct/pull/1528).
- Fixed GPU test failures: DPO `get_num_tokens` attention mask matching, DPO forward pass logps computation, mock model interface in `test_dpo_utils_gpu.py`, patch target in `test_olmo_core_callbacks_gpu.py`, reference logprobs cache `drop_last`, and flaky streaming dataloader tool test (https://github.com/allenai/open-instruct/pull/1514).
- Extended CONTRIBUTING.md with documentation on running tests, CI workflows, Beaker experiments, GRPO/DPO test scripts, and environment variables.

### Changed
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
- Exclude `CUDA_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` from the Ray `runtime_env` so Ray can manage per-worker GPU visibility correctly on heterogeneous clusters and avoid invalid GPU assignments (https://github.com/allenai/open-instruct/pull/1519).
- Include tokenizer configuration in per-transform dataset cache fingerprints so rerunning transformations with a different tokenizer does not silently reuse stale cached outputs (https://github.com/allenai/open-instruct/pull/1518).
- Fixed `grpo_fast` local eval rounds enqueueing 0 prompts after the first run by resetting `eval_data_loader` after each eval pass (stateful `DataLoaderBase` requires reset after epoch exhaustion); also switched eval prompt ID prefix from constant `0` to `training_step` to avoid cross-round metadata key collisions in vLLM request tracking (https://github.com/allenai/open-instruct/pull/1493).
- Force `generation_config="vllm"` in vLLM engine kwargs to prevent model HF generation defaults from capping OpenAI request `max_tokens` (https://github.com/allenai/open-instruct/pull/1512).

### Removed
