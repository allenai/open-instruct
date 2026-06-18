# Changelog

All notable changes to this project will be documented in this file.


### Added

### Changed
- Change the default generation `temperature` to 1.0 and make `SamplingConfig.temperature` a required field so `StreamingConfig.temperature` is the single source of truth (https://github.com/allenai/open-instruct/pull/1725).
- Bump OLMo-core to the latest `main` commit (`9aa3280`) (https://github.com/allenai/open-instruct/pull/1723).
- Refactor OLMo-core DPO metrics: reduce token-weighted metrics inline in `train_batch` with a single `all_reduce` over the DP group (matching `GRPOTrainModule`), align wandb keys with `dpo_tune_cache.py` (`train_loss`, `logps/*`, `rewards/*`, `perf/mfu_step`, `perf/tokens_per_second_step`/`_total`), add `train/padding_fraction`, `train/sequences_per_rank`, and `train/global_sequences_per_step` metrics, and make `get_num_sequences` always return an `int` (https://github.com/allenai/open-instruct/pull/1719).
- Add `ModelConfig.loss_implementation` to select olmo-core's LM loss implementation (e.g. `fused_linear` for Liger FLCE), applied in `setup_model` before the model is built (https://github.com/allenai/open-instruct/pull/1714).
- Expand type-checking coverage by replacing `# ty: ignore` directives with typed casts and fixing related type issues (https://github.com/allenai/open-instruct/pull/1688).
- Add TV divergence rho filtering for GRPO (https://github.com/allenai/open-instruct/pull/1681).
- Export `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPEN_INSTRUCT=0.0.0+debug` in `scripts/train/debug/grpo.sh` and `grpo_fast.sh` (local Ray debug scripts that disable torch compile) so setuptools-scm can resolve the package version (https://github.com/allenai/open-instruct/pull/1696).
- Simplify GRPO clip fraction handling by returning the final policy loss and clip fraction directly from `compute_grpo_loss` (https://github.com/allenai/open-instruct/pull/1679).
- Bring `grpo.py` (OLMo-core GRPO) to feature parity with `grpo_fast.py`: add `EvalCallback`, `setup_eval` actor RPC, unconditional vLLM-sync callback, `ConstantWithWarmup` scheduler support, and `StepTimingCallback` end-to-end step timing (https://github.com/allenai/open-instruct/pull/1672).
- Remove references to deleted `ppo_vllm_thread_ray_gtrl.py` script: delete broken launch scripts (`scripts/train/debug/ppo.sh`, `scripts/train/rlvr/tulu_rlvr.sh`, `scripts/train/tulu3/ppo_8b.sh`) and add historical-reference notes to `docs/tulu3.md` and `docs/archived_dev_scripts/olmoe_0125.sh` pointing to the deletion commit. Also drop the dead `update_command_args.py` references: delete `scripts/train/benchmark.sh` and its section in `docs/get_started/ai2_internal_setup.md`, and update the README RLVR quickstart to launch `grpo_fast.py` via `scripts/train/build_image_and_launch.sh`.
- Bump vllm to >=0.19.1 (and refresh `uv.lock`, including compressed-tensors v0.14.0.1 → v0.15.0.1).
- Move `maybe_evaluate` from `grpo_fast.py` to `grpo_utils.py` and drop the duplicate `PolicyTrainerRayProcess.calculate_token_counts` method, routing both trainer paths through the shared `grpo_utils.calculate_token_counts` (https://github.com/allenai/open-instruct/pull/1669).
- Rename `time/trainer_idle_waiting_for_inference` to `time/trainer_waiting_for_data` and `time/generation_idle_waiting_for_trainer` to `time/generation_waiting_for_trainer`, and emit per-Group generation timing (`time/group_generation_{mean,max,min}` plus `batch/per_group_generation_times` histogram) so latency vs. throughput in the inference pipeline is legible from wandb  (https://github.com/allenai/open-instruct/pull/1690).
- Add parameterized `combine_dataset` tests in `open_instruct/test_utils.py` against local jsonl fixtures (no network), covering varied fractional/sample-count weight combinations and split-count mismatch (would have caught the bug fixed in #1674). Extract the interleaved-list→dict parsing into a shared `utils.parse_dataset_mixer_list` helper (with its own parameterized unit tests) and tighten `combine_dataset` / `get_datasets` to accept dict-only `dataset_mixer`; the one external list-form caller (`rejection_sampling/generation.py`) now converts at the call site.
- Make `mason.py` `--output_dir` / `--checkpoint_state_dir` overrides idempotent via `replace_or_append_flag`, add `open_instruct/grpo.py` to `OPEN_INSTRUCT_COMMANDS` / `OPEN_INSTRUCT_RESUMABLES`, and wire OLMo-core checkpoint save/resume into `grpo.py` (`CheckpointerCallback` + `DataPreparationActorCheckpointCallback` + `LoadStrategy.if_available`) so resumable Beaker jobs actually resume (https://github.com/allenai/open-instruct/pull/1666).
- Make `--budget` optional in `mason.py` (falls back to the workspace's default budget) and drop the explicit `--budget` flag from launch scripts where it already matched the workspace default (https://github.com/allenai/open-instruct/pull/1673).
- Restore 🤡 to resample warnings and use `self.training_step` in `DataPreparationActor.run` (https://github.com/allenai/open-instruct/pull/1663).
- Add a unified `use_rho_correction` interface (clamp + mask, per-token or sequence-level) for the train/infer engine mismatch in GRPO loss; replaces `truncated_importance_sampling_ratio_cap` and the IcePop flags (https://github.com/allenai/open-instruct/pull/1650).
- Resample on filtered batches in `DataPreparationActor` instead of emitting empty `CollatedBatchData`, unifying the `grpo.py` and `grpo_fast.py` consumer paths and removing the now-dead empty-batch checks in `grpo_fast.py` (https://github.com/allenai/open-instruct/pull/1660).
- Update Beaker budget from `ai2/oe-omai` to `ai2/oe-other` across launch scripts and beaker configs.
- Update Beaker budget from `ai2/oe-adapt` to `ai2/oe-omai` across launch scripts and beaker configs to fix experiment launch failures from the retired budget (https://github.com/allenai/open-instruct/pull/1662).
- Log every filtered prompt in `accumulate_inference_batches` at INFO level with the zero/solved/nonzero breakdown, and add `batch/filtered_prompts_pct` to wandb so policy collapse / convergence is visible without spelunking debug logs (https://github.com/allenai/open-instruct/pull/1657).
- Aggregate prompt/response lengths across all DP ranks (deduplicating SP groups) when computing GRPO step token counts and utilization metrics, instead of using only rank 0 (https://github.com/allenai/open-instruct/pull/1659).
- Split `accumulate_inference_batches` into `process_single_result` and `combine_processed_results` for clarity (https://github.com/allenai/open-instruct/pull/1614).
- Match reference SFT run: `olmo_core_finetune.py` parity with pure olmo-core; default CP strategy switched to `ulysses` and ring-flash-attn dependency removed (https://github.com/allenai/open-instruct/pull/1620).
- Address review feedback on #1620: derive vocab size from the run's tokenizer (no longer hardcoded to dolma2), validate complete numpy artifacts before reusing the SFT cache, fold seed/max_seq_length into the cache directory, fix HF-vs-olmo-core checkpoint detection for relative local paths, and log which checkpoint format was detected (https://github.com/allenai/open-instruct/pull/1620).
- Stream SFT tokens/labels/boundaries directly to `_*.partial.bin` files and derive per-dataset stats at the end from disk, dropping the explicit `_checkpoint.json` file. `--resume` now works by truncating the partial files to a consistent sample boundary (https://github.com/allenai/open-instruct/pull/1631).
- Revert reapply of packaging fix from #1634 (https://github.com/allenai/open-instruct/pull/1637).
- Drop unused `data_types` import and inline `batch["batch"].to(device)` in `GRPOTrainModule` (https://github.com/allenai/open-instruct/pull/1635).
- Use incremental binary checkpoint for SFT tokenization resume, eliminating O(N²) re-serialization (https://github.com/allenai/open-instruct/pull/1633).
- Extract numpy SFT conversion helpers into `open_instruct.numpy_dataset_conversion` (https://github.com/allenai/open-instruct/pull/1622).
- Simplified model step tracking logic (https://github.com/allenai/open-instruct/pull/1616).
- Pass `attention_mask=None` in GRPO `forward_for_logprobs` calls — HF constructs the correct 3D intra-document mask from `position_ids` internally (https://github.com/allenai/open-instruct/pull/1617).
- Migrate GRPO trainer→vLLM weight sync to vLLM 0.16.0's native weight transfer API (`NCCLWeightTransferEngine`), replacing custom NCCL process-group and broadcast code (https://github.com/allenai/open-instruct/pull/1515).
- Extend pre-commit hook to also ban `nonlocal` keyword (https://github.com/allenai/open-instruct/pull/1613).
- Set checkpoint_state_freq default in data_loader.py, not mason.py (https://github.com/allenai/open-instruct/pull/1600).
- Inline data prep actor naming in `StreamingDataLoader` and GRPO, removing redundant helpers and parameter plumbing (https://github.com/allenai/open-instruct/pull/1326).
- Use local fixture for AceCode test instead of downloading from HuggingFace (https://github.com/allenai/open-instruct/pull/1593).
- Now, to disable `max_grad_norm` clipping, set None, not -1 (https://github.com/allenai/open-instruct/pull/1591).
- Inline GRPO utility functions and rename `ExperimentConfig` to `GRPOExperimentConfig` (https://github.com/allenai/open-instruct/pull/1578).
- Extract shared OLMo-core config classes and helpers into `olmo_core_utils.py`; refactor DPO to use shared configs (https://github.com/allenai/open-instruct/pull/1576).
- Decouple `mix_data.py` from `finetune.py` by replacing `FlatArguments` import with a lightweight `MixDataArguments` dataclass (https://github.com/allenai/open-instruct/pull/1573).
- Extracted shared `find_free_port` utility function (https://github.com/allenai/open-instruct/pull/1607).

### Deprecated

### Removed

### Fixed
