# Changelog

All notable changes to this project will be documented in this file.


### Added
- Enable the OLMo-core DPO path for Qwen3.5-9B (hybrid GatedDeltaNet): read `vocab_size`/`tie_word_embeddings` through `get_text_config()` in `setup_model` (VLM-wrapper configs), forward `tie_word_embeddings` to `qwen3_5*` presets, and add `open_instruct/qwen3_5_export.py` — the OLMo-core→HF inverse converter (GDN qkv/conv1d re-fusing, q/gate re-interleave, −1.0 norm offset) wired into `verify_can_save_as_hf`/`save_state_dict_as_hf`, with CPU round-trip tests against OLMo-core's own from-HF converter.
- Forward the `tools` column through the preference tokenization wrappers (both `preference_tulu_tokenize_and_truncate_v1` variants) so chosen/rejected render identical tool schemas; consume the column post-tokenization and bump `DATASET_CACHE_VERSION` to `v8`.
- Add `preference_safecomp_tail_tokenize_v1`: preference tokenization for prefix-matched multi-turn agentic pairs that labels exactly the divergent tail's assistant turns (via a new `label_from_message_idx` span filter) and drops pairs exceeding `max_seq_length` untruncated instead of cutting their closing turn.
- Normalize messages before chat templating (`_normalize_messages_for_chat_template`): deserialize JSON-string `tool_call.function.arguments` so templates that iterate `arguments|items` (e.g. Qwen3.5) render correctly.
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
- Raise a `ValueError` naming the packed instance count, global batch size, and epoch count when `olmo_core_finetune.py` computes fewer than one training step, instead of letting an undersized dataset surface as a bare `ZeroDivisionError` from olmo-core's LR scheduler (https://github.com/allenai/open-instruct/pull/1796).
- Track the CUDA 12 image suffix in the merge-queue Beaker workflow and allow enough time for the larger image build and upload (https://github.com/allenai/open-instruct/pull/1783).
- Exclude nested virtualenvs (e.g. `oe-eval-internal/.venv/`) from the Docker build context, so a uv venv inside a nested clone no longer fails the image build on a dangling host-interpreter symlink (https://github.com/allenai/open-instruct/pull/1786).
