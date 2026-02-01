# Changelog

All notable changes to this project will be documented in this file.


### Added
- Added generic RL environment support following the OpenEnv standard, with adapters for Prime Intellect verifiers (Wordle, Wiki-Search) and AppWorld. Environments integrate as tools with multi-turn reset/step/cleanup lifecycle. https://github.com/allenai/open-instruct/pull/1419
- Budget mode gradient checkpointing support for OLMo-core DPO training (https://github.com/allenai/open-instruct/pull/1444).
- PerfCallback for MFU metrics in OLMo-core DPO training (https://github.com/allenai/open-instruct/pull/1442).
- NVIDIA H200 GPU support in `GPU_SPECS` (https://github.com/allenai/open-instruct/pull/1441).
- Documentation and runtime warning for `dataset_mixer_list` format (float=proportion, int=count) (https://github.com/allenai/open-instruct/pull/1434).
- Added SLURM scripts for OLMo SFT training with checkpoint resume support and configurable shuffle seed. https://github.com/allenai/open-instruct/pull/1368
- Added retry logic with exponential backoff to `make_api_request` for tool API calls (retries on timeouts, connection errors, 429, and 5xx). Also added configurable `max_concurrency` parameter to tool configs for controlling Ray actor concurrency per-tool. https://github.com/allenai/open-instruct/pull/1388
- Added support for generic MCP tools during training, with some limitations (no changing tools, no tool discovery during training). For details: https://github.com/allenai/open-instruct/pull/1384
- Added the ability to set active tools on a per-sample basis. See the PR for more details: https://github.com/allenai/open-instruct/pull/1382
- Added a new changelog Github Action that makes sure you contribute to the changelog! https://github.com/allenai/open-instruct/pull/1276
- Now, we type check `open_instruct/dataset_transformation.py` (https://github.com/allenai/open-instruct/pull/1390).
- Added a linter rule that imports go at the top of the file (https://github.com/allenai/open-instruct/pull/1394).
- Refactors GRPO config into a grpo_utils.py file in preparation for Olmo-core implementation (https://github.com/allenai/open-instruct/pull/1396_.
- Now, we save the generated rollouts to disk during RL when the --save_traces flag is passed (https://github.com/allenai/open-instruct/pull/1406).
- Pulls out weight sync code from GRPO into a more generic function (https://github.com/allenai/open-instruct/pull/1411#pullrequestreview-3694117967)

### Changed

- Updated vllm version to 0.14.1 (https://github.com/allenai/open-instruct/pull/1433).
- Changed default wandb x-axis from `episode` to `training_step` for grpo_fast (https://github.com/allenai/open-instruct/pull/1437).

### Fixed
- Fixed evaluation responses being lost on timeout in grpo_fast by requeuing partial results (https://github.com/allenai/open-instruct/pull/1439).
- Beaker Experiment Launch now passes (https://github.com/allenai/open-instruct/pull/1424#pullrequestreview-3708034780).

### Removed
