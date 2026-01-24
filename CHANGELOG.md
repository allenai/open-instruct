# Changelog

All notable changes to this project will be documented in this file.


### Added
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
- Updated library versions via `uv lock --upgrade` (https://github.com/allenai/open-instruct/pull/1400).
- Now, `large_test_script.sh` exercises the `tp > 1` code path (https://github.com/allenai/open-instruct/pull/1413).

### Fixed
- Increased `MetricsTracker` max_metrics from 64 to 512 to fix `ValueError: Exceeded maximum number of metrics` when training with many tools or verifier functions (https://github.com/allenai/open-instruct/pull/1415).
- Fixed JSON serialization error in `LocalDatasetTransformationCache.save_config` when caching datasets locally (https://github.com/allenai/open-instruct/pull/1402).
- Now, we can support PRs from external contributors while still maintaining security for internal tokens (https://github.com/allenai/open-instruct/pull/1408).
- Improved error handling for tool calls with missing/invalid arguments - now returns a clear error message instead of crashing (https://github.com/allenai/open-instruct/pull/1404).
- Fixed `GenerationConfig` validation error when saving OLMo-3 models - config is now set after unwrapping the model, and OLMo-3 is detected from both `chat_template_name` and model name (https://github.com/allenai/open-instruct/pull/1404).
- Fixed the benchmark so that it runs (https://github.com/allenai/open-instruct/pull/1401).

### Removed
- Removed `open_instruct/ppo.py` and related PPO training scripts (https://github.com/allenai/open-instruct/pull/1395).
- Removed `scripts/train/debug/tool_grpo_fast.sh`; use `scripts/train/debug/tools/olmo_3_parser_multigpu.sh` for tool use experiments (https://github.com/allenai/open-instruct/pull/1404).
