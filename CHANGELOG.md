# Changelog

All notable changes to this project will be documented in this file.


### Added
- Budget mode gradient checkpointing support for OLMo-core DPO training (https://github.com/allenai/open-instruct/pull/1444).
- PerfCallback for MFU metrics in OLMo-core DPO training (https://github.com/allenai/open-instruct/pull/1442).
- NVIDIA H200 GPU support in `GPU_SPECS` (https://github.com/allenai/open-instruct/pull/1441).
- Documentation and runtime warning for `dataset_mixer_list` format (float=proportion, int=count) (https://github.com/allenai/open-instruct/pull/1434).

### Changed

- Updated vllm version to 0.14.1 (https://github.com/allenai/open-instruct/pull/1433).
- Changed default wandb x-axis from `episode` to `training_step` for grpo_fast (https://github.com/allenai/open-instruct/pull/1437).

### Fixed
- Fixed evaluation responses being lost on timeout in grpo_fast by requeuing partial results (https://github.com/allenai/open-instruct/pull/1439).
- Beaker Experiment Launch now passes (https://github.com/allenai/open-instruct/pull/1424#pullrequestreview-3708034780).

### Removed
