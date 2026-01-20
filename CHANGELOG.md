# Changelog

All notable changes to this project will be documented in this file.


### Added
- Added OLMo-core based DPO training module (`dpo.py`) using OLMo-core's TrainModule with HSDP support
- Added a new changelog Github Action that makes sure you contribute to the changelog! https://github.com/allenai/open-instruct/pull/1276

### Changed
- Moved `concatenated_forward` and `packing` config fields from DPOConfig to ModelConfig
- Removed duplicate config fields from ExperimentConfig that were already defined in parent classes

### Fixed

### Removed
