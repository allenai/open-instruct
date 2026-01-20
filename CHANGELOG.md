# Changelog

All notable changes to this project will be documented in this file.


### Added
- Added support for generic MCP tools during training, with some limitations (no changing tools, no tool discovery during training). For details: https://github.com/allenai/open-instruct/pull/1384
- Added the ability to set active tools on a per-sample basis. See the PR for more details: https://github.com/allenai/open-instruct/pull/1382
- Added a new changelog Github Action that makes sure you contribute to the changelog! https://github.com/allenai/open-instruct/pull/1276
- Now, we type check `open_instruct/dataset_transformation.py` (https://github.com/allenai/open-instruct/pull/1390).
- Added a linter rule that imports go at the top of the file (https://github.com/allenai/open-instruct/pull/1394).

### Changed

### Fixed

### Removed
