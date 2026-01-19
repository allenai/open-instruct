# Changelog

All notable changes to this project will be documented in this file.


### Added
- Added retry logic with exponential backoff to `make_api_request` for tool API calls (retries on timeouts, connection errors, 429, and 5xx). Also added configurable `max_concurrency` parameter to tool configs for controlling Ray actor concurrency per-tool. https://github.com/allenai/open-instruct/pull/1388
- Added a new changelog Github Action that makes sure you contribute to the changelog! https://github.com/allenai/open-instruct/pull/1276

### Changed

### Fixed

### Removed
