#!/bin/bash
set -eo pipefail

uv run pytest open_instruct/test_olmo_core_callbacks_gpu.py::TestPerfCallbackMFU::test_mfu_with_different_padding -v -s
