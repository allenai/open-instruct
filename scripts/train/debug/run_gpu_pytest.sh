#!/bin/bash
source configs/beaker_configs/ray_node_setup.sh
uv run pytest open_instruct/test_grpo_fast_gpu.py -xvs
