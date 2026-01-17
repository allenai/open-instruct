Please run, in order, the following scripts:

1. @scripts/train/debug/single_gpu_on_beaker.sh
2. @scripts/train/debug/tool_grpo_fast.sh
3. @scripts/train/debug/large_test_script.sh

Wait for each to finish successfully before starting the next one. Monitor the results and fix any errors.

If they pass, then update the PR with links to them in this format:

Ran single GPU GRPO ([Beaker](link)), single GPU GRPO with tools ([Beaker](link)), and multi-node GRPO ([Beaker](link)) scripts.
