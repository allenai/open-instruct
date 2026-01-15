Launch and monitor the following GRPO scripts:

1. @scripts/train/debug/single_gpu_on_beaker.sh (single GPU)
2. @scripts/train/debug/tool_grpo_fast.sh (single GPU with tools)
3. @scripts/train/debug/large_test_script.sh (multi-node)

Run them sequentially. Wait for each to finish before starting the next. Monitor the results and fix any errors.

If they pass, then update the PR with links to them in this format:

Ran single GPU GRPO ([Beaker](link)), single GPU GRPO with tools ([Beaker](link)), and multi-node GRPO ([Beaker](link)) scripts.
