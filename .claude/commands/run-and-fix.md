Please run, in order, the following scripts:

1. @scripts/train/debug/single_gpu_on_beaker.sh
2. @scripts/train/debug/tools/olmo_3_parser_multigpu.sh
3. @scripts/train/debug/large_test_script.sh

Wait for each to finish successfully before starting the next one. Monitor the results and fix any errors.

If they pass, then update the PR with links to them in this format:

Runs:
1. Single GPU GRPO: [Beaker](link)
2. Single GPU GRPO with tools: [Beaker](link)
3. Multi-node GRPO: [Beaker](link)
