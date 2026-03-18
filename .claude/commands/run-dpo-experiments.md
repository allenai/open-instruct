Run the following scripts, in order:

1. @scripts/train/debug/dpo/single_gpu.sh
2. @scripts/train/debug/dpo/multi_node.sh

Wait for the first to **succeed** before launching the second. Monitor the results and fix any errors. Only launch the multi-node experiment once single GPU has passed (exit code 0).

If they both pass, then update the PR with links to them in this format:

Runs:
1. Single GPU DPO: [Beaker](link)
2. Multi-node DPO: [Beaker](link)
