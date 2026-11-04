---
name: run-and-fix
description: Run the three GRPO (non-MILES) debug scripts on Beaker in order, fix failures, and add the runs to the PR body. Use when asked to run the GRPO debug scripts.
---

Commit all changes, then launch each of these with `./scripts/train/build_image_and_launch.sh <script>`, in order:

1. `scripts/train/debug/single_gpu_on_beaker.sh`
2. `scripts/train/debug/tools/olmo_3_parser_multigpu.sh`
3. `scripts/train/debug/large_test_script.sh`

Wait for each to finish successfully before starting the next one. Monitor the results and fix any errors.

If they pass, then update the PR with links to them in this format:

Runs:
1. Single GPU GRPO: [Beaker](link)
2. Single GPU GRPO with tools: [Beaker](link)
3. Multi-node GRPO: [Beaker](link)
