# Bash commands
- `uv run pytest`: Run the tests.
- `make style && make quality` run the linter + formatter.
- `uv run mkdocs serve`: View the documentation locally at http://127.0.0.1:8000/
- `uv run mkdocs build`: Build the documentation to the `site/` directory.

# Workflow
- Always run the linter and make sure the tests pass before finishing a task.
- Prefer running single tests, not the whole suite, when developing.
- To run the `./scripts/train/build_image_and_launch.sh` script, you must commit the current changes.
- Launch tool use experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/tool_grpo_fast.sh`.
- Launch multi-node non-tool experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/large_test_script.sh`.
- Launch the GPU tests with `./scripts/train/build_image_and_launch.sh scripts/train/debug/run_gpu_tests.sh`.

# OLMo-core GRPO Training Scripts
These scripts use `open_instruct/grpo.py` with OLMo-core's training infrastructure (FSDP instead of DeepSpeed):

- `./scripts/train/build_image_and_launch.sh scripts/train/debug/single_gpu_grpo.sh`: Single GPU test with math reasoning (GSM8K).
- `./scripts/train/build_image_and_launch.sh scripts/train/debug/tool_grpo.sh`: Single GPU test with tool use (code + search).
- `./scripts/train/build_image_and_launch.sh scripts/train/debug/multi_node_grpo.sh`: Multi-node (2 nodes, 8 GPUs each) test with code reasoning.

# Documentation
To verify that documentation changes don't alter the generated output:
1. Build docs on your branch: `uv run mkdocs build && cp -r site site-branch`
2. Switch to main branch and build: `cd /path/to/main && uv run mkdocs build`
3. Compare the builds: `diff -rq site-branch /path/to/main/site`
4. If no output, the docs are identical. If differences exist, review with: `diff -r site-branch /path/to/main/site`
