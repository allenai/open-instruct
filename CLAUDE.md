# Bash commands
- `uv run pytest`: Run the tests.
- `make style && make quality` run the linter + formatter.

# Workflow
- Always run the linter and make sure the tests pass before finishing a task.
- Prefer running single tests, not the whole suite, when developing.
- Launch tool use experiments by running ./scripts/train/build_image_and_launch.sh scripts/train/debug/tool_grpo_fast.sh
- Launch multi-node non-tool experiments by running ./scripts/train/build_image_and_launch.sh scripts/train/debug/large_test_script.sh