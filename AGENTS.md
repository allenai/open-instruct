# Bash commands
- `uv run pytest`: Run the tests.
- `make style && make quality` run the linter + formatter.
- `uv run mkdocs serve`: View the documentation locally at http://127.0.0.1:8000/
- `uv run mkdocs build`: Build the documentation to the `site/` directory.



# Workflow
- When creating a PR, always add a summary to `CHANGELOG.md` with a link to the PR (e.g., `- Description of change (https://github.com/allenai/open-instruct/pull/123).`).
- Always run the linter and make sure the tests pass before finishing a task.
- Prefer running single tests, not the whole suite, when developing.
- To run the `./scripts/train/build_image_and_launch.sh` script, you must commit the current changes.
- To launch experiment scripts, use the `build_image_and_launch.sh` script, like this: `./scripts/train/build_image_and_launch.sh $SOME_SCRIPT`.
- For GRPO, we have three test scripts:
  - `scripts/train/debug/single_gpu_on_beaker.sh`: single GPU, no tools (~8 minutes).
  - `scripts/train/debug/tools/olmo_3_parser_multigpu.sh`: multi GPU, with tools.
  - `scripts/train/debug/large_test_script.sh`: two 8x GPU nodes, no tools (~32 minutes).
- For OLMo-core SFT, we have two test scripts:
  - `scripts/train/debug/oc_sft.sh`: single GPU on Beaker.
  - `scripts/train/debug/oc_sft_multinode.sh`: two 8x GPU nodes on Beaker.
- For DPO, we have three test scripts:
  - `scripts/train/debug/dpo/local.sh`: local single GPU (no Beaker).
  - `scripts/train/debug/dpo/single_gpu.sh`: single GPU on Beaker.
  - `scripts/train/debug/dpo/multi_node.sh`: two 8x GPU nodes on Beaker.
- To run the `./scripts/train/build_image_and_launch.sh` script, you must commit the current changes.
- Launch tool use experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/tools/olmo_3_parser_multigpu.sh`.
- Launch multi-node non-tool experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/large_test_script.sh`.
- Launch OLMo-core SFT experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/oc_sft.sh`.
- Launch multi-node OLMo-core SFT experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/oc_sft_multinode.sh`.
- Launch DPO experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/dpo/single_gpu.sh`.
- Launch multi-node DPO experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/dpo/multi_node.sh`.
- Launch the GPU tests with `./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh`.
- When creating a PR that includes GPU test results, include `GPU_TESTS=[EXPERIMENT_ID](https://beaker.org/ex/EXPERIMENT_ID)` in the PR body. The CI will verify the experiment passed instead of re-running the tests. Use `GPU_TESTS=bypass` to skip GPU tests entirely. **IMPORTANT**: The experiment ID must be from actually running the GPU test script (`scripts/test/run_gpu_pytest.sh`), NOT from training or debug scripts. Training experiments and GPU tests are different things.
- If you are given a Beaker URL (beaker\.allen\.ai.*) use the Beaker CLI tool to interact with it.
- Experiment launch scripts that call `mason.py` must include `--no_auto_dataset_cache` (before the `--` separator) because vllm is not installed locally on macOS. Without this flag, mason.py tries to cache the dataset locally which fails on the `import vllm` in `data_loader.py`.
- The `oe-eval-internal` directory is required in the Docker image for experiments that use `--try_launch_beaker_eval_jobs_on_weka`. If it's missing (e.g. in a fresh clone or worktree), clone it with: `git clone --depth=1 https://github.com/allenai/oe-eval-internal.git oe-eval-internal`.

# Naming conventions
- Models OLMo and OLMo 2 (versions <=2) use the "OLMo" capitalization style.
- Olmo 3, Olmo Hybrid, and later models use "Olmo" (standard proper noun capitalization).
- Note: "OLMo-core" refers to the software repository and keeps its original capitalization.

# Coding conventions
- Never use `import logging` or `logging.info()` directly. Always use `logger = logger_utils.setup_logger(__name__)` and `logger.info()`.
- Imports always go at the top of the file, never inline.
- Use `from package import module` instead of `import package.module`.

# Documentation
To verify that documentation changes don't alter the generated output:
1. Build docs on your branch: `uv run mkdocs build && cp -r site site-branch`
2. Switch to main branch and build: `cd /path/to/main && uv run mkdocs build`
3. Compare the builds: `diff -rq site-branch /path/to/main/site`
4. If no output, the docs are identical. If differences exist, review with: `diff -r site-branch /path/to/main/site`
