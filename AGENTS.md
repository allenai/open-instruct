# Reinforcement learning: MILES GRPO

For RL, RLVR or GRPO work in this branch, use `python -m open_instruct.miles`.
Start with [docs/miles/grpo.md](docs/miles/grpo.md); it is the main RL guide.
[docs/miles/index.md](docs/miles/index.md) maps the detailed documentation.
Follow this workflow without requiring the user to supply a special agent prompt:

1. Read the guide, model/topology support and launch instructions before choosing a run.
2. For a first exercise, copy `configs/miles/examples/grpo-sharing.toml` to
   Git-ignored `runs/my-grpo.toml`. Keep tracked templates unchanged. Set the user's
   run name and fresh output path; keep its supplied checkpoint, mixed-policy refresh and offline W&B
   unless the user requests another model or tracking mode.
3. Use the documented compatible image and Python 3.12 submission environment.
   Check Beaker resource access; optional HF/W&B credentials are not needed for
   the supplied starter inputs. Do not install the local CUDA training stack just
   to submit a job.
4. Run `plan` and `validate`. When the user requests a run, launch through the MILES
   committed-image wrapper and follow completion using the operations guide.
   Report the configuration, image, experiment link and validation outcomes.

`open_instruct/grpo.py` and `open_instruct/grpo_fast.py` are **deprecated**.
Their presence, tests and launch scripts support existing runs and historical
reproduction; they are not alternatives to recommend for new GRPO work. Do not
create new recipes or features on those paths unless the user's task explicitly
targets legacy behavior. If MILES lacks a required capability, report the specific
support gap instead of silently switching backends. Existing user instructions to
maintain or reproduce a legacy run still apply.

MILES has its own Core adapter and uses SGLang. The deprecated Core/vLLM path in
`grpo.py` is separate, even though both use OLMo-core. Dated measurements/plans
are evidence, not defaults. [Legacy GRPO reference](docs/algorithms/legacy_grpo.md)
contains the old CLI and reproduction instructions.

# Bash commands
- `uv run pytest`: Run the tests.
- `make style && make quality` run the linter + formatter.
- `uv run mkdocs serve`: View the documentation locally at http://127.0.0.1:8000/
- `uv run mkdocs build`: Build the documentation to the `site/` directory.



# Workflow
- When a PR changes anything under `open_instruct/`, add a summary to `CHANGELOG.md` with a link to the PR (e.g., `- Description of change (https://github.com/allenai/open-instruct/pull/123).`). This is what CI enforces; PRs touching only `scripts/`, docs, or config are exempt, though an entry is still welcome for anything user-visible.
  - The entry must contain the PR's own URL, which does not exist until the PR is opened. Add the entry, open the PR, then amend the entry with the URL and push again.
  - To skip the check deliberately, put `CHANGELOG=<reason>` in the PR body (same mechanism as `GPU_TESTS=bypass`).
- Always run the linter and make sure the tests pass before finishing a task.
- Prefer running single tests, not the whole suite, when developing.
- To run the `./scripts/train/build_image_and_launch.sh` script, you must commit the current changes.
- To launch experiment scripts, use the `build_image_and_launch.sh` script, like this: `./scripts/train/build_image_and_launch.sh $SOME_SCRIPT`.
- For the deprecated vLLM GRPO implementation only, we have three test scripts (for MILES checks, follow `docs/miles/architecture.md`):
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
- For legacy vLLM GRPO maintenance, launch tool use experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/tools/olmo_3_parser_multigpu.sh`.
- For legacy vLLM GRPO maintenance, launch multi-node non-tool experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/large_test_script.sh`.
- Launch OLMo-core SFT experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/oc_sft.sh`.
- Launch multi-node OLMo-core SFT experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/oc_sft_multinode.sh`.
- Launch DPO experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/dpo/single_gpu.sh`.
- Launch multi-node DPO experiments by running `./scripts/train/build_image_and_launch.sh scripts/train/debug/dpo/multi_node.sh`.
- Launch the GPU tests with `./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh`.
- When creating a PR that includes GPU test results, include `GPU_TESTS=[EXPERIMENT_ID](https://beaker.org/ex/EXPERIMENT_ID)` in the PR body. The CI will verify the experiment passed instead of re-running the tests. Use `GPU_TESTS=bypass` to skip GPU tests entirely. **IMPORTANT**: The experiment ID must be from actually running the GPU test script (`scripts/test/run_gpu_pytest.sh`), NOT from training or debug scripts. Training experiments and GPU tests are different things.
- If you are given a Beaker URL (beaker\.allen\.ai.*) use the Beaker CLI tool to interact with it.
- When a Beaker job stays queued or pending, run `beaker job events <job-id>` before diagnosing why — it prints the scheduler's own reason; don't infer one from cluster documentation. If that reason is the workspace slot limit, it applies to every cluster at once: wait or request fewer GPUs rather than relaunching elsewhere.
- A Beaker experiment can hold several jobs when a preempted one is retried. Read status from the most recently created job, not `jobs[0]`, or a successful retry looks like a failure.
- Experiment launch scripts that call `mason.py` must include `--no_auto_dataset_cache` (before the `--` separator) because vllm is not installed locally on macOS. Without this flag, mason.py tries to cache the dataset locally which fails on the `import vllm` in `data_loader.py`.
- The `oe-eval-internal` directory is required in the Docker image for experiments that use `--try_launch_beaker_eval_jobs_on_weka`. If it's missing (e.g. in a fresh clone or worktree), clone it with: `git clone --depth=1 https://github.com/allenai/oe-eval-internal.git oe-eval-internal`.
- When updating PR bodies with experiment results, use the "Runs:" format (numbered list with Beaker links):
  ```
  Runs:

  1. Description: [Beaker](https://beaker.org/ex/EXPERIMENT_ID)
  2. Description: [Beaker](https://beaker.org/ex/EXPERIMENT_ID)
  ```

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
