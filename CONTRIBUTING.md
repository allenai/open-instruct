# Contributing to Open Instruct

Thank you for your interest in contributing to Open Instruct!

## Adding Olmo-core models

For our new infrastructure, which is based on [Olmo-core](https://github.com/allenai/OLMo-core), we need to add models in manually to convert them from Huggingface. You don't need to merge the PR to `olmo-core` (although we encourage it!) as you can modify `pyproject.toml` to use a specific commit of `olmo-core` (or a fork).

Here are some example PRs adding models: [Qwen3](https://github.com/allenai/OLMo-core/pull/533), [Gemma 3](https://github.com/allenai/OLMo-core/pull/534).

Once you have modified `pyproject.toml` to point to the specific commit, run `uv sync`, and then you should be able to run your experiment with the new model type.

## External contributors

### CI (Fork PRs)

When you submit a pull request from a fork, some CI checks behave differently due to GitHub's security restrictions on secrets:

#### GPU Tests

GPU tests require access to Beaker (our internal compute platform) and are **automatically skipped** for fork PRs. You'll see a message like:

```
Skipping GPU tests for fork PR
This PR is from a fork, and secrets are not available.
GPU tests will run automatically when this PR enters the merge queue.
```

This is expected behavior. A maintainer will manually run the GPU tests.

## Internal Contributors

Please name your branch `username/branch-description`. E.g. `finbarr/update-vllm-version`.

### GPU_TESTS Override (Internal PRs Only)

For internal PRs, you can skip running GPU tests by providing a link to an existing successful Beaker experiment in your PR description. This is useful when you've already run the tests locally or want to reuse results from a previous run. The format is `GPU_TESTS=[EXPERIMENT_ID](https://beaker.org/ex/EXPERIMENT_ID)`.

You can launch the GPU tests manually with `./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh`.

### GPU_TESTS Bypass (Internal PRs Only)

For changes that don't affect GPU functionality (e.g., documentation, CI config, minor refactors), you can bypass GPU tests entirely by adding to your PR description:

```
GPU_TESTS=bypass
```

**Warning**: Use this sparingly. Only bypass GPU tests when you are confident the changes cannot affect GPU-related code paths. When in doubt, let the tests run.

## Git LFS

This repository uses [Git LFS](https://git-lfs.com/) to store large test data files. The tracked files are defined in `.gitattributes`. Run `git lfs ls-files` to see the current set.

### Setup

1. Install Git LFS: https://git-lfs.com/
2. Run `git lfs install` (one-time setup per machine).
3. Clone the repository normally. LFS files are fetched automatically.

If you cloned the repository before installing Git LFS, you will have pointer files instead of the actual data. To fix this, first install the Git LFS client, then run `git lfs install` followed by `git lfs pull` in your repository to download the data.

### Adding new test data to LFS

Track new large or binary test data files with `git lfs track "path/to/file"` and commit the updated `.gitattributes`. CI is configured to fetch LFS objects automatically, so no workflow changes are needed.

## Running Tests

**Unit tests**: `uv run pytest` runs the tests in `tests/` (test_environments.py, test_generic_sandbox.py, test_merge_models.py).

**Linting and formatting**: `make style` formats code with ruff, `make quality` runs ruff lint, compileall, and the `ty` type checker. Both target `open_instruct/` and `*mason.py`.

**GPU tests**: The GPU test files live at `open_instruct/test_*_gpu.py` (5 files: data loader, DPO utils, GRPO fast, streaming data loader, OLMo-core callbacks). These require a GPU and are run via `uv run pytest open_instruct/test_*_gpu.py -xvs`. To run them on Beaker: `./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh`.

## CI Workflows

Four GitHub Actions workflows run on PRs:

1. **PR Checks** (`pr_checks.yml`): Runs `make style-check` and `make quality-check`. Also verifies that `CHANGELOG.md` was updated for changes to `open_instruct/` (bypass with `CHANGELOG=` in PR body).

2. **Unit Tests** (`tests.yml` → `unit-tests` job): Runs `uv run pytest` on an Ubuntu runner. 20-minute timeout.

3. **GPU Tests** (`tests.yml` → `gpu-tests` job): Builds a Docker image, uploads it to Beaker, and runs `open_instruct/test_*_gpu.py` on a single GPU. 45-minute timeout. Auto-skipped for fork PRs (no Beaker secrets). Can be overridden with `GPU_TESTS=[EXPERIMENT_ID]` or bypassed with `GPU_TESTS=bypass` in the PR body.

4. **Integration Tests** (`beaker-experiment.yml`): Runs in the merge queue (not on every PR push). Launches up to 3 Beaker experiments:
   - GRPO integration test (always runs)
   - DPO integration test (runs if DPO-related files changed)
   - SFT integration test (runs if `finetune.py` changed)

   Sends a Slack notification on failure.

## Launching Experiments on Beaker

All Beaker experiments are launched via `./scripts/train/build_image_and_launch.sh <script>`. This script:
- Requires a clean git working tree (no uncommitted changes)
- Builds a Docker image tagged with the current git branch and commit hash
- Caches images to avoid rebuilding for the same commit
- Passes the Beaker image name to the target script

Example: `./scripts/train/build_image_and_launch.sh scripts/train/debug/single_gpu_on_beaker.sh`

## GRPO Test Scripts

These are the main GRPO debug/test scripts. Use these to verify GRPO changes work end-to-end.

| Script | Hardware | Description | Runtime | Time to first step | Example |
|--------|----------|-------------|---------|--------------------|---------|
| `scripts/train/debug/grpo_fast.sh` | 1 GPU local | Minimal local test with Qwen3-0.6B, no tools | Fast | Unknown | Local only |
| `scripts/train/debug/grpo_fast_3_gpu.sh` | 3 GPUs local | Tests sequence parallelism (2 training + 1 inference) | Fast | Unknown | Local only |
| `scripts/train/debug/single_gpu_on_beaker.sh` | 1 GPU Beaker | Single GPU on Beaker, no tools, GSM8K dataset | ~4 min | ~2 min | [01KHC0ZX…](https://beaker.org/ex/01KHC0ZXVCNVGWM6QJSNFWN09R) |
| `scripts/train/debug/large_test_script.sh` | 2x8 GPUs Beaker | Multi-node with Qwen2.5-7B, DeepSpeed stage 3, seq parallelism | ~12 min | ~4 min | [01KK24AS…](https://beaker.org/ex/01KK24ASY65M27Q2P7RA85SA0Q) |
| `scripts/train/debug/tools/olmo_3_parser_multigpu.sh` | 2x8 GPUs Beaker | Multi-node with tool use (python, serper, jina), OLMo-3 model | ~10 min | ~4 min | [01KFEZBX…](https://beaker.org/ex/01KFEZBXWHDQ6PG6KZ63316M8J) |
| `scripts/train/debug/tools/tool_regression_beaker.sh` | 1 GPU Beaker | Tool use regression test with Qwen3-1.7B, hermes parser | ~4 min | ~3 min | [01KJE7T8…](https://beaker.org/ex/01KJE7T8N7S0Q1R55ACTJCJAQY) |

To launch any Beaker script: `./scripts/train/build_image_and_launch.sh <script_path>`

## DPO Test Scripts

| Script | Hardware | Description | Runtime | Time to first step | Example |
|--------|----------|-------------|---------|--------------------|---------|
| `scripts/train/debug/dpo/local.sh` | 1 GPU local | Local single-GPU DPO with OLMo-2-1B, no Beaker needed | Fast | Unknown | Local only |
| `scripts/train/debug/dpo/single_gpu.sh` | 1 GPU Beaker | Single GPU on Beaker with OLMo-2-1B | ~2 min | ~1 min | [01KHEJMG…](https://beaker.org/ex/01KHEJMG3HJ1MP6S3K0KMRGB1M) |
| `scripts/train/debug/dpo/multi_node.sh` | 2x8 GPUs Beaker | Multi-node DPO with OLMo-2-7B, FSDP + tensor parallelism | ~9 min | ~4 min | [01KH9RZD…](https://beaker.org/ex/01KH9RZD11EPNFEVJPZWT1A32G) |
| `scripts/train/debug/dpo/multi_node_cache.sh` | 2x8 GPUs Beaker | Multi-node cache-based DPO (`dpo_tune_cache.py`) with Qwen3-0.6B | ~2 min | ~1 min | [01KJX7JH…](https://beaker.org/ex/01KJX7JHZJETY6J20T5V02CD0T) |
| `scripts/train/debug/dpo/checkpoint_integration_test.sh` | 2x8 GPUs Beaker | Two-part test: trains, then resumes from checkpoint to verify checkpointing works | ~2 min | ~1 min | [01KH4TQA…](https://beaker.org/ex/01KH4TQA2F3JJV06081G3QT0FG) |

## Environment Variables

We set several environment variables for NCCL and vLLM to work around known issues and tune performance for our infrastructure.

### `NCCL_CUMEM_ENABLE=0` (set in Python source)

Disables NCCL's CUDA unified memory allocator. This works around a performance regression documented in [vllm-project/vllm#5723](https://github.com/vllm-project/vllm/issues/5723). It must be set before any NCCL imports take effect, which is why it's set via `os.environ` at the top of `grpo_fast.py`, `dpo_tune_cache.py`, `finetune.py`, and `utils.py` (before the `# isort: off` block).

### Default Beaker environment variables (set in `mason.py`)

These are injected into every Beaker experiment:

| Variable | Value | Why |
|----------|-------|-----|
| `VLLM_DISABLE_COMPILE_CACHE` | `1` | Torch compile caching is consistently broken in our setup, though compilation itself works fine |
| `VLLM_USE_V1` | `1` | Use the vLLM v1 engine (default for new work) |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `1` | Required for certain model serialization paths |
| `VLLM_ATTENTION_BACKEND` | `FLASH_ATTN` | Default vLLM attention backend; override via `--env VLLM_ATTENTION_BACKEND=...` or `--vllm_attention_backend ...` |
| `VLLM_LOGGING_LEVEL` | `WARNING` | Reduce vLLM log verbosity |
| `NCCL_DEBUG` | `ERROR` | Minimal NCCL logging (set to `INFO` or `WARN` when debugging communication issues) |
| `RAY_CGRAPH_get_timeout` | `300` | 5-minute timeout for Ray computation graph operations |
