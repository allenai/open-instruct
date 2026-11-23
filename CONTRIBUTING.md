# Contributing to Open Instruct

Thank you for your interest in contributing to Open Instruct!

## Adding Olmo-core models

For our new infrastructure, which is based on [Olmo-core](https://github.com/allenai/OLMo-core), we need to add models in manually to convert them from Huggingface. You don't need to merge the PR to `olmo-core` (although we encourage it!) as you can modify `pyproject.toml` to use a specific commit of `olmo-core` (or a fork).

Here are some example PRs adding models: [Qwen3](https://github.com/allenai/OLMo-core/pull/533), [Gemma 3](https://github.com/allenai/OLMo-core/pull/534).

Once you have modified `pyproject.toml` to point to the specific commit, run `uv sync`, and then you should be able to run your experiment with the new model type.

## External contributors

### CI and GPU validation

Automated CI runs on GitHub-hosted runners and does not require Beaker credentials or cluster capacity. GPU tests and training integration runs are manual for both fork and internal PRs. When a change needs GPU validation, an Ai2 maintainer can run it on Beaker, or a contributor can run the relevant tests on an available GPU.

## Internal Contributors

Please name your branch `username/branch-description`. E.g. `finbarr/update-vllm-version`.

### Recording manual GPU tests

Launch the GPU suite with `./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh`. Record a successful run as `GPU_TESTS=[EXPERIMENT_ID](https://beaker.org/ex/EXPERIMENT_ID)` in the PR body so reviewers can inspect the result and tested revision. For local GPU runs, record the command, revision, hardware, and result instead.

`GPU_TESTS` is a reviewer-facing record, not a CI override. CI does not parse or verify it, and `GPU_TESTS=bypass` has no effect. Training smoke runs should be recorded separately from the GPU pytest suite.

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

**Unit tests**: `uv run pytest` discovers CPU tests in `open_instruct/`, `tests/`, and the repository root. CUDA-only files under `open_instruct/` are excluded when CUDA is unavailable.

Data-loader sampling/resume/accounting (`test_hf_data_loader.py`), Beaker callbacks (`test_olmo_core_callbacks.py`), and DPO collation/forward/gradient checks (`test_dpo_forward.py`) run in ordinary unit-test CI without depending on Beaker. The DPO tests initialize tiny HF and OLMo-core models locally and use a local tokenizer, without model downloads. To run this coverage on CPU:

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run pytest \
    open_instruct/test_hf_data_loader.py \
    open_instruct/test_olmo_core_callbacks.py \
    open_instruct/test_dpo_forward.py
```

These checks do not replace CUDA/BF16 kernels, FlashAttention packing, optimizer offload, real vLLM generation, or distributed weight-transfer validation.

**Linting and formatting**: `make style` formats code with ruff, `make quality` runs ruff lint, compileall, and the `ty` type checker. Both target `open_instruct/` and `*mason.py`.

**GPU tests**: The GPU test files live at `open_instruct/test_*_gpu.py` (5 files: DPO utils, GRPO CPUAdam offload, GRPO generation/packing, streaming data loader, and vLLM weight transfer). These require a GPU and are run via `uv run pytest open_instruct/test_*_gpu.py -xvs`. To run them on Beaker: `./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh`.

## CI Workflows

Automated checks run on GitHub-hosted runners:

| Workflow | Checks |
|---|---|
| `pr_checks.yml` | Formatting, lint, types, and the PR changelog entry. |
| `tests.yml` | CPU unit tests on PRs and merge groups, with a 20-minute timeout. |
| `miles.yml` | CPU MILES contracts, runtime/integration style, types, and generated references for relevant PRs. |
| `docs.yml` | Documentation validation and deployment. |
| `codex-review.yml`, `claude-review.yml` | Automated code reviews. |

No workflow launches, waits for, or verifies Beaker experiments. GPU pytest and the training smoke scripts remain available for manual validation; they are not merge-queue jobs.

CI also no longer publishes `open_instruct_auto`, `open_instruct_auto_cuda12`, or `open_instruct_auto_cuda13`. Existing images are not refreshed automatically. Build an image from the intended code revision with `scripts/train/build_image_and_launch.sh` when launching training or GPU tests.

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
