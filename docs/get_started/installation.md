# Installation

Our setup follows our [Dockerfile](https://github.com/allenai/open-instruct/blob/main/Dockerfile). *Note that Open Instruct is a research codebase and does not guarantee backward compatibility.*

## Local installation with uv

We use [uv](https://docs.astral.sh/uv/) for installation and for running code. Install uv, then:

```bash
uv sync
```

This creates a `.venv/` from `uv.lock` and installs the `dev` and `cuda12` dependency groups by
default. The project requires Python 3.12; uv downloads and manages that interpreter for you, so
you do not need it installed system-wide.

To build against CUDA 13 instead of CUDA 12:

```bash
uv sync --no-default-groups --group dev --group cuda13
```

The `cuda12` and `cuda13` groups are mutually exclusive — they pin different builds of torch, vLLM,
and flash-attention — so select exactly one.

Run commands through uv rather than activating the venv:

```bash
uv run python open_instruct/finetune.py --help
```

### Git LFS

Test fixtures under `open_instruct/test_data/` are stored in [Git LFS](https://git-lfs.com/).
Install it and run `git lfs install` before cloning, otherwise those files are checked out as
pointer text and the tests that read them fail. If you already cloned, run `git lfs pull`. See
[CONTRIBUTING.md](https://github.com/allenai/open-instruct/blob/main/CONTRIBUTING.md#git-lfs).

### Platform support

The dependency set resolves for Linux (`x86_64` and `aarch64`) and macOS.

Training requires a CUDA GPU on Linux. macOS is only useful for editing, linting, and running part
of the test suite: `vllm`, `flash-attn`, `liger-kernel`, and `bitsandbytes` are all excluded on
Darwin, so any module that imports them cannot be collected. `uv run pytest` will report collection
errors for those files on macOS; that is expected, and CI runs the full suite on Linux.

### About `requirements.txt`

`requirements.txt` is generated from `uv.lock` by a pre-commit hook and exists for reference and
for tooling that expects it. It is not the supported installation path — use `uv sync`. Do not edit
it by hand; edit `pyproject.toml` and let the hook re-export it.

## Docker installation

You can also build a Docker image:

```bash
docker build . \
    --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) \
    --build-arg GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD) \
    -t open_instruct_dev
```

The image defaults to CUDA 12 (`nvidia/cuda:12.8.1-devel-ubuntu22.04`). Pass `CUDA_VERSION=13` to
build the CUDA 13 variant (`nvidia/cuda:13.0.3-devel-ubuntu22.04`) instead:

```bash
docker build . --build-arg CUDA_VERSION=13 -t open_instruct_dev
```

`GIT_COMMIT` and `GIT_BRANCH` are optional; they are recorded as environment variables in the image
so a running job can report which revision it was built from.

If you are internally at Ai2, you can create a Beaker image like this:

```bash
beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')
beaker image delete $beaker_user/open_instruct_dev
beaker image create open_instruct_dev -n open_instruct_dev -w ai2/$beaker_user
```

In practice most Ai2 experiments are launched with
`./scripts/train/build_image_and_launch.sh <script>`, which builds and uploads the image for you.
See [Ai2 internal setup](ai2_internal_setup.md).
