
# Installation

Our setup mostly follows our [Dockerfile](./Dockerfile), which uses Python 3.10. *Note that Open Instruct is a research codebase and does not guarantee backward compatibility.* We offer two installation strategies:

* **Local installation**: This is the recommended way to install Open Instruct. You can install the dependencies by running the following commands:
```bash
pip install --upgrade pip "setuptools<70.0.0" wheel
# TODO, unpin setuptools when this issue in flash attention is resolved
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install packaging
pip install flash-attn==2.7.2.post1 --no-build-isolation
pip install -r requirements.txt
pip install -e .
python -m nltk.downloader punkt
```

* **Local installation with uv (preview)**: We are experimenting with using [uv](https://docs.astral.sh/uv/). You can install via
```bash
uv sync
uv sync --extra compile # to install flash attention
```


* **Docker installation**: You can also use the Dockerfile to build a Docker image. You can build the image with the following command:

```bash
docker build . -t open_instruct_dev
# if you are interally at AI2, you can create an image like this:
beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')
beaker image delete $beaker_user/open_instruct_dev
beaker image create open_instruct_dev -n open_instruct_dev -w ai2/$beaker_user
```

Optionally you can build the base image with the following command:

```bash
docker build --build-arg CUDA=12.1.0 --build-arg TARGET=cudnn8-devel --build-arg DIST=ubuntu20.04 -f  Dockerfile.base . -t cuda-no-conda:12.1-cudnn8-dev-ubuntu20.04
```

* **Docker with uv**: You can also use the Dockerfile to build a Docker image with uv. You can build the image with the following command:

```bash
docker build -f Dockerfile.uv --build-arg UV_CACHE_DIR=$UV_CACHE_DIR -t open_instruct_dev_uv .
# if you are interally at AI2, you can create an image like this:
beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')
beaker image delete $beaker_user/open_instruct_dev_uv
beaker image create open_instruct_dev_uv -n open_instruct_dev_uv -w ai2/$beaker_user
```

If you are internally at AI2, you may launch experiments using our always-up-to-date auto-built image `nathanl/open_instruct_auto`.

## Running the local debug scripts outside Ai2

The training code itself does not require Beaker: integrations like auto-evaluation or saving to Beaker datasets only activate when the job runs on Beaker. On your own machine (or rented GPUs), you can train with the debug scripts directly, e.g. single-GPU DPO:

```bash
./scripts/train/debug/dpo/local.sh
```

A few defaults behave differently outside Ai2, all overridable with environment variables:

* Caches land next to your working directory instead of `/weka`: the DPO reference-logprobs cache goes to `local_reference_logprobs_cache/` (set `REFERENCE_LOGPROBS_CACHE_PATH` to move it), and dataset caches go to `local_dataset_cache/`.
* If you don't have a Weights & Biases API key, pass `--wandb_entity <anything>`; otherwise startup fails in `wandb.login()` because the client probes for the internal `ai2-llm` team even when tracking is off.
* To skip the local dataset-caching step when launching through `mason.py` (it imports `vllm`, which is not available on macOS), pass `--no_auto_dataset_cache`.
* On `push_to_hub`, the model is pushed to your own HuggingFace account unless you pass `--hf_entity`; you can also set `WANDB_ENTITY`/`WANDB_PROJECT` to control W&B.

Training still needs at least one GPU; the scripts above are smoke tests, not replacements for the Beaker-based multi-node orchestration.
