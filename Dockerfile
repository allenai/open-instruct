# Slim base: no gcloud/aws/vulkan/tools baked in
ARG BASE_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# Python (runtime only; the CUDA image doesn’t have it)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ca-certificates nginx \
  && rm -rf /var/lib/apt/lists/*

# uv
COPY --from=ghcr.io/astral-sh/uv:0.8.6 /uv /uvx /bin/

ENV HF_HUB_ENABLE_HF_TRANSFER=1 UV_COMPILE_BYTECODE=0
WORKDIR /stage/

# deps first
COPY pyproject.toml README.md build.py uv.lock ./
COPY open_instruct open_instruct

# Create the venv with your deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --link-mode=copy

# NLTK in one layer
RUN uv run -m nltk.downloader punkt punkt_tab

# app files
COPY eval eval
COPY configs configs
COPY scripts scripts
COPY mason.py mason.py
COPY oe-eval-internal oe-eval-internal

ENV PATH=/stage/.venv/bin:$PATH
