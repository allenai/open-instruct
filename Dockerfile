ARG BASE_IMAGE=ghcr.io/allenai/cuda:12.8-dev-ubuntu22.04-torch2.7.0-v1.2.170

FROM ${BASE_IMAGE}

# Set default cache directory but allow override from environment
ARG CACHE_DIR=/root/.cache/uv
ARG UV_CACHE_DIR
ENV UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE_DIR}
RUN echo "UV_CACHE_DIR: ${UV_CACHE_DIR}"

# setup files
WORKDIR /stage/

# Install nginx and create conf.d directory
RUN apt-get update --no-install-recommends && apt-get install -y nginx && mkdir -p /etc/nginx/conf.d && rm -rf /var/lib/apt/lists/*

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV UV_COMPILE_BYTECODE=0

# Copy only dependency-related files first
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --link-mode=copy

# rerun with extras
RUN uv sync --extra compile --extra liger

# punkt / punkt_tab
RUN uv run -m nltk.downloader punkt
RUN uv run -m nltk.downloader punkt_tab

WORKDIR /stage/

# Copy all runtime files directly to final stage
COPY open_instruct open_instruct
COPY eval eval
COPY configs configs
COPY scripts scripts
COPY oe-eval-internal oe-eval-internal
COPY mason.py mason.py
COPY .git/ ./.git/

# Set up the environment
ENV PATH=/stage/.venv/bin:$PATH
