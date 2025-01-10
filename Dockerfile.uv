FROM ghcr.io/allenai/oi-cuda-no-conda:12.1-cudnn8-dev-ubuntu20.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV UV_COMPILE_BYTECODE=1

# setup files
WORKDIR /stage/

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --extra compile

COPY open_instruct open_instruct
COPY eval eval
COPY configs configs
COPY scripts scripts
# hack: only copy oe-eval-internal if it exists
COPY mason.py oe-eval-internal* /stage/
COPY .git/ ./.git/
COPY pyproject.toml uv.lock .
RUN chmod +x scripts/*

# install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra compile

# for interactive session
RUN chmod -R 777 /stage/

# uv dockerfile magic: place executables in the environment at the front of the path
ENV PATH=/stage/.venv/bin:$PATH
