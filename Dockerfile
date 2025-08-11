FROM ghcr.io/allenai/cuda:12.8-dev-ubuntu22.04-torch2.7.0-v1.2.170

COPY --from=ghcr.io/astral-sh/uv:0.8.6 /uv /uvx /bin/

WORKDIR /stage/

# Install nginx and create conf.d directory
RUN apt-get update --no-install-recommends && apt-get install -y nginx && mkdir -p /etc/nginx/conf.d && rm -rf /var/lib/apt/lists/*

# TODO When updating flash-attn or torch in the future, make sure to update the version in the requirements.txt file. 
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
RUN pip install packaging --no-cache-dir
RUN pip install flash-attn==2.8.0.post2 flashinfer-python==0.2.8 --no-build-isolation --no-cache-dir
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Copy only dependency-related files first
COPY pyproject.toml build.py README.md uv.lock ./

# Annoyingly, we need this before `uv run`, or it complains.
COPY open_instruct open_instruct

# Install dependencies
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --link-mode=copy

RUN uv run -m nltk.downloader punkt punkt_tab

COPY oe-eval-internal oe-eval-internal

# install the package in editable mode
COPY pyproject.toml README.md build.py .
RUN pip install -e .
COPY .git/ ./.git/
COPY eval eval
COPY configs configs
COPY scripts scripts
COPY mason.py mason.py
RUN chmod +x scripts/*

# Set up the environment
ENV PATH=/stage/.venv/bin:$PATH