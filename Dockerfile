FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    make \
    sudo \
    nginx \
    && apt-get autoremove -y \
    && mkdir -p /etc/nginx/conf.d \
    && rm -rf /var/lib/apt/lists/*

# This ensures the dynamic linker (or NVIDIA's container runtime, I'm not sure)
# puts the right NVIDIA things in the right place (that THOR requires).
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Install DOCA OFED user-space drivers
# See https://docs.nvidia.com/doca/sdk/doca-host+installation+and+upgrade/index.html
# doca-ofed-userspace ver 2.10.0 depends on mft=4.31.0-149
ENV MFT_VER=4.31.0-149
RUN wget https://www.mellanox.com/downloads/MFT/mft-${MFT_VER}-x86_64-deb.tgz && \
    tar -xzf mft-${MFT_VER}-x86_64-deb.tgz && \
    mft-${MFT_VER}-x86_64-deb/install.sh --without-kernel && \
    rm mft-${MFT_VER}-x86_64-deb.tgz

ENV DOFED_VER=2.10.0
ENV OS_VER=ubuntu2204
RUN wget https://www.mellanox.com/downloads/DOCA/DOCA_v${DOFED_VER}/host/doca-host_${DOFED_VER}-093000-25.01-${OS_VER}_amd64.deb && \
    dpkg -i doca-host_${DOFED_VER}-093000-25.01-${OS_VER}_amd64.deb && \
    apt-get update && apt-get -y install --no-install-recommends doca-ofed-userspace && \
    apt-get autoremove -y && \
    rm doca-host_${DOFED_VER}-093000-25.01-${OS_VER}_amd64.deb

# Install Google Cloud CLI
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
        | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update -y && apt-get install -y --no-install-recommends google-cloud-sdk \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
# Taken from https://beaker.org/api/v3/release (add | jq -r '.version' if you want it programmatically).
ENV BEAKER_VERSION=v1.5.235
RUN curl --silent \
    --connect-timeout 5 \
    --max-time 10 \
    --retry 5 \
    --retry-delay 0 \
    --retry-max-time 40 \
    --output beaker.tar.gz \
    "https://beaker.org/api/v3/release/cli?os=linux&arch=amd64&version=${BEAKER_VERSION}" \
    && tar -zxf beaker.tar.gz -C /usr/local/bin/ ./beaker \
    && rm beaker.tar.gz

COPY --from=ghcr.io/astral-sh/uv:0.8.6 /uv /uvx /bin/

WORKDIR /stage/

ENV UV_CACHE_DIR=/root/.cache/uv
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV UV_COMPILE_BYTECODE=0

# Install dependencies
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-cache

RUN uv run --no-sync -m nltk.downloader punkt punkt_tab

# Copy all application code at the end
COPY eval eval
COPY configs configs
COPY scripts scripts
COPY mason.py mason.py
COPY oe-eval-internal oe-eval-internal
COPY open_instruct open_instruct

# Add build arguments for git information
ARG GIT_COMMIT=""
ARG GIT_BRANCH=""

# Set them as environment variables
ENV GIT_COMMIT=${GIT_COMMIT}
ENV GIT_BRANCH=${GIT_BRANCH}

# Set up the environment
ENV PATH=/stage/.venv/bin:$PATH
