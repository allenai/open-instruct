FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles"

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    make \
    sudo \
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
    apt-get update && apt-get -y install doca-ofed-userspace && \
    rm doca-host_${DOFED_VER}-093000-25.01-${OS_VER}_amd64.deb

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

# Install Beaker Gantry user-wide in an isolated venv
RUN uv tool install --no-cache-dir beaker-gantry

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

# Annoyingly, we need this before `uv run`, or it complains.
COPY open_instruct open_instruct

# Install dependencies
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

RUN uv run -m nltk.downloader punkt punkt_tab

WORKDIR /stage/

# Copy all runtime files directly to final stage
COPY eval eval
COPY configs configs
COPY scripts scripts
COPY oe-eval-internal oe-eval-internal
COPY mason.py mason.py
COPY .git/ ./.git/

# Set up the environment
ENV PATH=/stage/.venv/bin:$PATH