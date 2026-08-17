ARG CUDA_VERSION=12
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS cuda12
# SPIKE: 13.3.1 rather than 13.0.3, and the -cudnn- variant. transformer-engine's
# prebuilt cu13 wheel needs cublasLtGroupedMatrixLayoutInit_internal, absent from
# 13.0.3's libcublasLt (13.1.1.3), and its bindings need cudnn.h, absent from the
# plain devel image. Revert both if TE is ever dropped.
FROM nvidia/cuda:13.3.1-cudnn-devel-ubuntu22.04 AS cuda13
FROM cuda${CUDA_VERSION}

ARG CUDA_VERSION

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles" \
    LANG=en_US.UTF-8

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

ENV DOFED_VER=2.10.0 \
    OS_VER=ubuntu2204
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

ENV UV_CACHE_DIR=/root/.cache/uv \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    UV_COMPILE_BYTECODE=0 \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPEN_INSTRUCT=0.0.0+docker

# Install dependencies
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv run --frozen --no-default-groups --group dev --group cuda${CUDA_VERSION} \
        python -m nltk.downloader punkt punkt_tab words

# MoE v2's token permutation hard-requires transformer_engine's kernels
# (olmo_core/nn/moe/utils.py guards the import, olmo_core/nn/moe/v2/no_ep.py calls
# them unconditionally). Only the CUDA 13 image gets it: that's where the MoE spike
# runs, and the torch-binding sdist compiles against the venv's torch, so
# --no-build-isolation with the build tools preinstalled. transformer_engine_cu13
# itself is a prebuilt wheel; only the bindings compile here.
# No pyproject/uv.lock mounts here: the project's extra-build-dependencies
# config (torch, match-runtime, for flash-attn) would otherwise apply to these
# unrelated installs and fail them. uv pip targets /stage/.venv directly.
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    if [ "$CUDA_VERSION" = "13" ]; then \
        uv pip install cmake ninja pybind11 setuptools wheel && \
        CUDNN_DIR=$(/stage/.venv/bin/python -c "import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0])") && \
        CUDNN_PATH="$CUDNN_DIR" \
        CPLUS_INCLUDE_PATH="$CUDNN_DIR/include" \
        C_INCLUDE_PATH="$CUDNN_DIR/include" \
        LIBRARY_PATH="$CUDNN_DIR/lib" \
        MAX_JOBS=8 uv pip install --no-build-isolation "transformer-engine[pytorch]==2.16.1"; \
    fi

# Separate COPY commands required: Docker copies directory *contents*, not the directory itself
COPY configs configs
COPY scripts scripts
COPY mason.py mason.py
COPY open_instruct open_instruct
COPY oe-eval-interna[l] oe-eval-internal/

ARG GIT_COMMIT="" \
    GIT_BRANCH=""

ENV GIT_COMMIT=${GIT_COMMIT} \
    GIT_BRANCH=${GIT_BRANCH} \
    PATH=/stage/.venv/bin:$PATH
