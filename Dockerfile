FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

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
    UV_COMPILE_BYTECODE=0

# Install dependencies
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv run --frozen python -m nltk.downloader punkt punkt_tab words

# Hotfix for vllm-project/vllm#38574: layerwise.py warns "Failed to load weights"
# for zero-parameter modules and calls _place_kernel_tensors, corrupting model state.
# Remove once vllm>=0.19.1 is released.
RUN LAYERWISE=$(find /stage/.venv -path '*/model_loader/reload/layerwise.py' 2>/dev/null | head -1) && \
    if [ -n "$LAYERWISE" ]; then \
    sed -i '/place kernel tensors back as a fallback/{n;s/            else:/            elif info.load_numel_total > 0:  # type: ignore[operator]/;}' "$LAYERWISE"; \
    fi

# Weight validation: log NaN/Inf stats after finalize_layerwise_reload in gpu_worker.py
RUN GPU_WORKER=$(find /stage/.venv -path '*/v1/worker/gpu_worker.py' 2>/dev/null | head -1) && \
    if [ -n "$GPU_WORKER" ]; then \
    sed -i '/finalize_layerwise_reload(model, self.model_config)/a\
            import logging as _logging\
            _wv_logger = _logging.getLogger("weight_validation")\
            _bad = []\
            for _n, _p in model.named_parameters():\
                if _p.isnan().any():\
                    _bad.append((_n, "NaN", list(_p.shape)))\
                elif _p.isinf().any():\
                    _bad.append((_n, "Inf", list(_p.shape)))\
            if _bad:\
                _wv_logger.error("WEIGHT VALIDATION FAILED after layerwise reload: %s", _bad[:10])\
            else:\
                _wv_logger.warning("WEIGHT VALIDATION OK: all %d params clean after layerwise reload", sum(1 for _ in model.named_parameters()))' "$GPU_WORKER"; \
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
