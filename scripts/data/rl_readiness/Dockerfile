# Supply the locally available application image matching runtime/miles/runtime.lock.json.
# Build from the repository root with -f scripts/data/rl_readiness/Dockerfile.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY scripts/data/rl_readiness/requirements.txt /tmp/rl-readiness-requirements.txt
RUN python -m pip install --break-system-packages --no-cache-dir -r /tmp/rl-readiness-requirements.txt
COPY scripts/data/rl_readiness /opt/core-rl/scripts/data/rl_readiness
WORKDIR /opt/core-rl
RUN python -m scripts.data.rl_readiness.qualify_bundle --help && \
    python -m open_instruct.miles.datasets.candidate_sources --help && \
    python -m open_instruct.miles screen --help && \
    python -m scripts.data.rl_readiness.qualify --help && \
    python -m scripts.data.rl_readiness.audit --help && \
    python -m scripts.data.rl_readiness.local_coverage --help
