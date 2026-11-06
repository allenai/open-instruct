# syntax=docker/dockerfile:1
# Supply the locally available application image matching runtime/miles/runtime.lock.json.
# Build from the repository root with -f scripts/data/rl_readiness/Dockerfile.
# Supply --secret id=github_token,env=GH_TOKEN for the private code-execution client.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY scripts/data/rl_readiness/requirements.txt /tmp/rl-readiness-requirements.txt
RUN --mount=type=secret,id=github_token,required=true \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=credential.https://github.com.helper \
    GIT_CONFIG_VALUE_0='!f() { echo username=x-access-token; printf "password="; cat /run/secrets/github_token; echo; }; f' \
    python -m pip install --break-system-packages --no-cache-dir -r /tmp/rl-readiness-requirements.txt
COPY scripts/data/rl_readiness /opt/core-rl/scripts/data/rl_readiness
COPY scripts/data/rlvr /opt/core-rl/scripts/data/rlvr
WORKDIR /opt/core-rl
RUN python -m scripts.data.rl_readiness.qualify_bundle --help && \
    python -m open_instruct.miles.datasets.source_preparation --help && \
    python -m scripts.data.rl_readiness.qualify --help && \
    python -m scripts.data.rl_readiness.audit --help && \
    python -m scripts.data.rl_readiness.local_coverage --help
