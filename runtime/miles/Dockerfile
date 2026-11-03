# syntax=docker/dockerfile:1
# Binary dependencies, including FA2/FA3/FA4, are already in the locked base.
ARG BASE_IMAGE=open-instruct-miles-binary-base:20261027
FROM ${BASE_IMAGE} AS runtime-base
COPY runtime/miles/runtime.lock.json /opt/core-rl/build/runtime/miles/runtime.lock.json
COPY scripts/miles/prepare_runtime.py /opt/core-rl/build/scripts/miles/prepare_runtime.py
RUN --mount=type=secret,id=github_token,required=true \
    python /opt/core-rl/build/scripts/miles/prepare_runtime.py /opt/core-rl/sources \
    --github-token-file /run/secrets/github_token
RUN uv pip install --system --no-deps /opt/core-rl/sources/scaling-ladders/ladders/mainline/transformers_plugin
COPY runtime/miles/runtime.lock.json /opt/core-rl/runtime/miles/runtime.lock.json
# Install metadata/entrypoints; the existing source overlay does not load optional Rust extensions.
# SGLang and Megatron Bridge declare torch as a build dependency. Reuse the
# qualified binary stack for metadata installs (SGLang Rust modules are disabled).
RUN uv pip install --system "setuptools-scm>=8.0" "setuptools-rust>=1.11"
# Keep isolated-build downloads for the other packages out of the runtime image.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    SGLANG_BUILD_RUST_EXTS=none UV_LINK_MODE=copy uv pip install --system --no-deps \
    --no-build-isolation-package sglang --no-build-isolation-package megatron-bridge \
    -e /opt/core-rl/sources/olmo-core -e /opt/core-rl/sources/miles \
    -e /opt/core-rl/sources/olmo-sglang -e /opt/core-rl/sources/sglang/python \
    -e /opt/core-rl/sources/megatron-lm -e /opt/core-rl/sources/megatron-bridge
# The native OPD launcher explicitly prepends this historical Megatron location.
RUN mkdir -p /src && ln -sT /opt/core-rl/sources/megatron-lm /src/Megatron-LM

FROM runtime-base AS application
ARG SOURCE_REVISION
LABEL org.opencontainers.image.revision=${SOURCE_REVISION}
RUN printf '%s\n' "$SOURCE_REVISION" > /opt/core-rl/source-revision
COPY open_instruct /opt/core-rl/open_instruct
COPY tests/miles /opt/core-rl/tests/miles
COPY scripts/miles /opt/core-rl/scripts/miles
COPY runtime/miles/Dockerfile /opt/core-rl/runtime/miles/Dockerfile
COPY configs/miles /opt/core-rl/configs/miles
COPY docs/miles /opt/core-rl/docs/miles
COPY README.md /opt/core-rl/README.md
ENV PYTHONPATH=/opt/core-rl/sources/sglang/python:/opt/core-rl/sources/olmo-sglang/src:/opt/core-rl:/opt/core-rl/sources/olmo-core/src:/opt/core-rl/sources/miles:/opt/core-rl/sources/megatron-lm:/opt/core-rl/sources/megatron-bridge/src
WORKDIR /opt/core-rl
RUN python -m compileall -q /opt/core-rl/open_instruct/miles /opt/core-rl/sources/olmo-core/src/olmo_core /opt/core-rl/sources/miles/miles
RUN python -c "from open_instruct.miles.configuration.config import RunConfig"

RUN python -c "from open_instruct.miles.rewards.verifiers import GSM8KVerifier; v = GSM8KVerifier(); assert v([], 'The answer is 42', '42').score == 1; assert v([], 'The answer is 43', '42').score == 0"

RUN python -c "import olmo_sglang; assert olmo_sglang.__file__.startswith('/opt/core-rl/sources/olmo-sglang/src/'); from olmo_sglang.attention import PerHeadRMSNorm"

RUN python -c "import sglang; import sglang.srt.server_args; import sglang.srt.entrypoints.engine; from importlib.metadata import version; assert sglang.__file__.startswith('/opt/core-rl/sources/sglang/python/'); assert version('sglang-kernel') == '0.4.7'"

# Native serving imports must resolve before allocating GPU qualification runs.
RUN python -c "from sglang.srt.models import mainline_ladder; assert mainline_ladder.EntryClass.__name__ == 'MainlineLadderForCausalLM'"
