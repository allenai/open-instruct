# Keep the qualified binary toolchain, then prepare a reusable source/dependency layer.
ARG BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks
FROM ${BASE_IMAGE} AS runtime-base
COPY runtime/miles /opt/core-rl/build/runtime/miles
RUN uv pip install --system --no-deps -r /opt/core-rl/build/runtime/miles/verifier-requirements.txt
RUN uv pip install --system --no-deps --target /opt/core-rl/verifier-deps antlr4-python3-runtime==4.11.1
ENV OPEN_INSTRUCT_MATH_VERIFIER_PYTHONPATH=/opt/core-rl/verifier-deps
COPY scripts/miles/prepare_runtime.py /opt/core-rl/build/scripts/miles/prepare_runtime.py
RUN --mount=type=secret,id=github_token,required=true \
    python /opt/core-rl/build/scripts/miles/prepare_runtime.py /opt/core-rl/sources \
    --github-token-file /run/secrets/github_token
COPY runtime/miles/runtime.lock.json /opt/core-rl/runtime/miles/runtime.lock.json

FROM runtime-base AS application
# The background evaluator submitter runs in an isolated, deadline-bounded process.
RUN uv pip install --system beaker-py==2.5.7
ARG SOURCE_REVISION
LABEL org.opencontainers.image.revision=${SOURCE_REVISION}
COPY open_instruct /opt/core-rl/open_instruct
COPY tests/miles /opt/core-rl/tests/miles
COPY scripts/miles /opt/core-rl/scripts/miles
COPY scripts/train/debug /opt/core-rl/scripts/train/debug
COPY runtime/miles/Dockerfile /opt/core-rl/runtime/miles/Dockerfile
COPY configs/miles /opt/core-rl/configs/miles
COPY docs/miles /opt/core-rl/docs/miles
COPY README.md /opt/core-rl/README.md
ENV PYTHONPATH=/opt/core-rl/sources/olmo-sglang/src:/opt/core-rl:/opt/core-rl/sources/olmo-core/src:/opt/core-rl/sources/miles
WORKDIR /opt/core-rl
RUN python -m compileall -q /opt/core-rl/open_instruct/miles /opt/core-rl/sources/olmo-core/src/olmo_core /opt/core-rl/sources/miles/miles
RUN python -c "from open_instruct.miles.config import RunConfig"
RUN python -c "from scripts.miles.prepare_hero_basket import prepare"

RUN python -c "from open_instruct.ground_truth_utils import GSM8KVerifier; v = GSM8KVerifier(); assert v([], 'The answer is 42', '42').score == 1; assert v([], 'The answer is 43', '42').score == 0"

RUN python -c "import olmo_sglang; assert olmo_sglang.__file__.startswith('/opt/core-rl/sources/olmo-sglang/src/'); from olmo_sglang.attention import PerHeadRMSNorm"
