# syntax=docker/dockerfile:1
# Keep the qualified binary toolchain and build kernels before copying application sources.
ARG BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks
FROM ${BASE_IMAGE} AS flash-attention-build
# Compile FA3 against the base's Torch/CUDA ABI. No GPU is needed; limit compiler
# parallelism so a cold build does not exhaust the build host's memory.
ARG FLASH_ATTENTION_MAX_JOBS=2
WORKDIR /tmp/flash-attention
RUN git init . \
 && git remote add origin https://github.com/Dao-AILab/flash-attention.git \
 && git fetch --depth=1 origin 1fc0221183e799363001394a12dd9e78e15f5dda \
 && git checkout --detach FETCH_HEAD \
 && git submodule update --init csrc/cutlass \
 && cd hopper \
 && FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE \
    FLASH_ATTENTION_DISABLE_FP8=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE \
    MAX_JOBS=${FLASH_ATTENTION_MAX_JOBS} NVCC_THREADS=2 \
    python setup.py bdist_wheel --dist-dir /wheels
# FA2 is prebuilt for Python 3.12 / Torch 2.13 / CUDA 13.0. FA4 must be
# reinstalled after FA2, which also ships files under flash_attn/cute.
ADD --checksum=sha256:ab12a5b17f33f0d3381044f70006fa5891ec18404b412cc3a7517e5f2315534e \
    https://wheels.astral.sh/artifacts/ab12a5b17f33f0d3381044f70006fa5891ec18404b412cc3a7517e5f2315534e/flash_attn-2.8.3.post1+cu.13.0.torch.2.13-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl /wheels/
ADD --checksum=sha256:c84fedab172419f87bd1daf019ecd268595f512a5967ff90b74a0a601f578e07 \
    https://files.pythonhosted.org/packages/73/b3/85de5e9e393f5fc8a83c27258ba21c683c396c5ad79f15713283a801407b/flash_attn_4-4.0.0b27-py3-none-any.whl /wheels/

FROM ${BASE_IMAGE} AS runtime-base
RUN --mount=type=bind,from=flash-attention-build,source=/wheels,target=/tmp/flash-attention-wheels \
    uv pip install --system --no-deps /tmp/flash-attention-wheels/flash_attn_3-*.whl \
 && uv pip install --system --no-deps /tmp/flash-attention-wheels/flash_attn-2.8.3.post1*.whl \
 && uv pip install --system --no-deps --reinstall /tmp/flash-attention-wheels/flash_attn_4-4.0.0b27-*.whl
# Importing the extensions catches missing libraries and Torch ABI mismatches
# without requiring a GPU; each submitted job also checks forward/backward passes.
RUN python -c "import flash_attn; import flash_attn_2_cuda; import flash_attn_interface; import flash_attn_3; import flash_attn.cute"
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
RUN uv pip install --system --no-deps beaker-py==2.5.7
RUN python -c "from beaker import Beaker"
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
ENV PYTHONPATH=/opt/core-rl/sources/olmo-sglang/src:/opt/core-rl:/opt/core-rl/sources/olmo-core/src:/opt/core-rl/sources/miles
WORKDIR /opt/core-rl
RUN python -m compileall -q /opt/core-rl/open_instruct/miles /opt/core-rl/sources/olmo-core/src/olmo_core /opt/core-rl/sources/miles/miles
RUN python -c "from open_instruct.miles.configuration.config import RunConfig"

RUN python -c "from open_instruct.miles.rewards.verifiers import GSM8KVerifier; v = GSM8KVerifier(); assert v([], 'The answer is 42', '42').score == 1; assert v([], 'The answer is 43', '42').score == 0"

RUN python -c "import olmo_sglang; assert olmo_sglang.__file__.startswith('/opt/core-rl/sources/olmo-sglang/src/'); from olmo_sglang.attention import PerHeadRMSNorm"
