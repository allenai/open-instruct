# Beaker 01M38JPC4965R1PFTPMBTPWRHP, the actual H025/H030 CUDA 13 image.
FROM gcr.io/ai2-beaker-core/public/daq8brsgm01s73aj5t30
ARG GIT_COMMIT
ENV GIT_COMMIT=${GIT_COMMIT} GIT_BRANCH=robertb/atomic-think-sft
# Keep all historical libraries and kernels; replace only tracked project code.
COPY open_instruct /stage/open_instruct
COPY scripts /stage/scripts
COPY pyproject.toml uv.lock /stage/
