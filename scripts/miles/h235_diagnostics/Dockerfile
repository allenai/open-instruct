FROM sha256:ac0c8cdd6904ee11912571f11f47f13871572ca01ec3d291bb41c28708b94074
ARG SOURCE_REVISION
LABEL org.opencontainers.image.revision=$SOURCE_REVISION
COPY open_instruct/miles/infrastructure/coordinator_diagnostics.py /opt/core-rl/open_instruct/miles/infrastructure/coordinator_diagnostics.py
COPY scripts/miles/h235_diagnostics/miles.patch /opt/core-rl/h235-diagnostics.patch
RUN cd /opt/core-rl/sources/miles && git apply --check /opt/core-rl/h235-diagnostics.patch && git apply /opt/core-rl/h235-diagnostics.patch
RUN python -m compileall -q /opt/core-rl/open_instruct/miles/infrastructure/coordinator_diagnostics.py /opt/core-rl/sources/miles/miles/ray/rollout/rollout_executor.py /opt/core-rl/sources/miles/miles/ray/placement_group.py
RUN printf '%s\n' "$SOURCE_REVISION" > /opt/core-rl/diagnostic-source-revision
