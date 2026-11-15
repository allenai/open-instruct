FROM gcr.io/ai2-beaker-core/public/dbrcnemrl4sc73aaif5g@sha256:8fdfb51182a716d3d288e246e85dab0c4f6a2e027aae1dffa7844c24596007f5
ENV H235_ENABLE_RESOURCE_DIAGNOSTICS=1
ARG SOURCE_REVISION
LABEL org.opencontainers.image.revision=$SOURCE_REVISION
COPY open_instruct/miles/infrastructure/coordinator_diagnostics.py /opt/core-rl/open_instruct/miles/infrastructure/coordinator_diagnostics.py
COPY open_instruct/miles/infrastructure/resource_watch.py /opt/core-rl/open_instruct/miles/infrastructure/resource_watch.py
COPY open_instruct/miles/execution/cluster.py /opt/core-rl/open_instruct/miles/execution/cluster.py
COPY scripts/miles/h235_diagnostics/miles.patch /opt/core-rl/h235-diagnostics.patch
RUN cd /opt/core-rl/sources/miles && git apply --check /opt/core-rl/h235-diagnostics.patch && git apply /opt/core-rl/h235-diagnostics.patch
RUN python -m compileall -q /opt/core-rl/open_instruct/miles/infrastructure/coordinator_diagnostics.py /opt/core-rl/sources/miles/miles/ray/rollout/rollout_executor.py /opt/core-rl/sources/miles/miles/ray/placement_group.py
RUN printf '%s\n' "$SOURCE_REVISION" > /opt/core-rl/diagnostic-source-revision
