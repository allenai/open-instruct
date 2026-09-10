# Build as a source overlay on the existing qualified compiled runtime.
ARG BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks
FROM ${BASE_IMAGE}
COPY runtime/miles /opt/core-rl/build/runtime/miles
COPY scripts/miles/prepare_runtime.py /opt/core-rl/build/scripts/miles/prepare_runtime.py
RUN python /opt/core-rl/build/scripts/miles/prepare_runtime.py /opt/core-rl/sources
COPY open_instruct /opt/core-rl/open_instruct
COPY tests/miles /opt/core-rl/tests/miles
ENV PYTHONPATH=/opt/core-rl:/opt/core-rl/sources/olmo-core/src:/opt/core-rl/sources/miles
WORKDIR /opt/core-rl
RUN python -m compileall -q /opt/core-rl/open_instruct/miles /opt/core-rl/sources/olmo-core/src/olmo_core /opt/core-rl/sources/miles/miles
RUN python -c "from open_instruct.miles.config import RunConfig"
