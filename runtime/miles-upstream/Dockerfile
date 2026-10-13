# syntax=docker/dockerfile:1
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS runtime
ARG MILES_REVISION
ARG IMAGE_INPUTS
# Replace the editable checkout at the path expected by upstream's recipes.
# Keep the binary dependencies from the digest-pinned upstream image.
RUN rm -rf /root/miles && \
    git init /root/miles && \
    git -C /root/miles remote add origin https://github.com/radixark/miles.git && \
    git -C /root/miles fetch --depth=1 origin ${MILES_REVISION} && \
    git -C /root/miles checkout --detach FETCH_HEAD && \
    test "$(git -C /root/miles rev-parse HEAD)" = "${MILES_REVISION}" && \
    test "$(python /root/miles/docker/image_inputs.py)" = "${IMAGE_INPUTS}" && \
    pip install --no-deps -e /root/miles
COPY runtime.lock.json /opt/miles-upstream/runtime.lock.json
RUN python -c "from importlib import util; source = util.find_spec('miles.utils.arguments').origin; assert source == '/root/miles/miles/utils/arguments.py', source" && \
    pip freeze > /opt/miles-upstream/packages.txt && \
    git -C /root/Megatron-LM rev-parse HEAD > /opt/miles-upstream/megatron-revision && \
    git -C /sgl-workspace/sglang rev-parse HEAD > /opt/miles-upstream/sglang-revision
ARG BUILD_ID
LABEL org.opencontainers.image.source="https://github.com/radixark/miles" \
      org.opencontainers.image.revision=${MILES_REVISION} \
      ai2.miles-upstream.build-id=${BUILD_ID}
ENV MILES_UPSTREAM_BUILD_ID=${BUILD_ID}
WORKDIR /root/miles
ENTRYPOINT []
CMD ["/bin/bash"]

# Optional TOML/Beaker adapter. The runtime target above is usable on its own.
FROM runtime AS application
COPY adapter /opt/miles-upstream/adapter
ENV PYTHONPATH=/opt/miles-upstream/adapter:/root/Megatron-LM
# Parser imports load TransformerEngine. Use the bundled driver library only
# for this CPU metadata extraction; workers retain the upstream library path.
RUN LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH} \
    python -m scripts.miles_upstream.snapshot /opt/miles-upstream/schemas
