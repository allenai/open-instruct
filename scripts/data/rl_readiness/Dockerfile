# Supply the locally available application image matching runtime/miles/runtime.lock.json.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY requirements.txt /tmp/rl-readiness-requirements.txt
RUN python -m pip install --break-system-packages --no-cache-dir -r /tmp/rl-readiness-requirements.txt
