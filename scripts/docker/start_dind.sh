#!/bin/bash
# Start a local Docker daemon inside the Beaker task and point Docker SDK clients at it.
#
# This is intentionally separate from docker_login.sh, which starts Podman services.
# Source this before Ray starts so Ray worker actors inherit DOCKER_HOST.

DIND_DIR="${DIND_DIR:-/tmp/open-instruct-dind}"
DIND_SOCKET="${DIND_SOCKET:-${DIND_DIR}/docker.sock}"
DIND_DATA_ROOT="${DIND_DATA_ROOT:-/var/lib/docker-dind}"
DIND_EXEC_ROOT="${DIND_EXEC_ROOT:-/run/docker-dind}"
DIND_LOG_DIR="${DIND_LOG_DIR:-/output/tmp}"
DIND_LOG="${DIND_LOG:-${DIND_LOG_DIR}/dockerd.log}"
DIND_STORAGE_DRIVER="${DIND_STORAGE_DRIVER:-vfs}"
DIND_DISABLE_BRIDGE="${DIND_DISABLE_BRIDGE:-1}"
DIND_SMOKE_TEST="${DIND_SMOKE_TEST:-1}"
DIND_SMOKE_IMAGE="${DIND_SMOKE_IMAGE:-python:3.12-slim}"

unset DOCKER_TLS_VERIFY
unset DOCKER_CERT_PATH

mkdir -p "$DIND_DIR" "$DIND_DATA_ROOT" "$DIND_EXEC_ROOT" "$DIND_LOG_DIR"

if [ -x /usr/local/bin/setup_dockerio_mirror ]; then
    /usr/local/bin/setup_dockerio_mirror || true
else
    echo "WARNING: /usr/local/bin/setup_dockerio_mirror not found; skipping Docker Hub mirror setup"
fi

if [ -n "${DOCKER_PAT:-}" ]; then
    python -c "
import base64, json, os
username = os.environ.get('DOCKERHUB_USERNAME', 'hamishivi')
auth = base64.b64encode(f'{username}:'.encode() + os.environ['DOCKER_PAT'].encode()).decode()
config_dir = os.path.expanduser('~/.docker')
os.makedirs(config_dir, exist_ok=True)
config_path = os.path.join(config_dir, 'config.json')
config = {}
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
config.setdefault('auths', {})['https://index.docker.io/v1/'] = {'auth': auth}
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print('Docker Hub credentials written to', config_path)
"
else
    echo "WARNING: DOCKER_PAT not set, skipping Docker Hub login (pulls will be rate-limited)"
fi

if ! command -v dockerd >/dev/null 2>&1; then
    echo "ERROR: dockerd not found in PATH"
    return 1 2>/dev/null || exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker CLI not found in PATH"
    return 1 2>/dev/null || exit 1
fi

export DOCKER_HOST="unix://${DIND_SOCKET}"
# Reuse the existing EnvironmentPool fanout hook. With one entry, all sandbox
# actors get an explicit docker_host pointed at the local DinD socket.
export SWERL_PODMAN_DOCKER_HOSTS="${DOCKER_HOST}"

if [ -S "$DIND_SOCKET" ] && docker info >/dev/null 2>&1; then
    echo "Docker daemon already running at ${DOCKER_HOST}"
else
    rm -f "$DIND_SOCKET"

    dockerd_args=(
        --host "$DOCKER_HOST"
        --data-root "$DIND_DATA_ROOT"
        --exec-root "$DIND_EXEC_ROOT"
        --pidfile "$DIND_DIR/docker.pid"
        --storage-driver "$DIND_STORAGE_DRIVER"
    )

    if [ -n "${MIRROR_URL:-}" ]; then
        dockerd_args+=(
            --registry-mirror "http://${MIRROR_URL}"
            --insecure-registry "${MIRROR_URL}"
        )
    fi

    if [ "$DIND_DISABLE_BRIDGE" = "1" ]; then
        dockerd_args+=(
            --bridge=none
            --iptables=false
            --ip-forward=false
            --ip-masq=false
        )
    fi

    echo "Starting dockerd for Docker SDK at ${DOCKER_HOST}"
    echo "dockerd data-root=${DIND_DATA_ROOT} exec-root=${DIND_EXEC_ROOT} storage-driver=${DIND_STORAGE_DRIVER}"
    : >"$DIND_LOG"
    dockerd "${dockerd_args[@]}" >"$DIND_LOG" 2>&1 &
    export DIND_DOCKERD_PID=$!
    echo "dockerd PID: ${DIND_DOCKERD_PID}; log: ${DIND_LOG}"

    for attempt in $(seq 1 120); do
        if docker info >/dev/null 2>&1; then
            break
        fi
        if ! kill -0 "$DIND_DOCKERD_PID" >/dev/null 2>&1; then
            echo "ERROR: dockerd exited before becoming ready"
            if [ -s "$DIND_LOG" ]; then
                sed -n '1,240p' "$DIND_LOG"
            fi
            return 1 2>/dev/null || exit 1
        fi
        if [ "$attempt" = "10" ] || [ "$attempt" = "30" ] || [ "$attempt" = "60" ] || [ "$attempt" = "120" ]; then
            echo "Waiting for dockerd at ${DOCKER_HOST} (attempt ${attempt}/120)"
        fi
        sleep 0.5
    done
fi

if ! docker info; then
    echo "ERROR: docker info failed for ${DOCKER_HOST}"
    if [ -s "$DIND_LOG" ]; then
        sed -n '1,240p' "$DIND_LOG"
    fi
    return 1 2>/dev/null || exit 1
fi

if [ "$DIND_SMOKE_TEST" = "1" ]; then
    echo "Running DinD smoke test with ${DIND_SMOKE_IMAGE}"
    if ! docker run --rm "$DIND_SMOKE_IMAGE" python -c 'print("dind ok")'; then
        echo "ERROR: DinD smoke test failed"
        if [ -s "$DIND_LOG" ]; then
            sed -n '1,240p' "$DIND_LOG"
        fi
        return 1 2>/dev/null || exit 1
    fi
fi

echo "Docker SDK configured to use DinD at ${DOCKER_HOST}"
