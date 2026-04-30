#!/bin/bash
# Authenticate with Docker Hub using Beaker secret.
# Source this before training: source scripts/train/debug/envs/docker_login.sh
#
# Writes ~/.docker/config.json directly so all processes (including Ray workers)
# pick up the credentials when creating docker.from_env() clients.

PODMAN_SOCKET_PATH=/tmp/podman.sock
PODMAN_LOG_DIR=/output/tmp
PODMAN_SERVICE_LOG="${PODMAN_LOG_DIR}/podman-system-service.log"
PODMAN_INFO_LOG="${PODMAN_LOG_DIR}/podman-info.log"
export DOCKER_HOST="unix://${PODMAN_SOCKET_PATH}"
unset DOCKER_TLS_VERIFY
unset DOCKER_CERT_PATH

mkdir -p "$PODMAN_LOG_DIR"

if [ -x /usr/local/bin/setup_dockerio_mirror ]; then
    /usr/local/bin/setup_dockerio_mirror
else
    echo "WARNING: /usr/local/bin/setup_dockerio_mirror not found; skipping Docker Hub mirror setup"
fi

if command -v podman >/dev/null 2>&1; then
    echo "Podman found at $(command -v podman)"
    podman --version || true
    podman info --debug >"$PODMAN_INFO_LOG" 2>&1 || echo "WARNING: podman info failed; details in $PODMAN_INFO_LOG"

    if [ ! -S "$PODMAN_SOCKET_PATH" ]; then
        ulimit -n 1048576 || true
        ulimit -u 1048576 || true
        echo "Starting Podman service for Docker SDK: podman system service --time=0 $DOCKER_HOST"
        rm -f "$PODMAN_SOCKET_PATH"
        : >"$PODMAN_SERVICE_LOG"
        podman system service --time=0 "$DOCKER_HOST" >"$PODMAN_SERVICE_LOG" 2>&1 &
        export PODMAN_SYSTEM_SERVICE_PID=$!
        echo "Podman service PID: $PODMAN_SYSTEM_SERVICE_PID"

        for attempt in $(seq 1 50); do
            [ -S "$PODMAN_SOCKET_PATH" ] && break
            if ! kill -0 "$PODMAN_SYSTEM_SERVICE_PID" >/dev/null 2>&1; then
                echo "WARNING: Podman service process exited before socket appeared"
                wait "$PODMAN_SYSTEM_SERVICE_PID"
                echo "Podman service exit code: $?"
                break
            fi
            if [ "$attempt" = "10" ] || [ "$attempt" = "25" ] || [ "$attempt" = "50" ]; then
                echo "Waiting for Podman socket at $PODMAN_SOCKET_PATH (attempt $attempt/50)"
            fi
            sleep 0.1
        done
    else
        echo "Podman socket already exists at $PODMAN_SOCKET_PATH"
    fi

    if [ -S "$PODMAN_SOCKET_PATH" ]; then
        echo "Docker SDK configured to use Podman socket at $DOCKER_HOST"
    else
        echo "WARNING: Podman socket was not created at $PODMAN_SOCKET_PATH"
        if [ -e "$PODMAN_SOCKET_PATH" ]; then
            echo "A non-socket file exists at $PODMAN_SOCKET_PATH:"
            ls -l "$PODMAN_SOCKET_PATH" || true
        fi
        if [ -s "$PODMAN_SERVICE_LOG" ]; then
            echo "Podman service log ($PODMAN_SERVICE_LOG):"
            sed -n '1,200p' "$PODMAN_SERVICE_LOG"
        else
            echo "Podman service log is empty at $PODMAN_SERVICE_LOG"
        fi
        if [ -s "$PODMAN_INFO_LOG" ]; then
            echo "Podman info log ($PODMAN_INFO_LOG):"
            sed -n '1,200p' "$PODMAN_INFO_LOG"
        fi
        echo "Podman failed with 'cannot clone: Operation not permitted'; this usually means the Beaker task was not launched with subcontainer permissions. Is BEAKER_ALLOW_SUBCONTAINERS=1 and BEAKER_SKIP_DOCKER_SOCKET=1 set?"
    fi
else
    echo "WARNING: podman not found; Docker SDK will use $DOCKER_HOST if available"
fi

if [ -n "$DOCKER_PAT" ]; then
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
