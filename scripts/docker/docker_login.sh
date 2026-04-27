#!/bin/bash
# Authenticate with Docker Hub using Beaker secret.
# Source this before training: source scripts/train/debug/envs/docker_login.sh
#
# Writes ~/.docker/config.json directly so all processes (including Ray workers)
# pick up the credentials when creating docker.from_env() clients.

PODMAN_SOCKET_PATH=/tmp/podman.sock
export DOCKER_HOST="unix://${PODMAN_SOCKET_PATH}"
unset DOCKER_TLS_VERIFY
unset DOCKER_CERT_PATH

if [ -x /usr/local/bin/setup_dockerio_mirror ]; then
    /usr/local/bin/setup_dockerio_mirror
else
    echo "WARNING: /usr/local/bin/setup_dockerio_mirror not found; skipping Docker Hub mirror setup"
fi

if command -v podman >/dev/null 2>&1; then
    if [ ! -S "$PODMAN_SOCKET_PATH" ]; then
        rm -f "$PODMAN_SOCKET_PATH"
        podman system service --time=0 "$DOCKER_HOST" >/tmp/podman-system-service.log 2>&1 &
        export PODMAN_SYSTEM_SERVICE_PID=$!

        for _ in $(seq 1 50); do
            [ -S "$PODMAN_SOCKET_PATH" ] && break
            sleep 0.1
        done
    fi

    if [ -S "$PODMAN_SOCKET_PATH" ]; then
        echo "Docker SDK configured to use Podman socket at $DOCKER_HOST"
    else
        echo "WARNING: Podman socket was not created at $PODMAN_SOCKET_PATH"
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
