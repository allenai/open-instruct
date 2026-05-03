#!/bin/bash
# Authenticate with Docker Hub using Beaker secret.
# Source this before training: source scripts/train/debug/envs/docker_login.sh
#
# Writes ~/.docker/config.json directly so all processes (including Ray workers)
# pick up the credentials when creating docker.from_env() clients.

PODMAN_SOCKET_PATH=${PODMAN_SOCKET_PATH:-/tmp/podman.sock}
PODMAN_LOG_DIR=/output/tmp
PODMAN_SERVICE_LOG="${PODMAN_LOG_DIR}/podman-system-service.log"
PODMAN_INFO_LOG="${PODMAN_LOG_DIR}/podman-info.log"
SWERL_PODMAN_SERVICE_COUNT="${SWERL_PODMAN_SERVICE_COUNT:-1}"
SWERL_PODMAN_SERVICE_DIR="${SWERL_PODMAN_SERVICE_DIR:-/tmp/podman-services}"
SWERL_PODMAN_GRAPHROOT_BASE="${SWERL_PODMAN_GRAPHROOT_BASE:-/var/lib/containers/storage/swerl-podman-shards}"
SWERL_PODMAN_RUNROOT_BASE="${SWERL_PODMAN_RUNROOT_BASE:-/run/containers/storage/swerl-podman-shards}"
SWERL_PODMAN_TMPDIR_BASE="${SWERL_PODMAN_TMPDIR_BASE:-/var/tmp/swerl-podman-shards}"
SWERL_PODMAN_IMAGE_JANITOR_ENABLED="${SWERL_PODMAN_IMAGE_JANITOR_ENABLED:-0}"
SWERL_PODMAN_IMAGE_JANITOR_INTERVAL_S="${SWERL_PODMAN_IMAGE_JANITOR_INTERVAL_S:-300}"
SWERL_PODMAN_IMAGE_JANITOR_UNTIL="${SWERL_PODMAN_IMAGE_JANITOR_UNTIL:-30m}"
unset DOCKER_TLS_VERIFY
unset DOCKER_CERT_PATH

mkdir -p "$PODMAN_LOG_DIR"

if ! [[ "$SWERL_PODMAN_SERVICE_COUNT" =~ ^[0-9]+$ ]] || [ "$SWERL_PODMAN_SERVICE_COUNT" -lt 1 ]; then
    echo "WARNING: Invalid SWERL_PODMAN_SERVICE_COUNT=${SWERL_PODMAN_SERVICE_COUNT}; using 1"
    SWERL_PODMAN_SERVICE_COUNT=1
fi

declare -a PODMAN_SHARD_SOCKET_PATHS=()
declare -a PODMAN_SHARD_DOCKER_HOSTS=()
declare -a PODMAN_SHARD_ROOTS=()
declare -a PODMAN_SHARD_RUNROOTS=()
declare -a PODMAN_SHARD_TMPDIRS=()
declare -a PODMAN_SHARD_LOGS=()

for ((shard = 0; shard < SWERL_PODMAN_SERVICE_COUNT; shard++)); do
    if [ "$SWERL_PODMAN_SERVICE_COUNT" -eq 1 ]; then
        socket_path="$PODMAN_SOCKET_PATH"
        service_log="$PODMAN_SERVICE_LOG"
        graphroot=""
        runroot=""
        tmpdir=""
    else
        shard_dir="${SWERL_PODMAN_SERVICE_DIR}/${shard}"
        socket_path="${shard_dir}/podman.sock"
        service_log="${shard_dir}/podman-system-service.log"
        graphroot="${SWERL_PODMAN_GRAPHROOT_BASE}/${shard}"
        runroot="${SWERL_PODMAN_RUNROOT_BASE}/${shard}"
        tmpdir="${SWERL_PODMAN_TMPDIR_BASE}/${shard}"
        mkdir -p "$graphroot" "$runroot" "$tmpdir"
    fi
    mkdir -p "$(dirname "$socket_path")" "$(dirname "$service_log")"
    PODMAN_SHARD_SOCKET_PATHS[$shard]="$socket_path"
    PODMAN_SHARD_DOCKER_HOSTS[$shard]="unix://${socket_path}"
    PODMAN_SHARD_ROOTS[$shard]="$graphroot"
    PODMAN_SHARD_RUNROOTS[$shard]="$runroot"
    PODMAN_SHARD_TMPDIRS[$shard]="$tmpdir"
    PODMAN_SHARD_LOGS[$shard]="$service_log"
done

export DOCKER_HOST="${PODMAN_SHARD_DOCKER_HOSTS[0]}"
export SWERL_PODMAN_DOCKER_HOSTS
SWERL_PODMAN_DOCKER_HOSTS="$(IFS=,; echo "${PODMAN_SHARD_DOCKER_HOSTS[*]}")"

podman_for_shard() {
    local shard="$1"
    shift
    if [ -n "${PODMAN_SHARD_ROOTS[$shard]}" ]; then
        podman \
            --root "${PODMAN_SHARD_ROOTS[$shard]}" \
            --runroot "${PODMAN_SHARD_RUNROOTS[$shard]}" \
            --tmpdir "${PODMAN_SHARD_TMPDIRS[$shard]}" \
            "$@"
    else
        podman "$@"
    fi
}

if [ -x /usr/local/bin/setup_dockerio_mirror ]; then
    /usr/local/bin/setup_dockerio_mirror
else
    echo "WARNING: /usr/local/bin/setup_dockerio_mirror not found; skipping Docker Hub mirror setup"
fi

if command -v podman >/dev/null 2>&1; then
    echo "Podman found at $(command -v podman)"
    podman --version || true
    podman info --debug >"$PODMAN_INFO_LOG" 2>&1 || echo "WARNING: podman info failed; details in $PODMAN_INFO_LOG"

    ulimit -n 1048576 || true
    ulimit -u 1048576 || true
    podman_service_pids=()
    for ((shard = 0; shard < SWERL_PODMAN_SERVICE_COUNT; shard++)); do
        socket_path="${PODMAN_SHARD_SOCKET_PATHS[$shard]}"
        docker_host="${PODMAN_SHARD_DOCKER_HOSTS[$shard]}"
        service_log="${PODMAN_SHARD_LOGS[$shard]}"
        if [ ! -S "$socket_path" ]; then
            echo "Starting Podman service shard ${shard}/${SWERL_PODMAN_SERVICE_COUNT} for Docker SDK: ${docker_host}"
            echo "  socket=${socket_path}"
            if [ -n "${PODMAN_SHARD_ROOTS[$shard]}" ]; then
                echo "  root=${PODMAN_SHARD_ROOTS[$shard]} runroot=${PODMAN_SHARD_RUNROOTS[$shard]} tmpdir=${PODMAN_SHARD_TMPDIRS[$shard]}"
            fi
            rm -f "$socket_path"
            : >"$service_log"
            podman_for_shard "$shard" system service --time=0 "$docker_host" >"$service_log" 2>&1 &
            service_pid=$!
            podman_service_pids+=("$service_pid")
            if [ "$shard" -eq 0 ]; then
                export PODMAN_SYSTEM_SERVICE_PID="$service_pid"
            fi
            echo "Podman service shard $shard PID: $service_pid"

            for attempt in $(seq 1 50); do
                [ -S "$socket_path" ] && break
                if ! kill -0 "$service_pid" >/dev/null 2>&1; then
                    echo "WARNING: Podman service shard $shard exited before socket appeared"
                    wait "$service_pid"
                    echo "Podman service shard $shard exit code: $?"
                    break
                fi
                if [ "$attempt" = "10" ] || [ "$attempt" = "25" ] || [ "$attempt" = "50" ]; then
                    echo "Waiting for Podman socket at $socket_path (attempt $attempt/50)"
                fi
                sleep 0.1
            done
        else
            echo "Podman socket already exists at $socket_path"
        fi
    done
    export PODMAN_SYSTEM_SERVICE_PIDS="${podman_service_pids[*]}"

    missing_sockets=0
    for ((shard = 0; shard < SWERL_PODMAN_SERVICE_COUNT; shard++)); do
        if [ ! -S "${PODMAN_SHARD_SOCKET_PATHS[$shard]}" ]; then
            missing_sockets=1
        fi
    done

    if [ "$missing_sockets" = "0" ]; then
        echo "Docker SDK configured to use Podman socket at $DOCKER_HOST"
        echo "SWERL Podman Docker hosts: $SWERL_PODMAN_DOCKER_HOSTS"
        janitor_enabled="${SWERL_DOCKER_JANITOR_ENABLED:-}"
        if [ -z "$janitor_enabled" ] && [ "${SWERL_DOCKER_AUTO_REMOVE:-1}" = "0" ]; then
            janitor_enabled=1
        fi
        if [ "${janitor_enabled:-0}" = "1" ]; then
            JANITOR_LOG="${PODMAN_LOG_DIR}/podman-janitor.log"
            JANITOR_INTERVAL="${SWERL_DOCKER_JANITOR_INTERVAL_S:-60}"
            JANITOR_BATCH="${SWERL_DOCKER_JANITOR_BATCH_SIZE:-20}"
            echo "Starting Podman janitor (interval=${JANITOR_INTERVAL}s, batch=${JANITOR_BATCH})"
            (
                while true; do
                    for ((janitor_shard = 0; janitor_shard < SWERL_PODMAN_SERVICE_COUNT; janitor_shard++)); do
                        mapfile -t exited_ids < <(
                            podman_for_shard "$janitor_shard" ps -aq \
                                --filter status=exited \
                                --filter label=open_instruct=swerl_sandbox 2>>"$JANITOR_LOG" \
                                | head -n "$JANITOR_BATCH"
                        )
                        if [ "${#exited_ids[@]}" -gt 0 ]; then
                            echo "$(date -Is) shard=${janitor_shard} removing ${#exited_ids[@]} exited sandbox containers" >>"$JANITOR_LOG"
                        fi
                        for cid in "${exited_ids[@]}"; do
                            podman_for_shard "$janitor_shard" rm "$cid" >>"$JANITOR_LOG" 2>&1 || true
                        done
                    done
                    sleep "$JANITOR_INTERVAL"
                done
            ) &
            export PODMAN_JANITOR_PID=$!
            echo "Podman janitor PID: $PODMAN_JANITOR_PID"
        fi
        if [ "$SWERL_PODMAN_IMAGE_JANITOR_ENABLED" = "1" ]; then
            IMAGE_JANITOR_LOG="${PODMAN_LOG_DIR}/podman-image-janitor.log"
            IMAGE_JANITOR_INTERVAL="$SWERL_PODMAN_IMAGE_JANITOR_INTERVAL_S"
            IMAGE_JANITOR_UNTIL="$SWERL_PODMAN_IMAGE_JANITOR_UNTIL"
            echo "Starting Podman image janitor (interval=${IMAGE_JANITOR_INTERVAL}s, until=${IMAGE_JANITOR_UNTIL})"
            (
                while true; do
                    {
                        echo "$(date -Is) filesystem usage before image prune"
                        df -h /var/lib/containers /var/tmp /output 2>/dev/null || true
                    } >>"$IMAGE_JANITOR_LOG"
                    for ((image_janitor_shard = 0; image_janitor_shard < SWERL_PODMAN_SERVICE_COUNT; image_janitor_shard++)); do
                        {
                            echo "$(date -Is) shard=${image_janitor_shard} podman system df before image prune"
                            podman_for_shard "$image_janitor_shard" system df || true
                            echo "$(date -Is) shard=${image_janitor_shard} pruning unused images older than ${IMAGE_JANITOR_UNTIL}"
                            podman_for_shard "$image_janitor_shard" image prune -a --force --filter "until=${IMAGE_JANITOR_UNTIL}" || true
                            echo "$(date -Is) shard=${image_janitor_shard} podman system df after image prune"
                            podman_for_shard "$image_janitor_shard" system df || true
                        } >>"$IMAGE_JANITOR_LOG" 2>&1
                    done
                    sleep "$IMAGE_JANITOR_INTERVAL"
                done
            ) &
            export PODMAN_IMAGE_JANITOR_PID=$!
            echo "Podman image janitor PID: $PODMAN_IMAGE_JANITOR_PID; log: $IMAGE_JANITOR_LOG"
        fi
    else
        echo "WARNING: One or more Podman sockets were not created"
        for ((shard = 0; shard < SWERL_PODMAN_SERVICE_COUNT; shard++)); do
            socket_path="${PODMAN_SHARD_SOCKET_PATHS[$shard]}"
            service_log="${PODMAN_SHARD_LOGS[$shard]}"
            if [ -S "$socket_path" ]; then
                continue
            fi
            echo "Missing Podman socket for shard $shard at $socket_path"
            if [ -e "$socket_path" ]; then
                echo "A non-socket file exists at $socket_path:"
                ls -l "$socket_path" || true
            fi
            if [ -s "$service_log" ]; then
                echo "Podman service shard $shard log ($service_log):"
                sed -n '1,200p' "$service_log"
            else
                echo "Podman service shard $shard log is empty at $service_log"
            fi
        done
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
