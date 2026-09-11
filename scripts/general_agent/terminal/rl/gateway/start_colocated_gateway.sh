#!/bin/bash
# Run a LiteRegistry gateway inside the training job and point the sandbox backend at it.
#
# Source this on EVERY node before configs/beaker_configs/ray_node_setup.sh:
#   source scripts/general_agent/terminal/rl/gateway/start_colocated_gateway.sh && source configs/beaker_configs/ray_node_setup.sh && python ...
#
# The Beaker leader (Ray head) starts the gateway, supervised (restarted if it exits); all
# nodes export SWERL_GATEWAY_URL=http://<leader>:<port>, which GatewayBackend picks up when
# --tool_configs carries no gateway_url. The gateway finds the fleet's Redis through the head
# registry on weka, so a Redis restart on another host is followed automatically, and because
# the gateway shares the job's lifetime there is no separate gateway task to lose.
#
# Inputs (env):
#   LITEREGISTRY_REGISTRY      required, e.g. head+sqlite:///weka/oe-adapt-default/gfaria/podman_deployments/<deploy>/head.sqlite3
#   LITEREGISTRY_GATEWAY_VENV  venv with literegistry installed (image default /opt/literegistry-gateway-venv)
#   GATEWAY_PORT               default: derived from BEAKER_EXPERIMENT_ID into 20000-29999 (same on every node)
#   GATEWAY_WORKERS            default 8
#   GATEWAY_WAIT_S             seconds to wait for a healthy gateway before failing the job (default 600)
#   GATEWAY_MIN_REPLICAS       fail if the roster shows fewer podman replicas after the wait (default 1)

_gw_log() { echo "[start_colocated_gateway] $*"; }

if [ -z "${LITEREGISTRY_REGISTRY:-}" ]; then
    _gw_log "ERROR: LITEREGISTRY_REGISTRY is not set (e.g. head+sqlite:///weka/.../head.sqlite3)"
    return 1 2>/dev/null || exit 1
fi
LITEREGISTRY_GATEWAY_VENV="${LITEREGISTRY_GATEWAY_VENV:-/opt/literegistry-gateway-venv}"
if [ ! -x "${LITEREGISTRY_GATEWAY_VENV}/bin/python" ]; then
    _gw_log "ERROR: no python at ${LITEREGISTRY_GATEWAY_VENV}/bin/python (image built without the gateway venv?)"
    return 1 2>/dev/null || exit 1
fi

GATEWAY_WORKERS="${GATEWAY_WORKERS:-8}"
GATEWAY_WAIT_S="${GATEWAY_WAIT_S:-600}"
GATEWAY_MIN_REPLICAS="${GATEWAY_MIN_REPLICAS:-1}"
_gw_rank="${BEAKER_REPLICA_RANK:-0}"
_gw_leader="${BEAKER_LEADER_REPLICA_HOSTNAME:-$(hostname -f 2>/dev/null || hostname)}"
if [ -z "${GATEWAY_PORT:-}" ]; then
    # Deterministic per experiment so every replica computes the same URL without talking.
    GATEWAY_PORT=$("${LITEREGISTRY_GATEWAY_VENV}/bin/python" -c \
        "import hashlib,os; seed=os.environ.get('BEAKER_EXPERIMENT_ID') or os.environ.get('BEAKER_WORKLOAD_ID') or 'local'; print(20000 + int(hashlib.sha256(seed.encode()).hexdigest(), 16) % 10000)")
fi
export SWERL_GATEWAY_URL="http://${_gw_leader}:${GATEWAY_PORT}"
_gw_log "rank=${_gw_rank} leader=${_gw_leader} SWERL_GATEWAY_URL=${SWERL_GATEWAY_URL} registry=${LITEREGISTRY_REGISTRY}"

if [ "${_gw_rank}" = "0" ]; then
    if ! "${LITEREGISTRY_GATEWAY_VENV}/bin/python" -c "import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('0.0.0.0', ${GATEWAY_PORT})); s.close()" 2>/dev/null; then
        _gw_log "ERROR: port ${GATEWAY_PORT} is already in use on ${_gw_leader}; set GATEWAY_PORT to a free port"
        return 1 2>/dev/null || exit 1
    fi
    (
        while true; do
            "${LITEREGISTRY_GATEWAY_VENV}/bin/python" -m literegistry.gateway \
                --registry="${LITEREGISTRY_REGISTRY}" \
                --host=0.0.0.0 \
                --port="${GATEWAY_PORT}" \
                --workers="${GATEWAY_WORKERS}" \
                --register=False 2>&1 | sed -u 's/^/[literegistry-gateway] /'
            _gw_log "gateway process exited; restarting in 5s"
            sleep 5
        done
    ) &
    _gw_log "gateway supervisor started (pid $!)"
fi

# Every node waits: a worker whose leader gateway never comes up fails here instead of during rollouts.
_gw_deadline=$((SECONDS + GATEWAY_WAIT_S))
_gw_replicas=-1
while [ $SECONDS -lt $_gw_deadline ]; do
    if curl -fsS -m 10 "${SWERL_GATEWAY_URL}/health" >/dev/null 2>&1; then
        _gw_replicas=$(curl -fsS -m 10 "${SWERL_GATEWAY_URL}/v1/models" 2>/dev/null | "${LITEREGISTRY_GATEWAY_VENV}/bin/python" -c \
            "import json,sys; d=json.load(sys.stdin); print(sum(len(m.get('metadata', [])) for m in d['data'] if m.get('id') == 'podman'))" 2>/dev/null || echo -1)
        if [ "${_gw_replicas}" -ge "${GATEWAY_MIN_REPLICAS}" ] 2>/dev/null; then
            break
        fi
    fi
    sleep 5
done
if [ "${_gw_replicas}" -lt "${GATEWAY_MIN_REPLICAS}" ] 2>/dev/null || [ "${_gw_replicas}" = "-1" ]; then
    _gw_log "ERROR: gateway at ${SWERL_GATEWAY_URL} not healthy with >=${GATEWAY_MIN_REPLICAS} podman replicas after ${GATEWAY_WAIT_S}s (saw ${_gw_replicas})"
    return 1 2>/dev/null || exit 1
fi
_gw_log "gateway healthy: ${_gw_replicas} podman replicas on the roster"
