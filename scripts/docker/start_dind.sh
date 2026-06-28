#!/bin/bash
# Start a local Docker daemon inside the Beaker task and point Docker SDK clients at it.
#
# This is intentionally separate from docker_login.sh, which starts Podman services.
# Source this before Ray starts so Ray worker actors inherit DOCKER_HOST.

DOCKER_VERSION="${DIND_DOCKER_VERSION:-${DOCKER_VERSION:-27.3.1}}"
PREFIX="${DIND_PREFIX:-/opt/docker}"
RUNDIR="${DIND_RUN_DIR:-/run/docker}"
DATAROOT="${DIND_DATA_ROOT:-/var/lib/docker}"
SOCK="${DIND_SOCKET:-${RUNDIR}/docker.sock}"
DIND_LOG_DIR="${DIND_LOG_DIR:-/output/tmp}"
DIND_LOG="${DIND_LOG:-${DIND_LOG_DIR}/dockerd.log}"
DIND_SLIRP_LOG="${DIND_SLIRP_LOG:-${DIND_LOG_DIR}/slirp.log}"
DIND_NS_PID_FILE="${DIND_NS_PID_FILE:-/tmp/dind_ns_pid}"
DIND_SLIRP_READY_FILE="${DIND_SLIRP_READY_FILE:-/tmp/dind_slirp_ready}"
DIND_STORAGE_DRIVER="${DIND_STORAGE_DRIVER:-vfs}"
DIND_SMOKE_TEST="${DIND_SMOKE_TEST:-1}"
DIND_SMOKE_IMAGE="${DIND_SMOKE_IMAGE:-python:3.12-slim}"

export PATH="$PREFIX:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

unset DOCKER_TLS_VERIFY
unset DOCKER_CERT_PATH

export DOCKER_HOST="unix://${SOCK}"
export DIND_DOCKER_CLI="${PREFIX}/docker"
# Reuse the existing EnvironmentPool fanout hook. With one entry, all sandbox
# actors get an explicit docker_host pointed at the local DinD socket.
export SWERL_PODMAN_DOCKER_HOSTS="${DOCKER_HOST}"

dind_fail() {
    echo "ERROR: $*" >&2
    exit 1
}

dind_show_log() {
    local log_file="$1"
    if [ -s "$log_file" ]; then
        echo "===== ${log_file} ====="
        sed -n '1,240p' "$log_file"
    fi
}

mkdir -p "$PREFIX" "$RUNDIR" "$DATAROOT" "$DIND_LOG_DIR"

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

need_pkgs() {
    for p in slirp4netns jq iproute2 gcc libc6-dev curl ca-certificates; do
        dpkg -s "$p" >/dev/null 2>&1 || return 1
    done
}

if ! need_pkgs; then
    command -v apt-get >/dev/null 2>&1 || dind_fail "apt-get not found; cannot install DinD prerequisites"
    apt-get update -qq || dind_fail "apt-get update failed"
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        slirp4netns jq iproute2 gcc libc6-dev curl ca-certificates >/dev/null \
        || dind_fail "failed to install DinD prerequisites"
fi

if [ -n "${DIND_SLIRP4NETNS_BIN:-}" ]; then
    [ -x "$DIND_SLIRP4NETNS_BIN" ] || dind_fail "DIND_SLIRP4NETNS_BIN is not executable: $DIND_SLIRP4NETNS_BIN"
elif command -v slirp4netns >/dev/null 2>&1; then
    DIND_SLIRP4NETNS_BIN="$(command -v slirp4netns)"
elif [ -x /usr/bin/slirp4netns ]; then
    DIND_SLIRP4NETNS_BIN=/usr/bin/slirp4netns
elif [ -x /bin/slirp4netns ]; then
    DIND_SLIRP4NETNS_BIN=/bin/slirp4netns
else
    dind_fail "slirp4netns not found after installing DinD prerequisites"
fi
export DIND_SLIRP4NETNS_BIN

if [ ! -x "$PREFIX/dockerd" ]; then
    curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz" \
        | tar xz -C "$PREFIX" --strip-components=1 \
        || dind_fail "failed to install Docker ${DOCKER_VERSION}"
    ln -sf "$PREFIX/docker" /usr/local/bin/docker || dind_fail "failed to link Docker CLI"
fi

[ -x "$PREFIX/runc" ] || dind_fail "Docker install did not create $PREFIX/runc"
[ -e "$PREFIX/runc.real" ] || mv "$PREFIX/runc" "$PREFIX/runc.real"
cat >"$PREFIX/runc-wrap" <<WRAP
#!/bin/bash
REAL="$PREFIX/runc.real"
BUNDLE=
for ((i=1;i<=\$#;i++)); do
    a="\${!i}"
    if [ "\$a" = "--bundle" ] || [ "\$a" = "-b" ]; then j=\$((i+1)); BUNDLE="\${!j}"; break; fi
done
case " \$* " in *" create "*|*" run "*)
    if [ -n "\$BUNDLE" ] && [ -f "\$BUNDLE/config.json" ]; then
        cp "\$BUNDLE/config.json" "\$BUNDLE/config.json.orig"
        jq '
          (.linux.sysctl // {}) as \$s
          | .linux.sysctl = (\$s | with_entries(select(.key as \$k | (\$k | startswith("net.")) | not)))
          | del(.linux.cgroupsPath)
          | del(.linux.resources)
          | .mounts |= map(
              if .type=="proc"   then {destination:"/proc",type:"none",source:"/proc",options:["rbind","nosuid","noexec","nodev"]}
              elif .type=="sysfs"  then {destination:.destination,type:"none",source:"/sys",options:["rbind","nosuid","noexec","nodev","ro"]}
              elif .type=="mqueue" then {destination:.destination,type:"none",source:"/dev/mqueue",options:["rbind","nosuid","noexec","nodev"]}
              elif .type=="devpts" then {destination:.destination,type:"devpts",source:"devpts",options:["nosuid","noexec"]}
              else . end)
        ' "\$BUNDLE/config.json.orig" > "\$BUNDLE/config.json"
    fi ;;
esac
exec "\$REAL" "\$@"
WRAP
chmod +x "$PREFIX/runc-wrap"
ln -sf "$PREFIX/runc-wrap" "$PREFIX/runc"

if [ ! -x "$PREFIX/userns_launcher" ]; then
    cat >/tmp/userns_launcher.c <<'C'
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include <signal.h>
static int wf(const char *p, const char *d){int f=open(p,O_WRONLY);if(f<0)return -1;ssize_t r=write(f,d,strlen(d));close(f);return r==(ssize_t)strlen(d)?0:-1;}
int main(int argc,char**argv){
    if(argc<2)return 2;
    int sp[2],ap[2]; if(pipe(sp)||pipe(ap))return 1;
    int flags=CLONE_NEWUSER|CLONE_NEWNS|CLONE_NEWNET|CLONE_NEWCGROUP|CLONE_NEWIPC|CLONE_NEWUTS;
    pid_t pid=fork(); if(pid<0)return 1;
    if(pid==0){close(sp[1]);close(ap[0]);
        if(unshare(flags)<0){perror("unshare");_exit(1);}
        if(write(ap[1],"u",1)!=1)_exit(1); close(ap[1]);
        char c; if(read(sp[0],&c,1)!=1)_exit(1); close(sp[0]);
        execvp(argv[1],&argv[1]); perror("exec"); _exit(127);}
    close(sp[0]);close(ap[1]); char c;
    if(read(ap[0],&c,1)!=1){kill(pid,9);return 1;} close(ap[0]);
    char path[256];
    snprintf(path,sizeof path,"/proc/%d/uid_map",pid);
    if(wf(path,"0 0 1\n1 100000 65536\n")<0){perror("uid_map");kill(pid,9);return 1;}
    snprintf(path,sizeof path,"/proc/%d/gid_map",pid);
    if(wf(path,"0 0 1\n1 100000 65536\n")<0){perror("gid_map");kill(pid,9);return 1;}
    if(write(sp[1],"x",1)!=1){kill(pid,9);return 1;} close(sp[1]);
    fprintf(stderr,"USERNS_PID=%d\n",pid); fflush(stderr);
    int s; waitpid(pid,&s,0); return WIFEXITED(s)?WEXITSTATUS(s):1;}
C
    gcc -O2 -o "$PREFIX/userns_launcher" /tmp/userns_launcher.c || dind_fail "failed to build userns launcher"
fi
[ -x "$PREFIX/userns_launcher" ] || dind_fail "userns launcher missing at $PREFIX/userns_launcher"

cat >"$PREFIX/inside_dockerd.sh" <<EOF
#!/bin/bash
set -e
export PATH=$PREFIX:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:\$PATH
mount --make-rprivate /
mkdir -p /tmp/cg && mount -t cgroup2 none /tmp/cg && mount --bind /tmp/cg /sys/fs/cgroup
ip link set lo up
echo \$\$ > "$DIND_NS_PID_FILE"
while [ ! -f "$DIND_SLIRP_READY_FILE" ]; do sleep 0.1; done
for attempt in \$(seq 1 50); do
    if ip addr show tap0 >/dev/null 2>&1 && ip route get 10.0.2.3 >/dev/null 2>&1; then
        break
    fi
    if [ "\$attempt" = "50" ]; then
        echo "ERROR: slirp4netns did not configure tap0 networking" >&2
        exit 1
    fi
    sleep 0.1
done
echo "nameserver 10.0.2.3" > /tmp/resolv.conf
mount --bind /tmp/resolv.conf /etc/resolv.conf
mkdir -p "$DATAROOT" "$RUNDIR"
dockerd_args=(
    --data-root="$DATAROOT"
    --exec-root="$RUNDIR"
    --pidfile="$RUNDIR/docker.pid"
    --storage-driver="$DIND_STORAGE_DRIVER"
    --iptables=false
    --ip6tables=false
    --bridge=none
    --ip-forward=false
    --userland-proxy-path="$PREFIX/docker-proxy"
    --dns=10.0.2.3
    --host=unix://"$SOCK"
)
if [ -n "\${MIRROR_URL:-}" ]; then
    dockerd_args+=(--registry-mirror "http://\${MIRROR_URL}" --insecure-registry "\${MIRROR_URL}")
fi
exec "$PREFIX/dockerd" "\${dockerd_args[@]}"
EOF
chmod +x "$PREFIX/inside_dockerd.sh"

if [ -S "$SOCK" ] && "$DIND_DOCKER_CLI" info >/dev/null 2>&1; then
    echo "Docker daemon already running at ${DOCKER_HOST}"
else
    [ -e /dev/net/tun ] || { mkdir -p /dev/net && mknod /dev/net/tun c 10 200 && chmod 666 /dev/net/tun; } \
        || dind_fail "failed to create /dev/net/tun"
    rm -f "$DIND_NS_PID_FILE" "$DIND_SLIRP_READY_FILE" "$SOCK" "$RUNDIR/docker.pid"

    echo "Starting dockerd for Docker SDK at ${DOCKER_HOST}"
    echo "dockerd data-root=${DATAROOT} exec-root=${RUNDIR} storage-driver=${DIND_STORAGE_DRIVER}"
    : >"$DIND_LOG"
    : >"$DIND_SLIRP_LOG"
    "$PREFIX/userns_launcher" "$PREFIX/inside_dockerd.sh" >"$DIND_LOG" 2>&1 &
    export DIND_USERNS_LAUNCHER_PID=$!
    echo "userns launcher PID: ${DIND_USERNS_LAUNCHER_PID}; dockerd log: ${DIND_LOG}"

    for attempt in $(seq 1 50); do
        [ -f "$DIND_NS_PID_FILE" ] && break
        if ! kill -0 "$DIND_USERNS_LAUNCHER_PID" >/dev/null 2>&1; then
            dind_show_log "$DIND_LOG"
            dind_fail "userns launcher exited before writing ${DIND_NS_PID_FILE}"
        fi
        sleep 0.1
    done
    [ -f "$DIND_NS_PID_FILE" ] || { dind_show_log "$DIND_LOG"; dind_fail "timed out waiting for ${DIND_NS_PID_FILE}"; }

    NS_PID="$(sed -n '1p' "$DIND_NS_PID_FILE")"
    "$DIND_SLIRP4NETNS_BIN" --configure --mtu=65520 --disable-host-loopback "$NS_PID" tap0 >"$DIND_SLIRP_LOG" 2>&1 &
    export DIND_SLIRP_PID=$!
    echo "slirp4netns PID: ${DIND_SLIRP_PID}; log: ${DIND_SLIRP_LOG}"
    sleep 2
    if ! kill -0 "$DIND_SLIRP_PID" >/dev/null 2>&1; then
        dind_show_log "$DIND_SLIRP_LOG"
        dind_fail "slirp4netns exited before dockerd startup"
    fi
    touch "$DIND_SLIRP_READY_FILE"

    for attempt in $(seq 1 120); do
        if "$DIND_DOCKER_CLI" info >/dev/null 2>&1; then
            break
        fi
        if ! kill -0 "$DIND_USERNS_LAUNCHER_PID" >/dev/null 2>&1; then
            dind_show_log "$DIND_LOG"
            dind_show_log "$DIND_SLIRP_LOG"
            dind_fail "dockerd exited before becoming ready"
        fi
        if [ "$attempt" = "10" ] || [ "$attempt" = "30" ] || [ "$attempt" = "60" ] || [ "$attempt" = "120" ]; then
            echo "Waiting for dockerd at ${DOCKER_HOST} (attempt ${attempt}/120)"
        fi
        sleep 0.5
    done
fi

if ! "$DIND_DOCKER_CLI" info; then
    dind_show_log "$DIND_LOG"
    dind_show_log "$DIND_SLIRP_LOG"
    dind_fail "docker info failed for ${DOCKER_HOST}"
fi

if [ "$DIND_SMOKE_TEST" = "1" ]; then
    echo "Running DinD smoke test with ${DIND_SMOKE_IMAGE}"
    if ! "$DIND_DOCKER_CLI" run --rm "$DIND_SMOKE_IMAGE" python -c 'print("dind ok")'; then
        dind_show_log "$DIND_LOG"
        dind_show_log "$DIND_SLIRP_LOG"
        dind_fail "DinD smoke test failed"
    fi
fi

echo "Docker SDK configured to use DinD at ${DOCKER_HOST}"
