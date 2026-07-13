#!/bin/bash
# Install and configure Podman + crun the same way the training Dockerfile does,
# for sandbox tasks that need to launch subcontainers (e.g. on Beaker or a bare host).
#
# This mirrors the Podman section of the open-instruct Dockerfile so you can set up
# an existing machine/image without rebuilding from scratch. Build deps, the podman
# and crun versions, the /etc/containers config files, the docker->podman shim, the
# docker.io pull-through-mirror helper, and the subuid/subgid ranges all match the
# Dockerfile.
#
# Usage:
#   sudo bash scripts/docker/setup_podman.sh
#
# Overridable via env vars:
#   PODMAN_VERSION (default 5.6.2)
#   CRUN_VERSION   (default 1.14.3)
#   SKIP_APT=1     skip the apt-get install step (deps already present)

set -euo pipefail

PODMAN_VERSION="${PODMAN_VERSION:-5.6.2}"
CRUN_VERSION="${CRUN_VERSION:-1.14.3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/docker -> open-instruct root holds the docker/podman config files.
OI_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PODMAN_CONF_DIR="$OI_ROOT/docker/podman"

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

log() { echo "[setup_podman] $*"; }
fail() { echo "[setup_podman] ERROR: $*" >&2; exit 1; }

for f in containers.conf policy.json 10-unqualified-search-registries.conf setup_dockerio_mirror; do
    [ -f "$PODMAN_CONF_DIR/$f" ] || fail "missing config file: $PODMAN_CONF_DIR/$f"
done

# 1. Build/runtime dependencies (matches the Dockerfile package list).
if [ "${SKIP_APT:-0}" != "1" ]; then
    log "installing apt dependencies"
    command -v apt-get >/dev/null 2>&1 || fail "apt-get not found; this script targets Debian/Ubuntu"
    $SUDO apt-get update
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        conmon \
        gcc \
        go-md2man \
        golang-github-containers-common \
        golang-go \
        iptables \
        libassuan-dev \
        libbtrfs-dev \
        libcap-dev \
        libc6-dev \
        libdevmapper-dev \
        libglib2.0-dev \
        libgpg-error-dev \
        libgpgme-dev \
        libprotobuf-c-dev \
        libprotobuf-dev \
        libseccomp-dev \
        libselinux1-dev \
        libsystemd-dev \
        libtool \
        libyajl-dev \
        docker.io \
        netavark \
        passt \
        pkg-config \
        python3-sphinx \
        systemd \
        uidmap \
        curl \
        git \
        make \
        wget
    $SUDO rm -rf /var/lib/apt/lists/*
fi

# 2. Install the /etc/containers config files.
log "installing /etc/containers config"
$SUDO mkdir -p /etc/containers/registries.conf.d/
$SUDO cp "$PODMAN_CONF_DIR/containers.conf" /etc/containers/containers.conf
$SUDO cp "$PODMAN_CONF_DIR/policy.json" /etc/containers/policy.json
$SUDO cp "$PODMAN_CONF_DIR/10-unqualified-search-registries.conf" \
    /etc/containers/registries.conf.d/10-unqualified-search-registries.conf

# 2b. Install the docker.io pull-through-mirror helper (matches the Dockerfile's
#     COPY docker/podman/setup_dockerio_mirror /usr/local/bin/). Needed so mirror-based
#     pulls work the same as in the training image (see MIRROR_URL usage).
log "installing setup_dockerio_mirror -> /usr/local/bin"
$SUDO cp "$PODMAN_CONF_DIR/setup_dockerio_mirror" /usr/local/bin/setup_dockerio_mirror
$SUDO chmod +x /usr/local/bin/setup_dockerio_mirror

# 3. Build and install Podman from source.
log "building podman v${PODMAN_VERSION}"
PODMAN_SRC="/tmp/podman-${PODMAN_VERSION}"
rm -rf "$PODMAN_SRC"
wget -qO- "https://github.com/containers/podman/archive/refs/tags/v${PODMAN_VERSION}.tar.gz" \
    | tar xz -C /tmp
make -C "$PODMAN_SRC" BUILDTAGS="selinux seccomp" PREFIX=/usr
$SUDO make -C "$PODMAN_SRC" install PREFIX=/usr
rm -rf "$PODMAN_SRC"

# 4. Build and install crun.
log "building crun ${CRUN_VERSION}"
CRUN_SRC="/tmp/crun"
rm -rf "$CRUN_SRC"
git clone --depth 1 -b "$CRUN_VERSION" https://github.com/containers/crun.git "$CRUN_SRC"
(
    cd "$CRUN_SRC"
    ./autogen.sh
    ./configure --prefix=/usr --sysconfdir=/etc
    make
    $SUDO make install
)
rm -rf "$CRUN_SRC"

# 5. Translate Docker CLI calls to Podman by default. DinD scripts call /usr/bin/docker
#    explicitly when they need the real Docker CLI.
log "symlinking docker -> podman"
$SUDO ln -sf "$(command -v podman)" /usr/local/bin/docker

# 6. subuid/subgid ranges for rootless containers.
if ! grep -q "^root:10000:11165536" /etc/subuid 2>/dev/null; then
    echo "root:10000:11165536" | $SUDO tee -a /etc/subuid >/dev/null
fi
if ! grep -q "^root:10000:11165536" /etc/subgid 2>/dev/null; then
    echo "root:10000:11165536" | $SUDO tee -a /etc/subgid >/dev/null
fi


log "done. podman version:"
podman --version
crun --version | head -1
