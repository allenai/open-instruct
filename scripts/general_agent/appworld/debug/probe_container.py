"""Diagnostic probe: why does the AppWorld container die on Beaker podman, and is the
container's bridge IP reachable from here?

Runs as a tiny no-training Beaker job. It connects to a podman shard and starts the AppWorld
server container two ways — WITHOUT and WITH host-port publishing — and for each prints the
container status, exit code, error, log tail, bridge IP, published ports, and HTTP reachability
(bridge IP and, if published, the host port). This isolates:

  1. Whether host-port publishing is what makes the container exit instantly (the swerl contrast).
  2. Whether the trainer can reach the container's bridge IP (the open networking question for
     the stateful-server style).

Launch: ./scripts/train/build_image_and_launch_dirty.sh \
            scripts/general_agent/appworld/debug/probe_container_beaker.sh
"""

import os
import time

import docker
import requests

IMAGE = os.environ.get("APPWORLD_IMAGE", "shatu/appworld-data:latest")
PORT = 8000


def first_docker_host() -> str | None:
    hosts = [h.strip() for h in os.environ.get("SWERL_PODMAN_DOCKER_HOSTS", "").split(",") if h.strip()]
    if hosts:
        return hosts[0]
    # Fallback to the conventional shard-0 socket that docker_login.sh creates.
    sock = "/tmp/podman-services/0/podman.sock"
    return f"unix://{sock}" if os.path.exists(sock) else None


def make_client() -> docker.DockerClient:
    dh = first_docker_host()
    print(f"docker_host = {dh!r}  (SWERL_PODMAN_DOCKER_HOSTS={os.environ.get('SWERL_PODMAN_DOCKER_HOSTS')!r})", flush=True)
    return docker.DockerClient(base_url=dh) if dh else docker.from_env()


def run_test(cl: docker.DockerClient, publish: bool, host_network: bool = False, port: int = PORT) -> None:
    label = "network_mode=host" if host_network else ("WITH host-port publish" if publish else "NO publish (bridge IP only)")
    print(f"\n================ TEST: {label} (port {port}) ================", flush=True)
    kwargs = dict(
        image=IMAGE,
        command=["environment", "--port", str(port), "--no-show-usage"],
        environment={"APPWORLD_ROOT": "/run"},
        detach=True,
        auto_remove=False,
        mem_limit="4g",
        memswap_limit="4g",
        labels={"appworld_probe": "1"},
    )
    if host_network:
        kwargs["network_mode"] = "host"
    elif publish:
        kwargs["ports"] = {f"{port}/tcp": None}

    try:
        c = cl.containers.run(**kwargs)
    except Exception as e:
        print(f"containers.run FAILED to even create/start: {type(e).__name__}: {e}", flush=True)
        return

    try:
        # Watch the container for up to 30s; note if/when it stops running.
        last_status = None
        for _ in range(30):
            c.reload()
            if c.status != last_status:
                print(f"  t status -> {c.status}", flush=True)
                last_status = c.status
            if c.status != "running":
                break
            time.sleep(1)
        c.reload()
        state = c.attrs.get("State", {}) or {}
        print(f"final status: {c.status} | exit_code: {state.get('ExitCode')} | error: {state.get('Error')!r}", flush=True)
        print("--- container logs (tail 60) ---", flush=True)
        print(c.logs(tail=60).decode("utf-8", "replace"), flush=True)

        net = c.attrs.get("NetworkSettings", {}) or {}
        ip = net.get("IPAddress") or next(
            (n.get("IPAddress") for n in (net.get("Networks") or {}).values() if n.get("IPAddress")), None
        )
        ports = net.get("Ports")
        print(f"bridge IP: {ip} | published ports: {ports}", flush=True)
        print(f"full NetworkSettings: {net}", flush=True)

        urls = [f"http://127.0.0.1:{port}/"]  # works if host_network (shared netns)
        if ip:
            urls.append(f"http://{ip}:{port}/")
        if ports and ports.get(f"{port}/tcp"):
            urls.append(f"http://127.0.0.1:{ports[f'{port}/tcp'][0]['HostPort']}/")
        for url in urls:
            try:
                r = requests.get(url, timeout=5)
                print(f"  GET {url} -> {r.status_code}  {r.text[:80]!r}", flush=True)
            except Exception as e:
                print(f"  GET {url} -> ERROR {type(e).__name__}: {e}", flush=True)
    finally:
        try:
            c.remove(force=True)
        except Exception:
            pass


def main() -> None:
    cl = make_client()
    try:
        cl.images.get(IMAGE)
        print(f"image {IMAGE} already present", flush=True)
    except Exception:
        print(f"pulling {IMAGE} (through the mirror) ...", flush=True)
        cl.images.pull(IMAGE)
        print("pull done", flush=True)

    run_test(cl, publish=False)
    run_test(cl, publish=True)
    run_test(cl, publish=False, host_network=True, port=8137)  # candidate fix: reach via 127.0.0.1
    print("\nPROBE DONE", flush=True)


if __name__ == "__main__":
    main()
