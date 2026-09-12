"""Preflight check for a LiteRegistry Podman gateway deployment.

Run this before launching a gateway-backend terminal RL job to verify the
deployment end to end from the machine you care about:

1. gateway /health and the docker.io mirror route /v2/
2. GatewayBackend protocol: exec, state persistence, >60s commands, command
   timeouts, oversized output, chunked file IO, put_archive
3. one full SWERLVanilluxSandboxEnv episode on a real tmax task image,
   asserting the verifier awards reward 1.0

Usage:
    uv run python scripts/general_agent/terminal/rl/gateway/check_gateway_deployment.py \
        --gateway_url http://HOST:PORT [--skip_episode]

The episode check needs the allenai/tmax-15k-open-instruct task data (downloads
on first use) and pulls one small task image on a replica.
"""

import argparse
import asyncio
import io
import os
import sys
import tarfile
import time

import requests

from open_instruct import logger_utils
from open_instruct.environments.backends import GatewayBackend
from open_instruct.environments.base import EnvCall
from open_instruct.environments.swerl_vanillux_sandbox import SWERLVanilluxSandboxEnv

logger = logger_utils.setup_logger(__name__)

# A tmax-15k task whose verifier checks for an exact CSV we can write directly,
# so a passing run proves the tests-upload -> verifier -> reward-parse path.
EPISODE_TASK_ID = "task_000035_88acd16c"
EPISODE_IMAGE = "hamishi740/swerl-tmax-v3:d6f61374e053"
EPISODE_SOLVE_COMMAND = (
    "mkdir -p /home/user/pipeline && "
    "printf 'timestamp,distance\\n1000,22.36\\n1001,29.15\\n1003,7.07\\n' > /home/user/pipeline/cleaned.csv"
)

FAILURES: list[str] = []


def check(name: str, condition: bool, extra: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" ({extra})" if extra else ""))
    if not condition:
        FAILURES.append(name)


def check_gateway_health(gateway_url: str) -> None:
    health = requests.get(f"{gateway_url}/health", timeout=15).json()
    check("gateway /health", health.get("status") == "healthy", str(health))
    mirror_status = requests.get(f"{gateway_url}/v2/", timeout=15).status_code
    check("docker mirror /v2/", mirror_status == 200, f"HTTP {mirror_status}")


def check_backend_protocol(gateway_url: str) -> None:
    backend = GatewayBackend(image="python:3.12-slim", timeout=120, gateway_url=gateway_url)
    start_time = time.time()
    backend.start()
    check("handshake", True, f"{time.time() - start_time:.1f}s on {backend._instance_id}")
    try:
        result = backend.run_command("echo out && echo err >&2 && exit 3")
        check(
            "exec stdout/stderr/exit_code",
            (result.stdout.strip(), result.stderr.strip(), result.exit_code) == ("out", "err", 3),
            repr(result),
        )

        backend.run_command("echo state > /workspace/state.txt")
        result = backend.run_command("cat /workspace/state.txt")
        check("state persists across execs", result.stdout.strip() == "state")

        start_time = time.time()
        result = backend.run_command("sleep 70 && echo done", timeout=120)
        check(
            "command past the 60s replica cap",
            result.stdout.strip() == "done" and result.exit_code == 0,
            f"{time.time() - start_time:.1f}s",
        )

        result = backend.run_command("sleep 30", timeout=5)
        check("command timeout -> exit 124", result.exit_code == 124 and "timed out" in result.stderr)

        result = backend.run_command("head -c 5000000 /dev/zero | tr '\\0' 'a'", timeout=60)
        check("oversized stdout bounded, no wedge", result.exit_code == 0 and 0 < len(result.stdout) <= 512 * 1024)
        result = backend.run_command("echo alive")
        check("session alive after oversized stdout", result.stdout.strip() == "alive")

        blob = os.urandom(1_500_000)
        backend.write_file("/workspace/blob.bin", blob)
        check("chunked write/read 1.5MB", backend.read_file("/workspace/blob.bin", binary=True) == blob)

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            data = b"archive\n"
            info = tarfile.TarInfo(name="sub/file.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        backend.put_archive("/workspace/unpacked", tar_stream.getvalue())
        result = backend.run_command("cat /workspace/unpacked/sub/file.txt")
        check("put_archive", result.stdout == "archive\n")
    finally:
        backend.close()
    check("close", True)


async def check_episode(gateway_url: str) -> None:
    env = SWERLVanilluxSandboxEnv(
        backend="gateway",
        image="python:3.12-slim",
        task_data_hf_repo="allenai/tmax-15k-open-instruct",
        test_timeout=120,
        timeout=60,
        gateway_url=gateway_url,
    )
    await env.setup()
    start_time = time.time()
    await env.reset(task_id=EPISODE_TASK_ID, max_steps=10, image=EPISODE_IMAGE)
    check("episode reset (task image pull + task data)", True, f"{time.time() - start_time:.1f}s")
    result = await env.step(EnvCall(id="1", name="bash", args={"command": EPISODE_SOLVE_COMMAND}))
    check("episode solve step", result.metadata.get("exit_code") == 0, str(result.metadata))
    result = await env.step(
        EnvCall(id="2", name="bash", args={"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"})
    )
    check("episode verifier reward", result.done and result.reward == 1.0, f"reward={result.reward}")
    await env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway_url", required=True, help="LiteRegistry gateway base URL, e.g. http://host:port")
    parser.add_argument("--skip_episode", action="store_true", help="Skip the full tmax-task episode check")
    args = parser.parse_args()
    gateway_url = args.gateway_url.rstrip("/")

    check_gateway_health(gateway_url)
    check_backend_protocol(gateway_url)
    if not args.skip_episode:
        asyncio.run(check_episode(gateway_url))

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) FAILED: {FAILURES}")
        sys.exit(1)
    print("\nAll gateway deployment checks passed.")


if __name__ == "__main__":
    main()
