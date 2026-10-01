"""Validate explicit network-interface selection inside the job container before
starting distributed processes. An unavailable NCCL socket override can otherwise
cause confusing initialization failures. This check permits automatic selection
for the supported single-node bridge-network fallback and rejects invalid settings
where distributed startup requires a working interface.
"""

import argparse
import os
import socket

from open_instruct import logger_utils
from open_instruct.miles.errors import InputError

logger = logger_utils.setup_logger(__name__)


def matching_interfaces(selector, interfaces):
    """Apply NCCL's comma-separated prefixes, optional exclusion and exact match."""
    exclude = selector.startswith("^")
    if exclude:
        selector = selector[1:]
    exact = selector.startswith("=")
    if exact:
        selector = selector[1:]
    prefixes = [prefix for prefix in selector.split(",") if prefix]
    return [
        name
        for name in interfaces
        if any(name == prefix if exact else name.startswith(prefix) for prefix in prefixes) != exclude
    ]


def check(*, replicas, network_mode):
    selector = os.environ.get("NCCL_SOCKET_IFNAME")
    if not selector:
        return
    interfaces = [name for _, name in socket.if_nameindex()]
    if matching_interfaces(selector, interfaces):
        return
    details = f"NCCL_SOCKET_IFNAME={selector!r} matches no container interfaces; available interfaces: {interfaces}"
    if replicas == 1 and network_mode == "bridge" and selector == "ib":
        logger.warning(
            "%s. Removing the unavailable ib override for this single-node bridge-networked job; "
            "NCCL will select an interface automatically.",
            details,
        )
        del os.environ["NCCL_SOCKET_IFNAME"]
        return
    raise InputError(
        f"{details}. Remove the override or select an interface available in this job's network namespace."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--network-mode", choices=("bridge", "host"), required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.replicas < 1 or not command:
        parser.error("a positive replica count and a workload command are required")
    try:
        check(replicas=args.replicas, network_mode=args.network_mode)
    except InputError as error:
        parser.error(str(error))
    # Replace this process so attention checks, Ray and training all inherit any
    # fallback. A standalone preflight subprocess cannot unset its parent's env.
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
