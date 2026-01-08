"""
Monarch utilities for Beaker-based distributed training.

This module provides integration between Monarch (PyTorch's distributed actor framework)
and Beaker (AI2's job scheduler) for multi-node GRPO training.
"""

import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BeakerJobConfig:
    """Configuration for a Beaker-launched Monarch job."""

    num_replicas: int
    gpus_per_replica: int = 8
    rendezvous_timeout: int = 600


class BeakerHostDiscovery:
    """Discovers all Beaker replica hosts for forming a Monarch mesh.

    Beaker provides:
    - BEAKER_REPLICA_RANK: The rank of this replica (0, 1, 2, ...)
    - BEAKER_LEADER_REPLICA_HOSTNAME: The hostname of replica 0 (leader)
    - BEAKER_NODE_HOSTNAME: This replica's hostname

    Since Beaker doesn't provide all hostnames directly, we use a rendezvous approach:
    1. All replicas report their hostname to the leader
    2. Leader collects all hostnames and broadcasts them
    3. All replicas can then form a HostMesh
    """

    def __init__(self, num_replicas: int, rendezvous_port: int = 9999, timeout: int = 600):
        self.num_replicas = num_replicas
        self.rendezvous_port = rendezvous_port
        self.timeout = timeout

        self.replica_rank = int(os.environ.get("BEAKER_REPLICA_RANK", "0"))
        self.leader_hostname = os.environ.get("BEAKER_LEADER_REPLICA_HOSTNAME", "localhost")
        self.node_hostname = os.environ.get("BEAKER_NODE_HOSTNAME", socket.gethostname())

    def discover_hosts(self) -> list[str]:
        """Discover all replica hostnames via TCP rendezvous."""
        if self.num_replicas == 1:
            return [self.node_hostname]

        if self.replica_rank == 0:
            return self._leader_collect_hosts()
        else:
            return self._worker_report_and_receive()

    def _leader_collect_hosts(self) -> list[str]:
        """Leader collects hostnames from all workers and broadcasts back."""
        logger.info(f"Leader starting rendezvous on port {self.rendezvous_port}")

        hosts = [""] * self.num_replicas
        hosts[0] = self.node_hostname

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("0.0.0.0", self.rendezvous_port))
        server_socket.listen(self.num_replicas)
        server_socket.settimeout(self.timeout)

        connections: list[socket.socket] = []
        start_time = time.time()

        while len(connections) < self.num_replicas - 1:
            if time.time() - start_time > self.timeout:
                raise TimeoutError(
                    f"Rendezvous timeout: only {len(connections) + 1}/{self.num_replicas} replicas connected"
                )
            try:
                conn, _addr = server_socket.accept()
                data = conn.recv(1024).decode("utf-8")
                rank_str, hostname = data.split(",", 1)
                rank = int(rank_str)
                hosts[rank] = hostname
                connections.append(conn)
                logger.info(f"Leader received: rank={rank}, hostname={hostname}")
            except TimeoutError:
                continue

        host_list = ",".join(hosts)
        for conn in connections:
            conn.send(host_list.encode("utf-8"))
            conn.close()
        server_socket.close()

        logger.info(f"Leader rendezvous complete: {hosts}")
        return hosts

    def _worker_report_and_receive(self) -> list[str]:
        """Worker reports hostname to leader and receives full host list."""
        logger.info(
            f"Worker {self.replica_rank} connecting to leader at {self.leader_hostname}:{self.rendezvous_port}"
        )

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            try:
                client_socket.connect((self.leader_hostname, self.rendezvous_port))
                break
            except (ConnectionRefusedError, OSError):
                time.sleep(1)
                continue
        else:
            raise TimeoutError(f"Worker {self.replica_rank} failed to connect to leader")

        message = f"{self.replica_rank},{self.node_hostname}"
        client_socket.send(message.encode("utf-8"))

        host_list = client_socket.recv(4096).decode("utf-8")
        client_socket.close()

        hosts = host_list.split(",")
        logger.info(f"Worker {self.replica_rank} received hosts: {hosts}")
        return hosts


class BeakerJob:
    """Monarch job interface for Beaker-launched replicas.

    Usage:
        job = BeakerJob(num_replicas=4)
        hosts = job.discover_hosts()
        proc_mesh = job.create_proc_mesh(gpus_per_host=8)
        trainers = proc_mesh.spawn("trainer", TrainerActor, ...)
    """

    def __init__(self, config: BeakerJobConfig):
        self.config = config
        self.discovery = BeakerHostDiscovery(num_replicas=config.num_replicas, timeout=config.rendezvous_timeout)
        self._hosts: list[str] | None = None

    @property
    def replica_rank(self) -> int:
        return self.discovery.replica_rank

    @property
    def is_leader(self) -> bool:
        return self.replica_rank == 0

    def discover_hosts(self) -> list[str]:
        """Discover all Beaker replica hosts."""
        if self._hosts is None:
            self._hosts = self.discovery.discover_hosts()
        return self._hosts

    def create_proc_mesh(self, gpus_per_host: int | None = None) -> Any:
        """Create a Monarch ProcMesh from discovered hosts.

        Note: This requires Monarch to be running on each host.
        The actual implementation depends on Monarch's API which may vary.
        """
        from monarch.actor import this_host

        if gpus_per_host is None:
            gpus_per_host = self.config.gpus_per_replica

        if self.config.num_replicas == 1:
            return this_host().spawn_procs({"gpus": gpus_per_host})
        else:
            hosts = self.discover_hosts()
            logger.info(f"Creating ProcMesh across {len(hosts)} hosts with {gpus_per_host} GPUs each")
            raise NotImplementedError(
                "Multi-host ProcMesh creation requires Monarch's hosts_from_config or similar API. "
                "This is marked NYI in Monarch. Falling back to single-host mode."
            )


def get_beaker_job(num_replicas: int = 1, gpus_per_replica: int = 8) -> BeakerJob:
    """Create a BeakerJob from environment or defaults."""
    if "BEAKER_REPLICA_RANK" in os.environ:
        replica_rank = int(os.environ["BEAKER_REPLICA_RANK"])
        logger.info(f"Running on Beaker: replica_rank={replica_rank}, num_replicas={num_replicas}")
    else:
        logger.info("Not running on Beaker, using single-host mode")
        num_replicas = 1

    config = BeakerJobConfig(num_replicas=num_replicas, gpus_per_replica=gpus_per_replica)
    return BeakerJob(config)
