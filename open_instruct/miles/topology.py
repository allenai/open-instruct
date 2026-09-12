"""Deterministic GPU ownership before Ray starts; judges never enter its GPU pool."""

import ipaddress
from typing import Any

from open_instruct.miles import judging
from open_instruct.miles.errors import InputError


def plan(spec):
    miles = spec.compile().miles
    capacity = spec.launch.get("gpus_per_replica", miles["num_gpus_per_node"])
    trainer_nodes, trainer_per_node = miles["actor_num_nodes"], miles["actor_num_gpus_per_node"]
    trainer = trainer_nodes * trainer_per_node
    rollout = miles["rollout_num_gpus"]
    tp = miles["rollout_num_gpus_per_engine"]
    managed = {
        name: service["gpus"]
        for name, service in judging.registry(spec.judges)["judges"].items()
        if service["mode"] == "managed"
    }
    if trainer_per_node > capacity or tp > capacity or capacity % tp:
        raise InputError(
            "Trainer ranks and each rollout engine must fit within launch.gpus_per_replica; capacity must divide into whole engines"
        )
    if miles["num_gpus_per_node"] != capacity:
        raise InputError("inference.rollout_gpus_per_node must match launch.gpus_per_replica")
    if any(count > capacity for count in managed.values()):
        raise InputError("Each managed judge must fit within one replica")
    if miles.get("eval_num_gpus", 0) or miles.get("rollout_external", False):
        raise InputError("The config launcher does not yet allocate dedicated eval or external rollout pools")
    policy = trainer if miles["colocate"] else trainer + rollout
    total = policy + sum(managed.values())
    nodes: list[dict[str, Any]]
    if total <= capacity and trainer_nodes == 1:
        nodes = [
            {
                "trainer_gpus": trainer,
                "rollout_gpus": 0 if miles["colocate"] else rollout,
                "ray_gpus": policy,
                "judges": managed,
            }
        ]
        allocated = total
    else:
        if miles["colocate"]:
            raise InputError("Multi-node colocation is not yet qualified; choose disaggregated placement")
        nodes = [
            {"trainer_gpus": trainer_per_node, "rollout_gpus": 0, "ray_gpus": trainer_per_node, "judges": {}}
            for _ in range(trainer_nodes)
        ]
        remaining = rollout
        while remaining:
            count = min(capacity, remaining)
            nodes.append({"trainer_gpus": 0, "rollout_gpus": count, "ray_gpus": count, "judges": {}})
            remaining -= count
        # Do not put judges on trainer nodes; fill the final rollout node before
        # adding dedicated service nodes. Never subtract from requested rollouts.
        for name, count in sorted(managed.items()):
            last = nodes[-1]
            if last["trainer_gpus"] or last["ray_gpus"] + sum(last["judges"].values()) + count > capacity:
                nodes.append({"trainer_gpus": 0, "rollout_gpus": 0, "ray_gpus": 0, "judges": {}})
            nodes[-1]["judges"][name] = count
        allocated = capacity
    if (len(nodes) > 1 or managed) and spec.launch["auto_resume"]:
        raise InputError(
            "Multi-node/managed-judge launches require launch.auto_resume=false until coordinated restart is qualified"
        )
    if len(nodes) > 1 and not str(spec.output["root"]).startswith("/weka/"):
        raise InputError("Multi-node rendezvous requires output.root on shared WEKA")
    return {
        "replicas": len(nodes),
        "gpus_per_replica": allocated,
        "allocated_gpus": allocated * len(nodes),
        "policy_gpus": policy,
        "judge_gpus": sum(managed.values()),
        "nodes": nodes,
        "unused_gpus": allocated * len(nodes) - total,
    }


def assign(layout, addresses):
    if len(addresses) != layout["replicas"] or len(set(addresses)) != len(addresses):
        raise ValueError("Replicas must have distinct physical node addresses")
    return dict(zip(sorted(addresses, key=ipaddress.IPv4Address), layout["nodes"], strict=True))


def devices(node, visible):
    required = node["ray_gpus"] + sum(node["judges"].values())
    if len(visible) < required or len(set(visible)) != len(visible) or any(not item for item in visible):
        raise ValueError("CUDA device visibility does not cover the disjoint role assignment")
    offset = node["ray_gpus"]
    judges = {}
    for name, count in sorted(node["judges"].items()):
        judges[name] = visible[offset : offset + count]
        offset += count
    return visible[: node["ray_gpus"]], judges
