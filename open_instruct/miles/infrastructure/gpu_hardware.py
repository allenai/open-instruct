"""Cluster hints and a shared startup inventory; no generation-path queries."""

import csv
import math
import os
import subprocess

from open_instruct.miles.errors import InputError

# Hints for planning only. Named serving presets verify device inventories at startup.
CLUSTERS = {
    "ai2/holmes": {"gpu": "b300", "attention_backend": "flash_4"},
    "ai2/jupiter": {"gpu": "h100", "attention_backend": "flash_3"},
    "ai2/saturn": {"gpu": "a100", "attention_backend": "flash_2"},
}


def inventory():
    """Return physical GPU identities and capacity for the selected visible devices.

    MILES' pinned serving adapter requires numeric CUDA device IDs, not MIG
    instances. Unknown/N/A memory is retained as None and cannot qualify a profile.
    """
    raw_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_mask is None:
        mask = []  # Unset means all physical devices; explicitly empty means none.
    else:
        mask = [item.strip() for item in raw_mask.split(",")]
        if not mask or mask in ([""], ["-1"]):
            return []
        if len(set(mask)) != len(mask):
            raise InputError("CUDA_VISIBLE_DEVICES contains duplicate devices")
    if any(not device.isdecimal() for device in mask):
        raise InputError(
            "The pinned MILES serving adapter requires numeric CUDA_VISIBLE_DEVICES; MIG/UUID masks are unsupported"
        )
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"]
    if mask:
        command.append("--id=" + ",".join(mask))
    try:
        output = subprocess.check_output(command, text=True, timeout=15)
        devices = []
        for row in csv.reader(output.splitlines(), skipinitialspace=True):
            index, name, memory = (value.strip() for value in row)
            capacity = None if memory in ("N/A", "[N/A]", "[Not Supported]") else float(memory)
            if not index.isdecimal() or (capacity is not None and (not math.isfinite(capacity) or capacity <= 0)):
                raise ValueError(f"Invalid GPU inventory row: {row}")
            devices.append({"id": index, "name": name, "memory_mib": capacity})
        by_id = {device["id"]: device for device in devices}
        if len(by_id) != len(devices) or (mask and set(by_id) != set(mask)):
            raise ValueError("GPU inventory does not match requested devices")
        return [by_id[index] for index in mask] if mask else devices
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Cannot inspect GPU hardware: {error}") from error


def rollout_inventory(node, inventory):
    """Select a topology-sized slice, assuming homogeneous devices in the Ray pool.

    Ray assigns trainer and rollout devices within its pool; this slice does not
    identify the actual actor assignments. Judge devices are outside that pool.
    """
    start = node["trainer_gpus"]
    selected = inventory[start : start + node["rollout_gpus"]]
    if len(selected) != node["rollout_gpus"]:
        raise InputError("GPU inventory does not cover this replica's rollout allocation")
    return {
        "rollout_gpus": node["rollout_gpus"],
        "device_ids": [device["id"] for device in selected],
        "hardware": [(device["name"], device["memory_mib"]) for device in selected],
    }
