"""Validate small-slice mesh construction offline; this is not a CE parity test."""

import json
import sys
from pathlib import Path

from jax import sharding
from maxtext.configs import pyconfig
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.trainers.pre_train import train_compile


def main():
    name = sys.argv[1]
    assert name in ("candidate", "reference", "reference-cp8")
    mesh_args = {
        "reference-cp8": [
            "custom_mesh_and_rule=olmo35-cp-ep",
            "ici_fsdp_parallelism=16",
            "ici_context_parallelism=8",
            "ici_expert_parallelism=1",
        ],
        "candidate": ["custom_mesh_and_rule=olmo35-ep-as-cp", "ici_fsdp_parallelism=16", "ici_expert_parallelism=8"],
        "reference": [
            "custom_mesh_and_rule=olmo35-cp-ep",
            "ici_fsdp_parallelism=16",
            "ici_context_parallelism=2",
            "ici_expert_parallelism=4",
        ],
    }[name]
    args = [
        "check_mesh_shapes.py",
        "/deps/src/maxtext/configs/post_train/sft.yml",
        "run_name=h127-offline-mesh",
        "base_output_directory=/tmp/h127-offline",
        "model_name=olmo35-tiny",
        "override_model_config=true",
        "compile_topology=v4-256",
        "compile_topology_num_slices=1",
        "per_device_batch_size=0.125",
        "gradient_accumulation_steps=1",
        "max_target_length=65536",
        "steps=6",
        "enable_checkpointing=false",
        "context_parallel_load_balance=false",
        "allow_split_physical_axes=true",
        "load_balance_loss_weight=0",
        "emo_enabled=false",
        *mesh_args,
    ]
    config = pyconfig.initialize_pydantic(args)
    mesh = train_compile.get_topology_mesh(config)
    assert mesh.size == 128 and config.micro_batch_size_to_train_on == 16
    # KDA head/value axes have exact, nonempty shards in both small-slice meshes.
    assert config.gdn_num_key_heads % mesh.shape["expert"] == 0
    assert config.gdn_value_head_dim % mesh.shape.get("context", 1) == 0
    assert config.num_experts % (mesh.shape["expert"] * mesh.shape.get("context", 1)) == 0
    processes = input_pipeline_interface.get_process_loading_real_data(
        config.data_sharding,
        config.global_batch_size_to_load,
        config.global_batch_size_to_train_on,
        config.max_target_length,
        mesh,
    )
    indices = sharding.NamedSharding(mesh, sharding.PartitionSpec(*config.data_sharding)).devices_indices_map(
        (config.global_batch_size_to_load, config.max_target_length)
    )
    host_rows = {}
    for process in processes:
        devices = [d for d in mesh.devices.flat if d.process_index == process]
        host_rows[process] = [
            row < config.global_batch_size_to_train_on
            for device in devices
            for row in range(*indices[device][0].indices(config.global_batch_size_to_load))
        ]
    receipt = {
        "name": name,
        "argv": args,
        "mesh": dict(mesh.shape),
        "devices": mesh.size,
        "batch": config.micro_batch_size_to_train_on,
        "sequence": config.max_target_length,
        "kda_heads_per_shard": config.gdn_num_key_heads // mesh.shape["expert"],
        "kda_values_per_shard": config.gdn_value_head_dim // mesh.shape.get("context", 1),
        "effective_expert_partitions": mesh.shape["expert"] * mesh.shape.get("context", 1),
        "data_processes": processes,
        "data_host_keep_rows_in_mesh_order": host_rows,
        "data_host_global_rows": {
            str(p): [
                r
                for d in mesh.devices.flat
                if d.process_index == p
                for r in range(*indices[d][0].indices(config.global_batch_size_to_load))
            ]
            for p in processes
        },
        "scope": "Offline mesh construction and divisibility only; compilation/runtime/CE remain untested",
    }
    Path(f"/work/{name}-mesh-shape.json").write_text(json.dumps(receipt, indent=2))
    print("MESH-SHAPE-VALID", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
