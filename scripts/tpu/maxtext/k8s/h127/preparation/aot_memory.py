"""Offline memory feasibility estimate; pretraining compile is not Tunix parity evidence."""

import json
import sys
from pathlib import Path

import yaml
from maxtext.trainers.pre_train import train_compile


def main():
    name = sys.argv[1]
    assert name in ("candidate", "reference-cp8")
    spec = yaml.safe_load(Path("/work/control-config.yaml").read_text())
    env = dict(item.split("=", 1) for item in spec["configMapGenerator"][1]["literals"])
    extra = dict(item.split("=", 1) for item in env["EXTRA_ARGS"].split())
    extra.update(enable_checkpointing="false", load_balance_loss_weight="0")
    if name == "reference-cp8":
        extra.update(custom_mesh_and_rule="olmo35-cp-ep", ici_context_parallelism="8", ici_expert_parallelism="1")
    args = [
        "aot_memory.py",
        "/deps/src/maxtext/configs/post_train/sft.yml",
        "run_name=h127-aot-" + name,
        "base_output_directory=/tmp/h127-aot",
        "compile_topology=v4-256",
        "compile_topology_num_slices=1",
        "model_name=olmo35-tiny",
        "per_device_batch_size=0.125",
        "gradient_accumulation_steps=1",
        "max_target_length=65536",
        "packing=true",
        "steps=20",
        "learning_rate_schedule_steps=11768",
        "learning_rate=5e-5",
        "lr_schedule_type=wsd",
        "warmup_steps_fraction=0.03",
        "wsd_decay_steps_fraction=0.97",
        "wsd_decay_style=linear",
        "learning_rate_final_fraction=0",
        "opt_type=adamw",
        "adam_b1=0.9",
        "adam_b2=0.95",
        "adam_eps=1e-8",
        "adam_weight_decay=0",
        "gradient_clipping_threshold=1",
        "dtype=bfloat16",
        "weight_dtype=float32",
        "enable_dropout=false",
        *(f"{k}={v}" for k, v in extra.items()),
    ]
    original = train_compile.jit_and_compile

    def compile_and_report(*args, **kwargs):
        compiled = original(*args, **kwargs)
        memory = compiled.memory_analysis()
        fields = ["argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes"]
        report = {field: int(getattr(memory, field)) for field in fields}
        report["total_bytes"] = (
            report["argument_size_in_bytes"]
            + report["output_size_in_bytes"]
            - report["alias_size_in_bytes"]
            + report["temp_size_in_bytes"]
        )
        report["scope"] = (
            "Offline pretraining-step compiler estimate, not Tunix SFT runtime or numerical qualification"
        )
        Path(f"/work/{name}-aot-memory.json").write_text(json.dumps(report, indent=2))
        print("AOT-MEMORY", json.dumps(report), flush=True)
        return compiled

    train_compile.jit_and_compile = compile_and_report
    Path(f"/work/{name}-aot-args.json").write_text(json.dumps(args, indent=2))
    oom = train_compile.is_oom(args)
    print("AOT-FINISHED", name, "oom", oom, flush=True)
    if oom:
        raise RuntimeError("Offline compile reports accelerator out of memory")


if __name__ == "__main__":
    main()
