"""Run the bounded H077 hardware gate or one registered training arm."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from open_instruct import atomic_think

ARMS = ("control", "rounding-only", "atomic-safe", "atomic-historical", "control-repeat")


def link_reference_packing(root, name):
    """Reuse the validated packing in a fresh diagnostic's own loader directory."""
    source = (root / "gate-reference" / "dataset-common").resolve(strict=True)
    if not source.is_dir():
        raise ValueError(f"Reference packing is not a directory: {source}")
    output = root / name
    output.mkdir(parents=True, exist_ok=True)
    target = output / "dataset-common"
    try:
        target.symlink_to(source, target_is_directory=True)
    except FileExistsError:
        # Both node leaders execute this setup; accept only the same link.
        if not target.is_symlink() or target.resolve() != source:
            raise ValueError(f"Diagnostic packing differs from reference: {target}") from None


def train(
    root,
    name,
    mode,
    steps,
    probes,
    save_interval,
    resume=None,
    state_steps=(),
    offline=False,
    packed_cache=None,
    data_root=None,
    verify_resume=False,
    batch_steps=None,
):
    if verify_resume and (resume is None or not offline):
        raise ValueError("Qualified continuation requires an explicit resume and offline measurements")
    output = root / name
    output.mkdir(parents=True, exist_ok=True)
    atomic = mode.startswith("atomic-")
    os.environ["EXPECTED_NUMPY_CACHE"] = "3f12323c3b-6068a350" if atomic else "062b8a3d20-6068a350"
    cmd = [
        "torchrun",
        "--nnodes=2",
        f"--node_rank={os.environ['BEAKER_REPLICA_RANK']}",
        f"--master_addr={os.environ['BEAKER_LEADER_REPLICA_HOSTNAME']}",
        "--master_port=29400",
        "--nproc_per_node=8",
        "open_instruct/olmo_core_finetune.py",
        "--model_name_or_path",
        "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-4t/non-emo/step5961",
        "--config_name",
        "scripts/train/debug/kda_hero_small_sft.json",
        "--tokenizer_name_or_path",
        "allenai/olmo-3-tokenizer-instruct-dev",
        "--chat_template_name",
        "olmo123",
        "--max_seq_length",
        "65536",
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        "5e-5",
        "--warmup_ratio",
        "0.03",
        "--weight_decay",
        "0.0",
        "--max_grad_norm",
        "1.0",
        "--num_epochs",
        "1",
        "--max_train_steps",
        str(steps),
        "--scheduler_horizon",
        "11768",
        "--attn_implementation",
        "flash_2",
        "--no_compile_model",
        "--activation_checkpointing_mode",
        "budget",
        "--activation_memory_budget",
        "1",
        "--checkpointing_steps",
        str(save_interval),
        "--ephemeral_save_interval",
        "256" if steps == 512 else "0",
        "--keep_last_n_checkpoints",
        "-1",
        "--dist_timeout_hours",
        "1",
        "--no_save_async",
        "--with_tracking",
        "--run_name",
        f"h077-{root.name}-{name}",
        "--logging_steps",
        "1",
        "--mixer_list",
        "allenai/Dolci-Think-SFT",
        "1.0",
        "--local_cache_dir",
        "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache",
        "--sft_dataset_manifest",
        str((data_root or root) / ("dataset-atomic.json" if atomic else "dataset-original.json")),
        "--seed",
        "33333",
        "--data_loader_seed",
        "34521",
        "--output_dir",
        str(output),
        "--atomic_audit_mode",
        mode,
        "--atomic_probe_manifest",
        "scripts/train/atomic_think/probes.json",
        "--atomic_probe_steps",
        *map(str, probes),
    ]
    if atomic:
        cmd += ["--reserved_slot_tokens", "<think>", "</think>"]
    if resume:
        cmd += ["--resume_from_checkpoint", str(resume)]
    if state_steps:
        cmd += ["--atomic_state_steps", *map(str, state_steps)]
    if offline:
        cmd += ["--atomic_probe_offline"]
    if verify_resume:
        cmd += ["--atomic_verify_resume"]
    if batch_steps is not None:
        cmd += ["--atomic_batch_steps", *map(str, batch_steps)]
    if packed_cache is not None:
        cmd += ["--packed_sample_cache", str(packed_cache)]
    rank = os.environ["BEAKER_REPLICA_RANK"]
    atomic_think.write_json(output / f"command-node{rank}.json", cmd)
    start = time.time()
    # Keep Beaker logs live and retain a per-execution copy on WEKA.
    with (output / f"train-node{rank}.log").open("w") as log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        if process.wait():
            raise RuntimeError(f"Training failed: {name}, exit {process.returncode}")
    atomic_think.write_json(
        output / f"completion-node{rank}.json",
        {
            "started": start,
            "finished": time.time(),
            "steps": steps,
            "resume": str(resume),
            "checkpoint": str(output / f"step{steps}"),
            "retention": "keep",
        },
    )


def verify_gate(root):
    # Every rank writes these before its torchrun exits. The node leaders verify
    # their own ranks, avoiding races while the other node flushes its reports.
    first = int(os.environ["BEAKER_REPLICA_RANK"]) * 8
    for rank in range(first, first + 8):
        for step in (3, 4):
            for kind in ("batch", "state"):
                filename = f"{kind}-step{step}-rank{rank}.json"
                expected = json.loads((root / "gate-reference" / "atomic-audit" / filename).read_text())
                for name in ("gate-probed", "gate-resumed"):
                    actual = json.loads((root / name / "atomic-audit" / filename).read_text())
                    if expected != actual:
                        raise RuntimeError(f"Gate mismatch: {name}/{filename}; inspect before launching panel")
        resumed = root / "gate-resumed" / "atomic-audit"
        assert not (resumed / f"initialization-rank{rank}.json").exists(), "Resume repeated initialization"
        receipt = json.loads((resumed / f"restore-step2-rank{rank}.json").read_text())
        assert receipt["step"] == 2 and receipt["horizon"] == 11768
    atomic_think.write_json(
        root / f"gate-passed-node{os.environ['BEAKER_REPLICA_RANK']}.json",
        {
            "checks": [
                "batches",
                "model and optimizer state",
                "LR",
                "no reseeding",
                "step 2 restored; continued to 4",
            ],
            "time": time.time(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("gate", "gate-repeat", *ARMS))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.arm == "gate":
        train(args.root, "gate-reference", "control", 4, [999999], 2, state_steps=[3, 4])
        train(args.root, "gate-probed", "control", 4, [0, 1, 2], 4, state_steps=[3, 4])
        train(
            args.root,
            "gate-resumed",
            "control",
            4,
            [2],
            4,
            resume=args.root / "gate-reference" / "step2",
            state_steps=[3, 4],
        )
        verify_gate(args.root)
    elif args.arm == "gate-repeat":
        # Diagnose a failed exact gate without changing training or probe code.
        # These are prerequisite controls, never the scientific A-repeat arm.
        for name in ("gate-reference-repeat", "gate-resumed-unprobed"):
            link_reference_packing(args.root, name)
        train(args.root, "gate-reference-repeat", "control", 4, [999999], 2, state_steps=[3, 4])
        train(
            args.root,
            "gate-resumed-unprobed",
            "control",
            4,
            [999999],
            4,
            resume=args.root / "gate-reference" / "step2",
            state_steps=[3, 4],
        )
    else:
        mode = "control" if args.arm == "control-repeat" else args.arm
        train(args.root, args.arm, mode, 512, [0, 1, 2, 10, 100, 353, 512], 512, state_steps=[256, 512], offline=True)


if __name__ == "__main__":
    main()
