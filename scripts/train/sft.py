"""Launch one SFT run of a named recipe on Beaker.

    uv run python scripts/train/sft.py --recipe c12r
    uv run python scripts/train/sft.py --recipe c12r --checkpoint <olmo-core DCP dir> --name my-run

Each recipe pins the Beaker image that trained it, so its training code comes from that image, not from this
checkout: nothing needs building or committing first. Running the same command again resumes from the latest
checkpoint in the run's output directory, which has no timestamp.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

C12_PARQUETS = "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/data/parquets-a3"
RECIPES = {
    # Olmo 3.5 small. C12 SFT on B1 at peak LR 2e-4 for half an epoch, plus 10% long-context replay: the arm
    # b1-c12r-gpu of ledger hypothesis H182, which serves as the control for H183-H186. 2 x 8 B300 on holmes,
    # about 9 h of training.
    "c12r": {
        "image": "01M5KX4ZAN00BJ00ARS7FNG79T",
        "checkpoint": "/weka/olmo-3p5-post-checkpoints/bases/olmo35-small-mm-ckpt1-64k/step20000-lm-olmocore/model_and_optim",
        # The first step at which seed 34521's loader order has consumed half an epoch of C12 (H182's R4' gate).
        # Other seeds land within a few steps of it.
        "steps": 18410,
        "seed": 34521,
        # The frozen, pre-tokenized C12 cache. The job dies before training if the mixer, tokenizer, template or
        # --seed below resolve to a different cache.
        "cache": "bf82a8fab8-6068a350-rowaligned",
        "train_args": [
            "--config_name",
            "scripts/train/debug/kda_hero_small_sft.json",
            "--tokenizer_name_or_path",
            "allenai/dolma2-tokenizer-olmo35",
            "--tokenizer_revision",
            "8b9717061fae09d5be814373d189919a62a9a00d",
            "--chat_template_name",
            "olmo35",
            "--reserved_slot_tokens",
            "<think>",
            "</think>",
            "--row_aligned_parts",
            "--document_boundaries_from_metadata",
            "--max_seq_length",
            "65536",
            "--per_device_train_batch_size",
            "1",
            "--gradient_accumulation_steps",
            "1",
            "--learning_rate",
            "2e-4",
            "--warmup_ratio",
            "0.03",
            "--weight_decay",
            "0.0",
            "--max_grad_norm",
            "1.0",
            "--num_epochs",
            "1",
            "--attn_implementation",
            "flash_2",
            "--no_compile_model",
            "--activation_checkpointing_mode",
            "budget",
            "--activation_memory_budget",
            "1",
            "--ephemeral_save_interval",
            "1000",
            "--keep_last_n_checkpoints",
            "2",
            "--dist_timeout_hours",
            "2",
            "--no_save_async",
            "--with_tracking",
            "--logging_steps",
            "1",
            "--mixer_list",
            f"{C12_PARQUETS}/dolci-think-clean-v1-en-dbv-5bd68837-c12a3-9707a865.parquet",
            "1.0",
            f"{C12_PARQUETS}/nemotron-tool-oi-en-bd-825bdb83-c12a3-be9ec34a.parquet",
            "0.862426",
            f"{C12_PARQUETS}/nemotron-sci-v2-h083sel-bd-16f9aaaa-c12a3-25035bd6.parquet",
            "3.03277",
            "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/if/"
            "nemotron-ifchat-v3-if-decontam-1fee8794-c12-4e2daf40.parquet",
            "4.937456",
            # The job's cache dir holds a symlink to the frozen cache.
            "--local_cache_dir",
            "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache",
            # --seed names the cache; the data order is --data_loader_seed.
            "--seed",
            "33333",
            "--extra_numpy_dirs",
            "/weka/olmo-3p5-post-checkpoints/abhishekr/h182/replay-b1c12r",
        ],
        "cluster": "ai2/holmes",
        "nodes": 2,
        "timeout": "14h",
        # The first ephemeral checkpoint (step 1000) lands after about 41 minutes.
        "min_runtime": "45m",
    }
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    parser.add_argument("--checkpoint", help="OLMo-core checkpoint (DCP) to start from. Default: the recipe's base.")
    parser.add_argument("--name", help="Run name; also the output directory. Default: <recipe>-s<seed>.")
    parser.add_argument("--seed", type=int, help="Data-loader seed. Default: the recipe's.")
    parser.add_argument("--steps", type=int, help="Training steps. Default: the recipe's.")
    parser.add_argument("--workspace", default="ai2/olmo-instruct")
    parser.add_argument("--priority", default="high")
    parser.add_argument("--dry_run", action="store_true", help="Print the mason command instead of running it.")
    args = parser.parse_args()

    recipe = RECIPES[args.recipe]
    if args.checkpoint and not args.name:
        parser.error("--checkpoint needs a --name, so the run does not resume into the default run's directory.")
    seed = recipe["seed"] if args.seed is None else args.seed
    steps = recipe["steps"] if args.steps is None else args.steps
    name = args.name or f"{args.recipe}-s{seed}"
    user = json.loads(
        subprocess.run(
            ["beaker", "account", "whoami", "--format", "json"], capture_output=True, text=True, check=True
        ).stdout
    )[0]["name"]
    output_dir = f"/weka/olmo-3p5-post-checkpoints/{user}/sft/{name}"

    nodes = recipe["nodes"]
    train = [
        "torchrun", f"--nnodes={nodes}", "--node_rank=$BEAKER_REPLICA_RANK",
        "--master_addr=$BEAKER_LEADER_REPLICA_HOSTNAME", "--master_port=29400", "--nproc_per_node=8",
        "open_instruct/olmo_core_finetune.py",
        "--model_name_or_path", args.checkpoint or recipe["checkpoint"],
        *recipe["train_args"],
        "--max_train_steps", str(steps),
        # A permanent checkpoint half way through, plus the final one.
        "--checkpointing_steps", str((steps + 1) // 2),
        "--data_loader_seed", str(seed), "--run_name", name, "--output_dir", output_dir,
    ]  # fmt: skip
    mason = [
        sys.executable, "mason.py",
        "--cluster", recipe["cluster"], "--workspace", args.workspace, "--budget", "ai2/oe-other",
        "--image", recipe["image"], "--pure_docker_mode", "--num_nodes", str(nodes), "--gpus", "8",
        "--priority", args.priority, "--min_runtime", recipe["min_runtime"], "--auto_resume",
        "--timeout", recipe["timeout"],
        "--extra_weka_buckets", "olmo-3p5-checkpoints", "olmo-3p5-post-checkpoints",
        "--env", "OLMO_SHARED_FS=1", "--env", f"EXPECTED_NUMPY_CACHE={recipe['cache']}",
        "--task_name", name,
        "--description", f"SFT recipe {args.recipe}: {name}, {steps} steps, seed {seed}",
        "--",
        *train,
    ]  # fmt: skip
    print(f"Output: {output_dir}")
    if args.dry_run:
        print(shlex.join(mason))
        return
    subprocess.run(mason, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
