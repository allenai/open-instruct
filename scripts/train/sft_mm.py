"""Launch one Olmo 3.5 multimodal SFT run of a named recipe on Beaker (GPU).

    uv run python scripts/train/sft_mm.py --recipe 4n-emo256k
    uv run python scripts/train/sft_mm.py --recipe 4n-emo256k --seed 34522
    uv run python scripts/train/sft_mm.py --recipe 4n-emo256k --steps 30 --dry_run

The training code is OLMo-core's multimodal SFT recipe (allenai/OLMo-core#942), not open-instruct's trainer: every
replica fetches OLMo-core at OLMO_CORE_SHA, installs it into the pinned OLMo-core Beaker image and runs
src/scripts/train/Olmo35-Multimodal-SFT.py, as the ledger's H230/H231 runs did. Nothing needs building, and
open-instruct's own olmo-core pin (pyproject.toml) is untouched. Running the same command again resumes from the
latest checkpoint in the run's output directory, which has no timestamp.

See docs/algorithms/olmo35_mm_sft.md.
"""

import argparse
import datetime
import json
import os
import re
import shlex
import subprocess
import tempfile

# allenai/OLMo-core#942 (branch jasonr/olmo35-multimodal-sft) head: H230/H231 trained at its parent fd9bba1ba; this
# commit adds only #964 (HF export writes the <|im_end|> stop token), no training change. It is not on
# OLMo-core main: if that branch is squash-merged and deleted, GitHub may stop serving this commit.
OLMO_CORE_REPO = "https://github.com/allenai/OLMo-core.git"
OLMO_CORE_SHA = "7074cf408a157b71c5c74fbc749c4a757e047367"
# akshitab/olmo-core-tch2130cu130-fa4-rma-2026-09-11 (torch 2.13 + CUDA 13, FA4): the image H230/H231 ran on.
IMAGE = "01M28XX5N4KM6PD0G81KC41BDW"
CLUSTER = "ai2/holmes"
GPUS_PER_NODE = 8
BUCKETS = (
    "oe-training-default",
    "olmo-3p5-checkpoints",
    "olmo-3p5-mid-checkpoints",
    "olmo-3p5-post-checkpoints",
    "oe-adapt-default",
    "dolma-3p5",
    "oe-mm-default",
)
HF_CACHE = "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/hf_cache"
# The env of H231's launcher (OLMo-core src/scripts/train/olmo35_mm_sft_beaker.py at OLMO_CORE_SHA).
ENV = {
    "OLMO_SHARED_FS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "OLMO_SYMM_VDEV2D_AUTO_BUILD": "1",
    "HF_HOME": f"{HF_CACHE}/huggingface",
    # Only a writable index cache for the OCR/caption tar sources; the shared cache mason.py uses for every job.
    "HF_DATASETS_CACHE": "/weka/oe-adapt-default/allennlp/.cache/huggingface",
    "HF_HUB_CACHE": f"{HF_CACHE}/hub",
    "HF_HUB_OFFLINE": "1",
    "GRPC_ENABLE_FORK_SUPPORT": "false",
    "NCCL_DEBUG": "ERROR",
    "NCCL_SOCKET_IFNAME": "ib",
    "NCCL_IB_HCA": "^=mlx5_bond_0",
    "S3_PROFILE": "default",
}
# Every recipe: permanent saves half way and at the end, ephemeral saves for resuming every 250 steps (~16 min).
COMMON_OVERRIDES = ["--recipe.save_fraction=0.5", "--recipe.ephemeral_save_interval=250"]
DEFAULT_SEED = 34521

# Each recipe is an arm of ledger hypotheses H230/H231: its OLMo-core --recipe.* overrides, verbatim from the arm's
# `launch:` field. The step count is derived by the recipe from the base and the data dose (10,962 at the default
# 3.5e9 image tokens; 15,568 for tok5), so it is recorded here only for the docs and --steps.
RECIPES = {
    # The recommended recipe (H231 4n-emo256k): base B2e, the 256K causal-EMo multimodal midtrain (step 22240), its
    # causal EMo pools on in SFT; C12 + long-context replay + Molmo2 stage-2 image mixture v1 (3.5e9 image tokens);
    # 32 x 65,536-token batches, peak LR 2e-4 (connector 4e-4, ViT 2e-5), 10,962 steps. Data seeds 34521 and 34522
    # both ran. About 3.9 s/step on 4 x 8 B300.
    "4n-emo256k": {"overrides": ["--recipe.base=256k-emo-causal"], "steps": 10962, "timeout": "24h"},
    # H231 Amendment 1: 4n-emo256k with the image tokens' loss weighted so images carry 5% of the loss.
    "4n-emo256k-up5": {
        "overrides": ["--recipe.base=256k-emo-causal", "--recipe.image_loss_share=0.05"],
        "steps": 10962,
        "timeout": "24h",
    },
    # The dose sweep (ledger, proposed): 4n-emo256k with images at 3% of the loss (image tokens' weight x2.088).
    "4n-emo256k-up3": {
        "overrides": ["--recipe.base=256k-emo-causal", "--recipe.image_loss_share=0.03"],
        "steps": 10962,
        "timeout": "24h",
    },
    # H231 Amendment 1: 4n-emo256k with the image dose raised to 12.675B tokens (images ~5% of the loss mass).
    "4n-emo256k-tok5": {
        "overrides": ["--recipe.base=256k-emo-causal", "--recipe.image_dose_tokens=1.2675e10"],
        "steps": 15568,
        "timeout": "30h",
    },
    # H272 (proposed): the text-only control. 4n-emo256k's recipe in the script's text-only mode: the same C12 +
    # long-context replay text in open-instruct's data order, no images, so a text read against 4n-emo256k differs only
    # in the images. Steps are derived from the text alone: ceil(294,560 / 32) = 9,205.
    "4n-emo256k-textonly": {
        "overrides": ["--recipe.base=256k-emo-causal", "--recipe.text_order=open_instruct"],
        "steps": 9205,
        "timeout": "24h",
    },
    # H230 4n-base: the same recipe on base B1 (the 64K multimodal midtrain, step 20000).
    "4n-base": {"overrides": ["--recipe.base=64k"], "steps": 10962, "timeout": "16h"},
    # H230 4n-vis10: 4n-base with images at 10% of the loss.
    "4n-vis10": {
        "overrides": ["--recipe.base=64k", "--recipe.image_loss_share=0.10"],
        "steps": 10962,
        "timeout": "16h",
    },
    # H230 4n-vitlr: 4n-base with the ViT at LR / 2 instead of LR / 10.
    "4n-vitlr": {"overrides": ["--recipe.base=64k", "--recipe.vision_lr_divisor=2"], "steps": 10962, "timeout": "16h"},
    # H230 4n-hires: 4n-base with Molmo2's high-resolution draw (24 crops) on the document/chart/text sources.
    "4n-hires": {
        "overrides": [
            "--recipe.base=64k",
            "--recipe.high_res_sources=[doc_qa,info_qa,chart_qa_weighted,text_vqa,st_qa]",
        ],
        "steps": 10962,
        "timeout": "16h",
    },
    # H230 4n-rebal: 4n-base with pointing/counting cut to anchors, their tokens moved to multi-image/doc sources.
    "4n-rebal": {
        "overrides": ["--recipe.base=64k", "--recipe.image_data=stage2_v1_rebal"],
        "steps": 10962,
        "timeout": "16h",
    },
}
NODES = 4


def job_script(run_name: str, overrides: list[str], nodes: int) -> str:
    """The shell every replica runs: fetch and install the pinned OLMo-core, then torchrun its recipe."""
    src = "/tmp/olmo-core"
    train = [
        "torchrun", f"--nnodes={nodes}", "--node_rank=$BEAKER_REPLICA_RANK",
        "--master_addr=$BEAKER_LEADER_REPLICA_HOSTNAME", "--master_port=29400", f"--nproc_per_node={GPUS_PER_NODE}",
        f"{src}/src/scripts/train/Olmo35-Multimodal-SFT.py", "train", run_name, CLUSTER,
    ]  # fmt: skip
    # The $VARS above must expand, so only the run's own arguments are quoted.
    torchrun = " ".join(train[:6]) + " " + shlex.join([*train[6:], *overrides, f"--recipe.num_nodes={nodes}"])
    return f"""set -euo pipefail
# Single-replica jobs get neither variable from Beaker.
: "${{BEAKER_REPLICA_RANK:=0}}" "${{BEAKER_LEADER_REPLICA_HOSTNAME:=localhost}}"
echo "MMSFT-JOB-START $(date -u +%FT%TZ) job ${{BEAKER_JOB_ID:-}} replica $BEAKER_REPLICA_RANK"
git init -q {src}
git -C {src} fetch -q --depth 1 {OLMO_CORE_REPO} {OLMO_CORE_SHA}
git -C {src} checkout -q FETCH_HEAD
test "$(git -C {src} rev-parse HEAD)" = {OLMO_CORE_SHA}
echo "code $(git -C {src} rev-parse HEAD)"
pip install -q -e "{src}[fla,kernel-fun]" 'datasets>=4,<6' pypdfium2 h5py 2>&1 | tail -2
export PYTHONPATH={src}/src
python -c "import olmo_core, torch; print(olmo_core.__file__, torch.__version__, torch.cuda.get_device_name())"
exec {torchrun}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    parser.add_argument(
        "--checkpoint", help="Native OLMo-core multimodal checkpoint to start from. Default: the recipe's base."
    )
    parser.add_argument(
        "--name",
        help="Run name; also the output directory. Default: olmo35-mmsft-<recipe>-s<seed>, plus -<steps>st with "
        "--steps.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Data seed (data order).")
    parser.add_argument(
        "--steps",
        type=int,
        help="Stop after this many steps. The LR schedule keeps the recipe's full horizon, so this is for smoke tests.",
    )
    parser.add_argument("--workspace", default="ai2/olmo-instruct")
    parser.add_argument(
        "--output_root",
        help="Where the run saves (<output_root>/<name>). Default: the user's deletable scratch on oe-adapt-default; "
        "a run to keep saves under /weka/olmo-3p5-post-checkpoints/<user>/ (skills/storage in the ledger).",
    )
    parser.add_argument(
        "--work_dir", help="The data loader's per-run index dir. Default: beside the default output root."
    )
    parser.add_argument("--priority", default="high")
    parser.add_argument(
        "--min_runtime",
        default="8h",
        help="Beaker minRuntime. H231's runs with 1h were preempted repeatedly by allocation balancing; 8h (the "
        "maximum) is what finished them.",
    )
    parser.add_argument(
        "--gate_reference",
        action="store_true",
        help="Write per-rank batch hashes for every step and step 1's eval-mode CE to <run dir>/gates: the GPU "
        "reference for a TPU run's step-1 and batch-order gates.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the Beaker spec instead of submitting it.")
    args = parser.parse_args()

    recipe = RECIPES[args.recipe]
    if args.checkpoint and not args.name:
        parser.error("--checkpoint needs a --name, so the run does not resume into the default run's directory.")
    # A short run gets its own default directory, so a later full run does not resume from it.
    name = args.name or f"olmo35-mmsft-{args.recipe}-s{args.seed}" + ("" if args.steps is None else f"-{args.steps}st")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        parser.error(f"--name {name!r} must start with a letter or digit, then letters, digits, '.', '_' or '-'.")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be at least 1.")
    try:
        whoami = subprocess.run(["beaker", "account", "whoami", "--format", "json"], capture_output=True, text=True)
        user = json.loads(whoami.stdout)[0]["name"]
    except (OSError, ValueError, LookupError):
        if not args.dry_run:
            parser.error("needs a working `beaker account whoami`.")
        user = "<user>"
    scratch = f"/weka/oe-adapt-default/allennlp/deletable_checkpoint/{user}"
    # The recipe saves to <output_root>/<run name>; the work dir holds the data loader's per-run indices.
    output_root = args.output_root or f"{scratch}/sft"
    work_dir = args.work_dir or f"{scratch}/sft_mm_work"

    overrides = list(recipe["overrides"])
    if args.checkpoint:
        # --recipe.base takes a key of the recipe's BASES or a checkpoint directory.
        overrides = [f"--recipe.base={args.checkpoint}" if o.startswith("--recipe.base=") else o for o in overrides]
    overrides += [
        *COMMON_OVERRIDES,
        f"--recipe.data_seed={args.seed}",
        f"--recipe.output_root={output_root}",
        f"--recipe.work_dir={work_dir}",
    ]
    if args.steps is not None:
        overrides.append(f"--recipe.stop_at_step={args.steps}")
    if args.gate_reference:
        # The GPU side of the placement gates (docs/algorithms/olmo35_mm_sft_tpu.md): per-rank batch hashes for
        # every step, step 1's batch, and step 1 scored in eval mode (EMo eval pool = all 512 experts).
        overrides += [f"--recipe.batch_hash_dir={output_root}/{name}/gates", "--recipe.step1_reference=true"]

    task = {
        "name": name,
        "image": {"beaker": IMAGE},
        "command": ["/bin/bash", "-c"],
        "arguments": [job_script(name, overrides, NODES)],
        "envVars": [{"name": k, "value": v} for k, v in ENV.items()],
        "datasets": [{"mountPath": f"/weka/{b}", "source": {"weka": b}} for b in BUCKETS],
        "result": {"path": "/output"},
        "resources": {"gpuCount": GPUS_PER_NODE, "sharedMemory": "128GiB"},
        # autoResume restarts the job after a preemption; it resumes from the run's latest checkpoint.
        "context": {"priority": args.priority, "autoResume": True, "minRuntime": args.min_runtime},
        "constraints": {"cluster": [CLUSTER]},
        "timeout": recipe["timeout"],
        # NCCL_SOCKET_IFNAME=ib names the host's interfaces.
        "hostNetworking": True,
        "replicas": NODES,
        "leaderSelection": True,
        "propagateFailure": True,
        "propagatePreemption": True,
        "synchronizedStartTimeout": "20m",
    }
    steps = recipe["steps"] if args.steps is None else args.steps
    spec = {
        "version": "v2",
        "budget": "ai2/oe-other",
        "description": f"Olmo 3.5 multimodal SFT recipe {args.recipe}: {name}, {steps} steps, data seed {args.seed}",
        "tasks": [task],
    }
    print(f"Output: {output_root}/{name}", flush=True)
    if args.dry_run:
        print(json.dumps(spec, indent=1))
        return
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f, indent=1)
    # Experiment names are unique per workspace, so a rerun (a resume) gets a fresh one.
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    try:
        subprocess.run(
            ["beaker", "experiment", "create", f.name, "--workspace", args.workspace, "--name", f"{name}-{stamp}"],
            check=True,
        )
    finally:
        os.unlink(f.name)


if __name__ == "__main__":
    main()
