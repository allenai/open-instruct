"""H072 TPU overlays: r2's overlay (open-instruct 432c8d43, scripts/tpu/maxtext/k8s/olmo35-sft-4x4x8) with the batch-hash
image and MAXTEXT_LOG_BATCH_HASH=1, changing only the JobSet/ConfigMap names, queue, restarts, RUN_NAME, OUTPUT_DIR,
TRAIN_FILES, STEPS, NUM_EPOCH and the checkpoint settings.
  python mk_overlay.py <name> <run_name> <train_files> <steps> <num_epoch> <checkpoint_period> [queue] [max_restarts]"""
import os
import subprocess
import sys

name, run, files, steps, epochs, period = sys.argv[1:7]
queue = sys.argv[7] if len(sys.argv) > 7 else "dev"
restarts = sys.argv[8] if len(sys.argv) > 8 else "0"
IMG_TAG = os.environ.get("IMG_TAG", "51f8cc4b-p907d86f3-bh535e5782")
src = "scripts/tpu/maxtext/k8s/olmo35-sft-4x4x8"
show = lambda f: subprocess.run(["git", "-C", os.path.expanduser("~/repos/open-instruct"), "show", f"432c8d43:{src}/{f}"],
                                check=True, capture_output=True, text=True).stdout
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k8s", name)
os.makedirs(out, exist_ok=True)
js = show("jobset.yaml")
js = js.replace("name: olmo35-sft\n", f"name: {name}\n", 1).replace("queue-name: post-training", f"queue-name: {queue}")
js = js.replace("maxRestarts: 3", f"maxRestarts: {restarts}").replace("olmo35-sft-env", f"{name}-env")
js = js.replace("olmo35-sft-launcher", f"{name}-launcher")
k = show("kustomization.yaml")
k = k.replace("olmo35-sft-launcher", f"{name}-launcher").replace("olmo35-sft-env", f"{name}-env")
k = k.replace("RUN_NAME=olmo35-sft-h045-tpu-full-r2", f"RUN_NAME={run}")
PROD = os.environ.get("PROD") == "1"  # production: r2's output dir and checkpoint settings unchanged
if not PROD:
    k = k.replace("OUTPUT_DIR=gs://ai2-olmo-dev/post-training/runs", "OUTPUT_DIR=gs://ai2-olmo-dev/post-training/scratch")
k = k.replace("TRAIN_FILES=/mnt/post-training/data/dolci-think-sft-shuf-s0/*.parquet", f"TRAIN_FILES={files}")
k = k.replace("  - STEPS=11768\n", f"  - STEPS={steps}\n").replace("  - NUM_EPOCH=2\n", f"  - NUM_EPOCH={epochs}\n")
if not PROD:
    k = k.replace("checkpoint_period=300 max_num_checkpoints_to_keep=2", f"checkpoint_period={period} max_num_checkpoints_to_keep=2")
k = k.replace("  - TPU_MEGACORE=MEGACORE_DENSE\n", "  - TPU_MEGACORE=MEGACORE_DENSE\n  - MAXTEXT_LOG_BATCH_HASH=1\n")
k = k.replace("newTag: 51f8cc4b-p907d86f3", f"newTag: {IMG_TAG}")
for needle in (f"RUN_NAME={run}", f"STEPS={steps}", f"NUM_EPOCH={epochs}", "checkpoint_period=300" if PROD else f"checkpoint_period={period}", "MAXTEXT_LOG_BATCH_HASH=1",
               f"newTag: {IMG_TAG}", f"TRAIN_FILES={files}", "LR_SCHEDULE_STEPS=11768"):
    assert needle in k, needle
assert f"queue-name: {queue}" in js and f"name: {name}\n" in js
open(f"{out}/jobset.yaml", "w").write(js)
open(f"{out}/kustomization.yaml", "w").write(k)
open(f"{out}/train_sft.sh", "w").write(show("train_sft.sh"))
print(out)
