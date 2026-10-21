"""Overlays for H103 (4B, 4x batch at peak 1e-4 on the 8x8x8 shim) and H104 (7B, the same recipe on H089's overlay).

H103: from h078/bb4-lr1e4-s<seed> (6a38852e). Only RUN_NAME/names, checkpoint_period 200 (~15 min at 4.27 s/step),
ragged_buffer_factor F and num_vocab_tiling T change. P1 smokes: 5 steps, scratch output, no checkpoint, maxRestarts 0.
H104: from h089/olmo3-7b-s<seed> (2a6be850, copied to h089src/). GLOBAL_BATCH 64, STEPS = LR_SCHEDULE_STEPS 2942, LR 1e-4,
the r2 data shim 8:2:128:32:4 (hashing stays on), the big-slice image, and P1 candidate A/B/C's slice and mesh.
  python mk_h103_h104.py h103-p1 | h103-prod <seed> <F> <T> <suffix> | h103-prod-dc <seed> <suffix> | h104-p1 | h104-smoke <cand> | h104-prod <cand> <seed>"""
import pathlib, sys

K = pathlib.Path(__file__).parent
BIG = "51f8cc4b-p907d86f3-rfb46e215e-bs3ec936c3"
DC = BIG + "-dc88c6c7ea"  # BIG + the logging-only log_moe_dropped_tokens probe (abhishekraok/maxtext 1e612a76; H103 Amendment 1)
FILES = ("jobset.yaml", "kustomization.yaml", "train_sft.sh")


def sub(t, a, b):
    assert t.count(a) == 1, a
    return t.replace(a, b)


def write(src, dst, old, new, edit, head):
    dst.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        t = (src / f).read_text()
        t = edit(f, t)
        if f == "kustomization.yaml":
            t = head + t
        (dst / f).write_text(t.replace(old, new))
    print(dst)


def h103(seed, F, T, name, smoke, dropcount=False, smoke_steps=5):
    src = K / f"h078/bb4-lr1e4-s{seed}"

    def edit(f, t):
        if f == "jobset.yaml" and smoke:
            t = sub(t, "    maxRestarts: 3\n", "    maxRestarts: 0\n")
        if f == "kustomization.yaml":
            t = sub(t, " ragged_buffer_factor=3.0 ", f" ragged_buffer_factor={F} ")
            t = sub(t, " num_vocab_tiling=16 ", f" num_vocab_tiling={T} ")
            if smoke:
                t = sub(t, "  - OUTPUT_DIR=gs://ai2-olmo-dev/post-training/runs\n",
                        "  - OUTPUT_DIR=gs://ai2-olmo-dev/post-training/scratch\n")
                t = sub(t, "  - STEPS=2942\n", f"  - STEPS={smoke_steps}\n")
                t = sub(t, " checkpoint_period=250 ", " checkpoint_period=100000 ")
            else:
                t = sub(t, " checkpoint_period=250 ", " checkpoint_period=200 ")
        if f == "kustomization.yaml" and dropcount:
            t = sub(t, " log_required_ragged_buffer_factor=true ", " log_required_ragged_buffer_factor=true log_moe_dropped_tokens=true ")
            t = sub(t, f"  newTag: {BIG}\n", f"  newTag: {DC}\n")
        if f == "kustomization.yaml":
            for keep in ("  - GLOBAL_BATCH=64\n", "  - LR=1e-4\n", "  - LR_SCHEDULE_STEPS=2942\n",
                         "  - MAXTEXT_R2_DATA_EMULATION=8:2:128:32:4\n", f"  newTag: {DC if dropcount else BIG}\n"):
                assert keep in t, keep
        return t

    head = (f"# H103 {name} (tpu-worker, 2026-10-21; mk_h103_h104.py): H078's bb4-lr1e4-s{seed} with ragged_buffer_factor {F},\n"
            f"# num_vocab_tiling {T}{f', a {smoke_steps}-step smoke (scratch, no checkpoint)' if smoke else ', checkpoint_period 200'}"
            f"{'; Amendment 1: log_moe_dropped_tokens on the dropcount image' if dropcount else ''}.\n")
    write(src, K / "h103" / name, f"h078-bb4-lr1e4-s{seed}", f"h103-{name}", edit, head)


MESH = {  # candidate: (topology, hosts, devices, per-device batch, mesh args, checkpoint_period)
    "A": ("8x8x8", 128, 512, "0.125", "ici_fsdp_parallelism=16 ici_tensor_parallelism=8 ici_context_parallelism=4 allow_split_physical_axes=true", 100),
    "B": ("8x8x8", 128, 512, "0.125", "ici_fsdp_parallelism=64 ici_tensor_parallelism=8 allow_split_physical_axes=true", 100),
    "C": ("8x8x16", 256, 1024, "0.0625", "ici_fsdp_parallelism=64 ici_tensor_parallelism=8 ici_context_parallelism=2 allow_split_physical_axes=true", 180),
    # F1 (2026-10-21, after the 8x8x8 was resliced): AOT compile on CPU fits (temp 19.2 GB vs mesh B's 35.7), ~4.5 s/step est.
    "F1": ("8x8x16", 256, 1024, "0.0625", "ici_fsdp_parallelism=64 ici_tensor_parallelism=16 allow_split_physical_axes=true", 200),
}


def h104(cand, seed, name, smoke):
    topo, hosts, dev, pdb, mesh, ckpt = MESH[cand]
    src = K / f"h089src/olmo3-7b-s{seed}"

    def edit(f, t):
        if f == "jobset.yaml":
            t = sub(t, "        parallelism: 32 # Hosts per slice.\n", f"        parallelism: {hosts} # Hosts per slice.\n")
            t = sub(t, "        completions: 32\n", f"        completions: {hosts}\n")
            t = sub(t, "cloud.google.com/gke-tpu-topology: 4x4x8\n", f"cloud.google.com/gke-tpu-topology: {topo}\n")
            if smoke:
                t = sub(t, "    maxRestarts: 3\n", "    maxRestarts: 0\n")
        if f == "kustomization.yaml":
            t = sub(t, f"  - RUN_NAME=olmo3-7b-sft-h089-s{seed}\n", f"  - RUN_NAME=olmo3-7b-sft-h104-{name}\n")
            t = sub(t, "  - TOTAL_DEVICES=128\n", f"  - TOTAL_DEVICES={dev}\n")
            t = sub(t, "  - GLOBAL_BATCH=16\n", "  - GLOBAL_BATCH=64\n")
            t = sub(t, "  - PER_DEVICE_BATCH=0.125\n", f"  - PER_DEVICE_BATCH={pdb}\n")
            t = sub(t, "  - LR=5e-5\n", "  - LR=1e-4\n")
            t = sub(t, "  - LR_SCHEDULE_STEPS=11768\n", "  - LR_SCHEDULE_STEPS=2942\n")
            t = sub(t, "  - STEPS=11768\n", "  - STEPS=30\n" if smoke else "  - STEPS=2942\n")
            t = sub(t, " checkpoint_period=300 ", " checkpoint_period=100000 " if smoke else f" checkpoint_period={ckpt} ")
            t = sub(t, " ici_fsdp_parallelism=16 ici_tensor_parallelism=8 ", f" {mesh} ")
            t = sub(t, "  - TOKENIZER_PATH=", "  - MAXTEXT_R2_DATA_EMULATION=8:2:128:32:4\n  - TOKENIZER_PATH=")
            t = sub(t, "  newTag: 51f8cc4b-p907d86f3-rfb46e215e\n", f"  newTag: {BIG}\n")
            if smoke:
                t = sub(t, "  - OUTPUT_DIR=gs://ai2-olmo-dev/post-training/runs\n",
                        "  - OUTPUT_DIR=gs://ai2-olmo-dev/post-training/scratch\n")
            for keep in ("  - MAXTEXT_LOG_BATCH_HASH=1", " remat_policy=minimal ", "  - MODEL_NAME=olmo3-7b\n",
                         f"TRAIN_FILES=/mnt/post-training/data/h067/a0prime-s{seed}/", "  - NUM_EPOCH=2\n"):
                assert keep in t, keep
        return t

    head = (f"# H104 {name} (tpu-worker, 2026-10-21; mk_h103_h104.py): H089's olmo3-7b-s{seed} at 4x batch (64 rows, 2942 steps,\n"
            f"# peak 1e-4), r2 data shim 8:2:128:32:4, candidate {cand}: {topo}, {mesh}"
            f"{'; a 30-step P1 smoke (scratch, no checkpoint)' if smoke else f'; checkpoint_period {ckpt}'}.\n")
    write(src, K / "h104" / name, f"h089-olmo3-7b-s{seed}", f"h104-{name}", edit, head)


mode = sys.argv[1]
if mode == "h103-p1":
    for T in (16, 32):
        for F in ("5.0", "4.5", "4.0"):
            h103(1, F, T, f"p1-f{F.replace('.', '')}-t{T}", True)
elif mode == "h103-prod":
    seed, F, T, suffix = int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
    h103(seed, F, T, f"bb4-lr1e4-s{seed}{suffix}", False)
elif mode == "h103-prod-dc":  # H103 Amendment 1: factor 3.0 (s0r's), every step's dropped rows logged
    seed, suffix = int(sys.argv[2]), sys.argv[3]
    h103(seed, "3.0", 16, f"bb4-lr1e4-s{seed}{suffix}", False, dropcount=True)
elif mode == "h103-dc-smoke":  # Amendment 1 positive control: seed 1 at factor F on the dropcount image
    F, steps = sys.argv[2], int(sys.argv[3])
    h103(1, F, 16, f"dc-f{F.replace('.', '')}-s1", True, dropcount=True, smoke_steps=steps)
elif mode == "h104-p1":
    for c in MESH:
        h104(c, 0, f"p1-{c.lower()}", True)
elif mode == "h104-smoke":  # one candidate's 30-step smoke
    cand = sys.argv[2]
    h104(cand, 0, f"p1-{cand.lower()}", True)
elif mode == "h104-prod":
    cand, seed = sys.argv[2], int(sys.argv[3])
    h104(cand, seed, f"bb4-lr1e4-7b-{cand.lower()}-s{seed}", False)
