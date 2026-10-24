"""H115 overlays (P5): H076 lr2x s_i's overlay with only data, RUN_NAME, LR, warmup, STEPS, restore path (params only)
and the placement changed, for the four arms x two seeds; ledger hypotheses/H115-*.md, P5 and its 2026-10-23 4x4x8
placement note.

  python mk_h115.py <k8s dir>

Changes from h076/lr2x-s<i>:
- RUN_NAME olmo35-sft-h115-<arm>-s<i>; ConfigMap and JobSet names h115-<arm>-s<i>.
- TRAIN_FILES /mnt/post-training/data/h115/<arm>-s<i>/*.parquet (P4's streams: one file per TPU data host).
- LR 5e-5; WARMUP_FRAC 0.0304, so int(954 x 0.0304) = 29 warmup steps (3% of S; 0.03 would give 28);
  STEPS = LR_SCHEDULE_STEPS = 954.
- LOAD_PARAMETERS_PATH: lr2x s<i>'s final checkpoint (step 11,768), parameters only (MaxText restores the
  `model_params` item; fresh AdamW state).
- Placement: the ep8 4x4x8 terms H115's placement note takes from H102 (ep-as-cp fsdp16 x expert8 and factor 2.5, as
  lr2x already has) on H103 s1n16's drop-count image with log_moe_dropped_tokens=true (mk_ep8dc.py's change).
- (b) only: its rows are one-message LM rows, which need P1's lmrow.patch. The ConfigMap carries the patched
  input_pipeline_utils.py; the launcher checks the image's file is the one P1 patched (sha256 d9e2bac0...), installs
  the patched copy (sha256 fbff555b...) and sets MAXTEXT_SFT_LM_ROLE=lm. (a), (c) and (d) use lr2x's launcher unchanged.
Everything else (batch, sequence, labels, NUM_EPOCH=2, DATA_SEED=0, mesh flags, checkpointing, LIBTPU args) is lr2x's.
"""
import pathlib, sys

K = pathlib.Path(sys.argv[1])
TAG = "51f8cc4b-p907d86f3-rfb46e215e-bs3ec936c3-dc88c6c7ea"
ARMS = {"d": "anchor-data", "a": "long-agentic", "b": "long-replay", "c": "short-agentic"}
IPU_IMAGE = "d9e2bac0f0e86380b8a9984f9a6857f68c600b236d4ed2707f8cb8df709702ec"
IPU_PATCHED = "fbff555bcc2138821aed51854fb0a7b1ea100bde98cf73b531710bdd1d84fd52"
LMROW = '''
# H115 (b) only: install P1's lmrow.patch (one-message LM rows, every token trained) over the image's file, which must
# be the one P1 patched; the patched copy comes from this ConfigMap.
ipu=$(python3 -c "import os, maxtext.input_pipeline.input_pipeline_utils as u; print(os.path.realpath(u.__file__))")
[ "$(sha256sum "$ipu" | cut -d' ' -f1)" = "%s" ] || { echo "LMROW: image input_pipeline_utils.py is not P1's" >&2; exit 4; }
cp /launcher/input_pipeline_utils.py "$ipu"
[ "$(sha256sum "$ipu" | cut -d' ' -f1)" = "%s" ] || { echo "LMROW: patched copy sha mismatch" >&2; exit 4; }
export MAXTEXT_SFT_LM_ROLE=lm
echo "LMROW: lmrow.patch installed ($(sha256sum "$ipu" | cut -c1-16)), MAXTEXT_SFT_LM_ROLE=$MAXTEXT_SFT_LM_ROLE"
''' % (IPU_IMAGE, IPU_PATCHED)


def sub(t, a, b, n=1):
    assert t.count(a) == n, (a, t.count(a))
    return t.replace(a, b)


for s in (0, 1):
    src = K / "h076" / f"lr2x-s{s}"
    for arm, name in ARMS.items():
        j = f"h115-{arm}-s{s}"; dst = K / "h115" / f"{arm}-s{s}"; dst.mkdir(parents=True, exist_ok=True)
        for f in ("jobset.yaml", "kustomization.yaml", "train_sft.sh"):
            t = (src / f).read_text()
            if f != "train_sft.sh":
                t = t.replace(f"h076-lr2x-s{s}", j)
            if f == "kustomization.yaml":
                assert t.count(f"  - RUN_NAME=olmo35-sft-{j}\n") == 1  # from the h076-lr2x-s<i> -> h115-<arm>-s<i> rename
                t = sub(t, f"  - TRAIN_FILES=/mnt/post-training/data/h067/a0prime-s{s}/*.parquet\n",
                        f"  - TRAIN_FILES=/mnt/post-training/data/h115/{arm}-s{s}/*.parquet\n")
                t = sub(t, "  - LR=1e-4\n", "  - LR=5e-5\n")
                t = sub(t, "  - WARMUP_FRAC=0.03\n", "  - WARMUP_FRAC=0.0304\n")
                t = sub(t, "  - LR_SCHEDULE_STEPS=11768\n", "  - LR_SCHEDULE_STEPS=954\n")
                t = sub(t, "  - STEPS=11768\n", "  - STEPS=954\n")
                old_load = [l for l in t.splitlines() if l.startswith("  - LOAD_PARAMETERS_PATH=")]
                assert len(old_load) == 1, old_load
                t = t.replace(old_load[0] + "\n", f"  - LOAD_PARAMETERS_PATH=gs://ai2-olmo-dev/post-training/runs/olmo35-sft-h076-lr2x-s{s}/checkpoints/11768/model_params\n")
                t = sub(t, " ragged_buffer_factor=2.5 log_required_ragged_buffer_factor=true ",
                        " ragged_buffer_factor=2.5 log_required_ragged_buffer_factor=true log_moe_dropped_tokens=true ")
                t = sub(t, "  newTag: 51f8cc4b-p907d86f3-rfb46e215e\n", f"  newTag: {TAG}\n")
                for keep in (" ici_fsdp_parallelism=16 ici_expert_parallelism=8 ", " context_parallel_load_balance=false ",
                             "  - NUM_EPOCH=2\n", "  - DATA_SEED=0\n", " custom_mesh_and_rule=olmo35-ep-as-cp "):
                    assert keep in t, keep
                assert "MAXTEXT_R2_DATA_EMULATION" not in t
                if arm == "b":
                    line = [l for l in t.splitlines() if l.startswith("  - train_sft.sh")]
                    assert len(line) == 1, line
                    t = t.replace(line[0] + "\n", line[0] + "\n  - input_pipeline_utils.py # P1's lmrow.patch applied to the image's file (installed by train_sft.sh)\n")
                t = (f"# H115 arm ({arm}) {name}, seed {s}: lr2x s{s}'s overlay with data, RUN_NAME, LR, warmup, STEPS, restore\n"
                     f"# path (params only) and the ep8 drop-count placement changed (generated by mk_h115.py).\n" + t)
            if f == "train_sft.sh" and arm == "b":
                t = sub(t, "export PYTHONUNBUFFERED=1\n", LMROW.lstrip("\n") + "\nexport PYTHONUNBUFFERED=1\n")
            (dst / f).write_text(t)
        if arm == "b":
            (dst / "input_pipeline_utils.py").write_text((K / "h115" / "lmrow" / "input_pipeline_utils.py").read_text())
        print(dst)
