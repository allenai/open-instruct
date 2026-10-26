"""H131 (ledger, if-dose-c11-7b) overlays: H122's olmo3-7b-c11-s0 overlay with only names, TRAIN_FILES and the run name
changed, for c11if seeds 0-3 (C11 + 4x H123's IF build) and the c11 control seeds 2-3. A round-trip assert checks that
undoing the substitutions gives H122's files back byte for byte (the P3 config-identity check).
  python mk_h131.py      (run from this directory)"""
import os

SRC = "../h122/olmo3-7b-c11-s0"
FILES = ("jobset.yaml", "kustomization.yaml", "train_sft.sh")
NOTE = ("# H131 (ledger, if-dose-c11-7b): H122's overlay (../h122/olmo3-7b-c11-s0) with only names, TRAIN_FILES and the run\n"
        "# name changed; data from the ledger's experiments/h131/p1 (c11if: C11 + 4 copies of H123's IF build; c11: C11).\n")


def subs(arm, k):
    return [("h122-olmo3-7b-c11-s0", f"h131-olmo3-7b-{arm}-s{k}"),
            ("olmo3-7b-sft-h122-c11-s0", f"olmo3-7b-sft-h131-{arm}-s{k}"),
            ("/mnt/post-training/data/h122/c11-s0/", f"/mnt/post-training/data/h131/{DATA[arm]}-s{k}/")]


# H131 Amendment 1 (2026-10-26): c11if trains on the r3 streams (C12's filtered IF build); the control is unchanged.
DATA = {"c11if": "c11if-r3", "c11": "c11"}


for arm, seeds in (("c11if", (0, 1, 2, 3)), ("c11", (2, 3))):
    for k in seeds:
        d = f"{arm}-s{k}"
        os.makedirs(d, exist_ok=True)
        for f in FILES:
            src = open(os.path.join(SRC, f)).read()
            out = src
            for a, b in subs(arm, k):
                pass
                out = out.replace(a, b)
            if f == "kustomization.yaml":
                out = NOTE + out
            back = out[len(NOTE):] if f == "kustomization.yaml" else out
            for a, b in subs(arm, k):
                back = back.replace(b, a)
            assert back == src, (d, f)
            open(os.path.join(d, f), "w").write(out)
        print("WROTE", d)
