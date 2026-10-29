"""Narrow a MaxText SFT overlay's gcsfuse mounts to the folders the run reads.

Why: gcsfuseMetadataPrefetchOnMount lists everything under only-dir on every pod start, and only-dir=post-training also
holds every run's checkpoints (runs/, checkpoints/, scratch/). 32 hosts per slice x restarts x a growing checkpoint tree
made hundreds of millions of ListObjects a day on ai2-olmo-dev in October 2026 (olmo-gcp-infra#52). Checkpoints are read
and written over gs:// (OUTPUT_DIR, LOAD_PARAMETERS_PATH), never through the mount, so the mount only needs the
tokenizer folder and the run's parquet folder.

  python narrow_gcsfuse.py <overlay dir> [<overlay dir> ...]     rewrite each overlay's jobset.yaml in place
  python narrow_gcsfuse.py --check <overlay dir> ...             exit 1 if any overlay still mounts the whole folder

Reads TOKENIZER_PATH and TRAIN_FILES from the overlay's kustomization.yaml (both under /mnt/post-training/...), replaces
the single `post-training` volume and mount with one read-only mount per folder at the same paths, so the env values and
train_sft.sh are unchanged. Idempotent. Never run it on an overlay whose JobSet is running: the file would stop matching
the JobSet that ran (ledger provenance is by commit).
"""
import pathlib, re, sys

OPTS = ("implicit-dirs,metadata-cache:ttl-secs:-1,metadata-cache:stat-cache-max-size-mb:-1,"
        "metadata-cache:type-cache-max-size-mb:-1,file-system:kernel-list-cache-ttl-secs:-1")
MOUNT_RE = re.compile(r"              - name: post-training\n                mountPath: /mnt/post-training\n                readOnly: true\n")
VOL_RE = re.compile(r"            (?:#[^\n]*\n            )*- name: post-training\n              csi:\n(?:                [^\n]*\n)+")


def folders(overlay):
    k = (overlay / "kustomization.yaml").read_text()
    tok = re.search(r"^  - TOKENIZER_PATH=(/mnt/post-training/tokenizers/[^/\s]+)\s*$", k, re.M)
    data = re.search(r"^  - TRAIN_FILES=(/mnt/post-training/data/\S+?)/\*\.parquet\s*$", k, re.M)
    assert tok and data, (overlay, "TOKENIZER_PATH under /mnt/post-training/tokenizers/<name> and TRAIN_FILES as /mnt/post-training/data/<dir>/*.parquet")
    return tok.group(1), data.group(1)


def volume(name, path):
    sub = path[len("/mnt/"):]
    return (f"            - name: {name}\n              csi:\n                driver: gcsfuse.csi.storage.gke.io\n                readOnly: true\n"
            f"                volumeAttributes:\n                  bucketName: ai2-olmo-dev\n"
            f"                  mountOptions: only-dir={sub},{OPTS}\n"
            f"                  gcsfuseMetadataPrefetchOnMount: \"true\"\n"
            f"                  hostNetworkPodKSA: \"true\" # Authenticate the mount as serviceAccountName despite hostNetwork.\n")


def narrow(overlay, check=False):
    overlay = pathlib.Path(overlay); js = overlay / "jobset.yaml"; t = js.read_text()
    whole = "only-dir=post-training," in t
    if check:
        return not whole
    if not whole:
        print("already narrow", overlay); return True
    tok, data = folders(overlay)
    assert len(MOUNT_RE.findall(t)) == 1 and len(VOL_RE.findall(t)) == 1, (overlay, "expected one post-training mount and one volume")
    mounts = (f"              - name: pt-tokenizer\n                mountPath: {tok}\n                readOnly: true\n"
              f"              - name: pt-data\n                mountPath: {data}\n                readOnly: true\n")
    vols = ("            # Only the folders this run reads from the post-training folder of the project bucket, each its own\n"
            "            # read-only mount: the metadata prefetch lists everything under only-dir on every pod start, and the\n"
            "            # folder also holds every run's checkpoints (olmo-gcp-infra#52; narrow_gcsfuse.py). Checkpoints go\n"
            "            # over gs:// (OUTPUT_DIR, LOAD_PARAMETERS_PATH), not through a mount.\n"
            + volume("pt-tokenizer", tok) + volume("pt-data", data))
    t = MOUNT_RE.sub(mounts, t); t = VOL_RE.sub(vols, t)
    assert "only-dir=post-training," not in t and t.count("only-dir=") == 2
    js.write_text(t); print("narrowed", overlay, tok, data); return True


if __name__ == "__main__":
    args = sys.argv[1:]; check = "--check" in args; dirs = [a for a in args if a != "--check"]
    bad = [d for d in dirs if not narrow(d, check)]
    if check:
        for d in bad: print("WHOLE-FOLDER MOUNT", d)
        sys.exit(1 if bad else 0)
