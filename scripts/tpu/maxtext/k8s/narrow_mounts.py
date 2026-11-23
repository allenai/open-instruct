"""Narrow an overlay's gcsfuse mount from the whole post-training folder to its data/ and tokenizers/ subfolders.

gcsfuseMetadataPrefetchOnMount lists everything under only-dir on every pod start; post-training/ also holds runs/,
checkpoints/ and scratch/, so every host of every start listed every checkpoint (ai2-olmo-dev ListObjects reached
~690M/day on 10-23 and 10-27; allenai/olmo-gcp-infra#52 narrows the example jobs the same way). Each subfolder gets
its own read-only mount at its old path, so TRAIN_FILES and TOKENIZER_PATH are unchanged. Checkpoints are read and
written over gs:// (OUTPUT_DIR, LOAD_PARAMETERS_PATH), never through the mount.
  python narrow_mounts.py <jobset.yaml>...
"""

import pathlib
import sys

OPTS = (
    "implicit-dirs,metadata-cache:ttl-secs:-1,metadata-cache:stat-cache-max-size-mb:-1,"
    "metadata-cache:type-cache-max-size-mb:-1,file-system:kernel-list-cache-ttl-secs:-1"
)
SUBDIRS = ("data", "tokenizers")
OLD_MOUNT = "              - name: post-training\n                mountPath: /mnt/post-training\n                readOnly: true\n"
OLD_VOLUME = (
    "            # The post-training folder of the project bucket, read-only: SFT parquet files and tokenizers.\n"
    "            - name: post-training\n"
    "              csi:\n"
    "                driver: gcsfuse.csi.storage.gke.io\n"
    "                readOnly: true\n"
    "                volumeAttributes:\n"
    "                  bucketName: ai2-olmo-dev\n"
    f"                  mountOptions: only-dir=post-training,{OPTS}\n"
    '                  gcsfuseMetadataPrefetchOnMount: "true"\n'
    '                  hostNetworkPodKSA: "true" # Authenticate the mount as serviceAccountName despite hostNetwork.\n'
)


def narrow(t: str) -> str:
    assert t.count(OLD_MOUNT) == 1 and t.count(OLD_VOLUME) == 1, "post-training mount not found exactly once"
    mounts = "".join(
        f"              - name: post-training-{d}\n                mountPath: /mnt/post-training/{d}\n"
        "                readOnly: true\n"
        for d in SUBDIRS
    )
    volumes = (
        "            # Only the data/ and tokenizers/ subfolders of the project bucket's post-training folder, read-only, each at\n"
        "            # its old path: the metadata prefetch lists everything under only-dir on every pod start, and the\n"
        "            # whole folder also holds every run's checkpoints (narrow_mounts.py).\n"
    )
    volumes += "".join(
        f"            - name: post-training-{d}\n"
        "              csi:\n"
        "                driver: gcsfuse.csi.storage.gke.io\n"
        "                readOnly: true\n"
        "                volumeAttributes:\n"
        "                  bucketName: ai2-olmo-dev\n"
        f"                  mountOptions: only-dir=post-training/{d},{OPTS}\n"
        '                  gcsfuseMetadataPrefetchOnMount: "true"\n'
        '                  hostNetworkPodKSA: "true" # Authenticate the mount as serviceAccountName despite hostNetwork.\n'
        for d in SUBDIRS
    )
    return t.replace(OLD_MOUNT, mounts).replace(OLD_VOLUME, volumes)


if __name__ == "__main__":
    for p in map(pathlib.Path, sys.argv[1:]):
        p.write_text(narrow(p.read_text()))
        print(p)
