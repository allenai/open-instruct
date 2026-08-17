"""Probe whether a Beaker job can read olmo-core checkpoints straight from GCS.

Run as a 0-GPU job with the workspace's Google credential wired in:

    uv run python mason.py --cluster ai2/jupiter --workspace ai2/open-instruct-dev \
        --priority urgent --image "$BEAKER_IMAGE" --pure_docker_mode --preemptible \
        --num_nodes 1 --gpus 0 --non_resumable --no_auto_dataset_cache \
        --secret GOOGLE_APPLICATION_CREDENTIALS=GOOGLE_APPLICATION_CREDENTIALS \
        -- uv run python scripts/train/debug/probe_gcs_access.py

Beaker injects secrets as environment variables, but `google.auth` expects
GOOGLE_APPLICATION_CREDENTIALS to name a *file*. If the secret holds the service
account key JSON itself, this rewrites it to a file first. Nothing here prints
the credential.
"""

import os

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

S004_STEP = (
    "gs://ai2-llm/checkpoints/olmo3moe/"
    "OLMoE3-dev-260614-s004_1536d2048a_30L1536M1536S_128E8K1S_gdn/step69000"
)


def _materialize_credential() -> None:
    """Point GOOGLE_APPLICATION_CREDENTIALS at a file, whatever form it arrived in."""
    value = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    logger.info("GOOGLE_APPLICATION_CREDENTIALS present=%s length=%d", bool(value), len(value))
    if not value:
        return
    if value.lstrip().startswith("{"):
        path = "/tmp/gcp_service_account.json"
        with open(path, "w") as handle:
            handle.write(value)
        os.chmod(path, 0o600)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
        logger.info("secret held inline JSON; rewrote it to %s", path)
    else:
        logger.info("secret looks like a path; exists=%s", os.path.isfile(value))


def main() -> None:
    _materialize_credential()

    import google.auth

    credentials, project = google.auth.default()
    logger.info("auth OK: project=%s credential_type=%s", project, type(credentials).__name__)

    from olmo_core import io as olmo_core_io

    entries = list(olmo_core_io.list_directory(S004_STEP))
    logger.info("list_directory(%s) -> %d entries", S004_STEP, len(entries))
    for entry in entries[:3]:
        logger.info("  %s", entry)

    metadata = f"{S004_STEP}/model_and_optim/.metadata"
    size = olmo_core_io.file_size(metadata)
    logger.info("file_size(.metadata) = %d bytes", size)

    head = olmo_core_io.get_bytes_range(metadata, 0, 16)
    logger.info("get_bytes_range read %d bytes", len(head))

    logger.info("PROBE_OK")


if __name__ == "__main__":
    main()
