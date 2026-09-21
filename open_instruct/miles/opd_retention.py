"""Keep only the newest Megatron checkpoints of a Miles OPD run (``training.keep_checkpoints``).

Miles saves ``checkpoints/iter_<rollout>`` (weights plus optimizer state, ~50 GB for a 4B
learner) every ``save_interval`` rollouts and writes ``latest_checkpointed_iteration.txt`` once
the save is complete; a resume only needs the newest one, while the HF exports (``hf-<rollout>``)
and the data cursors (``checkpoints/rollout``) are what evaluation, the audit and the resume
read, so those stay. ``post_save`` is Miles's ``--custom-megatron-post-save-hook-path``: rank 0
calls it after each checkpoint and its HF export have been written.
"""

import os
import shutil
from pathlib import Path

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

KEEP_ENV = "OI_OPD_KEEP_CHECKPOINTS"
POST_SAVE_HOOK = "open_instruct.miles.opd_retention.post_save"


def completed_iterations(save_root):
    """``[(iteration, path)]`` of the ``iter_*`` directories the marker says are complete, oldest first."""
    save_root = Path(save_root)
    marker = save_root / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        return []
    latest = int(marker.read_text().strip())
    found = []
    for path in save_root.glob("iter_*"):
        suffix = path.name.removeprefix("iter_")
        if path.is_dir() and suffix.isdigit() and int(suffix) <= latest:
            found.append((int(suffix), path))
    return sorted(found)


def prune_checkpoints(save_root, keep):
    """Remove the oldest completed checkpoints under ``save_root`` so at most ``keep`` remain.

    Only iterations at or below the marker are candidates: a save in progress (a newer
    ``iter_*`` the marker does not name yet) and the marker's own iteration are never touched.
    ``keep <= 0`` disables pruning. Returns the removed paths.
    """
    if keep <= 0:
        return []
    removed = []
    for _, path in completed_iterations(save_root)[:-keep]:
        shutil.rmtree(path)
        removed.append(path)
    return removed


def post_save(args, rollout_id, checkpoint_dir, hf_checkpoint_dir):
    """Miles post-save hook: ``checkpoint_dir`` is ``<save>/iter_<rollout_id>``."""
    keep = int(os.environ.get(KEEP_ENV, "0"))
    for path in prune_checkpoints(Path(checkpoint_dir).parent, keep):
        logger.info("Removed checkpoint %s after saving rollout %s (keeping the newest %d)", path, rollout_id, keep)
