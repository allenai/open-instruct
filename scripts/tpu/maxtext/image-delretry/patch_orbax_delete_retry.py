"""Make Orbax's checkpoint delete retry, and survive a failed HNS folder cleanup once the step's objects are gone.

Orbax deletes old steps (max_to_keep) in the save-finalize thread; on an HNS GCS bucket that is `path.rmtree()` and then
one storage-control `delete_folder` call per empty folder, none retried. A transient GCS 500 in `delete_folder` propagated
through `wait_until_finished` at the next save and stopped the run (H157 s1, 2026-11-01). This replaces
`StandardCheckpointDeleter._rmtree`, so saves and restores are untouched. Both stages get up to ORBAX_DELETE_RETRY_TRIES
tries (default 5) with exponential backoff from ORBAX_DELETE_RETRY_BASE_SECS (default 4). After the last try:
- no objects left, only folders: log ORBAX-DELETE-LEAKED-FOLDERS and return;
- objects left: raise as Orbax does, since a half-deleted step can keep its commit marker and come back on restart.

Callers of `_rmtree`: retention (`delete`) and `CheckpointManager.delete(step)`, which MaxText does not call. The parallel
deleter (`num_threads` > 1, local paths only) and the rename/todelete branches bypass it and keep Orbax's behaviour.

    python3 patch_orbax_delete_retry.py   (in the image; refuses if the Orbax source differs from the one this targets)
"""

import pathlib
import sys

from orbax.checkpoint._src.path import deleter

OLD = '''  def _rmtree(self, path: epath.Path):
    """Recursively deletes a path.

    Args:
      path: the path to delete.
    """
    # TODO(b/493110683): Cleanup with refactoring of HNS GCS logic into
    # StorageBackend.
    if gcs_utils.is_gcs_path(path):
      gcs_utils.rmtree(path)
    else:
      path.rmtree()
'''

NEW = '''  def _rmtree(self, path: epath.Path):
    """Recursively deletes a path, retrying (patched, open-instruct image-delretry; see its README)."""
    tries = max(1, int(os.environ.get('ORBAX_DELETE_RETRY_TRIES', '5')))
    base = float(os.environ.get('ORBAX_DELETE_RETRY_BASE_SECS', '4'))
    for attempt in range(1, tries + 1):
      try:
        if not gcs_utils.is_gcs_path(path):
          path.rmtree()
          return
        # a failed try may already have removed every object; the HNS folder cleanup re-lists what is left
        if attempt == 1 or self._objects_left(path):
          path.rmtree()
        if gcs_utils.is_hierarchical_namespace_enabled(path):
          gcs_utils.cleanup_hns_folders(path)
        return
      except Exception as e:  # pylint: disable=broad-exception-caught
        if attempt < tries:
          delay = base * 2 ** (attempt - 1)
          logging.warning('ORBAX-DELETE-RETRY path=%s try=%d/%d err=%r; retrying in %.0fs', path, attempt, tries, e, delay)
          time.sleep(delay)
          continue
        # a half-deleted step may keep its commit marker, so only an object-free leftover is survivable
        if gcs_utils.is_gcs_path(path) and not self._objects_left(path):
          logging.error('ORBAX-DELETE-LEAKED-FOLDERS path=%s tries=%d err=%r; no objects left, continuing', path, tries, e)
          return
        raise

  @staticmethod
  def _objects_left(path: epath.Path) -> bool:
    # walks with iterdir: epath refuses recursive globs on GCS
    if not path.exists():
      return False
    dirs = [path]
    while dirs:
      for p in dirs.pop().iterdir():
        if not p.is_dir():
          return True
        dirs.append(p)
    return False
'''

src_path = pathlib.Path(deleter.__file__)
src = src_path.read_text()
if src.count(OLD) != 1:
    sys.exit(f"{src_path}: expected _rmtree source not found exactly once (not the target Orbax)")
for name in ("import os\n", "import time\n", "from absl import logging\n"):
    if name not in src:
        sys.exit(f"{src_path}: missing {name.strip()!r}, which the patched _rmtree needs")
src_path.write_text(src.replace(OLD, NEW))
print(f"patched {src_path}")
