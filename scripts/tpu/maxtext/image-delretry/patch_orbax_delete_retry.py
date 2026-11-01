"""Make Orbax's checkpoint retention delete retry, then give up without raising.

Orbax deletes old steps (max_to_keep) in the save-finalize thread; on an HNS GCS bucket that is `path.rmtree()` and then
one storage-control `delete_folder` call per empty folder, none retried. A transient GCS 500 there propagates through
`wait_until_finished` at the next save and stops the run (H157 s1, 2026-11-01). This replaces
`StandardCheckpointDeleter._rmtree`, which only old-step deletion calls, so saves and restores are untouched: up to
ORBAX_DELETE_RETRY_TRIES tries (default 5) with exponential backoff from ORBAX_DELETE_RETRY_BASE_SECS (default 4), then
an ORBAX-DELETE-LEAKED error line and return, leaving the old step on storage.

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
    """Recursively deletes a path, retrying; logs and returns if every try fails (patched, open-instruct image-delretry)."""
    tries = max(1, int(os.environ.get('ORBAX_DELETE_RETRY_TRIES', '5')))
    base = float(os.environ.get('ORBAX_DELETE_RETRY_BASE_SECS', '4'))
    for attempt in range(1, tries + 1):
      try:
        if not gcs_utils.is_gcs_path(path):
          path.rmtree()
          return
        # a failed try may already have removed every object; the HNS folder cleanup re-lists what is left
        if attempt == 1 or path.exists():
          path.rmtree()
        if gcs_utils.is_hierarchical_namespace_enabled(path):
          gcs_utils.cleanup_hns_folders(path)
        return
      except Exception as e:  # pylint: disable=broad-exception-caught
        if attempt == tries:
          logging.error('ORBAX-DELETE-LEAKED path=%s tries=%d err=%r; continuing, the old step stays on storage',
                        path, tries, e)
          return
        delay = base * 2 ** (attempt - 1)
        logging.warning('ORBAX-DELETE-RETRY path=%s try=%d/%d err=%r; retrying in %.0fs', path, attempt, tries, e, delay)
        time.sleep(delay)
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
