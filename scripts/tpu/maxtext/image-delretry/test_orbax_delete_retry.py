"""Build-time check of the patched Orbax retention delete: retries, gives up without raising, deletes local dirs."""

import os
import pathlib
import tempfile

os.environ["ORBAX_DELETE_RETRY_BASE_SECS"] = "0"

from etils import epath
from orbax.checkpoint._src.path import deleter, gcs_utils


class FakeGcsPath:
    def __init__(self):
        self.rmtree_calls = 0
        self.objects = True

    def rmtree(self):
        self.rmtree_calls += 1
        self.objects = False

    def exists(self):
        return self.objects


def run(fail_cleanups):
    calls = {"cleanup": 0}

    def cleanup(path):
        calls["cleanup"] += 1
        if calls["cleanup"] <= fail_cleanups:
            raise RuntimeError("500 We encountered an internal error. Please try again.")

    saved = (gcs_utils.is_gcs_path, gcs_utils.is_hierarchical_namespace_enabled, gcs_utils.cleanup_hns_folders)
    gcs_utils.is_gcs_path = lambda p: True
    gcs_utils.is_hierarchical_namespace_enabled = lambda p: True
    gcs_utils.cleanup_hns_folders = cleanup
    try:
        p = FakeGcsPath()
        deleter.StandardCheckpointDeleter._rmtree(None, p)  # must not raise
        return p.rmtree_calls, calls["cleanup"]
    finally:
        gcs_utils.is_gcs_path, gcs_utils.is_hierarchical_namespace_enabled, gcs_utils.cleanup_hns_folders = saved


assert run(0) == (1, 1), run(0)
assert run(2) == (1, 3), run(2)  # objects deleted once; the folder cleanup retried until it succeeds
assert run(99) == (1, 5), run(99)  # gives up after 5 tries without raising

d = pathlib.Path(tempfile.mkdtemp()) / "100"
(d / "items").mkdir(parents=True)
(d / "items" / "x").write_text("x")
deleter.StandardCheckpointDeleter._rmtree(None, epath.Path(d))
assert not d.exists()
print("orbax delete-retry checks passed")
