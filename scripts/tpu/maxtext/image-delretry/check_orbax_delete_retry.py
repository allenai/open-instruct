"""Build-time check of the patched Orbax delete: retries both stages, survives only an object-free leftover."""

import os
import pathlib
import tempfile

os.environ["ORBAX_DELETE_RETRY_BASE_SECS"] = "0"

from etils import epath
from orbax.checkpoint._src.path import deleter, gcs_utils

ERR = RuntimeError("500 We encountered an internal error. Please try again.")


class FakeFile:
    def is_dir(self):
        return False


class FakeGcsPath:
    def __init__(self, fail_rmtrees):
        self.fail_rmtrees = fail_rmtrees
        self.rmtree_calls = 0
        self.objects = True

    def rmtree(self):
        self.rmtree_calls += 1
        if self.rmtree_calls <= self.fail_rmtrees:
            raise ERR
        self.objects = False

    def exists(self):  # an HNS step dir still exists as folders after its objects are gone
        return True

    def iterdir(self):
        return [FakeFile()] if self.objects else []


def run(fail_rmtrees=0, fail_cleanups=0):
    calls = {"cleanup": 0}

    def cleanup(path):
        calls["cleanup"] += 1
        if calls["cleanup"] <= fail_cleanups:
            raise ERR

    saved = (gcs_utils.is_gcs_path, gcs_utils.is_hierarchical_namespace_enabled, gcs_utils.cleanup_hns_folders)
    gcs_utils.is_gcs_path = lambda p: True
    gcs_utils.is_hierarchical_namespace_enabled = lambda p: True
    gcs_utils.cleanup_hns_folders = cleanup
    p = FakeGcsPath(fail_rmtrees)
    try:
        deleter.StandardCheckpointDeleter._rmtree(deleter.StandardCheckpointDeleter, p)
        return "ok", p.rmtree_calls, calls["cleanup"]
    except RuntimeError:
        return "raised", p.rmtree_calls, calls["cleanup"]
    finally:
        gcs_utils.is_gcs_path, gcs_utils.is_hierarchical_namespace_enabled, gcs_utils.cleanup_hns_folders = saved


cases = {
    (0, 0): ("ok", 1, 1),
    (0, 2): ("ok", 1, 3),  # objects deleted once; the folder cleanup retried until it succeeds
    (0, 99): ("ok", 1, 5),  # objects gone, folder cleanup never succeeds: logged and survived
    (2, 0): ("ok", 3, 1),  # object delete retried until it succeeds
    (99, 0): ("raised", 5, 0),  # objects still there after 5 tries: raise as Orbax does
}
for args, want in cases.items():
    got = run(*args)
    assert got == want, (args, got, want)

d = pathlib.Path(tempfile.mkdtemp()) / "100"
(d / "items").mkdir(parents=True)
(d / "items" / "x").write_text("x")
deleter.StandardCheckpointDeleter._rmtree(deleter.StandardCheckpointDeleter, epath.Path(d))
assert not d.exists()

e = pathlib.Path(tempfile.mkdtemp()) / "200"
(e / "items" / "d").mkdir(parents=True)
assert not deleter.StandardCheckpointDeleter._objects_left(epath.Path(e))  # folders only
(e / "items" / "commit_success.txt").write_text("x")
assert deleter.StandardCheckpointDeleter._objects_left(epath.Path(e))
print("orbax delete-retry checks passed")
