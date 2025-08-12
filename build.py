import shutil
import subprocess
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def has_nvcc():
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        return False
    else:
        return True


class CustomHook(BuildHookInterface):
    def update_metadata(self, metadata):
        requires = list(metadata.get("requires_dist") or [])
        if has_nvcc():
            requires.append("flashinfer-python==0.2.8")
        metadata["requires_dist"] = requires
