import shutil
import subprocess
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

def has_nvcc():
    nvcc_path = shutil.which("nvcc")
    if not nvcc_path:
        return False
    try:
        subprocess.run([nvcc_path, "--version"],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)
        return True
    except Exception:
        return False

class CustomHook(BuildHookInterface):
    def update_metadata(self, metadata):
        requires = list(metadata.get("requires_dist") or [])
        if has_nvcc():
            requires.append("flashinfer-python==0.2.8")
        metadata["requires_dist"] = requires
