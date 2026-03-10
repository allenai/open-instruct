"""Patch vLLM's Qwen3.5 model to skip vision encoder when language_model_only=True.

Usage: python patch_qwen35_vision_skip.py
Modifies the installed vLLM package in-place.
"""

import site
import sys
from pathlib import Path


def find_qwen35_file() -> Path:
    """Find the qwen3_5.py file in the installed vLLM package."""
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        p = Path(sp) / "vllm" / "model_executor" / "models" / "qwen3_5.py"
        if p.exists():
            return p
    # Try venv
    venv = Path(sys.prefix) / "lib"
    for p in venv.rglob("vllm/model_executor/models/qwen3_5.py"):
        return p
    raise FileNotFoundError("Could not find vllm/model_executor/models/qwen3_5.py")


def patch(path: Path) -> None:
    content = path.read_text()

    old = """\
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )"""

    new = """\
        if not vllm_config.model_config.language_model_only:
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )"""

    count = content.count(old)
    if count == 0:
        # Check if already patched
        if "if not vllm_config.model_config.language_model_only:" in content:
            print("Already patched.")
            return
        raise ValueError("Could not find the vision encoder init block to patch")

    content = content.replace(old, new)
    path.write_text(content)
    print(f"Patched {count} occurrence(s) in {path}")


if __name__ == "__main__":
    path = find_qwen35_file()
    patch(path)
