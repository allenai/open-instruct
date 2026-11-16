"""Verify a content-addressed diagnostic source overlay before container-local installation."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import install_gradient_noise


def main():
    source = Path(os.environ["OI_H235_NOISE_SOURCE"])
    manifest = source / "MANIFEST.json"
    expected = os.environ["OI_H235_NOISE_MANIFEST_SHA256"]
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
        raise ValueError("Diagnostic manifest differs from the launch pin")
    for name, digest in json.loads(manifest.read_text())["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Diagnostic source hash mismatch: {name}")
    destination = Path("/opt/core-rl/scripts/miles/diagnostics")
    destination.mkdir(parents=True, exist_ok=True)
    for name in json.loads(manifest.read_text())["files"]:
        if name.endswith(".py"):
            shutil.copyfile(source / name, destination / name)
    actor = Path("/opt/core-rl/sources/miles/miles/backends/core_utils/actor.py")
    marker = "metrics = gradient_noise.measure(self, step_batches, objective, self._replay_context)"
    if marker not in actor.read_text():
        install_gradient_noise.install(actor)
    compile(actor.read_text(), str(actor), "exec")


if __name__ == "__main__":
    main()
