"""Repair only the routing logger's config lookup in the pinned H127 image."""

import hashlib
from pathlib import Path

SOURCE_SHA256 = "b6479f4c693998284920dc49da216c9aec21ba5fd0c5b121ca27262b6288779f"
SOURCE_PATH = Path("/deps/src/maxtext/trainers/post_train/sft/train_sft.py")
REPLACEMENTS = (
    (
        'if getattr(self.config, "log_moe_dropped_tokens", False):',
        'if getattr(self, "_h127_log_moe_dropped_tokens", False):',
    ),
    (
        "      trainer = MaxTextPeftTrainer(model, optimizer, tunix_config)\n",
        "      trainer = MaxTextPeftTrainer(model, optimizer, tunix_config)\n"
        "      trainer._h127_log_moe_dropped_tokens = bool(mt_config.log_moe_dropped_tokens)\n",
    ),
)


def patch_source(source):
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unexpected MaxText source; refusing to patch a different image")
    text = source.decode()
    for old, new in REPLACEMENTS:
        assert text.count(old) == 1, "Expected exactly one logging-fix target"
        text = text.replace(old, new)
    return text.encode()


def main():
    patched = patch_source(SOURCE_PATH.read_bytes())
    SOURCE_PATH.write_bytes(patched)
    print(
        "H127-ROUTING-LOG-FIX", "source=" + SOURCE_SHA256, "patched=" + hashlib.sha256(patched).hexdigest(), flush=True
    )


if __name__ == "__main__":
    main()
