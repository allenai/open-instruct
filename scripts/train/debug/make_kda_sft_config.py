"""Derive an SFT config from Jacob's OLMoE3 KDA checkpoint config.

The checkpoint config loads unmodified on the current branch -- no migration is
needed, unlike s004. The only change is memory: midtraining ran with expert
parallelism degree 8, so each rank held 1/8th of the 512 routed experts.
open-instruct builds no expert-parallel meshes, so every rank holds all of them,
and at top-16 routing the MoE activations overflow a 268 GiB B300
(01M0C12JF66MK017KA61VE67YT: 129.4 GiB static, then CUBLAS_STATUS_NOT_INITIALIZED
in a Linear forward -- OOM in disguise).

olmo-core's block has three checkpointing flags for precisely this; enabling them
trades compute for activation memory without touching the model's math.

    uv run python scripts/train/debug/make_kda_sft_config.py \
        <checkpoint>/config.json scripts/train/debug/kda_mt_sft.json
"""

import argparse
import json
import pathlib
import sys

CHECKPOINT_FLAGS = (
    "checkpoint_attn",
    "checkpoint_permute_moe_unpermute",
    "checkpoint_second_unpermute",
)

# The full-attention layers pretrain with the TransformerEngine backend, which
# raises "doesn't currently support intra-document masking" (01M0GJD4P8743W4T9NSS9C7B7X).
# SFT packs several documents per sequence and masks across their boundaries, so
# the backend has to be one that implements it.
TE_ATTENTION_BACKEND = "te"
SFT_ATTENTION_BACKEND = "flash_2"


def retarget_attention_backend(section: dict, label: str) -> list[str]:
    """Swap TE attention for a backend that supports intra-document masking."""
    mixer = section.get("sequence_mixer") or {}
    if mixer.get("backend") == TE_ATTENTION_BACKEND:
        mixer["backend"] = SFT_ATTENTION_BACKEND
        return [f"{label}: attention backend te -> {SFT_ATTENTION_BACKEND}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("dest", type=pathlib.Path)
    parser.add_argument(
        "--keep-ep",
        action="store_true",
        help="Leave the block's expert-parallel config in place (default: drop it, since open-instruct builds no EP meshes).",
    )
    args = parser.parse_args()

    payload = json.loads(args.source.read_text())
    block = payload["model"]["block"]

    for flag in CHECKPOINT_FLAGS:
        if flag not in block:
            raise SystemExit(f"block has no {flag!r}; checkpoint layout changed, re-check before training")
        block[flag] = True
    print(f"enabled: {', '.join(CHECKPOINT_FLAGS)}")

    if not args.keep_ep and block.pop("ep", None) is not None:
        print("dropped block.ep (no expert-parallel meshes in open-instruct)")

    changes = retarget_attention_backend(block, "block")
    for name, override in (payload["model"].get("block_overrides") or {}).items():
        changes += retarget_attention_backend(override, f"block_overrides.{name}")
        if not args.keep_ep:
            override.pop("ep", None)
    for change in changes:
        print(change)

    args.dest.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
