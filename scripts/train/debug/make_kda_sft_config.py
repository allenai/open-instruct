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

The Olmo 3.5 hero checkpoints (production-hero-small-lc, 2026-09) add one more
change: their KDA layers were pretrained with ``use_cute_kernel: true``, and the
cute kernel has no packed/variable-length path. Jacob's own SFT on these
checkpoints (OLMo-core ``src/examples/olmo_ddp/olmoe3_hero_sft.py``) turns it
off and lets KDA fall back to the flash-linear-attention kernels, which is what
the proxy SFT always ran. Do the same here. Their full-attention layers use the
``flash_4`` backend, which supports intra-document masking (OLMo-core 3847ce127
passes the varlen boundaries by keyword), so it is left alone.

    uv run python scripts/train/debug/make_kda_sft_config.py \
        <checkpoint>/config.json scripts/train/debug/kda_mt_sft.json
"""

import argparse
import json
import pathlib
import sys

CHECKPOINT_FLAGS = ("checkpoint_attn", "checkpoint_permute_moe_unpermute", "checkpoint_second_unpermute")

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


def disable_cute_kda_kernel(section: dict, label: str) -> list[str]:
    """Turn off the cute KDA kernel: it has no packed path, so SFT uses the fla kernels."""
    mixer = section.get("sequence_mixer") or {}
    if mixer.get("use_cute_kernel"):
        mixer["use_cute_kernel"] = False
        return [f"{label}: use_cute_kernel true -> false (fla kernels for packed SFT)"]
    return []


def assert_emo_off(section: dict, label: str) -> None:
    """The SFT path has no EMO handling; refuse a router that still carries it."""
    router = section.get("routed_experts_router") or {}
    if router.get("emo") is not None:
        raise SystemExit(
            f"{label}: routed_experts_router.emo is set; this generator only handles the "
            "non-EMO lineage. Jacob's SFT nulls the field -- decide that explicitly."
        )


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

    changes = retarget_attention_backend(block, "block") + disable_cute_kda_kernel(block, "block")
    assert_emo_off(block, "block")
    for name, override in (payload["model"].get("block_overrides") or {}).items():
        label = f"block_overrides.{name}"
        changes += retarget_attention_backend(override, label) + disable_cute_kda_kernel(override, label)
        assert_emo_off(override, label)
        if not args.keep_ep:
            override.pop("ep", None)
    for change in changes:
        print(change)

    args.dest.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
