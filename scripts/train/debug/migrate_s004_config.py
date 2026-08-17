"""Rewrite an OLMoE3 s004 pretrain config into the shape current olmo-core accepts.

The checkpoint was written by an older `moe-v2-core`, so two things drifted:

1. `OLMoDDPTransformerBlockConfig` used to carry separate `attention_norm` and
   `feed_forward_norm`; it now carries a single `layer_norm`.
2. `AttentionConfig` no longer takes `d_attn` -- it derives it from
   `n_heads * head_dim`.

Both rewrites are lossless *for this checkpoint*, but only because of facts that
have to hold and are therefore asserted rather than assumed: the two norms are
identical in every block, and `d_attn` already equals `n_heads * head_dim`. If a
future checkpoint violates either, this aborts instead of silently changing the
model.

    uv run python scratchpad/moe/migrate_s004.py s004_config.json s004_migrated.json
"""

import argparse
import json
import pathlib
import sys


def migrate_block(name: str, block: dict) -> list[str]:
    """Rewrite one block in place. Returns the list of changes applied."""
    changes = []

    attention_norm = block.pop("attention_norm", None)
    feed_forward_norm = block.pop("feed_forward_norm", None)
    if attention_norm is not None or feed_forward_norm is not None:
        if attention_norm is not None and feed_forward_norm is not None:
            if attention_norm != feed_forward_norm:
                raise SystemExit(
                    f"block {name!r}: attention_norm and feed_forward_norm differ, so collapsing "
                    f"them into a single layer_norm would change the model.\n"
                    f"  attention_norm    = {json.dumps(attention_norm, sort_keys=True)}\n"
                    f"  feed_forward_norm = {json.dumps(feed_forward_norm, sort_keys=True)}"
                )
        block["layer_norm"] = attention_norm if attention_norm is not None else feed_forward_norm
        changes.append("attention_norm+feed_forward_norm -> layer_norm")

    mixer = block.get("sequence_mixer", {})
    if "d_attn" in mixer:
        d_attn = mixer["d_attn"]
        derived = mixer.get("n_heads", 0) * mixer.get("head_dim", 0)
        if d_attn != derived:
            raise SystemExit(
                f"block {name!r}: d_attn={d_attn} but n_heads*head_dim={derived}. "
                f"Current AttentionConfig derives d_attn, so dropping it would change the model."
            )
        mixer.pop("d_attn")
        changes.append(f"dropped d_attn={d_attn} (== n_heads*head_dim)")

    # Execution-environment rewrites: these change how the model runs, not what it
    # computes. Weights and math are identical.
    if "ep" in block:
        # Expert parallelism (rowwise_nvshmem, degree 8 in pretraining). open-instruct's
        # train module builds no EP meshes, so the config would be inert at best and a
        # crash at parallelization time at worst.
        block.pop("ep")
        changes.append("dropped ep (open-instruct trains without expert parallelism)")
    if mixer.get("backend") == "flash_4":
        # flash-attention 4 is in the pretraining stack but not the open-instruct image;
        # flash_2 is what the hybrid runs used on the same hardware.
        mixer["backend"] = "flash_2"
        changes.append("backend flash_4 -> flash_2 (availability, not correctness)")

    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=pathlib.Path, help="config.json from the checkpoint")
    parser.add_argument("dest", type=pathlib.Path, help="where to write the migrated config")
    args = parser.parse_args()

    payload = json.loads(args.source.read_text())
    model = payload.get("model", payload)

    blocks = model.get("block")
    if not isinstance(blocks, dict):
        raise SystemExit(f"expected model.block to be a dict of named blocks, got {type(blocks)}")

    total = 0
    for name, block in blocks.items():
        changes = migrate_block(name, block)
        total += len(changes)
        print(f"{name}: {', '.join(changes) if changes else 'no change needed'}")

    args.dest.write_text(json.dumps(payload, indent=2))
    print(f"\n{total} rewrite(s) applied; wrote {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
