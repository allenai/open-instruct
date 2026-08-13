"""Check that an olmo-core config + HF conversion reproduce the HF model's logits.

Key-level checks (does every parameter name and shape line up?) cannot catch a
conversion that puts the right-shaped tensor behind the wrong name, and neither
can a state-dict round trip. Only running both models on the same input does.

Needs a GPU: olmo-core's transformer forward does not run on CPU.

Usage:
    python scripts/train/debug/check_olmo_core_matches_hf.py \
        --model-name allenai/Olmo-Hybrid-7B --config-name olmo3_hybrid_7B
"""

import argparse

import torch
import transformers

from open_instruct import logger_utils, olmo_core_hybrid, olmo_core_utils

logger = logger_utils.setup_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--attn-implementation", default="flash_2")
    parser.add_argument("--tolerance", type=float, default=0.02, help="Max abs diff as a fraction of the logit range.")
    args = parser.parse_args()

    device = torch.device("cuda")
    ids = torch.tensor([[100257, 3923, 374, 279, 6864, 315, 9822, 30, 578, 6864]], device=device)

    logger.info(f"Loading HF model {args.model_name}")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    hf_model.eval()
    with torch.no_grad():
        hf_out = hf_model(ids, output_hidden_states=True)
    hf_logits = hf_out.logits.float()
    # hidden_states[0] is the embedding output, so [i + 1] is the output of block i.
    hf_hidden = [h.float() for h in hf_out.hidden_states]
    hf_state = {k: v.clone() for k, v in hf_model.state_dict().items()}
    hf_config = hf_model.config
    del hf_model
    torch.cuda.empty_cache()

    logger.info(f"Building olmo-core model from --config_name {args.config_name}")
    model_config = olmo_core_utils.get_transformer_config(
        args.config_name, hf_config.vocab_size, args.attn_implementation
    )
    model = model_config.build(init_device="meta")
    layer_types = olmo_core_hybrid.layer_types_from_hf_config(hf_config)
    converted = olmo_core_hybrid.convert_hybrid_state_from_hf(hf_state, layer_types)
    model.load_state_dict(converted, assign=True)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    # Capture each block's output so a mismatch can be traced to the first layer that
    # diverges, which tells us whether the GDN or the attention block is at fault.
    olmo_hidden: dict[int, torch.Tensor] = {}

    def _record(idx: int):
        def hook(_module, _args, output):
            olmo_hidden[idx] = (output[0] if isinstance(output, (tuple, list)) else output).detach().float()

        return hook

    handles = [block.register_forward_hook(_record(int(i))) for i, block in model.blocks.items()]
    with torch.no_grad():
        out = model(ids)
    for handle in handles:
        handle.remove()
    olmo_logits = (out[0] if isinstance(out, (tuple, list)) else out).float()

    logger.info("per-layer divergence (block index, type, max abs diff vs HF):")
    first_bad = None
    for idx in sorted(olmo_hidden):
        if idx + 1 >= len(hf_hidden):
            break
        layer_diff = (hf_hidden[idx + 1] - olmo_hidden[idx]).abs().max().item()
        scale = hf_hidden[idx + 1].abs().max().item()
        relative_layer = layer_diff / max(scale, 1e-6)
        if relative_layer > 0.01 and first_bad is None:
            first_bad = idx
        if idx < 6 or relative_layer > 0.01:
            logger.info(
                f"  block {idx:2d} ({layer_types[idx]:17s}) max abs diff {layer_diff:9.4f} ({relative_layer:.2%})"
            )
    if first_bad is not None:
        logger.info(f"first diverging block: {first_bad} ({layer_types[first_bad]})")

    diff = (hf_logits - olmo_logits).abs()
    logit_range = hf_logits.abs().max().item()
    relative = diff.max().item() / max(logit_range, 1e-6)
    agreement = (hf_logits.argmax(-1) == olmo_logits.argmax(-1)).float().mean().item()
    logger.info(f"HF logits mean {hf_logits.mean().item():.4f}, olmo-core mean {olmo_logits.mean().item():.4f}")
    logger.info(f"max abs diff {diff.max().item():.4f} over a +-{logit_range:.2f} range ({relative:.4%})")
    logger.info(f"mean abs diff {diff.mean().item():.4f}, argmax agreement {agreement:.2%}")

    if relative > args.tolerance or agreement < 1.0:
        raise SystemExit(
            f"MISMATCH: relative max diff {relative:.4%} (tolerance {args.tolerance:.2%}), "
            f"argmax agreement {agreement:.2%}. The conversion does not reproduce the HF model."
        )
    logger.info("MATCH: the olmo-core model reproduces the HF model's logits.")


if __name__ == "__main__":
    main()
