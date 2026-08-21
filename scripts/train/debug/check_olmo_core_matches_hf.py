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

from open_instruct import logger_utils, olmo_core_utils

logger = logger_utils.setup_logger(__name__)

# olmo-core attention backend -> the HF implementation running the same kernel. Both
# models must run the same one: comparing flash against sdpa leaves a numerical
# difference that looks like a conversion error, and can flip a near-tied argmax.
HF_ATTN_IMPLEMENTATIONS = {"flash_2": "flash_attention_2", "flash_3": "flash_attention_3", "torch": "sdpa"}
FALLBACK_ATTN_BACKEND = "torch"


def load_hf_reference(model_name: str, hf_attn: str, device: torch.device) -> transformers.PreTrainedModel:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation=hf_attn
    ).to(device)
    return model.eval()


def block_labels(hf_config: transformers.PretrainedConfig, num_blocks: int) -> list[str]:
    """Label each block for the per-layer report, e.g. ``linear_attention`` vs ``full_attention``.

    Only mixed-architecture models carry ``layer_types``; for everything else every
    block is the same kind and the label carries no information.
    """
    layer_types = getattr(hf_config, "layer_types", None) or []
    return [layer_types[i] if i < len(layer_types) else "block" for i in range(num_blocks)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--config-name", required=True)
    # Restricted to the backends with an HF counterpart: olmo-core's flash_4 and te have
    # none, and running the two models on different kernels makes the comparison meaningless.
    parser.add_argument("--attn-implementation", default="flash_2", choices=sorted(HF_ATTN_IMPLEMENTATIONS))
    parser.add_argument("--tolerance", type=float, default=0.02, help="Max abs diff as a fraction of the logit range.")
    args = parser.parse_args()

    device = torch.device("cuda")
    # Encode with the checkpoint's own tokenizer rather than hard-coding ids, which sit
    # outside a smaller vocabulary and fail in the embedding lookup.
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    ids = tokenizer("What is the capital of France? The capital", return_tensors="pt").input_ids.to(device)

    attn_backend = args.attn_implementation
    logger.info(f"Loading HF model {args.model_name} with attn_implementation={HF_ATTN_IMPLEMENTATIONS[attn_backend]}")
    try:
        hf_model = load_hf_reference(args.model_name, HF_ATTN_IMPLEMENTATIONS[attn_backend], device)
    except (ValueError, ImportError) as exc:
        if attn_backend == FALLBACK_ATTN_BACKEND:
            raise
        # Not every architecture implements every backend. Falling back is fine only if
        # olmo-core falls back with it, which is why attn_backend is reassigned here and
        # read again when the olmo-core config is built.
        logger.warning(
            f"HF rejected {HF_ATTN_IMPLEMENTATIONS[attn_backend]!r} ({exc}); "
            f"using {FALLBACK_ATTN_BACKEND} for both models instead."
        )
        attn_backend = FALLBACK_ATTN_BACKEND
        hf_model = load_hf_reference(args.model_name, HF_ATTN_IMPLEMENTATIONS[attn_backend], device)
    with torch.no_grad():
        hf_out = hf_model(ids, output_hidden_states=True)
    hf_logits = hf_out.logits.float()
    # hidden_states[0] is the embedding output, so [i + 1] is the output of block i --
    # except for the last entry, which transformers appends *after* the final norm.
    # Comparing a raw block output against that would report a spurious mismatch.
    hf_hidden = [h.float() for h in hf_out.hidden_states[:-1]]
    hf_config = hf_model.config
    del hf_model
    torch.cuda.empty_cache()

    logger.info(f"Building olmo-core model from --config_name {args.config_name}")
    model_config = olmo_core_utils.get_transformer_config(args.config_name, hf_config.vocab_size, attn_backend)
    model = model_config.build(init_device="meta")
    labels = block_labels(hf_config, len(model.blocks))
    # Go through the same dispatch training uses, so this checks the branch that runs
    # rather than a converter called directly.
    converted = model.state_dict()
    olmo_core_utils.load_hf_weights_into_olmo_core(converted, args.model_name)
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
            logger.info(f"  block {idx:2d} ({labels[idx]:17s}) max abs diff {layer_diff:9.4f} ({relative_layer:.2%})")
    if first_bad is not None:
        logger.info(f"first diverging block: {first_bad} ({labels[first_bad]})")

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
