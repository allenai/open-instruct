"""Logprob parity probe between OLMo-core's Transformer and HF on a packed sequence.

Hypothesis (see match-grpo.md, "Bug 5"): OLMo-core's Transformer.forward silently
ignores the `attention_mask` and `position_ids` kwargs that grpo.py's logprob recompute
passes. With packed multi-document sequences, OLMo-core attends across document
boundaries while HF (and vLLM) do not, producing systematically different logprobs.

This script constructs a packed 2-document sequence and runs three forwards on identical
weights:

  1. HF reference: `model(input_ids, attention_mask=None, position_ids=...)` —
     HF Qwen3 builds the correct intra-doc attention from position_ids.
  2. OLMo-core "broken": `model(input_ids, attention_mask=..., position_ids=...)` —
     both kwargs are silently dropped via **kwargs; attention spans the whole row.
  3. OLMo-core "fixed": `model(input_ids, doc_lens=..., max_doc_lens=...)` —
     OLMo-core derives cu_doc_lens for intra-doc attention.

Expected:
  - mode (1) vs mode (3) within bf16 noise.
  - mode (1) vs mode (2) systematically larger.

Run on a GPU (Beaker, etc.); needs ~16GB VRAM for Qwen3-4B-Base in bf16.
"""

import argparse

import torch
import transformers
from olmo_core.nn.hf.checkpoint import load_hf_model

from open_instruct import logger_utils, model_utils, olmo_core_utils

logger = logger_utils.setup_logger(__name__)


def build_packed_inputs(tokenizer: transformers.PreTrainedTokenizer, device: torch.device):
    """Build a packed-sequence test input with two short documents.

    Returns input_ids (1, T), position_ids (1, T) that reset per doc, and the
    per-doc lengths needed for OLMo-core's intra-doc attention.
    """
    doc_a = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids[0]
    doc_b = tokenizer("In a hole in the ground there lived a hobbit.", return_tensors="pt").input_ids[0]
    input_ids = torch.cat([doc_a, doc_b], dim=0).unsqueeze(0).to(device)

    position_ids = torch.cat(
        [torch.arange(len(doc_a)), torch.arange(len(doc_b))], dim=0
    ).unsqueeze(0).to(device)

    doc_lens = torch.tensor([[len(doc_a), len(doc_b)]], dtype=torch.int32, device=device)
    max_doc_lens = [int(max(len(doc_a), len(doc_b)))]

    return input_ids, position_ids, doc_lens, max_doc_lens


def hf_logprobs(model, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """Reference: HF Qwen3 with attention_mask=None lets HF build intra-doc mask from position_ids."""
    out = model(input_ids=input_ids, attention_mask=None, position_ids=position_ids)
    logits = out.logits if hasattr(out, "logits") else out
    return model_utils.log_softmax_and_gather(logits[:, :-1], input_ids[:, 1:])


def olmo_logprobs(model, input_ids: torch.Tensor, **forward_kwargs) -> torch.Tensor:
    out = model(input_ids=input_ids, **forward_kwargs)
    logits = out if isinstance(out, torch.Tensor) else out.logits
    return model_utils.log_softmax_and_gather(logits[:, :-1], input_ids[:, 1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--attn", default="flash_2", help="OLMo-core attention backend")
    parser.add_argument("--work_dir", default="/tmp/olmo_core_parity")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    logger.info(f"Loading tokenizer + HF model from {args.model}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    logger.info("Loading OLMo-core Transformer with same weights")
    hf_config = transformers.AutoConfig.from_pretrained(args.model)
    olmo_config = olmo_core_utils.get_transformer_config(args.model, hf_config.vocab_size, attn_backend=args.attn)
    olmo_model = olmo_config.build(init_device="cpu")
    state_dict = olmo_model.state_dict()
    load_hf_model(args.model, state_dict, work_dir=args.work_dir)
    olmo_model.load_state_dict(state_dict)
    olmo_model = olmo_model.to(device=device, dtype=dtype).eval()

    input_ids, position_ids, doc_lens, max_doc_lens = build_packed_inputs(tokenizer, device)
    logger.info(f"Packed input shape: {input_ids.shape}, doc_lens: {doc_lens.tolist()}")

    with torch.no_grad():
        hf_lp = hf_logprobs(hf_model, input_ids, position_ids)
        olmo_broken = olmo_logprobs(olmo_model, input_ids, attention_mask=None, position_ids=position_ids)
        olmo_fixed = olmo_logprobs(olmo_model, input_ids, doc_lens=doc_lens, max_doc_lens=max_doc_lens)

    delta_broken = (olmo_broken - hf_lp).abs()
    delta_fixed = (olmo_fixed - hf_lp).abs()

    logger.info("=== per-token |olmo - hf| logprob delta ===")
    logger.info(f"OLMo-core BROKEN (silent kwarg drop): max={delta_broken.max():.4e}  mean={delta_broken.mean():.4e}")
    logger.info(f"OLMo-core FIXED  (doc_lens passed):  max={delta_fixed.max():.4e}  mean={delta_fixed.mean():.4e}")
    logger.info(f"Ratio (broken / fixed) by mean delta: {(delta_broken.mean() / delta_fixed.mean().clamp(min=1e-12)).item():.2f}x")


if __name__ == "__main__":
    main()
