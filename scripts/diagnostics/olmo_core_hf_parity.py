"""Logprob parity probe + per-layer bisection between OLMo-core's Transformer and HF.

Hypothesis (see match-grpo.md, "Bug 5"): OLMo-core's Transformer.forward silently
ignores the `attention_mask` and `position_ids` kwargs that grpo.py's logprob recompute
passes. The first run of this probe showed the silent-kwarg-drop bug accounts for ~1.66x
of the divergence but a much larger residual (~1.13 nats mean per token) remained even
after passing doc_lens — implying a deeper arch / weight-conversion mismatch.

This version:
  1. Runs both single-doc (rules out doc-handling) and packed 2-doc tests.
  2. Captures hidden states after embeddings, after each transformer block, and after
     the final RMSNorm. Reports per-layer |olmo - hf|.max() so divergence can be
     localized to a specific layer.

Run on a GPU; needs ~16GB VRAM for Qwen3-4B-Base in bf16.
"""

import argparse

import torch
import transformers
from olmo_core.nn.hf.checkpoint import load_hf_model

from open_instruct import logger_utils, model_utils, olmo_core_utils

logger = logger_utils.setup_logger(__name__)


def make_inputs(tokenizer, device, packed: bool):
    """Single-doc or packed 2-doc inputs."""
    doc_a = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids[0]
    if not packed:
        input_ids = doc_a.unsqueeze(0).to(device)
        position_ids = torch.arange(len(doc_a)).unsqueeze(0).to(device)
        return input_ids, position_ids, None, None

    doc_b = tokenizer("In a hole in the ground there lived a hobbit.", return_tensors="pt").input_ids[0]
    input_ids = torch.cat([doc_a, doc_b], dim=0).unsqueeze(0).to(device)
    position_ids = torch.cat([torch.arange(len(doc_a)), torch.arange(len(doc_b))]).unsqueeze(0).to(device)
    doc_lens = torch.tensor([[len(doc_a), len(doc_b)]], dtype=torch.int32, device=device)
    max_doc_lens = [int(max(len(doc_a), len(doc_b)))]
    return input_ids, position_ids, doc_lens, max_doc_lens


class HFCapture:
    """Forward hooks to capture HF Qwen3 hidden states at known layers."""

    def __init__(self, hf_model):
        self.hf_model = hf_model
        self.acts: dict[str, torch.Tensor] = {}
        self.handles: list = []

    def __enter__(self):
        m = self.hf_model.model
        self.handles.append(m.embed_tokens.register_forward_hook(self._save("embed")))
        for i, layer in enumerate(m.layers):
            self.handles.append(layer.register_forward_hook(self._save(f"block_{i}")))
        self.handles.append(m.norm.register_forward_hook(self._save("final_norm")))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()

    def _save(self, name):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self.acts[name] = t.detach().float().cpu()
        return hook


class OLMoCapture:
    """Forward hooks for OLMo-core Transformer; layer names matched to HFCapture."""

    def __init__(self, olmo_model):
        self.olmo_model = olmo_model
        self.acts: dict[str, torch.Tensor] = {}
        self.handles: list = []

    def __enter__(self):
        self.handles.append(self.olmo_model.embeddings.register_forward_hook(self._save("embed")))
        for key, block in self.olmo_model.blocks.items():
            self.handles.append(block.register_forward_hook(self._save(f"block_{int(key)}")))
        if self.olmo_model.lm_head is not None and getattr(self.olmo_model.lm_head, "norm", None) is not None:
            self.handles.append(self.olmo_model.lm_head.norm.register_forward_hook(self._save("final_norm")))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()

    def _save(self, name):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self.acts[name] = t.detach().float().cpu()
        return hook


def hf_logprobs(model, input_ids, position_ids):
    out = model(input_ids=input_ids, attention_mask=None, position_ids=position_ids)
    logits = out.logits if hasattr(out, "logits") else out
    return logits, model_utils.log_softmax_and_gather(logits[:, :-1], input_ids[:, 1:])


def olmo_logprobs(model, input_ids, **kwargs):
    out = model(input_ids=input_ids, **kwargs)
    logits = out if isinstance(out, torch.Tensor) else out.logits
    return logits, model_utils.log_softmax_and_gather(logits[:, :-1], input_ids[:, 1:])


def report_layers(hf_acts, olmo_acts, label):
    logger.info(f"--- per-layer |olmo - hf|.max() ({label}) ---")
    keys = sorted(set(hf_acts) & set(olmo_acts), key=lambda k: (k != "embed", k != "final_norm", k))
    # nicer ordering: embed, block_0..N, final_norm
    keys = (
        ["embed"] + sorted([k for k in hf_acts if k.startswith("block_")], key=lambda k: int(k.split("_")[1]))
        + ["final_norm"]
    )
    keys = [k for k in keys if k in hf_acts and k in olmo_acts]
    for k in keys:
        a, b = hf_acts[k], olmo_acts[k]
        if a.shape != b.shape:
            logger.warning(f"  {k}: shape mismatch hf={tuple(a.shape)} olmo={tuple(b.shape)} — skipping")
            continue
        d = (a - b).abs()
        logger.info(f"  {k:>14}  shape={tuple(a.shape)}  max={d.max():.3e}  mean={d.mean():.3e}")


def report_logprob_summary(label, olmo_lp, hf_lp, olmo_logits, hf_logits):
    d = (olmo_lp - hf_lp).abs()
    dlog = (olmo_logits - hf_logits).abs()
    olmo_top = olmo_logits.argmax(-1)
    hf_top = hf_logits.argmax(-1)
    top_agree = (olmo_top == hf_top).float().mean().item()
    logger.info(
        f"[{label}] logprob |Δ| max={d.max():.3e} mean={d.mean():.3e}  "
        f"logits |Δ| max={dlog.max():.3e} mean={dlog.mean():.3e}  "
        f"argmax-agree={top_agree*100:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--attn", default="flash_2", help="OLMo-core attention backend")
    parser.add_argument("--work_dir", default="/tmp/olmo_core_parity")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    logger.info(f"Loading HF + OLMo-core ({args.model})")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    hf_config = transformers.AutoConfig.from_pretrained(args.model)
    olmo_config = olmo_core_utils.get_transformer_config(args.model, hf_config.vocab_size, attn_backend=args.attn)
    olmo_model = olmo_config.build(init_device="cpu")
    sd = olmo_model.state_dict()
    load_hf_model(args.model, sd, work_dir=args.work_dir)
    olmo_model.load_state_dict(sd)
    olmo_model = olmo_model.to(device=device, dtype=dtype).eval()

    # Cross-check tied lm_head weight equality (Qwen3-4B-Base has tie_word_embeddings=True).
    with torch.no_grad():
        emb = olmo_model.embeddings.weight
        head = olmo_model.lm_head.w_out.weight
        same_storage = emb.data_ptr() == head.data_ptr()
        max_diff = (emb.float() - head.float()).abs().max().item()
        logger.info(f"OLMo-core embed vs lm_head: same_storage={same_storage}  max|Δ|={max_diff:.3e}")

    for packed in (False, True):
        label = "packed-2-doc" if packed else "single-doc"
        logger.info(f"\n========== {label} ==========")
        input_ids, position_ids, doc_lens, max_doc_lens = make_inputs(tokenizer, device, packed)
        logger.info(f"input_ids shape={tuple(input_ids.shape)}")

        with torch.no_grad():
            with HFCapture(hf_model) as hfc:
                hf_logits, hf_lp = hf_logprobs(hf_model, input_ids, position_ids)

            kwargs = {"attention_mask": None, "position_ids": position_ids}
            with OLMoCapture(olmo_model) as oc:
                olmo_logits_b, olmo_lp_b = olmo_logprobs(olmo_model, input_ids, **kwargs)
            report_logprob_summary(f"{label} olmo BROKEN ", olmo_lp_b, hf_lp, olmo_logits_b, hf_logits)
            report_layers(hfc.acts, oc.acts, f"{label} BROKEN")

            if packed:
                with OLMoCapture(olmo_model) as oc2:
                    olmo_logits_f, olmo_lp_f = olmo_logprobs(
                        olmo_model, input_ids, doc_lens=doc_lens, max_doc_lens=max_doc_lens
                    )
                report_logprob_summary(f"{label} olmo FIXED  ", olmo_lp_f, hf_lp, olmo_logits_f, hf_logits)
                report_layers(hfc.acts, oc2.acts, f"{label} FIXED")


if __name__ == "__main__":
    main()
