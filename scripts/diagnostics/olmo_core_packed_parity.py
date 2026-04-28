"""Minimal packed intra-doc attention parity probe for OLMo-core's Transformer.

Hypothesis: when running a packed batch [doc_a, doc_b] through OLMo-core's
Transformer with `doc_lens=[|a|,|b|]` and `max_doc_lens=[max]` (the
intra-document attention path), the hidden states for the doc_a positions
should be bit-identical (modulo bf16 noise) to running doc_a alone, and
similarly for doc_b. Today they aren't — block-level divergence at the
slice boundaries in the >5000 range.

Strategy: use OLMo-core *alone* as both reference (separate-doc forward)
and test (packed forward). No HF needed; the question is purely whether
OLMo-core's intra-doc attention reproduces the per-document forward.

Setup:
  1. Tokenize two short docs.
  2. Run OLMo-core Transformer on each doc independently → solo hidden states.
  3. Run OLMo-core Transformer on `cat([doc_a, doc_b])` with `doc_lens` and
     `max_doc_lens` set so attention is restricted intra-doc.
  4. Compare packed[a_slice] vs solo_a, packed[b_slice] vs solo_b at every
     block.

Also runs:
  - same packed forward with NO doc_lens (cross-doc attention) as a sanity check.

Run on a GPU; ~16 GB VRAM for Qwen3-4B-Base in bf16. Only deps are torch,
transformers, olmo_core.
"""

import torch
import transformers
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.checkpoint import load_hf_model
from olmo_core.nn.transformer.config import TransformerConfig


MODEL = "Qwen/Qwen3-4B-Base"
WORK_DIR = "/tmp/qwen3_olmo_core_packed_parity_work"


class Capture:
    """Hook every block + final norm; also drill into block_0 attention submodules.

    For block_0, captures the in/out of every attention submodule that exists
    (w_q, w_k, w_v, q_norm, k_norm, rope) so divergence inside a single layer
    can be localized to RoPE vs q_norm vs the attention kernel itself.
    """

    DRILL_BLOCK = 0

    def __init__(self, olmo_model):
        self.olmo_model = olmo_model
        self.acts: dict[str, torch.Tensor] = {}
        self.handles: list = []

    def __enter__(self):
        self.handles.append(self.olmo_model.embeddings.register_forward_hook(self._save("embed")))
        for key, block in self.olmo_model.blocks.items():
            i = int(key)
            self.handles.append(block.register_forward_hook(self._save(f"block_{i}")))
            if i == self.DRILL_BLOCK:
                attn = getattr(block, "attention", None)
                if attn is not None:
                    self.handles.append(attn.register_forward_pre_hook(self._save_pre("attn_in")))
                    self.handles.append(attn.register_forward_hook(self._save("attn_out")))
                    for name in ("w_q", "w_k", "w_v", "q_norm", "k_norm", "rope"):
                        sub = getattr(attn, name, None)
                        if sub is not None:
                            self.handles.append(sub.register_forward_pre_hook(self._save_pre(f"attn_{name}_in")))
                            self.handles.append(sub.register_forward_hook(self._save(f"attn_{name}_out")))
        if self.olmo_model.lm_head is not None and getattr(self.olmo_model.lm_head, "norm", None) is not None:
            self.handles.append(self.olmo_model.lm_head.norm.register_forward_hook(self._save("final_norm")))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()

    @staticmethod
    def _to_cpu(t):
        if isinstance(t, torch.Tensor):
            return t.detach().float().cpu()
        return t

    def _save(self, name):
        def hook(_module, _inp, out):
            if isinstance(out, tuple):
                self.acts[name] = tuple(self._to_cpu(x) for x in out)
            else:
                self.acts[name] = self._to_cpu(out)
        return hook

    def _save_pre(self, name):
        def hook(_module, args):
            self.acts[name] = tuple(self._to_cpu(x) for x in args)
        return hook


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Loading OLMo-core ({MODEL})")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf_config = transformers.AutoConfig.from_pretrained(MODEL)
    olmo_config = TransformerConfig.qwen3_4B(
        vocab_size=hf_config.vocab_size, attn_backend=AttentionBackendName("flash_2")
    )
    olmo_model = olmo_config.build(init_device="cpu")
    sd = olmo_model.state_dict()
    load_hf_model(MODEL, sd, work_dir=WORK_DIR)
    olmo_model.load_state_dict(sd)
    olmo_model = olmo_model.to(device=device, dtype=dtype).eval()

    doc_a = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids[0]
    doc_b = tokenizer("In a hole in the ground there lived a hobbit.", return_tensors="pt").input_ids[0]
    la, lb = len(doc_a), len(doc_b)
    print(f"doc_a len={la}, doc_b len={lb}")

    a_ids = doc_a.unsqueeze(0).to(device)
    b_ids = doc_b.unsqueeze(0).to(device)
    packed_ids = torch.cat([doc_a, doc_b], dim=0).unsqueeze(0).to(device)
    doc_lens = torch.tensor([[la, lb]], dtype=torch.int32, device=device)
    max_doc_lens = [int(max(la, lb))]

    with torch.no_grad():
        with Capture(olmo_model) as solo_a:
            olmo_model(input_ids=a_ids)
        with Capture(olmo_model) as solo_b:
            olmo_model(input_ids=b_ids)
        with Capture(olmo_model) as packed_no_doclens:
            olmo_model(input_ids=packed_ids)
        with Capture(olmo_model) as packed_with_doclens:
            olmo_model(input_ids=packed_ids, doc_lens=doc_lens, max_doc_lens=max_doc_lens)

    print("\n=== packed-with-doc_lens vs solo (intra-doc attention path) ===")
    print("If correct, max|Δ| should stay near bf16 noise (~1e-2) across all blocks.")
    print(f"{'layer':>14}  {'a_max':>10}  {'a_mean':>10}  {'b_max':>10}  {'b_mean':>10}")
    keys = (
        ["embed"]
        + sorted([k for k in solo_a.acts if k.startswith("block_")], key=lambda k: int(k.split("_")[1]))
        + ["final_norm"]
    )
    for k in keys:
        if k not in solo_a.acts or k not in packed_with_doclens.acts:
            continue
        sa = solo_a.acts[k][0]
        sb = solo_b.acts[k][0]
        p = packed_with_doclens.acts[k][0]
        pa = p[:la]
        pb = p[la:la + lb]
        if sa.shape != pa.shape or sb.shape != pb.shape:
            print(f"  {k}: shape mismatch")
            continue
        da = (sa - pa).abs()
        db = (sb - pb).abs()
        print(f"  {k:>14}  {da.max():.3e}  {da.mean():.3e}  {db.max():.3e}  {db.mean():.3e}")

    print("\n=== block_0 attention submodule drill: packed-with-doc_lens vs solo_a ===")
    print("Compares packed[:la] tensors to solo_a tensors, submodule by submodule.")
    sub_keys = [
        "attn_in", "attn_w_q_out", "attn_w_k_out", "attn_w_v_out",
        "attn_q_norm_in", "attn_q_norm_out",
        "attn_k_norm_in", "attn_k_norm_out",
        "attn_rope_in", "attn_rope_out",
        "attn_out",
    ]
    for k in sub_keys:
        if k not in solo_a.acts or k not in packed_with_doclens.acts:
            continue
        s = solo_a.acts[k]
        p = packed_with_doclens.acts[k]
        # Both may be tuples (forward_pre or rope returning q,k) or single tensors.
        if isinstance(s, tuple) and isinstance(p, tuple):
            n = min(len(s), len(p))
            for i in range(n):
                si, pi = s[i], p[i]
                if not (isinstance(si, torch.Tensor) and isinstance(pi, torch.Tensor)):
                    continue
                if si.dim() < 2 or pi.dim() < 2:
                    continue
                pi_a = pi[:, :la] if pi.shape[1] >= la else pi
                if si.shape != pi_a.shape:
                    print(f"  {k}[{i}]: shape mismatch solo={tuple(si.shape)} packed[:la]={tuple(pi_a.shape)}")
                    continue
                d = (si - pi_a).abs()
                print(f"  {k}[{i}]  shape={tuple(si.shape)}  max={d.max():.3e}  mean={d.mean():.3e}")
        elif isinstance(s, torch.Tensor) and isinstance(p, torch.Tensor):
            pa = p[:, :la] if p.dim() >= 2 and p.shape[1] >= la else p
            if s.shape != pa.shape:
                print(f"  {k}: shape mismatch solo={tuple(s.shape)} packed[:la]={tuple(pa.shape)}")
                continue
            d = (s - pa).abs()
            print(f"  {k}  shape={tuple(s.shape)}  max={d.max():.3e}  mean={d.mean():.3e}")

    print("\n=== packed-NO-doc_lens vs solo (cross-doc attention; sanity check) ===")
    print("Doc_a portion should still match solo_a (it's first, no future tokens to bleed in).")
    print("Doc_b portion will diverge as it attends back into doc_a (expected).")
    print(f"{'layer':>14}  {'a_max':>10}  {'a_mean':>10}  {'b_max':>10}  {'b_mean':>10}")
    for k in keys:
        if k not in solo_a.acts or k not in packed_no_doclens.acts:
            continue
        sa = solo_a.acts[k][0]
        sb = solo_b.acts[k][0]
        p = packed_no_doclens.acts[k][0]
        pa = p[:la]
        pb = p[la:la + lb]
        da = (sa - pa).abs()
        db = (sb - pb).abs()
        print(f"  {k:>14}  {da.max():.3e}  {da.mean():.3e}  {db.max():.3e}  {db.mean():.3e}")


if __name__ == "__main__":
    main()
