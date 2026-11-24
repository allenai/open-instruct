"""HF-vs-native logit check for an EMo-trained multimodal SFT export (256K EMo base sweep run).

The exports store ``emo: null``, so both the vision eval and the text HF export route with plain
top-k. That equals training-time EMo routing only when the eval pool covers all experts. This check
builds the native OLMo-core LM three ways from the same weights and compares logits with the HF
text export on the same inputs:

  native_null   the export's config as stored (emo null): what the converter validated
  native_emo512 EMo blocks copied from the training config, eval pool = all experts (512)
  native_emocfg EMo blocks copied from the training config, eval pool as trained (diagnostic)

Inputs: a real chat prompt, and random ids with EOS separators (several documents per row, so
causal document pools are exercised). PASS = native_emo512 matches native_null under the
converter's own tolerance (torch.testing.assert_close rtol=1e-4, atol=1e-4) on every input, i.e. the
routing the export drops changes nothing at the eval pool; HF-vs-native_null on these inputs closes
the chain (the converter already asserted it on its own random input). HF-vs-each and the as-trained
pool are reported as diagnostics. Prints EMO-LOGIT-CHECK PASS|FAIL.

    emo_logit_check.py EXPORT_DIR HF_TEXT_DIR TRAIN_CONFIG_JSON

A run trained without EMo has nothing to check: prints EMO-LOGIT-CHECK SKIP and exits 0.

From /root/mmsft-eval/export/emo_logit_check.py (H271; ledger copy olmo-post-training-ledger
experiments/mmsft/tpu/export/emo_logit_check.py), importing the copied transformers plugin
(scripts/export/olmo35_mm/transformers_plugin) instead of scaling-ladders'.
"""

import copy
import json
import os
import sys

import torch
import transformers
from olmo_core.nn.transformer import TransformerConfig
from olmoe3_transformers_plugin import convert_checkpoint as cc
from olmoe3_transformers_plugin.multimodal import split_multimodal_state_dict
from safetensors.torch import load_file


def routers(lm_cfg):
    block = lm_cfg.get("block") or {}
    if isinstance(block.get("routed_experts_router"), dict):
        yield "block", block["routed_experts_router"]
    for key, override in (lm_cfg.get("block_overrides") or {}).items():
        if isinstance(override, dict) and isinstance(override.get("routed_experts_router"), dict):
            yield key, override["routed_experts_router"]


def main():
    export_dir, hf_dir, train_cfg = sys.argv[1:4]
    with open(os.path.join(export_dir, "olmo_core_config.json")) as f:
        cfg = json.load(f)
    model_config = cfg["model"]
    cc._normalize_legacy_latent_moe_config(model_config)
    cc._use_fla_kda_kernels(model_config)
    cc._use_torch_attention(model_config)
    lm_null = model_config["lm"]
    with open(train_cfg) as f:
        src_lm = json.load(f)["model"]["lm"]
    src_emo = {k: r.get("emo") for k, r in routers(src_lm)}
    print("training EMo blocks:", {k: v for k, v in src_emo.items() if v is not None})
    if not any(v is not None for v in src_emo.values()):
        print("EMO-LOGIT-CHECK SKIP: the training config has no EMo router; nothing to check")
        return

    def with_emo(pool):
        lm = copy.deepcopy(lm_null)
        for k, r in routers(lm):
            if src_emo.get(k) is not None:
                r["emo"] = copy.deepcopy(src_emo[k])
                if pool is not None:
                    r["emo"]["eval_document_expert_pool"] = pool
        return lm

    state = load_file(os.path.join(export_dir, "model.safetensors"))
    lm_state, _, _ = split_multimodal_state_dict(state)
    del state
    dev = torch.device("cuda")

    def build(lm_cfg):
        m = TransformerConfig.from_dict(copy.deepcopy(lm_cfg)).build(init_device="meta")
        m.to_empty(device=dev)
        m.load_state_dict(lm_state)
        return m.to(torch.bfloat16).eval()

    tok = transformers.PreTrainedTokenizerFast.from_pretrained(hf_dir)
    eos = tok.eos_token_id
    chat = tok.apply_chat_template(
        [{"role": "user", "content": "Describe three uses of a paperclip."}],
        add_generation_prompt=True,
        tokenize=False,
    )
    g = torch.Generator().manual_seed(0)
    rnd = torch.randint(1000, 90000, (2, 120), generator=g)
    rnd[:, 40] = eos
    rnd[:, 85] = eos  # three documents per row
    inputs = {"chat": torch.tensor([tok.encode(chat, add_special_tokens=False)]), "random_3docs": rnd}

    hf = (
        transformers.AutoModelForCausalLM.from_pretrained(
            hf_dir, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
        )
        .to(dev)
        .eval()
    )
    V = hf.config.vocab_size
    with torch.no_grad():
        ref = {k: hf(input_ids=v.to(dev)).logits[..., :V].float() for k, v in inputs.items()}
    del hf

    ok = True
    outs = {}
    pool_cfg = {k: (v or {}).get("eval_document_expert_pool") for k, v in src_emo.items()}
    for label, lm_cfg in (
        ("native_null", lm_null),
        ("native_emo512", with_emo(512)),
        ("native_emocfg", with_emo(None)),
    ):
        m = build(lm_cfg)
        with torch.no_grad():
            for k, v in inputs.items():
                out = m(input_ids=v.to(dev))
                out = (out.logits if hasattr(out, "logits") else out)[..., :V].float()
                outs[(label, k)] = out.cpu()
                d = (out - ref[k]).abs()
                agree = (out.argmax(-1) == ref[k].argmax(-1)).float().mean().item()
                try:
                    torch.testing.assert_close(ref[k], out, rtol=1e-4, atol=1e-4)
                    close = True
                except AssertionError:
                    close = False
                print(
                    f"{label:14s} {k:13s} max|d|={d.max().item():.3e} mean|d|={d.mean().item():.3e} "
                    f"argmax_agree={agree:.4f} assert_close={close}"
                )
        del m
        torch.cuda.empty_cache()
    for k in inputs:
        a, b = outs[("native_null", k)], outs[("native_emo512", k)]
        try:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
            close = True
        except AssertionError:
            close = False
        print(f"native_emo512 vs native_null {k:13s} max|d|={(a - b).abs().max().item():.3e} assert_close={close}")
        ok &= close
    print("training eval pool per router config:", pool_cfg)
    print("EMO-LOGIT-CHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
