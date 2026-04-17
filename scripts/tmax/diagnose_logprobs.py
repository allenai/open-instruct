"""
Diagnose vllm vs HF logprob divergence for Qwen3.5 hybrid models.

Runs a single prompt through vLLM, captures per-token sampled logprobs, then scores
the same (prompt + completion) with HF and compares logprobs token-by-token.

Matches the training code path:
- bfloat16, enforce_eager, prefix caching controllable
- Applies the Qwen3.5 packing patch (open_instruct.qwen3_5_packing_patch) on HF
- Uses flash-linear-attention + causal-conv1d (imported via transformers fast path)
- Divides logits by temperature before log_softmax (matches forward_for_logprobs)
- Uses vllm's logprobs_mode='raw_logprobs' (no temperature applied)

This is a TWO-PHASE script:
  phase=vllm -> runs vllm, saves completion and logprobs to an .npz
  phase=hf   -> loads the npz, runs HF forward, compares and prints stats

Run them as separate processes so each has full GPU memory.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_DENSE = "Qwen/Qwen3-4B"  # for a dense-model sanity comparison


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--phase", choices=["vllm", "hf", "both"], default="both")
    p.add_argument("--prompt", default="Explain in detail how a transformer model is trained.")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_model_len", type=int, default=4096, help="vLLM max_model_len")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix_caching", type=int, default=0, help="0 or 1")
    p.add_argument("--apply_patch", type=int, default=1, help="0 or 1; HF Qwen3.5 packing patch")
    p.add_argument("--fp32_gating", type=int, default=0, help="extra: cast gate/beta to fp32 in HF (diagnostic only)")
    p.add_argument("--attn_impl", default="flash_attention_2")
    p.add_argument("--hf_chunk_size", type=int, default=0, help="override FLA chunk_size (0=default)")
    p.add_argument("--npz", default="/tmp/logprob_diag.npz")
    p.add_argument("--label", default="run")
    p.add_argument(
        "--pack_pad_left",
        type=int,
        default=0,
        help="pack N dummy tokens to the left of the target sequence (simulates packing)",
    )
    p.add_argument(
        "--pack_pad_right",
        type=int,
        default=0,
        help="pack N dummy tokens to the right of the target sequence",
    )
    p.add_argument(
        "--dummy_prompt",
        default="This is a different conversation. ",
        help="prompt for the dummy-packed neighbor sequence",
    )
    return p.parse_args()


def run_vllm(args):
    import vllm

    llm = vllm.LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=bool(args.prefix_caching),
        max_model_len=getattr(args, "max_model_len", 4096),
        gpu_memory_utilization=0.85,
        logprobs_mode="raw_logprobs",
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    chat = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    sp = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        seed=args.seed,
        max_tokens=args.max_new_tokens,
        min_tokens=args.max_new_tokens,
        logprobs=1,
    )
    out = llm.generate(prompts=[prompt_text], sampling_params=sp)[0]
    comp = out.outputs[0]

    # Get the actual tokenized prompt used (vllm may prepend BOS etc.)
    actual_prompt_ids = list(out.prompt_token_ids)
    prompt_ids = actual_prompt_ids

    token_ids = list(comp.token_ids)
    token_logprobs = []
    for i, tok_id in enumerate(token_ids):
        lp_dict = comp.logprobs[i]
        token_logprobs.append(float(lp_dict[tok_id].logprob))

    print(f"[vllm] Generated {len(token_ids)} tokens", flush=True)
    np.savez(
        args.npz,
        prompt_ids=np.array(prompt_ids, dtype=np.int64),
        comp_ids=np.array(token_ids, dtype=np.int64),
        vllm_lp=np.array(token_logprobs, dtype=np.float64),
        meta=json.dumps(
            {
                "model": args.model,
                "temperature": args.temperature,
                "prefix_caching": args.prefix_caching,
                "seed": args.seed,
            }
        ),
    )
    print(f"[vllm] Saved to {args.npz}", flush=True)


def run_hf(args):
    # Import open_instruct qwen3_5 packing patch like training does.
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo)

    if args.apply_patch:
        from open_instruct.qwen3_5_packing_patch import patch_qwen3_5_packing
        patch_qwen3_5_packing()

    from transformers import AutoModelForCausalLM

    data = np.load(args.npz, allow_pickle=True)
    prompt_ids = data["prompt_ids"].tolist()
    comp_ids = data["comp_ids"].tolist()
    vllm_lp = data["vllm_lp"]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    ).cuda().eval()

    target_ids = prompt_ids + comp_ids

    # Optionally pack dummy neighbors to simulate training-time packing.
    # Position ids reset at each sequence boundary, which triggers the
    # packing-aware path in forward_for_logprobs.
    pack_pieces: list[list[int]] = []
    if args.pack_pad_left > 0:
        # tokenize the dummy prompt, take first N tokens
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        dummy = tok.encode(args.dummy_prompt * 20, add_special_tokens=False)[: args.pack_pad_left]
        pack_pieces.append(dummy)
    pack_pieces.append(target_ids)
    if args.pack_pad_right > 0:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        dummy = tok.encode(args.dummy_prompt * 20, add_special_tokens=False)[: args.pack_pad_right]
        pack_pieces.append(dummy)

    flat_ids = [t for piece in pack_pieces for t in piece]
    flat_pos = []
    for piece in pack_pieces:
        flat_pos.extend(range(len(piece)))

    full_ids = torch.tensor([flat_ids], dtype=torch.long, device="cuda")
    position_ids = torch.tensor([flat_pos], dtype=torch.long, device="cuda")

    # Replicate forward_for_logprobs: if position_ids has a reset, pass
    # attention_mask=None and compute packing kwargs (seq_idx, cu_seqlens).
    extra_kwargs = {}
    attention_mask = None
    if (position_ids.diff(dim=-1) < 0).any():
        # use the exact helper from training for the same cu_seqlens/seq_idx shape
        from open_instruct.grpo_utils import _compute_packing_kwargs
        extra_kwargs = _compute_packing_kwargs(position_ids)
    else:
        attention_mask = torch.ones_like(position_ids)

    with torch.no_grad():
        out = model(
            input_ids=full_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **extra_kwargs,
        )

    # Locate the target sequence inside the pack
    target_start = sum(len(p) for p in pack_pieces[: 1 if args.pack_pad_left > 0 else 0])
    target_end = target_start + len(target_ids)

    logits = out.logits.float() / args.temperature  # match training path
    # position i predicts token i+1
    shift_logits = logits[:, :-1]
    shift_labels = full_ids[:, 1:]
    logprobs = torch.log_softmax(shift_logits, dim=-1)
    chosen = torch.gather(logprobs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)[0]

    # target tokens start at index `target_start` in full_ids;
    # their predictions live at chosen[target_start - 1 ..]
    prompt_len = len(prompt_ids)
    # comp tokens start at target_start + prompt_len; their logprobs are at
    # chosen[target_start + prompt_len - 1 .. + len(comp_ids)]
    comp_start_in_pack = target_start + prompt_len
    hf_lp = chosen[comp_start_in_pack - 1 : comp_start_in_pack - 1 + len(comp_ids)].cpu().numpy().astype(np.float64)

    n = min(len(vllm_lp), len(hf_lp))
    vllm_lp = vllm_lp[:n]
    hf_lp = hf_lp[:n]
    diff = np.abs(vllm_lp - hf_lp)
    print(
        f"[hf] Comparison for {args.label}: n={n}, mean|diff|={diff.mean():.5f}, "
        f"max|diff|={diff.max():.5f}, std|diff|={diff.std():.5f}"
    )

    # Bucketed drift — does disagreement grow with position?
    buckets = [64, 128, 256, 512, 1024, 2048]
    buckets = [b for b in buckets if b <= n]
    for b in buckets:
        window = diff[max(0, b - 64) : b]
        head = diff[:b]
        print(
            f"[hf]   positions 1..{b:>4}: mean|diff|={head.mean():.5f}  "
            f"  window[{max(0,b-64)}:{b}] mean|diff|={window.mean():.5f}"
        )
    # Save combined result
    out_npz = args.npz.replace(".npz", f"_hf_{args.label}.npz")
    np.savez(out_npz, vllm_lp=vllm_lp, hf_lp=hf_lp, diff=diff)
    print(f"[hf] Saved combined to {out_npz}", flush=True)


def main():
    args = parse_args()
    if args.phase == "vllm":
        run_vllm(args)
    elif args.phase == "hf":
        run_hf(args)
    else:
        raise SystemExit("Use --phase vllm then --phase hf in separate processes")


if __name__ == "__main__":
    main()
