"""
Diagnose vLLM vs HF logprob divergence in a multi-turn / tool-use scenario.

The script replicates what grpo_fast.py / process_request() does during training:
 - For each turn: vLLM completion on full accumulated context, collect decode-time logprobs.
 - Tool output: fake tool tokens appended, logprobs set to 0.0.
 - HF: single forward pass on the full sequence to get chunk-kernel logprobs.
 - Report mean|Δlogp| per turn to see if drift grows with turn depth.

Usage (two separate processes, same GPU):

    # Step 1: generate with vLLM and save per-token logprobs
    python scripts/tmax/diagnose_multiturn_logprobs.py --phase vllm \\
        --model Qwen/Qwen3.5-9B \\
        --num_turns 10 --turn_tokens 2000 --tool_tokens 1000 \\
        --npz /tmp/mt_diag.npz --max_model_len 36000

    # Step 2: score with HF and print comparison
    python scripts/tmax/diagnose_multiturn_logprobs.py --phase hf \\
        --model Qwen/Qwen3.5-9B \\
        --npz /tmp/mt_diag.npz

Flags:
  --prefix_caching 1  to re-run with prefix caching enabled (recreates old setup)
  --apply_patch 0     to run HF without the Qwen3.5 packing patch (sanity check)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
FAKE_TOOL_TEXT = (
    "Tool output: [{'status': 'ok', 'result': 42, 'stdout': 'Hello from the sandbox!\\n'}] "
    * 200  # enough tokens for large tool_tokens values
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--phase", choices=["vllm", "hf"], required=True)
    p.add_argument("--npz", default="/tmp/mt_logprob_diag.npz")
    p.add_argument("--label", default="multiturn")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix_caching", type=int, default=0, help="1 to enable vLLM prefix caching")
    p.add_argument("--apply_patch", type=int, default=1, help="1 to apply Qwen3.5 packing patch in HF")
    p.add_argument("--attn_impl", default="flash_attention_2")
    p.add_argument("--num_turns", type=int, default=2, help="number of assistant turns")
    p.add_argument("--turn_tokens", type=int, default=256, help="# assistant tokens per turn")
    p.add_argument("--tool_tokens", type=int, default=128, help="# fake tool-output tokens after each turn except last")
    p.add_argument(
        "--user_prompt",
        default=(
            "Write a Python function that computes the nth Fibonacci number using "
            "dynamic programming. Then call it with n=10 and print the result."
        ),
    )
    p.add_argument("--max_model_len", type=int, default=4096)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase: vLLM — one completion call per turn, like process_request()
# ---------------------------------------------------------------------------

def run_vllm(args):
    import vllm

    llm = vllm.LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=bool(args.prefix_caching),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        logprobs_mode="raw_logprobs",
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()

    chat = [
        {"role": "system", "content": "You are a helpful coding assistant. Use the sandbox tool to run code."},
        {"role": "user", "content": args.user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    tool_ids_pool = tokenizer.encode(FAKE_TOOL_TEXT, add_special_tokens=False)[: args.tool_tokens]

    # Accumulated context starts from the prompt
    current_ids = list(prompt_ids)
    actual_prompt_ids = None

    all_turn_ids = []
    all_turn_lp = []
    all_tool_ids_list = []

    for t in range(args.num_turns):
        sp = vllm.SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            seed=args.seed + t,
            max_tokens=args.turn_tokens,
            min_tokens=args.turn_tokens,
            logprobs=1,
        )
        out = llm.generate([current_ids], sampling_params=sp)[0]
        if actual_prompt_ids is None:
            actual_prompt_ids = list(out.prompt_token_ids)
            current_ids = list(actual_prompt_ids)

        turn_ids = list(out.outputs[0].token_ids)
        turn_lp = [float(out.outputs[0].logprobs[i][tid].logprob) for i, tid in enumerate(turn_ids)]
        mean_lp = sum(turn_lp) / len(turn_lp)
        print(f"[vllm] Turn {t+1}: {len(turn_ids)} tokens (ctx={len(current_ids)}), "
              f"mean logprob={mean_lp:.4f}", flush=True)

        all_turn_ids.append(turn_ids)
        all_turn_lp.append(turn_lp)
        current_ids = current_ids + turn_ids

        # Inject tool output after every turn except the last
        if t < args.num_turns - 1:
            tool_ids = tool_ids_pool[: args.tool_tokens]
            all_tool_ids_list.append(tool_ids)
            current_ids = current_ids + tool_ids
            print(f"[vllm] Tool output: {len(tool_ids)} tokens injected", flush=True)
        else:
            all_tool_ids_list.append([])

    # Build response = interleaved (turn, tool, turn, tool, ..., last_turn)
    response_ids = []
    response_mask = []
    response_logprobs = []
    turn_boundaries = []  # (turn_start, turn_end, tool_start, tool_end)

    for t in range(args.num_turns):
        t_start = len(response_ids)
        response_ids.extend(all_turn_ids[t])
        response_mask.extend([1] * len(all_turn_ids[t]))
        response_logprobs.extend(all_turn_lp[t])
        t_end = len(response_ids)

        tool_start = len(response_ids)
        response_ids.extend(all_tool_ids_list[t])
        response_mask.extend([0] * len(all_tool_ids_list[t]))
        response_logprobs.extend([0.0] * len(all_tool_ids_list[t]))
        tool_end = len(response_ids)

        turn_boundaries.append((t_start, t_end, tool_start, tool_end))

    np.savez(
        args.npz,
        prompt_ids=np.array(actual_prompt_ids, dtype=np.int64),
        response_ids=np.array(response_ids, dtype=np.int64),
        response_logprobs=np.array(response_logprobs, dtype=np.float64),
        response_mask=np.array(response_mask, dtype=np.int32),
        turn_boundaries=np.array(turn_boundaries, dtype=np.int64),
        meta=json.dumps({
            "model": args.model,
            "temperature": args.temperature,
            "prefix_caching": args.prefix_caching,
            "seed": args.seed,
            "num_turns": args.num_turns,
        }),
    )
    print(f"[vllm] Saved to {args.npz}", flush=True)
    print(f"[vllm] Total: prompt={len(actual_prompt_ids)}, response={len(response_ids)} tokens", flush=True)


# ---------------------------------------------------------------------------
# Phase: HF — single forward pass on full sequence, same as training scoring
# ---------------------------------------------------------------------------

def run_hf(args):
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo)

    if args.apply_patch:
        from open_instruct.qwen3_5_packing_patch import patch_qwen3_5_packing
        patch_qwen3_5_packing()
        print("[hf] Qwen3.5 packing patch applied", flush=True)

    from transformers import AutoModelForCausalLM

    data = np.load(args.npz, allow_pickle=True)
    prompt_ids = data["prompt_ids"].tolist()
    response_ids = data["response_ids"].tolist()
    vllm_resp_lp = data["response_logprobs"]
    response_mask = data["response_mask"]
    turn_boundaries = data["turn_boundaries"]  # (num_turns, 4): t_start, t_end, tool_start, tool_end

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    ).cuda().eval()

    full_ids = prompt_ids + response_ids
    full_ids_t = torch.tensor([full_ids], dtype=torch.long, device="cuda")
    position_ids = torch.arange(len(full_ids), device="cuda").unsqueeze(0)
    attention_mask = torch.ones_like(position_ids)

    with torch.no_grad():
        out = model(
            input_ids=full_ids_t,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

    # Compute logprobs in chunks to avoid OOM on large sequences with big vocab.
    # logits are bf16; materializing [1, N, vocab] in fp32 costs ~N * vocab * 4 bytes.
    logits_bf16 = out.logits  # [1, T, V]
    del out
    shift_labels = full_ids_t[0, 1:]  # [T-1]
    chunk_size = 2048
    lp_chunks = []
    for start in range(0, logits_bf16.shape[1] - 1, chunk_size):
        end = min(start + chunk_size, logits_bf16.shape[1] - 1)
        chunk = logits_bf16[0, start:end].float() / args.temperature  # [C, V]
        chunk_lp = torch.log_softmax(chunk, dim=-1)
        gathered = torch.gather(chunk_lp, 1, shift_labels[start:end].unsqueeze(-1)).squeeze(-1)
        lp_chunks.append(gathered.cpu())
    hf_all_lp = torch.cat(lp_chunks).numpy()

    prompt_len = len(prompt_ids)
    resp_len = len(response_ids)
    hf_resp_lp = hf_all_lp[prompt_len - 1 : prompt_len - 1 + resp_len]

    print(f"\n[hf] === {args.label} ===")
    print(f"[hf] prompt={prompt_len}  total_response={resp_len}  num_turns={len(turn_boundaries)}")

    non_tool_mask = response_mask != 0
    overall_diff = np.abs(vllm_resp_lp[non_tool_mask] - hf_resp_lp[non_tool_mask])
    print(f"[hf] OVERALL assistant tokens (n={non_tool_mask.sum()}): "
          f"mean|diff|={overall_diff.mean():.5f}  max|diff|={overall_diff.max():.5f}")

    for ti, (t_start, t_end, tool_start, tool_end) in enumerate(turn_boundaries):
        vl = vllm_resp_lp[t_start:t_end]
        hl = hf_resp_lp[t_start:t_end]
        mask = response_mask[t_start:t_end] != 0
        if not mask.any():
            continue
        diff = np.abs(vl[mask] - hl[mask])
        # Context length = prompt + everything before this turn
        ctx_len = prompt_len + int(t_start)
        print(f"[hf]   Turn {ti+1} (ctx={ctx_len}, n={mask.sum()}): "
              f"mean|diff|={diff.mean():.5f}  max|diff|={diff.max():.5f}")

    out_npz = args.npz.replace(".npz", f"_hf_{args.label}.npz")
    np.savez(out_npz, vllm_resp_lp=vllm_resp_lp, hf_resp_lp=hf_resp_lp,
             response_mask=response_mask)
    print(f"\n[hf] Saved to {out_npz}", flush=True)


def main():
    args = parse_args()
    if args.phase == "vllm":
        run_vllm(args)
    elif args.phase == "hf":
        run_hf(args)


if __name__ == "__main__":
    main()
