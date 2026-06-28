"""
Diagnose vLLM vs HF logprob divergence for Qwen3.5 under Ulysses SP=2.

Extends diagnose_logprobs.py by running the HF scoring phase across 2 GPUs
with Ulysses sequence parallelism + the per-batch FLA CP context fix.

Usage:
    # Step 1: generate with vLLM on a single GPU
    python scripts/tmax/diagnose_logprobs.py --phase vllm \\
        --model Qwen/Qwen3.5-4B --max_new_tokens 512 \\
        --npz /tmp/q35_sp.npz

    # Step 2: score with HF + SP=2 using the per-batch CP fix
    torchrun --nproc_per_node=2 scripts/tmax/diagnose_sp_logprobs.py \\
        --model Qwen/Qwen3.5-4B --npz /tmp/q35_sp.npz --pack_pad_left 256

    # Step 3: compare with old static context (should be significantly worse)
    torchrun --nproc_per_node=2 scripts/tmax/diagnose_sp_logprobs.py \\
        --model Qwen/Qwen3.5-4B --npz /tmp/q35_sp.npz \\
        --pack_pad_left 256 --static_context 1

Expected results with the fix:  mean|diff| ~0.01-0.02 (matches single-GPU HF)
Expected results static context: mean|diff| >> 0.02   (the bug)
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--npz", default="/tmp/logprob_diag.npz",
                   help="Path to .npz saved by diagnose_logprobs.py --phase vllm")
    p.add_argument("--label", default="sp2")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--attn_impl", default="flash_attention_2")
    p.add_argument(
        "--pack_pad_left", type=int, default=256,
        help="Pack N dummy tokens to the left of the target sequence.",
    )
    p.add_argument(
        "--pack_pad_right", type=int, default=0,
        help="Pack N dummy tokens to the right of the target sequence.",
    )
    p.add_argument(
        "--dummy_prompt", default="This is a different conversation. ",
        help="Text used to generate dummy packed tokens.",
    )
    p.add_argument(
        "--static_context", type=int, default=0,
        help="1 = use old static FLACPContext (the bug); 0 = per-batch fix.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo)

    import deepspeed
    from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF

    deepspeed.init_distributed()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    sp_world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Apply patch before loading model so the class is already patched.
    from open_instruct.qwen3_5_packing_patch import patch_qwen3_5_packing
    patch_qwen3_5_packing()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        device_map={"": device},
    ).eval()
    model.config.use_cache = False

    # Patch attention layers for all-to-all across the SP group.
    # register_with_transformers returns the parallel_state_sp module which owns
    # the SP group; use its accessors directly (groups._get_sequence_parallel_group
    # only works after a full deepspeed.initialize with mpu).
    mpu = UlyssesSPAttentionHF.register_with_transformers(
        model_name_or_path=model,
        core_attn_implementation=args.attn_impl,
        sequence_parallel_size=sp_world_size,
        micro_batch_size=1,
        seq_length_is_variable=True,
    )
    sp_rank = mpu.get_sequence_parallel_rank()
    sp_group = mpu.get_sequence_parallel_group()

    # ------------------------------------------------------------------ #
    # Build packed sequence (same logic as diagnose_logprobs.py run_hf)   #
    # ------------------------------------------------------------------ #
    data = np.load(args.npz, allow_pickle=True)
    prompt_ids = data["prompt_ids"].tolist()
    comp_ids = data["comp_ids"].tolist()
    vllm_lp = data["vllm_lp"]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    pad_token_id = tok.pad_token_id or tok.eos_token_id or 0

    target_ids = prompt_ids + comp_ids

    pack_pieces: list[list[int]] = []
    if args.pack_pad_left > 0:
        dummy = tok.encode(args.dummy_prompt * 20, add_special_tokens=False)[: args.pack_pad_left]
        pack_pieces.append(dummy)
    pack_pieces.append(target_ids)
    if args.pack_pad_right > 0:
        dummy = tok.encode(args.dummy_prompt * 20, add_special_tokens=False)[: args.pack_pad_right]
        pack_pieces.append(dummy)

    flat_ids = [t for piece in pack_pieces for t in piece]
    flat_pos: list[int] = []
    for piece in pack_pieces:
        flat_pos.extend(range(len(piece)))

    total_len = len(flat_ids)
    # Ensure total_len is divisible by sp_world_size so that UlyssesSP never
    # adds a padding token with pos_id=0, which would create a spurious
    # sub-sequence reset in causal_conv1d_fn and corrupt the SSM state.
    if total_len % sp_world_size != 0:
        trim = total_len % sp_world_size
        flat_ids = flat_ids[:-trim]
        flat_pos = flat_pos[:-trim]
        total_len -= trim

    # ------------------------------------------------------------------ #
    # Shard sequence for this SP rank (mirrors UlyssesSPSplitter)         #
    # ------------------------------------------------------------------ #
    padded_len = ((total_len + sp_world_size - 1) // sp_world_size) * sp_world_size
    chunk_len = padded_len // sp_world_size

    global_position_ids = torch.tensor([flat_pos], dtype=torch.long, device=device)
    full_ids_t = torch.tensor([flat_ids], dtype=torch.long, device=device)
    pad_amount = padded_len - total_len
    full_ids_padded = F.pad(full_ids_t, (0, pad_amount), value=pad_token_id)
    pos_pad_val = int(global_position_ids[0, -1].item()) + 1 if global_position_ids.shape[-1] > 0 else 1
    full_pos_padded = F.pad(global_position_ids, (0, pad_amount), value=pos_pad_val)

    start_idx = chunk_len * sp_rank
    end_idx = chunk_len * (sp_rank + 1)
    local_ids = full_ids_padded[:, start_idx:end_idx]  # [1, chunk_len]
    local_pos = full_pos_padded[:, start_idx:end_idx]  # [1, chunk_len]

    if rank == 0:
        print(
            f"[rank 0] total_len={total_len} padded_len={padded_len} chunk_len={chunk_len} "
            f"prompt={len(prompt_ids)} comp={len(comp_ids)}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # Build FLA CP context                                                 #
    # ------------------------------------------------------------------ #
    conv_kernel_size = getattr(model.config, "linear_conv_kernel_dim", None)
    text_cfg = getattr(model.config, "text_config", None)
    if conv_kernel_size is None and text_cfg is not None:
        conv_kernel_size = getattr(text_cfg, "linear_conv_kernel_dim", None)

    if args.static_context:
        # Reproduce old bug: rank metadata set from rank number alone,
        # cu_seqlens from local position resets (what old code did per-batch).
        is_start_local = (local_pos == 0)[0]
        starts_local = torch.where(is_start_local)[0].to(torch.int32)
        # Guard: if no local resets, the whole chunk is one continuation → [0, chunk_len].
        # (Without this, cu_seqlens=[chunk_len] only has 1 element and FLA crashes.)
        if len(starts_local) == 0:
            local_cu = torch.tensor([0, chunk_len], dtype=torch.int32, device=device)
        else:
            local_cu = torch.cat([starts_local, torch.tensor([chunk_len], dtype=torch.int32, device=device)])
        local_cu = torch.unique_consecutive(local_cu)
        from fla.ops.cp.context import FLACPContext
        cp_ctx = FLACPContext(
            group=sp_group,
            cu_seqlens=local_cu,
            cu_seqlens_cpu=local_cu.cpu(),
            is_first_rank=(sp_rank == 0),
            is_last_rank=(sp_rank == sp_world_size - 1),
            pre_num_ranks=sp_rank,
            post_num_ranks=sp_world_size - sp_rank - 1,
            conv1d_kernel_size=conv_kernel_size,
            pre_num_conv_tokens=0,
        )
        if rank == 0:
            print("[rank 0] Using OLD static FLACPContext (bug mode)", flush=True)
    else:
        from open_instruct.grpo_utils import build_fla_cp_context_for_sample
        cp_ctx = build_fla_cp_context_for_sample(
            global_position_ids=global_position_ids,
            sp_world_size=sp_world_size,
            sp_group=sp_group,
            conv_kernel_size=conv_kernel_size,
            local_seq_len=chunk_len,
        )
        if rank == 0:
            print("[rank 0] Using per-batch FLACPContext (fix)", flush=True)

    # ------------------------------------------------------------------ #
    # Forward pass (all ranks call simultaneously for all-to-all + CP)    #
    # ------------------------------------------------------------------ #
    from open_instruct.grpo_utils import _compute_packing_kwargs

    if (local_pos.diff(dim=-1) < 0).any():
        extra_kwargs = _compute_packing_kwargs(local_pos, cp_context=cp_ctx)
    else:
        # No local sequence boundary but CP context still needed for
        # cross-rank state passing in linear-attention layers.
        extra_kwargs = {"cp_context": cp_ctx} if cp_ctx is not None else {}

    dist.barrier()
    with torch.no_grad():
        out = model(input_ids=local_ids, position_ids=local_pos, use_cache=False, **extra_kwargs)

    logits_full = out.logits.float() / args.temperature  # [1, chunk_len, vocab]
    vocab_size = logits_full.shape[-1]

    # ------------------------------------------------------------------ #
    # Assemble per-token logprobs across ranks                            #
    # ------------------------------------------------------------------ #
    # logits_full[0, l] predicts local_ids[0, l+1] (i.e. global token r*chunk_len+l+1).
    # To predict local_ids[0, 0] (global token r*chunk_len), we need the last logit
    # row from rank r-1. We all-gather these last rows and combine.

    last_logit_row = logits_full[0, -1, :]  # [vocab]
    all_last_logits = [torch.zeros(vocab_size, device=device) for _ in range(sp_world_size)]
    dist.all_gather(all_last_logits, last_logit_row, group=sp_group)

    # inner_lp[i] = log P(local_ids[0, i+1] | ...) for i = 0..chunk_len-2
    inner_lp = torch.gather(
        torch.log_softmax(logits_full[0, :-1], dim=-1),
        dim=-1,
        index=local_ids[0, 1:].unsqueeze(-1),
    ).squeeze(-1)  # [chunk_len-1]

    if sp_rank > 0:
        prev_lp_row = torch.log_softmax(all_last_logits[sp_rank - 1], dim=-1)
        first_token_lp = prev_lp_row[local_ids[0, 0]].unsqueeze(0)  # [1]
    else:
        # Rank 0 first token has no predecessor — mark NaN (never a completion token).
        first_token_lp = torch.full((1,), float("nan"), device=device)

    # local_lp_full[i] = log P(global token r*chunk_len+i | previous context)
    local_lp_full = torch.cat([first_token_lp, inner_lp], dim=0)  # [chunk_len]

    all_lp = [torch.zeros(chunk_len, device=device) for _ in range(sp_world_size)]
    dist.all_gather(all_lp, local_lp_full, group=sp_group)

    if rank != 0:
        return

    # ------------------------------------------------------------------ #
    # Rank 0 only: compare against vLLM logprobs                          #
    # ------------------------------------------------------------------ #
    # full_lp[g] = log P(global token g | global context 0..g-1)
    full_lp = torch.cat(all_lp, dim=0).cpu().numpy()  # [padded_len]

    # Completion tokens start at: (dummy left tokens) + (prompt tokens)
    target_start = len(pack_pieces[0]) if args.pack_pad_left > 0 else 0
    comp_start = target_start + len(prompt_ids)
    hf_sp_lp = full_lp[comp_start : comp_start + len(comp_ids)]

    n = min(len(vllm_lp), len(hf_sp_lp))
    vllm_lp_n = vllm_lp[:n]
    hf_sp_lp_n = hf_sp_lp[:n]

    nan_mask = np.isnan(hf_sp_lp_n)
    if nan_mask.any():
        print(
            f"[rank 0] WARNING: {nan_mask.sum()} completion token(s) landed on a rank "
            "boundary and were skipped.",
            flush=True,
        )
    valid = ~nan_mask
    diff = np.abs(vllm_lp_n[valid] - hf_sp_lp_n[valid])

    label = args.label + ("_static" if args.static_context else "_fix")
    print(
        f"[rank 0] {label}: n={valid.sum()} mean|diff|={diff.mean():.5f} "
        f"max|diff|={diff.max():.5f} std={diff.std():.5f}",
        flush=True,
    )

    # Bucketed drift by token position.
    idx_array = np.arange(n)
    buckets = [64, 128, 256, 512, 1024, 2048]
    buckets = [b for b in buckets if b <= n]
    for b in buckets:
        mask_b = valid & (idx_array < b)
        win_mask = valid & (idx_array >= max(0, b - 64)) & (idx_array < b)
        if not mask_b.any():
            continue
        head_diff = np.abs(vllm_lp_n[mask_b] - hf_sp_lp_n[mask_b])
        win_diff = np.abs(vllm_lp_n[win_mask] - hf_sp_lp_n[win_mask])
        win_mean = win_diff.mean() if len(win_diff) > 0 else float("nan")
        print(
            f"[rank 0]   positions 1..{b:>4}: mean|diff|={head_diff.mean():.5f}  "
            f"  window[{max(0,b-64)}:{b}] mean|diff|={win_mean:.5f}",
            flush=True,
        )

    out_npz = args.npz.replace(".npz", f"_hf_{label}.npz")
    np.savez(out_npz, vllm_lp=vllm_lp_n, hf_sp_lp=hf_sp_lp_n, diff=diff)
    print(f"[rank 0] Saved to {out_npz}", flush=True)


if __name__ == "__main__":
    main()
