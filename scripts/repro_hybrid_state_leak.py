"""Standalone repro for OlmoHybrid recurrent state leak across packed sequences.

HuggingFace's OlmoHybridGatedDeltaNet.forward() doesn't pass cu_seqlens to
FLA recurrent kernels. When multiple sequences are packed into one forward pass,
the SSM layers leak state across sequence boundaries.

This script demonstrates the bug by comparing logprobs from a packed forward
pass against individual forward passes. The second sequence in a packed batch
should have identical logprobs whether or not the first sequence precedes it,
but due to the state leak, the logprobs differ significantly.

Dependencies: torch, transformers (no open-instruct internals).

Usage:
    HF_TOKEN=<token> uv run python scripts/repro_hybrid_state_leak.py
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "allenai/Olmo-Hybrid-Instruct-DPO-7B"

PROMPTS = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Explain photosynthesis in one sentence."}],
]

MAX_NEW_TOKENS = 32


def pack_sequences(seq1_ids, seq2_ids):
    input_ids = torch.cat([seq1_ids, seq2_ids])
    attention_mask = torch.cat([
        torch.ones(len(seq1_ids), dtype=torch.long),
        torch.full((len(seq2_ids),), 2, dtype=torch.long),
    ])
    position_ids = torch.cat([
        torch.arange(len(seq1_ids)),
        torch.arange(len(seq2_ids)),
    ])
    return input_ids, attention_mask, position_ids


def get_logprobs(model, input_ids, attention_mask, position_ids):
    with torch.no_grad():
        output = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
        )
    logits = output.logits[:, :-1]
    labels = input_ids[1:].unsqueeze(0)
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1).squeeze(0)


def main():
    token = os.environ.get("HF_TOKEN")
    device = "cuda"

    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model from {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=token,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    print("\nGenerating responses for two prompts...")
    full_sequences = []
    for prompt in PROMPTS:
        prompt_text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
        full_ids = output_ids[0]
        full_sequences.append(full_ids)
        decoded = tokenizer.decode(full_ids, skip_special_tokens=True)
        print(f"  Prompt: {prompt[0]['content']}")
        print(f"  Response: {decoded[:120]}...")

    seq1_ids = full_sequences[0]
    seq2_ids = full_sequences[1]

    print(f"\nSequence 1 length: {len(seq1_ids)}")
    print(f"Sequence 2 length: {len(seq2_ids)}")

    packed_input_ids, packed_attn_mask, packed_pos_ids = pack_sequences(
        seq1_ids, seq2_ids
    )
    packed_input_ids = packed_input_ids.to(device)
    packed_attn_mask = packed_attn_mask.to(device)
    packed_pos_ids = packed_pos_ids.to(device)

    print("\nRunning packed forward pass...")
    packed_logprobs = get_logprobs(
        model, packed_input_ids, packed_attn_mask, packed_pos_ids
    )

    seq1_len = len(seq1_ids)
    seq2_len = len(seq2_ids)
    packed_seq1_logprobs = packed_logprobs[:seq1_len - 1]
    packed_seq2_logprobs = packed_logprobs[seq1_len - 1:seq1_len + seq2_len - 2]

    print("Running individual forward pass for sequence 1...")
    indiv_attn_mask_1 = torch.ones(len(seq1_ids), dtype=torch.long, device=device)
    indiv_pos_ids_1 = torch.arange(len(seq1_ids), device=device)
    indiv_seq1_logprobs = get_logprobs(
        model, seq1_ids, indiv_attn_mask_1, indiv_pos_ids_1
    )

    print("Running individual forward pass for sequence 2...")
    indiv_attn_mask_2 = torch.ones(len(seq2_ids), dtype=torch.long, device=device)
    indiv_pos_ids_2 = torch.arange(len(seq2_ids), device=device)
    indiv_seq2_logprobs = get_logprobs(
        model, seq2_ids, indiv_attn_mask_2, indiv_pos_ids_2
    )

    seq1_diff = (packed_seq1_logprobs - indiv_seq1_logprobs).abs().mean().item()
    seq2_diff = (packed_seq2_logprobs - indiv_seq2_logprobs).abs().mean().item()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Sequence 1 (packed vs individual) mean abs diff: {seq1_diff:.6f}")
    print(f"  Expected: ~0 (first sequence is unaffected)")
    print(f"Sequence 2 (packed vs individual) mean abs diff: {seq2_diff:.6f}")
    print(f"  Expected: >> 0 if state leak exists (typically > 0.5)")
    print("=" * 60)

    if seq2_diff > 0.1:
        print("\nSTATE LEAK CONFIRMED: Second sequence logprobs differ significantly")
        print("when preceded by another sequence in packed input.")
    else:
        print("\nNo state leak detected (or bug has been fixed).")


if __name__ == "__main__":
    main()
