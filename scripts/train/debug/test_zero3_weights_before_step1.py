"""
Debug script: are ZeRO-3 BF16 weights valid before the first optimizer step?

Run with:
    torchrun --nproc_per_node=2 scripts/train/debug/test_zero3_weights_before_step1.py

Tests:
  1. Load a small model under deepspeed.zero.Init (ZeRO-3).
  2. Call deepspeed.initialize().
  3. Before any forward/backward, use GatheredParameters to read every parameter.
  4. Compare against the same weights loaded normally (rank 0 only).
  5. Report NaN count and max abs diff.
"""

import os
import sys
import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B")
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


def make_ds_config():
    return {
        "zero_optimization": {
            "stage": 3,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "bf16": {"enabled": True},
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": 1,
        "checkpoint": {"load_universal": False},
    }


def main():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)
    rank = dist.get_rank()

    ds_config = make_ds_config()

    # ── Load reference weights on rank 0 (no ZeRO) ──────────────────────────
    if rank == 0:
        print(f"Loading reference model (no ZeRO) on rank 0 from {MODEL} ...")
        ref = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        ref_params = {n: p.data.clone() for n, p in ref.named_parameters()}
        del ref
        torch.cuda.empty_cache()
        print(f"  Reference has {len(ref_params)} parameters.")
    else:
        ref_params = None

    dist.barrier()

    # ── Load model under deepspeed.zero.Init ─────────────────────────────────
    print(f"[rank {rank}] Activating HfDeepSpeedConfig for ZeRO-3 ...")
    dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841  must stay alive

    print(f"[rank {rank}] Loading model under zero.Init from {MODEL} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16
    )

    print(f"[rank {rank}] Calling deepspeed.initialize() ...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )

    dist.barrier()
    print(f"[rank {rank}] deepspeed.initialize() done. "
          "Checking weights with GatheredParameters BEFORE any training step ...")

    nan_params = []
    wrong_params = []
    total = 0

    for name, param in engine.module.named_parameters():
        total += 1
        with deepspeed.zero.GatheredParameters([param], enabled=True):
            if rank == 0:
                data = param.data  # full gathered parameter on rank 0
                has_nan = data.isnan().any().item()
                has_inf = data.isinf().any().item()

                if has_nan or has_inf:
                    nan_params.append((name, "NaN" if has_nan else "Inf"))

                if ref_params is not None and name in ref_params:
                    ref = ref_params[name].to(data.device)
                    max_diff = (data.float() - ref.float()).abs().max().item()
                    if max_diff > 1e-3:  # BF16 has ~1e-2 precision, but same dtype so should match exactly
                        wrong_params.append((name, max_diff))

    dist.barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Checked {total} parameters via GatheredParameters before step 1.")
        print(f"{'='*60}")
        if nan_params:
            print(f"❌ {len(nan_params)} parameters contain NaN/Inf:")
            for n, kind in nan_params[:10]:
                print(f"    {kind}: {n}")
        else:
            print("✅ No NaN/Inf in any gathered parameter.")

        if wrong_params:
            print(f"❌ {len(wrong_params)} parameters differ from reference (max_diff > 1e-3):")
            for n, d in sorted(wrong_params, key=lambda x: -x[1])[:10]:
                print(f"    max_diff={d:.6f}: {n}")
        else:
            print("✅ All gathered parameters match reference weights exactly.")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
