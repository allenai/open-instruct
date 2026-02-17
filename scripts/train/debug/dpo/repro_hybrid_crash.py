"""Repro: olmo3_2_hybrid crashes during model loading.

Single GPU (may not reproduce - the bug may require ZeRO-3 multi-GPU):
    python scripts/train/debug/dpo/repro_hybrid_crash.py

Multi-GPU with ZeRO-3 (should reproduce):
    accelerate launch --num_processes 2 --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
        scripts/train/debug/dpo/repro_hybrid_crash.py
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = "/weka/oe-adapt-default/nathanl/checkpoints/HYBRID_INSTRUCT_SFT_8e-5/step3256-hf"

print("Loading config...")
config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
print(f"model_type = {config.model_type}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    config=config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    dtype=torch.bfloat16,
    attn_implementation="eager",
)
print("Model loaded successfully!")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
