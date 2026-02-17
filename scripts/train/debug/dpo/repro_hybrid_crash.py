"""Repro: olmo3_2_hybrid crashes during model loading with ZeRO-3.

The bug is in _init_weights which accesses module.weight.data[padding_idx]
on ZeRO-3 partitioned Embedding weights (which can be size 0 on non-owning
ranks).

Launch with:
    accelerate launch --mixed_precision bf16 --num_processes 2 \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
        scripts/train/debug/dpo/repro_hybrid_crash.py
"""

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = "/weka/oe-adapt-default/nathanl/checkpoints/HYBRID_INSTRUCT_SFT_8e-5/step3256-hf"

deepspeed_plugin = DeepSpeedPlugin(
    hf_ds_config={
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "reduce_scatter": True,
            "contiguous_gradients": True,
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
    }
)

accelerator = Accelerator(
    mixed_precision="bf16",
    deepspeed_plugin=deepspeed_plugin,
)

print(f"[rank {accelerator.process_index}] Loading config...")
config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
print(f"[rank {accelerator.process_index}] model_type = {config.model_type}")

print(f"[rank {accelerator.process_index}] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    config=config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    dtype=torch.bfloat16,
    attn_implementation="eager",
)
print(f"[rank {accelerator.process_index}] Model loaded successfully!")
print(f"[rank {accelerator.process_index}] Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
