"""Repro script for olmo3_2_hybrid model_type not recognized by new transformers fork.

The new transformers fork (olmo-3.5-hybrid-clean branch) renamed the model type
from "olmo3_2_hybrid" to "olmo_hybrid", but existing HF checkpoints still have
"model_type": "olmo3_2_hybrid" in their config.json, causing AutoConfig to fail.

Usage: HF_TOKEN=<token> uv run python scripts/repro_olmo_hybrid_model_type.py
"""

import os

from transformers import AutoConfig

MODEL_NAME = "allenai/Olmo-Hybrid-Instruct-DPO-7B"


def main():
    token = os.environ.get("HF_TOKEN")
    print(f"Attempting AutoConfig.from_pretrained('{MODEL_NAME}')...")
    config = AutoConfig.from_pretrained(MODEL_NAME, token=token)
    print(f"SUCCESS: AutoConfig loaded. model_type={config.model_type}")


if __name__ == "__main__":
    main()
