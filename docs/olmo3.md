# OLMo 3

For details on reproducing OLMo 3 models, see the [OLMo 3 training scripts README](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/README.md).

## Training Pipeline

| Stage | Implementation | Notes |
|-------|---------------|-------|
| **SFT** | [OLMo-core](https://github.com/allenai/OLMo-core/tree/main/src/scripts/train/sft) | More GPU-efficient than `finetune.py` |
| **DPO** | open-instruct | Uses OLMo-core backend |
| **RL** | open-instruct | Uses OLMo-core backend |

## Supported Models

Models supported by OLMo-core (for SFT, DPO, RL) are listed in `open_instruct/olmo_core_utils.py`. Currently includes:

- OLMo-2 (1B, 7B, 13B, 32B)
- OLMo-3 (7B)
- OLMoE (1B-7B)
- Qwen3 (0.6B, 1.7B, 4B, 8B, 14B, 32B)

See [CONTRIBUTING.md](https://github.com/allenai/open-instruct/blob/main/CONTRIBUTING.md) for how to add new models.
