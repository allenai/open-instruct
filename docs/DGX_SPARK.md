# DGX Spark Support (Experimental)

Support for NVIDIA DGX Spark (GB10 Blackwell, CUDA 13, aarch64) is experimental.

## Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GB10 (Blackwell, sm_121) |
| CUDA | 13.0 (required for Blackwell) |
| Architecture | aarch64 / ARM64 |
| Memory | 128GB unified CPU/GPU |

## Known Limitations

1. **vLLM**: Uses cu130 wheel (v0.13.0) pinned to specific commit
2. **Flash Attention**: Not available - uses PyTorch SDPA instead
3. **PyTorch**: Shows sm_121 warning (safe to ignore - binary compatible with sm_120)

## Installation

```bash
cd open-instruct
uv sync  # Automatically pulls correct wheels for aarch64
```

## Running Training

### SFT (Working)
```bash
uv run python -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/finetune.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --use_flash_attn false \
    ...
```

### GRPO/RL (Single GPU)

`VLLM_ALLOW_INSECURE_SERIALIZATION=1` is needed due to a vLLM v1 msgspec serialization bug on aarch64. See [huggingface/trl#3676](https://github.com/huggingface/trl/issues/3676).

```bash
VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run python open_instruct/grpo_fast.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --attn_implementation sdpa \
    --vllm_enforce_eager \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --single_gpu_mode \
    ...
```

**Required flags for DGX Spark single-GPU:**
- `--attn_implementation sdpa` - Use PyTorch SDPA instead of flash-attn
- `--vllm_enforce_eager` - Disable CUDA graphs (more compatible)
- `--vllm_sync_backend gloo` - Use Gloo instead of NCCL (supports same-GPU ranks)
- `--vllm_gpu_memory_utilization 0.3` - Lower memory since sharing GPU
- `--single_gpu_mode` - Collocate vLLM and policy on same GPU

## Troubleshooting

### "libcudart.so.12 not found"
You have a CUDA 12 wheel installed. The DGX Spark only has CUDA 13. Check:
```bash
pip list | grep -E "vllm|flash"
```
Remove any CUDA 12 packages and reinstall with `uv sync`.

### "sm_121 not supported" warning
Safe to ignore. sm_120 and sm_121 are binary compatible.

## References

- [vLLM Issue #31128: Blackwell SM121 Support](https://github.com/vllm-project/vllm/issues/31128)
- [vLLM Issue #28669: CUDA 13 Mismatch on ARM64](https://github.com/vllm-project/vllm/issues/28669)
- [Flash-Attention Issue #1969: SM_121 Support](https://github.com/Dao-AILab/flash-attention/issues/1969)
