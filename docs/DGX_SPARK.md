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
4. **GRPO/RL**: Requires 2+ GPUs - NCCL weight sync doesn't support single-GPU mode

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

### GRPO/RL (Requires 2+ GPUs)

> **Note**: GRPO requires at least 2 GPUs because it uses NCCL distributed weight sync between the policy model and vLLM inference engine. Single-GPU training causes "Duplicate GPU detected" NCCL errors.

```bash
uv run python open_instruct/grpo_fast.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --attn_implementation sdpa \
    --vllm_enforce_eager \
    --vllm_gpu_memory_utilization 0.7 \
    ...
```

**Required flags for DGX Spark:**
- `--attn_implementation sdpa` - Use PyTorch SDPA instead of flash-attn
- `--vllm_enforce_eager` - Disable CUDA graphs (more compatible)
- `--vllm_gpu_memory_utilization 0.7` - Leave headroom for unified memory

## Troubleshooting

### "libcudart.so.12 not found"
You have a CUDA 12 wheel installed. The DGX Spark only has CUDA 13. Check:
```bash
pip list | grep -E "vllm|flash"
```
Remove any CUDA 12 packages and reinstall with `uv sync`.

### "sm_121 not supported" warning
Safe to ignore. sm_120 and sm_121 are binary compatible.

### vLLM serialization errors
If you see `torch.dtype is not serializable`, try:
```bash
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
```
This is a known issue with vLLM v1 nightly builds.

## Building vLLM from Source (Fallback)

If the pinned wheel stops working, build vLLM from source:

```bash
# Install PyTorch cu130 first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Clone and build vLLM
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.13.0  # or desired version

# Set CUDA 13 environment
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="12.1a"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Build
python use_existing_torch.py
pip install -r requirements/build.txt
pip install --no-build-isolation -e .
```

Build time: ~20-30 minutes.

## References

- [vLLM Issue #31128: Blackwell SM121 Support](https://github.com/vllm-project/vllm/issues/31128)
- [vLLM Issue #28669: CUDA 13 Mismatch on ARM64](https://github.com/vllm-project/vllm/issues/28669)
- [Flash-Attention Issue #1969: SM_121 Support](https://github.com/Dao-AILab/flash-attention/issues/1969)
