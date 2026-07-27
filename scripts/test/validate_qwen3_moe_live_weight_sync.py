"""Compatibility entry point for the collected Qwen3-MoE GPU test."""

from open_instruct.test_qwen3_moe_weight_sync_gpu import validate_qwen3_moe_live_weight_sync

if __name__ == "__main__":
    validate_qwen3_moe_live_weight_sync()
