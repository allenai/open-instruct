"""GPU benchmark: compare forward+backward speed of GatedDeltaNet vs FlashAttention layers.

Compares per-layer and full-model speed between the hybrid model (OLMo 3.5
Hybrid, GatedDeltaNet linear attention) and standard OLMo3 (FlashAttention-2).

Findings:
- Individual linear attention layers ARE slower than full attention layers.
- The full hybrid model is roughly comparable to OLMo3 (the hybrid's smaller
  hidden_size 3840 vs 4096 offsets the slower linear attention layers).
- The ~3.5x training slowdown in hybrid DPO runs is NOT caused by model
  architecture; it likely comes from distributed training overhead.

Run on Beaker GPU nodes:
    uv run pytest open_instruct/test_hybrid_layer_speed_gpu.py -v -s
"""

import logging
import time
import unittest

import torch
from transformers import AutoModelForCausalLM
from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
from transformers.models.olmo3_5_hybrid import modeling_olmo3_5_hybrid
from transformers.models.olmo3_5_hybrid.configuration_olmo3_5_hybrid import Olmo3_5HybridConfig

logger = logging.getLogger(__name__)

HYBRID_CONFIG_KWARGS = {
    "vocab_size": 100352,
    "hidden_size": 3840,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 30,
    "num_key_value_heads": 30,
    "hidden_act": "silu",
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": False,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8,
    "linear_num_key_heads": 30,
    "linear_num_value_heads": 30,
    "linear_key_head_dim": 96,
    "linear_value_head_dim": 192,
    "linear_conv_kernel_dim": 4,
    "linear_use_gate": True,
    "linear_allow_neg_eigval": True,
}

OLMO3_CONFIG_KWARGS = {
    "vocab_size": 100278,
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "hidden_act": "silu",
    "max_position_embeddings": 65536,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": False,
    "rope_theta": 500000,
}


def _build_hybrid_config():
    return Olmo3_5HybridConfig(**HYBRID_CONFIG_KWARGS)


def _build_olmo3_config():
    return Olmo3Config(**OLMO3_CONFIG_KWARGS)


def _time_forward_backward(layer, hidden_states, num_warmup=3, num_iters=10, **kwargs):
    """Time forward+backward passes for a layer, returning median seconds."""
    for _ in range(num_warmup):
        out = layer(hidden_states, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        out.sum().backward()
        layer.zero_grad()

    torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = layer(hidden_states, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        out.sum().backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        layer.zero_grad()
    times.sort()
    return times[len(times) // 2]


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA GPU")
class TestHybridLayerSpeed(unittest.TestCase):
    def test_linear_attention_slower_than_full_attention(self):
        config = _build_hybrid_config()

        device = torch.device("cuda")

        full_attn_layer_idx = config.layer_types.index("full_attention")
        linear_attn_layer_idx = config.layer_types.index("linear_attention")

        full_attn_block = modeling_olmo3_5_hybrid.Olmo3_5HybridDecoderLayer(config, layer_idx=full_attn_layer_idx).to(
            device=device, dtype=torch.bfloat16
        )

        linear_attn_block = modeling_olmo3_5_hybrid.Olmo3_5HybridDecoderLayer(
            config, layer_idx=linear_attn_layer_idx
        ).to(device=device, dtype=torch.bfloat16)

        batch_size = 1
        seq_len = 4096
        hidden = torch.randn(
            batch_size, seq_len, config.hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        full_time = _time_forward_backward(full_attn_block, hidden, position_ids=position_ids)
        linear_time = _time_forward_backward(linear_attn_block, hidden)

        speedup = linear_time / full_time
        logger.info(
            "seq_len=%d | full_attention: %.3fms | linear_attention: %.3fms | linear/full ratio: %.2fx",
            seq_len,
            full_time * 1000,
            linear_time * 1000,
            speedup,
        )
        self.assertGreater(
            speedup,
            1.0,
            f"Expected linear attention to be slower than full attention, "
            f"but got ratio {speedup:.2f}x (linear={linear_time * 1000:.1f}ms, "
            f"full={full_time * 1000:.1f}ms)",
        )

    def test_hybrid_model_comparable_to_standard_olmo3(self):
        device = torch.device("cuda")
        batch_size = 1
        seq_len = 2048

        def time_model(model, input_ids, labels, num_warmup=2, num_iters=5):
            for _ in range(num_warmup):
                out = model(input_ids=input_ids, labels=labels)
                out.loss.backward()
                model.zero_grad()

            torch.cuda.synchronize()
            times = []
            for _ in range(num_iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = model(input_ids=input_ids, labels=labels)
                out.loss.backward()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                model.zero_grad()
            times.sort()
            return times[len(times) // 2]

        hybrid_config = _build_hybrid_config()
        hybrid_model = AutoModelForCausalLM.from_config(
            hybrid_config, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to(device)
        input_ids = torch.randint(0, hybrid_config.vocab_size, (batch_size, seq_len), device=device)
        hybrid_time = time_model(hybrid_model, input_ids, input_ids.clone())
        del hybrid_model
        torch.cuda.empty_cache()

        olmo3_config = _build_olmo3_config()
        olmo3_model = AutoModelForCausalLM.from_config(
            olmo3_config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to(device)
        input_ids = torch.randint(0, olmo3_config.vocab_size, (batch_size, seq_len), device=device)
        olmo3_time = time_model(olmo3_model, input_ids, input_ids.clone())
        del olmo3_model
        torch.cuda.empty_cache()

        ratio = hybrid_time / olmo3_time

        logger.info(
            "Full model fwd+bwd (seq_len=%d) | OLMo3: %.1fms | Hybrid: %.1fms | hybrid/olmo3 ratio: %.2fx",
            seq_len,
            olmo3_time * 1000,
            hybrid_time * 1000,
            ratio,
        )
        self.assertGreater(ratio, 0.5, f"Hybrid model unexpectedly >2x faster than OLMo3: ratio {ratio:.2f}x")
        self.assertLess(ratio, 2.0, f"Hybrid model unexpectedly >2x slower than OLMo3: ratio {ratio:.2f}x")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
