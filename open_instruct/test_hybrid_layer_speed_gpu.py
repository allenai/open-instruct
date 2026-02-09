"""GPU benchmark: compare forward+backward speed of GatedDeltaNet vs FlashAttention layers.

Verifies the hypothesis that the hybrid model's linear attention layers
(GatedDeltaNet) are slower per step than standard full attention layers
(FlashAttention-2), explaining the ~3.5x training slowdown observed in
hybrid DPO runs.

Run on Beaker GPU nodes:
    uv run pytest open_instruct/test_hybrid_layer_speed_gpu.py -v -s
"""

import logging
import time
import unittest

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.olmo3_5_hybrid import modeling_olmo3_5_hybrid

logger = logging.getLogger(__name__)


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
        config = AutoConfig.from_pretrained("allenai/OLMo-3.5-7B-Hybrid-February-2026", trust_remote_code=True)

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

    def test_hybrid_model_slower_than_standard_olmo3(self):
        device = torch.device("cuda")
        batch_size = 1
        seq_len = 2048

        hybrid_config = AutoConfig.from_pretrained("allenai/OLMo-3.5-7B-Hybrid-February-2026", trust_remote_code=True)
        hybrid_model = AutoModelForCausalLM.from_config(
            hybrid_config, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to(device)

        olmo3_config = AutoConfig.from_pretrained("allenai/OLMo-3-7B-0225")
        olmo3_model = AutoModelForCausalLM.from_config(
            olmo3_config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to(device)

        input_ids = torch.randint(
            0, min(hybrid_config.vocab_size, olmo3_config.vocab_size), (batch_size, seq_len), device=device
        )
        labels = input_ids.clone()

        def time_model(model, num_warmup=2, num_iters=5):
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

        hybrid_time = time_model(hybrid_model)
        olmo3_time = time_model(olmo3_model)
        speedup = hybrid_time / olmo3_time

        logger.info(
            "Full model fwd+bwd (seq_len=%d) | OLMo3: %.1fms | Hybrid: %.1fms | hybrid/olmo3 ratio: %.2fx",
            seq_len,
            olmo3_time * 1000,
            hybrid_time * 1000,
            speedup,
        )
        self.assertGreater(
            speedup, 1.0, f"Expected hybrid model to be slower than OLMo3, but got ratio {speedup:.2f}x"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
