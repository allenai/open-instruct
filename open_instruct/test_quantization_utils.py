"""Tests for GPTQ quantization utilities."""

import unittest

import torch
import torch.nn as nn
import transformers

from open_instruct import quantization


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestQuantizationUtils(unittest.TestCase):
    """Test GPTQ quantization utilities."""

    @classmethod
    def setUpClass(cls):
        cls.model_name = "sshleifer/tiny-gpt2"
        cls.device = torch.device("cuda")

    def test_calibrate_and_compress_single_module(self):
        """Test calibrating and compressing a single Linear module."""
        module = nn.Linear(128, 256).to(self.device)

        calibration_data = torch.randn(4, 32, 128, device=self.device)

        hessian, num_samples = quantization.calibrate_module_with_data(
            module=module,
            calibration_data=calibration_data,
            existing_hessian=None,
            existing_num_samples=0,
            offload_to_cpu=False,
        )

        self.assertIsNotNone(hessian)
        self.assertEqual(hessian.shape, (128, 128))
        self.assertGreater(num_samples, 0)

        from llmcompressor.modifiers.quantization.gptq import QuantizationScheme

        quant_args = QuantizationScheme(
            targets=["Linear"],
            weights={"num_bits": 8, "type": "int", "symmetric": True, "strategy": "group", "group_size": 128},
        )

        quantized_weight, scale, zero_point, g_idx, loss = quantization.compress_module_weights(
            module=module,
            hessian=hessian,
            num_samples=num_samples,
            quant_args=quant_args,
            block_size=128,
            dampening_frac=0.01,
        )

        self.assertEqual(quantized_weight.dtype, torch.int8)
        self.assertEqual(quantized_weight.shape, module.weight.shape)
        self.assertGreater(loss, 0.0)

    def test_is_quantizable_layer(self):
        """Test layer quantizability checks."""
        linear = nn.Linear(10, 20)
        self.assertTrue(quantization.is_quantizable_layer("model.layer.0", linear))

        embedding = nn.Embedding(100, 50)
        self.assertFalse(quantization.is_quantizable_layer("model.embedding", embedding))

        lm_head = nn.Linear(10, 20)
        self.assertFalse(quantization.is_quantizable_layer("model.lm_head", lm_head))

    def test_end_to_end_model_calibration_and_compression(self):
        """Test end-to-end: load model, calibrate, compress weights."""
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map=self.device
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        text = ["Hello world! This is a test.", "Another test sentence for calibration."]
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        config = quantization.QuantizationConfig(
            enable_gptq=True, block_size=128, dampening_frac=0.01, offload_hessians=False
        )

        hessians = {}
        num_samples_dict = {}
        hooks = []

        def make_hook(module):
            def hook(mod, args, output):
                inp = args[0]
                hessian, num_samples = quantization.calibrate_module_with_data(
                    module=module,
                    calibration_data=inp,
                    existing_hessian=hessians.get(module),
                    existing_num_samples=num_samples_dict.get(module, 0),
                    offload_to_cpu=False,
                )
                hessians[module] = hessian
                num_samples_dict[module] = num_samples

            return hook

        for name, module in model.named_modules():
            if quantization.is_quantizable_layer(name, module):
                hook = module.register_forward_hook(make_hook(module))
                hooks.append(hook)

        with torch.no_grad():
            model(**inputs)

        for hook in hooks:
            hook.remove()

        self.assertGreater(len(hessians), 0, "Should have calibrated at least one module")

        from llmcompressor.modifiers.quantization.gptq import QuantizationScheme

        quant_args = QuantizationScheme(
            targets=["Linear"],
            weights={"num_bits": 8, "type": "int", "symmetric": True, "strategy": "group", "group_size": 128},
        )

        for module in list(hessians.keys())[:1]:
            quantized_weight, scale, zero_point, g_idx, loss = quantization.compress_module_weights(
                module=module,
                hessian=hessians[module],
                num_samples=num_samples_dict[module],
                quant_args=quant_args,
                block_size=config.block_size,
                dampening_frac=config.dampening_frac,
            )

            self.assertEqual(quantized_weight.dtype, torch.int8)
            self.assertEqual(quantized_weight.shape, module.weight.shape)

            quantization.store_quantized_weights(module, quantized_weight, scale, zero_point, g_idx)

            self.assertTrue(hasattr(module, "weight_quantized"))
            self.assertTrue(hasattr(module, "weight_scale"))


if __name__ == "__main__":
    unittest.main()
