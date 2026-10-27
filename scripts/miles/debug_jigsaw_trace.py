"""Opt-in hooks for a copied, pinned serving package in numerical diagnostics.

The payload builder appends ``install(Olmo3MoeVisionForConditionalGeneration)`` to its copy of
the serving model module. No installed or production source is modified.
"""

import hashlib
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F


def install(model_class):
    destination = os.environ.get("OI_JIGSAW_TRACE")
    if not destination:
        return
    original_init = model_class.__init__
    original_forward = model_class.forward

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if os.environ.get("OI_JIGSAW_SPLIT_QKV") == "1":
            for layer in self.language_model.model.layers:
                attention = layer.self_attn
                if not hasattr(attention, "key_dim"):
                    continue
                projection = attention.qkv_proj
                widths = (attention.key_dim, attention.key_dim, attention.value_dim)

                def split_projection(value, projection=projection, widths=widths):
                    return torch.cat([F.linear(value, w) for w in projection.weight.split(widths)], dim=-1), None

                projection.forward = split_projection
        self._jigsaw_capture = None
        self._jigsaw_captured = False

        def capture(name):
            def hook(module, inputs, output):
                if self._jigsaw_capture is not None:
                    value = output[0] if isinstance(output, tuple) else output
                    if isinstance(value, torch.Tensor):
                        self._jigsaw_capture[name] = value.detach().clone()

            return hook

        def inputs_hook(module, inputs):
            if self._jigsaw_capture is not None:
                self._jigsaw_capture["input_embeddings"] = inputs[1].detach().clone()

        self.language_model.model.layers[0].register_forward_pre_hook(inputs_hook)
        for index, layer in enumerate(self.language_model.model.layers):
            layer.register_forward_hook(capture(f"layer_{index}"))
        self.language_model.model.norm.register_forward_hook(capture("head_input"))
        if os.environ.get("OI_JIGSAW_TRACE_DETAIL") == "1":
            for name, module in self.language_model.model.layers[0].named_modules():
                if name:
                    module.register_forward_hook(capture("detail." + name))

    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        mm_inputs = getattr(forward_batch, "mm_inputs", None)
        recording = not self._jigsaw_captured and mm_inputs and any(mm_inputs) and input_ids.numel() > 1500
        if recording:
            self._jigsaw_capture = {"token_ids_before_placement": input_ids.detach().clone()}
        result = original_forward(self, input_ids, positions, forward_batch, *args, **kwargs)
        if recording:
            self._jigsaw_captured = True
            values = {key: value.cpu() for key, value in self._jigsaw_capture.items()}
            self._jigsaw_capture = None
            target = Path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(values, target)
            weights = {}
            for name, parameter in self.named_parameters():
                value = parameter.detach().cpu().contiguous()
                weights[name] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
                }
            properties = torch.cuda.get_device_properties(0)
            metadata = {
                "hardware": properties.name,
                "sm_count": properties.multi_processor_count,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "core_compat_mode": self.language_model.core_compat_mode,
                "fp32_lm_head": self.language_model.logits_processor.use_fp32_lm_head,
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "split_qkv": os.environ.get("OI_JIGSAW_SPLIT_QKV") == "1",
                "weights": weights,
            }
            target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        return result

    model_class.__init__ = initialize
    model_class.forward = forward
