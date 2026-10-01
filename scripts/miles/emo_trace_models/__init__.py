"""Capture the actual serving computation without changing its tensor operations."""

import os
from pathlib import Path

import torch


class TraceMixin:
    def __init__(self, config, *args, **kwargs):
        if config.hidden_size > 256 or config.num_hidden_layers > 4 or config.n_routed_experts > 16:
            raise ValueError("Layer tracing is restricted to tiny diagnostic checkpoints")
        super().__init__(config, *args, **kwargs)
        self._trace = None
        self._trace_path = Path(os.environ["EMO_NUMERICS_TRACE"])
        for name, module in self.named_modules():
            if name:
                module.register_forward_hook(self._capture(name))

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        # SGLang calls model.forward directly, bypassing root nn.Module hooks.
        self._trace = None
        if 1 < input_ids.numel() <= 128:
            self._trace = {"tokens": input_ids.detach().cpu().clone(), "outputs": {}, "inputs": {}, "routers": {}}
        try:
            output = super().forward(input_ids, positions, forward_batch, input_embeds)
            self._finish()
            return output
        finally:
            self._trace = None

    def _capture(self, name):
        def capture(module, args, output):
            if self._trace is None:
                return
            if args and isinstance(args[0], torch.Tensor):
                self._trace["inputs"][name] = args[0].detach().cpu().clone()
            if name.endswith(".topk") and hasattr(output, "topk_ids"):
                self._trace["routers"][name.removesuffix(".topk")] = {
                    "logits": args[1].detach().cpu().clone(),
                    "weights": output.topk_weights.detach().cpu().clone(),
                    "ids": output.topk_ids.detach().cpu().clone(),
                }
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                self._trace["outputs"][name] = output.detach().cpu().clone()

        return capture

    def _finish(self):
        if self._trace is not None:
            self._trace["parameters"] = {name: param.detach().cpu().clone() for name, param in self.named_parameters()}
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._trace, self._trace_path)
