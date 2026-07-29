from typing import Any

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter


class RouterReplayWorkerExtension:
    model_runner: Any
    """Expose native vLLM router-capture state through collective RPC."""

    def get_router_replay_status(self) -> dict[str, Any]:
        model_runner = self.model_runner
        context = model_runner.compilation_config.static_forward_context
        fused_moe_layers = [module for module in context.values() if isinstance(module, FusedMoE)]
        capturable_layers = [module for module in fused_moe_layers if isinstance(module.router, BaseRouter)]
        bound_layers = [module for module in capturable_layers if module.router.capture_fn is not None]
        capturer = RoutedExpertsCapturer.get_instance()
        device_buffer = None if capturer is None else capturer._device_buffer

        return {
            "model_class": type(model_runner.model).__name__,
            "config_enabled": model_runner.model_config.enable_return_routed_experts,
            "capturer_initialized": model_runner.routed_experts_initialized,
            "capturer_present": capturer is not None,
            "device_buffer_shape": None if device_buffer is None else list(device_buffer.shape),
            "static_context_modules": len(context),
            "fused_moe_layers": len(fused_moe_layers),
            "capturable_layers": len(capturable_layers),
            "bound_layers": len(bound_layers),
            "layer_ids": [module.layer_id for module in bound_layers],
            "router_classes": sorted({type(module.router).__name__ for module in fused_moe_layers}),
        }
